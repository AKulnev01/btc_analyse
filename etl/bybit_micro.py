# etl/bybit_micro.py
# 1m Bybit → микроструктурные фичи (imbalance, spread proxy, event intensity, spike/cooldown, regime)
import os
import time
import math
import json
import requests
import numpy as np
import pandas as pd

import config as CFG

BYBIT_BASE = "https://api.bybit.com"
SYMBOL = getattr(CFG, "BYBIT_SYMBOL", "BTCUSDT")
CATEGORY = getattr(CFG, "BYBIT_CATEGORY", "linear")  # linear|inverse|spot
MINUTE_PATH = getattr(CFG, "PX_1M_FILE", "data/interim/btc_1m.parquet")

# --- низкоуровневый fetch 1m-клинов (исторически; пагинация по start/end) ---
def _bybit_klines_1m(start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
    """
    Возвращает 1m-клины [open,high,low,close,volume] в UTC.
    Bybit v5: /v5/market/kline?category=linear&symbol=BTCUSDT&interval=1
    """
    url = f"{BYBIT_BASE}/v5/market/kline"
    params = {
        "category": CATEGORY,
        "symbol": SYMBOL,
        "interval": "1",
        "start": start_ms,
        "end": end_ms,
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    if str(j.get("retCode", 0)) != "0":
        raise RuntimeError(f"Bybit error: {j.get('retCode')} {j.get('retMsg')}")
    rows = j.get("result", {}).get("list", []) or []
    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    # формат: [start, open, high, low, close, volume, turnover]
    recs = []
    for arr in rows:
        # бывает как список строк
        ts = int(arr[0])
        recs.append({
            "timestamp": pd.to_datetime(ts, unit="ms", utc=True),
            "open": float(arr[1]),
            "high": float(arr[2]),
            "low": float(arr[3]),
            "close": float(arr[4]),
            "volume": float(arr[5]),
        })
    df = pd.DataFrame(recs).set_index("timestamp").sort_index()
    return df

def _ms(dt: pd.Timestamp) -> int:
    return int(dt.value // 10**6)

def _now_utc_floor_min() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize("UTC").floor("T")

def ensure_1m_prices(days: int = 365, path: str = MINUTE_PATH) -> pd.DataFrame:
    """
    Гарантирует локальный parquet 1m клинов за 'days' дней назад до текущего.
    Аппендит недостающий хвост, берёт из диска что уже есть.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    end = _now_utc_floor_min()
    start = end - pd.Timedelta(days=days)

    if os.path.exists(path):
        try:
            old = pd.read_parquet(path)
            old = old[~old.index.duplicated(keep="last")].sort_index()
        except Exception:
            old = pd.DataFrame()
    else:
        old = pd.DataFrame()

    need_from = start
    if len(old) > 0:
        # если уже есть покрытие — тянем только «хвост»
        last_ts = old.index.max()
        if last_ts >= start:
            need_from = (last_ts + pd.Timedelta(minutes=1)).floor("T")

    if need_from > end:
        return old

    # пагинация рывками (~1000 минут за раз)
    cur = need_from
    chunks = []
    while cur <= end:
        seg_end = min(cur + pd.Timedelta(minutes=999), end)
        try:
            df = _bybit_klines_1m(_ms(cur), _ms(seg_end))
            chunks.append(df)
        except Exception as e:
            # подождём и попробуем шагнуть дальше (бывает 429/пусто)
            time.sleep(0.4)
        cur = seg_end + pd.Timedelta(minutes=1)
        time.sleep(0.12)  # мягкий рейт-лимит

    add = pd.concat(chunks, axis=0).sort_index() if chunks else pd.DataFrame()
    if len(old) > 0 and len(add) > 0:
        out = pd.concat([old, add]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
    else:
        out = add if len(add) > 0 else old

    out.to_parquet(path)
    return out

# --- фичи на 1m и агрегация до базового TF ---
def _zscore(x: pd.Series, win: int) -> pd.Series:
    m = x.rolling(win, min_periods=max(5, win//5)).mean()
    s = x.rolling(win, min_periods=max(5, win//5)).std()
    return (x - m) / s.replace(0, np.nan)

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False, min_periods=max(5, span//5)).mean()

def _atr_from_ohlc(df: pd.DataFrame, win: int) -> pd.Series:
    # ATR( win ) по 1m
    hl = df["high"] - df["low"]
    c_prev = df["close"].shift(1)
    tr = np.maximum.reduce([hl, (df["high"] - c_prev).abs(), (df["low"] - c_prev).abs()])
    return tr.rolling(win, min_periods=max(5, win//5)).mean()

def build_micro_features(minute_df: pd.DataFrame,
                         resample_minutes: int = 60,
                         event_thr: float = 0.001,
                         spike_thr_z: float = 3.0) -> pd.DataFrame:
    """
    На входе 1m OHLCV (UTC, индекс — минутные метки).
    Возвращает фичтаб на сетке 'resample_minutes' минут.
    """
    d = minute_df.copy().sort_index()
    # базовые минутные расчёты
    d["ret_1m"] = d["close"].pct_change()
    d["hl_spread_proxy"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    # buy/sell по tick-rule (на минуте): если ret>0 → покупатели агрессоры
    buy_vol = np.where(d["ret_1m"] > 0, d["volume"], 0.0)
    sell_vol = np.where(d["ret_1m"] < 0, d["volume"], 0.0)
    d["buy_vol_1m"] = buy_vol
    d["sell_vol_1m"] = sell_vol

    # интенсивность событий: доля минут с |ret| > event_thr
    evt = (d["ret_1m"].abs() > event_thr).astype(int)
    # momentum z-score по минутным ретурнам
    d["ret_z_60"] = _zscore(d["ret_1m"], 60)

    # regime: тренд/флэт (через расхождение EMA и ATR)
    ema_fast = _ema(d["close"], 50)
    ema_slow = _ema(d["close"], 200)
    atr200 = _atr_from_ohlc(d, 200)
    regime_score = (ema_fast - ema_slow).abs() / atr200.replace(0, np.nan)
    d["regime_trend_score"] = regime_score
    d["regime_trend_flag"] = (regime_score > 1.0).astype(int)  # грубый порог

    # spike/cooldown: z(|ret|) по 60м окну; «сильный спайк» и сколько минут прошло
    absz = d["ret_z_60"].abs()
    spike_flag = (absz >= spike_thr_z).astype(int)
    d["spike_flag"] = spike_flag
    # minutes since last spike
    last_spike_idx = np.where(spike_flag.values == 1)[0]
    cooldown = np.full(len(d), np.nan)
    last = -1e9
    for i in range(len(d)):
        if spike_flag.iat[i] == 1:
            last = i
        cooldown[i] = (i - last) if last > -1e9 else np.nan
    d["spike_cooldown_min"] = cooldown

    # ресемплинг до желаемого TF
    rule = f"{int(resample_minutes)}T"
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        "hl_spread_proxy": "mean",
        "buy_vol_1m": "sum", "sell_vol_1m": "sum",
        "ret_1m": "sum",           # суммарный минутный ретурн ≈ логика add, но для pct_change это ок для малых рет
        "ret_z_60": "mean",
        "regime_trend_score": "last", "regime_trend_flag": "last",
        "spike_flag": "max",       # был ли спайк в интервале
        "spike_cooldown_min": "last",
        # интенсивность событий (доля минут с |ret|>thr)
    }
    g = d.resample(rule).agg(agg)

    # после агрегации — derived features на TF
    g["imbalance"] = (g["buy_vol_1m"] - g["sell_vol_1m"]) / (g["buy_vol_1m"] + g["sell_vol_1m"]).replace(0, np.nan)
    # доля «событийных» минут
    evt_share = evt.resample(rule).mean()
    g["event_intensity"] = evt_share.reindex(g.index)

    # gap-на TF: gap против предыдущего close
    g["gap_up"] = ((g["open"] > g["close"].shift(1) * 1.002)).astype(int)   # > +0.2%
    g["gap_down"] = ((g["open"] < g["close"].shift(1) * 0.998)).astype(int) # < -0.2%

    # proxy POC shift (очень грубо): сдвиг VWAP в интервале относительно предыдущего
    vwap = (g["close"] * g["volume"]).replace([np.inf, -np.inf], np.nan)
    vwap = vwap.rolling(3, min_periods=1).mean()  # сглаженно
    g["vwap_shift"] = (vwap - vwap.shift(1)) / vwap.shift(1).replace(0, np.nan)

    # приберём чисто минутные колонки из финального набора
    drop_cols = ["buy_vol_1m", "sell_vol_1m", "ret_1m"]
    g = g.drop(columns=[c for c in drop_cols if c in g.columns])

    # префиксируем, чтобы не конфликтовать с базовыми OHLCV
    rename = {
        "hl_spread_proxy": "micro_spread_proxy",
        "ret_z_60": "micro_ret_z_60",
        "regime_trend_score": "regime_trend_score",
        "regime_trend_flag": "regime_trend_flag",
        "spike_flag": "spike_flag",
        "spike_cooldown_min": "spike_cooldown_min",
        "event_intensity": "event_intensity",
        "imbalance": "order_imbalance",
        "gap_up": "gap_up",
        "gap_down": "gap_down",
        "vwap_shift": "vwap_shift",
        "volume": "vol_sum",
    }
    g = g.rename(columns=rename)
    # оставим только фичи, не пересекаясь с базовыми OHLCV (их у нас уже есть из hourly/30m)
    keep = [
        "micro_spread_proxy", "micro_ret_z_60",
        "order_imbalance", "event_intensity",
        "regime_trend_score", "regime_trend_flag",
        "spike_flag", "spike_cooldown_min",
        "gap_up", "gap_down", "vwap_shift", "vol_sum"
    ]
    g = g[keep].copy()
    return g

def build_micro_feature_table(days: int = 365, resample_minutes: int = None) -> pd.DataFrame:
    """
    Полный цикл: ensure 1m → агрегировать → вернуть фичи на базовой сетке (CFG.BASE_TF_MINUTES).
    """
    if resample_minutes is None:
        resample_minutes = int(getattr(CFG, "BASE_TF_MINUTES", 60))
    m1 = ensure_1m_prices(days=days, path=MINUTE_PATH)
    if m1.empty:
        return pd.DataFrame()
    feats = build_micro_features(m1, resample_minutes=resample_minutes)
    return feats