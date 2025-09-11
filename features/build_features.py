"""Сборка фичей на 1H сетке и таргета на +CFG.TARGET_HOURS (по умолчанию 4ч).
   Включены: цены/деривы/время/вола/дивергенции/новости/глобальные метрики
   + дешёвые фичи для бинарных голов.
"""
from typing import Optional, List
import numpy as np
import pandas as pd

import config as CFG
from etl.prices import load_btc_prices
from etl.bybit import load_bybit_derivs
from etl.news import load_news_counts
from etl.vol_indices import realized_vol
from indicators.divergence import detect_divergence


# ----------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ----------------------
def _time_feats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dow"] = d.index.dayofweek
    d["hod"] = d.index.hour
    d["sin_h"] = np.sin(2 * np.pi * d["hod"] / 24.0)
    d["cos_h"] = np.cos(2 * np.pi * d["hod"] / 24.0)
    dow_oh = pd.get_dummies(d["dow"], prefix="dow", dtype="int8")
    return pd.concat([d, dow_oh], axis=1)


def _rolling(px: pd.DataFrame) -> pd.DataFrame:
    d = px.copy()
    r = d["close"].pct_change()
    d["ret_1h"] = r
    d["ret_4h"] = d["close"].pct_change(4)
    d["ret_1d"] = d["close"].pct_change(24)
    d["vol_24h"] = r.rolling(24, min_periods=12).std()
    d["atr_24h"] = (d["high"] - d["low"]).rolling(24, min_periods=12).mean()
    d["vol_of_vol"] = d["vol_24h"].rolling(24, min_periods=12).std()
    return d


def _head_extra_feats(df: pd.DataFrame) -> pd.DataFrame:
    """Дешёвые фичи для бинарных голов и коротких горизонтов."""
    d = df.copy()
    # abs returns
    for c in ("ret_1h", "ret_4h", "ret_1d"):
        if c in d.columns:
            d[f"abs_{c}"] = d[c].abs()

    # локальная/недельная волатильность (если есть rv и sigma_*)
    if "rv" in d.columns:
        sigma_col = "sigma_ewma_7d" if "sigma_ewma_7d" in d.columns else ("sigma_7d" if "sigma_7d" in d.columns else None)
        if sigma_col is not None:
            tmp = d[sigma_col].replace(0, np.nan)
            d["rv_over_sigma7d"] = (d["rv"] / tmp).replace([np.inf, -np.inf], np.nan)

    # экстремумы за 7д (168 часов)
    win = 168
    roll_max = d["close"].rolling(win, min_periods=max(24, win // 3)).max()
    roll_min = d["close"].rolling(win, min_periods=max(24, win // 3)).min()
    d["dist_to_max7d"] = (d["close"] / roll_max) - 1.0
    d["dist_to_min7d"] = (d["close"] / roll_min) - 1.0
    return d


def _ensure_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """Если нет sigma_7d / sigma_ewma_7d — посчитаем из часовых рентов."""
    d = df.copy()
    r = d["close"].pct_change()
    if "sigma_7d" not in d.columns:
        d["sigma_7d"] = r.rolling(168, min_periods=96).std()
    if "sigma_ewma_7d" not in d.columns:
        # полураспад ~3.5 дня (половина недели)
        d["sigma_ewma_7d"] = r.ewm(halflife=84, min_periods=48).std()
    return d


# ----------------------
# ОСНОВНОЙ БИЛДЕР ФИЧ
# ----------------------
def build_feature_table(
    price_path: str = CFG.PX_FILE,
    oi_path: str = CFG.OI_FILE,
    funding_path: str = CFG.FUND_FILE,
    news_path: Optional[str] = CFG.NEWS_FILE,
    macro_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Собираем единую часовую таблицу фичей и цель на +CFG.TARGET_HOURS часов.
    Все источники приводим к UTC 1H. Внешние источники (OI/Funding/News/Meta)
    мягко фоллбэкаем, чтобы пайплайн не падал.
    """
    horizon = int(getattr(CFG, "TARGET_HOURS", 4))  # по умолчанию 4 часа

    # 1) База: цены
    px = load_btc_prices(price_path)
    base = px.copy()

    # 2) Деривативы (OI/Funding) — опционально
    try:
        dv = load_bybit_derivs(oi_path, funding_path)
        base = base.join(dv, how="left")
    except Exception:
        pass

    base = base.sort_index()

    # 3) RSI/дивергенции — опционально
    try:
        div = detect_divergence(base[["close"]].dropna(), rsi_period=14, lb=48)
        base = base.join(div, how="left")
    except Exception:
        pass

    # 4) Реализованная вола и стандартные окна
    base["rv"] = realized_vol(px)
    base = _rolling(base)
    base = _time_feats(base)

    # 5) Новости (агрегаты) — опционально
    if news_path:
        try:
            news = load_news_counts(news_path)  # уже на 1H
            # редкие события → заполняем нулями, чтобы не терять строки
            base = base.join(news, how="left").fillna({c: 0 for c in news.columns})
        except Exception:
            pass

    # 6) Макро — если переданы
    if macro_df is not None:
        try:
            base = base.join(macro_df, how="left")
        except Exception:
            pass

    # 7) Глобальные метрики рынка (TOTAL, BTC.D, USDT.D) — опционально
    try:
        gm = pd.read_parquet("data/interim/global_meta.parquet")
        base = base.join(gm, how="left")
    except Exception:
        pass

    # медленные каналы (oi/funding/meta) тянем вперёд
    slow_cols = [c for c in base.columns if c.startswith("oi") or c.startswith("fund") or "dominance" in c or "market_cap" in c]
    if slow_cols:
        base[slow_cols] = base[slow_cols].ffill()

    # 8) Доп. фичи
    base = _head_extra_feats(base)

    # 9) sigma_* для масштабирования/меток
    base = _ensure_sigma(base)

    # 10) Цель на +horizon
    base["y"] = np.log(base["close"].shift(-horizon)) - np.log(base["close"])

    # 11) Чистим пропуски, появившиеся из-за окон/сдвигов
    required = [
        "y", "ret_1h", "ret_4h", "ret_1d", "vol_24h", "atr_24h", "rv",
        "sin_h", "cos_h", "open", "high", "low", "close", "volume",
        "sigma_7d", "sigma_ewma_7d",
    ]
    req_here = [c for c in required if c in base.columns]
    base = base.dropna(subset=req_here)

    return base