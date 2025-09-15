"""Сборка фичей на базовой сетке (CFG.BASE_TF_MINUTES) и таргетов на несколько горизонтов.
Создаёт: y_h1, y_h4, y_h24 и алиас y = y_h{CFG.TARGET_HOURS}.
Включены: цены/деривы/время/вола/дивергенции/новости/глобальные метрики + лёгкие head-фичи.
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
# ВСПОМОГАТЕЛЬНЫЕ
# ----------------------
def _steps(hours: int) -> int:
    base_min = int(getattr(CFG, "BASE_TF_MINUTES", 60))
    return max(1, int(round(hours * 60.0 / base_min)))

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
    # returns в единицах базового шага
    d["ret_1step"] = d["close"].pct_change(1)
    d["ret_1h"] = d["close"].pct_change(_steps(1))
    d["ret_4h"] = d["close"].pct_change(_steps(4))
    d["ret_1d"] = d["close"].pct_change(_steps(24))
    d["vol_24h"] = d["ret_1step"].rolling(_steps(24), min_periods=max(12, _steps(6))).std()
    d["atr_24h"] = (d["high"] - d["low"]).rolling(_steps(24), min_periods=max(12, _steps(6))).mean()
    d["vol_of_vol"] = d["vol_24h"].rolling(_steps(24), min_periods=max(12, _steps(6))).std()
    return d

def _head_extra_feats(df: pd.DataFrame) -> pd.DataFrame:
    """Дешёвые фичи для бинарных голов и коротких горизонтов."""
    d = df.copy()
    for c in ("ret_1step", "ret_1h", "ret_4h", "ret_1d"):
        if c in d.columns:
            d[f"abs_{c}"] = d[c].abs()

    if "rv" in d.columns:
        sigma_col = "sigma_ewma_7d" if "sigma_ewma_7d" in d.columns else ("sigma_7d" if "sigma_7d" in d.columns else None)
        if sigma_col is not None:
            tmp = d[sigma_col].replace(0, np.nan)
            d["rv_over_sigma7d"] = (d["rv"] / tmp).replace([np.inf, -np.inf], np.nan)

    # экстремумы за 7д (в шагах базы)
    win = _steps(168)
    roll_max = d["close"].rolling(win, min_periods=max(_steps(24), win // 3)).max()
    roll_min = d["close"].rolling(win, min_periods=max(_steps(24), win // 3)).min()
    d["dist_to_max7d"] = (d["close"] / roll_max) - 1.0
    d["dist_to_min7d"] = (d["close"] / roll_min) - 1.0
    return d

def _ensure_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """Если нет sigma_7d / sigma_ewma_7d — посчитаем из базовых рентов."""
    d = df.copy()
    r = d["close"].pct_change(1)
    if "sigma_7d" not in d.columns:
        d["sigma_7d"] = r.rolling(_steps(168), min_periods=max(48, _steps(24))).std()
    if "sigma_ewma_7d" not in d.columns:
        d["sigma_ewma_7d"] = r.ewm(halflife=_steps(84), min_periods=max(24, _steps(12))).std()
    return d

# ----------------------
# ОСНОВНОЙ БИЛДЕР
# ----------------------
def build_feature_table(
    price_path: str = CFG.PX_FILE,
    oi_path: str = CFG.OI_FILE,
    funding_path: str = CFG.FUND_FILE,
    news_path: Optional[str] = CFG.NEWS_FILE,
    macro_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Собираем единую таблицу на базовой сетке (CFG.BASE_TF_MINUTES) и таргеты: y_h1, y_h4, y_h24.
    Алиас y = y_h{CFG.TARGET_HOURS}.
    """
    target_hours = int(getattr(CFG, "TARGET_HOURS", 4))
    # можно переопределить список горизонтов через конфиг
    multi_hours = list(dict.fromkeys(
        [1, 4, 24] + list(getattr(CFG, "MULTI_TARGET_HOURS", [])) + [target_hours]
    ))

    # 1) цены
    px = load_btc_prices(price_path)
    base = px.copy().sort_index()

    # 2) деривы (опционально)
    try:
        dv = load_bybit_derivs(oi_path, funding_path)
        base = base.join(dv, how="left")
    except Exception:
        pass

    # 3) RSI/дивергенции (опц.)
    try:
        div = detect_divergence(base[["close"]].dropna(), rsi_period=14, lb=max(48, _steps(24)))
        base = base.join(div, how="left")
    except Exception:
        pass

    # 4) вола/окна/время
    base["rv"] = realized_vol(px)  # функция должна уметь работать на базовом шаге (или ок)
    base = _rolling(base)
    base = _time_feats(base)

    # 5) новости (агрегаты)
    if news_path:
        try:
            news = load_news_counts(news_path)  # уже resample-нутая в 1H — приведём к base через ffill
            news = news.reindex(base.index, method="ffill")
            base = base.join(news, how="left").fillna({c: 0 for c in news.columns})
        except Exception:
            pass

    # 6) макро (если передали)
    if macro_df is not None:
        try:
            macro_df = macro_df.reindex(base.index, method="ffill")
            base = base.join(macro_df, how="left")
        except Exception:
            pass

    # 7) глобальные метрики
    try:
        gm = pd.read_parquet("data/interim/global_meta.parquet")
        gm = gm.reindex(base.index, method="ffill")
        base = base.join(gm, how="left")
    except Exception:
        pass

    # медленные каналы тянем вперёд
    slow_cols = [c for c in base.columns if c.startswith("oi") or c.startswith("fund") or "dominance" in c or "market_cap" in c]
    if slow_cols:
        base[slow_cols] = base[slow_cols].ffill()

    # 8) extra фичи
    base = _head_extra_feats(base)

    # 9) сигмы
    base = _ensure_sigma(base)

    # 10) несколько таргетов
    for h in multi_hours:
        k = _steps(h)
        base[f"y_h{h}"] = np.log(base["close"].shift(-k)) - np.log(base["close"])

    # алиас для текущего бизнес-горизонта
    alias_col = f"y_h{target_hours}"
    if alias_col in base.columns:
        base["y"] = base[alias_col]

    # 11) чистка обязательных колонок
    required = [
        "y", "ret_1h", "ret_4h", "ret_1d", "vol_24h", "atr_24h", "rv",
        "sin_h", "cos_h", "open", "high", "low", "close", "volume",
        "sigma_7d", "sigma_ewma_7d",
    ]
    req_here = [c for c in required if c in base.columns]
    base = base.dropna(subset=req_here)

    return base