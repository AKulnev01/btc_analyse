"""Сборка фичей на базовой сетке (CFG.BASE_TF_MINUTES) и таргетов на несколько горизонтов.
Создаёт: y_h1, y_h4, y_h24 и алиас y = y_h{CFG.TARGET_HOURS}.
Включены: цены/деривы/время/вола/дивергенции/новости/глобальные метрики + внешние (ончейн/фандинг/новсентимент) + лёгкие head-фичи.
"""
from typing import Optional, List
import os
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


def _safe_join(left: pd.DataFrame, right: Optional[pd.DataFrame], how="left") -> pd.DataFrame:
    """джойн, который молча пропускает отсутствующие/пустые таблицы"""
    if right is None or isinstance(right, pd.DataFrame) and right.empty:
        return left
    # приводим индекс к UTC DatetimeIndex для единообразия
    if right.index.tz is None:
        right = right.copy()
        right.index = pd.to_datetime(right.index, utc=True)
    return left.join(right, how=how)


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
    if base.index.tz is None:
        base.index = pd.to_datetime(base.index, utc=True)

    # 2) деривы (Bybit OI/funding — если используешь)
    try:
        if getattr(CFG, "FEATURE_FLAGS", {}).get("USE_DERIVS", True):
            dv = load_bybit_derivs(oi_path, funding_path)
            base = base.join(dv, how="left")
    except Exception as e:
        print(f"[WARN] derivs join: {e}")

    # 3) RSI/дивергенции (опц.)
    try:
        if getattr(CFG, "FEATURE_FLAGS", {}).get("USE_DIVERGENCE", True):
            div = detect_divergence(base[["close"]].dropna(), rsi_period=14, lb=max(48, _steps(24)))
            base = base.join(div, how="left")
    except Exception as e:
        print(f"[WARN] divergence: {e}")

    # 4) вола/окна/время
    base["rv"] = realized_vol(px)  # функция должна уметь работать на базовом шаге
    base = _rolling(base)
    base = _time_feats(base)

    # 5) новости-«счётчики» (если есть твой news_counts)
    if news_path and getattr(CFG, "FEATURE_FLAGS", {}).get("USE_NEWS", True):
        try:
            news = load_news_counts(news_path)  # обычно уже на 1H; приведём к base через ffill
            news = news.reindex(base.index, method="ffill")
            base = base.join(news, how="left").fillna({c: 0 for c in news.columns})
        except Exception as e:
            print(f"[WARN] news_counts: {e}")

    # 5.1) новостной сентимент (новый кэш)
    if getattr(CFG, "FEATURE_FLAGS", {}).get("USE_NEWS_SENTI", True):
        try:
            sf = getattr(CFG, "SENTI_FILE", "")
            if sf and os.path.exists(sf):
                senti = pd.read_parquet(sf)
                if senti.index.tz is None:
                    senti.index = pd.to_datetime(senti.index, utc=True)
                # приведём к сетке base ffill-ом (редкие агрегаты)
                senti = senti.reindex(base.index, method="ffill")
                base = base.join(senti, how="left")
        except Exception as e:
            print(f"[WARN] news_sentiment join: {e}")

    # 6) макро (если передали)
    if macro_df is not None and getattr(CFG, "FEATURE_FLAGS", {}).get("USE_MACRO", False):
        try:
            macro_df = macro_df.copy()
            if macro_df.index.tz is None:
                macro_df.index = pd.to_datetime(macro_df.index, utc=True)
            macro_df = macro_df.reindex(base.index, method="ffill")
            base = base.join(macro_df, how="left")
        except Exception as e:
            print(f"[WARN] macro join: {e}")

    # 7) глобальные метрики (как у тебя было)
    try:
        if getattr(CFG, "FEATURE_FLAGS", {}).get("USE_GLOBAL_META", True):
            gm_path = getattr(CFG, "GLOBAL_META_FILE", "data/interim/global_meta.parquet")
            if gm_path and os.path.exists(gm_path):
                gm = pd.read_parquet(gm_path)
                if gm.index.tz is None:
                    gm.index = pd.to_datetime(gm.index, utc=True)
                gm = gm.reindex(base.index, method="ffill")
                base = base.join(gm, how="left")
    except Exception as e:
        print(f"[WARN] global_meta: {e}")

    # 8) ВНЕШНИЕ НОВЫЕ ИСТОЧНИКИ (ончейн/фандинг-кроссбиржевой)
    # on-chain (Glassnode bundle)
    if getattr(CFG, "FEATURE_FLAGS", {}).get("USE_ONCHAIN", True):
        try:
            oc_path = getattr(CFG, "ONCHAIN_FILE", "")
            if oc_path and os.path.exists(oc_path):
                onchain = pd.read_parquet(oc_path)
                base = _safe_join(base, onchain, how="left")
        except Exception as e:
            print(f"[WARN] onchain join: {e}")

    # funding (Binance/Bybit/OKX + агрегаты)
    if getattr(CFG, "FEATURE_FLAGS", {}).get("USE_FUNDING", True):
        try:
            f_path = getattr(CFG, "FUNDING_FILE", "")
            if f_path and os.path.exists(f_path):
                funding_all = pd.read_parquet(f_path)
                # сгладим быстрые шумы и добавим простые производные
                for c in funding_all.columns:
                    if "funding" in c:
                        funding_all[c + "_ema"] = funding_all[c].ewm(span=12, min_periods=3).mean()
                base = _safe_join(base, funding_all, how="left")
        except Exception as e:
            print(f"[WARN] funding join: {e}")

    # медленные каналы тянем вперёд
    slow_cols = [c for c in base.columns if c.startswith("oi")
                 or c.startswith("fund")
                 or "dominance" in c
                 or "market_cap" in c]
    if slow_cols:
        base[slow_cols] = base[slow_cols].ffill()

    # 9) extra фичи
    base = _head_extra_feats(base)

    # 10) сигмы
    base = _ensure_sigma(base)

    # 11) несколько таргетов
    for h in multi_hours:
        k = _steps(h)
        base[f"y_h{h}"] = np.log(base["close"].shift(-k)) - np.log(base["close"])

    # алиас для текущего бизнес-горизонта
    alias_col = f"y_h{target_hours}"
    if alias_col in base.columns:
        base["y"] = base[alias_col]

    # 12) чистка обязательных колонок (требовательная часть)
    required = [
        "y", "ret_1h", "ret_4h", "ret_1d", "vol_24h", "atr_24h", "rv",
        "sin_h", "cos_h", "open", "high", "low", "close", "volume",
        "sigma_7d", "sigma_ewma_7d",
    ]
    req_here = [c for c in required if c in base.columns]
    base = base.dropna(subset=req_here)

    return base