# features/news_surprise.py
"""
Строим 'news_shock' на произвольной BAR-сетке.
Работает как с часовыми counts (Cryptopanic), так и с минутными, если они есть.
Идея:
  - нормализуем интенсивность новостей в [0..1]
  - лаг в минутах (чтоб учесть задержку реакции рынка)
  - эксп. распад импульса в минутах
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config as CFG

def _ensure_series(df: pd.DataFrame) -> pd.Series:
    """Принимаем DF с колонкой 'news_cnt' или любой одной колонкой — возвращаем Series."""
    if isinstance(df, pd.Series):
        return df
    if "news_cnt" in df.columns:
        s = df["news_cnt"]
    else:
        # возьмём первую числовую
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num:
            raise ValueError("news_surprise: нет числовых столбцов для новостей")
        s = df[num[0]]
    return s

def _to_bar(df: pd.DataFrame, bar: str) -> pd.DataFrame:
    """Ресемплим новостные counts к BAR через sum (сохраняем редкие всплески)."""
    if pd.infer_freq(df.index) == bar:
        return df
    s = _ensure_series(df)
    d = s.resample(bar).sum().to_frame("news_cnt")
    return d

def build_news_shock(
    news_df: pd.DataFrame,
    bar: str = CFG.BAR,
    lag_min: int = CFG.NEWS_LAG_MIN,
    decay_min: int = CFG.NEWS_DECAY_MIN,
    max_per_hour: int = CFG.NEWS_MAX_PER_HOUR,
) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонкой 'news_shock' на BAR-сетке [0..1].
    - ресемпл до BAR
    - нормализация: x / max_per_hour (клип до 1)
    - лаг (мин)
    - эксп. распад (по минутной шкале, пересчитанный на шаг BAR)
    """
    if news_df is None or len(news_df) == 0:
        return pd.DataFrame()

    d = news_df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        for cand in ("timestamp","ts","time","date"):
            if cand in d.columns:
                d = d.set_index(cand)
                break
    d.index = pd.to_datetime(d.index, utc=True)
    d = d.sort_index()

    d = _to_bar(d, bar)
    # нормализуем интенсивность (клипнем)
    x = d["news_cnt"].astype(float) / float(max_per_hour)
    x = x.clip(lower=0.0, upper=1.0)

    # пересчёт минутного лага/распада в "кол-во баров"
    bar_minutes = {"1H": 60, "30T": 30, "1T": 1}[bar]
    lag_bars = max(int(round(lag_min / bar_minutes)), 0)
    # экспон. фильтр: эквивалентный альфа на шаге BAR
    # decay_min — это tau (в минутах) ~ время распада до ~37%
    # преобразуем: alpha = 1 - exp(-bar_minutes / tau)
    tau = max(decay_min, 1)
    alpha = 1.0 - np.exp(-bar_minutes / tau)

    # применим лаг и EWM
    if lag_bars > 0:
        x = x.shift(lag_bars)

    shock = x.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    out = pd.DataFrame({"news_shock": shock.fillna(0.0)})
    return out