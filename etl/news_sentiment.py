# etl/news_sentiment.py
from __future__ import annotations
import math
import pandas as pd
from textblob import TextBlob
import config as CFG

# нормализуем BAR к pandas-частоте
def _norm_bar(bar: str) -> str:
    bar = str(bar or "60min").upper()
    if bar.endswith("T"):  # '30T' -> '30min'
        return f"{bar[:-1]}min"
    if bar.endswith("H"):
        return f"{bar[:-1]}H"
    return bar.lower()

_TIME_CANDIDATES = [
    "timestamp", "published_at", "created_at", "time", "date",
    "pub_time", "published", "datetime", "dt"
]
_TEXT_CANDIDATES = ["title", "text", "headline", "summary"]

def _find_time_column(df: pd.DataFrame) -> str | None:
    cols_l = {c.lower(): c for c in df.columns}
    for lc in _TIME_CANDIDATES:
        if lc in cols_l:
            return cols_l[lc]
    return None

def _find_text_column(df: pd.DataFrame) -> str | None:
    cols_l = {c.lower(): c for c in df.columns}
    for lc in _TEXT_CANDIDATES:
        if lc in cols_l:
            return cols_l[lc]
    return None

def _to_datetime_utc(s: pd.Series) -> pd.Series:
    # сначала пробуем как секунды/мс, потом обычный parse
    s2 = pd.to_datetime(s, utc=True, errors="coerce")
    if s2.notna().mean() < 0.5:  # плохо распарсилось — попробуем как ms
        try_ms = pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="ms", utc=True, errors="coerce")
        if try_ms.notna().sum() > s2.notna().sum():
            s2 = try_ms
        else:
            try_s = pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="s", utc=True, errors="coerce")
            if try_s.notna().sum() > s2.notna().sum():
                s2 = try_s
    return s2

def build_news_sentiment(news_df: pd.DataFrame, resample_to: str | None = None) -> pd.DataFrame:
    """
    Возвращает агрегаты на сетке CFG.BAR (по умолчанию):
      - news_count
      - news_pos_share
      - news_polarity_mean
      - news_shock  (скачок интенсивности относительно скользящей базы)
    """
    if news_df is None or len(news_df) == 0:
        return pd.DataFrame(columns=["news_count","news_pos_share","news_polarity_mean","news_shock"])

    d = news_df.copy()
    # 1) время
    tcol = _find_time_column(d)
    if tcol is not None:
        d[tcol] = _to_datetime_utc(d[tcol])
        d = d.dropna(subset=[tcol]).set_index(tcol)
    else:
        # fallback: используем индекс, если он уже datetime-like
        if not isinstance(d.index, pd.DatetimeIndex):
            raise ValueError("news_df must contain a timestamp column or have DatetimeIndex")
        d.index = pd.to_datetime(d.index, utc=True, errors="coerce")
        d = d.dropna(axis=0, how="any")

    d = d.sort_index()

    # 2) текст
    ttxt = _find_text_column(d)
    if ttxt is None:
        # если текста нет — считаем только количества
        d["polarity"] = 0.0
    else:
        d["polarity"] = d[ttxt].astype(str).apply(lambda s: TextBlob(s).sentiment.polarity if isinstance(s, str) else 0.0)

    # 3) частота ресемплинга
    freq = _norm_bar(resample_to or getattr(CFG, "BAR", "60min"))
    g = d.resample(freq)

    out = pd.DataFrame({
        "news_count": g.size(),
    })
    # доля положительных
    out["news_pos_share"] = g.apply(lambda x: float((x["polarity"] > 0).mean()) if len(x) else 0.0)
    # средняя полярность
    out["news_polarity_mean"] = g["polarity"].mean()

    # 4) shock: отношение к скользящему среднему за ~1 сутки
    # берём окно ~ 24h: если BAR=30min, то 48; если 1H, то 24…
    bars_per_hour = getattr(CFG, "BARS_PER_HOUR", 2)
    win = max(12, 24 * bars_per_hour)
    roll = out["news_count"].rolling(win, min_periods=max(6, win // 3)).mean().clip(lower=1e-9)
    out["news_shock"] = (out["news_count"] / roll) - 1.0

    # заполним NaN начального участка
    out = out.fillna({"news_count": 0, "news_pos_share": 0.0, "news_polarity_mean": 0.0, "news_shock": 0.0})
    return out