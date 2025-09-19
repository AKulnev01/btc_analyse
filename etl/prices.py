# etl/prices.py
import pandas as pd
import numpy as np
import config as CFG

OHLC_AGG = {
    "open": "first",
    "high": "max",
    "low":  "min",
    "close":"last",
    "volume":"sum",
}

def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    idx = pd.to_datetime(d.index, utc=True)
    d.index = idx
    d = d.sort_index()
    return d

def _norm_bar(bar: str) -> str:
    bar = str(bar)
    if bar.endswith("T"):   # '30T' -> '30min'
        return f"{bar[:-1]}min"
    return bar

def _maybe_resample(df: pd.DataFrame, bar: str) -> pd.DataFrame:
    d = df.copy()
    try:
        infer = pd.infer_freq(d.index)
    except Exception:
        infer = None
    # сравниваем нормализованные частоты
    if infer and _norm_bar(infer) == _norm_bar(bar):
        return d
    d = (
        d.resample(_norm_bar(bar))
         .agg(OHLC_AGG)
         .dropna(subset=["open","high","low","close"])
    )
    if "volume" not in d.columns:
        d["volume"] = 0.0
    return d

def _maybe_resample(df: pd.DataFrame, bar: str) -> pd.DataFrame:
    """Ресемплим к нужной сетке BAR, если исходные данные не совпадают."""
    d = df.copy()
    # если уже ровно в нужной частоте
    try:
        infer = pd.infer_freq(d.index)
    except Exception:
        infer = None
    if infer == bar:
        return d

    # ресемпл до BAR
    d = (
        d.resample(bar)
         .agg(OHLC_AGG)
         .dropna(subset=["open","high","low","close"])
    )
    # объём, если колонки нет — создадим нули
    if "volume" not in d.columns:
        d["volume"] = 0.0
    return d

def load_btc_prices(path: str = CFG.PX_FILE, bar: str = None) -> pd.DataFrame:
    """
    Грузим цены (parquet/csv), приводим к UTC и к CFG.BAR (или переданному).
    Требуем колонки: open/high/low/close (+ volume по возможности).
    """
    bar = bar or CFG.BAR
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        # попробуем найти колонку с временем
        for cand in ("timestamp","ts","time","date"):
            if cand in df.columns:
                df = df.set_index(cand)
                break

    df = _ensure_utc_index(df)

    # гарантируем нужные колонки
    need = {"open","high","low","close"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"load_btc_prices: отсутствуют колонки: {missing}")

    # объём, если нет — создадим
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = _maybe_resample(df, bar)
    return df