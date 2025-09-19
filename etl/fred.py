# etl/fred.py
import os
import requests
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("data/external/fred")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://api.stlouisfed.org/fred/series/observations"

# Топ полезных индикаторов
FRED_SERIES = {
    "DXY": "DTWEXBGS",        # US Dollar Index
    "VIX": "VIXCLS",          # Volatility Index
    "SP500": "SP500",         # S&P500 Index
    "GOLD": "GOLDAMGBD228NLBM",  # Gold USD
}

FRED_API_KEY = os.getenv("FRED_API_KEY")  # можно получить бесплатно на https://fred.stlouisfed.org/

def _fetch_series(series_id: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "file_type": "json",
    }
    if FRED_API_KEY:
        params["api_key"] = FRED_API_KEY

    r = requests.get(BASE, params=params, timeout=15)
    r.raise_for_status()
    js = r.json()

    obs = js.get("observations", [])
    df = pd.DataFrame(obs)
    df["t"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df.set_index("t", inplace=True)
    df = df.rename(columns={"value": series_id})[[series_id]]
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df

def load_fred_metrics(refresh: bool = False) -> pd.DataFrame:
    cache_file = CACHE_DIR / "fred.parquet"
    if cache_file.exists() and not refresh:
        return pd.read_parquet(cache_file)

    dfs = []
    for alias, series_id in FRED_SERIES.items():
        print(f"[fred] fetch {alias} ({series_id})")
        try:
            df = _fetch_series(series_id)
            df = df.rename(columns={series_id: alias})
            dfs.append(df)
        except Exception as e:
            print(f"[fred] fail {alias}: {e}")

    if not dfs:
        raise RuntimeError("Не удалось загрузить макро-индикаторы FRED")

    out = pd.concat(dfs, axis=1).sort_index()
    out.to_parquet(cache_file)
    return out