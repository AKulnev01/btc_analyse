# etl/macro_fred.py
# FRED: тянем ключевые ряды и строим «сюрпризы» как отклонение от простого nowcast (EWMA).
import os, pandas as pd, numpy as np, requests
from datetime import datetime, timezone

FRED_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "CPIAUCSL": "cpi",      # CPI (YoY можно производить отдельно)
    "UNRATE": "unemp",      # Unemployment Rate
    "PAYEMS": "nfp",        # Nonfarm Payrolls
    "FEDFUNDS": "ffr",      # Fed Funds rate
    "PCEPILFE": "core_pce", # Core PCE Price Index
}

def _fred_series(series_id):
    params = {"series_id": series_id, "api_key": FRED_KEY, "file_type":"json"}
    r = requests.get(FRED_BASE, params=params, timeout=20); r.raise_for_status()
    obs = r.json().get("observations", [])
    rows = []
    for o in obs:
        try:
            ts = pd.to_datetime(o["date"], utc=True)
            v = float(o["value"])
            rows.append({"timestamp": ts, series_id: v})
        except Exception:
            pass
    return pd.DataFrame(rows).set_index("timestamp").sort_index()

def fetch_macro_nowcast():
    frames = []
    for sid in SERIES:
        try:
            frames.append(_fred_series(sid))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    macro = pd.concat(frames, axis=1).sort_index()
    # EWMA-«прогноз» и сюрприз (actual - ewma)
    out = {}
    for sid, name in SERIES.items():
        s = macro[sid].dropna()
        if s.empty:
            continue
        ew = s.ewm(halflife=6, min_periods=6).mean()
        # сюрприз на дате новой публикации
        spr = (s - ew).rename(f"surprise_{name}")
        out[f"level_{name}"] = s
        out[f"surprise_{name}"] = spr
    df = pd.DataFrame(out, index=macro.index).dropna(how="all")
    # Приводим к 1H: метка публикации -> 13:30/14:30 UTC сложно из FRED; используем дневную,
    # а затем forward-fill до конца суток (легкие флаги)
    df_h = df.resample("1H").ffill()
    return df_h

if __name__=="__main__":
    df = fetch_macro_nowcast()
    if not df.empty:
        path = "data/interim/macro_fred.parquet"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)
        print("[OK] macro_fred ->", path, "rows=", len(df))
    else:
        print("[WARN] macro_fred empty")