import requests
import pandas as pd
import config as CFG

FRED_KEY = CFG.FRED_API_KEY
BASE = "https://api.stlouisfed.org/fred/series/observations"

# подберём несколько полезных серий (можно расширять)
SERIES = {
    "CPIAUCSL": "cpi",      # CPI (м/м) — здесь уровень, дальше можно брать pct_change
    "UNRATE":   "unemp",    # Unemployment rate
    "DFF":      "dff",      # Fed Funds rate
    "T10Y2Y":   "t10y2y",   # 10Y-2Y Treasury spread
    "ICSA":     "claims",   # Initial Jobless Claims
}

def fred_series(series_id: str) -> pd.DataFrame:
    params = dict(series_id=series_id, api_key=FRED_KEY, file_type="json", observation_start="2010-01-01")
    r = requests.get(BASE, params=params, timeout=20); r.raise_for_status()
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)
    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["timestamp","value"]].set_index("timestamp").sort_index()

def fetch_fred_macro() -> pd.DataFrame:
    out = []
    for sid, name in SERIES.items():
        df = fred_series(sid).rename(columns={"value": name})
        out.append(df)
    df = pd.concat(out, axis=1).sort_index()
    # ресемплим в 1h, ffill «ступеньками»
    df_h = df.resample("1h").ffill()
    # производные признаки (z-score грубо, и изменения)
    for c in df_h.columns:
        df_h[f"{c}_chg"] = df_h[c].diff()
    return df_h

if __name__ == "__main__":
    df = fetch_fred_macro()
    df.to_parquet(CFG.MACRO_FILE)
    print(df.tail(3))