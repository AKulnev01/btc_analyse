# etl/macro_fetch.py
import os, requests, pandas as pd

FRED_KEY = os.getenv("FRED_API_KEY","")
TE_KEY   = os.getenv("TE_API_KEY","")

def fred_series(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(url, params={"series_id": series_id, "api_key": FRED_KEY,
                                  "file_type": "json"}, timeout=20).json()
    obs = pd.DataFrame(r["observations"])
    obs["timestamp"] = pd.to_datetime(obs["date"], utc=True)
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    return obs.set_index("timestamp")[["value"]].rename(columns={"value":series_id})

def te_calendar(d1, d2, country="United States"):
    url = "https://api.tradingeconomics.com/calendar"
    r = requests.get(url, params={"d1": d1, "d2": d2, "c": country, "format":"json",
                                  "client": TE_KEY.split(":")[0],
                                  "client_secret": TE_KEY.split(":")[1]},
                     timeout=30).json()
    rows=[]
    for it in r:
        if not it.get("Date") or not it.get("Event"):
            continue
        ts = pd.to_datetime(it["Date"], utc=True)
        actual = it.get("Actual")
        forecast = it.get("Forecast")
        rows.append({"timestamp": ts, "event": it["Event"],
                     "actual": actual, "forecast": forecast})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    # окна вокруг CPI/NFP/FOMC
    def flag(event_substr, win_before="2H", win_after="4H"):
        mask = df["event"].str.contains(event_substr, case=False, na=False)
        idx = pd.date_range(df.index.min().floor("h")-pd.Timedelta("2D"),
                            df.index.max().ceil("h")+pd.Timedelta("2D"), freq="h", tz="UTC")
        out = pd.Series(0, index=idx)
        for ts in df.index[mask]:
            s = (ts - pd.Timedelta(win_before)).floor("h")
            e = (ts + pd.Timedelta(win_after)).ceil("h")
            out.loc[s:e] = 1
        return out.astype("int8")
    f = pd.DataFrame({
        "macro_cpi_win":  flag("CPI"),
        "macro_nfp_win":  flag("Non-Farm|Employment", win_before="1H", win_after="6H"),
        "macro_fomc_win": flag("FOMC|Federal Funds|Rate Decision")
    })
    return f, df  # флажки + «сырые» значения с actual/forecast

if __name__ == "__main__":
    os.makedirs("data/interim", exist_ok=True)
    # FRED time series
    series = ["CPIAUCSL","UNRATE","FEDFUNDS"]
    fred = [fred_series(s) for s in series if FRED_KEY]
    if fred:
        fred_df = pd.concat(fred, axis=1).resample("1h").ffill()
        fred_df.to_parquet("data/interim/fred_series.parquet")
        print("[OK] FRED series -> data/interim/fred_series.parquet")
    # TE calendar (пример: 6 месяцев назад… + 2 месяца вперёд)
    if TE_KEY:
        flags, raw = te_calendar(d1=pd.Timestamp.utcnow().date()-pd.Timedelta("180D"),
                                 d2=pd.Timestamp.utcnow().date()+pd.Timedelta("60D"))
        flags.to_parquet("data/interim/macro_flags.parquet")
        raw.to_parquet("data/interim/macro_raw.parquet")
        print("[OK] TE calendar flags/raw saved")