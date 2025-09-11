# etl/fear_greed.py
import os, requests, pandas as pd
URL = "https://api.alternative.me/fng/?limit=0&format=json"
if __name__ == "__main__":
    r = requests.get(URL, timeout=20).json()["data"]
    df = pd.DataFrame(r)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).floor("h")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    out = (df[["timestamp","value"]]
           .groupby("timestamp").last()
           .rename(columns={"value":"fear_greed"}))
    out.to_parquet("data/interim/fear_greed.parquet")
    print(out.tail(3))