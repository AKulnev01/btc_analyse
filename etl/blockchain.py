# etl/blockchain.py
import os
import time
import requests
import pandas as pd
from pathlib import Path

BASE = "https://api.blockchain.info"
CACHE_DIR = Path("data/external/blockchain")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CHARTS = {
    "n-transactions": "tx_per_day",
    "hash-rate": "hashrate",
    "difficulty": "difficulty",
    "miners-revenue": "miners_rev_usd",
    "utxo-count": "utxo_count",
    "mempool-size": "mempool_size",
    "market-price": "btc_usd",
}

def _fetch_chart(chart_name: str, timespan="4years") -> pd.DataFrame:
    url = f"{BASE}/charts/{chart_name}?timespan={timespan}&format=json"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js["values"])
    df["t"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df.set_index("t", inplace=True)
    df = df.rename(columns={"y": chart_name})
    return df[[chart_name]]

def load_blockchain_metrics(refresh: bool = False) -> pd.DataFrame:
    """Загружает и кэширует ончейн метрики в parquet"""
    cache_file = CACHE_DIR / "onchain.parquet"
    if cache_file.exists() and not refresh:
        return pd.read_parquet(cache_file)

    dfs = []
    for api_name, alias in CHARTS.items():
        print(f"[blockchain] fetch {api_name} → {alias}")
        try:
            df = _fetch_chart(api_name)
            df = df.rename(columns={api_name: alias})
            dfs.append(df)
            time.sleep(1.5)  # защита от rate limit
        except Exception as e:
            print(f"[blockchain] fail {api_name}: {e}")

    if not dfs:
        raise RuntimeError("Не удалось загрузить метрики Blockchain.com")

    out = pd.concat(dfs, axis=1).sort_index()
    out.to_parquet(cache_file)
    return out