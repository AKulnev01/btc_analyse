# etl/onchain_glassnode.py
from __future__ import annotations
import os, time, math
from typing import List, Dict, Optional
import requests
import pandas as pd

def _fetch_glassnode(metric: str, api_key: str, asset: str = "BTC", interval: str = "1h",
                     params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Универсальный загрузчик Glassnode v1.
    metric: e.g. 'transactions/transfers_volume_sum', 'addresses/active_count'
    interval: '1h' | '24h'
    Возвращает DataFrame с индексом DatetimeIndex (UTC) и колонкой 'value'.
    """
    if not api_key:
        raise ValueError("GLASSNODE_API_KEY is empty")
    url = f"https://api.glassnode.com/v1/metrics/{metric}"
    q = {"api_key": api_key, "a": asset, "i": interval}
    if params:
        q.update(params)
    r = requests.get(url, params=q, timeout=60)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        return pd.DataFrame(columns=["t","v"]).assign(t=pd.NaT).dropna()
    df = pd.DataFrame(arr)
    # glassnode: keys could be time -> t / value -> v (или 'value')
    if "t" in df.columns and "v" in df.columns:
        df = df.rename(columns={"t": "time", "v": "value"})
    elif "time" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    elif "value" not in df.columns and "o" in df.columns:
        df = df.rename(columns={"o": "value"})
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    return df[["value"]].rename(columns={"value": metric.replace("/", "_")})

def fetch_onchain_bundle(api_key: str, interval: str = "1h") -> pd.DataFrame:
    """
    Сбор базового набора сигналов:
      - transactions/transfers_volume_sum         (объём трансферов)
      - addresses/active_count                    (активные адреса)
      - market/realized_cap_usd                   (реализ. капитализация)
      - supply/current                            (текущий supply)
      - miners/revenue_sum                        (доход майнеров, прокси продаж)
      - aSOPR                                     (spent output profit ratio)
      - derivatives/futures_open_interest_sum     (если доступно)
    """
    metrics = [
        "transactions/transfers_volume_sum",
        "addresses/active_count",
        "market/realized_cap_usd",
        "supply/current",
        "miners/revenue_sum",
        "indicators/asopr",  # aSOPR
    ]
    dfs = []
    for m in metrics:
        try:
            dfs.append(_fetch_glassnode(m, api_key, "BTC", interval))
            time.sleep(0.7)  # щадим API
        except Exception as e:
            print(f"[WARN] glassnode {m}: {e}")
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, axis=1).sort_index()
    # лёгкая обработка: лог-шкалы для объёмов
    for c in out.columns:
        if "volume" in c or "cap" in c or "revenue" in c or "supply" in c:
            out[f"log_{c}"] = (out[c].clip(lower=1e-9)).apply(math.log)
    return out