# etl/funding_all.py
from __future__ import annotations
import time
from typing import Optional
import requests
import pandas as pd

def _fetch_binance_funding(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    """
    Binance futures funding history. Interval ~8h; вернём как time series.
    """
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if not js:
        return pd.DataFrame()
    df = pd.DataFrame(js)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df = df.rename(columns={"fundingRate": "binance_funding"}).set_index("fundingTime").sort_index()
    df["binance_funding"] = df["binance_funding"].astype(float)
    return df[["binance_funding"]]

def _fetch_bybit_funding(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """
    Bybit funding history (v5). Вернём последние ~200 записей.
    """
    url = "https://api.bybit.com/v5/market/funding/history"
    params = {"category": "linear", "symbol": symbol, "limit": 200}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = js.get("result", {}).get("list", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["fundingRateTimestamp"] = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
    df = df.set_index("fundingRateTimestamp").sort_index()
    return df.rename(columns={"fundingRate": "bybit_funding"})[["bybit_funding"]]

def _fetch_okx_funding(instId: str = "BTC-USDT-SWAP", limit: int = 100) -> pd.DataFrame:
    """
    OKX funding history (public).
    """
    url = "https://www.okx.com/api/v5/public/funding-rate-history"
    params = {"instId": instId, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = js.get("data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df = df.set_index("fundingTime").sort_index()
    return df.rename(columns={"fundingRate": "okx_funding"})[["okx_funding"]]

def fetch_all_funding() -> pd.DataFrame:
    b = _fetch_binance_funding()
    time.sleep(0.5)
    y = _fetch_bybit_funding()
    time.sleep(0.5)
    o = _fetch_okx_funding()
    pieces = [x for x in (b, y, o) if not x.empty]
    if not pieces:
        return pd.DataFrame()
    df = pd.concat(pieces, axis=1).sort_index()
    # добавим агрегаты
    df["funding_mean"] = df.mean(axis=1, skipna=True)
    df["funding_spread_bybit_binance"] = df.get("bybit_funding", pd.Series(index=df.index)) - df.get("binance_funding", pd.Series(index=df.index))
    df["funding_spread_okx_binance"] = df.get("okx_funding", pd.Series(index=df.index)) - df.get("binance_funding", pd.Series(index=df.index))
    return df