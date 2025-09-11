import time
import math
import requests
import pandas as pd

BASE = "https://api.bybit.com"  # public REST v5

def fetch_klines(symbol="BTCUSDT", category="linear", interval="60", start_ms=None, end_ms=None, limit=1000):
    """
    interval: '60' = 1h (Bybit v5), limit<=1000
    """
    url = f"{BASE}/v5/market/kline"
    params = dict(category=category, symbol=symbol, interval=interval, limit=limit)
    if start_ms: params["start"] = int(start_ms)
    if end_ms:   params["end"]   = int(end_ms)
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("retCode") != 0:
        raise RuntimeError(f"Bybit error: {j}")
    rows = j["result"]["list"]  # newest first (по доке)
    # формат: [start, open, high, low, close, volume, turnover]
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","turnover"])
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df = df.drop(columns=["ts"]).sort_values("timestamp")
    return df

def fetch_klines_days(symbol="BTCUSDT", category="linear", interval="60", days=1200):
    """Тянем ~days назад крупными окнами, не упираясь в лимиты."""
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=days)
    # шажок: 1000×1h ≈ ~41 день за запрос
    step = pd.Timedelta(hours=1000)
    out = []
    cur = start
    while cur < end:
        nxt = min(cur + step, end)
        df = fetch_klines(symbol=symbol, category=category, interval=interval,
                          start_ms=int(cur.timestamp()*1000), end_ms=int(nxt.timestamp()*1000), limit=1000)
        out.append(df)
        cur = nxt
        time.sleep(0.2)  # не душим API
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    # унифицируем OHLCV к часовому индексу (на случай дыр)
    df = df.set_index("timestamp").sort_index()
    df = pd.DataFrame({
        "close":  df["close"].resample("1h").last(),
        "open":   df["open"].resample("1h").first(),
        "high":   df["high"].resample("1h").max(),
        "low":    df["low"].resample("1h").min(),
        "volume": df["volume"].resample("1h").sum(),
    }).dropna()
    return df