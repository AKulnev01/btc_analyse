# etl/bybit_rest.py
import time
import math
import json
import requests
import pandas as pd

BYBIT_BASE = "https://api.bybit.com"

# Bybit v5 /market/kline (публичный, без ключа)
# Документация: GET /v5/market/kline?category=linear|spot&symbol=...&interval=1&start=...&end=...&limit=... (<=1000)
# interval: "1","3","5","15","30","60","120","240","360","720","D","M","W"
INTERVAL_TO_MIN = {
    "1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60,
    "120": 120, "240": 240, "360": 360, "720": 720
}

def _to_ms(ts):
    """Принимает строки/дату/таймстамп -> миллисекунды UTC."""
    if isinstance(ts, (int, float)):
        # уже unix сек или мс — нормализуем
        if ts > 1e12:
            return int(ts)
        return int(ts * 1000)
    t = pd.to_datetime(ts, utc=True)
    return int(t.value // 10**6)

def fetch_klines(symbol="BTCUSDT", category="linear", interval="1",
                 start=None, end=None, limit=1000, sleep=0.25) -> pd.DataFrame:
    """
    Скачивает свечи [start, end] включительно. interval="1" == 1m.
    category: "linear" (перпеты), "spot" (спот), "inverse" — по необходимости.
    Возвращает DF с индексом UTC DatetimeIndex и колонками:
    open, high, low, close, volume, turnover.
    """
    assert interval in INTERVAL_TO_MIN or interval in ("D", "W", "M")
    if start is None or end is None:
        raise ValueError("start/end обязательны")

    start_ms = _to_ms(start)
    end_ms   = _to_ms(end)

    out = []
    params_base = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": min(int(limit), 1000),
    }

    # шаг по времени: для минут/часов считаем из таблицы, для D/W/M — возьмём суток
    if interval in INTERVAL_TO_MIN:
        step_min = INTERVAL_TO_MIN[interval]
        step_ms = step_min * 60 * 1000 * params_base["limit"]
    else:
        # D/W/M — оценочно: 1D * limit
        step_ms = 24 * 60 * 60 * 1000 * params_base["limit"]

    cur_start = start_ms
    last_progress = -1

    while cur_start <= end_ms:
        cur_end = min(end_ms, cur_start + step_ms - 1)
        params = dict(params_base)
        params["start"] = cur_start
        params["end"]   = cur_end

        r = requests.get(f"{BYBIT_BASE}/v5/market/kline", params=params, timeout=20)
        # 429/5xx — подождём и повторим
        if r.status_code >= 500 or r.status_code == 429:
            time.sleep(1.0)
            r = requests.get(f"{BYBIT_BASE}/v5/market/kline", params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        if j.get("retCode") != 0:
            # мягкая пауза и ещё попытка
            time.sleep(1.0)
            r = requests.get(f"{BYBIT_BASE}/v5/market/kline", params=params, timeout=20)
            r.raise_for_status()
            j = r.json()
            if j.get("retCode") != 0:
                raise RuntimeError(f"Bybit error: {j.get('retCode')} {j.get('retMsg')} | {json.dumps(params)}")

        rows = j.get("result", {}).get("list", []) or []
        # Bybit возвращает в порядке от новых к старым для v5 — перевернём
        rows = list(reversed(rows))

        for item in rows:
            # формат: [startTime(ms), open, high, low, close, volume, turnover]
            ts_ms = int(item[0])
            out.append({
                "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                "open":     float(item[1]),
                "high":     float(item[2]),
                "low":      float(item[3]),
                "close":    float(item[4]),
                "volume":   float(item[5]),  # qty
                "turnover": float(item[6]),  # quote value
            })

        if rows:
            # следующий блок — от последней свечи + 1шаг
            last_ts = int(rows[-1][0])
            # шаг = интервал в мс
            if interval in INTERVAL_TO_MIN:
                inc_ms = INTERVAL_TO_MIN[interval] * 60 * 1000
            else:
                inc_ms = 24 * 60 * 60 * 1000  # для D/W/M
            cur_start = last_ts + inc_ms
        else:
            # данных нет — выходим
            break

        # прогресс-лог (каждые ~10%)
        total_span = max(end_ms - start_ms, 1)
        done = cur_start - start_ms
        pct = int(100.0 * done / total_span)
        if pct >= last_progress + 10:
            print(f"[Bybit] progress ~{pct}%")
            last_progress = pct

        if sleep:
            time.sleep(sleep)

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out).set_index("timestamp").sort_index()
    # гарантируем, что все числа — float
    for c in ("open","high","low","close","volume","turnover"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df