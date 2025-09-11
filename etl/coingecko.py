# etl/coingecko.py
import time
import requests
import pandas as pd
from datetime import datetime, timezone
import config as CFG

# попробуем сначала pro, затем публичный
_BASES = [
    "https://pro-api.coingecko.com/api/v3",
    "https://api.coingecko.com/api/v3",
]

def _headers():
    h = {"Accept": "application/json"}
    if CFG.COINGECKO_KEY:
        # CoinGecko Pro принимает заголовок x-cg-pro-api-key
        h["x-cg-pro-api-key"] = CFG.COINGECKO_KEY
        # некоторые клиенты используют снейк-кейс — добавим на всякий случай
        h["x_cg_pro_api_key"] = CFG.COINGECKO_KEY
    return h

def _get(path: str, params: dict | None = None, retries: int = 3, cooldown: float = 0.5) -> dict:
    params = dict(params or {})
    last_err = None
    for _ in range(retries):
        for base in _BASES:
            try:
                r = requests.get(f"{base}{path}", params=params, headers=_headers(), timeout=20)
                if r.status_code in (429, 503):
                    time.sleep(cooldown); continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                continue
        time.sleep(cooldown)
    raise last_err  # type: ignore[misc]

def _current_hour_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).floor("h")

def fetch_global_meta_cg() -> pd.DataFrame:
    """
    Возвращает одну строку на текущий час:
      total_market_cap_usd, btc_dominance_pct, usdt_market_cap_usd
    """
    ts = _current_hour_utc()

    g = _get("/global", {})
    total_mc = float(g["data"]["total_market_cap"]["usd"])
    mcap_pct = g["data"].get("market_cap_percentage", {}) or {}
    btc_dom = float(mcap_pct.get("btc", 0.0))  # %
    # попробуем сразу взять USDT из процента, если он есть
    if "usdt" in mcap_pct and mcap_pct["usdt"] not in (None, 0):
        usdt_mc = total_mc * float(mcap_pct["usdt"]) / 100.0
    else:
        # fallback: напрямую спросим market cap Tether
        sp = _get("/simple/price", {
            "ids": "tether",
            "vs_currencies": "usd",
            "include_market_cap": "true"
        })
        usdt_mc = float(sp.get("tether", {}).get("usd_market_cap", 0.0))

    df = pd.DataFrame([{
        "total_market_cap_usd": total_mc,
        "btc_dominance_pct": btc_dom,
        "usdt_market_cap_usd": usdt_mc,
    }], index=[ts])
    df.index.name = "timestamp"
    return df