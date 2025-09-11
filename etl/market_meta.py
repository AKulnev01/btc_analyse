# etl/market_meta.py
import os
import time
import requests
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime, timezone

CG_PRO_BASE = "https://pro-api.coingecko.com/api/v3"
CG_PUB_BASE = "https://api.coingecko.com/api/v3"
CG_KEY = os.getenv("COINGECKO_API_KEY", "")

def _now_hour_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).floor("h")

def _get_json(path: str, params: Optional[dict] = None) -> Tuple[dict, str]:
    """Сначала пробуем PRO, потом public. Возвращаем (json, base_used)."""
    params = dict(params or {})
    headers_base = {"User-Agent": "btc-forecast/1.0"}
    bases = []
    if CG_KEY:
        bases.append((CG_PRO_BASE, {**headers_base, "x-cg-pro-api-key": CG_KEY}))
    bases.append((CG_PUB_BASE, headers_base))

    last_err: Optional[Exception] = None
    for base, hdr in bases:
        try:
            url = f"{base}{path}"
            r = requests.get(url, params=params, headers=hdr, timeout=20)
            if r.status_code == 429:
                time.sleep(1.0)
                r = requests.get(url, params=params, headers=hdr, timeout=20)
            r.raise_for_status()
            return r.json(), base
        except requests.HTTPError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("CoinGecko request failed")

def fetch_coingecko_global() -> pd.DataFrame:
    # /global
    gjson, base_used = _get_json("/global")
    g = gjson.get("data", gjson)

    total_usd = None
    btc_d = None
    try:
        total_usd = g.get("total_market_cap", {}).get("usd")
    except Exception:
        pass
    try:
        btc_d = g.get("market_cap_percentage", {}).get("btc")
    except Exception:
        pass

    # /coins/markets (tether) — чтобы получить USDT market cap
    try:
        mjson, _ = _get_json(
            "/coins/markets",
            params={"vs_currency": "usd", "ids": "tether", "per_page": 1, "page": 1, "sparkline": "false"},
        )
        usdt_mcap = mjson[0]["market_cap"] if (isinstance(mjson, list) and mjson) else None
    except Exception:
        usdt_mcap = None

    usdt_d = (usdt_mcap / total_usd * 100.0) if (usdt_mcap is not None and total_usd not in (None, 0)) else None

    out = pd.DataFrame([{
        "timestamp": _now_hour_utc(),
        "total_market_cap_usd": total_usd,
        "btc_dominance_pct": btc_d,
        "usdt_market_cap_usd": usdt_mcap,
        "usdt_dominance_pct": usdt_d,
        "cg_base_used": base_used,
    }]).set_index("timestamp")
    return out

if __name__ == "__main__":
    df_new = fetch_coingecko_global()
    path = "data/interim/global_meta.parquet"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df_old = pd.read_parquet(path)
        df = pd.concat([df_old, df_new]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
    except Exception:
        df = df_new
    df.to_parquet(path)
    print(df.tail(1))