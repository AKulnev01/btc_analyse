# scripts/fetch_bybit_klines.py
import os
import sys
import argparse
import pandas as pd

# чтобы работал import etl.*
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config as CFG
from etl.bybit_rest import fetch_klines

def main():
    ap = argparse.ArgumentParser(description="Fetch Bybit klines (v5) and save to Parquet.")
    ap.add_argument("--symbol", default="BTCUSDT", help="e.g., BTCUSDT")
    ap.add_argument("--category", default="linear", choices=["linear","spot","inverse"])
    ap.add_argument("--interval", default="1", help="Bybit interval code, e.g. '1' for 1m, '60' for 1h")
    ap.add_argument("--start", required=True, help="start datetime (e.g. 2024-01-01T00:00:00Z)")
    ap.add_argument("--end",   required=True, help="end datetime (e.g. 2025-01-01T00:00:00Z)")
    ap.add_argument("--out", default=CFG.PX_FILE, help="output parquet path")
    args = ap.parse_args()

    df = fetch_klines(
        symbol=args.symbol,
        category=args.category,
        interval=args.interval,
        start=args.start,
        end=args.end,
        limit=1000,
        sleep=0.2,
    )
    if df.empty:
        print("[WARN] no klines returned.")
        return

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(f"[OK] saved {len(df):,} rows -> {out_path}")

if __name__ == "__main__":
    main()