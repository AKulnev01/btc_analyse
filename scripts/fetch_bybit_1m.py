# scripts/fetch_bybit_1m.py
import os
import argparse
from etl.bybit_micro import ensure_1m_prices, MINUTE_PATH

def main():
    p = argparse.ArgumentParser(description="Fetch/append Bybit 1m klines and store parquet")
    p.add_argument("--days", type=int, default=365, help="how many days back to fetch")
    p.add_argument("--out", default=MINUTE_PATH, help="parquet path for 1m data")
    args = p.parse_args()
    df = ensure_1m_prices(days=args.days, path=args.out)
    print(f"[OK] saved {len(df)} rows -> {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()