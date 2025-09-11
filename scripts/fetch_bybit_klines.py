import argparse, config as CFG
from etl.bybit_rest import fetch_klines_days

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--category", default="linear")
    ap.add_argument("--interval", default="60")  # 60 = 1h
    ap.add_argument("--days", type=int, default=1200)
    args = ap.parse_args()

    df = fetch_klines_days(args.symbol, args.category, args.interval, args.days)
    df.to_parquet(CFG.PX_FILE)
    print(f"[OK] saved {len(df)} rows -> {CFG.PX_FILE}")