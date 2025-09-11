# scripts/fetch_news_cryptopanic.py
import os
import sys
import argparse

# чтобы импортировать etl.*
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from etl.news_cryptopanic import fetch_and_save_parquet

def main():
    ap = argparse.ArgumentParser(description="Fetch CryptoPanic → hourly parquet")
    ap.add_argument("--hours", type=int, default=7*24, help="сколько часов назад тянуть")
    ap.add_argument("--out", default="data/interim/news_cryptopanic.parquet", help="куда сохранить parquet")
    ap.add_argument("--currencies", default=None, help="например 'BTC,ETH'")
    ap.add_argument("--kinds", default=None, help="'news', 'media' или 'news,media'")
    ap.add_argument("--regions", default=None, help="например 'en,ru'")
    args = ap.parse_args()

    fetch_and_save_parquet(hours=args.hours, out_path=args.out,
                           currencies=args.currencies, kinds=args.kinds, regions=args.regions)

if __name__ == "__main__":
    main()