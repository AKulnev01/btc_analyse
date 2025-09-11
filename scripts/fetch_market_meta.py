# scripts/fetch_market_meta.py
import os
import pandas as pd
from etl.coingecko import fetch_global_meta_cg

OUT = "data/interim/global_meta.parquet"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    row = fetch_global_meta_cg()

    if os.path.exists(OUT):
        old = pd.read_parquet(OUT)
        df = pd.concat([old, row]).sort_index()
        # Удаляем дубликаты по индексу (timestamp), оставляем последнюю запись
        df = df[~df.index.duplicated(keep="last")]
    else:
        df = row

    df.to_parquet(OUT)
    print(df.tail(1))
    print(f"[OK] saved -> {OUT}")

if __name__ == "__main__":
    main()