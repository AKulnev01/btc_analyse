# scripts/fetch_funding_all.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config as CFG
from etl.funding_all import fetch_all_funding

def main():
    import pandas as pd
    os.makedirs(CFG.EXT_DIR, exist_ok=True)
    df = fetch_all_funding()
    if df.empty:
        print("[WARN] funding empty")
        return
    df.to_parquet(CFG.FUNDING_FILE)
    print(f"[OK] funding saved -> {CFG.FUNDING_FILE}, rows={len(df)}")

if __name__ == "__main__":
    main()