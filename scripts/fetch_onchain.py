# scripts/fetch_onchain.py
import os, sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config as CFG
from etl.onchain_glassnode import fetch_onchain_bundle

def main():
    os.makedirs(CFG.EXT_DIR, exist_ok=True)
    df = fetch_onchain_bundle(CFG.GLASSNODE_API_KEY, interval="1h")
    if df.empty:
        print("[WARN] on-chain empty")
        return
    df.to_parquet(CFG.ONCHAIN_FILE)
    print(f"[OK] on-chain saved -> {CFG.ONCHAIN_FILE}, rows={len(df)}")

if __name__ == "__main__":
    main()