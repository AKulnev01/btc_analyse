# scripts/build_news_sentiment.py
import os, sys, pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config as CFG
from etl.news_sentiment import build_news_sentiment

def main():
    os.makedirs(CFG.EXT_DIR, exist_ok=True)
    if not (CFG.NEWS_FILE and os.path.exists(CFG.NEWS_FILE)):
        print("[WARN] NEWS_FILE not found; skip")
        return
    news = pd.read_parquet(CFG.NEWS_FILE)
    s = build_news_sentiment(news)
    s.to_parquet(CFG.SENTI_FILE)
    print(f"[OK] news sentiment saved -> {CFG.SENTI_FILE}, rows={len(s)}")

if __name__ == "__main__":
    main()