"""Готовые счётчики новостей (например, из CryptoPanic), ресемплим в 1H."""
import pandas as pd

def load_news_counts(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.set_index('timestamp').sort_index().resample('1H').sum()
