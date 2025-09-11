"""Загрузка OHLCV и приведение к часовой сетке.
Поддерживает файлы, где timestamp либо в колонке, либо уже в индексе.
"""
import pandas as pd

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантируем DatetimeIndex в UTC, независимо от того, в колонке ли timestamp или уже в индексе."""
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.set_index('timestamp')
    else:
        # попытка интерпретировать индекс как datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
        elif df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
    return df.sort_index()

def load_btc_prices(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    df = _ensure_dt_index(df)

    # Если данные уже в 1H — ресемпл не повредит; агрегируем явно
    out = pd.DataFrame({
        'open':   df['open'].resample('1H').first(),
        'high':   df['high'].resample('1H').max(),
        'low':    df['low'].resample('1H').min(),
        'close':  df['close'].resample('1H').last(),
        'volume': df['volume'].resample('1H').sum(),
    })
    return out.dropna()