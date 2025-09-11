"""Прокси имплайд-волы: реализованная волатильность по часам."""
import pandas as pd

def realized_vol(df_prices: pd.DataFrame, hours: int = 24) -> pd.Series:
    ret = df_prices['close'].pct_change()
    rv = ret.rolling(hours).std()
    return rv.rename('rv')
