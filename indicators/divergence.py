"""
RSI + простая детекция классической дивергенции Цена vs RSI.
Это эвристика: ищем пара локальных экстремумов с противоположным направлением.
"""
import numpy as np
import pandas as pd

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    roll_dn = pd.Series(dn, index=close.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def _local_extrema(s: pd.Series, lookback: int = 5):
    # простой способ пометить локальные минимумы/максимумы в окне
    min_mask = (s.shift(1) > s) & (s.shift(-1) > s)
    max_mask = (s.shift(1) < s) & (s.shift(-1) < s)
    return min_mask, max_mask

def detect_divergence(df: pd.DataFrame, rsi_period: int = 14, lb: int = 48) -> pd.DataFrame:
    """
    Возвращает флаги bull_div / bear_div.
    Bullish: цена делает ниже минимум, RSI делает выше минимум.
    Bearish: цена делает выше максимум, RSI делает ниже максимум.
    """
    d = df.copy()
    d["rsi"] = rsi(d["close"], period=rsi_period)

    price_min, price_max = _local_extrema(d["close"])
    rsi_min, rsi_max     = _local_extrema(d["rsi"])

    d["bull_div"] = False
    d["bear_div"] = False

    # смотрим последние два минимума/максимума в окне lb
    for i in range(lb, len(d)):
        window = d.iloc[i-lb:i]
        p_mins = window.index[price_min.loc[window.index]]
        r_mins = window.index[rsi_min.loc[window.index]]
        if len(p_mins) >= 2 and len(r_mins) >= 2:
            p1, p2 = p_mins[-2], p_mins[-1]
            r1, r2 = r_mins[-2], r_mins[-1]
            if d.loc[p2,"close"] < d.loc[p1,"close"] and d.loc[r2,"rsi"] > d.loc[r1,"rsi"]:
                d.iloc[i, d.columns.get_loc("bull_div")] = True

        p_maxs = window.index[price_max.loc[window.index]]
        r_maxs = window.index[rsi_max.loc[window.index]]
        if len(p_maxs) >= 2 and len(r_maxs) >= 2:
            p1, p2 = p_maxs[-2], p_maxs[-1]
            r1, r2 = r_maxs[-2], r_maxs[-1]
            if d.loc[p2,"close"] > d.loc[p1,"close"] and d.loc[r2,"rsi"] < d.loc[r1,"rsi"]:
                d.iloc[i, d.columns.get_loc("bear_div")] = True

    return d[["rsi","bull_div","bear_div"]]