# indicators/regime.py
# Доп. утилиты — можно подключить позже для тонкой настройки режима рынка
import numpy as np
import pandas as pd

def volatility_regime(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """
    Простой индикатор режима: |EMA(fast)-EMA(slow)| / rolling ATR(slow)
    """
    c = close.astype(float)
    ema_f = c.ewm(span=fast, adjust=False, min_periods=max(5, fast//5)).mean()
    ema_s = c.ewm(span=slow, adjust=False, min_periods=max(5, slow//5)).mean()
    hl = c.rolling(1).max() - c.rolling(1).min()  # заглушка для совместимости
    cprev = c.shift(1)
    tr = np.maximum.reduce([hl, (c - cprev).abs(), (c - cprev).abs()])
    atr = tr.rolling(slow, min_periods=max(5, slow//5)).mean()
    score = (ema_f - ema_s).abs() / atr.replace(0, np.nan)
    out = pd.DataFrame({"regime_score": score})
    out["regime_flag"] = (out["regime_score"] > 1.0).astype(int)
    return out