# models/lambda_news.py
# Динамическая ширина интервала: базовая λ из train + расширение по новостям/режиму волы.
import numpy as np

def dynamic_lambda(base_lambda, news_burst_z=None, sigma=None, sigma_q1=None, sigma_q2=None):
    lam = float(base_lambda)
    # новостной множитель: z>0 -> расширяем, saturate
    if news_burst_z is not None:
        z = max(0.0, float(news_burst_z))
        lam *= (1.0 + min(0.6, 0.15 * z))  # до +60%

    # режим волы: HIGH расширяем, LOW можно чуть сжать
    if sigma is not None and sigma_q1 is not None and sigma_q2 is not None:
        if sigma <= sigma_q1:
            lam *= 0.95
        elif sigma > sigma_q2:
            lam *= 1.10
    # clip
    return float(np.clip(lam, 0.5, 2.0))