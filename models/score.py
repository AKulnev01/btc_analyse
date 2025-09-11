# models/score.py
import pickle
import numpy as np
import pandas as pd


def predict_7d_price(df: pd.DataFrame, models_path: str = 'models/quantile_models.pkl') -> dict:
    """
    Прогноз цены через 7д на основе квантильных моделей.
    Ожидаем, что в models_path лежит payload вида:
      {'models': {q: LGBMRegressor(...), ...},
       'features': [list],
       'scale_col': 'sigma_ewma_7d' | 'sigma_7d',
       'lambda_width': float}
    Возвращает словарь с уровнями P10/P50/P90 и шириной диапазона.
    """
    with open(models_path, 'rb') as f:
        payload = pickle.load(f)

    qmodels = payload['models']
    feature_cols = payload['features']
    scale_col = payload.get('scale_col', 'sigma_7d')
    lam = float(payload.get('lambda_width', 1.0))

    last = df.iloc[[-1]]
    now_price = float(last['close'].iloc[0])
    sigma = float(last[scale_col].clip(lower=1e-8).iloc[0])

    # предсказания в масштабе y* (scaled), затем обратно в y
    preds_scaled = {q: float(qmodels[q].predict(last[feature_cols])[0]) for q in qmodels}
    yq = {q: preds_scaled[q] * sigma for q in preds_scaled}

    # non-crossing + shrink к медиане через λ
    qs = sorted(yq.keys())
    arr = np.array([yq[q] for q in qs])
    arr.sort()
    yq = {qs[i]: arr[i] for i in range(len(qs))}
    if 0.5 in yq and 0.1 in yq and 0.9 in yq:
        med = yq[0.5]
        yq[0.1] = med + lam * (yq[0.1] - med)
        yq[0.9] = med + lam * (yq[0.9] - med)

    P10 = now_price * np.exp(yq.get(0.1, np.nan))
    P50 = now_price * np.exp(yq.get(0.5, np.nan))
    P90 = now_price * np.exp(yq.get(0.9, np.nan))

    width_pct = 100.0 * (P90 - P10) / now_price if now_price > 0 else np.nan

    return {
        "now_price": now_price,
        "P10": float(P10),
        "P50": float(P50),
        "P90": float(P90),
        "width_pct": float(width_pct),
    }