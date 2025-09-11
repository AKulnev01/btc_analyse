"""Простая оценка: MAE, hit±$200, покрытие [P10;P90]."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def evaluate(df_preds: pd.DataFrame) -> dict:
    y_true = df_preds['y_true']               # фактическая 7д лог-доходность
    p0 = df_preds['price_now']
    yhat50 = np.log(df_preds['P50']/p0)       # предсказанная лог-доходность
    mae = mean_absolute_error(y_true, yhat50)

    target_price = p0 * np.exp(y_true)
    hit200 = (np.abs(df_preds['P50'] - target_price) <= 200).mean()

    lower = np.log(df_preds['P10']/p0)
    upper = np.log(df_preds['P90']/p0)
    inside = ((y_true >= lower) & (y_true <= upper)).mean()

    return {'MAE_log': float(mae), 'hit_±200': float(hit200), 'inside_10_90': float(inside)}
