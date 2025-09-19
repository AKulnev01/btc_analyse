# models/walkforward.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class WFConfig:
    train_days: int = 365
    val_days: int = 60
    step_days: int = 30          # шаг окна вперёд
    min_rows: int = 2000         # защита от слишком коротких срезов

def _as_days(freq_minutes: int, days: int) -> int:
    return int((days * 24 * 60) // max(1, freq_minutes))

def walkforward_slices(
    df: pd.DataFrame,
    cfg: WFConfig,
    freq_minutes: int = 60
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Возвращает генератор (idx_train, idx_val) по времени.
    df — индекс по времени (DatetimeIndex) уже отсортирован.
    """
    assert isinstance(df.index, pd.DatetimeIndex), "df.index must be DatetimeIndex"
    n_train = _as_days(freq_minutes, cfg.train_days)
    n_val   = _as_days(freq_minutes, cfg.val_days)
    n_step  = _as_days(freq_minutes, cfg.step_days)

    if len(df) < (n_train + n_val + 1) or len(df) < cfg.min_rows:
        return

    start = 0
    end = n_train + n_val
    while end <= len(df):
        tr_slice = np.arange(start, start + n_train)
        va_slice = np.arange(start + n_train, start + n_train + n_val)
        yield tr_slice, va_slice
        start += n_step
        end = start + n_train + n_val

def wf_evaluate_quantiles(
    df: pd.DataFrame,
    feature_cols: List[str],
    scale_col: str,
    backend_fit,          # функция (X_tr, y_tr_scaled, q) -> model
    backend_predict,      # функция (model, X) -> y_scaled_pred
    quantiles: List[float] = [0.1, 0.5, 0.9],
    freq_minutes: int = 60,
    cfg: Optional[WFConfig] = None,
) -> Dict[str, float]:
    """
    Универсальная оценка через walk-forward для квантильных моделей.
    Возвращает средние по фолдам: MAE_50 (в лог-единицах), COVER_10_90.
    """
    from sklearn.metrics import mean_absolute_error
    cfg = cfg or WFConfig()
    work = df.dropna(subset=["y", scale_col]).copy()
    X_all = work[feature_cols].astype(float)
    y = work["y"].astype(float).values
    sigma = work[scale_col].astype(float).clip(1e-8).values

    maes: List[float] = []
    covers: List[float] = []

    for tr_idx, va_idx in walkforward_slices(work, cfg, freq_minutes=freq_minutes):
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue

        # train (по каждому q)
        models = {}
        y_scaled_tr = (y[tr_idx] / sigma[tr_idx])
        for q in quantiles:
            models[q] = backend_fit(X_all.iloc[tr_idx], y_scaled_tr, q)

        # predict на валидации
        q_preds_scaled = {}
        for q in quantiles:
            q_preds_scaled[q] = backend_predict(models[q], X_all.iloc[va_idx])

        # восстановим лог-доходность и метрики
        yhat50 = q_preds_scaled[0.5] * sigma[va_idx]
        mae = mean_absolute_error(y[va_idx], yhat50)
        q10 = q_preds_scaled[0.1] * sigma[va_idx]
        q90 = q_preds_scaled[0.9] * sigma[va_idx]
        cover = np.mean((y[va_idx] >= np.minimum(q10, q90)) & (y[va_idx] <= np.maximum(q10, q90)))

        maes.append(mae)
        covers.append(float(cover))

    return {
        "MAE50_log_mean": float(np.mean(maes)) if maes else np.nan,
        "COVER_10_90_mean": float(np.mean(covers)) if covers else np.nan,
        "folds": int(len(maes)),
    }