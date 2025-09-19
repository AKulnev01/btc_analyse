# models/xgb_quantile.py
import os
import pickle
from typing import List, Dict
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray

# ---- сглаженный pinball (квантильный) лосс ----
def _pinball_objective(alpha: float, eps: float = 1e-3):
    # -> callable: (preds: NDArray, dtrain: xgb.DMatrix) -> tuple[NDArray, NDArray]
    def obj(preds: NDArray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        diff = y - preds  # t - yhat, shape (n,)
        # grad, hess должны быть массивами той же длины
        grad = -(alpha - 0.5 * (1.0 + np.tanh(diff / eps)))
        hess = (0.5 / eps) * (1.0 / np.cosh(diff / eps) ** 2) + 1e-6
        # гарантируем тип/форму
        grad = np.asarray(grad, dtype=np.float32)
        hess = np.asarray(hess, dtype=np.float32)
        return grad, hess
    return obj

def _pinball_metric(alpha: float):
    # -> callable: (preds: NDArray, dtrain: xgb.DMatrix) -> tuple[str, float]
    def feval(preds: NDArray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        e = y - preds
        loss = np.maximum(alpha * e, (alpha - 1) * e)
        return "pinball", float(np.mean(loss))
    return feval

# ---- параметры DART ----
XGB_PARAMS_DART = dict(
    booster="dart",
    rate_drop=0.1,
    skip_drop=0.5,
    eta=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    lambda_=1.0,
    alpha=0.0,
    tree_method="hist",
    nthread=0,
    # objective зададим callable-ом при train()
)

NUM_BOOST_ROUND = 4000
EARLY_STOPPING_ROUNDS = 300

class _BoosterWrapper:
    """Унифицированная обёртка под Booster, чтобы был метод predict(X)."""
    def __init__(self, booster: xgb.Booster):
        self.booster = booster
    def predict(self, X: np.ndarray) -> np.ndarray:
        dm = xgb.DMatrix(X)
        return self.booster.predict(dm)

def _train_one_quantile(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    alpha: float
) -> _BoosterWrapper:
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)

    evals = [(dtr, "train"), (dva, "validation_0")]
    evals_result = {}

    booster = xgb.train(
        params=XGB_PARAMS_DART,
        dtrain=dtr,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtr, "train"), (dva, "validation_0")],
        obj=_pinball_objective(alpha),  # <-- objective: градиент+гессиан (ndarray, ndarray)
        feval=_pinball_metric(alpha),  # <-- метрика: (name, value)
        maximize=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result={},
        verbose_eval=False,
    )
    # можно при желании распечатать лучшую итерацию/метрику:
    # best_it = booster.best_iteration
    # best_score = booster.best_score
    return _BoosterWrapper(booster)

def train_quantiles_xgb_dart(
    df: pd.DataFrame,
    feature_cols: List[str],
    out_path: str = "models/quantile_models_xgb.pkl",
    horizons: List[int] = [4],                  # расширь при мульти-таргете
    quantiles: List[float] = [0.1, 0.5, 0.9],
    scale_col: str = None,
) -> Dict:
    if scale_col is None:
        for c in ["sigma_ewma_7d", "sigma_7d"]:
            if c in df.columns:
                scale_col = c
                break
        if scale_col is None:
            raise ValueError("Не найдена sigma_ewma_7d/sigma_7d")

    n = len(df); split = int(n * 0.8)
    tr = df.iloc[:split].copy()
    va = df.iloc[split:].copy()

    X_tr = tr[feature_cols].astype(float).values
    X_va = va[feature_cols].astype(float).values
    y_tr = (tr["y"].astype(float) / tr[scale_col].astype(float).clip(1e-8)).values
    y_va = (va["y"].astype(float) / va[scale_col].astype(float).clip(1e-8)).values

    models = {}
    for h in horizons:
        models[h] = {}
        for q in quantiles:
            m = _train_one_quantile(X_tr, y_tr, X_va, y_va, alpha=q)
            models[h][q] = m

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            dict(models=models, feature_cols=feature_cols,
                 scale_col=scale_col, quantiles=quantiles, horizons=horizons),
        )
    return models