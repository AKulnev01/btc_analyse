# models/model_zoo.py
# Унифицированный интерфейс обучения квантилей через разные бэкенды.
import os
import pickle
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

# LGBM
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# CatBoost
from catboost import CatBoostRegressor, Pool


DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
SCALE_COLS = ["sigma_ewma_7d", "sigma_7d"]


def _pick_scale_col(df: pd.DataFrame) -> str:
    for c in SCALE_COLS:
        if c in df.columns:
            return c
    raise ValueError("Не найдено ни 'sigma_ewma_7d', ни 'sigma_7d'.")


def _feature_cols(df: pd.DataFrame, feature_cols: Optional[List[str]]) -> List[str]:
    if feature_cols is None:
        bad = set(["y", "close"])
        cols = [c for c in df.columns if c not in bad and df[c].dtype.kind != "O"]
    else:
        cols = [c for c in feature_cols if c in df.columns and df[c].dtype.kind != "O"]
    return cols


# ---------- LGBM квантиль ----------
LGB_PARAMS_REG = dict(
    n_estimators=6000,
    learning_rate=0.03,
    num_leaves=255,
    max_depth=-1,
    min_data_in_leaf=25,
    min_sum_hessian_in_leaf=1e-5,
    feature_fraction=0.95,
    subsample=0.9,
    subsample_freq=1,
    reg_lambda=3.0,
    min_split_gain=0.0,
    max_bin=1023,
    force_row_wise=True,
    deterministic=True,
    seed=42,
    bagging_seed=42,
    feature_fraction_seed=42,
    verbosity=-1,
    extra_trees=True,
    path_smooth=5.0,
)

def _train_one_quantile_lgbm(X: pd.DataFrame, y_scaled: pd.Series, q: float) -> LGBMRegressor:
    split = int(len(X) * 0.8)
    Xtr, Xva = X.iloc[:split], X.iloc[split:]
    ytr, yva = y_scaled.iloc[:split], y_scaled.iloc[split:]

    model = LGBMRegressor(objective="quantile", alpha=q, **LGB_PARAMS_REG)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="l1",
        callbacks=[early_stopping(stopping_rounds=200), log_evaluation(0)]
    )
    pred = model.predict(Xva)
    mae = mean_absolute_error(yva, pred)
    print(f"[LGBM Q{int(q*100)}] CV MAE (scaled): {mae:.6f}")
    return model


# ---------- CatBoost квантиль ----------
CAT_PARAMS_BASE = dict(
    depth=8,
    learning_rate=0.03,
    iterations=6000,
    random_seed=42,
    od_type="Iter",
    od_wait=200,
    l2_leaf_reg=3.0,
    loss_function=None,   # выставим ниже как Quantile:alpha=...
    verbose=False,
)

def _train_one_quantile_cat(X: pd.DataFrame, y_scaled: pd.Series, q: float) -> CatBoostRegressor:
    split = int(len(X) * 0.8)
    Xtr, Xva = X.iloc[:split], X.iloc[split:]
    ytr, yva = y_scaled.iloc[:split], y_scaled.iloc[split:]

    params = CAT_PARAMS_BASE.copy()
    params["loss_function"] = f"Quantile:alpha={q}"
    model = CatBoostRegressor(**params)
    model.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), use_best_model=True, verbose=False)
    pred = model.predict(Xva)
    mae = mean_absolute_error(yva, pred)
    print(f"[CAT Q{int(q*100)}] CV MAE (scaled): {mae:.6f}")
    return model


def _calibrate_lambda(models, X: pd.DataFrame, y: pd.Series, sigma: pd.Series, coverage_target: float = 0.70) -> float:
    if not all(q in models for q in (0.1, 0.5, 0.9)):
        return 1.0
    split = int(len(X) * 0.8)
    Xva = X.iloc[split:]
    yva = y.iloc[split:]
    sva = sigma.iloc[split:]

    q10s = models[0.1].predict(Xva) * sva.values
    q50s = models[0.5].predict(Xva) * sva.values
    q90s = models[0.9].predict(Xva) * sva.values

    grid = np.linspace(0.5, 1.0, 51)
    covers = []
    for lam in grid:
        lo = q50s + lam * (q10s - q50s)
        hi = q50s + lam * (q90s - q50s)
        cover = np.mean((yva.values >= np.minimum(lo, hi)) & (yva.values <= np.maximum(lo, hi)))
        covers.append(cover)
    covers = np.array(covers)
    lam = float(grid[int(np.argmin(np.abs(covers - coverage_target)))])
    print(f"[calib] target={coverage_target:.2f}, lambda={lam:.2f}, raw_cover={covers.min():.3f}..{covers.max():.3f}")
    return lam


def train_quantiles_backend(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]],
    out_path: str,
    quantiles: Optional[List[float]] = None,
    coverage_target: float = 0.70,
    lambda_fixed: Optional[float] = None,
    backend: str = "lgbm",   # "lgbm" | "cat"
) -> Dict[float, object]:
    """
    Обучает квантильные модели выбранным бэкендом, сохраняет пакет в pickle.
    """
    quantiles = quantiles or DEFAULT_QUANTILES
    feats = _feature_cols(df, feature_cols)
    scale_col = _pick_scale_col(df)

    work = df.dropna(subset=["y", scale_col]).copy()
    X_all = work[feats].astype(float)
    y = work["y"].astype(float)
    sigma = work[scale_col].clip(lower=1e-8).astype(float)
    y_scaled = (y / sigma).astype(float)

    print(f"[train {backend}] rows={len(work)}, features={len(feats)}, scale_col={scale_col}")
    models: Dict[float, object] = {}

    for q in quantiles:
        print(f"[train] quantile={q}")
        if backend == "lgbm":
            models[q] = _train_one_quantile_lgbm(X_all, y_scaled, q)
        elif backend == "cat":
            models[q] = _train_one_quantile_cat(X_all, y_scaled, q)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    if lambda_fixed is not None:
        lambda_width = float(lambda_fixed)
        print(f"[calib] lambda fixed: {lambda_width:.2f}")
    else:
        lambda_width = _calibrate_lambda(models, X_all, y, sigma, coverage_target=coverage_target)

    payload = {
        "backend": backend,
        "models": models,
        "features": feats,
        "scale_col": scale_col,
        "lambda_width": float(lambda_width),
        "quantiles": quantiles,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    return models