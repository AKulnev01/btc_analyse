# models/model_zoo.py
# Унифицированный интерфейс обучения квантилей (LGBM / CatBoost / XGB-DART) + аугментации и калибровка lambda.
from __future__ import annotations

import os
import pickle
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# LGBM
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# CatBoost (может не быть установлен)
try:
    from catboost import CatBoostRegressor, Pool
    _CAT_OK = True
except Exception:
    _CAT_OK = False

# XGB делегат (опционально)
try:
    from models.xgb_quantile import train_quantiles_xgb_dart as _train_xgb_dart
    _XGB_OK = True
except Exception:
    _XGB_OK = False


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


# ---------- базовые параметры ----------
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
    allow_writing_files=False,
)


# ---------- КАЛИБРОВКА LAMBDA (ОБЯЗАТЕЛЬНО ДО ВЫЗОВА) ----------
def _calibrate_lambda(models: Dict[float, object],
                      X: pd.DataFrame,
                      y: pd.Series,
                      sigma: pd.Series,
                      coverage_target: float = 0.70) -> float:
    """
    Подбираем масштаб интервала вокруг медианы так, чтобы доля покрытий ~ coverage_target.
    Ожидаем, что models содержит квантили 0.1/0.5/0.9 (LGBM/CatBoost интерфейс .predict).
    """
    need = (0.1, 0.5, 0.9)
    if not all(q in models for q in need):
        # если нет полного набора — не калибруем
        return 1.0

    split = int(len(X) * 0.8)
    Xva = X.iloc[split:]
    yva = y.iloc[split:].values
    sva = sigma.iloc[split:].values

    q10s = np.asarray(models[0.1].predict(Xva)) * sva
    q50s = np.asarray(models[0.5].predict(Xva)) * sva
    q90s = np.asarray(models[0.9].predict(Xva)) * sva

    grid = np.linspace(0.5, 1.2, 71)
    best = 1.0
    best_diff = 1e9
    cov_min, cov_max = 1.0, 0.0

    for lam in grid:
        lo = q50s + lam * (q10s - q50s)
        hi = q50s + lam * (q90s - q50s)
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        cover = float(np.mean((yva >= lo2) & (yva <= hi2)))
        cov_min = min(cov_min, cover)
        cov_max = max(cov_max, cover)
        diff = abs(cover - coverage_target)
        if diff < best_diff:
            best_diff = diff
            best = float(lam)

    print(f"[calib] target={coverage_target:.2f}, lambda={best:.2f}, raw_cover={cov_min:.3f}..{cov_max:.3f}")
    return best


# ---------- AUGMENTATION HELPERS ----------
def _augment_xy(
    X: pd.DataFrame,
    y_scaled: pd.Series,
    sigma: Optional[pd.Series] = None,
    p_feat_drop: float = 0.0,
    x_jitter: float = 0.0,
    y_jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Простые аугментации:
      - случайный дроп некоторых фич (имитируем пропуски/шум в данных)
      - add-noise к X (джиттер)
      - шум к y_scaled (стабилизация квантилей)
    """
    rng = rng or np.random.default_rng(42)
    X_aug = X.copy()
    y_aug = y_scaled.copy()

    if x_jitter > 0:
        X_aug = X_aug + rng.normal(0.0, x_jitter, size=X_aug.shape)

    if p_feat_drop > 0:
        mask = rng.random(size=X_aug.shape) < p_feat_drop
        X_aug = X_aug.mask(mask)
        X_aug = X_aug.fillna(X_aug.mean(numeric_only=True))

    if y_jitter > 0:
        y_aug = y_aug + rng.normal(0.0, y_jitter, size=len(y_aug))

    return X_aug, y_aug


# ---------- ТРЕНЕРЫ ДЛЯ КВАНТИЛЕЙ ----------
def _train_one_quantile_lgbm(
    X: pd.DataFrame,
    y_scaled: pd.Series,
    q: float,
    aug: Optional[dict] = None
) -> LGBMRegressor:
    split = int(len(X) * 0.8)
    Xtr, Xva = X.iloc[:split], X.iloc[split:]
    ytr, yva = y_scaled.iloc[:split], y_scaled.iloc[split:]

    if aug:
        Xtr, ytr = _augment_xy(
            Xtr, ytr,
            p_feat_drop=float(aug.get("p_feat_drop", 0.0)),
            x_jitter=float(aug.get("x_jitter", 0.0)),
            y_jitter=float(aug.get("y_jitter", 0.0)),
        )

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


def _train_one_quantile_cat(
    X: pd.DataFrame,
    y_scaled: pd.Series,
    q: float,
    aug: Optional[dict] = None
) -> "CatBoostRegressor":
    if not _CAT_OK:
        raise ImportError("catboost is not installed")
    split = int(len(X) * 0.8)
    Xtr, Xva = X.iloc[:split], X.iloc[split:]
    ytr, yva = y_scaled.iloc[:split], y_scaled.iloc[split:]

    if aug:
        Xtr, ytr = _augment_xy(
            Xtr, ytr,
            p_feat_drop=float(aug.get("p_feat_drop", 0.0)),
            x_jitter=float(aug.get("x_jitter", 0.0)),
            y_jitter=float(aug.get("y_jitter", 0.0)),
        )
        Xtr = Xtr.fillna(Xtr.mean(numeric_only=True))

    params = CAT_PARAMS_BASE.copy()
    params["loss_function"] = f"Quantile:alpha={q}"

    train_pool = Pool(Xtr, ytr)
    valid_pool = Pool(Xva, yva)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)
    pred = model.predict(Xva)
    mae = mean_absolute_error(yva, pred)
    print(f"[CAT Q{int(q*100)}] CV MAE (scaled): {mae:.6f}")
    return model


# ---------- ГЛАВНАЯ ТОЧКА ВХОДА ----------
def train_quantiles_backend(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]],
    out_path: str,
    quantiles: Optional[List[float]] = None,
    coverage_target: float = 0.70,
    lambda_fixed: Optional[float] = None,
    backend: str = "lgbm",   # "lgbm" | "cat" | "xgb"
    augment: Optional[dict] = None,
) -> Dict[float, object]:
    """
    Обучает квантильные модели выбранным бэкендом, сохраняет пакет в pickle.
    Возвращает dict из моделей (для lgbm/cat). Для xgb делегирует во внешний тренер.
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

    # XGB: делегируем и выходим (файл сохранит внешний тренер)
    if backend == "xgb":
        if not _XGB_OK:
            raise ImportError("models.xgb_quantile.train_quantiles_xgb_dart not available")
        return _train_xgb_dart(
            df=work,
            feature_cols=feats,
            out_path=out_path,
            quantiles=quantiles,
            coverage_target=coverage_target,
            lambda_fixed=lambda_fixed,
        )

    # LGBM/CAT локально
    models: Dict[float, object] = {}
    for q in quantiles:
        print(f"[train] quantile={q}")
        if backend == "lgbm":
            models[q] = _train_one_quantile_lgbm(X_all, y_scaled, q, aug=augment)
        elif backend == "cat":
            models[q] = _train_one_quantile_cat(X_all, y_scaled, q, aug=augment)
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
        "feature_cols": feats,     # NB: ensemble ожидает именно 'feature_cols'
        "scale_col": scale_col,
        "lambda_width": float(lambda_width),
        "quantiles": quantiles,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    return models