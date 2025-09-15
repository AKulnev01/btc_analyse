# models/train_quantile.py
# Мульти-таргет: обучаем квантильные модели на нескольких горизонтах (1h/4h/24h и т.д.).
# Масштабируем таргет на sigma и используем volatility-weighted sample_weight.
import os
import pickle
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error

# --------- настройки ---------
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
    verbosity=-1,
    extra_trees=True,
    path_smooth=5.0,
)

DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
SCALE_COLS = ["sigma_ewma_7d", "sigma_7d"]  # предпочтение EWMA

try:
    import config as CFG
    VOL_WEIGHT_ALPHA = float(getattr(CFG, "VOL_WEIGHT_ALPHA", 1.0))  # 0.0 чтобы выключить
    TARGET_HOURS = int(getattr(CFG, "TARGET_HOURS", 4))
    MULTI_HOURS = list(dict.fromkeys(
        [1, 4, 24] + list(getattr(CFG, "MULTI_TARGET_HOURS", [])) + [TARGET_HOURS]
    ))
except Exception:
    VOL_WEIGHT_ALPHA = 1.0
    TARGET_HOURS = 4
    MULTI_HOURS = [1, 4, 24, 4]

def _pick_scale_col(df: pd.DataFrame) -> str:
    for c in SCALE_COLS:
        if c in df.columns:
            return c
    raise ValueError("Нет 'sigma_ewma_7d' или 'sigma_7d' в фичах.")

def _feature_cols(df: pd.DataFrame, feature_cols: Optional[List[str]]) -> List[str]:
    if feature_cols is None:
        bad = set(["y", "close"])
        cols = [c for c in df.columns if c not in bad and df[c].dtype.kind != "O"]
    else:
        cols = [c for c in feature_cols if c in df.columns and df[c].dtype.kind != "O"]
    return cols

def _vol_weights(sig: pd.Series, alpha: float = 1.0) -> np.ndarray:
    if alpha <= 0:
        return np.ones(len(sig), dtype=float)
    med = float(np.nanmedian(sig.values))
    if med <= 0 or not np.isfinite(med):
        return np.ones(len(sig), dtype=float)
    w = (sig.values / med) ** alpha
    # защитные клипы, чтобы не разгонять экстремумы
    return np.clip(w, 0.25, 4.0)

def _train_one_quantile(X: pd.DataFrame, y_scaled: pd.Series, q: float, sample_weight: np.ndarray) -> LGBMRegressor:
    split = int(len(X) * 0.8)
    Xtr, Xva = X.iloc[:split], X.iloc[split:]
    ytr, yva = y_scaled.iloc[:split], y_scaled.iloc[split:]
    wtr, wva = sample_weight[:split], sample_weight[split:]

    model = LGBMRegressor(objective="quantile", alpha=q, **LGB_PARAMS_REG)
    model.fit(
        Xtr, ytr,
        sample_weight=wtr,
        eval_set=[(Xva, yva)],
        eval_metric="l1",
        callbacks=[early_stopping(stopping_rounds=200), log_evaluation(0)]
    )
    pred = model.predict(Xva)
    mae = mean_absolute_error(yva, pred)
    print(f"Q{int(q*100)} CV MAE (scaled): {mae:.6f}")
    return model

def _calibrate_lambda(models: Dict[float, LGBMRegressor],
                      X: pd.DataFrame,
                      y: pd.Series,
                      sigma: pd.Series,
                      coverage_target: float = 0.70) -> float:
    """Подбираем λ так, чтобы покрытие [P10,P90] ≈ coverage_target (на валидации)."""
    if not all(q in models for q in (0.1, 0.5, 0.9)):
        return 1.0
    split = int(len(X) * 0.8)
    Xva, yva, sva = X.iloc[split:], y.iloc[split:], sigma.iloc[split:]

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
    print(f"[calib] target={coverage_target:.2f}, picked lambda_width={lam:.2f}, raw_cover={covers.min():.3f}..{covers.max():.3f}")
    return lam

def train_quantiles(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    out_path: str = "models/quantile_models.pkl",
    quantiles: Optional[List[float]] = None,
    coverage_target: float = 0.70,
    lambda_fixed: Optional[float] = None,
    horizons: Optional[List[int]] = None,
) -> Dict[int, Dict[float, LGBMRegressor]]:
    """
    Обучает квантильные модели на нескольких горизонтах (по лог-доходности y_h{H}).
    Сохраняет пакет в pickle со структурой by_horizon[H].
    """
    quantiles = quantiles or DEFAULT_QUANTILES
    feats = _feature_cols(df, feature_cols)
    scale_col = _pick_scale_col(df)
    horizons = horizons or list(dict.fromkeys(MULTI_HOURS))

    # базовые наборы (одни для всех горизонтов)
    work = df.dropna(subset=[scale_col]).copy()
    X_all = work[feats].astype(float)
    sigma = work[scale_col].clip(lower=1e-8).astype(float)
    sample_weight = _vol_weights(sigma, alpha=VOL_WEIGHT_ALPHA)

    print(f"[train] rows={len(work)}, features={len(feats)}, scale_col={scale_col}, horizons={horizons}")

    by_horizon: Dict[int, Dict] = {}
    for H in horizons:
        ycol = f"y_h{H}" if f"y_h{H}" in work.columns else "y"
        wH = work.dropna(subset=[ycol]).copy()
        if wH.empty:
            print(f"[WARN] skip horizon {H}h: no target {ycol}")
            continue

        X = wH[feats].astype(float)
        y = wH[ycol].astype(float)
        s = wH[scale_col].clip(lower=1e-8).astype(float)
        y_scaled = (y / s).astype(float)
        sw = _vol_weights(s, alpha=VOL_WEIGHT_ALPHA)

        models: Dict[float, LGBMRegressor] = {}
        print(f"[train:H={H}] quantiles={quantiles}")
        for q in quantiles:
            models[q] = _train_one_quantile(X, y_scaled, q, sw)

        if lambda_fixed is not None:
            lambda_width = float(lambda_fixed)
            print(f"[calib:H={H}] lambda_width fixed: {lambda_width:.2f}")
        else:
            lambda_width = _calibrate_lambda(models, X, y, s, coverage_target=coverage_target)

        by_horizon[H] = dict(
            models=models,
            features=feats,
            scale_col=scale_col,
            lambda_width=float(lambda_width),
            quantiles=quantiles,
            params=LGB_PARAMS_REG,
        )

    payload = dict(
        by_horizon=by_horizon,
        horizons=sorted(list(by_horizon.keys())),
        default_horizon=int(getattr(CFG, "TARGET_HOURS", TARGET_HOURS)),
        coverage_target=float(coverage_target),
        vol_weight_alpha=float(VOL_WEIGHT_ALPHA),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    return {H: by_horizon[H]["models"] for H in by_horizon}