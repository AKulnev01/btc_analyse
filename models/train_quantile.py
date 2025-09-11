# models/train_quantile.py
# Обучение квантильных моделей LGBM с масштабированием таргета и калибровкой ширины интервала.
import os
import pickle
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error

# Параметры LGBM для квантилей (ослаблены, чтобы реже ловить "No further splits")
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
    seed=42,
    bagging_seed=42,
    feature_fraction_seed=42,
    verbosity=-1,
    # если твоя сборка lightgbm поддерживает — можно вернуть:
    # deterministic=True,
    # extra_trees=True,
    # path_smooth=5.0,
)

# Квантили по умолчанию
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
SCALE_COLS = ["sigma_ewma_7d", "sigma_7d"]  # предпочтение EWMA


def _pick_scale_col(df: pd.DataFrame) -> str:
    for c in SCALE_COLS:
        if c in df.columns:
            return c
    raise ValueError("Не найдено ни 'sigma_ewma_7d', ни 'sigma_7d' — добавь одну из них в features.")


def _feature_cols(df: pd.DataFrame, feature_cols: Optional[List[str]]) -> List[str]:
    """Вернём только числовые фичи, исключая 'y' и 'close'."""
    if feature_cols is None:
        bad = set(["y", "close"])
        cols = [c for c in df.columns if c not in bad and df[c].dtype.kind != "O"]
    else:
        cols = [c for c in feature_cols if c in df.columns and df[c].dtype.kind != "O"]
    return cols


def _select_feats(df_tr: pd.DataFrame,
                  feats_all: List[str],
                  min_cov: float = 0.90,
                  min_unique: int = 5) -> List[str]:
    """
    Отбираем фичи в train-окне:
      - покрытие not-null >= min_cov
      - число уникальных значений >= min_unique
      - ненулевая std
    Это снижает шанс "No further splits ..." и «схлопнутых» деревьев.
    """
    cov = df_tr[feats_all].notna().mean()
    cand = cov[cov >= min_cov].index.tolist()
    if not cand:
        return []

    nunique = df_tr[cand].nunique()
    cand = nunique[nunique >= min_unique].index.tolist()
    if not cand:
        return []

    std = df_tr[cand].std(numeric_only=True)
    cand = std[std > 0].index.tolist()
    return cand


def _train_one_quantile(X: pd.DataFrame, y_scaled: pd.Series, q: float) -> LGBMRegressor:
    """Учим один квантиль с валидационным хвостом 20%."""
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
    # отчёт MAE (scaled)
    pred = model.predict(Xva)
    mae = mean_absolute_error(yva, pred)
    print(f"Q{int(q*100)} CV MAE (scaled): {mae:.6f}")
    return model


def _calibrate_lambda(models: Dict[float, LGBMRegressor],
                      X: pd.DataFrame,
                      y: pd.Series,
                      sigma: pd.Series,
                      coverage_target: float = 0.70) -> float:
    """
    Подбираем λ для сжатия [Q10,Q90] к медиане так, чтобы покрытие было близко к coverage_target.
    Делается на валидационном хвосте (последние 20% трен-окна).
    """
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
    print(f"[calib] target={coverage_target:.2f}, picked lambda_width={lam:.2f}, raw_cover={covers.min():.3f}..{covers.max():.3f}")
    return lam


def train_quantiles(df: pd.DataFrame,
                    feature_cols: Optional[List[str]] = None,
                    out_path: str = "models/quantile_models.pkl",
                    quantiles: Optional[List[float]] = None,
                    coverage_target: float = 0.70,
                    lambda_fixed: Optional[float] = None,
                    min_cov: float = 0.90,
                    min_unique: int = 5) -> Dict[float, LGBMRegressor]:
    """
    Обучает квантильные модели на масштабе y/sigma и сохраняет пакет в pickle.

    Параметры:
      - coverage_target: цель покрытия для [P10,P90] при калибровке λ.
      - lambda_fixed: если задано (например, 0.69) — используем фиксированную λ.
      - min_cov / min_unique: фильтры для отбора признаков.
    """
    quantiles = quantiles or DEFAULT_QUANTILES
    feats_all = _feature_cols(df, feature_cols)
    scale_col = _pick_scale_col(df)

    # чистим строки, где нет y или σ
    work = df.dropna(subset=["y", scale_col]).copy()

    # отбор фич на всей имеющейся «work»-истории как train-окне
    feats_used = _select_feats(work, feats_all, min_cov=min_cov, min_unique=min_unique)
    if len(feats_used) == 0:
        raise ValueError("После фильтра покрытий/вариативности не осталось фичей для обучения")

    # LightGBM умеет NaN, но для стабильности уберём строки с NaN в критических фичах
    X_all = work[feats_used]
    y = work["y"].astype(float)
    sigma = work[scale_col].clip(lower=1e-8).astype(float)
    mask_rows = X_all.notna().all(axis=1) & y.notna() & sigma.notna()
    X_all = X_all.loc[mask_rows].astype(float)
    y = y.loc[mask_rows]
    sigma = sigma.loc[mask_rows]
    y_scaled = (y / sigma).astype(float)

    print(f"[train] rows={len(X_all)}, features={len(feats_used)}, scale_col={scale_col}")
    models: Dict[float, LGBMRegressor] = {}
    for q in quantiles:
        print(f"[train] quantile={q}")
        models[q] = _train_one_quantile(X_all, y_scaled, q)

    # калибровка λ
    if lambda_fixed is not None:
        lambda_width = float(lambda_fixed)
        print(f"[calib] lambda_width fixed by user: {lambda_width:.2f}")
    else:
        lambda_width = _calibrate_lambda(models, X_all, y, sigma, coverage_target=coverage_target)

    payload = {
        "models": models,
        "features": feats_used,
        "scale_col": scale_col,
        "lambda_width": float(lambda_width),
        "quantiles": quantiles,
        "params": LGB_PARAMS_REG,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    return models