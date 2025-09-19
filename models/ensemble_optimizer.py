from __future__ import annotations
import json
import pickle
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _project_to_simplex(w: np.ndarray) -> np.ndarray:
    """Проекция на симплекс {w_i>=0, sum w_i = 1} (Chen & Ye, 2011)."""
    if w.ndim != 1:
        w = w.ravel()
    n = w.size
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    w = np.maximum(w - theta, 0.0)
    s = w.sum()
    return w if s == 0 else w / s


def _predict_pack_series(df: pd.DataFrame, pack_path: str) -> pd.DataFrame:
    """
    Батч-прогноз для классического квантильного пака (LGBM/CAT/XGB).
    Возвращает DataFrame с колонками ['P10','P50','P90'] на подмножестве индекса df.
    """
    with open(pack_path, "rb") as f:
        pack = pickle.load(f)

    models = pack["models"]
    feature_cols = pack.get("feature_cols") or pack.get("features")
    if feature_cols is None:
        raise ValueError(f"{pack_path}: missing 'feature_cols'/'features'")
    scale_col = pack["scale_col"]
    quantiles = pack.get("quantiles") or [0.1, 0.5, 0.9]

    work = df.dropna(subset=["y", "close", scale_col])[feature_cols + [scale_col, "close", "y"]].copy()
    X = work[feature_cols].astype(float).values
    sigma = work[scale_col].astype(float).clip(1e-8).values
    now_price = work["close"].astype(float).values

    # структура models[q] или models[h][q]
    if isinstance(models, dict) and models:
        first_val = next(iter(models.values()))
        if isinstance(first_val, dict):
            # берём первый доступный горизонт h
            h = sorted(models.keys())[0]
            get_model = lambda q: models[h][q]
        else:
            get_model = lambda q: models[q]
    else:
        raise ValueError(f"{pack_path}: unsupported 'models' structure")

    preds_scaled = {}
    for q in quantiles:
        m = get_model(q)
        preds_scaled[q] = np.asarray(m.predict(X), dtype=float)

    # на всякий пожарный — non-crossing
    stacked = np.vstack([preds_scaled[q] for q in sorted(quantiles)])
    stacked.sort(axis=0)

    # обратно в лог-y и затем в цену
    yq = (stacked * sigma.reshape(1, -1))
    p10 = now_price * np.exp(yq[0, :])
    p50 = now_price * np.exp(yq[1, :])
    p90 = now_price * np.exp(yq[2, :])

    out = pd.DataFrame({"P10": p10, "P50": p50, "P90": p90}, index=work.index)
    return out


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if len(a) else np.nan


def optimize_weights_static(
    df: pd.DataFrame,
    classical_packs: List[str],
    target_col: str = "y",
    max_iter: int = 500,
    lr: float = 0.1,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Подбор весов по MAE(P50) на валидации для набора классических паков.
    Ограничения: w_i >= 0, sum w_i = 1 (проекция на симплекс).
    Возвращает dict с 'weights', 'mae_p50', 'n_samples'.
    """
    if len(classical_packs) == 0:
        raise ValueError("No classical packs provided")

    # Собираем P50 каждого пака и таргет (лог-y → цена для сравнения медианы в ценах)
    preds_list, idx_list = [], []
    for p in classical_packs:
        s = _predict_pack_series(df, p)["P50"]
        preds_list.append(s)
        idx_list.append(s.index)

    # пересечение индекса
    idx = idx_list[0]
    for i in range(1, len(idx_list)):
        idx = idx.intersection(idx_list[i])

    if "close" not in df.columns:
        raise ValueError("df must contain 'close' to compare in price space")

    # Истинную цену через лог-return: price_t * exp(y)
    price = df.loc[idx, "close"].astype(float).values
    y_true = df.loc[idx, target_col].astype(float).values
    price_true = price * np.exp(y_true)

    P = np.vstack([s.loc[idx].astype(float).values for s in preds_list])  # shape (M, N)
    M, N = P.shape

    # Инициализация весов равномерно
    rng = np.random.default_rng(seed)
    w = np.ones(M, dtype=float) / M

    # Простой projected gradient descent по MAE (субградиент)
    for _ in range(max_iter):
        # ансамбль
        ens = np.dot(w, P)  # (N,)
        # субградиент MAE
        g = np.sign(ens - price_true)  # (N,)
        grad = (P * g).mean(axis=1)    # (M,)
        w = w - lr * grad
        w = _project_to_simplex(w)

    mae_p50 = _mae(np.dot(w, P), price_true)

    out = {
        "weights": w.tolist(),
        "packs": classical_packs,
        "mae_p50": float(mae_p50),
        "n_samples": int(N),
    }
    if save_path:
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


def optimize_weights_with_seq(
    df: pd.DataFrame,
    classical_packs: List[str],
    seq_models: List[Dict],
    target_col: str = "y",
    max_iter: int = 500,
    lr: float = 0.1,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> Dict:
    """
    То же, но добавляем seq-модели (GRU/TCN).
    seq_models: [{"backend":"gru"|"tcn","model":"...pt","meta":"...pkl","device":"cpu|cuda"}, ...]
    """
    # классические
    preds_p50 = []
    idx_list = []
    for p in classical_packs:
        s = _predict_pack_series(df, p)["P50"]
        preds_p50.append(s)
        idx_list.append(s.index)

    # seq-медианы
    from models.seq_batch_infer import batch_predict_seq_median
    for m in seq_models or []:
        s = batch_predict_seq_median(
            df, backend=m["backend"],
            model_path=m["model"], meta_path=m["meta"],
            device=m.get("device", "cpu"),
        )  # это лог-y медиана
        # переведём в цену: price * exp(yhat50)
        idx_list.append(s.index)
        preds_p50.append((df.loc[s.index, "close"].astype(float) * np.exp(s.values)).rename("P50"))

    if len(preds_p50) == 0:
        raise ValueError("No models provided")

    # пересечение индекса
    idx = idx_list[0]
    for i in range(1, len(idx_list)):
        idx = idx.intersection(idx_list[i])

    price = df.loc[idx, "close"].astype(float).values
    y_true = df.loc[idx, target_col].astype(float).values
    price_true = price * np.exp(y_true)

    P = np.vstack([s.loc[idx].astype(float).values for s in preds_p50])  # (M, N)
    M, N = P.shape

    rng = np.random.default_rng(seed)
    w = np.ones(M, dtype=float) / M

    for _ in range(max_iter):
        ens = np.dot(w, P)
        g = np.sign(ens - price_true)
        grad = (P * g).mean(axis=1)
        w = w - lr * grad
        w = _project_to_simplex(w)

    mae_p50 = _mae(np.dot(w, P), price_true)

    out = {
        "weights": w.tolist(),
        "sources": {
            "classical_packs": classical_packs,
            "seq_models": seq_models or [],
        },
        "mae_p50": float(mae_p50),
        "n_samples": int(N),
    }
    if save_path:
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
    return out