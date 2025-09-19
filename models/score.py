# models/score.py — инференс квантилей по заданному горизонту
import pickle
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

def _unscale_to_prices(now_price: float, yq_scaled: np.ndarray, sigma: float, quantiles: List[float]) -> Dict[str,float]:
    yq = yq_scaled * sigma
    yq.sort()
    return {
        "P10": float(now_price * np.exp(yq[0])),
        "P50": float(now_price * np.exp(yq[1])),
        "P90": float(now_price * np.exp(yq[2])),
    }

def _predict_quantiles_for_row(row: pd.Series,
                               bundle: Dict,
                               qmodels: Dict[float, object]) -> Dict[str, float]:
    feats = bundle["features"]
    scale_col = bundle.get("scale_col", "sigma_ewma_7d")
    lam = float(bundle.get("lambda_width", 1.0))

    x = row[feats].to_frame().T.astype(float)
    now_price = float(row["close"])
    sigma = float(max(float(row.get(scale_col, np.nan)), 1e-8))

    # scaled y*
    yq = {q: float(qmodels[q].predict(x)[0]) * sigma for q in qmodels}
    qs = sorted(yq.keys())
    vals = np.array([yq[q] for q in qs], dtype=float)
    vals.sort()
    yq = {qs[i]: float(vals[i]) for i in range(len(qs))}

    med = float(yq.get(0.5, 0.0))
    y_lo = med + lam * (float(yq.get(0.1, med)) - med)
    y_hi = med + lam * (float(yq.get(0.9, med)) - med)

    P10 = now_price * np.exp(y_lo)
    P50 = now_price * np.exp(med)
    P90 = now_price * np.exp(y_hi)
    width_pct = 100.0 * (P90 - P10) / now_price if now_price > 0 else float("nan")
    return {"now_price": now_price, "P10": P10, "P50": P50, "P90": P90, "width_pct": float(width_pct)}

def predict_price(df: pd.DataFrame,
                  models_path: str = "models/quantile_models.pkl",
                  horizon_hours: Optional[int] = None) -> Dict[str, float]:
    """Берём последнюю строку df и считаем P10/P50/P90 для горизонта."""
    with open(models_path, "rb") as f:
        payload = pickle.load(f)

    last = df.iloc[[-1]]
    # новый формат: payload['by_horizon'][H]
    if "by_horizon" in payload:
        if horizon_hours is None:
            horizon_hours = int(payload.get("default_horizon", 4))
        # ближайший доступный
        Hs = sorted([int(h) for h in payload["by_horizon"].keys()])
        H = min(Hs, key=lambda x: abs(x - int(horizon_hours)))
        bundle = payload["by_horizon"][H]
        return _predict_quantiles_for_row(last.iloc[0], bundle, bundle["models"])
    else:
        # старый формат
        bundle = payload
        return _predict_quantiles_for_row(last.iloc[0], bundle, bundle["models"])

def predict_quantiles(df: pd.DataFrame, models_path: str, backend: str = "lgbm") -> Dict[str, float]:
    """
    Унифицированный предикт бустингов (LGBM/Cat/XGB). Предполагаем, что модели обучались на scaled y.
    """
    with open(models_path, "rb") as f:
        pack = pickle.load(f)
    feature_cols = pack["feature_cols"]; scale_col = pack["scale_col"]; quantiles = pack["quantiles"]
    models = pack["models"]  # dict[h][q] -> booster
    # берём последнюю точку
    row = df.dropna(subset=["y", scale_col]).iloc[-1]
    X = row[feature_cols].astype(float).values.reshape(1, -1)
    sigma = float(row[scale_col]); now_price=float(row["close"])

    # возьмём, например, горизонт h=4 (или любой доступный)
    h = sorted(models.keys())[0]
    preds_scaled = []
    for q in quantiles:
        m = models[h][q]
        preds_scaled.append(float(m.predict(X)[0]))
    preds_scaled = np.array(preds_scaled)
    out = _unscale_to_prices(now_price, preds_scaled, sigma, quantiles)
    out["now_price"]=now_price
    return out

def predict_quantiles_seq(df: pd.DataFrame, model_path: str, meta_path: str, backend: str = "gru", device: str = "cpu") -> Dict[str, float]:
    """
    Унифицированный предикт seq-моделей.
    """
    if backend == "gru":
        from .nn_seq import predict_now as _pred
    elif backend == "tcn":
        from .tcn_seq import predict_now_tcn as _pred
    else:
        raise ValueError(f"unknown seq backend: {backend}")
    return _pred(df, model_path=model_path, meta_path=meta_path, device=device)