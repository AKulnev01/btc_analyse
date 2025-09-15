# models/score.py — инференс квантилей по заданному горизонту
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd

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