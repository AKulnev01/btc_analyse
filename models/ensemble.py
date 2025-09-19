# models/ensemble.py
from __future__ import annotations
import os
import pickle
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# опционально: GP-калибратор
try:
    from models.gp_calibration import GPCalibrator
except Exception:
    GPCalibrator = None  # не критично, если модуля нет

# опционально: читаем коэффициенты адаптивной ламбды из конфигурации
try:
    import config as CFG
    _LAMBDA_BASE       = float(getattr(CFG, "LAMBDA_BASE", 0.85))
    _LAMBDA_VOL_ALPHA  = float(getattr(CFG, "LAMBDA_VOL_ALPHA", 0.25))
    _LAMBDA_NEWS_ALPHA = float(getattr(CFG, "LAMBDA_NEWS_ALPHA", 0.15))
    _BAR               = getattr(CFG, "BAR", "1H")
except Exception:
    _LAMBDA_BASE, _LAMBDA_VOL_ALPHA, _LAMBDA_NEWS_ALPHA = 0.85, 0.25, 0.15
    _BAR = "1H"

__all__ = ["predict_pack", "ensemble_predict"]


def predict_pack(df: pd.DataFrame, pack_path: str) -> Dict[str, float]:
    """Единый предикт классических квантильных паков (LGBM/CAT/XGB)."""
    with open(pack_path, "rb") as f:
        pack = pickle.load(f)
    models = pack["models"]
    feature_cols = pack.get("feature_cols") or pack.get("features")
    if feature_cols is None:
        raise ValueError("Pack missing 'feature_cols'/'features'.")
    scale_col = pack["scale_col"]
    quantiles = pack["quantiles"]

    # определить структуру: models[q] или models[h][q]
    h = None
    if isinstance(models, dict) and models:
        first_val = next(iter(models.values()))
        if isinstance(first_val, dict):  # models[h][q]
            h = sorted(models.keys())[0]

    row = df.dropna(subset=["y", scale_col]).iloc[-1]
    X = row[feature_cols].astype(float).values.reshape(1, -1)
    sigma = float(row[scale_col])
    now_price = float(row["close"])

    preds_scaled = []
    if h is not None:
        for q in quantiles:
            m = models[h][q]
            preds_scaled.append(float(m.predict(X)[0]))
    else:
        for q in quantiles:
            m = models[q]
            preds_scaled.append(float(m.predict(X)[0]))

    preds_scaled = np.array(preds_scaled, dtype=float)
    preds_scaled.sort()  # safety
    yq = preds_scaled * sigma
    return {
        "now_price": now_price,
        "P10": float(now_price * np.exp(yq[0])),
        "P50": float(now_price * np.exp(yq[1])),
        "P90": float(now_price * np.exp(yq[2])),
    }


# back-compat alias
_predict_pack = predict_pack


def _predict_seq(df: pd.DataFrame, backend: str, model_path: str, meta_path: str, device: str = "cpu") -> Dict[str, float]:
    if backend == "gru":
        from models.nn_seq import predict_now as _pred
    elif backend == "tcn":
        from models.tcn_seq import predict_now_tcn as _pred
    elif backend in ("trans", "transformer"):
        from models.transformer_seq import predict_now_transformer as _pred
    else:
        raise ValueError(f"unknown seq backend: {backend}")
    return _pred(df, model_path=model_path, meta_path=meta_path, device=device)


def _adaptive_lambda_from_context(df: pd.DataFrame) -> float:
    """
    Простая адаптивная ламбда на основе относительной волатильности и news_surprise.
    Используем последнюю строку и медиану за «окно» для нормализации.
    """
    if df.empty:
        return 1.0

    row = df.iloc[-1]
    # --- вола: sigma_ewma_7d / медиана за окно
    vol_col = "sigma_ewma_7d" if "sigma_ewma_7d" in df.columns else ("sigma_7d" if "sigma_7d" in df.columns else None)
    if _BAR == "1H":
        bars_per_day = 24
    elif _BAR == "30T":
        bars_per_day = 48
    elif _BAR == "1T":
        bars_per_day = 1440
    else:
        bars_per_day = 24

    win = max(7 * bars_per_day, 500)
    lam = float(_LAMBDA_BASE)

    if vol_col:
        ref = np.nanmedian(df[vol_col].tail(win).astype(float))
        if ref and np.isfinite(ref) and ref > 0:
            vol_rel = float(row[vol_col]) / ref
        else:
            vol_rel = 1.0
        lam *= (1.0 + _LAMBDA_VOL_ALPHA * (vol_rel - 1.0))

    # --- новости: news_surprise (если есть)
    if "news_surprise" in df.columns:
        ns = float(row["news_surprise"]) if np.isfinite(row.get("news_surprise", np.nan)) else 0.0
        # мягкая нормализация в -1..1
        ns = float(np.tanh(ns))
        lam *= (1.0 + _LAMBDA_NEWS_ALPHA * ns)

    # границы безопасности
    return float(np.clip(lam, 0.6, 1.6))


def ensemble_predict(
    df: pd.DataFrame,
    classical_packs: Optional[List[str]] = None,
    seq_models: Optional[List[Dict]] = None,
    weights: Optional[List[float]] = None,
    gp_path: Optional[str] = None,            # <-- NEW: путь к GP-калибратору
    use_adaptive_lambda: bool = True,         # <-- NEW: включать адаптивную ламбду
) -> Dict[str, float]:
    classical_packs = classical_packs or []
    seq_models = seq_models or []
    preds: List[Dict[str, float]] = []

    for p in classical_packs:
        preds.append(predict_pack(df, p))
    for m in seq_models:
        preds.append(_predict_seq(df, m["backend"], m["model"], m["meta"], device=m.get("device", "cpu")))

    if not preds:
        raise ValueError("No models provided for ensemble.")

    if weights is None:
        weights = [1.0] * len(preds)
    w = np.array(weights, dtype=float)
    w = w / w.sum()

    now_price = preds[0]["now_price"]
    P10 = float(np.sum([w[i] * preds[i]["P10"] for i in range(len(preds))]))
    P50 = float(np.sum([w[i] * preds[i]["P50"] for i in range(len(preds))]))
    P90 = float(np.sum([w[i] * preds[i]["P90"] for i in range(len(preds))]))

    # --- комбинированная калибровка ширины интервала ---
    lam_total = 1.0

    # 1) адаптивная ламбда по контексту (вола/новости)
    if use_adaptive_lambda:
        try:
            lam_ctx = _adaptive_lambda_from_context(df)
            lam_total *= float(lam_ctx)
        except Exception:
            pass

    # 2) GP-калибровка (если есть обученный калибратор)
    if gp_path and os.path.exists(gp_path) and GPCalibrator is not None:
        try:
            gp = GPCalibrator.load(gp_path)
            lam_gp = float(gp.predict_lambda(df.tail(1))[-1])
            lam_total *= lam_gp
        except Exception:
            pass

    # применяем общий множитель симметрично вокруг медианы
    if not np.isfinite(lam_total):
        lam_total = 1.0
    P10 = float(P50 + lam_total * (P10 - P50))
    P90 = float(P50 + lam_total * (P90 - P50))

    # безопасность: не дать интервалу схлопнуться/перекреститься
    P10, P90 = float(min(P10, P50)), float(max(P90, P50))
    return {"now_price": now_price, "P10": P10, "P50": P50, "P90": P90}