# models/calibration_gp.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

@dataclass
class GPCalib:
    gp_bias: GaussianProcessRegressor
    gp_scale: Optional[GaussianProcessRegressor]
    features: List[str]
    scale_col: str

def _design(df: pd.DataFrame, features: List[str]) -> np.ndarray:
    X = []
    for f in features:
        v = df[f].astype(float).values
        med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-9
        X.append((v - med)/mad)
    return np.vstack(X).T

def fit_gp_calibration(
    df: pd.DataFrame,
    pack_path: str,
    features: List[str] = ["sigma_ewma_7d", "vol_of_vol"],
    scale_col: Optional[str] = None,
) -> GPCalib:
    import pickle
    with open(pack_path, "rb") as f: pack = pickle.load(f)
    models = pack["models"]
    feat_cols = pack["feature_cols"]
    scale_col = scale_col or pack["scale_col"]
    quantiles = pack["quantiles"]

    # валидация — хвост 20%
    n = len(df); split = int(n*0.8)
    va = df.iloc[split:].dropna(subset=["y", scale_col]).copy()
    Xva = va[feat_cols].astype(float).values
    sigma = va[scale_col].astype(float).clip(1e-8).values
    y = va["y"].astype(float).values

    q10s = models[0.1].predict(Xva) * sigma
    q50s = models[0.5].predict(Xva) * sigma
    q90s = models[0.9].predict(Xva) * sigma
    hw = 0.5*((q90s - q50s)+(q50s - q10s))

    # таргеты для GP
    r_bias = y - q50s          # смещение медианы
    r_scale = np.abs(y - q50s) / (hw + 1e-9)  # относительная «ширина»

    Xdrv = _design(va, features)
    # ядро для gp_bias
    kernel_b = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(Xdrv.shape[1]), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)
    gp_bias = GaussianProcessRegressor(kernel=kernel_b, alpha=1e-6, normalize_y=True)
    gp_bias.fit(Xdrv, r_bias)

    # можно обучить GP и на scale (необязательно)
    kernel_s = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(Xdrv.shape[1]), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)
    gp_scale = GaussianProcessRegressor(kernel=kernel_s, alpha=1e-6, normalize_y=True)
    gp_scale.fit(Xdrv, r_scale)

    return GPCalib(gp_bias=gp_bias, gp_scale=gp_scale, features=features, scale_col=scale_col)

def apply_gp_calibration(
    df_last: pd.DataFrame,
    base_pred: Dict[str, float],
    gpcal: GPCalib,
    widen_beta: float = 0.5,   # насколько учитывать неопределённость bias для расширения
) -> Dict[str, float]:
    Xdrv = _design(df_last, gpcal.features)
    # скорректируем медиану
    mu_bias, std_bias = gpcal.gp_bias.predict(Xdrv, return_std=True)
    P50c = base_pred["P50"] * float(np.exp(mu_bias[0]))  # переводим из лог-масштаба в цену

    # расширим интервал: комбинация базовой ширины и неопределённости bias
    base_hw = 0.5 * ((base_pred["P90"] - base_pred["P50"]) + (base_pred["P50"] - base_pred["P10"]))
    hw_add = float(widen_beta * std_bias[0] * base_pred["now_price"])  # грубая шкала к цене
    hw_new = max(1.0, base_hw + hw_add)

    P10c = float(P50c - hw_new)
    P90c = float(P50c + hw_new)
    return {"now_price": base_pred["now_price"], "P10": P10c, "P50": P50c, "P90": P90c}