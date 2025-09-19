# evaluation/adaptive_calibration.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

@dataclass
class AdaptiveLambdaModel:
    coef_: np.ndarray
    intercept_: float
    features_: List[str]
    base_lambda_: float
    target_cover_: float

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        x = X[self.features_].values.astype(float)
        lam = self.intercept_ + x @ self.coef_
        return np.clip(lam, 0.5, 2.0)

def _robust_z(v: pd.Series) -> pd.Series:
    v = v.astype(float)
    med = v.median()
    mad = (v - med).abs().median() + 1e-9
    return (v - med) / mad

def _build_design(df: pd.DataFrame, scale_col: str, extra_feats: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Формируем дизайн-матрицу драйверов для адаптивной λ.
    Всегда включаем z-score волатильности, опционально — news_surprise и др.
    """
    feats = {}
    # волатильность
    if scale_col not in df.columns:
        raise ValueError(f"scale_col '{scale_col}' not found")
    sigma = df[scale_col].astype(float).clip(1e-8)
    feats["z_sigma"] = _robust_z(sigma)

    # auto-активация news_surprise, если колонка есть
    auto_news = "news_surprise" in df.columns
    extra_feats = list(extra_feats) if extra_feats is not None else []
    if auto_news and "news_surprise" not in extra_feats:
        extra_feats.append("news_surprise")

    # дополнительные драйверы (в т.ч. news_surprise)
    for name in extra_feats:
        if name in df.columns:
            z = _robust_z(df[name])
            feats[f"z_{name}"] = z
            # добавим модуль — часто лучше коррелирует с расширением интервала
            feats[f"z_abs_{name}"] = _robust_z(df[name].abs())

    return pd.DataFrame(feats, index=df.index)

def fit_adaptive_lambda(
    df: pd.DataFrame,
    pack_path: str,
    target_cover: float = 0.70,
    extra_feats: Optional[List[str]] = None,
) -> AdaptiveLambdaModel:
    """
    Подгоняем λ(x) так, чтобы покрытие попадало в target в среднем, учитывая драйверы
    (волатильность + news_surprise и т.п.).
    """
    with open(pack_path, "rb") as f:
        pack = pickle.load(f)

    models = pack["models"]
    feature_cols = pack["feature_cols"]
    scale_col = pack["scale_col"]
    quantiles = pack["quantiles"]

    # валидация — хвост 20%
    n = len(df)
    split = int(n * 0.8)
    va = df.iloc[split:].dropna(subset=["y", scale_col]).copy()
    if len(va) < 100:
        raise ValueError("Слишком мало данных для адаптивной калибровки")

    Xva = va[feature_cols].astype(float).values
    sigma = va[scale_col].astype(float).clip(1e-8).values
    y = va["y"].astype(float).values

    # предсказанные квантили (в sigma-единицах)
    q10s = models[0.1].predict(Xva) * sigma
    q50s = models[0.5].predict(Xva) * sigma
    q90s = models[0.9].predict(Xva) * sigma

    # базовая половинная ширина вокруг медианы (в лог-единицах *sigma* уже учтено)
    hw = 0.5 * ((q90s - q50s) + (q50s - q10s))
    eps = 1e-9
    r = np.abs(y - q50s) / (hw + eps)  # «насколько точка далеко от медианы» в half-width’ах
    r = np.clip(r, 0.0, 2.0)

    # дизайн-матрица драйверов
    Xdrv = _build_design(va, scale_col, extra_feats=extra_feats)
    F = Xdrv.values.astype(float)

    # регрессируем r ~ F (линейно), потом глобальным сдвигом добиваемся нужного покрытия
    reg = LinearRegression().fit(F, r)
    lam_pred = reg.predict(F)

    # найдём константу c для таргет-покрытия
    grid = np.linspace(-0.5, 0.5, 101)
    covers = []
    for c in grid:
        lam = lam_pred + c
        cover = np.mean(np.abs(y - q50s) <= lam * hw)
        covers.append(cover)
    c_star = float(grid[int(np.argmin(np.abs(np.array(covers) - target_cover)))])

    model = AdaptiveLambdaModel(
        coef_=reg.coef_.astype(float),
        intercept_=float(reg.intercept_ + c_star),
        features_=list(Xdrv.columns),
        base_lambda_=float(np.median(lam_pred + c_star)),
        target_cover_=float(target_cover),
    )
    return model

def apply_adaptive_lambda(
    p50: np.ndarray, p10: np.ndarray, p90: np.ndarray,
    adapt_model: AdaptiveLambdaModel,
    Xdrivers: pd.DataFrame
) -> Dict[str, np.ndarray]:
    lam = adapt_model.predict_lambda(Xdrivers)
    mid = p50
    lo = mid + lam * (p10 - mid)
    hi = mid + lam * (p90 - mid)
    return {"P10": np.minimum(lo, hi), "P50": mid, "P90": np.maximum(lo, hi)}