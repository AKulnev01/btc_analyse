# models/gp_calibration.py
from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Optional

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class GPCalibrator:
    """
    Байесовская GP-коррекция доверительных интервалов (P10–P90).
    Идея: учим GP предсказывать покрытие интервала по контекстным фичам
    (вола, новости, объём...), затем конвертируем прогноз покрытия в
    множитель lambda для ширины интервала.
    """

    def __init__(self):
        kernel = 1.0 * RBF(length_scale=10.0) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.fitted = False

    # ---- фичи для GP
    def _make_features(self, df: pd.DataFrame) -> np.ndarray:
        feats = []
        if "sigma_ewma_7d" in df.columns:
            feats.append(df["sigma_ewma_7d"].astype(float).values)
        elif "sigma_7d" in df.columns:
            feats.append(df["sigma_7d"].astype(float).values)
        if "news_surprise" in df.columns:
            feats.append(df["news_surprise"].fillna(0).astype(float).values)
        if "volume" in df.columns:
            feats.append(np.log1p(df["volume"].clip(lower=0)).astype(float).values)

        if not feats:
            feats.append(np.arange(len(df), dtype=float))
        X = np.vstack(feats).T
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    # ---- обучение
    def fit(self, df_val: pd.DataFrame, y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray):
        cover = ((y_true >= np.minimum(p10, p90)) & (y_true <= np.maximum(p10, p90))).astype(float)
        X = self._make_features(df_val)
        self.gp.fit(X, cover)
        self.fitted = True

    # ---- прогноз множителя λ
    def predict_lambda(self, df_slice: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("GPCalibrator is not fitted")
        X = self._make_features(df_slice)
        y_pred, _ = self.gp.predict(X, return_std=True)  # ожидаемое покрытие [0..1]
        # чем ниже ожидаемое покрытие, тем шире интервал:  lam = 1 + (0.5 - cover)
        lam = np.clip(1.0 + (0.5 - y_pred), 0.5, 1.5)
        return lam

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "GPCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)