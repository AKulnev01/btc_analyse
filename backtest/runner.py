# backtest/runner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from models.ensemble import predict_pack as _predict_pack
from models.seq_batch_infer import batch_predict_seq_median

try:
    from models.ensemble_optimizer import optimize_ensemble_weights
except Exception:
    optimize_ensemble_weights = None

@dataclass
class ClassicalPack:
    path: str  # путь к .pkl паку (LGBM/Cat/XGB)

@dataclass
class SeqModel:
    backend: str  # "gru" | "tcn"
    model: str    # .pt
    meta: str     # .pkl
    device: str = "cpu"

@dataclass
class BTConfig:
    bars_per_day: int
    train_days: int = 365
    val_days: int = 60
    step_days: int = 30
    quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9)
    optimize_weights: bool = False

def _wf_splits(n_rows: int, bars_per_day: int, train_days: int, val_days: int, step_days: int):
    train_len = train_days * bars_per_day
    val_len = val_days * bars_per_day
    step_len = step_days * bars_per_day
    i = train_len
    while i + val_len <= n_rows:
        yield slice(i - train_len, i), slice(i, i + val_len)
        i += step_len

def _p50_from_pack(df_va: pd.DataFrame, pack_path: str) -> pd.Series:
    # быстрый батчевый способ для п50 (без цикла по строкам): считаем как обычно, но по всей валидации
    # для простоты: используем pack напрямую «как есть» на последних строках поштучно (надёжно, хоть и не супер быстро).
    preds = []
    for idx in df_va.index:
        sub = df_va.loc[:idx].iloc[-1:]  # одна строка
        out = _predict_pack(sub, pack_path)
        preds.append((idx, np.log(out["P50"]/out["now_price"])))
    return pd.Series({i:v for i,v in preds}, name="p50_log")

def _p10_p90_from_pack(df_va: pd.DataFrame, pack_path: str) -> Tuple[pd.Series, pd.Series]:
    p10, p90 = [], []
    for idx in df_va.index:
        sub = df_va.loc[:idx].iloc[-1:]
        out = _predict_pack(sub, pack_path)
        p10.append((idx, np.log(out["P10"]/out["now_price"])))
        p90.append((idx, np.log(out["P90"]/out["now_price"])))
    return (pd.Series(dict(p10), name="p10_log"),
            pd.Series(dict(p90), name="p90_log"))

class BacktestRunner:
    def __init__(self, df: pd.DataFrame, scale_col: str, feature_cols: List[str], cfg: BTConfig):
        self.df = df
        self.scale_col = scale_col
        self.feature_cols = feature_cols
        self.cfg = cfg
        self.work = df.dropna(subset=["y", scale_col]).copy()

    def _metrics(self, y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> Dict[str, float]:
        mae50 = float(np.mean(np.abs(y_true - p50)))
        lo = np.minimum(p10, p50)
        hi = np.maximum(p90, p50)
        cover = float(np.mean((y_true >= lo) & (y_true <= hi)))
        # простой pinball@0.5
        alpha = 0.5
        diff = y_true - p50
        pinball50 = float(np.mean(np.maximum(alpha*diff, (alpha-1)*diff)))
        return {"MAE50_log": mae50, "COVER_10_90": cover, "Pinball50": pinball50}

    def run(self,
            classical: Optional[List[ClassicalPack]] = None,
            seq_models: Optional[List[SeqModel]] = None,
            weights: Optional[List[float]] = None) -> Dict[str, float]:
        classical = classical or []
        seq_models = seq_models or []
        bars_per_day = self.cfg.bars_per_day

        folds = 0
        mae_list, cov_list, pin_list = [], [], []

        for tr_sl, va_sl in _wf_splits(len(self.work), bars_per_day,
                                       self.cfg.train_days, self.cfg.val_days, self.cfg.step_days):
            df_tr = self.work.iloc[tr_sl]
            df_va = self.work.iloc[va_sl]
            y_va = df_va["y"].values

            # собираем предсказания каждой модели на валид-окне
            preds_p10, preds_p50, preds_p90 = [], [], []

            # классические паки
            for pack in classical:
                p10_s, p90_s = _p10_p90_from_pack(df_va, pack.path)
                p50_s = _p50_from_pack(df_va, pack.path)
                preds_p10.append(p10_s.values)
                preds_p50.append(p50_s.values)
                preds_p90.append(p90_s.values)

            # seq модели (сейчас доступен быстрый батч только для медианы)
            for sm in seq_models:
                # медиана:
                p50_seq = batch_predict_seq_median(self.df.loc[df_va.index[0]:df_va.index[-1]],
                                                   backend=sm.backend, model_path=sm.model,
                                                   meta_path=sm.meta, device=sm.device)
                # приведём к тому же окну и порядку
                p50_aligned = p50_seq.reindex(df_va.index).values
                preds_p50.append(p50_aligned)
                # пока п10/п90 нет — возьмём симметрию от медианы (мягкий допущение): ширина = медиана ширин по пакетам, если есть
                if preds_p10 and preds_p90:
                    width_lo = np.median([p50 - p10 for p10, p50 in zip(preds_p10[-len(classical):], preds_p50[-len(classical):])], axis=0)
                    width_hi = np.median([p90 - p50 for p50, p90 in zip(preds_p50[-len(classical):], preds_p90[-len(classical):])], axis=0)
                else:
                    width_lo = width_hi = np.zeros_like(p50_aligned)
                preds_p10.append(p50_aligned - width_lo)
                preds_p90.append(p50_aligned + width_hi)

            # ансамбль
            n_models = len(preds_p50)
            if n_models == 0:
                raise ValueError("Нет моделей для бэктеста.")
            W = np.ones(n_models, dtype=float) / n_models if weights is None else np.array(weights, float)
            W = W / W.sum()

            P10 = np.sum(np.stack(preds_p10, axis=0) * W[:, None], axis=0)
            P50 = np.sum(np.stack(preds_p50, axis=0) * W[:, None], axis=0)
            P90 = np.sum(np.stack(preds_p90, axis=0) * W[:, None], axis=0)

            m = self._metrics(y_va, P10, P50, P90)
            mae_list.append(m["MAE50_log"]); cov_list.append(m["COVER_10_90"]); pin_list.append(m["Pinball50"])
            folds += 1

        return {
            "folds": folds,
            "MAE50_log_mean": float(np.mean(mae_list)) if mae_list else None,
            "COVER_10_90_mean": float(np.mean(cov_list)) if cov_list else None,
            "Pinball50_mean": float(np.mean(pin_list)) if pin_list else None,
        }