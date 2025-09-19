# models/seq_batch_infer.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import pickle
import torch


def _pick_device(pref: str = "cpu") -> torch.device:
    """Выбираем доступный девайс с учётом предпочтения."""
    pref = (pref or "cpu").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    # авто-выбор: cuda -> mps -> cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_sequences(Xz: np.ndarray, idxs: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Собирает батч последовательностей (B, L, F) для индексов idxs,
    где для каждого i берём Xz[i-L+1 : i+1].
    """
    B = len(idxs)
    L, F = seq_len, Xz.shape[1]
    out = np.empty((B, L, F), dtype=np.float32)
    for j, i in enumerate(idxs):
        out[j] = Xz[i - L + 1 : i + 1]
    return out


def batch_predict_seq_median(
    df: pd.DataFrame,
    backend: str,
    model_path: str,
    meta_path: str,
    device: str = "cpu",
    batch_size: int = 512,
) -> pd.Series:
    """
    Возвращает серию yhat50 (лог-доходность, уже размасштабированная),
    для всех валидных точек (там, где есть полная история seq_len).
    Индекс серии соответствует исходному df (подмножество).
    """
    device_t = _pick_device(device)

    if backend == "gru":
        # импортируем классы/функции из твоего модуля
        from models.nn_seq import GRUQuantile

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        feat_cols = meta["feature_cols"]
        scale_col = meta["scale_col"]
        quantiles = meta["quantiles"]
        seq_len = int(meta["seq_len"])
        scaler = meta["scaler"]

        # сбор данных
        work = df.dropna(subset=["y", scale_col]).copy()
        X = work[feat_cols].astype(float).values
        Xz = scaler.transform(X).astype(np.float32)
        sigma = work[scale_col].astype(float).clip(1e-8).values

        valid_idx = np.arange(seq_len - 1, len(work))
        # строим модель и грузим веса
        model = GRUQuantile(
            in_dim=len(feat_cols),
            hidden=int(meta["hidden"]),
            num_layers=int(meta["num_layers"]),
            quantiles=quantiles,
        ).to(device_t)
        state = torch.load(model_path, map_location=device_t)
        model.load_state_dict(state)
        model.eval()

        # индекс медианного квантили
        try:
            qi = int(np.argwhere(np.isclose(np.array(quantiles), 0.5)).ravel()[0])
        except Exception as e:
            raise ValueError("В meta.quantiles нет 0.5 — нужен медианный квантиль.") from e

        preds_scaled = np.empty(len(valid_idx), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(valid_idx), batch_size):
                chunk = valid_idx[start : start + batch_size]
                xb_np = _batch_sequences(Xz, chunk, seq_len)       # (B, L, F)
                xb = torch.from_numpy(xb_np).to(device_t)           # float32
                q = model(xb)                                       # (B, Q)
                preds_scaled[start : start + len(chunk)] = q[:, qi].detach().cpu().numpy()

        # размасштабируем обратно в лог-y
        yhat50 = preds_scaled * sigma[valid_idx]
        idx = work.index[valid_idx]
        return pd.Series(yhat50.astype(float), index=idx, name="yhat50_seq")

    elif backend == "tcn":
        # ожидаем, что в models/tcn_seq.py есть совместимые meta и класс TCNQuantile
        from models.tcn_seq import TCNQuantile

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        feat_cols = meta["feature_cols"]
        scale_col = meta["scale_col"]
        quantiles = meta["quantiles"]
        seq_len = int(meta["seq_len"])
        scaler = meta["scaler"]

        work = df.dropna(subset=["y", scale_col]).copy()
        X = work[feat_cols].astype(float).values
        Xz = scaler.transform(X).astype(np.float32)
        sigma = work[scale_col].astype(float).clip(1e-8).values

        valid_idx = np.arange(seq_len - 1, len(work))

        model = TCNQuantile(
            in_dim=len(feat_cols),
            channels=int(meta["channels"]),
            levels=int(meta["levels"]),
            kernel_size=int(meta["kernel_size"]),
            quantiles=quantiles,
            dropout=float(meta.get("dropout", 0.0)),
        ).to(device_t)
        state = torch.load(model_path, map_location=device_t)
        model.load_state_dict(state)
        model.eval()

        try:
            qi = int(np.argwhere(np.isclose(np.array(quantiles), 0.5)).ravel()[0])
        except Exception as e:
            raise ValueError("В meta.quantiles нет 0.5 — нужен медианный квантиль.") from e

        preds_scaled = np.empty(len(valid_idx), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(valid_idx), batch_size):
                chunk = valid_idx[start : start + batch_size]
                xb_np = _batch_sequences(Xz, chunk, seq_len)       # (B, L, F)
                # для TCN часто нужен формат (B, F, L); перевернём оси
                xb_np = np.transpose(xb_np, (0, 2, 1))             # (B, F, L)
                xb = torch.from_numpy(xb_np).to(device_t)
                q = model(xb)                                      # (B, Q)
                preds_scaled[start : start + len(chunk)] = q[:, qi].detach().cpu().numpy()

        yhat50 = preds_scaled * sigma[valid_idx]
        idx = work.index[valid_idx]
        return pd.Series(yhat50.astype(float), index=idx, name="yhat50_seq")

    else:
        raise ValueError(f"Unsupported backend: {backend}")