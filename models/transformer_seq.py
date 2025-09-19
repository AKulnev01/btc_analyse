# models/transformer_seq.py
# Quantile Transformer (lite): TransformerEncoder + pinball loss
from __future__ import annotations
import os, pickle, argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
SCALE_COLS = ["sigma_ewma_7d", "sigma_7d"]


def _pick_scale_col(df: pd.DataFrame) -> str:
    for c in SCALE_COLS:
        if c in df.columns:
            return c
    raise ValueError("Нет sigma_ewma_7d/sigma_7d в фичах.")


def _feature_cols(df: pd.DataFrame, explicit: Optional[List[str]] = None) -> List[str]:
    if explicit is not None:
        cols = [c for c in explicit if c in df.columns and df[c].dtype.kind != "O"]
    else:
        bad = set(["y", "close"])
        cols = [c for c in df.columns if c not in bad and df[c].dtype.kind != "O"]
    return cols


class SequenceDataset(Dataset):
    """(B, L, F), цель: y* в σ-единицах для последнего шага окна"""
    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        scale_col: str,
        seq_len: int = 96,
        scaler: Optional[StandardScaler] = None,
        x_jitter_std: float = 0.0,
    ):
        self.df = df.copy()
        self.feat_cols = feat_cols
        self.scale_col = scale_col
        self.seq_len = int(seq_len)
        self.x_jitter_std = float(x_jitter_std or 0.0)

        req = ["y", scale_col] + feat_cols
        mask = self.df[req].notna().all(axis=1)
        self.df = self.df.loc[mask]

        X = self.df[feat_cols].astype(float).values
        if scaler is None:
            self.scaler = StandardScaler()
            Xz = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            Xz = self.scaler.transform(X)

        Xz = np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)
        self.Xz = Xz.astype(np.float32)
        self.sigma = self.df[scale_col].astype(float).values.clip(1e-8).astype(np.float32)
        y = self.df["y"].astype(float).values.astype(np.float32)
        self.y_scaled = (y / self.sigma).astype(np.float32)

        self.valid_idx = np.arange(self.seq_len - 1, len(self.df))

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, i: int):
        idx = self.valid_idx[i]
        sl = slice(idx - self.seq_len + 1, idx + 1)
        x = self.Xz[sl]  # (L, F)
        if self.x_jitter_std > 0.0:
            x = x + np.random.normal(0.0, self.x_jitter_std, size=x.shape).astype(np.float32)
        y_scaled = self.y_scaled[idx]
        sigma = self.sigma[idx]
        now_price = float(self.df["close"].iloc[idx])
        return (
            torch.from_numpy(x),                                           # (L, F)
            torch.tensor([y_scaled], dtype=torch.float32),                 # (1,)
            torch.tensor([sigma], dtype=torch.float32),                    # (1,)
            torch.tensor([now_price], dtype=torch.float32),                # (1,)
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TransQuantile(nn.Module):
    """Простой TransformerEncoder → квантильная голова"""
    def __init__(self, in_dim: int, d_model: int, nhead: int, num_layers: int, quantiles: List[float], dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
                                                   dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.head = nn.Linear(d_model, len(quantiles))
        self.quantiles = quantiles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        z = self.in_proj(x)             # (B, L, D)
        z = self.pos(z)                 # (B, L, D)
        z = self.encoder(z)             # (B, L, D)
        h_last = z[:, -1, :]            # (B, D)
        q_scaled = self.head(h_last)    # (B, Q) в σ-единицах
        return q_scaled


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    B, Q = pred.shape
    t = target.repeat(1, Q)
    qs = torch.tensor(quantiles, device=pred.device).view(1, Q).repeat(B, 1)
    diff = t - pred
    return torch.maximum(qs * diff, (qs - 1.0) * diff).mean()


def train_transformer_seq(
    df: pd.DataFrame,
    feat_cols: List[str],
    scale_col: str,
    quantiles: List[float] = DEFAULT_QUANTILES,
    seq_len: int = 96,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    out_path: str = "models/trans_seq.pt",
    meta_path: str = "models/trans_seq_meta.pkl",
    early_stopping_patience: int = 5,
    x_jitter_std: float = 0.0,
    grad_clip: Optional[float] = 1.0,
) -> Tuple['TransQuantile', Dict]:
    n = len(df)
    split = int(n * 0.8)
    df_tr = df.iloc[:split].copy()
    df_va = df.iloc[split:].copy()

    # scaler по валидным train-строкам
    fit_mask = df_tr[feat_cols + [scale_col, "y"]].notna().all(axis=1)
    Xfit = df_tr.loc[fit_mask, feat_cols].astype(float).values
    if Xfit.size == 0:
        raise ValueError("Нет валидных строк для фитинга скейлера.")
    scaler = StandardScaler().fit(Xfit)

    ds_tr = SequenceDataset(df_tr, feat_cols, scale_col, seq_len, scaler=scaler, x_jitter_std=x_jitter_std)
    ds_va = SequenceDataset(df_va, feat_cols, scale_col, seq_len, scaler=scaler, x_jitter_std=0.0)

    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise ValueError("Слишком мало данных для seq-обучения.")

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    model = TransQuantile(len(feat_cols), d_model, nhead, num_layers, quantiles, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val, patience = 1e9, 0

    for ep in range(1, epochs + 1):
        # train
        model.train()
        tr_loss = 0.0
        for xb, yb, _, _ in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            q = model(xb)
            loss = pinball_loss(q, yb, quantiles)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
        tr_loss /= max(1, len(ds_tr))

        # valid
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb, _, _ in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                q = model(xb)
                loss = pinball_loss(q, yb, quantiles)
                va_loss += float(loss.item()) * xb.size(0)
        va_loss /= max(1, len(ds_va))
        print(f"epoch {ep} | train={tr_loss:.5f}  val={va_loss:.5f}")

        if va_loss + 1e-6 < best_val:
            best_val, patience = va_loss, 0
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(model.state_dict(), out_path)
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print("[early_stop] no improvement")
                break

    meta = dict(
        feature_cols=feat_cols,
        scale_col=scale_col,
        quantiles=quantiles,
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=float(dropout),
        scaler=scaler,
    )
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    model.load_state_dict(torch.load(out_path, map_location=device))
    model.eval()
    return model, meta


def predict_now_transformer(
    df: pd.DataFrame,
    model_path: str = "models/trans_seq.pt",
    meta_path: str = "models/trans_seq_meta.pkl",
    device: str = "cpu",
) -> Dict[str, float]:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    feat_cols = meta["feature_cols"]
    scale_col = meta["scale_col"]
    quantiles = meta["quantiles"]
    seq_len = int(meta["seq_len"])
    scaler = meta["scaler"]

    model = TransQuantile(
        in_dim=len(feat_cols),
        d_model=int(meta["d_model"]),
        nhead=int(meta["nhead"]),
        num_layers=int(meta["num_layers"]),
        quantiles=quantiles,
        dropout=float(meta.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    work = df.dropna(subset=["y", scale_col]).copy()
    if len(work) < seq_len:
        raise ValueError("Недостаточно строк для последней последовательности.")

    X = work[feat_cols].astype(float).values
    Xz = scaler.transform(X)
    Xz = np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)

    xb = torch.from_numpy(Xz[-seq_len:]).float().unsqueeze(0).to(device)  # (1, L, F)
    sigma = float(work[scale_col].iloc[-1])
    now_price = float(work["close"].iloc[-1])

    with torch.no_grad():
        q_scaled = model(xb)[0].cpu().numpy()
    q_scaled.sort()

    yq = q_scaled * sigma
    P10 = now_price * np.exp(yq[0])
    P50 = now_price * np.exp(yq[1])
    P90 = now_price * np.exp(yq[2])
    return {"now_price": now_price, "P10": float(P10), "P50": float(P50), "P90": float(P90)}


def main():
    ap = argparse.ArgumentParser(description="Transformer quantile seq — train / predict")
    ap.add_argument("--features", default="./data/features/btc_1h_features.parquet")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", choices=["train", "predict"], default="train")
    ap.add_argument("--out", default="models/trans_seq.pt")
    ap.add_argument("--meta", default="models/trans_seq_meta.pkl")
    ap.add_argument("--quantiles", default="0.1,0.5,0.9")
    ap.add_argument("--x-jitter-std", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.features).sort_index()
    feat_cols = _feature_cols(df, None)
    scale_col = _pick_scale_col(df)
    quants = [float(x) for x in args.quantiles.split(",")]

    if args.mode == "train":
        train_transformer_seq(
            df, feat_cols, scale_col,
            quantiles=quants,
            seq_len=args.seq_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            out_path=args.out,
            meta_path=args.meta,
            x_jitter_std=args.x_jitter_std,
            grad_clip=args.grad_clip,
        )
    else:
        out = predict_now_transformer(df, model_path=args.out, meta_path=args.meta, device=args.device)
        print({k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in out.items()})


if __name__ == "__main__":
    main()