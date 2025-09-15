# models/nn_seq.py
# GRU-квантильная регрессия: pinball loss для [0.1,0.5,0.9]
import os
import pickle
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ---- конфиг по умолчанию (совместим с твоим проектом) ----
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

# ----- Dataset: последовательности из таблички -----
class SequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        scale_col: str,
        seq_len: int = 48,
        scaler: Optional[StandardScaler] = None,
    ):
        self.df = df.copy()
        self.feat_cols = feat_cols
        self.scale_col = scale_col
        self.seq_len = int(seq_len)

        # Маскирование валидных строк: есть y и sigma
        mask = self.df["y"].notna() & self.df[scale_col].notna()
        self.df = self.df.loc[mask]

        # Стандартизация фич (только X, не цель)
        X = self.df[feat_cols].astype(float).values
        if scaler is None:
            self.scaler = StandardScaler()
            Xz = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            Xz = self.scaler.transform(X)

        self.Xz = Xz
        self.sigma = self.df[scale_col].astype(float).values.clip(min=1e-8)
        self.y = self.df["y"].astype(float).values
        # y* = y / sigma
        self.y_scaled = self.y / self.sigma

        # индексы, где есть полная история seq_len
        self.valid_idx = np.arange(self.seq_len - 1, len(self.df))

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, i: int):
        idx = self.valid_idx[i]
        sl = slice(idx - self.seq_len + 1, idx + 1)
        x_seq = self.Xz[sl]                       # (seq_len, F)
        y_scaled = self.y_scaled[idx]             # scalar (регрессия)
        sigma = self.sigma[idx]                   # scalar (для обратного масштаба)
        # для удобства вернём now_price тоже (брать из DataFrame)
        now_price = float(self.df["close"].iloc[idx])
        return (
            torch.from_numpy(x_seq).float(),      # (L, F)
            torch.tensor([y_scaled]).float(),     # (1,)
            torch.tensor([sigma]).float(),        # (1,)
            torch.tensor([now_price]).float(),    # (1,)
        )

# ----- Модель -----
class GRUQuantile(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_layers: int, quantiles: List[float]):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden, len(quantiles))
        self.quantiles = quantiles

    def forward(self, x):
        # x: (B, L, F)
        out, _ = self.gru(x)            # (B, L, H)
        h_last = out[:, -1, :]          # (B, H)
        q_scaled = self.head(h_last)    # (B, Q) — в σ-единицах
        return q_scaled

# ----- Pinball loss для набора квантилей -----
def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    # pred: (B, Q), target: (B, 1)
    B, Q = pred.shape
    t = target.repeat(1, Q)
    qs = torch.tensor(quantiles, device=pred.device).view(1, Q).repeat(B, 1)
    diff = t - pred
    loss = torch.maximum(qs * diff, (qs - 1.0) * diff).mean()
    return loss

# ----- Обучение -----
def train_model(
    df: pd.DataFrame,
    feat_cols: List[str],
    scale_col: str,
    quantiles: List[float],
    seq_len: int = 48,
    hidden: int = 128,
    num_layers: int = 1,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    out_path: str = "models/nn_seq.pt",
    meta_path: str = "models/nn_seq_meta.pkl",
    early_stopping_patience: int = 5,
) -> Tuple[GRUQuantile, Dict]:
    # time split 80/20
    n = len(df)
    split = int(n * 0.8)
    df_tr = df.iloc[:split].copy()
    df_va = df.iloc[split:].copy()

    # скейлер по train
    scaler = StandardScaler()
    _ = scaler.fit(df_tr[feat_cols].astype(float).values)

    ds_tr = SequenceDataset(df_tr, feat_cols, scale_col, seq_len, scaler=scaler)
    ds_va = SequenceDataset(df_va, feat_cols, scale_col, seq_len, scaler=scaler)

    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise ValueError("Слишком мало данных для seq-обучения.")

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    model = GRUQuantile(in_dim=len(feat_cols), hidden=hidden, num_layers=num_layers, quantiles=quantiles).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = 1e9
    patience = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb, _, _ in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            q_pred = model(xb)  # (B, Q), в σ-единицах
            loss = pinball_loss(q_pred, yb, quantiles)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
        tr_loss /= max(1, len(ds_tr))

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb, _, _ in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                q_pred = model(xb)
                loss = pinball_loss(q_pred, yb, quantiles)
                va_loss += float(loss.item()) * xb.size(0)
        va_loss /= max(1, len(ds_va))

        print("epoch %d | train=%.5f  val=%.5f" % (ep, tr_loss, va_loss))

        # ES
        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            patience = 0
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(model.state_dict(), out_path)
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print("[early_stop] no improvement")
                break

    # финальная мета
    meta = dict(
        feature_cols=feat_cols,
        scale_col=scale_col,
        quantiles=quantiles,
        seq_len=seq_len,
        scaler=scaler,
        hidden=hidden,
        num_layers=num_layers,
    )
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    # загрузим лучшие веса
    model.load_state_dict(torch.load(out_path, map_location=device))
    model.eval()
    return model, meta

# ----- Прогноз «на сейчас» -----
def predict_now(
    df: pd.DataFrame,
    model_path: str = "models/nn_seq.pt",
    meta_path: str = "models/nn_seq_meta.pkl",
    device: str = "cpu",
) -> Dict[str, float]:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    feat_cols = meta["feature_cols"]
    scale_col = meta["scale_col"]
    quantiles = meta["quantiles"]
    seq_len = int(meta["seq_len"])
    scaler = meta["scaler"]

    model = GRUQuantile(
        in_dim=len(feat_cols),
        hidden=int(meta["hidden"]),
        num_layers=int(meta["num_layers"]),
        quantiles=quantiles,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # отфильтруем валид
    work = df.dropna(subset=["y", scale_col]).copy()
    if len(work) < seq_len:
        raise ValueError("Недостаточно строк для последней последовательности.")

    X = work[feat_cols].astype(float).values
    Xz = scaler.transform(X)

    x_seq = torch.from_numpy(Xz[-seq_len:]).float().unsqueeze(0).to(device)  # (1, L, F)
    sigma = float(work[scale_col].iloc[-1])
    now_price = float(work["close"].iloc[-1])

    with torch.no_grad():
        q_scaled = model(x_seq)[0].cpu().numpy()  # (Q,)
    # non-crossing: отсортируем
    q_scaled.sort()

    # обратно в y и в цены
    yq = q_scaled * sigma
    P10 = now_price * np.exp(yq[0])
    P50 = now_price * np.exp(yq[1])
    P90 = now_price * np.exp(yq[2])

    return {"now_price": now_price, "P10": float(P10), "P50": float(P50), "P90": float(P90)}

# ----- CLI -----
def main():
    ap = argparse.ArgumentParser(description="GRU quantile model (seq) — train / predict")
    ap.add_argument("--features", default="./data/features/btc_1h_features.parquet")
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", choices=["train", "predict"], default="train")
    ap.add_argument("--out", default="models/nn_seq.pt")
    ap.add_argument("--meta", default="models/nn_seq_meta.pkl")
    ap.add_argument("--quantiles", default="0.1,0.5,0.9")
    args = ap.parse_args()

    df = pd.read_parquet(args.features).sort_index()
    feat_cols = _feature_cols(df, None)
    scale_col = _pick_scale_col(df)
    quants = [float(x) for x in args.quantiles.split(",")]

    if args.mode == "train":
        train_model(
            df=df,
            feat_cols=feat_cols,
            scale_col=scale_col,
            quantiles=quants,
            seq_len=args.seq_len,
            hidden=args.hidden,
            num_layers=args.layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            out_path=args.out,
            meta_path=args.meta,
        )
    else:
        out = predict_now(df, model_path=args.out, meta_path=args.meta, device=args.device)
        print({k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in out.items()})

if __name__ == "__main__":
    main()