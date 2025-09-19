# models/tcn_seq.py
import os
import pickle
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .seq_common import (
    feature_cols_auto, pick_scale_col,
    sanitize_dataframe, StandardScalerSafe, SequenceDataset,
    pinball_loss,
)
# в начале файла

# ----- TCN слои -----
class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp
    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.0):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

        # init
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_dim: int, channels: List[int], kernel_size: int, dropout: float, quantiles: List[float]):
        super().__init__()
        layers = []
        prev = in_dim
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(prev, ch, kernel_size, dilation, dropout))
            prev = ch
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(prev, len(quantiles))
        self.quantiles = quantiles

    def forward(self, x):  # x: (B,L,F)
        x = x.transpose(1, 2)           # -> (B,F,L)
        y = self.network(x)              # -> (B,C,L)
        h_last = y[:, :, -1]             # (B,C)
        out = self.head(h_last)          # (B,Q) в σ-единицах
        return out

# ----- Train / Predict -----
def train_tcn(
    df: pd.DataFrame,
    feat_cols: Optional[List[str]] = None,
    seq_len: int = 96,
    hidden_channels: List[int] = [64, 64, 64, 64],
    kernel_size: int = 3,
    dropout: float = 0.1,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    out_path: str = "models/tcn_seq.pt",
    meta_path: str = "models/tcn_seq_meta.pkl",
) -> Tuple[nn.Module, Dict]:

    feat_cols = feat_cols or feature_cols_auto(df)
    scale_col = pick_scale_col(df)

    # split & sanitize (train/val одинаково)
    n = len(df); split = int(n * 0.8)
    df_tr, feat_cols_tr = sanitize_dataframe(df.iloc[:split], feat_cols, scale_col, na_frac_threshold=0.0)
    df_va, _             = sanitize_dataframe(df.iloc[split:], feat_cols_tr, scale_col, na_frac_threshold=0.0)

    scaler = StandardScalerSafe().fit(df_tr[feat_cols_tr].to_numpy(dtype=float), cols=feat_cols_tr)

    ds_tr = SequenceDataset(df_tr, feat_cols_tr, scale_col, seq_len, scaler=scaler)
    ds_va = SequenceDataset(df_va, feat_cols_tr, scale_col, seq_len, scaler=scaler)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    model = TCN(in_dim=len(feat_cols_tr), channels=hidden_channels, kernel_size=kernel_size, dropout=dropout, quantiles=quantiles).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float("inf"); patience=0; saved=False

    for ep in range(1, epochs+1):
        model.train(); tr_loss=0.0
        for xb, yb, _, _ in dl_tr:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad()
            q = model(xb)
            loss = pinball_loss(q, yb, quantiles)
            if torch.isnan(loss): raise ValueError("NaN loss (TCN)")
            loss.backward(); opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
        tr_loss/=max(1,len(ds_tr))

        model.eval(); va_loss=0.0
        with torch.no_grad():
            for xb, yb, _, _ in dl_va:
                xb=xb.to(device); yb=yb.to(device)
                q = model(xb)
                loss = pinball_loss(q, yb, quantiles)
                va_loss += float(loss.item()) * xb.size(0)
        va_loss/=max(1,len(ds_va))
        print(f"[TCN] epoch {ep} | train={tr_loss:.5f} val={va_loss:.5f}")

        if va_loss + 1e-6 < best:
            best = va_loss; patience=0
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(model.state_dict(), out_path); saved=True
        else:
            patience += 1
            if patience >= 6:
                print("[TCN] early stop"); break

    meta = dict(feature_cols=feat_cols_tr, scale_col=scale_col, quantiles=quantiles, seq_len=seq_len,
                scaler=scaler, hidden_channels=hidden_channels, kernel_size=kernel_size, dropout=dropout)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "wb") as f: pickle.dump(meta, f)
    if not saved: raise FileNotFoundError("TCN checkpoint not saved")

    model.load_state_dict(torch.load(out_path, map_location=device)); model.eval()
    return model, meta

def predict_now_tcn(
    df: pd.DataFrame,
    model_path: str = "models/tcn_seq.pt",
    meta_path: str  = "models/tcn_seq_meta.pkl",
    device: str = "cpu",
) -> Dict[str, float]:
    with open(meta_path, "rb") as f: meta = pickle.load(f)
    feat_cols = meta["feature_cols"]; scale_col = meta["scale_col"]; quantiles = meta["quantiles"]
    seq_len = int(meta["seq_len"]); scaler: StandardScalerSafe = meta["scaler"]
    hidden_channels=meta["hidden_channels"]; kernel_size=meta["kernel_size"]; dropout=meta["dropout"]

    work, _ = sanitize_dataframe(df, feat_cols, scale_col, na_frac_threshold=0.0)
    for c in feat_cols:
        if c not in work.columns: work[c]=0.0
    work = work[[*feat_cols, "y", scale_col, "close"]]
    if len(work) < seq_len: raise ValueError("Недостаточно строк для последовательности TCN")

    model = TCN(in_dim=len(feat_cols), channels=hidden_channels, kernel_size=kernel_size, dropout=dropout, quantiles=quantiles).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)); model.eval()

    X = work[feat_cols].to_numpy(dtype=float)
    Xz = scaler.transform(X)

    x_seq = torch.from_numpy(Xz[-seq_len:]).float().unsqueeze(0).to(device)
    sigma = float(work[scale_col].iloc[-1]); now_price=float(work["close"].iloc[-1])

    with torch.no_grad():
        q_scaled = model(x_seq)[0].cpu().numpy()
    q_scaled.sort()
    yq = q_scaled * sigma
    return {
        "now_price": now_price,
        "P10": float(now_price * np.exp(yq[0])),
        "P50": float(now_price * np.exp(yq[1])),
        "P90": float(now_price * np.exp(yq[2])),
    }

# CLI для быстрого прогона
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="./data/features/btc_30m_features.parquet")
    ap.add_argument("--mode", choices=["train","predict"], default="train")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="models/tcn_seq.pt")
    ap.add_argument("--meta", default="models/tcn_seq_meta.pkl")
    args = ap.parse_args()

    df = pd.read_parquet(args.features).sort_index()
    if args.mode=="train":
        train_tcn(df, seq_len=args.seq_len, device=args.device, out_path=args.out, meta_path=args.meta)
    else:
        print(predict_now_tcn(df, model_path=args.out, meta_path=args.meta, device=args.device))

if __name__ == "__main__":
    _cli()