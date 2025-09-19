# models/nn_seq.py
# GRU-квантильная регрессия: pinball loss для [0.1, 0.5, 0.9]
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


# ---- конфиг по умолчанию ----
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
SCALE_COLS = ["sigma_ewma_7d", "sigma_7d"]


# ---- утилиты ----
def _pick_scale_col(df: pd.DataFrame) -> str:
    for c in SCALE_COLS:
        if c in df.columns:
            return c
    raise ValueError("Нет sigma_ewma_7d/sigma_7d в фичах.")

def _feature_cols(df: pd.DataFrame, explicit: Optional[List[str]] = None) -> List[str]:
    if explicit is not None:
        cols = [c for c in explicit if c in df.columns and df[c].dtype.kind != "O"]
    else:
        bad = {"y", "close"}
        cols = [c for c in df.columns if c not in bad and df[c].dtype.kind != "O"]
    return cols

def _auto_device(name: str) -> str:
    name = (name or "cpu").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        return _auto_device("auto")
    if name == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            return _auto_device("auto")
    return name

def _nan_to_num_(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

def _set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----- Dataset: последовательности из таблички -----
class SequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        scale_col: str,
        seq_len: int = 48,
        scaler: Optional[StandardScaler] = None,
        x_jitter_std: float = 0.0,    # шум признаков (используем только на train)
    ):
        self.df = df.copy()
        self.feat_cols = feat_cols
        self.scale_col = scale_col
        self.seq_len = int(seq_len)
        self.x_jitter_std = float(max(0.0, x_jitter_std or 0.0))

        # Маскирование валидных строк: есть y, sigma и все фичи
        req = ["y", scale_col] + feat_cols
        mask = self.df[req].notna().all(axis=1)
        self.df = self.df.loc[mask].copy()

        # Стандартизация фич (только X, не цель)
        X = self.df[feat_cols].astype(float).values
        if scaler is None:
            self.scaler = StandardScaler()
            Xz = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            Xz = self.scaler.transform(X)

        Xz = _nan_to_num_(Xz).astype(np.float32, copy=False)

        self.Xz = Xz
        self.sigma = np.clip(self.df[scale_col].astype(float).values, 1e-8, None).astype(np.float32)
        self.y = self.df["y"].astype(float).values
        self.y_scaled = (self.y / self.sigma).astype(np.float32)

        # индексы, где есть полная история seq_len
        self.valid_idx = np.arange(self.seq_len - 1, len(self.df))

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, i: int):
        idx = self.valid_idx[i]
        sl = slice(idx - self.seq_len + 1, idx + 1)
        x_seq = self.Xz[sl].copy()                      # (L, F)

        # лёгкий джиттер признаков (если включён)
        if self.x_jitter_std > 0.0:
            noise = np.random.normal(0.0, self.x_jitter_std, size=x_seq.shape).astype(np.float32)
            x_seq += noise

        y_scaled = np.float32(self.y_scaled[idx])      # scalar (регрессия, в σ-ед.)
        sigma = np.float32(self.sigma[idx])            # scalar (для обратного масштаба)
        now_price = float(self.df["close"].iloc[idx])

        return (
            torch.from_numpy(x_seq),                               # (L, F)
            torch.tensor([y_scaled], dtype=torch.float32),         # (1,)
            torch.tensor([sigma], dtype=torch.float32),            # (1,)
            torch.tensor([now_price], dtype=torch.float32),        # (1,)
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
    # делаем квантильный вектор на нужном устройстве
    qs = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).view(1, Q).repeat(B, 1)
    t = target.to(pred.dtype).repeat(1, Q)
    diff = t - pred
    # pinball: max(q*diff, (q-1)*diff)
    loss = torch.maximum(qs * diff, (qs - 1.0) * diff).mean()
    # защита от NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        # очень редкий случай — вернём нулевой лосс, чтобы не сорвать шаг
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
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
    device: str = "cpu",              # "cpu" | "cuda" | "mps" | "auto"
    out_path: str = "models/nn_seq.pt",
    meta_path: str = "models/nn_seq_meta.pkl",
    early_stopping_patience: int = 5,
    x_jitter_std: float = 0.0,        # шум на train
    grad_clip: Optional[float] = 1.0, # клиппирование градиентов
    seed: int = 42,
) -> Tuple["GRUQuantile", Dict]:
    _set_seed(seed)
    device = _auto_device(device)
    device_t = torch.device(device)

    # --- time split 80/20 ---
    n = len(df)
    split = int(n * 0.8)
    df_tr = df.iloc[:split].copy()
    df_va = df.iloc[split:].copy()

    # --- 1) дроп слишком «дырявых» фич в train (NaN > 50%) ---
    na_ratio = df_tr[feat_cols].isna().mean().sort_values(ascending=False)
    bad_feats = na_ratio[na_ratio > 0.50].index.tolist()
    if bad_feats:
        print(f"[SEQ] drop {len(bad_feats)} sparse features (>50% NaN in train), e.g.: {bad_feats[:5]}")
        feat_cols = [c for c in feat_cols if c not in bad_feats]
    if not feat_cols:
        raise ValueError("После фильтрации пустых фич не осталось ни одной колонки для обучения.")

    # --- 2) фитим скейлер по валидным train-строкам; если 0 — делаем фолбэк на весь df ---
    req_cols = feat_cols + [scale_col, "y"]
    fit_mask_tr = df_tr[req_cols].notna().all(axis=1)
    Xfit = df_tr.loc[fit_mask_tr, feat_cols].astype(float).values

    if Xfit.size == 0:
        print("[SEQ][WARN] no valid rows in train split for scaler — fallback to full df mask")
        fit_mask_full = df[req_cols].notna().all(axis=1)
        Xfit = df.loc[fit_mask_full, feat_cols].astype(float).values

    if Xfit.size == 0:
        holes = df_tr[req_cols].isna().sum().sort_values(ascending=False).head(12)
        raise ValueError(
            "Нет валидных строк для фитинга скейлера. Топ «дырявых» колонок (кол-во NaN в train):\n"
            + str(holes)
        )

    scaler = StandardScaler().fit(Xfit)

    # --- 3) датасеты/даталоадеры ---
    ds_tr = SequenceDataset(df_tr, feat_cols, scale_col, seq_len, scaler=scaler, x_jitter_std=x_jitter_std)
    ds_va = SequenceDataset(df_va, feat_cols, scale_col, seq_len, scaler=scaler, x_jitter_std=0.0)

    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise ValueError(
            f"Слишком мало данных для seq-обучения: len_tr={len(ds_tr)}, len_va={len(ds_va)}. "
            f"seq_len={seq_len}, feat_cols={len(feat_cols)}"
        )

    pin_memory = (device_t.type == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory)

    # --- 4) модель/оптимайзер ---
    model = GRUQuantile(in_dim=len(feat_cols), hidden=hidden, num_layers=num_layers, quantiles=quantiles).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    patience = 0

    # --- 5) обучение ---
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        seen_tr = 0

        for xb, yb, _, _ in dl_tr:
            xb = xb.to(device_t, non_blocking=True)
            yb = yb.to(device_t, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            q_pred = model(xb)  # (B, Q), в σ-единицах
            loss = pinball_loss(q_pred, yb, quantiles)
            if torch.isnan(loss) or torch.isinf(loss):
                # пропускаем «плохой» батч
                continue
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            bs = xb.size(0)
            tr_loss_sum += float(loss.item()) * bs
            seen_tr += bs

        tr_loss = tr_loss_sum / max(1, seen_tr)

        # --- 6) валидация ---
        model.eval()
        va_loss_sum = 0.0
        seen_va = 0
        with torch.no_grad():
            for xb, yb, _, _ in dl_va:
                xb = xb.to(device_t, non_blocking=True)
                yb = yb.to(device_t, non_blocking=True)
                q_pred = model(xb)
                loss = pinball_loss(q_pred, yb, quantiles)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                bs = xb.size(0)
                va_loss_sum += float(loss.item()) * bs
                seen_va += bs
        va_loss = va_loss_sum / max(1, seen_va)

        print(f"epoch {ep} | train={tr_loss:.5f}  val={va_loss:.5f}")

        # --- 7) early stopping + чекпоинт ---
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

    # --- 8) мета и загрузка лучших весов ---
    meta = dict(
        feature_cols=feat_cols,
        scale_col=scale_col,
        quantiles=quantiles,
        seq_len=seq_len,
        scaler=scaler,
        hidden=hidden,
        num_layers=num_layers,
        x_jitter_std=float(x_jitter_std),
        seed=int(seed),
    )
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    model.load_state_dict(torch.load(out_path, map_location=device_t))
    model.eval()
    return model, meta


# ----- Прогноз «на сейчас» -----
def predict_now(
    df: pd.DataFrame,
    model_path: str = "models/nn_seq.pt",
    meta_path: str = "models/nn_seq_meta.pkl",
    device: str = "cpu",
) -> Dict[str, float]:
    device = _auto_device(device)
    device_t = torch.device(device)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    feat_cols = meta["feature_cols"]
    scale_col = meta["scale_col"]
    quantiles = meta["quantiles"]
    seq_len = int(meta["seq_len"])
    scaler: StandardScaler = meta["scaler"]

    model = GRUQuantile(
        in_dim=len(feat_cols),
        hidden=int(meta["hidden"]),
        num_layers=int(meta["num_layers"]),
        quantiles=quantiles,
    ).to(device_t)
    model.load_state_dict(torch.load(model_path, map_location=device_t))
    model.eval()

    # отфильтруем валид
    work = df.dropna(subset=["y", scale_col]).copy()
    if len(work) < seq_len:
        raise ValueError("Недостаточно строк для последней последовательности.")

    X = work[feat_cols].astype(float).values
    Xz = scaler.transform(X)
    Xz = _nan_to_num_(Xz)

    x_seq = torch.from_numpy(Xz[-seq_len:]).float().unsqueeze(0).to(device_t)  # (1, L, F)
    sigma = float(np.clip(work[scale_col].iloc[-1], 1e-8, None))
    now_price = float(work["close"].iloc[-1])

    with torch.no_grad():
        q_scaled = model(x_seq)[0].detach().cpu().numpy()  # (Q,)
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
    ap.add_argument("--device", default="auto", help='"cpu" | "cuda" | "mps" | "auto"')
    ap.add_argument("--mode", choices=["train", "predict"], default="train")
    ap.add_argument("--out", default="models/nn_seq.pt")
    ap.add_argument("--meta", default="models/nn_seq_meta.pkl")
    ap.add_argument("--quantiles", default="0.1,0.5,0.9")
    ap.add_argument("--x-jitter-std", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
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
            x_jitter_std=args.x_jitter_std,
            grad_clip=args.grad_clip,
            seed=args.seed,
        )
    else:
        out = predict_now(df, model_path=args.out, meta_path=args.meta, device=args.device)
        # компактный принт
        nice = {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in out.items()}
        print(nice)


if __name__ == "__main__":
    main()