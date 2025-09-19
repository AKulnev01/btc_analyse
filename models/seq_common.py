# models/seq_common.py
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
SCALE_COLS = ["sigma_ewma_7d", "sigma_7d"]

def pick_scale_col(df: pd.DataFrame) -> str:
    for c in SCALE_COLS:
        if c in df.columns:
            return c
    raise ValueError("Нет sigma_ewma_7d/sigma_7d")

def feature_cols_auto(df: pd.DataFrame, explicit: Optional[List[str]] = None) -> List[str]:
    if explicit is not None:
        return [c for c in explicit if c in df.columns and df[c].dtype.kind != "O"]
    bad = {"y","close"}
    return [c for c in df.columns if c not in bad and df[c].dtype.kind != "O"]

def sanitize_dataframe(
    df: pd.DataFrame,
    feat_cols: List[str],
    scale_col: str,
    na_frac_threshold: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    work = df.copy()
    work = work.loc[work["y"].notna() & work[scale_col].notna()].copy()
    work.replace([np.inf, -np.inf], np.nan, inplace=True)

    feat_cols = [c for c in feat_cols if c in work.columns and work[c].dtype.kind != "O"]
    if not feat_cols:
        raise ValueError("Нет числовых фич")
    na_frac = work[feat_cols].isna().mean()
    keep = [c for c in feat_cols if float(na_frac.get(c,0.0)) <= na_frac_threshold]
    if not keep:
        raise ValueError("Все фичи отфильтрованы по NaN")
    work[keep] = work[keep].fillna(0.0)
    work = work[work[scale_col] > 1e-8].copy()

    needed = keep + ["y", scale_col, "close"]
    arr = work[needed].to_numpy(dtype=float)
    row_mask = np.isfinite(arr).all(axis=1)
    work = work.iloc[row_mask].copy()
    if len(work)==0:
        raise ValueError("После sanitize нет строк")
    return work, keep

class StandardScalerSafe:
    def __init__(self):
        self.mean_=None; self.std_=None; self.cols=None
    def fit(self, X: np.ndarray, cols: Optional[List[str]]=None):
        self.cols = cols
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd = np.where(sd<=0.0, 1.0, sd)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd), sd, 1.0)
        self.mean_=mu; self.std_=sd
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.nan_to_num(X, copy=False) - self.mean_) / self.std_
        if not np.isfinite(Z).all():
            raise ValueError("[StandardScalerSafe] NaN/inf in transform")
        return Z

class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feat_cols: List[str], scale_col: str, seq_len: int, scaler: StandardScalerSafe):
        self.df=df.copy(); self.feat_cols=list(feat_cols); self.scale_col=scale_col; self.seq_len=int(seq_len); self.scaler=scaler
        X = self.df[self.feat_cols].to_numpy(dtype=float)
        self.Xz = self.scaler.transform(X)
        self.sigma = self.df[self.scale_col].to_numpy(dtype=float).clip(min=1e-8)
        self.y = self.df["y"].to_numpy(dtype=float)
        self.y_scaled = self.y / self.sigma
        self.valid_idx = np.arange(self.seq_len-1, len(self.df))
    def __len__(self): return len(self.valid_idx)
    def __getitem__(self, i:int):
        idx=self.valid_idx[i]; sl=slice(idx-self.seq_len+1, idx+1)
        x_seq=self.Xz[sl]; y_scaled=self.y_scaled[idx]; sigma=self.sigma[idx]; p=float(self.df["close"].iloc[idx])
        import torch
        return (torch.from_numpy(x_seq).float(),
                torch.tensor([y_scaled]).float(),
                torch.tensor([sigma]).float(),
                torch.tensor([p]).float())

def pinball_loss(pred, target, quantiles: List[float]):
    import torch
    B,Q = pred.shape
    t = target.repeat(1,Q)
    qs = torch.tensor(quantiles, device=pred.device).view(1,Q).repeat(B,1)
    diff = t - pred
    return torch.maximum(qs*diff, (qs-1.0)*diff).mean()