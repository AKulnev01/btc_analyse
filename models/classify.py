# models/classify.py — бинарные головы: P(up > X%), P(down > X%) с Platt-калибровкой, по нескольким горизонтам
import pickle
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression

# Базовые параметры
LGB_PARAMS_CLF = dict(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_data_in_leaf=50,
    min_sum_hessian_in_leaf=1e-3,
    feature_fraction=0.9,
    subsample=0.9,
    subsample_freq=1,
    reg_lambda=5.0,
    min_split_gain=0.0,
    max_bin=511,
    force_row_wise=True,
    objective='binary',
    n_jobs=-1,
    verbosity=-1,
)

# Пороги для "up/down > X%" — в долях (0.005 = 0.5%)
try:
    import config as CFG
    UP_THR_PCT = getattr(CFG, "UP_THR_PCT", {1: 0.002, 4: 0.005, 24: 0.02})
    TARGET_HOURS = int(getattr(CFG, "TARGET_HOURS", 4))
    MULTI_HOURS = list(dict.fromkeys(
        [1, 4, 24] + list(getattr(CFG, "MULTI_TARGET_HOURS", [])) + [TARGET_HOURS]
    ))
except Exception:
    UP_THR_PCT = {1: 0.002, 4: 0.005, 24: 0.02}
    TARGET_HOURS = 4
    MULTI_HOURS = [1, 4, 24, 4]

def _log_thr(pct):
    # перевод порога в лог-доходность
    return float(np.log1p(float(pct)))

class PlattWrapper:
    def __init__(self, base, platt):
        self.base = base
        self.platt = platt
    def predict_proba(self, X):
        p_raw = self.base.predict_proba(X)[:, 1]
        p_cal = self.platt.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        return np.vstack([1.0 - p_cal, p_cal]).T

def _fit_one_head(Xtr, ytr, Xva, yva, class_weight) -> (PlattWrapper, Dict[str, float]):
    base = LGBMClassifier(**LGB_PARAMS_CLF, class_weight=class_weight)
    base.fit(Xtr, ytr)
    # калибровка Platt по train
    p_tr = base.predict_proba(Xtr)[:, 1]
    p_va = base.predict_proba(Xva)[:, 1]
    cal = LogisticRegression(max_iter=1000)
    # защита от одного класса
    if len(np.unique(ytr)) < 2:
        cal.classes_ = np.array([0, 1])
        cal.coef_ = np.zeros((1, 1))
        cal.intercept_ = np.zeros(1)
        cal.n_features_in_ = 1
    else:
        cal.fit(p_tr.reshape(-1, 1), ytr)
    p_va_cal = cal.predict_proba(p_va.reshape(-1, 1))[:, 1]
    # метрики
    out = dict(
        logloss=float(log_loss(yva, p_va_cal, labels=[0, 1])),
        brier=float(brier_score_loss(yva, p_va_cal)),
        auc=float(roc_auc_score(yva, p_va_cal)) if len(np.unique(yva)) == 2 else float("nan"),
    )
    return PlattWrapper(base, cal), out

def train_binary_heads(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizons: Optional[List[int]] = None,
    out_path: str = 'models/binary_heads.pkl'
) -> Dict[str, PlattWrapper]:
    """
    Для каждого горизонта H из horizons создаёт две головы:
      - y_up_thr_h{H}: 1 если y_h{H} >= log(1+thr)
      - y_down_thr_h{H}: 1 если y_h{H} <= -log(1+thr)
    Возвращает словарь {label_name: PlattWrapper} и сохраняет его.
    """
    horizons = horizons or list(dict.fromkeys(MULTI_HOURS))
    feats = [c for c in feature_cols if (c in df.columns and df[c].dtype.kind != 'O')]

    results: Dict[str, PlattWrapper] = {}
    logs: Dict[str, Dict[str, float]] = {}

    for H in horizons:
        ycol = f"y_h{H}" if f"y_h{H}" in df.columns else "y"
        if ycol not in df.columns:
            continue
        thr = _log_thr(UP_THR_PCT.get(H, 0.005))

        d = df.dropna(subset=[ycol]).copy()
        X = d[feats].astype(float)
        y_up = (d[ycol] >= thr).astype(int)
        y_dn = (d[ycol] <= -thr).astype(int)

        split = int(len(d) * 0.8)
        Xtr, Xva = X.iloc[:split], X.iloc[split:]
        yup_tr, yup_va = y_up.iloc[:split], y_up.iloc[split:]
        ydn_tr, ydn_va = y_dn.iloc[:split], y_dn.iloc[split:]

        for name, ytr, yva in [ (f"y_up_thr_h{H}", yup_tr, yup_va), (f"y_down_thr_h{H}", ydn_tr, ydn_va) ]:
            classes = np.array([0, 1])
            weights = compute_class_weight('balanced', classes=classes, y=ytr.values)
            class_weight = {0: float(weights[0]), 1: float(weights[1])}
            model, mlog = _fit_one_head(Xtr, ytr, Xva, yva, class_weight)
            results[name] = model
            logs[name] = mlog
            print(f"{name}: logloss={mlog['logloss']:.4f}, brier={mlog['brier']:.4f}, auc={mlog['auc']:.4f}")

    # бэквард-совместимость: сделаем alias для текущего TARGET_HOURS
    H0 = int(getattr(CFG, "TARGET_HOURS", TARGET_HOURS))
    if f"y_up_thr_h{H0}" in results:
        results["y_up"] = results[f"y_up_thr_h{H0}"]

    payload = {'models': results, 'features': feats, 'logs': logs, 'horizons': horizons, 'up_thr_pct': UP_THR_PCT}
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)
    return results