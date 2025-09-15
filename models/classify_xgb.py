# models/classify_xgb.py
import pickle
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


BIG_MOVE_PCT = 0.06  # 6% по |y|

def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["y_up"]  = (out["y"] > 0).astype(int)
    out["y_big"] = (out["y"].abs() > BIG_MOVE_PCT).astype(int)
    return out

XGB_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=3.0,
    n_jobs=-1,
    objective="binary:logistic",
    eval_metric="logloss",
)

def _fit_platt(p_raw: np.ndarray, y: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(max_iter=1000)
    if len(np.unique(y)) < 2:
        lr.classes_ = np.array([0, 1])
        lr.coef_ = np.zeros((1, 1))
        lr.intercept_ = np.zeros(1)
        lr.n_features_in_ = 1
        return lr
    lr.fit(p_raw.reshape(-1, 1), y)
    return lr

def train_binary_heads_xgb(df: pd.DataFrame, feature_cols: List[str], save_path: str = "models/binary_heads_xgb.pkl"):
    df = make_labels(df)
    cv = TimeSeriesSplit(n_splits=5)
    results: Dict[str, Dict] = {}

    for target in ["y_up", "y_big"]:
        ll_list, br_list, auc_list = [], [], []

        classes = np.array([0, 1])
        weights = compute_class_weight("balanced", classes=classes, y=df[target].values)
        scale_pos_weight = float(weights[1] / weights[0]) if weights[0] > 0 else 1.0

        for tr_idx, va_idx in cv.split(df):
            Xtr, ytr = df.iloc[tr_idx][feature_cols].astype(float), df.iloc[tr_idx][target].astype(int)
            Xva, yva = df.iloc[va_idx][feature_cols].astype(float), df.iloc[va_idx][target].astype(int)

            base = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
            base.fit(Xtr, ytr)
            p_tr = base.predict_proba(Xtr)[:, 1]
            p_va = base.predict_proba(Xva)[:, 1]

            cal = _fit_platt(p_tr, ytr)
            p_va_cal = cal.predict_proba(p_va.reshape(-1, 1))[:, 1]

            ll_list.append(log_loss(yva, p_va_cal, labels=[0, 1]))
            br_list.append(brier_score_loss(yva, p_va_cal))
            try:
                auc_list.append(roc_auc_score(yva, p_va_cal))
            except ValueError:
                pass

        print(f"[XGB {target}] logloss={np.mean(ll_list):.4f}, brier={np.mean(br_list):.4f}, auc={np.mean(auc_list) if auc_list else float('nan'):.4f}")

        # финальная
        X_all, y_all = df[feature_cols].astype(float), df[target].astype(int)
        base_full = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
        base_full.fit(X_all, y_all)
        p_all = base_full.predict_proba(X_all)[:, 1]
        cal_full = _fit_platt(p_all, y_all)

        results[target] = dict(base=base_full, cal=cal_full, features=feature_cols)

    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    return results