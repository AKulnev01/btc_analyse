# models/classify.py — бинарные головы: P(up), P(big_move) + калибровка
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from typing import Dict, Optional, Union, List

# Порог «крупного хода» (лог-доходность за 7д ~ 6%)
BIG_MOVE_PCT = 0.06  # 6%

# Общие параметры для классификатора
LGB_PARAMS_CLF = dict(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_data_in_leaf=50,
    min_sum_hessian_in_leaf=1e-3,
    feature_fraction=0.9,     # используем feature_fraction вместо colsample_bytree
    subsample=0.9,
    subsample_freq=1,
    reg_lambda=5.0,
    min_split_gain=0.0,
    max_bin=511,
    force_row_wise=True,
    objective="binary",
    n_jobs=-1,
    verbosity=-1,
)

class PlattWrapper:
    """Обертка: LGBM + Platt-калибровка вероятностей."""
    def __init__(self, base, platt):
        self.base = base
        self.platt = platt
    def predict_proba(self, X):
        p_raw = self.base.predict_proba(X)[:, 1]
        p_cal = self.platt.predict_proba(p_raw.reshape(-1,1))[:,1]
        return np.vstack([1.0 - p_cal, p_cal]).T


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Из непрерывной цели y делаем две бинарные метки."""
    out = df.copy()
    out['y_up']  = (out['y'] > 0).astype(int)
    out['y_big'] = (out['y'].abs() > BIG_MOVE_PCT).astype(int)
    return out

def _fit_base(X, y, class_weight):
    return LGBMClassifier(
        **LGB_PARAMS_CLF,
        class_weight=class_weight
    ).fit(X, y)

def _fit_platt(p_raw: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Плати-калибровка: логрег по scalar-признаку p_raw.
    Защищаемся от случая одного класса (возвращаем «тривиальный» калибратор).
    """
    lr = LogisticRegression(max_iter=1000)
    if len(np.unique(y)) < 2:
        lr.classes_ = np.array([0, 1])
        lr.coef_ = np.zeros((1, 1))
        lr.intercept_ = np.zeros(1)
        lr.n_features_in_ = 1
        return lr
    lr.fit(p_raw.reshape(-1, 1), y)
    return lr

def train_binary_heads(df, feature_cols, save_path='models/binary_heads.pkl'):
    """
    Обучаем две головы (y_up, y_big) с Platt-калибровкой вероятностей.
    Возвращаем dict {target: PlattWrapper} и логируем метрики.
    """
    df = make_labels(df)
    feature_cols = [c for c in feature_cols if c in df.columns and df[c].dtype.kind != 'O']
    cv = TimeSeriesSplit(n_splits=5)

    results = {}
    for target in ['y_up', 'y_big']:
        ll_list, br_list, auc_list = [], [], []

        classes = np.array([0, 1])
        weights = compute_class_weight('balanced', classes=classes, y=df[target].values)
        class_weight = {0: float(weights[0]), 1: float(weights[1])}

        # walk-forward оценка
        for tr_idx, va_idx in cv.split(df):
            Xtr, ytr = df.iloc[tr_idx][feature_cols].astype(float), df.iloc[tr_idx][target].astype(int)
            Xva, yva = df.iloc[va_idx][feature_cols].astype(float), df.iloc[va_idx][target].astype(int)

            base = _fit_base(Xtr, ytr, class_weight)
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

        print(f"{target}: logloss={np.mean(ll_list):.4f}, "
              f"brier={np.mean(br_list):.4f}, "
              f"auc={np.mean(auc_list) if auc_list else float('nan'):.4f}")

        # финальная модель на всём датасете
        base_full = LGBMClassifier(**LGB_PARAMS_CLF, class_weight=class_weight)
        X_all, y_all = df[feature_cols], df[target]
        base_full.fit(X_all, y_all)
        p_all = base_full.predict_proba(X_all)[:, 1]
        platt_full = _fit_platt(p_all, y_all)

        results[target] = PlattWrapper(base_full, platt_full)

    payload = {'models': results, 'features': feature_cols}
    with open(save_path, 'wb') as f:
        pickle.dump(payload, f)

    return results

def _ensure_df_row(x) -> pd.DataFrame:
    """Принимаем Series/DataFrame/словарь — возвращаем DataFrame с 1 строкой."""
    if isinstance(x, pd.Series):
        return x.to_frame().T
    if isinstance(x, dict):
        return pd.DataFrame([x])
    return x  # уже DataFrame

def predict_probs(models_path: str, last_features) -> dict:
    """
    Инференс вероятностей P_up и P_big из сохранённых бинарных голов.
    """
    with open(models_path, 'rb') as f:
        payload = pickle.load(f)

    models = payload.get('models', payload)
    feat = payload.get('features', None)

    Xall = _ensure_df_row(last_features)
    if feat is None:
        feat = [c for c in Xall.columns if c not in ('y','close') and Xall[c].dtype.kind != 'O']
    X = Xall[feat].astype(float)

    out = {}
    for key, model in models.items():
        tgt = 'P_up' if 'up' in key else ('P_big' if 'big' in key else f"P_{key}")
        proba = model.predict_proba(X)[:, 1]
        out[tgt] = float(proba[-1])
    return out