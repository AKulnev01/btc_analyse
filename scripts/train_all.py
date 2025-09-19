# scripts/train_all.py
"""
CLI: собрать фичи (по желанию), обучить квантили и бинарные головы,
дополнительно: быстрая walk-forward CV (--wf-cv) для lgbm / cat / xgb,
и опционально seq-модели.
"""
import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd

# гарантируем корень в пути
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config as CFG
from features.build_features import build_feature_table

# классические тренеры
from models.train_quantile import train_quantiles as train_quantiles_lgbm
from models.classify import train_binary_heads as train_heads_lgbm

# новые
from models.model_zoo import train_quantiles_backend
from models.classify_xgb import train_binary_heads_xgb
from models.xgb_quantile import train_quantiles_xgb_dart
from etl.blockchain import load_blockchain_metrics
from etl.fred import load_fred_metrics

from utils.logging_why import log_dataframe, log_predictions

MACRO_FILE = "./data/interim/macro.parquet"

# ====== helpers ======
def _maybe_make_features(features_path: str) -> None:
    if os.path.exists(features_path):
        return
    print(f"[INFO] features not found -> building: {features_path}")

    macro_df = None
    try:
        if getattr(CFG, "MACRO_FILE", None) and os.path.exists(CFG.MACRO_FILE):
            macro_df = pd.read_parquet(CFG.MACRO_FILE)
    except Exception:
        macro_df = None

    news_path = None
    try:
        if getattr(CFG, "NEWS_FILE", None) and os.path.exists(CFG.NEWS_FILE):
            news_path = CFG.NEWS_FILE
    except Exception:
        news_path = None

    df = build_feature_table(
        price_path=CFG.PX_FILE,
        oi_path=getattr(CFG, "OI_FILE", ""),
        funding_path=getattr(CFG, "FUND_FILE", ""),
        news_path=news_path,
        macro_df=macro_df,
    )

    # --- новые внешние источники ---
    try:
        print("[INFO] merging blockchain metrics...")
        df_bc = load_blockchain_metrics()
        df = df.join(df_bc, how="left")
    except Exception as e:
        print(f"[WARN] blockchain metrics skipped: {e}")

    try:
        print("[INFO] merging FRED macro metrics...")
        df_fred = load_fred_metrics()
        df = df.join(df_fred, how="left")
    except Exception as e:
        print(f"[WARN] FRED metrics skipped: {e}")

    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    df.to_parquet(features_path)
    print(f"[OK] saved features -> {features_path} (rows={len(df)})")

# ====== Walk-forward CV (LGBM / CatBoost / XGBoost) ======
def _wf_make_splits(n_rows: int, bars_per_day: int, train_days=365, val_days=60, step_days=30):
    train_len = train_days * bars_per_day
    val_len = val_days * bars_per_day
    step_len = step_days * bars_per_day
    i_start = train_len
    while i_start + val_len <= n_rows:
        tr_slice = slice(i_start - train_len, i_start)
        va_slice = slice(i_start, i_start + val_len)
        yield tr_slice, va_slice
        i_start += step_len

def _wf_cv(df: pd.DataFrame, feature_cols, scale_col: str,
           backend: str = "lgbm",
           quantiles=(0.1, 0.5, 0.9),
           train_days=365, val_days=60, step_days=30):
    """
    Унифицированная WF-оценка для lgbm/cat/xgb.
    Для xgb используем xgboost.train с кастомным pinball-объектом.
    """
    work = df.dropna(subset=["y", scale_col]).copy()
    X = work[feature_cols].astype(float).values
    y = work["y"].astype(float).values
    sigma = work[scale_col].astype(float).clip(1e-8).values

    bars_per_hour = {"1H": 1, "30T": 2, "1T": 60}[CFG.BAR]
    bars_per_day = bars_per_hour * 24

    mae_list, cover_list, folds = [], [], 0

    if backend == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as e:
            raise ImportError("LightGBM required for --wf-cv backend=lgbm") from e

    if backend == "cat":
        try:
            from catboost import CatBoostRegressor, Pool
        except Exception as e:
            raise ImportError("CatBoost required for --wf-cv backend=cat") from e

    if backend == "xgb":
        try:
            import xgboost as xgb
        except Exception as e:
            raise ImportError("XGBoost required for --wf-cv backend=xgb") from e

        def _pinball_obj(alpha: float):
            # кастомный градиент/гессиан для pinball (квантили)
            def obj(preds: np.ndarray, dtrain: "xgb.DMatrix"):
                y_true = dtrain.get_label()
                diff = y_true - preds
                grad = np.where(diff >= 0.0, -alpha, 1.0 - alpha)
                hess = np.ones_like(grad) * 1e-6  # псевдо-гессиан (константа)
                return grad, hess
            return obj

        def _train_xgb_quantile(Xtr, ytr_scaled, alpha: float):
            dtr = xgb.DMatrix(Xtr, label=ytr_scaled)
            params = {
                "max_depth": 6,
                "eta": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.95,
                "min_child_weight": 1.0,
                "lambda": 2.0,
                "verbosity": 0,
                "tree_method": "hist",
                "objective": "reg:squarederror",  # будет переопределён кастомным obj
            }
            bst = xgb.train(
                params,
                dtr,
                num_boost_round=600,   # можно увеличить
                obj=_pinball_obj(alpha),
            )
            return bst

    for tr_sl, va_sl in _wf_make_splits(len(work), bars_per_day, train_days, val_days, step_days):
        Xtr, Xva = X[tr_sl], X[va_sl]
        ytr, yva = y[tr_sl], y[va_sl]
        s_tr, s_va = sigma[tr_sl], sigma[va_sl]

        # y_scaled для обучения
        ytr_scaled = ytr / s_tr

        models = {}
        if backend == "lgbm":
            for q in quantiles:
                m = LGBMRegressor(
                    objective="quantile", alpha=float(q),
                    n_estimators=1200, learning_rate=0.05,
                    num_leaves=127, subsample=0.9, subsample_freq=1,
                    feature_fraction=0.95, reg_lambda=2.0,
                    force_row_wise=True, verbosity=-1, seed=42
                )
                m.fit(Xtr, ytr_scaled)
                models[q] = m

            q10 = models[0.1].predict(Xva) * s_va
            q50 = models[0.5].predict(Xva) * s_va
            q90 = models[0.9].predict(Xva) * s_va

        elif backend == "cat":
            train_pool = Pool(Xtr, ytr_scaled)
            valid_pool = Pool(Xva, yva)  # валид не обязателен, но пусть будет для консистентности
            for q in quantiles:
                m = CatBoostRegressor(
                    loss_function=f"Quantile:alpha={float(q)}",
                    iterations=2000,  # fast WF
                    depth=8, learning_rate=0.05,
                    l2_leaf_reg=2.0,
                    random_seed=42,
                    od_type="Iter", od_wait=100,
                    verbose=False, allow_writing_files=False
                )
                m.fit(train_pool)
                models[q] = m

            q10 = models[0.1].predict(Xva) * s_va
            q50 = models[0.5].predict(Xva) * s_va
            q90 = models[0.9].predict(Xva) * s_va

        elif backend == "xgb":
            import xgboost as xgb
            dva = xgb.DMatrix(Xva)
            for q in quantiles:
                bst = _train_xgb_quantile(Xtr, ytr_scaled, alpha=float(q))
                models[q] = bst

            q10 = models[0.1].predict(dva) * s_va
            q50 = models[0.5].predict(dva) * s_va
            q90 = models[0.9].predict(dva) * s_va

        else:
            raise ValueError(f"Unknown backend for WF: {backend}")

        # метрики
        mae50 = float(np.mean(np.abs(yva - q50)))
        lo = np.minimum(q10, q50)
        hi = np.maximum(q90, q50)
        cover = float(np.mean((yva >= lo) & (yva <= hi)))

        mae_list.append(mae50)
        cover_list.append(cover)
        folds += 1

    out = {
        "MAE50_log_mean": float(np.mean(mae_list)) if mae_list else None,
        "COVER_10_90_mean": float(np.mean(cover_list)) if cover_list else None,
        "folds": folds,
    }
    return out


def main():
    p = argparse.ArgumentParser(description="Train quantiles + binary heads; optionally build features and predict.")
    p.add_argument("--features", default=getattr(CFG, "FEAT_FILE", "./data/features/btc_1h_features.parquet"))
    p.add_argument("--make-features", action="store_true")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--quant-backend", choices=["lgbm", "cat", "xgb"], default="lgbm", help="бэкенд для квантилей")
    p.add_argument("--clf-backend", choices=["lgbm", "xgb"], default="lgbm", help="бэкенд для бинарных голов")
    p.add_argument("--seq-train", action="store_true")
    p.add_argument("--seq-backend", choices=["gru", "tcn"], default="gru")
    p.add_argument("--out-quant", default="models/quantile_models.pkl")
    p.add_argument("--out-heads", default=None)
    p.add_argument("--wf-cv", action="store_true", help="быстрая walk-forward оценка выбранного бэкенда (lgbm/cat/xgb)")
    # WF гиперпараметры (по желанию)
    p.add_argument("--wf-train-days", type=int, default=365)
    p.add_argument("--wf-val-days", type=int, default=60)
    p.add_argument("--wf-step-days", type=int, default=30)
    args = p.parse_args()

    if args.make_features:
        _maybe_make_features(args.features)

    if not os.path.exists(args.features):
        raise FileNotFoundError(f"features file not found: {args.features}. Запусти с --make-features или создай вручную.")

    df = pd.read_parquet(args.features)
    log_dataframe(df, name="train_features")
    feature_cols = [c for c in df.columns if c not in ("y", "close") and df[c].dtype.kind != "O"]

    # --- WF-CV: быстрый тест без сохранения моделей ---
    if args.wf_cv:
        scale_col = "sigma_ewma_7d" if "sigma_ewma_7d" in df.columns else "sigma_7d"
        metrics = _wf_cv(
            df, feature_cols, scale_col,
            backend=args.quant_backend,
            quantiles=(0.1, 0.5, 0.9),
            train_days=args.wf_train_days,
            val_days=args.wf_val_days,
            step_days=args.wf_step_days,
        )
        print("[WF]", metrics)
        return

    # --- квантильные модели ---
    if args.quant_backend == "lgbm":
        train_quantiles_lgbm(df, feature_cols, out_path=args.out_quant)
    elif args.quant_backend == "cat":
        train_quantiles_backend(df, feature_cols, out_path=args.out_quant, backend="cat")
    else:
        # XGBoost DART
        print("[train xgb] delegating to train_quantiles_xgb_dart()")
        train_quantiles_xgb_dart(df, feature_cols, out_path=args.out_quant)

    # --- бинарные головы ---
    out_heads = args.out_heads or ("models/binary_heads.pkl" if args.clf_backend == "lgbm" else "models/binary_heads_xgb.pkl")
    if args.clf_backend == "lgbm":
        # ВНИМАНИЕ: в этой функции параметр называется out_path
        train_heads_lgbm(df, feature_cols, out_path=out_heads)
    else:
        train_binary_heads_xgb(df, feature_cols, save_path=out_heads)

    # --- опционально: обучение seq-модели (в том же запуске) ---
    if args.seq_train:
        if args.seq_backend == "gru":
            cmd = [
                sys.executable, "-m", "models.nn_seq",
                "--features", args.features,
                "--mode", "train",
                "--seq-len", "96",
                "--device", "cuda",
                "--out", "models/nn_seq_30m.pt",
                "--meta", "models/nn_seq_30m_meta.pkl",
                "--epochs", "5",
            ]
            print("[SEQ] launching GRU:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            print("[SEQ] training TCN in-process")
            from models.tcn_seq import train_tcn
            train_tcn(df, seq_len=96, device="cuda", out_path="models/tcn_seq.pt", meta_path="models/tcn_seq_meta.pkl")

    # --- опционально — прогноз «на сейчас» от классических квантильных моделей ---
    if args.predict:
        try:
            from models.score import predict_quantiles
            out = predict_quantiles(df, models_path=args.out_quant)
            print("[NOW]", out)
            log_predictions(out, name="predictions")
        except Exception as e:
            print(f"[WARN] predict failed: {e}")


if __name__ == "__main__":
    main()