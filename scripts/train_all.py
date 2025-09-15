# scripts/train_all.py
"""
CLI: собрать фичи (по желанию), обучить квантили и бинарные головы,
и опционально распечатать прогноз на +TARGET_HOURS.
"""
import os
import sys
import argparse
import pandas as pd

# гарантируем корень в пути
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config as CFG
from features.build_features import build_feature_table

# наши тренеры
from models.train_quantile import train_quantiles as train_quantiles_lgbm
from models.classify import train_binary_heads as train_heads_lgbm

# новые
from models.model_zoo import train_quantiles_backend
from models.classify_xgb import train_binary_heads_xgb


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
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    df.to_parquet(features_path)
    print(f"[OK] saved features -> {features_path} (rows={len(df)})")


def main():
    p = argparse.ArgumentParser(description="Train quantiles + binary heads; optionally build features and predict.")
    p.add_argument("--features", default=getattr(CFG, "FEAT_FILE", "./data/features/btc_1h_features.parquet"))
    p.add_argument("--make-features", action="store_true")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--quant-backend", choices=["lgbm", "cat"], default="lgbm",
                  help="бэкенд для квантилей")
    p.add_argument("--clf-backend", choices=["lgbm", "xgb"], default="lgbm",
                  help="бэкенд для бинарных голов")
    p.add_argument("--out-quant", default="models/quantile_models.pkl")
    p.add_argument("--out-heads", default=None)
    args = p.parse_args()

    if args.make-features:
        _maybe_make_features(args.features)

    if not os.path.exists(args.features):
        raise FileNotFoundError(f"features file not found: {args.features}. Запусти с --make-features или создай вручную.")

    df = pd.read_parquet(args.features)
    feature_cols = [c for c in df.columns if c not in ("y", "close") and df[c].dtype.kind != "O"]

    # квантили
    if args.quant-backend if False else None:  # заглушка для линтера
        pass
    if args.quant_backend == "lgbm":
        train_quantiles_lgbm(df, feature_cols, out_path=args.out_quant)
    else:
        train_quantiles_backend(df, feature_cols, out_path=args.out_quant, backend="cat")

    # головы
    out_heads = args.out_heads or ( "models/binary_heads.pkl" if args.clf_backend=="lgbm" else "models/binary_heads_xgb.pkl" )
    if args.clf_backend == "lgbm":
        train_heads_lgbm(df, feature_cols, save_path=out_heads)
    else:
        train_binary_heads_xgb(df, feature_cols, save_path=out_heads)

    # опционально — прогноз «на сейчас»
    if args.predict:
        try:
            from models.score import predict_7d_price
            out = predict_7d_price(df, models_path=args.out_quant)
            print(out)
        except Exception as e:
            print(f"[WARN] predict failed: {e}")


if __name__ == "__main__":
    main()