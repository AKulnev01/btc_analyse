"""
CLI: собрать фичи (по желанию), обучить квантили и бинарные головы,
и опционально сразу распечатать прогноз на +7д.
"""
import os
import sys
import argparse
import pandas as pd

# гарантируем, что корень проекта в пути
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config as CFG
from models.train_quantile import train_quantiles
from models.classify import train_binary_heads, predict_probs
from models.score import predict_7d_price


def _maybe_make_features(features_path: str) -> None:
    """Если файла фичей нет — собираем на лету."""
    if os.path.exists(features_path):
        return

    print(f"[INFO] features not found -> building: {features_path}")
    from features.build_features import build_feature_table

    macro_df = None
    try:
        if getattr(CFG, "MACRO_FILE", None) and os.path.exists(CFG.MACRO_FILE):
            macro_df = pd.read_parquet(CFG.MACRO_FILE)
    except Exception:
        pass

    news_path = None
    try:
        if getattr(CFG, "NEWS_FILE", None) and os.path.exists(CFG.NEWS_FILE):
            news_path = CFG.NEWS_FILE
    except Exception:
        pass

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


def _predict_now(df: pd.DataFrame) -> None:
    """Распечатать прогноз P10/P50/P90 (уровни цены) + вероятности бинарных голов."""
    out = {}
    try:
        out.update(predict_7d_price(df))
    except Exception as e:
        print(f"[WARN] predict_7d_price failed: {e}")

    try:
        probs = predict_probs('models/binary_heads.pkl', df.iloc[[-1]])
        out.update(probs)
    except Exception as e:
        print(f"[INFO] no binary heads or scoring failed: {e}")

    if out:
        def _r(x):
            try:
                return round(float(x), 4)
            except Exception:
                return x
        pretty = {k: _r(v) for k, v in out.items()}
        print(pretty)
    else:
        print("[INFO] no output produced")


def main():
    p = argparse.ArgumentParser(description="Train quantiles + binary heads; optionally build features and predict.")
    p.add_argument('--features', default=getattr(CFG, 'FEAT_FILE', './data/features/btc_1h_features.parquet'),
                  help='путь к parquet с фичами')
    p.add_argument('--make-features', action='store_true', help='если указано — соберём фичи перед обучением')
    p.add_argument('--predict', action='store_true', help='после обучения распечатать прогноз на «сейчас»')
    args = p.parse_args()

    if args.make_features:
        _maybe_make_features(args.features)

    if not os.path.exists(args.features):
        raise FileNotFoundError(f"features file not found: {args.features}. "
                                f"Запусти с --make-features или создай вручную.")

    df = pd.read_parquet(args.features)
    feature_cols = [c for c in df.columns if c not in ('y', 'close') and df[c].dtype.kind != 'O']

    train_quantiles(df, feature_cols)
    train_binary_heads(df, feature_cols)

    if args.predict:
        _predict_now(df)


if __name__ == '__main__':
    main()