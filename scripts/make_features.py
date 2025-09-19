# scripts/make_features.py
"""CLI: собрать фичи и сохранить в parquet (с опц. on-chain + FRED, whylogs-логированием)."""
import argparse
import sys
import pandas as pd
import numpy as np

import config as CFG
from features.build_features import build_feature_table

# внешние источники (не обязательны)
try:
    from etl.blockchain import load_blockchain_metrics  # free explorer charts
except Exception:
    load_blockchain_metrics = None

try:
    from etl.fred import load_fred_metrics
except Exception:
    load_fred_metrics = None

# whylogs (не обязателен)
try:
    from utils.logging_why import log_dataframe
except Exception:
    def log_dataframe(*_args, **_kwargs):
        print("[whylogs] not installed, skip logging")


def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        # пытаемся найти временную колонку
        for cand in ("timestamp", "ts", "time", "date", "datetime"):
            if cand in d.columns:
                d = d.set_index(cand)
                break
    d.index = pd.to_datetime(d.index, utc=True, errors="coerce")
    d = d[~d.index.isna()].sort_index()
    d = d[~d.index.duplicated(keep="last")]
    return d


def _safe_join(base: pd.DataFrame, extra: pd.DataFrame, name: str) -> pd.DataFrame:
    if extra is None or len(extra) == 0:
        print(f"[make_features] {name}: nothing to merge (empty)")
        return base
    extra = _to_utc_index(extra)
    # подгоняем к сетке base, тянем вперёд «медленные» метрики
    extra = extra.reindex(base.index, method="ffill")
    before = base.shape[1]
    out = base.join(extra, how="left")
    added = out.shape[1] - before
    print(f"[make_features] {name}: merged, +{added} cols")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prices',   default=CFG.PX_FILE)
    ap.add_argument('--oi',       default=getattr(CFG, 'OI_FILE', ''))
    ap.add_argument('--funding',  default=getattr(CFG, 'FUND_FILE', ''))
    ap.add_argument('--news',     default=getattr(CFG, 'NEWS_FILE', ''))
    ap.add_argument('--macro',    default=getattr(CFG, 'MACRO_FILE', ''))
    ap.add_argument('--onchain',  action="store_true", help="подключить on-chain метрики Blockchain.com (free explorer)")
    ap.add_argument('--fred',     action="store_true", help="подключить макро из FRED API (нужен FRED_API_KEY в .env)")
    ap.add_argument('--out',      default=CFG.FEAT_FILE)
    args = ap.parse_args()

    # --- базовые данные (цены, деривы, новости, внутр. макро) ---
    macro_df = None
    if args.macro and args.macro.endswith('.parquet'):
        try:
            macro_df = pd.read_parquet(args.macro)
            macro_df = _to_utc_index(macro_df)
        except Exception as e:
            print(f"[WARN] macro read failed: {e}")

    print("[make_features] building base feature table…")
    df = build_feature_table(
        args.prices,
        args.oi,
        args.funding,
        news_path=args.news,
        macro_df=macro_df,
    )
    df = _to_utc_index(df)
    print(f"[make_features] base features: rows={len(df)}, cols={df.shape[1]}")

    # --- внешние источники ---
    if args.onchain:
        if load_blockchain_metrics is None:
            print("[WARN] etl.blockchain not available. Skip on-chain.")
        else:
            try:
                print("[make_features] merge blockchain.com onchain metrics")
                df_onchain = load_blockchain_metrics()  # уже UTC index
                df = _safe_join(df, df_onchain, "onchain")
            except Exception as e:
                print(f"[WARN] onchain merge failed: {e}")

    if args.fred:
        if load_fred_metrics is None:
            print("[WARN] etl.fred not available. Skip FRED.")
        else:
            try:
                print("[make_features] merge FRED macro indicators")
                df_fred = load_fred_metrics()  # уже UTC index
                df = _safe_join(df, df_fred, "fred")
            except Exception as e:
                print(f"[WARN] fred merge failed: {e}")

    # --- финальные штрихи: числовые типы, пропуски по медленным метрикам ---
    num_cols = [c for c in df.columns if df[c].dtype.kind in "iufcb"]
    df[num_cols] = df[num_cols].astype(np.float32)
    slow_cols = [c for c in df.columns if any(k in c.lower() for k in ("hashrate", "difficulty", "tx_count", "fees", "supply", "fred_", "macro_", "miner"))]
    if slow_cols:
        df[slow_cols] = df[slow_cols].ffill()

    # --- save ---
    df.to_parquet(args.out)
    print("Saved features ->", args.out)

    # whylogs: лог профиля данных
    try:
        log_dataframe(df, name="features_make")
    except Exception as e:
        print(f"[whylogs] skip log: {e}")


if __name__ == '__main__':
    main()