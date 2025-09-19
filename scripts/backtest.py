# scripts/backtest.py
import os, sys, argparse
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config as CFG

from backtest.runner import BacktestRunner, BTConfig, ClassicalPack, SeqModel

def main():
    ap = argparse.ArgumentParser(description="Unified WF backtest (classical + seq + future transformers)")
    ap.add_argument("--features", default=CFG.FEAT_FILE)
    ap.add_argument("--packs", nargs="*", default=[], help="paths to classical packs .pkl")
    ap.add_argument("--seq", nargs="*", default=[], help='seq models: backend:model.pt:meta.pkl[:device], e.g. "gru:models/nn_seq.pt:models/nn_seq_meta.pkl:cuda"')
    ap.add_argument("--device", default="cpu", help="default device for seq/transformers")
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--val-days", type=int, default=60)
    ap.add_argument("--step-days", type=int, default=30)
    ap.add_argument("--optimize-weights", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.features).sort_index()
    scale_col = "sigma_ewma_7d" if "sigma_ewma_7d" in df.columns else "sigma_7d"
    feat_cols = [c for c in df.columns if c not in ("y","close") and df[c].dtype.kind != "O"]

    bars_per_hour = {"1H":1,"30T":2,"1T":60}[CFG.BAR]
    cfg = BTConfig(
        bars_per_day=bars_per_hour*24,
        train_days=args.train_days,
        val_days=args.val_days,
        step_days=args.step_days,
        optimize_weights=args.optimize_weights,
    )

    classical = [ClassicalPack(p) for p in args.packs]
    seq_models = []
    for s in args.seq:
        parts = s.split(":")
        if len(parts) < 3:
            raise ValueError(f"Bad --seq spec: {s}")
        backend, model, meta = parts[:3]
        device = parts[3] if len(parts) >= 4 else args.device
        seq_models.append(SeqModel(backend=backend, model=model, meta=meta, device=device))

    runner = BacktestRunner(df, scale_col, feat_cols, cfg)
    res = runner.run(classical=classical, seq_models=seq_models)
    print("[WF]", res)

if __name__ == "__main__":
    main()