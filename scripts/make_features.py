"""CLI: собрать фичи и сохранить в parquet."""
import argparse, pandas as pd, config as CFG
from features.build_features import build_feature_table

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prices', default=CFG.PX_FILE)
    ap.add_argument('--oi', default=CFG.OI_FILE)
    ap.add_argument('--funding', default=CFG.FUND_FILE)
    ap.add_argument('--news', default=CFG.NEWS_FILE)
    ap.add_argument('--macro', default=getattr(CFG, 'MACRO_FILE', ''))
    ap.add_argument('--out', default=CFG.FEAT_FILE)
    args = ap.parse_args()
    macro_df = pd.read_parquet(args.macro) if args.macro and args.macro.endswith('.parquet') else None
    df = build_feature_table(args.prices, args.oi, args.funding, news_path=args.news, macro_df=macro_df)
    df.to_parquet(args.out)
    print('Saved features ->', args.out)
