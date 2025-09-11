"""CLI: вывести текущий прогноз уровня цены и вероятностей."""
import pickle, numpy as np, pandas as pd, config as CFG
from models.score import predict_7d_price

if __name__ == '__main__':
    df = pd.read_parquet(CFG.FEAT_FILE)
    out = predict_7d_price(df)

    # вероятности (если обучены)
    try:
        with open('models/binary_heads.pkl','rb') as f:
            payload = pickle.load(f)
        models, feat = payload['models'], payload['features']
        last = df.iloc[[-1]]
        out['P_up']  = float(models['y_up'].predict_proba(last[feat])[:,1][0])
        out['P_big_move'] = float(models['y_big'].predict_proba(last[feat])[:,1][0])
    except Exception:
        pass

    print(out)
