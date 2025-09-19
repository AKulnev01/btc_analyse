# models/ensemble_weight_opt.py
from __future__ import annotations
import json
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


@dataclass
class WeightFitResult:
    weights: List[float]
    model_names: List[str]
    val_mae_log: float
    n_val: int


# ---------- helpers to read packs & batch-predict ----------
def _load_pack(pack_path: str) -> Dict:
    with open(pack_path, "rb") as f:
        pack = pickle.load(f)
    return pack

def _batch_pred_pack_yq50(df_va: pd.DataFrame, pack: Dict) -> np.ndarray:
    models = pack["models"]
    feature_cols = pack["feature_cols"] if "feature_cols" in pack else pack.get("features", None)
    if feature_cols is None:
        raise ValueError("Pack missing 'feature_cols'/'features'.")
    scale_col = pack["scale_col"]
    quantiles: List[float] = pack["quantiles"]
    if 0.5 not in quantiles:
        raise ValueError("Pack quantiles must include 0.5 (median).")

    has_h = isinstance(models, dict) and models and isinstance(next(iter(models.values())), dict)
    if has_h:
        any_h = sorted(models.keys())[0]
        m50 = models[any_h][0.5]
    else:
        m50 = models[0.5]

    X = df_va[feature_cols].astype(float).values
    sigma = df_va[scale_col].astype(float).clip(1e-8).values
    yhat50_scaled = m50.predict(X)
    yhat50 = yhat50_scaled * sigma
    return np.asarray(yhat50, dtype=float)

def _train_val_split(df: pd.DataFrame, scale_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.dropna(subset=["y", scale_col]).copy()
    n = len(work)
    split = int(n * 0.8)
    return work.iloc[:split], work.iloc[split:]


# ---------- seq batch predict ----------
def _batch_pred_seq_yq50(df_va: pd.DataFrame, df_full: pd.DataFrame, seq_cfg: Dict) -> np.ndarray:
    """
    Получаем yhat50 для seq-модели по валидации.
    Важно: seq-модель требует L последних шагов для каждой точки, поэтому
    сначала считаем по всему df_full (скользящим окном), затем выравниваем
    по индексу валидации.
    seq_cfg: {"backend": "gru"|"tcn", "model": "...pt", "meta": "...pkl", "device": "cpu"}
    """
    from models.seq_batch_infer import batch_predict_seq_median

    yhat_full = batch_predict_seq_median(
        df=df_full,
        backend=seq_cfg["backend"],
        model_path=seq_cfg["model"],
        meta_path=seq_cfg["meta"],
        device=seq_cfg.get("device", "cpu"),
        batch_size=int(seq_cfg.get("batch_size", 512)),
    )
    # пересекаем индексы с df_va
    idx = df_va.index.intersection(yhat_full.index)
    if len(idx) == 0:
        raise ValueError("No overlapping indices between validation slice and seq predictions.")
    return yhat_full.loc[idx].values.astype(float)


# ---------- core: fit weights ----------
def _mae_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def _optimize_weights_slsqp(y_true: np.ndarray, Ymat: np.ndarray) -> np.ndarray:
    n_models = Ymat.shape[1]
    w0 = np.full(n_models, 1.0 / n_models, dtype=float)

    def obj(w):
        yhat = Ymat @ w
        return np.mean(np.abs(y_true - yhat))

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n_models

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000})
    if not res.success:
        w = np.clip(res.x, 0.0, None)
        s = w.sum()
        return w / s if s > 0 else w0
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else w0

def _optimize_weights_grid(y_true: np.ndarray, Ymat: np.ndarray) -> np.ndarray:
    n_models = Ymat.shape[1]
    if n_models == 1:
        return np.array([1.0], dtype=float)

    best_mae = 1e18
    best_w = None

    if n_models == 2:
        grid = np.linspace(0.0, 1.0, 51)
        for a in grid:
            w = np.array([a, 1.0 - a])
            mae = _mae_log(y_true, Ymat @ w)
            if mae < best_mae:
                best_mae, best_w = mae, w
    elif n_models == 3:
        grid = np.linspace(0.0, 1.0, 21)
        for a in grid:
            for b in grid:
                c = 1.0 - a - b
                if c < 0 or c > 1:
                    continue
                w = np.array([a, b, c])
                mae = _mae_log(y_true, Ymat @ w)
                if mae < best_mae:
                    best_mae, best_w = mae, w
    else:
        rng = np.random.default_rng(42)
        for _ in range(20000):
            x = rng.random(n_models)
            w = x / x.sum()
            mae = _mae_log(y_true, Ymat @ w)
            if mae < best_mae:
                best_mae, best_w = mae, w

    return np.array(best_w, dtype=float)


def fit_ensemble_weights(
    df: pd.DataFrame,
    classical_packs: List[str],
    save_path: Optional[str] = None,
    model_names: Optional[List[str]] = None,
    seq_models: Optional[List[Dict]] = None,  # добавлено: [{"backend":"gru","model":"...pt","meta":"...pkl","device":"cuda"}, ...]
) -> WeightFitResult:
    """
    Подбирает веса ансамбля на валидации (последние 20%) среди:
      - классических паков (обязательно хотя бы один),
      - опционально seq-моделей (GRU/TCN) через батч-инференс.

    Минимизируем MAE по медиане (лог-доходности).
    """
    if not classical_packs and not seq_models:
        raise ValueError("Provide at least one classical pack or seq model for weight fitting.")

    # scale_col возьмём из первого доступного пака, иначе — из df
    if classical_packs:
        first_pack = _load_pack(classical_packs[0])
        scale_col = first_pack["scale_col"]
    else:
        # если нет паков — попробуем угадать
        for c in ("sigma_ewma_7d", "sigma_7d"):
            if c in df.columns:
                scale_col = c
                break
        else:
            raise ValueError("Cannot determine scale_col: provide at least one classical pack.")

    df_tr, df_va = _train_val_split(df, scale_col)
    y_true = df_va["y"].astype(float).values

    # формируем Ymat и имена
    Ycols = []
    names = []

    # классические паки
    for i, path in enumerate(classical_packs or []):
        pack = _load_pack(path)
        yhat50 = _batch_pred_pack_yq50(df_va, pack)
        Ycols.append(np.asarray(yhat50, dtype=float))
        names.append(model_names[i] if model_names and i < len(model_names) else f"pack_{i+1}")

    # seq-модели
    if seq_models:
        for j, scfg in enumerate(seq_models):
            yhat50_seq = _batch_pred_seq_yq50(df_va, df, scfg)
            # Важно: df_va может иметь индекс шире, чем у seq-предиктов (первые L-1 точек выбывают),
            # поэтому согласуем длины по пересечению индексов в _batch_pred_seq_yq50.
            # Тут Ycols уже построены на df_va; чтобы сопоставить, срезаем y_true/Ycols по пересечению позже.
            Ycols.append(np.asarray(yhat50_seq, dtype=float))
            names.append(scfg.get("name", f"seq_{scfg['backend']}_{j+1}"))

        # если есть и паки, и seq — нужно обрезать до общего индекса
        # Самый простой способ — пересчитать всё по пересечению индексов.
        # Но мы уже получили массивы без индексов. Проще пересчитать df_va до idx пересечения первых yhat50_seq.
        # Сделаем это один раз: построим общий idx = пересечение всех источников.
        # Для простоты: пересечение классических = весь df_va (они не теряют точки).
        # Пересечение с seq: берём длину по seq (они уже срезаны по пересечению). Тогда усечём y_true и Ycols классических.
        if classical_packs:
            # длина у последовательных предиктов равна длине df_va ∩ валид индексов seq.
            # Чтобы честно обрезать, пересчитаем по хвосту len_min:
            len_min = min(map(len, Ycols))
            # усечём все столбцы и y_true с конца, чтобы выровнять
            Ycols = [col[-len_min:] for col in Ycols]
            y_true = y_true[-len_min:]

    if not Ycols:
        raise ValueError("No prediction sources collected for weight fitting.")

    Ymat = np.column_stack(Ycols)  # shape (n_val_eff, n_models)

    # оптимизация
    if _HAS_SCIPY:
        w = _optimize_weights_slsqp(y_true, Ymat)
    else:
        w = _optimize_weights_grid(y_true, Ymat)

    val_mae = _mae_log(y_true, Ymat @ w)

    result = WeightFitResult(weights=list(map(float, w)), model_names=names, val_mae_log=float(val_mae), n_val=int(len(y_true)))

    if save_path:
        with open(save_path, "w") as f:
            json.dump({
                "weights": result.weights,
                "model_names": result.model_names,
                "val_mae_log": result.val_mae_log,
                "n_val": result.n_val,
                "packs": classical_packs,
                "seq_models": seq_models or [],
            }, f, indent=2)

    return result


# ---------- apply weights to "now" ----------
def predict_now_with_weights(
    df: pd.DataFrame,
    classical_packs: List[str],
    weights: List[float],
) -> Dict[str, float]:
    from models.ensemble import predict_pack
    assert len(classical_packs) == len(weights), "weights length must match number of packs"
    preds = []
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    for p in classical_packs:
        preds.append(predict_pack(df, p))

    now_price = preds[0]["now_price"]
    P10 = float(np.sum([w[i] * preds[i]["P10"] for i in range(len(preds))]))
    P50 = float(np.sum([w[i] * preds[i]["P50"] for i in range(len(preds))]))
    P90 = float(np.sum([w[i] * preds[i]["P90"] for i in range(len(preds))]))
    P10, P90 = float(min(P10, P50)), float(max(P90, P50))
    return {"now_price": now_price, "P10": P10, "P50": P50, "P90": P90}

# ---------- CLI ----------
def _parse_list(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fit ensemble weights over validation (last 20%).")
    ap.add_argument("--features", required=True, help="path to features parquet")
    ap.add_argument("--packs", default="", help="comma-separated list of classical pack .pkl paths")
    ap.add_argument("--names", default="", help="comma-separated list of model names (optional, for packs)")
    ap.add_argument("--seq", default="", help="JSON string with list of seq models, e.g. '[{\"backend\":\"gru\",\"model\":\"models/nn_seq.pt\",\"meta\":\"models/nn_seq_meta.pkl\",\"device\":\"cuda\",\"name\":\"GRU\"}]'")
    ap.add_argument("--save", default="models/ensemble_weights.json", help="where to save weights json")
    args = ap.parse_args()

    df = pd.read_parquet(args.features).sort_index()
    packs = _parse_list(args.packs) if args.packs else []
    names = _parse_list(args.names) if args.names else None
    seq_models = json.loads(args.seq) if args.seq else None

    res = fit_ensemble_weights(df, packs, save_path=args.save, model_names=names, seq_models=seq_models)
    print(f"[OK] fitted weights: {res.weights}  (val MAE_log={res.val_mae_log:.6f}, n_val={res.n_val})")
    print(f"[SAVED] {args.save}")

if __name__ == "__main__":
    main()