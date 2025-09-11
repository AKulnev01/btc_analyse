import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# чтобы работали config и общий код
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    import config as CFG
    HORIZON_H: int = getattr(CFG, "TARGET_HOURS", 168)
    DEFAULT_FEAT: str = getattr(CFG, "FEAT_FILE", "./data/features/btc_1h_features.parquet")
    QUANTILES: List[float] = getattr(CFG, "QUANTILES", [0.1, 0.5, 0.9])
except Exception:
    HORIZON_H = 168
    DEFAULT_FEAT = "./data/features/btc_1h_features.parquet"
    QUANTILES = [0.1, 0.5, 0.9]

SCALE_COLS = ["sigma_ewma_7d", "sigma_7d"]  # предпочтение EWMA


def _pick_scale_col(df: pd.DataFrame) -> str:
    for c in SCALE_COLS:
        if c in df.columns:
            return c
    raise ValueError("Не найдено ни sigma_ewma_7d, ни sigma_7d — добавь в build_features.py")


def _feature_cols(df: pd.DataFrame) -> List[str]:
    bad = set(["y", "close"])
    return [c for c in df.columns if c not in bad and df[c].dtype.kind != "O"]


def _train_quantile_block(
    df_tr: pd.DataFrame,
    feats_all: List[str],
    q_list: List[float],
    coverage_target: float = 0.70,
    min_cov: float = 0.90,
) -> Tuple[Dict[float, LGBMRegressor], Tuple[Dict[str, float], float, float, Dict[str, float]], str, List[str]]:
    """
    Обучает LGBM-квантили на масштабе y/sigma, калибрует λ по бакетам σ,
    плюс split-conformal полу-ширины в σ-единицах по бакетам.
    Возвращает: models, (lam_map, q1, q2, qmap), scale_col, feats_used
    """
    scale_col = _pick_scale_col(df_tr)
    s = df_tr[scale_col].clip(lower=1e-8).astype(float)
    y_scaled = (df_tr["y"] / s).astype(float)

    # фильтр по покрытию фичей
    cov = df_tr[feats_all].notna().mean()
    feats_used = [c for c in feats_all if float(cov.get(c, 0.0)) >= min_cov]
    if len(feats_used) == 0:
        raise ValueError("После фильтра покрытий не осталось фичей для обучения")

    X_all = df_tr[feats_used].astype(float)
    mask_rows = X_all.notna().all(axis=1) & y_scaled.notna()
    X = X_all.loc[mask_rows]
    y_scaled = y_scaled.loc[mask_rows]
    s = s.loc[mask_rows]

    if len(X) < 500:
        raise ValueError("Слишком мало строк после фильтрации для обучения")

    # простая валидация: последний 20%
    split = int(len(X) * 0.8)
    Xtr, Xva = X.iloc[:split], X.iloc[split:]
    ytr, yva = y_scaled.iloc[:split], y_scaled.iloc[split:]
    s_va = s.iloc[split:]

    # LGBM — параметры, чтобы избежать “No further splits”
    params = dict(
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=255,
        max_depth=-1,
        min_data_in_leaf=25,
        min_sum_hessian_in_leaf=1e-5,
        feature_fraction=0.95,
        subsample=0.9,
        subsample_freq=1,
        reg_lambda=3.0,
        min_split_gain=0.0,
        max_bin=1023,
        force_row_wise=True,
        deterministic=True,
        seed=42,
        bagging_seed=42,
        feature_fraction_seed=42,
        verbosity=-1,
        extra_trees=True,
        path_smooth=5.0,
    )

    models: Dict[float, LGBMRegressor] = {}
    for q in q_list:
        m = LGBMRegressor(objective="quantile", alpha=q, **params)
        m.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="l1",
            callbacks=[early_stopping(stopping_rounds=200), log_evaluation(0)],
        )
        models[q] = m

    # базовые квантили на валидации (в σ-единицах -> затем вернёмся в y)
    lam_map: Dict[str, float] = {"LOW": 1.0, "MID": 1.0, "HIGH": 1.0}
    qmap: Dict[str, float] = {"LOW": 0.0, "MID": 0.0, "HIGH": 0.0}
    q1 = q2 = float("nan")

    try:
        preds_scaled = {q: models[q].predict(Xva) for q in q_list}
        has_all = all(q in preds_scaled for q in (0.1, 0.5, 0.9))
        if has_all and len(Xva) > 100:
            q10_s = preds_scaled[0.1]
            q50_s = preds_scaled[0.5]
            q90_s = preds_scaled[0.9]

            # вернёмся в y (домножаем на σ валидации)
            q10 = q10_s * s_va.values
            q50 = q50_s * s_va.values
            q90 = q90_s * s_va.values

            # y_true на тех же индексах
            y_true = df_tr.loc[X.index].iloc[split:]["y"].values

            # бакеты по σ на валидации
            q1, q2 = np.quantile(s_va.values, [1.0 / 3, 2.0 / 3])
            def vb(sig: float) -> str:
                return "LOW" if sig <= q1 else ("MID" if sig <= q2 else "HIGH")

            grid = np.linspace(0.5, 1.0, 51)
            s_va_arr = s_va.values

            # подбор λ по каждому бакету
            for b in ("LOW", "MID", "HIGH"):
                mask_b = np.array([vb(float(z)) == b for z in s_va_arr])
                if mask_b.sum() < 100:
                    lam_map[b] = 1.0
                    continue
                q10_b, q50_b, q90_b = q10[mask_b], q50[mask_b], q90[mask_b]
                y_b = y_true[mask_b]
                best_lam, best_diff = 1.0, 1e9
                for lam in grid:
                    lo = q50_b + lam * (q10_b - q50_b)
                    hi = q50_b + lam * (q90_b - q50_b)
                    cov = float(np.mean((y_b >= lo) & (y_b <= hi)))
                    diff = abs(cov - coverage_target)
                    if diff < best_diff:
                        best_lam, best_diff = float(lam), float(diff)
                lam_map[b] = best_lam

            # split-conformal: нормы остатков к медиане в σ-единицах на хвосте трейна (90 дней)
            tail_start = X.index.max() - pd.Timedelta(days=90)
            cal_mask = X.index >= tail_start
            if (0.5 in models) and (cal_mask.sum() > 500):
                med_cal = models[0.5].predict(X.loc[cal_mask])
                y_tail = df_tr.loc[X.index[cal_mask], "y"].astype(float)
                s_tail = s.loc[X.index[cal_mask]].astype(float)
                resid = np.abs((y_tail / s_tail) - med_cal)

                # бакеты по σ на хвосте
                q1_cal, q2_cal = np.quantile(s_tail.values, [1.0 / 3, 2.0 / 3])
                def vb_tail(sig: float) -> str:
                    return "LOW" if sig <= q1_cal else ("MID" if sig <= q2_cal else "HIGH")

                for b in ("LOW", "MID", "HIGH"):
                    r_b = resid[[vb_tail(float(z)) == b for z in s_tail.values]]
                    if len(r_b) > 50:
                        qmap[b] = float(np.quantile(r_b, coverage_target))
                    else:
                        qmap[b] = float(np.quantile(resid, coverage_target))
            else:
                qmap = {"LOW": 0.0, "MID": 0.0, "HIGH": 0.0}

    except Exception:
        pass

    print(
        f"[retrain] feats_used={len(feats_used)}/{len(feats_all)} "
        f"rows={len(X)} | lam_map={lam_map} | q_conf={qmap}"
    )
    return models, (lam_map, float(q1), float(q2), qmap), scale_col, feats_used


def _predict_one(
    df_row: pd.Series,
    feats_used: List[str],
    models: Dict[float, LGBMRegressor],
    scale_col: str,
    lam_pack: Tuple[Dict[str, float], float, float, Dict[str, float]],
) -> Tuple[float, float, float]:
    lam_map, q1, q2, qmap = lam_pack

    # подаём в LGBM DataFrame с именованными фичами (иначе warning)
    x = df_row[feats_used].to_frame().T.astype(float)
    now_price = float(df_row["close"])
    sigma = float(max(float(df_row.get(scale_col, np.nan)), 1e-8))

    # бакет σ
    if (not np.isfinite(q1)) or (not np.isfinite(q2)):
        b = "MID"
    else:
        b = "LOW" if sigma <= q1 else ("MID" if sigma <= q2 else "HIGH")

    # предсказываем scaled y*, возвращаемся в y (домножаем на σ)
    yq: Dict[float, float] = {}
    for q, m in models.items():
        yq[q] = float(m.predict(x)[0]) * sigma

    # non-crossing: сортируем значения
    qs = sorted(yq.keys())
    vals = np.array([yq[q] for q in qs], dtype=float)
    vals.sort()
    yq = {qs[i]: float(vals[i]) for i in range(len(qs))}

    # центр и первичная полу-ширина
    med = float(yq.get(0.5, 0.0))
    w0 = max(
        float(yq.get(0.9, med)) - med,
        med - float(yq.get(0.1, med)),
    )

    # shrink/expand по λ(σ-бакет)
    lam = float(lam_map.get(b, 1.0))
    y_lo_q = med - lam * (med - float(yq.get(0.1, med)))
    y_hi_q = med + lam * (float(yq.get(0.9, med)) - med)

    # split-conformal: минимальная полу-ширина на бакет
    w_conf = float(qmap.get(b, 0.0)) * sigma
    w = max(w_conf, (y_hi_q - y_lo_q) / 2.0, w0)

    # «шок-защита»: если сильный внутринедельный/4h ход, мягко расширим
    ret_4h = float(df_row.get("ret_4h", 0.0)) if "ret_4h" in df_row else 0.0
    ret_1d = float(df_row.get("ret_1d", 0.0)) if "ret_1d" in df_row else 0.0
    extreme = (abs(ret_4h) > 0.03) or (abs(ret_1d) > 0.06)
    if extreme:
        w *= 1.25

    # финальные уровни
    y_lo = med - w
    y_hi = med + w
    P10 = now_price * np.exp(y_lo)
    P50 = now_price * np.exp(med)
    P90 = now_price * np.exp(y_hi)
    return P10, P50, P90


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Walk-forward бэктест за последний год (ежечасные предикты на +HORIZON_H)."
    )
    ap.add_argument("--features", default=DEFAULT_FEAT, help="parquet фичей")
    ap.add_argument("--start-days", type=int, default=365, help="длина окна бэктеста в днях")
    ap.add_argument("--retrain-hours", type=int, default=168, help="как часто переобучаем модели")
    ap.add_argument("--coverage", type=float, default=0.70, help="целевое покрытие для калибровки [P10,P90]")
    ap.add_argument("--out-csv", default="./data/backtests/btc_last_year.csv", help="куда сохранить покадровые результаты")
    ap.add_argument("--hit-usd", type=float, default=200.0, help="порог попадания относительно P50, $")
    ap.add_argument("--hit-pct", type=float, default=0.005, help="порог попадания относительно P50, % (0.005 = 0.5%)")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.read_parquet(args.features).sort_index()
    feats_all = _feature_cols(df)
    scale_col = _pick_scale_col(df)

    # ограничим бэктест последним годом с запасом на горизонт
    t_end_eval = df.index.max() - pd.Timedelta(hours=HORIZON_H)
    t_start_eval = t_end_eval - pd.Timedelta(days=args.start_days)

    results: List[Dict[str, float]] = []
    last_train_time: Optional[pd.Timestamp] = None
    models: Optional[Dict[float, LGBMRegressor]] = None
    lam_pack: Tuple[Dict[str, float], float, float, Dict[str, float]] = (
        {"LOW": 1.0, "MID": 1.0, "HIGH": 1.0},
        float("nan"),
        float("nan"),
        {"LOW": 0.0, "MID": 0.0, "HIGH": 0.0},
    )
    feats_used: List[str] = feats_all[:]  # обновится после тренировки
    scale_col_used: str = scale_col

    iter_ts = df.loc[(df.index >= t_start_eval) & (df.index <= t_end_eval)].index
    for ts in iter_ts:
        need_retrain = (
            models is None
            or last_train_time is None
            or ((ts - last_train_time) >= pd.Timedelta(hours=args.retrain_hours))
        )
        if need_retrain:
            train_cutoff = ts - pd.Timedelta(hours=1)  # на момент t доступно всё до t-1h
            train_mask = df.index <= (train_cutoff - pd.Timedelta(hours=HORIZON_H))
            df_tr = df.loc[train_mask].dropna(subset=["y", scale_col])
            if len(df_tr) < 2000:
                last_train_time = ts
                continue
            try:
                models, lam_pack, scale_col_used, feats_used = _train_quantile_block(
                    df_tr, feats_all, QUANTILES, coverage_target=args.coverage, min_cov=0.90
                )
                last_train_time = ts
            except Exception as e:
                print(f"[WARN] retrain failed at {ts}: {e}")
                last_train_time = ts
                continue

        row = df.loc[ts]
        # пропускаем, если не хватает масштаба или фичей
        if pd.isna(row.get(scale_col_used, np.nan)):
            continue
        if any(pd.isna(row.get(c, np.nan)) for c in feats_used):
            continue

        P10, P50, P90 = _predict_one(row, feats_used, models, scale_col_used, lam_pack)

        # истинное значение на +HORIZON_H (log-return) и «реальная» цена
        y_true = float(df.loc[ts, "y"])
        price_now = float(row["close"])
        price_true = price_now * np.exp(y_true)

        cover = (price_true >= min(P10, P90)) and (price_true <= max(P10, P90))
        width_pct = 100.0 * (P90 - P10) / price_now if price_now > 0 else np.nan
        abs_err_usd = abs(P50 - price_true)
        abs_err_pct = abs(P50 - price_true) / price_now if price_now > 0 else np.nan
        hit200 = float(abs_err_usd <= args.hit_usd)
        hitpct = float(abs_err_pct <= args.hit_pct)

        results.append(
            {
                "ts": ts,
                "price_now": price_now,
                "price_true": price_true,
                "P10": P10,
                "P50": P50,
                "P90": P90,
                "cover": int(cover),
                "width_pct": width_pct,
                "abs_err_usd": abs_err_usd,
                "abs_err_pct": abs_err_pct,
                "hit_usd": hit200,
                "hit_pct": hitpct,
            }
        )

    if not results:
        print("[WARN] нет результатов (мало данных/фичей?)")
        return

    rp = pd.DataFrame(results).set_index("ts")
    rp.to_csv(args.out_csv, index=True)
    print(f"[OK] saved backtest -> {args.out_csv} (rows={len(rp)})")

    # сводка
    cover_rate = rp["cover"].mean()
    mean_width = rp["width_pct"].mean()
    mae_usd = rp["abs_err_usd"].mean()
    hit_usd = rp["hit_usd"].mean() * 100.0
    hit_pct = rp["hit_pct"].mean() * 100.0
    print(f"Coverage [P10,P90]: {cover_rate*100:.1f}%  |  mean width: {mean_width:.2f}%  |  MAE@P50: ${mae_usd:,.0f}")
    print(f"Hit ±${int(args.hit_usd)}: {hit_usd:.1f}%   |   Hit ±{args.hit_pct*100:.2f}%: {hit_pct:.1f}%")

if __name__ == "__main__":
    main()