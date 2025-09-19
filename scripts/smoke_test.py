"""
Лёгкий smoke-тест проекта: проверки импортов и базовых точек входа.
НЕ запускает обучение и НЕ требует готовых фичей/моделей.
Запуск:  python -m scripts.smoke_test
Выход: код 0 если всё ок (с WARN допускаются), иначе 1.
"""
import sys, importlib, traceback

OK = 0
def ok(msg):  print(f"[OK]   {msg}")
def warn(msg):print(f"[WARN] {msg}")
def err(msg):
    global OK
    OK = 1
    print(f"[ERR]  {msg}")

def try_import(name, attrs=None):
    try:
        m = importlib.import_module(name)
        ok(f"import {name}")
        if attrs:
            for a in attrs:
                if not hasattr(m, a):
                    warn(f"{name}: missing attr '{a}' (может быть не обязательно)")
                else:
                    ok(f"{name}.{a} exists")
        return m
    except Exception:
        err(f"import {name} failed")
        traceback.print_exc(limit=1)
        return None

def main():
    # --- config ---
    cfg = try_import("config", attrs=[
        "BASE_DIR","DATA_DIR","FEATURES_DIR","MODELS_DIR",
        "BAR","FEAT_FILE","PX_FILE"
    ])

    # --- utils / logging ---
    try_import("utils.logging_why", attrs=["log_dataframe"])

    # --- ETL ---
    try_import("etl.prices", attrs=["load_btc_prices"])
    try_import("etl.bybit", attrs=["load_bybit_derivs"])
    try_import("etl.news", attrs=["load_news_counts"])
    try_import("etl.blockchain", attrs=["load_blockchain_metrics"])
    try_import("etl.fred", attrs=["load_fred_metrics"])

    # --- features ---
    try_import("features.build_features", attrs=["build_feature_table"])

    # --- classical models ---
    try_import("models.train_quantile", attrs=["train_quantiles"])
    try_import("models.classify", attrs=["train_binary_heads"])
    try_import("models.model_zoo", attrs=["train_quantiles_backend"])
    try_import("models.classify_xgb", attrs=["train_binary_heads_xgb"])
    try_import("models.xgb_quantile", attrs=["train_quantiles_xgb_dart"])

    # --- seq models ---
    try_import("models.nn_seq", attrs=["train_model","predict_now","GRUQuantile"])
    try_import("models.tcn_seq", attrs=["train_tcn","predict_now_tcn","TCNQuantile"])
    try_import("models.seq_batch_infer", attrs=["batch_predict_seq_median"])

    # --- post-processing / ensemble / calibration ---
    try_import("models.ensemble", attrs=["ensemble_predict","predict_pack"])
    try_import("models.ensemble_optimizer", attrs=["optimize_weights_static","optimize_weights_with_seq"])
    try_import("models.gp_calibration", attrs=["GPCalibrator"])

    # --- scripts entrypoints (как модули) ---
    try:
        import scripts.make_features as _mf
        ok("scripts.make_features import")
    except Exception:
        err("scripts.make_features import failed")
        traceback.print_exc(limit=1)

    try:
        import scripts.train_all as _ta
        ok("scripts.train_all import")
    except Exception:
        err("scripts.train_all import failed")
        traceback.print_exc(limit=1)

    # --- whylogs наличие (не запускаем логирование) ---
    try:
        import whylogs as _w
        ok("whylogs present")
    except Exception:
        warn("whylogs not installed (не критично для обучения, но для логирования понадобится)")

    # --- финал ---
    if OK == 0:
        print("\n=== SMOKE TEST: PASSED ===")
    else:
        print("\n=== SMOKE TEST: FAILED (см. ERR выше) ===")
    sys.exit(OK)

if __name__ == "__main__":
    main()