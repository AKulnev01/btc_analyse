# utils/logging_why.py
"""
Whylogs-интеграция для логгирования данных и предсказаний.
Лог сохраняется в ./logs/whylogs/YYYYMMDD/...
"""
from __future__ import annotations
import os
import pandas as pd
import whylogs as why
from datetime import datetime

def log_dataframe(df, name: str = "features", outdir: str = "./logs/whylogs"):
    """
    Логируем датафрейм в whylogs local writer.
    Совместимо с whylogs>=1.4 (используем base_dir вместо path).
    """
    try:
        outdir = os.path.join(outdir, datetime.utcnow().strftime("%Y%m%d"))
        os.makedirs(outdir, exist_ok=True)
        profile = why.log(df)
        # ⬇️ ключевая строчка — передаём base_dir один раз при создании writer
        writer = profile.writer("local", base_dir=outdir)
        writer.write(base_name=name)  # больше не передаём path/base_dir здесь
        return True
    except Exception as e:
        print(f"[whylogs][WARN] logging skipped: {e}")
        return False

def log_predictions(preds: dict, name: str = "predictions") -> None:
    """
    Логирует словарь прогнозов (например, {"P10":..., "P50":..., "P90":...})
    """
    if not preds:
        print(f"[whylogs] skip empty preds: {name}")
        return

    df = pd.DataFrame([preds])
    log_dataframe(df, name=name)