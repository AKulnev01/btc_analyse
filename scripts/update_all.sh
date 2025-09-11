#!/usr/bin/env bash
set -e
export PYTHONPATH=.

# 1) рынок
python scripts/fetch_bybit_klines.py --symbol BTCUSDT --category linear --interval 1h --days 1200
python etl/market_meta.py

# 2) новости/сентимент
python etl/news.py
python etl/fear_greed.py

# 3) макро
python etl/macro_fetch.py

# 4) фичи + тренировка/скоринг
python scripts/make_features.py
python scripts/train_all.py --predict