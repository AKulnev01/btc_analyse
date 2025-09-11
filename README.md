# BTC Weekly Forecast — MVP
Ежечасный прогноз цены BTC на +7 дней: квантили (P10/P50/P90) + вероятности сценариев.

## Быстрый старт
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# подготовь промежуточные данные в data/interim/:
#  - btc_prices.parquet: index=timestamp(UTC), cols: open, high, low, close, volume
#  - bybit_oi.parquet: index=timestamp(UTC), col: open_interest
#  - bybit_funding.parquet: index=timestamp(UTC), col: funding_rate
#  - news_counts.parquet: index=timestamp(UTC), агрегаты новостей по часу
#  - macro_flags.parquet: index=timestamp(UTC), cols: macro_fomc_win, macro_cpi_win, macro_nfp_win

python scripts/make_features.py
python scripts/train_all.py
python scripts/predict_now.py
