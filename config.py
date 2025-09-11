from pathlib import Path
import os

# опционально: подхватывать .env автоматически
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# ── пути и общие настройки ───────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
INTERIM_DIR   = DATA_DIR / "interim"
FEATURES_DIR  = DATA_DIR / "features"
BACKTESTS_DIR = DATA_DIR / "backtests"
MODELS_DIR    = BASE_DIR / "models"

PX_FILE    = str(INTERIM_DIR / "btc_prices.parquet")
OI_FILE    = str(INTERIM_DIR / "bybit_oi.parquet")
FUND_FILE  = str(INTERIM_DIR / "bybit_funding.parquet")
NEWS_FILE  = str(INTERIM_DIR / "news_counts.parquet")
FEAT_FILE  = str(FEATURES_DIR / "btc_1h_features.parquet")

TARGET_HOURS = 168
QUANTILES    = [0.1, 0.5, 0.9]

# динамическая ширина (пример базовых коэфов — можешь тюнить)
LAMBDA_BASE        = 0.85
LAMBDA_VOL_ALPHA   = 0.25   # реакция на рост волы
LAMBDA_NEWS_ALPHA  = 0.15   # реакция на «горячие» новости

# ── API ключи (из окружения / .env) ─────────────────────────────────────
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
GLOBAL_META_FILE = "./data/interim/global_meta.parquet"

CRYPTOPANIC_KEY   = os.getenv("CRYPTOPANIC_KEY", "")
GNEWS_KEY         = os.getenv("GNEWS_KEY", "")
FRED_API_KEY      = os.getenv("FRED_API_KEY", "")

TE_CLIENT         = os.getenv("TE_CLIENT", "")
TE_SECRET         = os.getenv("TE_SECRET", "")
TE_API_KEY        = f"{TE_CLIENT}:{TE_SECRET}" if TE_CLIENT and TE_SECRET else ""

BYBIT_API_KEY     = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET  = os.getenv("BYBIT_API_SECRET", "")

HTTP_PROXY        = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY       = os.getenv("HTTPS_PROXY", "")

TARGET_HOURS = 4
FEAT_FILE = "./data/features/btc_1h_features.parquet"
NEWS_FILE = "./data/interim/news_counts.parquet"
