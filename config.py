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

BAR = "30T"              # варианты: "1H" (час), "30T" (полчаса), "1T" (минута)
BARS_PER_HOUR = {"1H": 1, "30T": 2, "1T": 60}[BAR]

TARGET_HOURS = 24        # бизнес-горизонт в ЧАСАХ
MULTI_TARGET_HOURS = [1, 4]
BASE_TF_MINUTES = 30
TARGET_BARS  = TARGET_HOURS * BARS_PER_HOUR

# config.py
FEAT_FILE = f"./data/features/btc_{BASE_TF_MINUTES}m_features.parquet"
PX_FILE   = "./data/interim/btc_prices.parquet"
NEWS_FILE = "./data/interim/news_counts.parquet"

# === BAR / TARGET ===
BAR = "30T"              # варианты: "1H" (час), "30T" (полчаса), "1T" (минута)
BARS_PER_HOUR = {"1H": 1, "30T": 2, "1T": 60}[BAR]

# === NEWS SHOCK ===
NEWS_LAG_MIN = 5          # сдвиг новости (минуты) до попадания в рынок
NEWS_DECAY_MIN = 90       # эксп. распад импульса новости (мин)
NEWS_MAX_PER_HOUR = 20    # нормализация счётчика в [0..1] (поджать всплески)

# === ПУТИ (если ещё не заданы) ===
OI_FILE = ""              # если используешь
FUND_FILE = ""            # если используешь

# config.py (добавь)
FEATURE_FLAGS = {
    "USE_DERIVS": True,
    "USE_DIVERGENCE": True,
    "USE_NEWS": True,
    "USE_MACRO": False,      # включим позже
    "USE_GLOBAL_META": True,
    "USE_MICROSTRUCTURE": True,  # 1m → базовый TF
}

VOL_WEIGHT_ALPHA = 1.0  # 0.0 чтобы выключить
UP_THR_PCT = {1: 0.002, 4: 0.005, 24: 0.02}