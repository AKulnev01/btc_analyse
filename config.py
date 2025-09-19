# config.py
from pathlib import Path
import os

# ── .env (опционально) ──────────────────────────────────────────────────
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# ── БАЗОВЫЕ ПУТИ ────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
INTERIM_DIR   = DATA_DIR / "interim"
FEATURES_DIR  = DATA_DIR / "features"
BACKTESTS_DIR = DATA_DIR / "backtests"
MODELS_DIR    = BASE_DIR / "models"
EXT_DIR       = DATA_DIR / "external"          # новые внешние кэши (on-chain / funding / sentiment)

# ── ТФ / ТАЙМИНГ ────────────────────────────────────────────────────────
# BAR: "1H" (час), "30T" (полчаса), "1T" (минута)
BAR = os.getenv("BAR", "30T").upper()
BARS_PER_HOUR = {"1H": 1, "30T": 2, "1T": 60}[BAR]
BASE_TF_MINUTES = {"1H": 60, "30T": 30, "1T": 1}[BAR]

# Основной горизонт для бизнес-задач (часы)
TARGET_HOURS = int(os.getenv("TARGET_HOURS", "24"))
# Мульти-таргеты параллельно (часы)
MULTI_TARGET_HOURS = [1, 4, 24]
TARGET_BARS = TARGET_HOURS * BARS_PER_HOUR

# ── ФАЙЛЫ (основные) ────────────────────────────────────────────────────
PX_FILE    = str(INTERIM_DIR / "btc_prices.parquet")
OI_FILE    = str(INTERIM_DIR / "bybit_oi.parquet")            # опционально (если используешь)
FUND_FILE  = str(INTERIM_DIR / "bybit_funding.parquet")       # историческое; для новых фич см. FUNDING_FILE ниже
NEWS_FILE  = str(INTERIM_DIR / "news_counts.parquet")         # если у тебя есть локальный кэш новостей
FEAT_FILE  = str(FEATURES_DIR / f"btc_{BASE_TF_MINUTES}m_features.parquet")
GLOBAL_META_FILE = str(INTERIM_DIR / "global_meta.parquet")

# ── ФАЙЛЫ (внешние источники/новые кэши) ────────────────────────────────
ONCHAIN_FILE  = str(EXT_DIR / "onchain_btc.parquet")          # Glassnode bundle
FUNDING_FILE  = str(EXT_DIR / "funding_all.parquet")          # Binance/Bybit/OKX funding + агрегаты
BASIS_FILE    = str(EXT_DIR / "basis_all.parquet")            # опц.: перп-премия vs spot (если добавим)
SENTI_FILE    = str(EXT_DIR / "news_sentiment.parquet")       # агрегаты/полярность новостей
MACRO_FILE = "./data/interim/macro.parquet"
# ── КВАНТИЛИ / МОДЕЛИ ───────────────────────────────────────────────────
QUANTILES = [0.1, 0.5, 0.9]

# ── ДИНАМИЧЕСКАЯ ШИРИНА ИНТЕРВАЛОВ (Conformal / Adaptive λ) ─────────────
LAMBDA_BASE       = 0.85   # базовая ширина
LAMBDA_VOL_ALPHA  = 0.25   # реакция на рост волатильности
LAMBDA_NEWS_ALPHA = 0.15   # реакция на «горячие» новости (shock/surprise)

# ── NEWS SHOCK (инжиниринг новостного импульса) ─────────────────────────
NEWS_LAG_MIN       = 5     # задержка, пока новость «доходит» до рынка (мин)
NEWS_DECAY_MIN     = 90    # эксп. распад импульса (мин)
NEWS_MAX_PER_HOUR  = 20    # кап на счётчик в час (нормализация)

# ── ФИЧЕ-ФЛАГИ ──────────────────────────────────────────────────────────
FEATURE_FLAGS = {
    "USE_DERIVS": True,
    "USE_DIVERGENCE": True,
    "USE_NEWS": True,
    "USE_MACRO": False,          # можно включить позже
    "USE_GLOBAL_META": True,
    "USE_MICROSTRUCTURE": True,  # на 1m-барах для базового TF
    "USE_ONCHAIN": True,         # <— новые
    "USE_FUNDING": True,         # <— новые
    "USE_NEWS_SENTI": True,      # <— новые
}

# ── ВЕСА/ПОРОГИ ДЛЯ КЛАССИФ. ГОЛОВ ──────────────────────────────────────
VOL_WEIGHT_ALPHA = 1.0  # 0.0 чтобы выключить веса по волатильности
UP_THR_PCT = {1: 0.002, 4: 0.005, 24: 0.02}

# ── API КЛЮЧИ / ПРОКСИ (из окружения/.env) ─────────────────────────────
COINGECKO_API_KEY   = os.getenv("COINGECKO_API_KEY", "")
CRYPTOPANIC_KEY     = os.getenv("CRYPTOPANIC_KEY", "")
GNEWS_KEY           = os.getenv("GNEWS_KEY", "")
FRED_API_KEY        = os.getenv("FRED_API_KEY", "")

# Trading Economics (если используешь)
TE_CLIENT           = os.getenv("TE_CLIENT", "")
TE_SECRET           = os.getenv("TE_SECRET", "")
TE_API_KEY          = f"{TE_CLIENT}:{TE_SECRET}" if TE_CLIENT and TE_SECRET else ""

# Bybit (если нужен приватный REST)
BYBIT_API_KEY       = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET    = os.getenv("BYBIT_API_SECRET", "")
FEATURE_FLAGS["USE_DERIVS"] = False  # включим, когда появится bybit_oi.parquet

# Glassnode / CryptoQuant для ончейн
GLASSNODE_API_KEY   = os.getenv("GLASSNODE_API_KEY", "")
CRYPTOQUANT_API_KEY = os.getenv("CRYPTOQUANT_API_KEY", "")   # опционально

# Прокси (если нужно)
HTTP_PROXY          = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY         = os.getenv("HTTPS_PROXY", "")

# ── ДЕФОЛТЫ ДЛЯ УСТРОЙСТВА (не обязательно) ────────────────────────────
# Можно переопределить через переменную окружения DEVICE: "cuda"|"mps"|"cpu"
DEVICE_DEFAULT = os.getenv("DEVICE", "cpu").lower()