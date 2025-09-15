# etl/news_cryptopanic.py
import os
import time
import requests
import pandas as pd
from typing import Optional, List
import config as CFG

BASE = "https://cryptopanic.com/api/developer/v2"
KEY  = getattr(CFG, "CRYPTOPANIC_KEY", os.getenv("CRYPTOPANIC_KEY", ""))

def _get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_posts(
    hours: int = 7*24,
    currencies: Optional[str] = None,   # 'BTC,ETH' и т.п.
    kinds: str = "news",                 # news | media
    regions: str = "en",                 # 'en' и/или др.
    filters: str = "rising,hot,important",
    pages_max: int = 50
) -> pd.DataFrame:
    """
    Тянем новости CryptoPanic за последние `hours` часов, с пагинацией.
    Возвращает сырые посты (одна строка = один пост).
    """
    if not KEY:
        raise RuntimeError("CRYPTOPANIC_KEY не найден ни в config.py, ни в ENV")

    # фиксированное «сейчас» в UTC, без .tz_localize()
    now_utc = pd.Timestamp.now(tz="UTC")
    cutoff  = now_utc - pd.Timedelta(hours=hours)

    params = {
        "auth_token": KEY,
        "public": "true",
        "kind": kinds,
        "regions": regions,
        "filter": filters,
    }
    if currencies:
        params["currencies"] = currencies

    out = []
    cursor = None
    for _ in range(pages_max):
        if cursor:
            params["cursor"] = cursor
        j = _get(f"{BASE}/posts/", params)
        results = j.get("results", [])
        if not results:
            break

        for p in results:
            ts = pd.to_datetime(p.get("published_at"), utc=True)
            if ts < cutoff:
                # дальше только старее — можно прервать весь цикл
                results = []
                break
            # currencies в dev v2 иногда массив объектов; нормализуем
            curr = p.get("currencies", [])
            if isinstance(curr, list):
                # элементы бывают строками или словарями с 'code'
                try:
                    codes = []
                    for c in curr:
                        if isinstance(c, str):
                            codes.append(c)
                        elif isinstance(c, dict):
                            # варианты ключей
                            codes.append(c.get("code") or c.get("symbol") or "")
                    currencies_str = ",".join([c for c in codes if c])
                except Exception:
                    currencies_str = ""
            else:
                currencies_str = str(curr or "")
            out.append({
                "timestamp": ts,
                "title": p.get("title", ""),
                "source": p.get("source", ""),
                "currencies": currencies_str,
                "sentiment": p.get("vote", "") or p.get("sentiment", ""),
                "url": p.get("url", ""),
                "domain": p.get("domain", ""),
            })

        cursor = j.get("next_cursor")
        if not cursor or not results:
            break
        time.sleep(0.25)

    df = pd.DataFrame(out)
    if df.empty:
        return df

    # аккуратный индекс-таймстемп по часам (UTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def to_hourly_counts(df_posts: pd.DataFrame) -> pd.DataFrame:
    """Агрегация до 1H: просто счётчик новостей."""
    if df_posts.empty:
        return pd.DataFrame()
    s = df_posts.set_index("timestamp")["title"].resample("1h").count().rename("news_cnt")
    return s.to_frame()

def fetch_and_save_parquet(
    hours: int = 7*24,
    currencies: Optional[str] = None,
    kinds: str = "news",
    regions: str = "en",
    out_path: Optional[str] = None
) -> str:
    """
    Композит: тянем посты -> в 1H счётчики -> дописываем в parquet (dedup по индексу).
    Возвращаем путь к parquet.
    """
    df_posts = fetch_posts(hours=hours, currencies=currencies, kinds=kinds, regions=regions)
    hourly = to_hourly_counts(df_posts)
    if hourly.empty:
        print("[WARN] CryptoPanic: пусто (ничего не сохранили)")
        return out_path or getattr(CFG, "NEWS_FILE", "data/interim/news_counts.parquet")

    path = out_path or getattr(CFG, "NEWS_FILE", "data/interim/news_counts.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        old = pd.read_parquet(path)
        hourly = pd.concat([old, hourly]).sort_index()
        hourly = hourly[~hourly.index.duplicated(keep="last")]
    except Exception:
        pass
    hourly.to_parquet(path)
    print(f"[OK] news hours: {len(hourly)} -> {path}")
    return path

if __name__ == "__main__":
    # по умолчанию 7 дней англ. новостей, BTC/крипто фокус можно задать currencies="BTC"
    fetch_and_save_parquet(hours=7*24, regions="en")