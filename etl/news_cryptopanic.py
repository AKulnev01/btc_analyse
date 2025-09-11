# etl/news_cryptopanic.py
import os
import time
import math
import json
import logging
from typing import Optional, List, Dict

import pandas as pd
import requests

# ====== конфиг / ключ ======
CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_KEY", "")  # положи ключ в .env и экспортни его
BASE_URL = "https://cryptopanic.com/api/developer/v2"

# ====== логгер ======
log = logging.getLogger("news_cp")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def _to_utc_hour(ts) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True).floor("H")

def _one_page(params: Dict) -> Dict:
    """Вызов одного page у /posts/ с таймаутом и мягкой обработкой."""
    url = f"{BASE_URL}/posts/"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_posts(hours: int = 168,
                currencies: Optional[str] = None,
                kinds: Optional[str] = None,
                regions: Optional[str] = None,
                max_pages: int = 200) -> pd.DataFrame:
    """
    Тянем новости за последние `hours` часов.
    Параметры CryptoPanic:
      - currencies: строка через запятую (например: "BTC,ETH")
      - kinds: "news,media" и т.п.
      - regions: "en,ru,..." (если нужно)
    """
    if not CRYPTOPANIC_KEY:
        raise RuntimeError("ENV CRYPTOPANIC_KEY не задан")

    cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=hours)
    params = {
        "auth_token": CRYPTOPANIC_KEY,
        "page": 1,
    }
    if currencies:
        params["currencies"] = currencies
    if kinds:
        params["kind"] = kinds
    if regions:
        params["regions"] = regions

    posts = []
    page = 1
    while page <= max_pages:
        params["page"] = page
        try:
            data = _one_page(params)
        except requests.HTTPError as e:
            log.warning(f"CryptoPanic HTTP error on page={page}: {e}")
            break
        except Exception as e:
            log.warning(f"CryptoPanic error on page={page}: {e}")
            break

        # API v2 обычно кладёт посты в "results"
        items = data.get("results") or data.get("data") or []
        if not items:
            break

        for it in items:
            # published_at может называться по-разному в v2 — пробуем варианты
            ts = it.get("published_at") or it.get("created_at") or it.get("timestamp")
            if not ts:
                continue
            ts = pd.to_datetime(ts, utc=True, errors="coerce")
            if ts is None or pd.isna(ts):
                continue
            if ts < cutoff:
                # достигли отсечки
                page = max_pages + 1
                break

            # Вытаскиваем сигналы:
            # - currencies: список словарей [{code: "BTC", title: "Bitcoin"}...]
            cur_list = it.get("currencies") or []
            codes = [c.get("code") for c in cur_list if isinstance(c, dict) and c.get("code")]
            codes = [str(c).upper() for c in codes]

            # - votes/metrics (например: important / liked и др.) — используем как proxy важности
            votes = it.get("votes") or it.get("metrics") or {}
            important = 0
            try:
                # в старом API был votes['important']; в v2 может быть другое поле
                important = int(votes.get("important", 0))
            except Exception:
                important = 0

            title = it.get("title") or it.get("title_raw") or ""

            posts.append({
                "timestamp": ts,
                "title": title,
                "codes": ",".join(codes) if codes else "",
                "important": important,
            })

        # пагинация (v2 может вернуть "next" url)
        next_url = data.get("next")
        if not next_url:
            # попробуем листать page++
            page += 1
        else:
            # если вдруг выдали прямую ссылку — подменим
            page += 1
        time.sleep(0.2)  # слегка бережём API

    if not posts:
        return pd.DataFrame(columns=["timestamp", "title", "codes", "important"])

    df = pd.DataFrame(posts)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    return df

def aggregate_hourly(df_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегируем в часовые счётчики:
      - news_cnt_total
      - news_cnt_btc  (если в codes упоминается BTC или в title есть 'bitcoin')
      - news_cnt_imp  (сумма important)
    """
    if df_posts.empty:
        idx = pd.date_range(pd.Timestamp.utcnow().floor("H") - pd.Timedelta("7D"),
                            periods=7*24+1, freq="H", tz="UTC")
        return pd.DataFrame(index=idx, columns=["news_cnt_total","news_cnt_btc","news_cnt_imp"]).fillna(0)

    d = df_posts.copy()
    d["hour"] = d["timestamp"].dt.floor("H")

    # BTC-маркер
    has_btc = d["codes"].str.contains(r"\bBTC\b", na=False) | d["title"].str.contains("bitcoin", case=False, na=False)
    grp = d.groupby("hour")

    out = pd.DataFrame({
        "news_cnt_total": grp.size(),
        "news_cnt_btc": grp.apply(lambda g: int(has_btc.loc[g.index].sum())),
        "news_cnt_imp": grp["important"].sum(),
    })
    out.index.name = "timestamp"
    # на всякий — приведём к int
    out["news_cnt_total"] = out["news_cnt_total"].astype("int64")
    out["news_cnt_btc"] = out["news_cnt_btc"].astype("int64")
    out["news_cnt_imp"] = out["news_cnt_imp"].astype("int64")
    return out

def fetch_and_save_parquet(hours: int = 168,
                           out_path: str = "data/interim/news_cryptopanic.parquet",
                           currencies: Optional[str] = None,
                           kinds: Optional[str] = None,
                           regions: Optional[str] = None) -> pd.DataFrame:
    """
    Главная функция: качаем сырые посты, аггрегируем в часовые счётчики, сохраняем parquet.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df_posts = fetch_posts(hours=hours, currencies=currencies, kinds=kinds, regions=regions)
    agg = aggregate_hourly(df_posts)

    # сделаем пару производных (скользящие окна), полезно для трейна
    for w in (6, 24):
        agg[f"news_tot_{w}h"] = agg["news_cnt_total"].rolling(w, min_periods=1).sum()
        agg[f"news_btc_{w}h"] = agg["news_cnt_btc"].rolling(w, min_periods=1).sum()
        agg[f"news_imp_{w}h"] = agg["news_cnt_imp"].rolling(w, min_periods=1).sum()

    agg.to_parquet(out_path)
    log.info(f"[OK] saved CryptoPanic hourly -> {out_path} rows={len(agg)}")
    return agg

if __name__ == "__main__":
    # пример: последние 7 дней, все новости, английский регион
    fetch_and_save_parquet(hours=7*24, regions="en")