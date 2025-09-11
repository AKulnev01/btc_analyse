import time
import requests
import pandas as pd
import config as CFG

BASE = "https://cryptopanic.com/api/developer/v2"
KEY  = CFG.CRYPTOPANIC_KEY

def _page(params):
    r = requests.get(f"{BASE}/posts/", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_posts(pages=10, filter_btc=True):
    """Тянем несколько страниц, складываем в единый df."""
    params = {
        "auth_token": KEY,
        "filter": "rising,hot,important",
        "public": "true",
        "kind": "news",   # news | media
        "regions": "en",  # английский
    }
    out = []
    cursor = None
    for i in range(pages):
        if cursor:
            params["cursor"] = cursor
        j = _page(params)
        for p in j.get("results", []):
            ts = pd.to_datetime(p["published_at"], utc=True)
            cats = ",".join(p.get("currencies", [])) if isinstance(p.get("currencies"), list) else ""
            title = p.get("title","")
            source = p.get("source","")
            sentiment = p.get("vote","")  # dev v2: бывает vote/sentiment, нормализуем
            out.append({
                "timestamp": ts,
                "title": title,
                "source": source,
                "currencies": cats,
                "sentiment": sentiment,
                "url": p.get("url",""),
                "domain": p.get("domain",""),
            })
        cursor = j.get("next_cursor")
        if not cursor:
            break
        time.sleep(0.25)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if filter_btc:
        # простая фильтрация BTC по символам в currencies или в тексте
        mask = df["currencies"].str.contains("BTC|XBT|Bitcoin", case=False, na=False) | \
               df["title"].str.contains("Bitcoin|BTC|XBT", case=False, na=False)
        df = df[mask]
    return df.sort_values("timestamp")

def hourly_counts(df_posts: pd.DataFrame) -> pd.DataFrame:
    """Грубая агрегация: почасовое количество новостей + простая «важность» (hot/important, если их отмечать)."""
    if df_posts.empty:
        return pd.DataFrame()
    df = df_posts.copy().set_index("timestamp")
    # счётчики
    cnt = df["title"].resample("1h").count().rename("news_cnt")
    # на будущее можно распарсить sentiment/importance и агрегировать по весам
    out = pd.concat([cnt], axis=1)
    return out

if __name__ == "__main__":
    dfp = fetch_posts(pages=20, filter_btc=True)
    hc = hourly_counts(dfp)
    if not hc.empty:
        path = CFG.NEWS_FILE
        try:
            old = pd.read_parquet(path)
            hc = pd.concat([old, hc]).sort_index()
            hc = hc[~hc.index.duplicated(keep="last")]
        except Exception:
            pass
        hc.to_parquet(path)
        print(f"[OK] news hours: {len(hc)} -> {path}")
    else:
        print("[WARN] no news fetched")