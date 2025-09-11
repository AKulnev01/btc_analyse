# etl/news_sources.py
import os, math, requests, pandas as pd

def _to_hour(ts):
    return pd.to_datetime(ts, utc=True).floor("h")

def fetch_cryptopanic_counts(hours=72):
    key = os.getenv("CRYPTOPANIC_KEY", "")
    if not key:
        return pd.DataFrame()
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {"auth_token": key, "kind": "news", "public": "true"}
    rows = []
    for page in range(1, 6):  # 5 страниц хватит для последних ~100–150 постов
        r = requests.get(url, params={**params, "page": page}, timeout=20).json()
        for it in r.get("results", []):
            ts = _to_hour(it["published_at"])
            cats = it.get("currencies") or []
            is_btc = any(c.get("code","").lower()=="btc" for c in cats)
            # sentiment: it.get("votes") -> {"positive":..,"negative":..}
            votes = it.get("votes") or {}
            rows.append({"timestamp": ts,
                         "is_btc": int(is_btc),
                         "pos": votes.get("positive",0),
                         "neg": votes.get("negative",0)})
        if not r.get("next"):
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("timestamp")
    agg = df.resample("1h").agg(
        news_total=("is_btc","size"),
        news_btc=("is_btc","sum"),
        news_pos=("pos","sum"),
        news_neg=("neg","sum"),
    )
    return agg

def fetch_gnews_counts(query="bitcoin", hours=72):
    key = os.getenv("GNEWS_KEY", "")
    if not key:
        return pd.DataFrame()
    url = "https://gnews.io/api/v4/search"
    rows = []
    # gnews = 100/стр; пробегаем последние несколько страниц по времени
    for page in range(1, 6):
        r = requests.get(url, params={
            "q": query, "lang":"en", "max": 100, "token": key, "page": page,
            "sortby":"publishedAt"
        }, timeout=20).json()
        for a in r.get("articles", []):
            ts = _to_hour(a["publishedAt"])
            rows.append({"timestamp": ts})
        if len(r.get("articles",[])) < 100:
            break
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
              .set_index("timestamp")
              .assign(dummy=1)
              .resample("1h").sum()
              .rename(columns={"dummy":"gnews_"+query.replace(" ","_")}))