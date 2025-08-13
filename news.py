import feedparser
from datetime import datetime

def fetch_news_rss(ticker: str, limit: int = 10):
    # Yahoo Finance RSS
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    d = feedparser.parse(url)
    items = []
    for e in d.entries[:limit]:
        items.append({
            "title": e.get("title", ""),
            "link": e.get("link", ""),
            "published": e.get("published", ""),
            "summary": e.get("summary", ""),
        })
    return items
