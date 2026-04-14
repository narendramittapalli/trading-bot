"""
Layer 3a — News ingestion.
Pulls financial and macro headlines from NewsAPI (free tier) and RSS feeds.
Filters for relevance to multi-asset universe and macro themes.
"""

import os
from datetime import datetime, timedelta, timezone

import feedparser
import requests


# Keywords for filtering headlines by macro relevance
MACRO_KEYWORDS = [
    # Rates / bonds
    "interest rate", "fed", "federal reserve", "treasury", "yield", "bond",
    "inflation", "cpi", "ppi", "fomc", "rate hike", "rate cut", "monetary policy",
    "fixed income",
    # Equities
    "stock market", "s&p 500", "sp500", "spy", "nasdaq", "qqq", "dow",
    "equity", "earnings", "gdp", "jobs report", "unemployment", "payroll",
    "tech stocks", "growth stocks", "market rally", "market sell-off",
    "russell 2000", "iwm", "small cap",
    # Crypto
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto",
    "blockchain", "defi", "stablecoin", "sec crypto",
    # Commodities
    "gold", "gld", "commodity", "oil", "crude", "precious metal",
    "silver", "copper", "natural gas",
    # International
    "emerging market", "china", "india", "japan", "europe",
    "efa", "eem", "ewj", "inda",
    # General macro
    "recession", "economic", "trade war", "tariff", "geopolitical",
    "central bank", "ecb", "boj", "pboc",
]


def _flatten_universe(config: dict) -> list[str]:
    """Extract all instrument symbols from hierarchical or flat universe config."""
    universe_cfg = config.get("universe", {})
    if isinstance(universe_cfg, list):
        return universe_cfg
    symbols = []
    for class_cfg in universe_cfg.values():
        if isinstance(class_cfg, dict) and "instruments" in class_cfg:
            symbols.extend(class_cfg["instruments"])
    return symbols


class NewsIngestion:
    """Fetches and filters financial headlines from NewsAPI and RSS feeds."""

    def __init__(self, config: dict):
        self.config = config
        news_cfg = config.get("news", {})
        self.lookback_hours = news_cfg.get("lookback_hours", 48)
        self.sources = news_cfg.get("sources", ["newsapi", "rss"])
        self.rss_feeds = news_cfg.get("rss_feeds", [])
        self.universe = _flatten_universe(config)
        self.newsapi_key = os.environ.get("NEWSAPI_KEY", "")

    def fetch_newsapi(self) -> list[dict]:
        """Fetch headlines from NewsAPI free tier."""
        if not self.newsapi_key:
            print("[NEWS] Warning: NEWSAPI_KEY not set. Skipping NewsAPI.")
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")

        headlines = []
        queries = [
            "stock market",
            "federal reserve economy",
            "gold commodity bonds",
            "bitcoin crypto ethereum",
            "emerging markets international",
        ]

        for query in queries:
            try:
                resp = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "from": cutoff_str,
                        "sortBy": "relevancy",
                        "language": "en",
                        "pageSize": 20,
                        "apiKey": self.newsapi_key,
                    },
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for article in data.get("articles", []):
                        headlines.append({
                            "title": article.get("title", ""),
                            "description": article.get("description", ""),
                            "source": article.get("source", {}).get("name", "NewsAPI"),
                            "published": article.get("publishedAt", ""),
                            "url": article.get("url", ""),
                            "origin": "newsapi",
                        })
                else:
                    print(f"[NEWS] NewsAPI returned {resp.status_code} for query '{query}'")
            except Exception as e:
                print(f"[NEWS] NewsAPI error for query '{query}': {e}")

        return headlines

    def fetch_rss(self) -> list[dict]:
        """Fetch headlines from configured RSS feeds."""
        headlines = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)

        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                feed_name = feed.feed.get("title", feed_url)

                for entry in feed.entries[:30]:  # Limit per feed
                    # Check published date if available
                    published = ""
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                        if pub_dt < cutoff:
                            continue
                        published = pub_dt.isoformat()

                    headlines.append({
                        "title": entry.get("title", ""),
                        "description": entry.get("summary", "")[:300],
                        "source": feed_name,
                        "published": published,
                        "url": entry.get("link", ""),
                        "origin": "rss",
                    })
            except Exception as e:
                print(f"[NEWS] RSS error for {feed_url}: {e}")

        return headlines

    def fetch_all(self) -> list[dict]:
        """Fetch headlines from all configured sources."""
        all_headlines = []

        if "newsapi" in self.sources:
            all_headlines.extend(self.fetch_newsapi())

        if "rss" in self.sources:
            all_headlines.extend(self.fetch_rss())

        # Deduplicate by title (case-insensitive)
        seen = set()
        unique = []
        for h in all_headlines:
            key = (h.get("title") or "").lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(h)

        return unique

    def filter_relevant(self, headlines: list[dict]) -> list[dict]:
        """
        Filter headlines for relevance to the instrument universe and macro themes.
        Checks title and description against keyword list and instrument symbols.
        """
        relevant = []
        for h in headlines:
            text = ((h.get("title") or "") + " " + (h.get("description") or "")).lower()

            # Check for universe symbols
            matched_symbols = [s for s in self.universe if s.lower() in text]

            # Check for macro keywords
            matched_keywords = [k for k in MACRO_KEYWORDS if k in text]

            if matched_symbols or matched_keywords:
                h["matched_symbols"] = matched_symbols
                h["matched_keywords"] = matched_keywords[:5]  # Limit for readability
                relevant.append(h)

        return relevant

    def get_headline_digest(self, max_age_hours: int = None) -> dict:
        """
        Full pipeline: fetch, filter, and format a headline digest.
        Returns a dict with raw count, filtered count, and the digest text.

        max_age_hours: if set, overrides config lookback_hours for this call.
          - Prefetch (9:30 AM): use default 48h for broad context
          - Rebalance (9:35 AM): pass 12 to get only overnight/morning news
        """
        if max_age_hours is not None:
            original = self.lookback_hours
            self.lookback_hours = max_age_hours

        all_headlines = self.fetch_all()
        relevant = self.filter_relevant(all_headlines)

        if max_age_hours is not None:
            self.lookback_hours = original  # restore

        # Format digest for Claude
        digest_lines = []
        for i, h in enumerate(relevant[:30], 1):  # Cap at 30 headlines
            source = h.get("source", "Unknown")
            title = h.get("title", "No title")
            desc = h.get("description", "")[:150]
            symbols = ", ".join(h.get("matched_symbols", []))
            keywords = ", ".join(h.get("matched_keywords", []))

            line = f"{i}. [{source}] {title}"
            if desc:
                line += f"\n   {desc}"
            if symbols:
                line += f"\n   Symbols: {symbols}"
            if keywords:
                line += f"\n   Themes: {keywords}"
            digest_lines.append(line)

        digest_text = "\n\n".join(digest_lines) if digest_lines else "No relevant headlines found."

        return {
            "total_fetched": len(all_headlines),
            "relevant_count": len(relevant),
            "headlines": relevant[:30],
            "digest_text": digest_text,
        }

    def print_digest(self, digest: dict):
        """Pretty-print the headline digest."""
        print("\n" + "=" * 70)
        print(f"NEWS DIGEST — Last {self.lookback_hours} hours")
        print(f"Fetched: {digest['total_fetched']} | Relevant: {digest['relevant_count']}")
        print("=" * 70)
        print(digest["digest_text"][:2000])  # Truncate for console
        print("=" * 70 + "\n")
