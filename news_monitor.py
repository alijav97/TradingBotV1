"""
news_monitor.py - Economic News Monitor for TradingBotV1

Fetches headlines from free RSS/JSON news feeds, filters them for
trading relevance, tags each item with an asset category, and returns
a structured list ready for downstream analysis or Claude API calls.
"""

import os
import json
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Asset / topic keyword lists ───────────────────────────────────────────────

GOLD_KEYWORDS = [
    "gold", "XAU", "XAUUSD", "bullion",
    "precious metals", "GLD", "safe haven"
]

SILVER_KEYWORDS = [
    "silver", "XAG", "XAGUSD", "palladium",
    "platinum", "metals"
]

OIL_KEYWORDS = [
    "oil", "crude", "WTI", "Brent", "OPEC",
    "petroleum", "energy", "natural gas"
]

INDICES_KEYWORDS = [
    "S&P 500", "SPX", "Nasdaq", "Dow Jones",
    "FTSE", "DAX", "Nikkei", "stock market",
    "Wall Street", "equities", "risk on", "risk off"
]

FOREX_KEYWORDS = [
    "dollar", "USD", "DXY", "euro", "EUR",
    "pound", "GBP", "yen", "JPY", "franc",
    "CHF", "aussie", "AUD", "loonie", "CAD",
    "forex", "currency", "exchange rate"
]

MACRO_KEYWORDS = [
    "fed", "federal reserve", "interest rate",
    "inflation", "CPI", "NFP", "non-farm",
    "FOMC", "Powell", "ECB", "BOE", "BOJ",
    "central bank", "GDP", "recession",
    "treasury", "bonds", "yields"
]

RISK_KEYWORDS = [
    "war", "conflict", "crisis", "sanctions",
    "geopolitical", "China", "Russia",
    "Middle East", "Ukraine", "Taiwan",
    "earthquake", "pandemic", "terror",
    # Extended geopolitical / macro event triggers
    "ceasefire", "peace deal", "OPEC", "Fed",
    "rate decision", "CPI", "NFP",
    "Iran", "Israel", "Gaza",
    "China tariffs", "oil embargo",
    "gulf", "UAE", "Saudi", "Middle East tension",
    "nuclear", "airstrike", "missile",
    "coup", "regime change", "election outcome",
    "debt ceiling", "default", "credit downgrade",
]

CRYPTO_KEYWORDS = [
    "bitcoin", "BTC", "ethereum", "ETH",
    "crypto", "cryptocurrency", "blockchain",
    "stablecoin", "binance", "coinbase"
]

# Ordered mapping used by categorise_item() — first match wins.
# MACRO and RISK are checked last so specific assets take priority.
CATEGORY_MAP: list[tuple[str, list[str]]] = [
    ("GOLD",    GOLD_KEYWORDS),
    ("SILVER",  SILVER_KEYWORDS),
    ("OIL",     OIL_KEYWORDS),
    ("INDICES", INDICES_KEYWORDS),
    ("FOREX",   FOREX_KEYWORDS),
    ("CRYPTO",  CRYPTO_KEYWORDS),
    ("MACRO",   MACRO_KEYWORDS),
    ("RISK",    RISK_KEYWORDS),
]

# ── RSS news feed lists ───────────────────────────────────────────────────────

GENERAL_FINANCIAL_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "https://www.investing.com/rss/news.rss",
    "https://apnews.com/rss",
    "https://www.ft.com/?format=rss",
]

GEOPOLITICAL_FEEDS = [
    "https://rss.app/feeds/Al-Jazeera-English.xml",
    "https://www.arabnews.com/rss.xml",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://apnews.com/rss",
]

GOLD_AND_COMMODITIES_FEEDS = [
    "https://www.kitco.com/rss/kitconews.rss",
    "https://www.fxstreet.com/rss/news",
    "https://www.forexlive.com/feed/news",
    "https://www.investing.com/rss/news_25.rss",
    "https://oilprice.com/rss/main",
]

FOREX_AND_MACRO_FEEDS = [
    "https://www.dailyfx.com/feeds/all",
    "https://www.fxempire.com/api/v1/en/articles/rss",
    "https://www.forexcrunch.com/feed/",
    "https://www.financemagnates.com/feed/",
]

INDICES_AND_STOCKS_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://www.zerohedge.com/fullrss2.xml",
    "https://seekingalpha.com/market_currents.xml",
]

X_TWITTER_FEEDS = [
    "https://nitter.net/KitcoNewsNOW/rss",
    "https://nitter.net/ForexLive/rss",
    "https://nitter.net/zerohedge/rss",
    "https://nitter.net/markets/rss",
    "https://nitter.net/Schuldensuehner/rss",
    "https://nitter.net/federalreserve/rss",
    "https://nitter.net/WSJmarkets/rss",
    "https://nitter.net/ReutersBiz/rss",
    "https://nitter.net/business/rss",
    "https://nitter.net/BloombergMarkets/rss",
]

# All feeds combined — used as the default source list
ALL_FEEDS: list[str] = list(dict.fromkeys(
    GENERAL_FINANCIAL_FEEDS
    + GEOPOLITICAL_FEEDS
    + GOLD_AND_COMMODITIES_FEEDS
    + FOREX_AND_MACRO_FEEDS
    + INDICES_AND_STOCKS_FEEDS
    + X_TWITTER_FEEDS
))


# ── Helper: categorise a single news item ────────────────────────────────────

def categorise_item(title: str, description: str = "") -> str:
    """
    Return the best-matching category label for a headline + description.

    Matching is case-insensitive. The CATEGORY_MAP order ensures specific
    asset classes (Gold, Silver, Oil …) take priority over broad macro/risk.

    Args:
        title:       Headline text.
        description: Optional body / summary text.

    Returns:
        One of: GOLD | SILVER | OIL | INDICES | FOREX | CRYPTO |
                MACRO | RISK | OTHER
    """
    combined = (title + " " + description).lower()

    for category, keywords in CATEGORY_MAP:
        for kw in keywords:
            if kw.lower() in combined:
                return category

    return "OTHER"


# ── Core function: filter_relevant_news ──────────────────────────────────────

def filter_relevant_news(raw_items: list[dict]) -> list[dict]:
    """
    Filter a list of raw news dicts for trading relevance and tag each one.

    A news item is considered relevant if it matches at least one keyword
    from any category (i.e. its category is NOT "OTHER").

    Each returned dict has these guaranteed keys:
        title       – headline string
        description – summary / body (may be empty string)
        link        – URL to full article
        published   – ISO-8601 datetime string (or empty string)
        category    – one of the category labels above
        source      – feed hostname or label

    Args:
        raw_items: List of raw dicts produced by fetch_news().

    Returns:
        Filtered and categorised list, sorted newest-first where dates exist.
    """
    relevant: list[dict] = []

    for item in raw_items:
        title       = item.get("title", "").strip()
        description = item.get("description", "").strip()
        link        = item.get("link", "").strip()
        published   = item.get("published", "").strip()
        source      = item.get("source", "").strip()

        if not title:
            continue

        category = categorise_item(title, description)
        if category == "OTHER":
            continue  # not relevant to any tracked asset

        relevant.append({
            "title":       title,
            "description": description,
            "link":        link,
            "published":   published,
            "category":    category,
            "source":      source,
        })

    # Sort: items with a published date go first (newest first),
    # items without a date go to the end.
    def sort_key(item: dict):
        pub = item.get("published", "")
        return pub if pub else ""

    relevant.sort(key=sort_key, reverse=True)
    return relevant


# ── RSS fetch helper ──────────────────────────────────────────────────────────

def _parse_rss_date(date_str: str) -> str:
    """
    Try to normalise an RSS pubDate string to ISO-8601.
    Returns the original string unchanged if parsing fails.
    """
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    return date_str


def fetch_rss_feed(url: str) -> list[dict]:
    """
    Fetch and parse a single RSS 2.0 feed URL.

    Args:
        url: RSS feed URL.

    Returns:
        List of raw news dicts (title, description, link, published, source).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    items: list[dict] = []

    # Derive a short source label from the URL hostname
    try:
        source = url.split("/")[2].replace("www.", "").replace("feeds.", "")
    except IndexError:
        source = url

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        # Handle both RSS 2.0 (<channel><item>) and Atom (<entry>) formats
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        rss_items  = root.findall(".//item")
        atom_items = root.findall(".//atom:entry", ns)
        nodes      = rss_items if rss_items else atom_items

        for node in nodes:
            def _text(tag: str, namespace: str = "") -> str:
                el = node.find(f"{namespace}{tag}")
                return (el.text or "").strip() if el is not None else ""

            # RSS 2.0 fields
            title       = _text("title")
            description = _text("description") or _text("summary")
            link        = _text("link")
            published   = _text("pubDate") or _text("updated") or _text("published")

            # Atom: <link href="..."> has no text content
            if not link:
                link_el = node.find("atom:link", ns)
                if link_el is not None:
                    link = link_el.get("href", "")

            if published:
                published = _parse_rss_date(published)

            items.append({
                "title":       title,
                "description": description,
                "link":        link,
                "published":   published,
                "source":      source,
            })

    except requests.RequestException:
        pass  # silently skip failed feeds
    except ET.ParseError:
        pass  # silently skip unparseable feeds

    return items


# ── Main public function ──────────────────────────────────────────────────────

def fetch_news(feeds: list[str] | None = None) -> list[dict]:
    """
    Fetch all RSS feeds in parallel and return filtered, categorised news items.

    Uses a ThreadPoolExecutor with up to 10 workers so all feeds are fetched
    concurrently. Any feed that fails is silently skipped.

    Args:
        feeds: List of RSS URLs. Defaults to ALL_FEEDS if not provided.

    Returns:
        Filtered and categorised list of relevant news dicts.
    """
    if feeds is None:
        feeds = ALL_FEEDS

    print(f"  Fetching {len(feeds)} feeds in parallel (max 10 workers)...")

    all_raw: list[dict] = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(fetch_rss_feed, url): url for url in feeds}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                items = future.result()
                if items:
                    print(f"  ✓ {url.split('/')[2]:<40} {len(items):>3} items")
                else:
                    print(f"  ✗ {url.split('/')[2]:<40} (no items / failed)")
                all_raw.extend(items)
            except Exception:
                # Any unexpected exception from the worker is silently skipped
                pass

    relevant = filter_relevant_news(all_raw)
    return relevant


def save_news(items: list[dict], output_path: str = "outputs/news_latest.json") -> None:
    """
    Save the filtered news items to a JSON file in outputs/.

    Args:
        items:       List of categorised news dicts.
        output_path: Destination file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {len(items)} relevant news items → {output_path}")


# ── Market sentiment analysis via Claude ───────────────────────────────────

SENTIMENT_PROMPT = """Based on these market headlines, analyze sentiment for each asset class. Return ONLY this exact JSON format, no other text:
{
  "gold": {
    "sentiment": "bullish/bearish/neutral",
    "confidence": 1,
    "bias": "buy/sell/wait",
    "reason": "one sentence"
  },
  "silver": {
    "sentiment": "bullish/bearish/neutral",
    "confidence": 1,
    "bias": "buy/sell/wait",
    "reason": "one sentence"
  },
  "oil": {
    "sentiment": "bullish/bearish/neutral",
    "confidence": 1,
    "bias": "buy/sell/wait",
    "reason": "one sentence"
  },
  "indices": {
    "sentiment": "bullish/bearish/neutral",
    "confidence": 1,
    "bias": "buy/sell/wait",
    "reason": "one sentence"
  },
  "forex_usd": {
    "sentiment": "bullish/bearish/neutral",
    "confidence": 1,
    "bias": "buy/sell/wait",
    "reason": "one sentence"
  },
  "overall_risk": "low/medium/high",
  "key_event_today": "most important event in one sentence"
}"""


def get_market_sentiment(news_items: list[dict]) -> dict:
    """
    Send the top filtered headlines to Claude and get per-asset sentiment.

    Args:
        news_items: Filtered and categorised news dicts from fetch_news().

    Returns:
        Parsed sentiment dict with keys: gold, silver, oil, indices,
        forex_usd, overall_risk, key_event_today.
        Returns an empty dict on failure.
    """
    if not news_items:
        print("  [WARN] No news items to analyse.")
        return {}

    # Build a compact headline list (cap at 60 to stay within token limits)
    headline_lines = []
    for item in news_items[:60]:
        cat   = item.get("category", "")
        title = item.get("title", "").strip()
        if title:
            headline_lines.append(f"[{cat}] {title}")

    headlines_text = "\n".join(headline_lines)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"{SENTIMENT_PROMPT}\n\nHEADLINES:\n{headlines_text}",
                }
            ],
        )

        raw = message.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError as exc:
        print(f"  [ERROR] Claude returned invalid JSON for sentiment: {exc}")
        return {}
    except Exception as exc:
        print(f"  [ERROR] Sentiment analysis failed: {exc}")
        return {}


def print_news_briefing(sentiment: dict) -> None:
    """
    Print the Market Sentiment Dashboard to stdout.

    Args:
        sentiment: Dict returned by get_market_sentiment().
    """
    # Current time in Gulf Standard Time (UTC+4)
    now_gst = datetime.now(timezone.utc)
    from datetime import timedelta
    gst_offset = timedelta(hours=4)
    now_gst_str = (now_gst + gst_offset).strftime("%Y-%m-%d %H:%M GST")

    print("\n" + "=" * 50)
    print("=== MARKET SENTIMENT DASHBOARD ===")
    print(f"=== Updated: {now_gst_str} ===")
    print("=" * 50)

    ASSET_DISPLAY = [
        ("gold",      "GOLD     "),
        ("silver",    "SILVER   "),
        ("oil",       "OIL      "),
        ("indices",   "INDICES  "),
        ("forex_usd", "USD/FOREX"),
    ]

    # Colour helpers (ANSI — work in PowerShell 7 / Windows Terminal)
    COLORS = {
        "bullish": "\033[32m",  # green
        "bearish": "\033[31m",  # red
        "neutral": "\033[33m",  # yellow
    }
    RESET = "\033[0m"

    for key, label in ASSET_DISPLAY:
        asset = sentiment.get(key, {})
        if not isinstance(asset, dict):
            continue

        raw_sentiment = str(asset.get("sentiment", "neutral")).lower()
        confidence    = asset.get("confidence", "?")
        bias          = str(asset.get("bias", "wait")).upper()
        reason        = asset.get("reason", "")
        color         = COLORS.get(raw_sentiment, "")
        disp_sentiment = raw_sentiment.upper()

        print(f"\n{label} → {color}{disp_sentiment:<8}{RESET} [{confidence}/10] → {bias}")
        if reason:
            # Wrap reason at ~50 chars for readability
            print(f"           Reason: {reason}")

    print()
    risk = str(sentiment.get("overall_risk", "unknown")).upper()
    print(f"OVERALL RISK LEVEL: {risk}")

    key_event = sentiment.get("key_event_today", "None identified")
    print(f"KEY EVENT TODAY: {key_event}")
    print("=" * 50)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from collections import Counter

    print("=" * 60)
    print("TradingBotV1 — News Monitor")
    print("=" * 60)

    news = fetch_news()

    counts = Counter(item["category"] for item in news)
    print(f"\nTotal relevant items : {len(news)}")
    print("\nBy category:")
    for cat, count in sorted(counts.items()):
        print(f"  {cat:<10} : {count}")

    save_news(news)

    print("\nAnalysing sentiment with Claude...")
    sentiment = get_market_sentiment(news)

    if sentiment:
        print_news_briefing(sentiment)
    else:
        print("  [WARN] Could not generate sentiment dashboard.")

    print("\n" + "=" * 60)
    print("News fetch complete. Results saved to outputs/news_latest.json")
    print("=" * 60)
