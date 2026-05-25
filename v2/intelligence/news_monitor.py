"""
intelligence/news_monitor.py — Financial RSS news monitor for TradingBotV2.

Improvements over V1:
- Per-feed 8-second timeout (one feed can never block others).
- Circuit breaker: a feed is skipped for 10 minutes after two consecutive
  timeouts.
- FinBERT (ProsusAI/finbert) for local sentiment inference — no Claude API.
- Model cached in memory after first load.
- Graceful fallback to sentiment_score=0.0 when transformers is not installed.
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSS feed list
# ---------------------------------------------------------------------------

RSS_FEEDS: list[str] = [
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://www.investing.com/rss/news.rss",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.marketwatch.com/rss/topstories",
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "https://www.ft.com/?format=rss",
]

# ---------------------------------------------------------------------------
# Circuit breaker state  (per-feed)
# ---------------------------------------------------------------------------
# Maps feed URL -> {"consecutive_timeouts": int, "skip_until": float}
_CIRCUIT: dict[str, dict] = {}

_CIRCUIT_TIMEOUT_THRESHOLD = 2        # consecutive timeouts before opening
_CIRCUIT_COOLDOWN_SECONDS  = 600      # 10 minutes


def _is_circuit_open(url: str) -> bool:
    state = _CIRCUIT.get(url)
    if state is None:
        return False
    if state["consecutive_timeouts"] >= _CIRCUIT_THRESHOLD:
        if time.monotonic() < state["skip_until"]:
            return True
        # Cooldown expired — reset
        _CIRCUIT[url] = {"consecutive_timeouts": 0, "skip_until": 0.0}
    return False


def _record_timeout(url: str) -> None:
    state = _CIRCUIT.setdefault(url, {"consecutive_timeouts": 0, "skip_until": 0.0})
    state["consecutive_timeouts"] += 1
    if state["consecutive_timeouts"] >= _CIRCUIT_THRESHOLD:
        state["skip_until"] = time.monotonic() + _CIRCUIT_COOLDOWN_SECONDS
        logger.warning(
            "Circuit open for %s after %d consecutive timeouts — skipping for %ds",
            url,
            state["consecutive_timeouts"],
            _CIRCUIT_COOLDOWN_SECONDS,
        )


def _record_success(url: str) -> None:
    if url in _CIRCUIT:
        _CIRCUIT[url]["consecutive_timeouts"] = 0


# Private name referenced from _is_circuit_open / _record_timeout
_CIRCUIT_THRESHOLD = _CIRCUIT_TIMEOUT_THRESHOLD

# ---------------------------------------------------------------------------
# FinBERT sentiment (lazy-loaded, cached)
# ---------------------------------------------------------------------------

_finbert_pipeline = None          # cached pipeline object
_finbert_available: Optional[bool] = None  # None = not yet checked


def _get_finbert():
    """
    Return a loaded transformers sentiment-analysis pipeline for FinBERT, or
    None if the transformers library is not installed.

    The model is loaded once and reused for every subsequent call.
    """
    global _finbert_pipeline, _finbert_available

    if _finbert_available is False:
        return None
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    try:
        from transformers import pipeline  # type: ignore

        logger.info("Loading FinBERT model (ProsusAI/finbert) — first call only")
        _finbert_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,          # return all three labels
        )
        _finbert_available = True
        logger.info("FinBERT model loaded and cached")
        return _finbert_pipeline

    except ImportError:
        logger.warning("transformers not installed — sentiment will default to 0.0")
        _finbert_available = False
        return None
    except Exception as exc:
        logger.error("Failed to load FinBERT: %s", exc)
        _finbert_available = False
        return None


def _score_text(text: str) -> float:
    """
    Run FinBERT on *text* and return a score in [-1.0, +1.0].

    Score = (positive_conf - negative_conf)

    Returns 0.0 on any failure or if transformers is unavailable.
    """
    if not text:
        return 0.0

    pipe = _get_finbert()
    if pipe is None:
        return 0.0

    try:
        # Truncate to 512 chars — BERT has a 512-token limit
        truncated = text[:512]
        results = pipe(truncated)

        # results is a list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
        # When top_k=None the pipeline returns a list-of-lists for batch input.
        if results and isinstance(results[0], list):
            results = results[0]

        scores: dict[str, float] = {r["label"].lower(): r["score"] for r in results}
        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)
        return round(positive - negative, 4)

    except Exception as exc:
        logger.debug("FinBERT inference error: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# RSS helpers
# ---------------------------------------------------------------------------

_RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

_DATE_FORMATS = [
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S GMT",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
]


def _parse_rss_date(date_str: str) -> Optional[datetime]:
    """Return a UTC-aware datetime from an RSS date string, or None."""
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _source_label(url: str) -> str:
    try:
        host = urlparse(url).hostname or url
        return host.removeprefix("www.").removeprefix("feeds.")
    except Exception:
        return url


def _fetch_single_feed(url: str) -> list[dict]:
    """
    Fetch and parse one RSS feed.  Raises requests.Timeout on timeout so the
    caller can update the circuit breaker state.

    Returns a list of raw dicts with keys: title, url, published, source_text.
    """
    source = _source_label(url)
    items: list[dict] = []

    response = requests.get(url, headers=_RSS_HEADERS, timeout=8)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    ns   = {"atom": "http://www.w3.org/2005/Atom"}

    rss_nodes  = root.findall(".//item")
    atom_nodes = root.findall(".//atom:entry", ns)
    nodes      = rss_nodes if rss_nodes else atom_nodes

    for node in nodes:
        def _text(tag: str, namespace: str = "") -> str:
            el = node.find(f"{namespace}{tag}")
            return (el.text or "").strip() if el is not None else ""

        title     = _text("title")
        link      = _text("link")
        pub_raw   = _text("pubDate") or _text("updated") or _text("published")

        # Atom <link href="..."> has no text content
        if not link:
            link_el = node.find("atom:link", ns)
            if link_el is not None:
                link = link_el.get("href", "")

        if not title:
            continue

        published_dt = _parse_rss_date(pub_raw) if pub_raw else None

        items.append({
            "title":        title,
            "url":          link,
            "published":    published_dt,
            "source":       source,
        })

    return items


# ---------------------------------------------------------------------------
# NewsMonitor
# ---------------------------------------------------------------------------

class NewsMonitor:
    """
    Fetches financial news from multiple RSS feeds concurrently, scores each
    headline with FinBERT, and exposes query methods for downstream use.
    """

    def __init__(self, feeds: list[str] | None = None, max_workers: int = 8) -> None:
        self._feeds      = feeds or RSS_FEEDS
        self._max_workers = max_workers
        # Cache of fetched + scored articles: list[dict]
        self._cache:      list[dict] = []
        self._cache_until: float     = 0.0  # monotonic timestamp
        self._cache_ttl:   float     = 300.0  # 5 minutes

    # ------------------------------------------------------------------
    # Internal: fetch all feeds with per-feed timeout + circuit breaker
    # ------------------------------------------------------------------

    def _fetch_all(self) -> list[dict]:
        """
        Fetch all configured RSS feeds concurrently.

        Each feed has an 8-second network timeout enforced inside
        _fetch_single_feed().  If a feed raises Timeout twice in a row
        the circuit opens and the feed is skipped for 10 minutes.

        Returns a flat list of raw article dicts (unsorted, unscored).
        """
        results: list[dict] = []

        active_feeds = [f for f in self._feeds if not _is_circuit_open(f)]
        skipped      = len(self._feeds) - len(active_feeds)
        if skipped:
            logger.debug("Circuit open for %d feed(s) — skipped", skipped)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_map = {pool.submit(_fetch_single_feed, url): url for url in active_feeds}

            for future in as_completed(future_map):
                url = future_map[future]
                try:
                    items = future.result()
                    _record_success(url)
                    results.extend(items)
                    logger.debug("Feed %s → %d items", _source_label(url), len(items))

                except requests.Timeout:
                    _record_timeout(url)
                    logger.warning("Timeout fetching %s", url)

                except requests.RequestException as exc:
                    logger.warning("Request error for %s: %s", url, exc)

                except ET.ParseError as exc:
                    logger.warning("XML parse error for %s: %s", url, exc)

                except Exception as exc:
                    logger.error("Unexpected error fetching %s: %s", url, exc)

        return results

    # ------------------------------------------------------------------
    # Internal: score articles with FinBERT
    # ------------------------------------------------------------------

    @staticmethod
    def _score_articles(raw: list[dict]) -> list[dict]:
        """Score each article with FinBERT and return the enriched list."""
        scored = []
        for item in raw:
            sentiment = _score_text(item["title"])
            scored.append({
                "title":           item["title"],
                "source":          item["source"],
                "published":       item["published"],
                "sentiment_score": sentiment,
                "url":             item["url"],
            })
        return scored

    # ------------------------------------------------------------------
    # Internal: refresh cache when stale
    # ------------------------------------------------------------------

    def _refresh_if_stale(self) -> None:
        if time.monotonic() < self._cache_until:
            return
        raw            = self._fetch_all()
        self._cache    = self._score_articles(raw)
        self._cache_until = time.monotonic() + self._cache_ttl
        logger.info("News cache refreshed: %d articles", len(self._cache))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_latest_headlines(self, max_age_hours: float = 4.0) -> list[dict]:
        """
        Return all headlines published within *max_age_hours*.

        Each item contains:
            title           – headline string
            source          – feed hostname label
            published       – UTC-aware datetime (or None)
            sentiment_score – float in [-1.0, +1.0]
            url             – link to full article
        """
        self._refresh_if_stale()

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        filtered = [
            a for a in self._cache
            if a["published"] is None or a["published"] >= cutoff
        ]

        return sorted(
            filtered,
            key=lambda a: a["published"] or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

    def get_sentiment_for(self, symbol: str) -> dict:
        """
        Aggregate sentiment for *symbol* from recent headlines that mention it.

        Returns:
            score          – average sentiment in [-1.0, +1.0]
            headline_count – number of matching headlines
            top_headline   – text of highest-|score| headline (or "")
            bias           – "bullish" | "bearish" | "neutral"
        """
        symbol_lower = symbol.lower()
        headlines    = self.get_latest_headlines()

        matching = [
            h for h in headlines
            if symbol_lower in h["title"].lower()
        ]

        if not matching:
            return {
                "score":          0.0,
                "headline_count": 0,
                "top_headline":   "",
                "bias":           "neutral",
            }

        avg_score   = sum(h["sentiment_score"] for h in matching) / len(matching)
        top         = max(matching, key=lambda h: abs(h["sentiment_score"]))
        avg_score   = round(avg_score, 4)

        if avg_score > 0.15:
            bias = "bullish"
        elif avg_score < -0.15:
            bias = "bearish"
        else:
            bias = "neutral"

        return {
            "score":          avg_score,
            "headline_count": len(matching),
            "top_headline":   top["title"],
            "bias":           bias,
        }

    def is_market_risk_on(self) -> bool:
        """
        Return True when aggregate sentiment across all recent headlines > 0.2,
        indicating broadly risk-on conditions.
        """
        headlines = self.get_latest_headlines()
        if not headlines:
            return False

        avg = sum(h["sentiment_score"] for h in headlines) / len(headlines)
        return avg > 0.2
