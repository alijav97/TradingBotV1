"""
intelligence/tweet_monitor.py — Nitter RSS feed monitor for TradingBotV2.

Monitors key financial/political Twitter accounts via nitter proxy instances.
Parses each tweet for financial keywords and assigns a market impact score.
Results are cached for 5 minutes to avoid hammering public nitter instances.
All nitter instance failures are handled gracefully — never crashes the bot.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRACKED_ACCOUNTS: list[str] = [
    "realDonaldTrump",
    "elonmusk",
    "federalreserve",
    "GoldmanSachs",
]

NITTER_INSTANCES: list[str] = [
    "nitter.net",
    "nitter.privacydev.net",
    "nitter.cz",
]

# Keyword → impact score in [0.0, 1.0].
# Longer/more-specific phrases must come before their sub-strings so the
# regex alternation matches the most specific phrase first.
KEYWORD_IMPACT: dict[str, float] = {
    # Monetary policy — very high impact
    "emergency rate cut":  1.0,
    "rate hike":           0.9,
    "rate cut":            0.9,
    "quantitative easing": 0.9,
    "quantitative tightening": 0.9,
    "federal reserve":     0.85,
    "fomc":                0.85,
    "interest rate":       0.85,
    "powell":              0.8,
    # Macro / economic data
    "inflation":           0.8,
    "cpi":                 0.75,
    "nfp":                 0.75,
    "non-farm payroll":    0.75,
    "gdp":                 0.7,
    "recession":           0.85,
    "debt ceiling":        0.8,
    "default":             0.8,
    "credit downgrade":    0.8,
    # Geopolitical
    "sanctions":           0.8,
    "war":                 0.8,
    "nuclear":             0.9,
    "airstrike":           0.8,
    "ceasefire":           0.75,
    "peace deal":          0.7,
    "tariff":              0.75,
    "trade war":           0.8,
    # Market / asset specific
    "gold":                0.6,
    "oil":                 0.65,
    "opec":                0.75,
    "bitcoin":             0.65,
    "crypto":              0.6,
    "dollar":              0.6,
    "treasury":            0.65,
    "yields":              0.65,
    "stock market":        0.65,
    "wall street":         0.6,
    "nasdaq":              0.6,
    "s&p":                 0.6,
    # Corporate / finance
    "bailout":             0.8,
    "bankruptcy":          0.8,
    "layoffs":             0.65,
    "earnings":            0.55,
    "merger":              0.55,
    "acquisition":         0.55,
    # General financial signals
    "market crash":        0.95,
    "black swan":          0.95,
    "bank run":            0.9,
    "liquidity crisis":    0.9,
    "pandemic":            0.85,
    "crisis":              0.75,
}

# Pre-compile a single regex for efficiency: longest phrases first (already
# ordered in the dict above for multi-word entries).
_KEYWORD_PATTERN = re.compile(
    r"(" + "|".join(re.escape(kw) for kw in KEYWORD_IMPACT) + r")",
    re.IGNORECASE,
)

# Regex to extract ticker-like symbols (e.g. $AAPL, $BTC, $SPX)
_SYMBOL_PATTERN = re.compile(r"\$([A-Z]{1,6})", re.IGNORECASE)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_HEADERS = {
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


def _parse_date(date_str: str) -> Optional[datetime]:
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _compute_impact(text: str) -> tuple[float, list[str]]:
    """
    Return (impact_score, matched_keywords) for *text*.

    impact_score is the maximum single-keyword score found in the text.
    If no keyword matches, returns (0.0, []).
    """
    matches = _KEYWORD_PATTERN.findall(text)
    if not matches:
        return 0.0, []

    unique  = list({m.lower() for m in matches})
    scores  = [KEYWORD_IMPACT.get(kw, 0.0) for kw in unique]
    return max(scores), unique


def _extract_symbols(text: str) -> list[str]:
    """Return any $TICKER symbols mentioned in *text*, uppercased and deduplicated."""
    found = _SYMBOL_PATTERN.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for sym in found:
        up = sym.upper()
        if up not in seen:
            seen.add(up)
            result.append(up)
    return result


# ---------------------------------------------------------------------------
# Feed fetching with nitter instance fallback
# ---------------------------------------------------------------------------

def _fetch_nitter_rss(username: str) -> list[dict]:
    """
    Try each nitter instance in order and return the parsed tweet list for
    *username*.  Returns [] if every instance fails.
    """
    last_exc: Optional[Exception] = None

    for instance in NITTER_INSTANCES:
        url = f"https://{instance}/{username}/rss"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()

            root = ET.fromstring(resp.content)
            items: list[dict] = []

            for node in root.findall(".//item"):
                def _text(tag: str) -> str:
                    el = node.find(tag)
                    return (el.text or "").strip() if el is not None else ""

                title      = _text("title")
                pub_raw    = _text("pubDate")
                link       = _text("link")

                # Nitter sometimes puts the full tweet body in <description>
                description = _text("description")
                tweet_text  = description if description else title

                published = _parse_date(pub_raw) if pub_raw else None

                impact, keywords = _compute_impact(tweet_text)
                symbols          = _extract_symbols(tweet_text)

                items.append({
                    "username":         username,
                    "text":             tweet_text,
                    "published":        published,
                    "impact_score":     impact,
                    "symbols_mentioned": symbols,
                    "_matched_keywords": keywords,  # internal, stripped before return
                    "url":              link,
                })

            logger.debug(
                "Fetched %d tweets for @%s from %s", len(items), username, instance
            )
            return items

        except requests.Timeout:
            logger.warning("Timeout on nitter instance %s for @%s", instance, username)
            last_exc = None  # timeout is expected, try next

        except requests.RequestException as exc:
            logger.warning(
                "Request error on nitter instance %s for @%s: %s", instance, username, exc
            )
            last_exc = exc

        except ET.ParseError as exc:
            logger.warning(
                "XML parse error from nitter instance %s for @%s: %s", instance, username, exc
            )
            last_exc = exc

    if last_exc:
        logger.error("All nitter instances failed for @%s: %s", username, last_exc)
    else:
        logger.warning("All nitter instances timed out for @%s", username)

    return []


# ---------------------------------------------------------------------------
# TweetMonitor
# ---------------------------------------------------------------------------

class TweetMonitor:
    """
    Monitors key Twitter/X accounts via nitter RSS feeds.

    All accounts are fetched concurrently.  Results are cached for 5 minutes.
    If every nitter instance is unreachable the methods return [] without
    raising any exception.
    """

    CACHE_TTL_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        accounts: list[str] | None = None,
        max_workers: int = 4,
    ) -> None:
        self._accounts   = accounts or TRACKED_ACCOUNTS
        self._max_workers = max_workers
        self._cache:       list[dict] = []
        self._cache_until: float      = 0.0  # monotonic timestamp

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_if_stale(self) -> None:
        if time.monotonic() < self._cache_until:
            return

        raw: list[dict] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(_fetch_nitter_rss, acct): acct
                for acct in self._accounts
            }
            for future in as_completed(futures):
                acct = futures[future]
                try:
                    items = future.result()
                    raw.extend(items)
                except Exception as exc:
                    # _fetch_nitter_rss already handles all expected errors;
                    # this guard is a last resort to keep the bot alive.
                    logger.error(
                        "Unhandled error fetching tweets for @%s: %s", acct, exc
                    )

        # Strip internal keys, store clean dicts
        self._cache = [
            {k: v for k, v in item.items() if not k.startswith("_")}
            for item in raw
        ]
        self._cache_until = time.monotonic() + self.CACHE_TTL_SECONDS
        logger.info("Tweet cache refreshed: %d tweets", len(self._cache))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_recent_tweets(self, hours: float = 2.0) -> list[dict]:
        """
        Return tweets published within the last *hours* hours.

        Each item contains:
            username          – Twitter/X handle (no @)
            text              – tweet body
            published         – UTC-aware datetime (or None)
            impact_score      – float in [0.0, 1.0]
            symbols_mentioned – list of $TICKER strings found
            url               – link to the tweet
        """
        self._refresh_if_stale()

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        recent = [
            t for t in self._cache
            if t["published"] is None or t["published"] >= cutoff
        ]

        return sorted(
            recent,
            key=lambda t: t["published"] or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

    def get_market_moving_tweets(self, threshold: float = 0.7) -> list[dict]:
        """
        Return recent tweets whose impact_score >= *threshold*.

        Uses the same 2-hour window as get_recent_tweets().
        """
        return [t for t in self.get_recent_tweets() if t["impact_score"] >= threshold]
