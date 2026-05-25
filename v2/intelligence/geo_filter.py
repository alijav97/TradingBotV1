"""
intelligence/geo_filter.py — Geopolitical Risk Filter for TradingBotV2
=======================================================================
Scores geopolitical risk from news headlines using a keyword-weighted
scoring system.  The returned dict can be used by the confluence engine
to widen stop-loss ATR multipliers and adjust per-instrument bias.

Risk scoring overview
---------------------
  0.0 – 0.3  calm / normal  → no SL adjustment, neutral bias
  0.3 – 0.5  elevated       → +0.25 SL multiplier, slightly bullish gold
  0.5 – 0.7  high           → +0.50 SL multiplier, bullish gold, bearish equities
  0.7 – 1.0  extreme        → +0.75 SL multiplier, strongly bullish gold

Usage
-----
    from v2.intelligence.geo_filter import get_geo_score

    geo = get_geo_score()
    # {
    #   "score": 0.65,
    #   "sl_multiplier": 0.5,
    #   "notes": "HIGH geo risk — widen SL, favour safe-haven longs",
    #   "instrument_biases": {
    #       "XAUUSD":  "bullish",
    #       "GBPJPY":  "bearish",
    #       "WTI":     "neutral",
    #       "NAS100":  "bearish",
    #       "BTCUSDT": "neutral",
    #       "ETHUSDT": "neutral",
    #   },
    # }

Notes
-----
- If no headline provider is available the function returns a safe fallback
  (score=0.0, neutral biases, sl_multiplier=0.0) and logs a warning.
- Designed to accept pre-fetched headlines for testing; calls the
  v2.intelligence.news_filter fetcher by default.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

_EXTREME_KW: list[str] = [
    "nuclear", "world war", "ww3", "nato invade", "nato strikes",
    "missile strike", "ballistic missile", "chemical weapon", "biological weapon",
    "hypersonic", "carrier strike group deploy",
]

_HIGH_KW: list[str] = [
    "war", "invasion", "military offensive", "airstrikes", "airstrike",
    "ground offensive", "troops deploy", "troops cross", "troops advance",
    "conflict escalat", "escalat", "ceasefire collapses", "ceasefire fail",
    "sanctions escalat", "oil embargo", "supply shock", "terror attack",
    "assassination", "coup", "regime change", "emergency declaration",
    "refugee crisis", "market panic",
    "tariff", "trade war", "tariffs escalat", "sanctions",
    "emergency rate", "emergency cut", "emergency meeting",
    "iran strike", "iran nuclear", "strait of hormuz", "houthi",
    "taiwan invasion", "taiwan strait", "south china sea conflict",
    "russia ukraine offensive", "ukraine offensive", "kyiv attack",
]

_MEDIUM_KW: list[str] = [
    "tension", "tensions", "standoff", "border clash", "border incident",
    "protest", "unrest", "riot", "strike action", "political crisis",
    "debt ceiling", "government shutdown", "election uncertainty",
    "opec cut", "opec+ cut", "production cut", "supply disruption",
    "fed warning", "powell warning", "inflation surprise",
    "geopolit", "risk-off", "flight to safety", "safe haven demand",
    "gold rally", "gold surge", "gold spike",
    "china slowdown", "china crisis", "bank run", "banking crisis",
    "credit downgrade", "sovereign downgrade",
]

_RISK_OFF_KW: list[str] = [
    "risk-off", "safe haven", "flight to safety", "gold demand",
    "bonds rally", "yen surge", "swiss franc", "dollar strengthen",
    "vix spike", "volatility surge", "market sell-off", "equity sell-off",
]

# (entity_substring, integer_bonus_points)
_ENTITY_BONUSES: list[tuple[str, int]] = [
    ("iran",            2),
    ("nuclear",         3),
    ("taiwan",          2),
    ("china sea",       2),
    ("russia",          1),
    ("ukraine",         1),
    ("middle east",     1),
    ("hamas",           2),
    ("hezbollah",       2),
    ("houthi",          2),
    ("isis",            2),
    ("opec",            1),
    ("trump",           1),
    ("federal reserve", 1),
    ("powell",          1),
    ("yellen",          1),
    ("bank of japan",   1),
    ("boj",             1),
]

# ---------------------------------------------------------------------------
# Risk-level thresholds and parameters
# Raw score is integer; normalised to [0,1] by dividing by _MAX_RAW_SCORE.
# ---------------------------------------------------------------------------

_MAX_RAW_SCORE = 10  # raw scores are capped here before normalisation

# (level_name, min_normalised_score)  — evaluated in descending order
_LEVEL_THRESHOLDS: list[tuple[str, float]] = [
    ("extreme",  0.8),
    ("high",     0.5),
    ("elevated", 0.3),
    ("normal",   0.1),
    ("calm",     0.0),
]

# sl_multiplier: added on top of base ATR multiplier from risk engine
_LEVEL_PARAMS: dict[str, dict[str, Any]] = {
    "extreme": {
        "sl_multiplier": 0.75,
        "notes":         "EXTREME geo risk — strongly widen SL, favour safe-haven longs",
    },
    "high": {
        "sl_multiplier": 0.50,
        "notes":         "HIGH geo risk — widen SL, favour safe-haven longs",
    },
    "elevated": {
        "sl_multiplier": 0.25,
        "notes":         "ELEVATED geo risk — slight SL buffer; monitor for escalation",
    },
    "normal": {
        "sl_multiplier": 0.0,
        "notes":         "Normal geo risk — technicals dominate",
    },
    "calm": {
        "sl_multiplier": 0.0,
        "notes":         "Calm environment — no geo premium required",
    },
}

# Instrument bias rules per risk level
# (xauusd gets safe-haven bid; equities/risk assets are sold)
_INSTRUMENT_BIAS_BY_LEVEL: dict[str, dict[str, str]] = {
    "extreme": {
        "XAUUSD":  "bullish",
        "GBPJPY":  "bearish",
        "WTI":     "bullish",   # supply shock premium
        "NAS100":  "bearish",
        "BTCUSDT": "bearish",   # crypto sells off in extreme risk-off
        "ETHUSDT": "bearish",
    },
    "high": {
        "XAUUSD":  "bullish",
        "GBPJPY":  "bearish",
        "WTI":     "bullish",
        "NAS100":  "bearish",
        "BTCUSDT": "neutral",
        "ETHUSDT": "neutral",
    },
    "elevated": {
        "XAUUSD":  "bullish",
        "GBPJPY":  "neutral",
        "WTI":     "neutral",
        "NAS100":  "neutral",
        "BTCUSDT": "neutral",
        "ETHUSDT": "neutral",
    },
    "normal": {
        "XAUUSD":  "neutral",
        "GBPJPY":  "neutral",
        "WTI":     "neutral",
        "NAS100":  "neutral",
        "BTCUSDT": "neutral",
        "ETHUSDT": "neutral",
    },
    "calm": {
        "XAUUSD":  "neutral",
        "GBPJPY":  "neutral",
        "WTI":     "neutral",
        "NAS100":  "neutral",
        "BTCUSDT": "neutral",
        "ETHUSDT": "neutral",
    },
}

_NEUTRAL_BIASES: dict[str, str] = {k: "neutral" for k in
                                    ["XAUUSD", "GBPJPY", "WTI", "NAS100", "BTCUSDT", "ETHUSDT"]}

_FALLBACK: dict[str, Any] = {
    "score":              0.0,
    "sl_multiplier":      0.0,
    "notes":              "Geo filter unavailable — defaulting to neutral",
    "instrument_biases":  dict(_NEUTRAL_BIASES),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_headline(title: str) -> int:
    """
    Return a raw integer score for one headline.

    Caps per-headline contributions to avoid one explosive headline
    dominating the entire score:
      extreme keywords: max 2 hits × 3 pts = 6
      high keywords:    max 3 hits × 2 pts = 6
      medium keywords:  max 3 hits × 1 pt  = 3
    """
    low    = title.lower()
    points = 0

    extreme_hits = 0
    for kw in _EXTREME_KW:
        if kw in low and extreme_hits < 2:
            points      += 3
            extreme_hits += 1

    high_hits = 0
    for kw in _HIGH_KW:
        if kw in low and high_hits < 3:
            points    += 2
            high_hits += 1

    med_hits = 0
    for kw in _MEDIUM_KW:
        if kw in low and med_hits < 3:
            points  += 1
            med_hits += 1

    if any(kw in low for kw in _RISK_OFF_KW):
        points += 1

    for entity, bonus in _ENTITY_BONUSES:
        if entity in low:
            points += bonus

    return points


def _normalise(raw_score: int) -> float:
    """Map integer raw score → float in [0.0, 1.0]."""
    capped = min(raw_score, _MAX_RAW_SCORE)
    return round(capped / _MAX_RAW_SCORE, 3)


def _resolve_level(normalised: float) -> str:
    """Return the risk level string for a normalised score."""
    for level, threshold in _LEVEL_THRESHOLDS:
        if normalised >= threshold:
            return level
    return "calm"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_geo_score(headlines: list[dict] | None = None) -> dict[str, Any]:
    """
    Score current geopolitical risk and return bias adjustments.

    Parameters
    ----------
    headlines : list[dict] | None
        Optional pre-fetched list of news items, each containing at least
        a ``"title"`` key.  When *None*, the function attempts to fetch
        headlines via ``v2.intelligence.news_filter`` (gracefully degrading
        if it is unavailable).

    Returns
    -------
    dict with keys:
        score              float         — normalised risk score 0.0–1.0
        sl_multiplier      float         — additional ATR multiplier to add to SL
        notes              str           — human-readable risk summary
        instrument_biases  dict[str,str] — per-symbol bias (bullish/bearish/neutral)
    """
    # --- Obtain headlines ---------------------------------------------------
    if headlines is None:
        try:
            from v2.intelligence.news_filter import fetch_ff_calendar  # type: ignore[import]
            headlines = fetch_ff_calendar()
            logger.debug("geo_filter: fetched %d headlines from news_filter", len(headlines))
        except ImportError:
            logger.warning("geo_filter: v2.intelligence.news_filter not available; trying direct fetch")
            headlines = _fetch_headlines_direct()

    if not headlines:
        logger.warning("geo_filter: no headlines available — returning fallback")
        return dict(_FALLBACK)

    # --- Filter to risk-relevant categories (soft filter) ------------------
    relevant = [
        h for h in headlines
        if str(h.get("category", "")).upper() in ("RISK", "MACRO", "GOLD", "GEOPOLITICS")
    ]
    if not relevant:
        relevant = list(headlines)

    # --- Score headlines ---------------------------------------------------
    raw_total = 0
    for h in relevant:
        title = str(h.get("title", ""))
        if title:
            raw_total += _score_headline(title)

    normalised = _normalise(raw_total)
    level      = _resolve_level(normalised)
    params     = _LEVEL_PARAMS[level]

    logger.debug(
        "geo_filter: raw=%d normalised=%.3f level=%s sl_mult=%.2f",
        raw_total, normalised, level, params["sl_multiplier"],
    )

    return {
        "score":             normalised,
        "sl_multiplier":     params["sl_multiplier"],
        "notes":             params["notes"],
        "instrument_biases": dict(_INSTRUMENT_BIAS_BY_LEVEL[level]),
    }


def _fetch_headlines_direct() -> list[dict]:
    """
    Minimal fallback fetcher: retrieve Forex Factory XML calendar directly.
    Returns an empty list on any failure so the caller can degrade gracefully.
    """
    try:
        import xml.etree.ElementTree as ET
        import requests  # type: ignore[import]

        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items: list[dict] = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            title    = (title_el.text or "").strip() if title_el is not None else ""
            if title:
                items.append({"title": title, "category": "MACRO"})
        return items
    except ImportError as exc:
        logger.warning("geo_filter: requests not available for direct fetch: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("geo_filter: direct headline fetch failed: %s", exc)
        return []
