"""
geo_filter.py — Geopolitical Risk Scoring Layer for TradingBotV1
────────────────────────────────────────────────────────────────
Scores real-time geopolitical risk from news headlines fetched via
news_monitor.fetch_news(). Returns a structured context dict that can
be used to:
  • Adjust signal confidence scores
  • Widen SL ATR multipliers under extreme/high risk
  • Display a colour-coded risk level in the sidebar

Usage:
    from geo_filter import get_geopolitical_score
    geo_ctx = get_geopolitical_score()
"""

from __future__ import annotations

# ── Keywords that contribute to geo risk scoring ──────────────────────────────

_EXTREME_KEYWORDS: list[str] = [
    "nuclear", "world war", "world war iii", "ww3", "nato invade", "nato strikes",
    "missile strike", "ballistic missile", "chemical weapon", "biological weapon",
    "hypersonic", "carrier strike group deploy",
]

_HIGH_KEYWORDS: list[str] = [
    "war", "invasion", "invasion of", "military offensive", "airstrikes", "airstrike",
    "ground offensive", "troops deploy", "troops cross", "troops advance",
    "conflict escalat", "escalat", "ceasefire collapses", "ceasefire fail",
    "sanctions escalat", "oil embargo", "supply shock", "terror attack",
    "assassination", "coup", "regime change", "emergency declaration",
    "refugee crisis", "market panic",
    # Trump / policy shock
    "tariff", "trade war", "tariffs escalat", "sanctions",
    # Central bank
    "emergency rate", "emergency cut", "emergency meeting",
    # Middle East
    "iran strike", "iran nuclear", "strait of hormuz", "houthi",
    # China
    "taiwan invasion", "taiwan strait", "south china sea conflict",
    # Russia / Ukraine
    "russia ukraine offensive", "ukraine offensive", "kyiv attack",
]

_MEDIUM_KEYWORDS: list[str] = [
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

_RISK_OFF_KEYWORDS: list[str] = [
    "risk-off", "safe haven", "flight to safety", "gold demand",
    "bonds rally", "yen surge", "swiss franc", "dollar strengthen",
    "vix spike", "volatility surge", "market sell-off", "equity sell-off",
]

# ── Source / entity bonus scores ──────────────────────────────────────────────

_SOURCE_BONUSES: list[tuple[str, int]] = [
    ("iran",          2),
    ("nuclear",       3),
    ("taiwan",        2),
    ("china sea",     2),
    ("russia",        1),
    ("ukraine",       1),
    ("middle east",   1),
    ("hamas",         2),
    ("hezbollah",     2),
    ("houthi",        2),
    ("isis",          2),
    ("opec",          1),
    ("trump",         1),
    ("federal reserve", 1),
    ("powell",        1),
    ("yellen",        1),
    ("bank of japan", 1),
    ("boj",           1),
]

# ── Risk level thresholds ──────────────────────────────────────────────────────

_LEVEL_THRESHOLDS: list[tuple[str, int]] = [
    ("extreme",  8),
    ("high",     5),
    ("elevated", 3),
    ("normal",   1),
    ("calm",     0),
]

# ── Per-level parameters ───────────────────────────────────────────────────────

_LEVEL_PARAMS: dict[str, dict] = {
    "extreme":  {"confidence_adjustment": +0.5, "sl_atr_multiplier": 1.5,
                 "gold_bias": "bullish",          "colour": "#E05555"},
    "high":     {"confidence_adjustment": +0.5, "sl_atr_multiplier": 1.0,
                 "gold_bias": "bullish",          "colour": "#E08020"},
    "elevated": {"confidence_adjustment":  0.0, "sl_atr_multiplier": 0.5,
                 "gold_bias": "slightly_bullish", "colour": "#F4C542"},
    "normal":   {"confidence_adjustment":  0.0, "sl_atr_multiplier": 0.0,
                 "gold_bias": "neutral",          "colour": "#2ecc71"},
    "calm":     {"confidence_adjustment":  0.0, "sl_atr_multiplier": 0.0,
                 "gold_bias": "neutral",          "colour": "#2ecc71"},
}

_FALLBACK: dict = {
    "available":            False,
    "geo_score":            0,
    "geo_risk_level":       "normal",
    "gold_bias":            "neutral",
    "triggered_events":     [],
    "top_headlines":        [],
    "sl_atr_multiplier":    0.0,
    "confidence_adjustment": 0.0,
    "recommendation":       "Geo filter unavailable — rely on technicals.",
    "colour":               "#2ecc71",
}


def _score_headline(title: str) -> tuple[int, list[str]]:
    """Return (raw_points, list_of_matched_keywords) for one headline."""
    low       = title.lower()
    points    = 0
    matched:  list[str] = []

    # Extreme: cap contribution at 2 per headline
    extreme_hits = 0
    for kw in _EXTREME_KEYWORDS:
        if kw in low and extreme_hits < 2:
            points      += 3
            extreme_hits += 1
            matched.append(kw)

    # High: cap contribution at 3 per headline
    high_hits = 0
    for kw in _HIGH_KEYWORDS:
        if kw in low and high_hits < 3:
            points    += 2
            high_hits += 1
            matched.append(kw)

    # Medium: cap at 3 per headline
    med_hits = 0
    for kw in _MEDIUM_KEYWORDS:
        if kw in low and med_hits < 3:
            points  += 1
            med_hits += 1
            matched.append(kw)

    # Risk-off modifier: +1 total if any risk-off keyword present
    if any(kw in low for kw in _RISK_OFF_KEYWORDS):
        points += 1
        matched.append("risk-off")

    # Source / entity bonus (applied per headline, not capped)
    for entity, bonus in _SOURCE_BONUSES:
        if entity in low:
            points += bonus
            matched.append(f"entity:{entity}")

    return points, matched


def get_geopolitical_score(headlines: list[dict] | None = None) -> dict:
    """
    Score geopolitical risk from news headlines.

    Parameters
    ----------
    headlines : list[dict] | None
        Pre-fetched news items (each dict has 'title' and 'category' keys).
        If None, news_monitor.fetch_news() is called automatically.

    Returns
    -------
    dict with keys:
        available, geo_score, geo_risk_level, gold_bias,
        triggered_events, top_headlines, sl_atr_multiplier,
        confidence_adjustment, recommendation, colour
    """
    # ── Fetch headlines if not provided ──────────────────────────────────────
    if headlines is None:
        try:
            from news_monitor import fetch_news
            headlines = fetch_news()
        except Exception:
            return dict(_FALLBACK)

    if not headlines:
        return dict(_FALLBACK)

    # ── Filter to RISK + MACRO categories only ────────────────────────────────
    relevant = [
        h for h in headlines
        if str(h.get("category", "")).upper() in ("RISK", "MACRO", "GOLD", "GEOPOLITICS")
    ]
    # If no category-filtered items, fall back to all headlines
    if not relevant:
        relevant = list(headlines)

    # ── Score each headline ───────────────────────────────────────────────────
    raw_score        = 0
    triggered_events: list[str] = []
    scored_headlines: list[tuple[int, str]] = []   # (pts, title)

    for h in relevant:
        title = str(h.get("title", ""))
        pts, matched = _score_headline(title)
        if pts > 0:
            raw_score += pts
            triggered_events.extend(matched)
            scored_headlines.append((pts, title))

    # Cap total score at 10
    geo_score = min(10, raw_score)

    # ── Determine risk level ──────────────────────────────────────────────────
    geo_risk_level = "calm"
    for level, threshold in _LEVEL_THRESHOLDS:
        if geo_score >= threshold:
            geo_risk_level = level
            break

    params = _LEVEL_PARAMS[geo_risk_level]

    # ── Top headlines (highest scoring, max 5) ────────────────────────────────
    scored_headlines.sort(key=lambda x: x[0], reverse=True)
    top_headlines = [t for _, t in scored_headlines[:5]]

    # ── Deduplicate triggered events ──────────────────────────────────────────
    seen:           set[str] = set()
    unique_events:  list[str] = []
    for ev in triggered_events:
        if ev not in seen:
            seen.add(ev)
            unique_events.append(ev)

    # ── Build recommendation ──────────────────────────────────────────────────
    if geo_risk_level == "extreme":
        recommendation = (
            "⚠️ EXTREME geo risk — widen SL, reduce size, favour LONG (safe-haven gold bid)."
        )
    elif geo_risk_level == "high":
        recommendation = (
            "⚠️ HIGH geo risk — widen SL by 1.0× ATR, LONG bias amplified."
        )
    elif geo_risk_level == "elevated":
        recommendation = (
            "⚡ ELEVATED geo risk — monitor for escalation; slight SL buffer recommended."
        )
    else:
        recommendation = "✅ Low geopolitical risk — technicals dominate."

    return {
        "available":             True,
        "geo_score":             geo_score,
        "geo_risk_level":        geo_risk_level,
        "gold_bias":             params["gold_bias"],
        "triggered_events":      unique_events[:15],
        "top_headlines":         top_headlines,
        "sl_atr_multiplier":     params["sl_atr_multiplier"],
        "confidence_adjustment": params["confidence_adjustment"],
        "recommendation":        recommendation,
        "colour":                params["colour"],
    }


# ── CLI quick-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json
    result = get_geopolitical_score()
    print(_json.dumps(result, indent=2))
