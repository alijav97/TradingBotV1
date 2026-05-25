"""
analysis/candle_patterns.py — TA-Lib candlestick pattern wrapper for TradingBotV2.

Detects 30 key patterns across bullish / bearish / neutral categories.
Falls back gracefully if TA-Lib is not installed (returns empty result).

Usage:
    from v2.analysis.candle_patterns import detect_patterns
    patterns = detect_patterns(df)
    # patterns["bullish"]  → list of active bullish pattern names
    # patterns["bearish"]  → list of active bearish pattern names
    # patterns["bias"]     → "bullish" | "bearish" | "neutral"
    # patterns["score"]    → int (-5 to +5)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import talib
    _TALIB_OK = True
except ImportError:
    talib = None  # type: ignore
    _TALIB_OK = False
    logger.warning("TA-Lib not installed — candle pattern detection disabled")


# ── Pattern registry ──────────────────────────────────────────────────────────
# (talib_fn_name, human_name, direction)
# direction: +1 = bullish, -1 = bearish, 0 = both (check sign of return value)

_PATTERNS: list[tuple[str, str, int]] = [
    # Bullish patterns
    ("CDL_HAMMER",           "Hammer",              +1),
    ("CDL_INVERTEDHAMMER",   "Inverted Hammer",     +1),
    ("CDL_MORNINGSTAR",      "Morning Star",        +1),
    ("CDL_MORNINGDOJISTAR",  "Morning Doji Star",   +1),
    ("CDL_BULLISHENGULFING", "Bullish Engulfing",   +1),
    ("CDL_PIERCINGLINE",     "Piercing Line",       +1),
    ("CDL_3WHITESOLDIERS",   "Three White Soldiers",+1),
    ("CDL_DRAGONFLYDOJI",    "Dragonfly Doji",      +1),
    ("CDL_HARAMIBULLISH",    "Bullish Harami",      +1),
    ("CDL_TWEEZERBOTTOM",    "Tweezer Bottom",      +1),
    ("CDL_RISINGTHREEMETHODS","Rising Three Methods",+1),
    ("CDL_UPSIDEGAPTWOCROWS","Upside Gap Two Crows", 0),
    # Bearish patterns
    ("CDL_HANGINGMAN",       "Hanging Man",         -1),
    ("CDL_SHOOTINGSTAR",     "Shooting Star",       -1),
    ("CDL_EVENINGSTAR",      "Evening Star",        -1),
    ("CDL_EVENINGDOJISTAR",  "Evening Doji Star",   -1),
    ("CDL_BEARISHENGULFING", "Bearish Engulfing",   -1),
    ("CDL_DARKCLOUDCOVER",   "Dark Cloud Cover",    -1),
    ("CDL_3BLACKCROWS",      "Three Black Crows",   -1),
    ("CDL_GRAVESTONEDOJI",   "Gravestone Doji",     -1),
    ("CDL_HARAMIBEARISH",    "Bearish Harami",      -1),
    ("CDL_TWEEZERTOP",       "Tweezer Top",         -1),
    ("CDL_FALLINGTHREEMETHODS","Falling Three Methods",-1),
    # Neutral (sign of return determines direction)
    ("CDL_DOJI",             "Doji",                0),
    ("CDL_DOJISTAR",         "Doji Star",           0),
    ("CDL_HARAMI",           "Harami",              0),
    ("CDL_ENGULFING",        "Engulfing",           0),
    ("CDL_KICKING",          "Kicking",             0),
    ("CDL_LADDERBOTTOM",     "Ladder Bottom",       0),
    ("CDL_BELTHOLD",         "Belt Hold",           0),
]


def detect_patterns(df: pd.DataFrame) -> dict:
    """
    Detect candlestick patterns on the last candle of df.

    Parameters
    ----------
    df : OHLCV DataFrame with columns open, high, low, close

    Returns
    -------
    {
        "bullish":  [list of pattern names],
        "bearish":  [list of pattern names],
        "neutral":  [list of pattern names],
        "bias":     "bullish" | "bearish" | "neutral",
        "score":    int  (+5 max bullish, -5 max bearish),
        "strongest": str  (name of highest-weight pattern, or ""),
    }
    """
    empty = {"bullish": [], "bearish": [], "neutral": [], "bias": "neutral", "score": 0, "strongest": ""}

    if not _TALIB_OK:
        return empty

    if df.empty or len(df) < 5:
        return empty

    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)

    bullish_hits: list[str] = []
    bearish_hits: list[str] = []
    neutral_hits: list[str]  = []
    score = 0

    for fn_name, human_name, direction in _PATTERNS:
        fn = getattr(talib, fn_name, None)
        if fn is None:
            # Some older TA-Lib builds use different names — skip gracefully
            continue
        try:
            result = fn(o, h, l, c)
            val = int(result[-1])
            if val == 0:
                continue

            if direction == +1 or (direction == 0 and val > 0):
                bullish_hits.append(human_name)
                score += 1
            elif direction == -1 or (direction == 0 and val < 0):
                bearish_hits.append(human_name)
                score -= 1
            else:
                neutral_hits.append(human_name)

        except Exception as exc:
            logger.debug("Pattern %s failed: %s", fn_name, exc)

    # Cap score at ±5
    score = max(-5, min(5, score))

    if   score > 0: bias = "bullish"
    elif score < 0: bias = "bearish"
    else:           bias = "neutral"

    # Find strongest single pattern (whichever list is longer wins)
    strongest = ""
    if bullish_hits and score > 0:
        strongest = bullish_hits[0]
    elif bearish_hits and score < 0:
        strongest = bearish_hits[0]

    return {
        "bullish":  bullish_hits,
        "bearish":  bearish_hits,
        "neutral":  neutral_hits,
        "bias":     bias,
        "score":    score,
        "strongest": strongest,
    }


def get_pattern_summary(df: pd.DataFrame) -> str:
    """Return a short human-readable string for alerts."""
    p = detect_patterns(df)
    if not p["bullish"] and not p["bearish"]:
        return "No notable candle patterns"
    parts = []
    if p["bullish"]:
        parts.append("BULL: " + ", ".join(p["bullish"][:3]))
    if p["bearish"]:
        parts.append("BEAR: " + ", ".join(p["bearish"][:3]))
    return " | ".join(parts)
