"""
fundamental_bias.py
───────────────────
Reads macro fundamentals and detects when they conflict with technical
signals — e.g. inflation causing gold to rise even when charts say bearish.

Main entry points:
  get_fundamental_bias()                   -> dict
  check_fundamental_conflict(tech_dir)     -> dict
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_BASE_DIR, "data")
_CACHE_FILE = os.path.join(_DATA_DIR, "fundamental_cache.json")
_GST        = timezone(timedelta(hours=4))
_CACHE_TTL  = 30 * 60   # 30 minutes in seconds

os.makedirs(_DATA_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cache helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_cache() -> dict | None:
    try:
        if not os.path.exists(_CACHE_FILE):
            return None
        with open(_CACHE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        saved_ts = data.get("_cache_ts", 0)
        age = datetime.now(_GST).timestamp() - float(saved_ts)
        if age < _CACHE_TTL:
            return data
    except Exception:
        pass
    return None


def _save_cache(result: dict) -> None:
    try:
        result["_cache_ts"] = datetime.now(_GST).timestamp()
        with open(_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Headline scanner helper
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_headlines() -> list[str]:
    """Return a flat list of lowercase headline strings from news_monitor."""
    try:
        from news_monitor import fetch_news
        items = fetch_news()
        return [str(item.get("title", "") + " " + item.get("summary", "")).lower()
                for item in (items or [])]
    except Exception:
        return []


def _any_kw(headlines: list[str], keywords: list[str]) -> bool:
    return any(kw in h for h in headlines for kw in keywords)


# ══════════════════════════════════════════════════════════════════════════════
# Factor scorers
# ══════════════════════════════════════════════════════════════════════════════

def _score_inflation(headlines: list[str]) -> dict:
    cpi_bullish_kw = [
        "inflation rising", "cpi higher", "inflation hot",
        "prices surge", "inflation above", "inflation spike",
        "hotter than expected", "core inflation", "cpi beat",
    ]
    cpi_bearish_kw = [
        "inflation falling", "cpi lower", "inflation cooling",
        "prices ease", "deflation", "inflation slows",
        "cpi miss", "disinflation",
    ]
    if _any_kw(headlines, cpi_bullish_kw):
        return {
            "score": 2,
            "bias":  "bullish_gold",
            "note":  "Rising inflation → gold hedge demand",
        }
    elif _any_kw(headlines, cpi_bearish_kw):
        return {
            "score": -1,
            "bias":  "bearish_gold",
            "note":  "Falling inflation → less gold demand",
        }
    return {
        "score": 0,
        "bias":  "neutral",
        "note":  "Inflation data unclear",
    }


def _score_oil() -> dict:
    try:
        import yfinance as _yf
        cl = _yf.Ticker("CL=F")
        hist = cl.history(period="10d", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            raise ValueError("empty")
        oil_price     = float(hist["Close"].iloc[-1])
        oil_5d_ago    = float(hist["Close"].iloc[-5]) if len(hist) >= 5 else oil_price
        oil_change_pct = (oil_price - oil_5d_ago) / max(oil_5d_ago, 1) * 100

        if oil_price > 100 or oil_change_pct > 3:
            return {
                "score": 2,
                "bias":  "bullish_gold",
                "note":  f"Oil ${oil_price:.0f} → inflation fears → gold demand",
                "price": round(oil_price, 2),
            }
        elif oil_price < 70 or oil_change_pct < -3:
            return {
                "score": -1,
                "bias":  "bearish_gold",
                "note":  f"Oil ${oil_price:.0f} falling → lower inflation risk",
                "price": round(oil_price, 2),
            }
        return {
            "score": 0,
            "bias":  "neutral",
            "note":  f"Oil ${oil_price:.0f} neutral",
            "price": round(oil_price, 2),
        }
    except Exception:
        return {"score": 0, "bias": "neutral", "note": "Oil data unavailable", "price": 0.0}


def _score_fed(headlines: list[str]) -> dict:
    fed_dovish_kw = [
        "rate cut", "fed dovish", "fed pause", "lower rates",
        "fed pivot", "powell dovish", "rates fall", "fed holds",
        "rate reduction", "fed easing",
    ]
    fed_hawkish_kw = [
        "rate hike", "fed hawkish", "higher rates", "tighten",
        "powell hawkish", "rates rise", "fed raises", "rate increase",
        "aggressive fed",
    ]
    if _any_kw(headlines, fed_dovish_kw):
        return {"score": 2, "bias": "bullish_gold", "note": "Fed dovish → gold bullish"}
    elif _any_kw(headlines, fed_hawkish_kw):
        return {"score": -2, "bias": "bearish_gold", "note": "Fed hawkish → gold bearish"}
    return {"score": 0, "bias": "neutral", "note": "Fed stance unclear"}


def _score_dxy() -> dict:
    try:
        from dxy_correlation import get_dxy_context
        ctx      = get_dxy_context()
        dxy_trend = str(ctx.get("dxy_trend", "sideways"))
        dxy_rsi   = float(ctx.get("dxy_rsi", 50) or 50)

        if dxy_trend == "down" and dxy_rsi < 45:
            return {
                "score": 2,
                "bias":  "bullish_gold",
                "note":  f"DXY falling (RSI {dxy_rsi:.0f}) → dollar weak → gold up",
            }
        elif dxy_trend == "up" and dxy_rsi > 55:
            return {
                "score": -2,
                "bias":  "bearish_gold",
                "note":  f"DXY rising (RSI {dxy_rsi:.0f}) → dollar strong → gold down",
            }
        return {"score": 0, "bias": "neutral", "note": "DXY neutral"}
    except Exception:
        return {"score": 0, "bias": "neutral", "note": "DXY unavailable"}


def _score_geo() -> dict:
    try:
        from geo_filter import get_geopolitical_score
        geo       = get_geopolitical_score()
        geo_level = str(geo.get("geo_risk_level", "normal"))
        geo_raw   = int(geo.get("geo_score", 0) or 0)

        if geo_level in ("extreme", "high"):
            return {
                "score": 2,
                "bias":  "bullish_gold",
                "note":  f"Geo risk {geo_level} → safe haven demand → gold up",
            }
        elif geo_level == "elevated":
            return {
                "score": 1,
                "bias":  "bullish_gold",
                "note":  "Elevated geo risk → mild safe haven demand",
            }
        elif geo_level == "calm":
            return {
                "score": -1,
                "bias":  "bearish_gold",
                "note":  "Calm markets → safe haven demand low",
            }
        return {"score": 0, "bias": "neutral", "note": "Normal geo conditions"}
    except Exception:
        return {"score": 0, "bias": "neutral", "note": "Geo data unavailable"}


# ══════════════════════════════════════════════════════════════════════════════
# Conflict detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_conflict(technical_direction: str, fundamental_bias: str) -> dict:
    """
    Compare a technical trade direction against the fundamental bias and
    return a conflict dict.
    """
    tech_bullish = str(technical_direction).lower() in ("long", "buy")
    fund_bullish = "BULLISH" in fundamental_bias
    fund_bearish = "BEARISH" in fundamental_bias

    if tech_bullish and fund_bearish:
        severity = "HIGH" if "STRONGLY" in fundamental_bias else "MODERATE"
        rec = "SKIP or use 25% size" if severity == "HIGH" else "Reduce size to 50%"
        return {
            "conflict":  True,
            "severity":  severity,
            "message": (
                f"⚠ FUNDAMENTAL CONFLICT\n"
                f"Technical: LONG setup\n"
                f"Fundamental: {fundamental_bias}\n"
                f"Macro is working AGAINST this long.\n"
                f"Recommendation: {rec}"
            ),
        }
    elif not tech_bullish and fund_bullish:
        severity = "HIGH" if "STRONGLY" in fundamental_bias else "MODERATE"
        rec = "SKIP or use 25% size" if severity == "HIGH" else "Take TP1 only"
        return {
            "conflict":  True,
            "severity":  severity,
            "message": (
                f"⚠ FUNDAMENTAL CONFLICT\n"
                f"Technical: SHORT setup\n"
                f"Fundamental: {fundamental_bias}\n"
                f"Macro is working AGAINST this short.\n"
                f"Recommendation: {rec}"
            ),
        }
    return {
        "conflict":  False,
        "severity":  "NONE",
        "message":   "✓ Technical and fundamental aligned",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main functions
# ══════════════════════════════════════════════════════════════════════════════

def get_fundamental_bias() -> dict:
    """
    Fetch and score 5 macro factors for gold.  Caches result 30 minutes.
    Never raises — returns a safe fallback dict on any error.
    """
    try:
        cached = _load_cache()
        if cached:
            return cached
        return _compute_fundamental_bias()
    except Exception:
        return _fallback()


def _compute_fundamental_bias() -> dict:
    headlines = _fetch_headlines()

    f_inflation = _score_inflation(headlines)
    f_oil       = _score_oil()
    f_fed       = _score_fed(headlines)
    f_dxy       = _score_dxy()
    f_geo       = _score_geo()

    total_score = (
        f_inflation["score"] + f_oil["score"] +
        f_fed["score"] + f_dxy["score"] + f_geo["score"]
    )

    if total_score >= 5:
        bias    = "STRONGLY_BULLISH"
        summary = "Strong macro tailwinds for gold"
    elif total_score >= 2:
        bias    = "BULLISH"
        summary = "Macro supports gold prices"
    elif total_score >= -1:
        bias    = "NEUTRAL"
        summary = "Mixed macro signals"
    elif total_score >= -4:
        bias    = "BEARISH"
        summary = "Macro headwinds for gold"
    else:
        bias    = "STRONGLY_BEARISH"
        summary = "Strong macro headwinds for gold"

    # Max possible positive = 9 (2+2+2+2+1 geo elevated / 2+2+2+2+2=10 but geo max=2)
    max_pos = 9
    confidence = round(min(10.0, max(0.0, (total_score + 8) / (max_pos + 8) * 10)), 1)

    display_line = f"📊 Fundamental: {bias.replace('_',' ').title()} ({total_score:+d})"

    result: dict = {
        "fundamental_bias": bias,
        "total_score":      total_score,
        "summary":          summary,
        "factors": {
            "inflation":    f_inflation,
            "oil":          f_oil,
            "fed":          f_fed,
            "dxy":          f_dxy,
            "geopolitical": f_geo,
        },
        "available":    True,
        "display_line": display_line,
        "confidence":   confidence,
        "timeframe":    "medium_term (1-4 weeks)",
    }
    _save_cache(result)
    return result


def _fallback() -> dict:
    return {
        "fundamental_bias": "NEUTRAL",
        "total_score":       0,
        "summary":           "Fundamental data unavailable",
        "factors": {
            "inflation":    {"score": 0, "bias": "neutral", "note": "unavailable"},
            "oil":          {"score": 0, "bias": "neutral", "note": "unavailable", "price": 0.0},
            "fed":          {"score": 0, "bias": "neutral", "note": "unavailable"},
            "dxy":          {"score": 0, "bias": "neutral", "note": "unavailable"},
            "geopolitical": {"score": 0, "bias": "neutral", "note": "unavailable"},
        },
        "available":    False,
        "display_line": "📊 Fundamental: Unavailable",
        "confidence":   5.0,
        "timeframe":    "medium_term (1-4 weeks)",
    }


def check_fundamental_conflict(technical_direction: str) -> dict:
    """Check if a technical direction conflicts with the current fundamental bias."""
    try:
        bias = get_fundamental_bias()
        return detect_conflict(technical_direction, bias["fundamental_bias"])
    except Exception:
        return {"conflict": False, "severity": "NONE", "message": "Fundamental check unavailable"}
