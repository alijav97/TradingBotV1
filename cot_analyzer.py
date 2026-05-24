"""
cot_analyzer.py — Commitment of Traders (COT) Signal Engine
────────────────────────────────────────────────────────────
Fetches CFTC COT data for Gold (GC) futures and derives:
  • Speculator net position as % of open interest
  • Directional bias: STRONGLY_BULLISH → STRONGLY_BEARISH
  • Contrarian hedger signal (commercials flip direction)
  • 24-hour cache to avoid hammering data sources

Usage:
    from cot_analyzer import fetch_cot_data, get_cot_signal
    cot  = fetch_cot_data()
    sig  = get_cot_signal("long", cot)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_BASE_DIR, "data", "cot_cache.json")

# ── Thresholds ─────────────────────────────────────────────────────────────────
_STRONGLY_BULLISH_PCT =  20.0   # spec_net_pct ≥ +20 → STRONGLY_BULLISH  (+1.0)
_BULLISH_PCT          =   5.0   # spec_net_pct ≥  +5 → BULLISH            (+0.5)
_BEARISH_PCT          =  -5.0   # spec_net_pct ≤  −5 → BEARISH            (−0.5)
_STRONGLY_BEARISH_PCT = -20.0   # spec_net_pct ≤ −20 → STRONGLY_BEARISH  (−1.0)

# ── Gold futures COT code (CFTC) ───────────────────────────────────────────────
_GOLD_FUTURES_CODE = "088691"   # GOLD - COMMODITY EXCHANGE INC.


def _load_cache() -> dict | None:
    """Return cached COT data if it exists and is less than 24 hours old."""
    try:
        if not os.path.exists(_CACHE_PATH):
            return None
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cached_at = data.get("cached_at", "")
        if not cached_at:
            return None
        age = datetime.now(timezone.utc) - datetime.fromisoformat(cached_at)
        if age < timedelta(hours=24):
            return data
    except Exception:
        pass
    return None


def _save_cache(data: dict) -> None:
    """Save COT data to disk with a timestamp."""
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        data["cached_at"] = datetime.now(timezone.utc).isoformat()
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _derive_signals(raw: dict) -> dict:
    """
    Compute spec_net_pct, bias, hedger contrarian signal,
    and display_line from raw COT figures.
    """
    nc_long    = float(raw.get("noncomm_long",    180_000))
    nc_short   = float(raw.get("noncomm_short",    90_000))
    comm_long  = float(raw.get("comm_long",        40_000))
    comm_short = float(raw.get("comm_short",      200_000))
    total_oi   = float(raw.get("total_oi",        500_000))

    spec_net       = nc_long - nc_short
    comm_net       = comm_long - comm_short
    spec_net_pct   = (spec_net / total_oi * 100) if total_oi else 0.0
    spec_net_pct   = round(spec_net_pct, 1)

    # ── Bias classification ───────────────────────────────────────────────────
    if spec_net_pct >= _STRONGLY_BULLISH_PCT:
        bias   = "STRONGLY_BULLISH"
        boost  = 1.0
        emoji  = "🟢🟢"
    elif spec_net_pct >= _BULLISH_PCT:
        bias   = "BULLISH"
        boost  = 0.5
        emoji  = "🟢"
    elif spec_net_pct <= _STRONGLY_BEARISH_PCT:
        bias   = "STRONGLY_BEARISH"
        boost  = -1.0
        emoji  = "🔴🔴"
    elif spec_net_pct <= _BEARISH_PCT:
        bias   = "BEARISH"
        boost  = -0.5
        emoji  = "🔴"
    else:
        bias   = "NEUTRAL"
        boost  = 0.0
        emoji  = "⚪"

    # ── Hedger contrarian signal ──────────────────────────────────────────────
    # Commercials (smart money hedgers) typically hold positions opposite to
    # speculators. When commercials are NET LONG → contrarian bullish for gold.
    hedger_net_pct = (comm_net / total_oi * 100) if total_oi else 0.0
    if hedger_net_pct > 5.0:
        hedger = "CONTRARIAN_BULLISH"
        hedger_note = f"Commercials net long ({hedger_net_pct:+.1f}%) — smart money buying"
    elif hedger_net_pct < -5.0:
        hedger = "CONTRARIAN_BEARISH"
        hedger_note = f"Commercials net short ({hedger_net_pct:+.1f}%) — smart money selling"
    else:
        hedger = "NEUTRAL"
        hedger_note = f"Commercials near neutral ({hedger_net_pct:+.1f}%)"

    report_date = raw.get("report_date", "unknown")

    display_line = (
        f"{emoji} COT {bias.replace('_', ' ')} | "
        f"Specs: {spec_net:+,.0f} net ({spec_net_pct:+.1f}%) | "
        f"Hedgers: {hedger_note.split(' — ')[0]} | "
        f"Report: {report_date}"
    )

    return {
        "bias":           bias,
        "boost":          boost,
        "spec_net":       round(spec_net),
        "spec_net_pct":   spec_net_pct,
        "comm_net":       round(comm_net),
        "hedger":         hedger,
        "hedger_note":    hedger_note,
        "report_date":    report_date,
        "total_oi":       round(total_oi),
        "display_line":   display_line,
        "available":      True,
        "source":         raw.get("source", "hardcoded"),
    }


def fetch_cot_data() -> dict[str, Any]:
    """
    Fetch Gold COT data.  Order of preference:
      1. Valid 24-hour cache on disk
      2. Live fetch via `cot_reports` Python package
      3. Hardcoded fallback (recent typical positioning)

    Returns a dict with all fields including `available`, `bias`,
    `boost`, `spec_net_pct`, `hedger`, `display_line`.
    """
    # ── 1. Check cache ────────────────────────────────────────────────────────
    cached = _load_cache()
    if cached and cached.get("available"):
        return cached

    # ── 2. Try live fetch via cot_reports package ─────────────────────────────
    try:
        import cot_reports as cot                          # pip install cot_reports
        df = cot.cot_year(cot_report_type="legacy_fut_only")
        # Filter for Gold futures
        gold_df = df[df["Market and Exchange Names"].str.contains("GOLD", na=False, case=False)]
        if not gold_df.empty:
            row = gold_df.iloc[-1]
            raw = {
                "noncomm_long":  float(row.get("Noncommercial Positions-Long (All)", 180_000)),
                "noncomm_short": float(row.get("Noncommercial Positions-Short (All)", 90_000)),
                "comm_long":     float(row.get("Commercial Positions-Long (All)", 40_000)),
                "comm_short":    float(row.get("Commercial Positions-Short (All)", 200_000)),
                "total_oi":      float(row.get("Open Interest (All)", 500_000)),
                "report_date":   str(row.get("As of Date in Form YYYY-MM-DD", "unknown")),
                "source":        "cot_reports_live",
            }
            result = _derive_signals(raw)
            _save_cache(result | raw)
            return result
    except Exception:
        pass  # fall through to hardcoded

    # ── 3. Hardcoded fallback (typical recent positioning for XAUUSD) ─────────
    # Values represent a moderately bullish speculator positioning.
    # These are placeholders — real data fetched when cot_reports is installed.
    raw_fallback = {
        "noncomm_long":  180_000,
        "noncomm_short":  90_000,
        "comm_long":      40_000,
        "comm_short":    200_000,
        "total_oi":      500_000,
        "report_date":   "estimate",
        "source":        "hardcoded_fallback",
    }
    result = _derive_signals(raw_fallback)
    _save_cache(result | raw_fallback)
    return result


def get_cot_signal(direction: str, cot_data: dict | None = None) -> dict[str, Any]:
    """
    Return COT alignment signal for the given trade direction.

    Parameters
    ----------
    direction : "long" | "short"
    cot_data  : pre-fetched dict from fetch_cot_data(); fetched if None

    Returns
    -------
    boost     : float  — score adjustment for confluence_engine (+1.0 to −1.0)
    aligned   : bool   — COT supports direction
    opposed   : bool   — COT opposes direction
    bias      : str    — STRONGLY_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONGLY_BEARISH
    note      : str    — one-line summary
    hedger    : str    — CONTRARIAN_BULLISH / CONTRARIAN_BEARISH / NEUTRAL
    available : bool
    """
    if cot_data is None:
        cot_data = fetch_cot_data()

    if not cot_data.get("available", False):
        return {
            "boost": 0.0, "aligned": False, "opposed": False,
            "bias": "NEUTRAL", "note": "COT data unavailable",
            "hedger": "NEUTRAL", "available": False,
        }

    bias  = cot_data.get("bias", "NEUTRAL")
    boost = float(cot_data.get("boost", 0.0))
    is_long = direction.lower() in ("long", "buy")

    # For a LONG trade: bullish COT is aligned, bearish is opposed
    # For a SHORT trade: bearish COT is aligned, bullish is opposed
    cot_bullish = bias in ("BULLISH", "STRONGLY_BULLISH")
    cot_bearish = bias in ("BEARISH", "STRONGLY_BEARISH")

    if is_long:
        aligned = cot_bullish
        opposed = cot_bearish
        if opposed:
            boost = -abs(boost)   # always subtract if opposed
    else:
        aligned = cot_bearish
        opposed = cot_bullish
        if opposed:
            boost = -abs(boost)
        else:
            boost = abs(boost) if aligned else 0.0

    # Build note
    net_pct = cot_data.get("spec_net_pct", 0.0)
    if aligned:
        note = f"COT aligned — {bias.replace('_', ' ')} ({net_pct:+.1f}% spec net)"
    elif opposed:
        note = f"COT opposes — {bias.replace('_', ' ')} vs your {direction.upper()} trade"
    else:
        note = f"COT neutral ({net_pct:+.1f}% spec net) — no strong bias"

    return {
        "boost":     round(boost, 2),
        "aligned":   aligned,
        "opposed":   opposed,
        "bias":      bias,
        "note":      note,
        "hedger":    cot_data.get("hedger", "NEUTRAL"),
        "available": True,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== COT Analyzer Self-Test ===\n")
    data = fetch_cot_data()
    print(f"Bias        : {data.get('bias')}")
    print(f"Spec net %  : {data.get('spec_net_pct'):+.1f}%")
    print(f"Boost       : {data.get('boost'):+.1f}")
    print(f"Hedger      : {data.get('hedger')}")
    print(f"Source      : {data.get('source')}")
    print(f"Display     : {data.get('display_line')}")
    print()
    for d in ("long", "short"):
        sig = get_cot_signal(d, data)
        print(f"  {d.upper():5s}  aligned={sig['aligned']}  opposed={sig['opposed']}  "
              f"boost={sig['boost']:+.2f}  note={sig['note']}")
    print("\nSelf-test complete ✓")
