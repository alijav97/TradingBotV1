"""
intelligence/cot_analyzer.py — CFTC COT Institutional Positioning for TradingBotV2
====================================================================================
Fetches Commitment of Traders (COT) report data and derives an institutional
positioning bias for a given instrument.

Data sources (tried in order):
  1. 24-hour on-disk cache (avoids redundant network calls across sessions)
  2. ``cot_reports`` Python package (pip install cot_reports) — live CFTC data
  3. Direct CFTC CSV download (free, no API key required)
  4. Hardcoded plausible fallback with ``available=False`` to signal degraded mode

Supported symbols and their CFTC futures codes
-----------------------------------------------
  XAUUSD  → Gold futures         (088691)
  WTI     → Crude Oil WTI        (067651)
  BTCUSDT → Bitcoin futures      (133741)
  ETHUSDT → Ether futures        (146021)
  NAS100  → Nasdaq-100 e-mini    (20974+ / 209742)
  GBPJPY  → British Pound futures (096742) — GBP proxy

Usage
-----
    from v2.intelligence.cot_analyzer import get_cot_bias

    result = get_cot_bias("XAUUSD")
    # {
    #   "net_position": 90000,
    #   "bias": "bullish",          # "bullish" | "bearish" | "neutral"
    #   "note": "Speculator net +18.0% of OI — moderately bullish",
    #   "available": True,
    # }
"""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR  = os.path.join(_BASE_DIR, "..", "data")
_CACHE_FILE = os.path.join(_CACHE_DIR, "cot_cache.json")

_CACHE_TTL_HOURS = 24

# CFTC COT report data (legacy futures-only, annual)
_CFTC_DEACOTR_URL = (
    "https://www.cftc.gov/files/dea/history/deacotr_2024.zip"
)
_CFTC_COLUMNS = {
    "market":     "Market and Exchange Names",
    "nc_long":    "Noncommercial Positions-Long (All)",
    "nc_short":   "Noncommercial Positions-Short (All)",
    "comm_long":  "Commercial Positions-Long (All)",
    "comm_short": "Commercial Positions-Short (All)",
    "oi":         "Open Interest (All)",
    "date":       "As of Date in Form YYYY-MM-DD",
}

# CFTC market search strings per instrument (case-insensitive substring match)
_SYMBOL_TO_MARKET: dict[str, str] = {
    "XAUUSD":  "GOLD",
    "WTI":     "CRUDE OIL",
    "BTCUSDT": "BITCOIN",
    "ETHUSDT": "ETHER",
    "NAS100":  "NASDAQ",
    "GBPJPY":  "BRITISH POUND",
}

# Positioning thresholds (spec net as % of total OI)
_BULLISH_PCT          =  5.0
_STRONGLY_BULLISH_PCT = 20.0
_BEARISH_PCT          = -5.0
_STRONGLY_BEARISH_PCT = -20.0

# Hardcoded fallback values per symbol (moderately bullish gold, neutral others)
_FALLBACK_POSITIONS: dict[str, dict[str, float]] = {
    "XAUUSD":  {"nc_long": 180_000, "nc_short":  90_000, "comm_long":  40_000, "comm_short": 200_000, "oi": 500_000},
    "WTI":     {"nc_long": 250_000, "nc_short": 140_000, "comm_long": 100_000, "comm_short": 300_000, "oi": 700_000},
    "BTCUSDT": {"nc_long":   8_000, "nc_short":   6_000, "comm_long":   1_000, "comm_short":   4_000, "oi":  20_000},
    "ETHUSDT": {"nc_long":   5_000, "nc_short":   4_000, "comm_long":     500, "comm_short":   2_000, "oi":  12_000},
    "NAS100":  {"nc_long":  60_000, "nc_short":  55_000, "comm_long":  10_000, "comm_short":  25_000, "oi": 150_000},
    "GBPJPY":  {"nc_long":  35_000, "nc_short":  30_000, "comm_long":  15_000, "comm_short":  20_000, "oi":  90_000},
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(symbol: str) -> dict[str, Any] | None:
    """Return cached COT result for *symbol* if fresher than TTL, else None."""
    try:
        if not os.path.exists(_CACHE_FILE):
            return None
        with open(_CACHE_FILE, "r", encoding="utf-8") as fh:
            store: dict = json.load(fh)
        entry = store.get(symbol)
        if not entry:
            return None
        cached_at_str = entry.get("_cached_at", "")
        if not cached_at_str:
            return None
        cached_at = datetime.fromisoformat(cached_at_str)
        if datetime.now(timezone.utc) - cached_at < timedelta(hours=_CACHE_TTL_HOURS):
            return entry
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.debug("COT cache read failed (%s): %s", symbol, exc)
    return None


def _save_cache(symbol: str, data: dict[str, Any]) -> None:
    """Persist *data* for *symbol* into the shared cache file."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        store: dict = {}
        if os.path.exists(_CACHE_FILE):
            try:
                with open(_CACHE_FILE, "r", encoding="utf-8") as fh:
                    store = json.load(fh)
            except (OSError, json.JSONDecodeError):
                store = {}
        entry = dict(data)
        entry["_cached_at"] = datetime.now(timezone.utc).isoformat()
        store[symbol] = entry
        with open(_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2)
    except OSError as exc:
        logger.warning("COT cache write failed (%s): %s", symbol, exc)


# ---------------------------------------------------------------------------
# Bias derivation
# ---------------------------------------------------------------------------

def _derive_bias(
    nc_long: float,
    nc_short: float,
    oi: float,
    source: str,
    report_date: str,
) -> dict[str, Any]:
    """
    Compute net speculator position and map to a bias string.

    Returns a dict matching the get_cot_bias() schema plus internal
    fields used by the cache.
    """
    spec_net     = nc_long - nc_short
    spec_net_pct = (spec_net / oi * 100.0) if oi else 0.0

    if spec_net_pct >= _STRONGLY_BULLISH_PCT:
        bias = "bullish"
        note = (
            f"Speculator net +{spec_net_pct:.1f}% of OI — strongly bullish "
            f"(source={source}, report={report_date})"
        )
    elif spec_net_pct >= _BULLISH_PCT:
        bias = "bullish"
        note = (
            f"Speculator net +{spec_net_pct:.1f}% of OI — moderately bullish "
            f"(source={source}, report={report_date})"
        )
    elif spec_net_pct <= _STRONGLY_BEARISH_PCT:
        bias = "bearish"
        note = (
            f"Speculator net {spec_net_pct:.1f}% of OI — strongly bearish "
            f"(source={source}, report={report_date})"
        )
    elif spec_net_pct <= _BEARISH_PCT:
        bias = "bearish"
        note = (
            f"Speculator net {spec_net_pct:.1f}% of OI — moderately bearish "
            f"(source={source}, report={report_date})"
        )
    else:
        bias = "neutral"
        note = (
            f"Speculator net {spec_net_pct:+.1f}% of OI — neutral positioning "
            f"(source={source}, report={report_date})"
        )

    return {
        "net_position": round(spec_net),
        "bias":         bias,
        "note":         note,
        "available":    True,
        "_source":      source,
        "_report_date": report_date,
        "_spec_net_pct": round(spec_net_pct, 2),
    }


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _fetch_via_cot_reports_pkg(symbol: str) -> dict[str, Any] | None:
    """
    Attempt to fetch COT data using the ``cot_reports`` package.
    Returns a bias dict or None if the package is unavailable or the
    market cannot be found.
    """
    market_kw = _SYMBOL_TO_MARKET.get(symbol)
    if not market_kw:
        return None
    try:
        import cot_reports as cot  # type: ignore[import]  # pip install cot_reports
        df = cot.cot_year(cot_report_type="legacy_fut_only")
        mask    = df[_CFTC_COLUMNS["market"]].str.upper().str.contains(market_kw, na=False)
        matched = df[mask]
        if matched.empty:
            logger.debug("cot_reports: no rows matched market_kw=%s for %s", market_kw, symbol)
            return None
        row = matched.iloc[-1]
        result = _derive_bias(
            nc_long=float(row.get(_CFTC_COLUMNS["nc_long"],    0)),
            nc_short=float(row.get(_CFTC_COLUMNS["nc_short"],  0)),
            oi=float(row.get(_CFTC_COLUMNS["oi"],              1)),
            source="cot_reports_live",
            report_date=str(row.get(_CFTC_COLUMNS["date"], "unknown")),
        )
        logger.info("COT (%s): loaded from cot_reports package", symbol)
        return result
    except ImportError:
        logger.debug("cot_reports package not installed — skipping")
        return None
    except (KeyError, ValueError, IndexError) as exc:
        logger.warning("cot_reports fetch error for %s: %s", symbol, exc)
        return None


def _fetch_via_cftc_direct(symbol: str) -> dict[str, Any] | None:
    """
    Download the CFTC annual legacy COT CSV directly, parse it with pandas,
    and return a bias dict, or None on any failure.
    """
    market_kw = _SYMBOL_TO_MARKET.get(symbol)
    if not market_kw:
        return None
    try:
        import requests  # type: ignore[import]
        import pandas as pd  # type: ignore[import]

        resp = requests.get(_CFTC_DEACOTR_URL, timeout=20)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return None
            with z.open(csv_names[0]) as csv_fh:
                df = pd.read_csv(csv_fh, low_memory=False)

        col_market = _CFTC_COLUMNS["market"]
        if col_market not in df.columns:
            logger.debug("CFTC CSV: expected column '%s' not found", col_market)
            return None

        mask    = df[col_market].str.upper().str.contains(market_kw, na=False)
        matched = df[mask]
        if matched.empty:
            return None

        row = matched.iloc[-1]
        result = _derive_bias(
            nc_long=float(row.get(_CFTC_COLUMNS["nc_long"],   0)),
            nc_short=float(row.get(_CFTC_COLUMNS["nc_short"], 0)),
            oi=float(row.get(_CFTC_COLUMNS["oi"],             1)),
            source="cftc_direct",
            report_date=str(row.get(_CFTC_COLUMNS["date"], "unknown")),
        )
        logger.info("COT (%s): loaded from CFTC direct CSV", symbol)
        return result
    except ImportError as exc:
        logger.debug("requests/pandas not available for CFTC direct fetch: %s", exc)
        return None
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as exc:
        logger.warning("CFTC direct fetch error for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cot_bias(symbol: str) -> dict[str, Any]:
    """
    Return institutional positioning bias for *symbol*.

    Fetch order:
      1. On-disk cache (24-hour TTL)
      2. ``cot_reports`` Python package
      3. Direct CFTC CSV download
      4. Hardcoded fallback (``available=False``)

    Parameters
    ----------
    symbol : str
        One of: XAUUSD, GBPJPY, WTI, NAS100, BTCUSDT, ETHUSDT.

    Returns
    -------
    dict with keys:
        net_position  int   — speculator net contracts (longs minus shorts)
        bias          str   — "bullish" | "bearish" | "neutral"
        note          str   — human-readable explanation
        available     bool  — False when only the hardcoded fallback is used
    """
    symbol_upper = symbol.upper()

    # 1. Cache -----------------------------------------------------------------
    cached = _load_cache(symbol_upper)
    if cached is not None:
        logger.debug("COT (%s): serving from cache", symbol_upper)
        return {
            "net_position": cached.get("net_position", 0),
            "bias":         cached.get("bias", "neutral"),
            "note":         cached.get("note", ""),
            "available":    cached.get("available", True),
        }

    # 2. cot_reports package ---------------------------------------------------
    result = _fetch_via_cot_reports_pkg(symbol_upper)

    # 3. Direct CFTC CSV -------------------------------------------------------
    if result is None:
        result = _fetch_via_cftc_direct(symbol_upper)

    # 4. Hardcoded fallback ----------------------------------------------------
    if result is None:
        fallback_pos = _FALLBACK_POSITIONS.get(symbol_upper, {
            "nc_long": 100_000, "nc_short": 100_000, "oi": 300_000,
        })
        result = _derive_bias(
            nc_long=fallback_pos["nc_long"],
            nc_short=fallback_pos["nc_short"],
            oi=fallback_pos["oi"],
            source="hardcoded_fallback",
            report_date="estimate",
        )
        # Override available flag — callers should treat this as unreliable
        result["available"] = False
        result["note"] = (
            f"COT data unavailable for {symbol_upper} — using hardcoded estimate; "
            "install 'cot_reports' for live data"
        )
        logger.warning("COT (%s): all live sources failed; using hardcoded fallback", symbol_upper)

    # Persist successful live result ------------------------------------------
    if result.get("available"):
        _save_cache(symbol_upper, result)

    return {
        "net_position": result.get("net_position", 0),
        "bias":         result.get("bias", "neutral"),
        "note":         result.get("note", ""),
        "available":    result.get("available", False),
    }
