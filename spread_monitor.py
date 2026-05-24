"""
spread_monitor.py — XAUUSD Bid/Ask Spread Monitor

Checks whether the current spread is acceptable before allowing a signal
through the pipeline.

Sources (in priority order):
  1. MT5 live tick data  (mt5.symbol_info_tick)
  2. data/price_cache.json  (ask/bid fields)
  3. Unavailable fallback

Spread thresholds for XAUUSD (dollars):
  ACCEPTABLE : spread <= $1.00
  WARNING    : $1.00 < spread <= $2.00
  BLOCKED    : spread > $2.00

Also BLOCKED during rollover windows (UTC):
  00:00–01:00  and  21:00–24:00
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE  = os.path.join(_BASE_DIR, "data", "price_cache.json")

# ── Thresholds ────────────────────────────────────────────────────────────────
SPREAD_ACCEPTABLE  = 1.00   # <= this → green
SPREAD_WARNING     = 2.00   # <= this → yellow
# > SPREAD_WARNING → blocked

# UTC hours when spreads are structurally wide (market open/close rollover)
_ROLLOVER_RANGES = [(0, 1), (21, 24)]   # (start_hour_inclusive, end_hour_exclusive)


def _is_rollover_window() -> bool:
    """Return True if current UTC time is inside a rollover window."""
    h = datetime.now(timezone.utc).hour
    return any(start <= h < end for start, end in _ROLLOVER_RANGES)


def _load_cache_prices() -> tuple[float | None, float | None]:
    """Read bid/ask from data/price_cache.json. Returns (bid, ask) or (None, None)."""
    try:
        with open(_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        bid = float(data.get("bid") or data.get("price", 0) or 0)
        ask = float(data.get("ask") or 0)
        if bid > 0 and ask > 0:
            return bid, ask
        # If only price available, estimate with a typical spread
        price = float(data.get("price") or 0)
        if price > 0:
            return price - 0.15, price + 0.15   # estimated ±$0.15
    except Exception:
        pass
    return None, None


def _get_mt5_tick(symbol: str) -> tuple[float | None, float | None]:
    """
    Try to read live bid/ask from MetaTrader5.
    Returns (bid, ask) or (None, None) if MT5 is unavailable.
    """
    try:
        import MetaTrader5 as mt5  # type: ignore[import]
        if not mt5.initialize():
            return None, None
        tick = mt5.symbol_info_tick(symbol)
        mt5.shutdown()
        if tick is None:
            return None, None
        return float(tick.bid), float(tick.ask)
    except Exception:
        return None, None


def check_spread(symbol: str = "XAUUSD") -> dict:
    """
    Check the current bid/ask spread for *symbol* and return a verdict dict.

    Returns
    -------
    {
        "spread_usd":     float | None,
        "status":         "acceptable" | "warning" | "blocked" | "unavailable",
        "blocked":        bool,
        "reason":         str,
        "recommendation": str,
        "bid":            float | None,
        "ask":            float | None,
    }
    """
    bid: float | None = None
    ask: float | None = None
    source = "unavailable"

    # Priority 1 — MT5 live tick
    bid, ask = _get_mt5_tick(symbol)
    if bid is not None and ask is not None:
        source = "mt5"

    # Priority 2 — price_cache.json
    if bid is None or ask is None:
        bid, ask = _load_cache_prices()
        if bid is not None and ask is not None:
            source = "cache"

    # No price data at all
    if bid is None or ask is None:
        return {
            "spread_usd":     None,
            "status":         "unavailable",
            "blocked":        False,        # don't hard-block if we can't measure
            "reason":         "Spread unavailable — no MT5 connection and no price cache",
            "recommendation": "Connect MT5 or run setup.py to populate price cache.",
            "bid":            None,
            "ask":            None,
        }

    spread = round(ask - bid, 4)

    # ── Rollover window check (always blocked regardless of spread value) ─────
    if _is_rollover_window():
        h = datetime.now(timezone.utc).hour
        window = "00:00–01:00 UTC" if h < 1 else "21:00–24:00 UTC"
        return {
            "spread_usd":     spread,
            "status":         "blocked",
            "blocked":        True,
            "reason":         f"Market rollover window ({window}) — spreads always spike",
            "recommendation": "Wait until after 01:00 UTC (or trade only after 01:00 UTC).",
            "bid":            bid,
            "ask":            ask,
        }

    # ── Spread threshold classification ──────────────────────────────────────
    if spread <= SPREAD_ACCEPTABLE:
        return {
            "spread_usd":     spread,
            "status":         "acceptable",
            "blocked":        False,
            "reason":         f"Spread ${spread:.2f} — within normal range",
            "recommendation": "Normal conditions — proceed with signal.",
            "bid":            bid,
            "ask":            ask,
        }

    if spread <= SPREAD_WARNING:
        return {
            "spread_usd":     spread,
            "status":         "warning",
            "blocked":        False,
            "reason":         f"Spread ${spread:.2f} — wider than usual",
            "recommendation": "Trade allowed but spread will eat into profit. Consider waiting.",
            "bid":            bid,
            "ask":            ask,
        }

    # > SPREAD_WARNING
    return {
        "spread_usd":     spread,
        "status":         "blocked",
        "blocked":        True,
        "reason":         f"Spread ${spread:.2f} — too wide (max ${SPREAD_WARNING:.2f})",
        "recommendation": "Wait for spread to normalise before entering.",
        "bid":            bid,
        "ask":            ask,
    }
