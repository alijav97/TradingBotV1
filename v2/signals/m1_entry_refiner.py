"""
signals/m1_entry_refiner.py — ICT M1 IFVG precision entry for XAUUSD.

After the H1 confluence engine fires a signal for XAUUSD, this module
drops to the M1 timeframe to find an Inverse Fair Value Gap (IFVG) for a
tighter, higher-probability entry.

ICT Charter Model — Step 4:
  "1Min Institutional Entry Only With IFVG's"

Logic:
  1. Fetch the last 30 M1 bars for XAUUSD.
  2. Scan for active, unfilled FVGs aligned with the trade direction.
     - Long  signal → bullish FVG (gap above, price may return to fill)
     - Short signal → bearish FVG (gap below, price may return to fill)
  3. If a qualifying FVG is found within MAX_PROXIMITY_PCT of current price:
       entry = FVG midpoint
       sl    = FVG bottom (long) or FVG top (short)  ← tighter than H1 SL
       tp1, tp2 recalculated from new entry/SL
  4. If no qualifying FVG → return original H1 signal unchanged (fallback).

Only applied to XAUUSD.  All other instruments pass through unchanged.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2.connectors.unified_data import DataFeed

from v2.analysis.smart_money import SmartMoneyAnalyzer
from v2.risk.position_sizer import calculate_tp_prices

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Only apply M1 refinement to this instrument
REFINE_SYMBOL = "XAUUSD"

# Maximum distance (as % of price) between current price and FVG midpoint.
# At $2 300 gold this is ~$11.50.  Keeps us from latching onto stale far-away gaps.
MAX_PROXIMITY_PCT = 0.005   # 0.5%

# Minimum FVG gap size as % of price — filters out tiny noise gaps
MIN_GAP_PCT = 0.0002        # 0.02% (~$0.46 at $2 300)

# Number of M1 bars to fetch (30 minutes of context)
M1_BARS = 30

# SL buffer: add this many pips beyond FVG boundary as the stop loss
# 0.1 pip_size per pip, 3 pips buffer = 0.3 for XAUUSD
SL_BUFFER_PIPS = 3
XAUUSD_PIP_SIZE = 0.1       # from instrument_config


# ── Public API ────────────────────────────────────────────────────────────────

def refine_entry(signal: dict, feed: "DataFeed") -> dict:
    """
    Attempt to refine a XAUUSD H1 signal to M1 IFVG precision.

    Parameters
    ----------
    signal : dict
        The signal dict produced by auto_trader._scan_one() — must contain
        symbol, direction, entry_price, stop_loss, tp1_price, tp2_price.
    feed : DataFeed
        Connected data feed for fetching M1 bars.

    Returns
    -------
    dict
        The (possibly refined) signal dict.  If refinement succeeds,
        adds ``m1_refined=True`` and ``m1_fvg`` fields.
        If no qualifying FVG found, adds ``m1_refined=False`` and
        returns signal unchanged.
    """
    symbol    = signal.get("symbol", "")
    direction = signal.get("direction", "").lower()

    if symbol != REFINE_SYMBOL:
        return signal

    entry_h1 = float(signal.get("entry_price") or 0)
    sl_h1    = float(signal.get("stop_loss") or 0)
    if entry_h1 <= 0 or sl_h1 <= 0:
        return signal

    # ── 1. Fetch M1 bars ──────────────────────────────────────────────────────
    try:
        df_m1 = feed.get_ohlcv(symbol, "M1", M1_BARS)
    except Exception as exc:
        logger.debug("M1 fetch failed for %s: %s", symbol, exc)
        signal["m1_refined"] = False
        return signal

    if df_m1 is None or df_m1.empty or len(df_m1) < 5:
        logger.debug("Insufficient M1 data for %s — skipping IFVG refinement", symbol)
        signal["m1_refined"] = False
        return signal

    current_price = float(df_m1["close"].iloc[-1])

    # ── 2. Find FVGs on M1 ───────────────────────────────────────────────────
    try:
        sma  = SmartMoneyAnalyzer()
        fvgs = sma.find_fair_value_gaps(df_m1)
    except Exception as exc:
        logger.debug("FVG detection error for %s M1: %s", symbol, exc)
        signal["m1_refined"] = False
        return signal

    # ── 3. Filter to qualifying IFVGs ────────────────────────────────────────
    target_type = "bullish" if direction in ("long", "buy") else "bearish"
    candidates  = []

    for fvg in fvgs:
        if fvg.get("filled"):
            continue  # gap already mitigated — not an entry zone
        if fvg.get("fvg_type") != target_type:
            continue

        mid = float(fvg.get("fvg_midpoint", 0))
        if mid <= 0:
            continue

        # Gap size filter
        gap_size = abs(float(fvg["fvg_top"]) - float(fvg["fvg_bottom"]))
        if gap_size < current_price * MIN_GAP_PCT:
            continue

        # Proximity filter
        proximity = abs(current_price - mid) / current_price
        if proximity > MAX_PROXIMITY_PCT:
            continue

        # Directional sanity: for long, FVG midpoint should be AT or BELOW entry
        # (we're looking for support zones to long from, not resistance above)
        if direction in ("long", "buy") and mid > entry_h1 * 1.001:
            continue
        if direction in ("short", "sell") and mid < entry_h1 * 0.999:
            continue

        candidates.append((proximity, fvg))

    if not candidates:
        logger.debug(
            "%s: no qualifying M1 %s IFVG found — using H1 entry", symbol, target_type
        )
        signal["m1_refined"] = False
        return signal

    # Pick the closest FVG to current price
    candidates.sort(key=lambda x: x[0])
    _, best_fvg = candidates[0]

    fvg_top    = float(best_fvg["fvg_top"])
    fvg_bottom = float(best_fvg["fvg_bottom"])
    fvg_mid    = float(best_fvg["fvg_midpoint"])

    # ── 4. Build refined entry / SL ──────────────────────────────────────────
    buf = SL_BUFFER_PIPS * XAUUSD_PIP_SIZE

    if direction in ("long", "buy"):
        new_entry = fvg_mid
        new_sl    = fvg_bottom - buf   # stop below FVG bottom
    else:
        new_entry = fvg_mid
        new_sl    = fvg_top + buf      # stop above FVG top

    # Sanity check: new SL must be meaningfully inside the H1 SL (not wider)
    sl_h1_dist = abs(entry_h1 - sl_h1)
    new_sl_dist = abs(new_entry - new_sl)
    if new_sl_dist >= sl_h1_dist:
        logger.debug(
            "%s: M1 IFVG SL (%.2f) not tighter than H1 SL (%.2f) — using H1 entry",
            symbol, new_sl_dist, sl_h1_dist
        )
        signal["m1_refined"] = False
        return signal

    # ── 5. Recalculate TP1 / TP2 from refined entry/SL ───────────────────────
    new_tp1, new_tp2 = calculate_tp_prices(new_entry, new_sl, direction)

    # ── 6. Apply to signal dict ───────────────────────────────────────────────
    refined = signal.copy()
    refined.update({
        "entry_price":  round(new_entry, 2),
        "stop_loss":    round(new_sl, 2),
        "tp1_price":    round(new_tp1, 2),
        "tp2_price":    round(new_tp2, 2),
        "m1_refined":   True,
        "m1_fvg": {
            "fvg_type":    best_fvg.get("fvg_type"),
            "fvg_top":     round(fvg_top, 2),
            "fvg_bottom":  round(fvg_bottom, 2),
            "fvg_midpoint": round(fvg_mid, 2),
            "h1_entry":    round(entry_h1, 2),
            "h1_sl":       round(sl_h1, 2),
            "sl_improvement_pct": round(
                (1 - new_sl_dist / sl_h1_dist) * 100, 1
            ) if sl_h1_dist > 0 else 0,
        },
    })

    logger.info(
        "%s M1 IFVG entry refined: entry %.2f→%.2f  SL %.2f→%.2f  (%.0f%% tighter SL)",
        symbol,
        entry_h1, new_entry,
        sl_h1,    new_sl,
        refined["m1_fvg"]["sl_improvement_pct"],
    )

    return refined
