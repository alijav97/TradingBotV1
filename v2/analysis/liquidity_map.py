"""
analysis/liquidity_map.py — Liquidity Heatmap Engine for TradingBotV2.

Identifies where stop-loss clusters and volume concentration sit relative to
current price, then predicts the most likely price move based on which cluster
is larger.  Works on any OHLCV instrument — not tied to a specific asset.

Components:
  • Swing high / low detection  (±_SWING_LOOKBACK bar comparison)
  • Stop-loss cluster zones      (swing ± _SL_CLUSTER_PCT)
  • Volume profile               (adaptive bucket size)
  • Point of Control (POC)       — highest-volume bucket midpoint
  • Value Area (70 % of volume)
  • Void zones                   — buckets with < _VOID_THRESHOLD × mean volume
  • Likely-move prediction

Usage:
    from v2.analysis.liquidity_map import build_liquidity_map, format_liquidity_map
    liq = build_liquidity_map(df, current_price)
    print(format_liquidity_map(liq, current_price))
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_SWING_LOOKBACK  = 2        # bars each side for swing detection
_SL_CLUSTER_PCT  = 0.001    # ±0.1% around swing → stop cluster zone
_MERGE_PCT       = 0.005    # merge nearby swings within 0.5%
_VALUE_AREA_PCT  = 0.70     # 70% of volume = value area
_VOID_THRESHOLD  = 0.30     # < 30% of mean vol = void zone
# Bucket size adapts to price: bucket = price * _VOL_BUCKET_PCT
_VOL_BUCKET_PCT  = 0.001    # 0.1% of current price per bucket


# ── Internal helpers ───────────────────────────────────────────────────────────

def _detect_swings(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    Detect swing highs and swing lows using ±_SWING_LOOKBACK bar comparison.

    Returns
    -------
    swing_highs : list of prices (from high column)
    swing_lows  : list of prices (from low column)
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(highs)
    lb    = _SWING_LOOKBACK

    swing_highs: list[float] = []
    swing_lows:  list[float] = []

    for i in range(lb, n - lb):
        window_h = highs[i - lb: i + lb + 1]
        if highs[i] == np.max(window_h):
            swing_highs.append(float(highs[i]))

        window_l = lows[i - lb: i + lb + 1]
        if lows[i] == np.min(window_l):
            swing_lows.append(float(lows[i]))

    return swing_highs, swing_lows


def _build_sl_clusters(
    swing_highs:   list[float],
    swing_lows:    list[float],
    current_price: float,
) -> tuple[list[dict], list[dict]]:
    """
    Build stop-loss cluster zones from swing levels.

    Each cluster dict:
        price        : float  — representative swing level
        zone_low     : float
        zone_high    : float
        side         : "above" | "below"
        count        : int    — how many swings merged into this cluster
        distance_abs : float  — absolute distance from current price

    Nearby swings within _MERGE_PCT are merged together.

    Returns (clusters_above, clusters_below), each sorted nearest-first.
    """
    def _cluster_levels(levels: list[float], side: str) -> list[dict]:
        if not levels:
            return []
        clusters: list[dict] = []
        for lvl in sorted(set(levels)):
            merged = False
            for c in clusters:
                if abs(lvl - c["price"]) / max(c["price"], 1e-9) < _MERGE_PCT:
                    # running weighted average
                    new_count  = c["count"] + 1
                    c["price"] = round(
                        (c["price"] * c["count"] + lvl) / new_count, 5
                    )
                    c["count"]      = new_count
                    c["zone_low"]   = round(c["price"] * (1 - _SL_CLUSTER_PCT), 5)
                    c["zone_high"]  = round(c["price"] * (1 + _SL_CLUSTER_PCT), 5)
                    merged = True
                    break
            if not merged:
                clusters.append({
                    "price":        round(lvl, 5),
                    "zone_low":     round(lvl * (1 - _SL_CLUSTER_PCT), 5),
                    "zone_high":    round(lvl * (1 + _SL_CLUSTER_PCT), 5),
                    "side":         side,
                    "count":        1,
                    "distance_abs": round(abs(lvl - current_price), 5),
                })
        # Refresh distances after all merges
        for c in clusters:
            c["distance_abs"] = round(abs(c["price"] - current_price), 5)
        return sorted(clusters, key=lambda x: x["distance_abs"])

    above_swings = [s for s in swing_highs if s > current_price]
    below_swings = [s for s in swing_lows  if s < current_price]

    return _cluster_levels(above_swings, "above"), _cluster_levels(below_swings, "below")


def _choose_bucket_size(current_price: float,
                        price_min: float,
                        price_max: float) -> float:
    """
    Choose a volume-profile bucket size that is proportional to the instrument
    price so the profile resolution is meaningful for any asset.
    """
    raw    = current_price * _VOL_BUCKET_PCT
    span   = price_max - price_min
    # Ensure at least 20 buckets, at most 500
    bucket = max(raw, span / 500)
    bucket = min(bucket, span / 20) if span > 0 else raw
    return max(bucket, 1e-9)


def _build_volume_profile(df: pd.DataFrame,
                          current_price: float) -> dict[str, Any]:
    """
    Build a volume profile using adaptive price buckets.

    Returns
    -------
    poc        : float  — price of highest-volume bucket midpoint
    va_high    : float  — top of value area
    va_low     : float  — bottom of value area
    buckets    : list[dict] with keys: price, volume, pct
    voids      : list[dict] — ranges with < _VOID_THRESHOLD × mean volume
    """
    _empty = {"poc": 0.0, "va_high": 0.0, "va_low": 0.0, "buckets": [], "voids": []}

    if "volume" not in df.columns:
        return _empty
    if "high" not in df.columns or "low" not in df.columns:
        return _empty

    prices  = (df["high"].values + df["low"].values) / 2.0
    volumes = df["volume"].values.astype(float)

    price_min = float(np.min(df["low"].values))
    price_max = float(np.max(df["high"].values))
    if price_min >= price_max:
        return _empty

    bucket_size  = _choose_bucket_size(current_price, price_min, price_max)
    price_floor  = np.floor(price_min  / bucket_size) * bucket_size
    price_ceil   = np.ceil( price_max  / bucket_size) * bucket_size
    bucket_edges = np.arange(price_floor, price_ceil + bucket_size, bucket_size)
    bucket_vols  = np.zeros(max(len(bucket_edges) - 1, 1), dtype=float)

    for p, v in zip(prices, volumes):
        idx = int((p - price_floor) / bucket_size)
        if 0 <= idx < len(bucket_vols):
            bucket_vols[idx] += v

    total_vol = float(bucket_vols.sum())
    if total_vol == 0:
        return _empty

    # POC
    poc_idx = int(np.argmax(bucket_vols))
    poc     = round(float(bucket_edges[poc_idx]) + bucket_size / 2.0, 5)

    # Value Area — expand from POC until 70 % volume accumulated
    va_vol_target = total_vol * _VALUE_AREA_PCT
    accumulated   = bucket_vols[poc_idx]
    lo_idx        = poc_idx
    hi_idx        = poc_idx

    while accumulated < va_vol_target:
        expand_down = lo_idx - 1 >= 0
        expand_up   = hi_idx + 1 < len(bucket_vols)
        if not expand_down and not expand_up:
            break
        vol_down = bucket_vols[lo_idx - 1] if expand_down else -1.0
        vol_up   = bucket_vols[hi_idx + 1] if expand_up   else -1.0
        if vol_down >= vol_up:
            lo_idx      -= 1
            accumulated += bucket_vols[lo_idx]
        else:
            hi_idx      += 1
            accumulated += bucket_vols[hi_idx]

    va_low  = round(float(bucket_edges[lo_idx]), 5)
    va_high = round(float(bucket_edges[min(hi_idx + 1, len(bucket_edges) - 1)]), 5)

    # Bucket list + voids
    nonzero_vols = bucket_vols[bucket_vols > 0]
    mean_vol     = float(np.mean(nonzero_vols)) if len(nonzero_vols) > 0 else 1.0
    buckets: list[dict] = []
    voids:   list[dict] = []

    for v, lo, hi in zip(bucket_vols, bucket_edges[:-1], bucket_edges[1:]):
        mid = round(float(lo) + bucket_size / 2.0, 5)
        pct = round(float(v) / total_vol * 100.0, 1) if total_vol else 0.0
        buckets.append({"price": mid, "volume": round(float(v)), "pct": pct})
        if 0 < v < mean_vol * _VOID_THRESHOLD:
            voids.append({
                "low":    round(float(lo), 5),
                "high":   round(float(hi), 5),
                "mid":    mid,
                "volume": round(float(v)),
            })

    return {
        "poc":     poc,
        "va_high": va_high,
        "va_low":  va_low,
        "buckets": buckets,
        "voids":   voids,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def build_liquidity_map(df: pd.DataFrame, current_price: float) -> dict[str, Any]:
    """
    Build the complete liquidity map for the given OHLCV DataFrame.

    Parameters
    ----------
    df            : pd.DataFrame — OHLCV with columns: high, low, close, volume
    current_price : float        — most recent price (df["close"].iloc[-1])

    Returns
    -------
    available       : bool
    clusters_above  : list[dict]  — stop clusters above current price, nearest-first
    clusters_below  : list[dict]  — stop clusters below current price, nearest-first
    poc             : float       — point of control
    va_high         : float       — value area high
    va_low          : float       — value area low
    voids           : list[dict]  — volume void zones
    likely_move     : str         — "UP" | "DOWN" | "NEUTRAL"
    likely_reason   : str         — plain-English explanation
    largest_cluster : dict | None — single largest stop cluster by count
    current_price   : float
    """
    _empty_result: dict[str, Any] = {
        "available":       False,
        "clusters_above":  [],
        "clusters_below":  [],
        "poc":             0.0,
        "va_high":         0.0,
        "va_low":          0.0,
        "voids":           [],
        "likely_move":     "NEUTRAL",
        "likely_reason":   "Insufficient data",
        "largest_cluster": None,
        "current_price":   current_price,
    }

    if df is None or len(df) < 10:
        return _empty_result

    # Normalise column names
    df_work = df.copy()
    df_work.columns = [c.lower().strip() for c in df_work.columns]

    required = {"high", "low", "close"}
    missing  = required - set(df_work.columns)
    if missing:
        logger.warning("liquidity_map: missing columns %s", missing)
        return _empty_result

    try:
        swing_highs, swing_lows = _detect_swings(df_work)
        clusters_above, clusters_below = _build_sl_clusters(
            swing_highs, swing_lows, current_price
        )
        vol_profile = _build_volume_profile(df_work, current_price)

        # Likely move: price tends to hunt the LARGER stop pool
        top_above = clusters_above[0]["count"] if clusters_above else 0
        top_below = clusters_below[0]["count"] if clusters_below else 0

        all_clusters    = clusters_above + clusters_below
        largest_cluster = (max(all_clusters, key=lambda c: c["count"])
                           if all_clusters else None)

        if top_above > top_below:
            likely_move   = "UP"
            likely_reason = (
                f"Largest stop cluster ABOVE at {clusters_above[0]['price']:,.5g} "
                f"({clusters_above[0]['count']} swing levels) — price likely hunts BSL"
            )
        elif top_below > top_above:
            likely_move   = "DOWN"
            likely_reason = (
                f"Largest stop cluster BELOW at {clusters_below[0]['price']:,.5g} "
                f"({clusters_below[0]['count']} swing levels) — price likely hunts SSL"
            )
        else:
            likely_move   = "NEUTRAL"
            likely_reason = "Clusters above and below are equal — no directional liquidity bias"

        logger.debug(
            "liquidity_map: %d clusters above, %d below, POC=%.5g, move=%s",
            len(clusters_above), len(clusters_below), vol_profile["poc"], likely_move
        )

        return {
            "available":       True,
            "clusters_above":  clusters_above[:5],
            "clusters_below":  clusters_below[:5],
            "poc":             vol_profile["poc"],
            "va_high":         vol_profile["va_high"],
            "va_low":          vol_profile["va_low"],
            "voids":           vol_profile["voids"][:5],
            "likely_move":     likely_move,
            "likely_reason":   likely_reason,
            "largest_cluster": largest_cluster,
            "current_price":   current_price,
        }

    except (KeyError, ValueError, IndexError) as exc:
        logger.error("liquidity_map: data error: %s", exc)
        _empty_result["likely_reason"] = f"Data error: {exc}"
        return _empty_result
    except Exception as exc:
        logger.exception("liquidity_map: unexpected error: %s", exc)
        _empty_result["likely_reason"] = f"Error: {exc}"
        return _empty_result


def format_liquidity_map(liq: dict, current_price: float) -> str:
    """
    Format the liquidity map dict as a readable string block.

    Example output:
        STOPS ABOVE (buy-side liquidity)
          2655.00  (+12.00)  [2 swings]
        ──────── current price: 2643.00 ────────
        STOPS BELOW (sell-side liquidity)
          2631.00  (−12.00)  [3 swings]
        POC:  2640.00  |  VA:  2635.00 – 2648.00
        Likely move: UP — hunts BSL above
    """
    if not liq.get("available"):
        return "Liquidity map unavailable."

    lines: list[str] = []

    lines.append("STOPS ABOVE (buy-side liquidity)")
    for c in liq.get("clusters_above", [])[:3]:
        dist = f"+{c['distance_abs']:,.5g}"
        lines.append(
            f"  {c['price']:>12,.5g}  ({dist})  "
            f"[{c['count']} swing{'s' if c['count'] != 1 else ''}]"
        )
    if not liq.get("clusters_above"):
        lines.append("  None detected in range")

    lines.append(f"  ─── current price: {current_price:,.5g} ───")

    lines.append("STOPS BELOW (sell-side liquidity)")
    for c in liq.get("clusters_below", [])[:3]:
        dist = f"-{c['distance_abs']:,.5g}"
        lines.append(
            f"  {c['price']:>12,.5g}  ({dist})  "
            f"[{c['count']} swing{'s' if c['count'] != 1 else ''}]"
        )
    if not liq.get("clusters_below"):
        lines.append("  None detected in range")

    poc    = liq.get("poc",    0.0)
    va_low = liq.get("va_low", 0.0)
    va_hi  = liq.get("va_high", 0.0)
    if poc:
        lines.append(f"POC: {poc:,.5g}  |  VA: {va_low:,.5g} – {va_hi:,.5g}")

    voids = liq.get("voids", [])
    if voids:
        void_str = ", ".join(f"{v['mid']:,.5g}" for v in voids[:3])
        lines.append(f"Vol voids: {void_str}")

    move   = liq.get("likely_move",   "NEUTRAL")
    reason = liq.get("likely_reason", "")
    arrow  = "UP" if move == "UP" else ("DOWN" if move == "DOWN" else "NEUTRAL")
    lines.append(f"Likely move: {arrow} — {reason}")

    return "\n".join(lines)
