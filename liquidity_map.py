"""
liquidity_map.py — Liquidity Heatmap Engine
─────────────────────────────────────────────
Identifies where stop-loss clusters and volume concentration sit
relative to current price, then predicts the most likely price move
based on which cluster is larger.

Components built:
  • Swing high / low detection (±2 bar lookback)
  • Stop-loss cluster zones (swing ± 0.1%)
  • Volume profile in $5 price buckets
  • Point of Control (POC) — highest-volume bucket
  • Value Area (70% of total volume)
  • Void zones — buckets with < 30% of mean volume
  • Likely move prediction

Usage:
    from liquidity_map import build_liquidity_map, format_liquidity_map
    liq = build_liquidity_map(df, current_price)
    print(format_liquidity_map(liq, current_price))
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────
_SWING_LOOKBACK  = 2        # bars each side for swing detection
_SL_CLUSTER_PCT  = 0.001    # ±0.1% around swing → stop cluster zone
_VOL_BUCKET_SIZE = 5.0      # $5 buckets for volume profile
_VALUE_AREA_PCT  = 0.70     # 70% of volume = value area
_VOID_THRESHOLD  = 0.30     # < 30% of mean vol = void zone


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_swings(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    Detect swing highs and swing lows using ±lookback bar comparison.

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
        window_h = highs[i - lb : i + lb + 1]
        if highs[i] == np.max(window_h):
            swing_highs.append(float(highs[i]))

        window_l = lows[i - lb : i + lb + 1]
        if lows[i] == np.min(window_l):
            swing_lows.append(float(lows[i]))

    return swing_highs, swing_lows


def _build_sl_clusters(
    swing_highs: list[float],
    swing_lows:  list[float],
    current_price: float,
) -> tuple[list[dict], list[dict]]:
    """
    Build stop-loss cluster zones from swing levels.

    Each cluster is a dict:
        price       : float  — the swing level itself
        zone_low    : float
        zone_high   : float
        side        : "above" | "below"
        count       : int    — how many swings within merge distance

    Nearby swings within 0.5% of each other are merged.
    """
    MERGE_PCT = 0.005

    def _cluster_levels(levels: list[float], side: str) -> list[dict]:
        if not levels:
            return []
        sorted_lvls = sorted(set(levels))
        clusters: list[dict] = []
        for lvl in sorted_lvls:
            merged = False
            for c in clusters:
                if abs(lvl - c["price"]) / c["price"] < MERGE_PCT:
                    # merge into existing cluster
                    c["count"] += 1
                    c["price"]     = round((c["price"] * c["count"] + lvl) / (c["count"] + 1), 2)
                    c["zone_low"]  = round(c["price"] * (1 - _SL_CLUSTER_PCT), 2)
                    c["zone_high"] = round(c["price"] * (1 + _SL_CLUSTER_PCT), 2)
                    merged = True
                    break
            if not merged:
                clusters.append({
                    "price":      round(lvl, 2),
                    "zone_low":   round(lvl * (1 - _SL_CLUSTER_PCT), 2),
                    "zone_high":  round(lvl * (1 + _SL_CLUSTER_PCT), 2),
                    "side":       side,
                    "count":      1,
                    "distance_usd": round(abs(lvl - current_price), 2),
                })
        # Update distances
        for c in clusters:
            c["distance_usd"] = round(abs(c["price"] - current_price), 2)
        return sorted(clusters, key=lambda x: x["distance_usd"])

    above_swings = [s for s in swing_highs if s > current_price]
    below_swings = [s for s in swing_lows  if s < current_price]

    clusters_above = _cluster_levels(above_swings, "above")
    clusters_below = _cluster_levels(below_swings, "below")

    return clusters_above, clusters_below


def _build_volume_profile(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build a volume profile in $5 price buckets.

    Returns
    -------
    poc        : float  — price of highest-volume bucket
    va_high    : float  — top of 70% value area
    va_low     : float  — bottom of 70% value area
    buckets    : list[dict] with keys price, volume, pct
    voids      : list[dict] — price ranges with < 30% mean volume
    """
    if "volume" not in df.columns:
        return {"poc": 0.0, "va_high": 0.0, "va_low": 0.0, "buckets": [], "voids": []}

    prices  = (df["high"].values + df["low"].values) / 2
    volumes = df["volume"].values.astype(float)

    price_min = np.floor(np.min(df["low"].values)  / _VOL_BUCKET_SIZE) * _VOL_BUCKET_SIZE
    price_max = np.ceil( np.max(df["high"].values) / _VOL_BUCKET_SIZE) * _VOL_BUCKET_SIZE

    bucket_edges = np.arange(price_min, price_max + _VOL_BUCKET_SIZE, _VOL_BUCKET_SIZE)
    bucket_vols  = np.zeros(len(bucket_edges) - 1, dtype=float)

    for p, v in zip(prices, volumes):
        idx = int((p - price_min) / _VOL_BUCKET_SIZE)
        if 0 <= idx < len(bucket_vols):
            bucket_vols[idx] += v

    total_vol = bucket_vols.sum()
    if total_vol == 0:
        return {"poc": 0.0, "va_high": 0.0, "va_low": 0.0, "buckets": [], "voids": []}

    # POC — highest-volume bucket midpoint
    poc_idx  = int(np.argmax(bucket_vols))
    poc      = round(bucket_edges[poc_idx] + _VOL_BUCKET_SIZE / 2, 2)

    # Value Area — 70% of total volume centred on POC
    va_vol_target = total_vol * _VALUE_AREA_PCT
    va_indices    = [poc_idx]
    accumulated   = bucket_vols[poc_idx]
    lo_idx        = poc_idx
    hi_idx        = poc_idx

    while accumulated < va_vol_target:
        expand_down = (lo_idx - 1 >= 0)
        expand_up   = (hi_idx + 1 < len(bucket_vols))
        if not expand_down and not expand_up:
            break
        vol_down = bucket_vols[lo_idx - 1] if expand_down else -1
        vol_up   = bucket_vols[hi_idx + 1] if expand_up   else -1
        if vol_down >= vol_up:
            lo_idx   -= 1
            accumulated += bucket_vols[lo_idx]
        else:
            hi_idx   += 1
            accumulated += bucket_vols[hi_idx]

    va_low  = round(bucket_edges[lo_idx], 2)
    va_high = round(bucket_edges[hi_idx + 1], 2)

    # Build bucket list
    mean_vol = float(np.mean(bucket_vols[bucket_vols > 0])) if np.any(bucket_vols > 0) else 1.0
    buckets  = []
    voids    = []

    for i, (v, lo, hi) in enumerate(
        zip(bucket_vols, bucket_edges[:-1], bucket_edges[1:])
    ):
        mid = round(lo + _VOL_BUCKET_SIZE / 2, 2)
        pct = round(v / total_vol * 100, 1) if total_vol else 0.0
        buckets.append({"price": mid, "volume": round(float(v)), "pct": pct})
        if v > 0 and v < mean_vol * _VOID_THRESHOLD:
            voids.append({"low": round(float(lo), 2), "high": round(float(hi), 2),
                          "mid": mid, "volume": round(float(v))})

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
    Build the complete liquidity map for the given OHLCV dataframe.

    Parameters
    ----------
    df            : pd.DataFrame — enriched OHLCV with columns high, low, close, volume
    current_price : float        — most recent price (df["close"].iloc[-1])

    Returns
    -------
    clusters_above  : list[dict]  — stop clusters above current price, nearest first
    clusters_below  : list[dict]  — stop clusters below current price, nearest first
    poc             : float       — point of control
    va_high         : float       — value area high
    va_low          : float       — value area low
    voids           : list[dict]  — volume void zones
    likely_move     : str         — "UP" | "DOWN" | "NEUTRAL"
    likely_reason   : str         — plain English explanation
    largest_cluster : dict | None — the single largest stop cluster by count
    available       : bool
    """
    if df is None or len(df) < 10:
        return {"available": False, "clusters_above": [], "clusters_below": [],
                "poc": 0.0, "va_high": 0.0, "va_low": 0.0, "voids": [],
                "likely_move": "NEUTRAL", "likely_reason": "Insufficient data",
                "largest_cluster": None}

    try:
        swing_highs, swing_lows = _detect_swings(df)
        clusters_above, clusters_below = _build_sl_clusters(
            swing_highs, swing_lows, current_price
        )
        vol_profile = _build_volume_profile(df)

        # Likely move: price hunts the LARGER stop pool
        # Largest cluster above → price likely moves UP to grab those stops
        # Largest cluster below → price likely moves DOWN
        top_above = clusters_above[0]["count"] if clusters_above else 0
        top_below = clusters_below[0]["count"] if clusters_below else 0

        all_clusters = clusters_above + clusters_below
        largest_cluster = max(all_clusters, key=lambda c: c["count"]) if all_clusters else None

        if top_above > top_below:
            likely_move   = "UP"
            likely_reason = (
                f"Largest stop cluster ABOVE at ${clusters_above[0]['price']:,.2f} "
                f"({clusters_above[0]['count']} swing levels) — price likely hunts BSL"
            )
        elif top_below > top_above:
            likely_move   = "DOWN"
            likely_reason = (
                f"Largest stop cluster BELOW at ${clusters_below[0]['price']:,.2f} "
                f"({clusters_below[0]['count']} swing levels) — price likely hunts SSL"
            )
        else:
            likely_move   = "NEUTRAL"
            likely_reason = "Clusters above and below are equal — no directional bias from liquidity"

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

    except Exception as e:
        return {
            "available":       False,
            "clusters_above":  [],
            "clusters_below":  [],
            "poc":             0.0,
            "va_high":         0.0,
            "va_low":          0.0,
            "voids":           [],
            "likely_move":     "NEUTRAL",
            "likely_reason":   f"Error: {e}",
            "largest_cluster": None,
            "current_price":   current_price,
        }


def format_liquidity_map(liq: dict, current_price: float) -> str:
    """
    Format the liquidity map dict as a readable string block.

    Example output:
        🔴 STOPS ABOVE (buy-side liquidity)
          $2,655.00  (+$12.00)  [2 swings]
        ──────────────── $2,643.00 ────────────────
        🟢 STOPS BELOW (sell-side liquidity)
          $2,631.00  (−$12.00)  [3 swings]
        POC:  $2,640.00  |  VA:  $2,635.00 – $2,648.00
        ⬆ Likely move: UP  (hunts BSL above)
    """
    if not liq.get("available"):
        return "Liquidity map unavailable."

    SEP = "─" * 44
    lines = []

    # Clusters above
    lines.append("🔴 STOPS ABOVE (buy-side liquidity)")
    ca = liq.get("clusters_above", [])
    if ca:
        for c in ca[:3]:
            dist = f"+${c['distance_usd']:,.2f}"
            lines.append(f"  ${c['price']:>10,.2f}  ({dist})  [{c['count']} swing{'s' if c['count'] != 1 else ''}]")
    else:
        lines.append("  None detected in range")

    lines.append(f"{SEP} ${current_price:,.2f} {SEP}"[:len(SEP) + 2])
    lines.append(f"  ─── current price: ${current_price:,.2f} ───")

    # Clusters below
    lines.append("🟢 STOPS BELOW (sell-side liquidity)")
    cb = liq.get("clusters_below", [])
    if cb:
        for c in cb[:3]:
            dist = f"−${c['distance_usd']:,.2f}"
            lines.append(f"  ${c['price']:>10,.2f}  ({dist})  [{c['count']} swing{'s' if c['count'] != 1 else ''}]")
    else:
        lines.append("  None detected in range")

    # POC / VA
    poc     = liq.get("poc", 0.0)
    va_low  = liq.get("va_low", 0.0)
    va_high = liq.get("va_high", 0.0)
    if poc:
        lines.append(f"POC:  ${poc:,.2f}  |  VA: ${va_low:,.2f} – ${va_high:,.2f}")

    # Voids
    voids = liq.get("voids", [])
    if voids:
        _void_prices = ", ".join(f"${v['mid']:,.0f}" for v in voids[:3])
        lines.append(f"⚡ Vol voids: {_void_prices}")

    # Likely move
    move   = liq.get("likely_move", "NEUTRAL")
    reason = liq.get("likely_reason", "")
    arrow  = "⬆" if move == "UP" else ("⬇" if move == "DOWN" else "↔")
    lines.append(f"{arrow} Likely move: {move}  — {reason}")

    return "\n".join(lines)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    print("=== Liquidity Map Self-Test ===\n")
    df_test = yf.download("GC=F", period="5d", interval="5m", progress=False)
    df_test.columns = [c[0].lower() for c in df_test.columns]
    price = float(df_test["close"].iloc[-1])
    liq   = build_liquidity_map(df_test, price)
    print(f"Available    : {liq['available']}")
    print(f"Clusters ↑   : {len(liq['clusters_above'])}")
    print(f"Clusters ↓   : {len(liq['clusters_below'])}")
    print(f"POC          : ${liq['poc']:,.2f}")
    print(f"VA           : ${liq['va_low']:,.2f} – ${liq['va_high']:,.2f}")
    print(f"Likely move  : {liq['likely_move']}")
    print()
    print(format_liquidity_map(liq, price))
    print("\nSelf-test complete ✓")
