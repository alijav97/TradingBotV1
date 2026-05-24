"""
sr_mapper.py
────────────
S/R Auto-Mapper for TradingBotV1.

Automatically identifies key support and resistance levels from:
  1. Previous week high/low  (Mon-Sun UTC)
  2. Previous day high/low   (yesterday UTC)
  3. Round numbers           ($50 increments, $100 = KEY ROUND)
  4. Swing highs/lows        (last 100 candles)
  5. EMA levels              (EMA50 / EMA200)

Merges nearby levels into clusters and scores proximity to current price.

Usage:
    from sr_mapper import get_sr_levels, is_at_support, is_at_resistance
    sr = get_sr_levels(df, current_price)
    print(sr["summary"])
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────
CLUSTER_TOLERANCE_USD   = 5.0    # levels within $5 → merge into cluster
ROUND_NUM_INTERVAL      = 50.0   # $50 increments
ROUND_NUM_KEY_INTERVAL  = 100.0  # $100 multiples = KEY (MAJOR strength)
ROUND_NUM_RANGE         = 200.0  # scan ±$200 from current price
SWING_LOOKBACK          = 100    # candles to scan for swings
SWING_MIN_BARS          = 3      # bars either side to qualify as swing

# Proximity thresholds (percentage of current price)
PROX_IMMEDIATE = 0.001   # 0.1%
PROX_NEAR      = 0.005   # 0.5%
PROX_WATCH     = 0.010   # 1.0%

_EMPTY_LEVEL: dict[str, Any] = {
    "price":        0.0,
    "label":        "—",
    "strength":     "MODERATE",
    "proximity":    "DISTANT",
    "distance_usd": 999.0,
    "distance_pct": 99.0,
    "sources":      [],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_datetime_series(df: pd.DataFrame) -> pd.Series | None:
    """
    Try to extract a datetime Series from the DataFrame.
    Checks: DatetimeIndex, then common column names.
    Returns a Series aligned with df.index, or None.
    """
    # Check index
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.to_series().reset_index(drop=True)

    # Check common column names
    for col in ("time", "date", "datetime", "timestamp", "open_time", "Datetime", "Time", "Date"):
        if col in df.columns:
            try:
                series = pd.to_datetime(df[col], utc=True, errors="coerce")
                if series.notna().sum() > 0:
                    return series.reset_index(drop=True)
            except Exception:
                continue
    return None


def _proximity_tag(distance_pct: float) -> str:
    if distance_pct < PROX_IMMEDIATE * 100:
        return "IMMEDIATE"
    if distance_pct < PROX_NEAR * 100:
        return "NEAR"
    if distance_pct < PROX_WATCH * 100:
        return "WATCH"
    return "DISTANT"


def _make_level(price: float, label: str, strength: str,
                current_price: float, sources: list[str]) -> dict[str, Any]:
    dist_usd = abs(price - current_price)
    dist_pct = dist_usd / max(current_price, 1.0) * 100.0
    return {
        "price":        round(price, 2),
        "label":        label,
        "strength":     strength,
        "proximity":    _proximity_tag(dist_pct),
        "distance_usd": round(dist_usd, 2),
        "distance_pct": round(dist_pct, 4),
        "sources":      sources,
    }


def _find_swing_highs_lows(df: pd.DataFrame, lookback: int = 100
                           ) -> tuple[list[float], list[float]]:
    """
    Detect local swing highs and swing lows in the last `lookback` candles.
    Returns (swing_highs, swing_lows) as lists of price values.
    """
    window = df.tail(lookback).reset_index(drop=True)
    n = len(window)
    highs = window["high"].values if "high" in window.columns else np.array([])
    lows  = window["low"].values  if "low"  in window.columns else np.array([])

    swing_highs: list[float] = []
    swing_lows:  list[float] = []

    bars = SWING_MIN_BARS
    for i in range(bars, n - bars):
        # Swing high: local max within ±bars
        if len(highs) > 0:
            h = float(highs[i])
            if all(h > float(highs[i - j]) for j in range(1, bars + 1)) and \
               all(h > float(highs[i + j]) for j in range(1, bars + 1)):
                swing_highs.append(h)
        # Swing low: local min within ±bars
        if len(lows) > 0:
            lo = float(lows[i])
            if all(lo < float(lows[i - j]) for j in range(1, bars + 1)) and \
               all(lo < float(lows[i + j]) for j in range(1, bars + 1)):
                swing_lows.append(lo)

    return swing_highs, swing_lows


def _merge_clusters(levels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge levels within CLUSTER_TOLERANCE_USD of each other into a single
    CLUSTER level with upgraded strength.
    """
    if not levels:
        return levels

    # Sort by price
    sorted_levels = sorted(levels, key=lambda x: x["price"])
    merged: list[dict[str, Any]] = []
    used = [False] * len(sorted_levels)

    for i, lvl in enumerate(sorted_levels):
        if used[i]:
            continue
        cluster = [lvl]
        for j in range(i + 1, len(sorted_levels)):
            if not used[j] and abs(sorted_levels[j]["price"] - lvl["price"]) <= CLUSTER_TOLERANCE_USD:
                cluster.append(sorted_levels[j])
                used[j] = True
        used[i] = True

        if len(cluster) >= 2:
            # Merge: average price, combine sources and labels, upgrade strength
            avg_price   = sum(c["price"] for c in cluster) / len(cluster)
            all_labels  = [c["label"] for c in cluster]
            all_sources = list({s for c in cluster for s in c["sources"]})
            has_major   = any(c["strength"] == "MAJOR" for c in cluster)
            merged_label = "Cluster: " + " + ".join(all_labels)
            if len(merged_label) > 60:
                merged_label = f"Cluster ({len(cluster)} levels @ ${avg_price:,.0f})"
            merged.append({
                "price":        round(avg_price, 2),
                "label":        merged_label,
                "strength":     "MAJOR" if has_major else "STRONG",
                "proximity":    cluster[0]["proximity"],  # recalculated after
                "distance_usd": cluster[0]["distance_usd"],
                "distance_pct": cluster[0]["distance_pct"],
                "sources":      all_sources,
            })
        else:
            merged.append(lvl)

    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  Main: get_sr_levels()
# ══════════════════════════════════════════════════════════════════════════════

def get_sr_levels(df: pd.DataFrame, current_price: float) -> dict[str, Any]:
    """
    Identify key support/resistance levels from 5 sources.

    Parameters
    ----------
    df            : OHLCV + indicator DataFrame (expects low, high, close columns)
    current_price : current market price for proximity calculations

    Returns
    -------
    dict with keys: resistance_levels, support_levels, nearest_resistance,
                    nearest_support, current_price, at_key_level,
                    key_level_detail, prev_week_high, prev_week_low,
                    prev_day_high, prev_day_low, round_numbers, summary
    """
    _FALLBACK: dict[str, Any] = {
        "resistance_levels": [],
        "support_levels":    [],
        "nearest_resistance": dict(_EMPTY_LEVEL),
        "nearest_support":    dict(_EMPTY_LEVEL),
        "current_price":     current_price,
        "at_key_level":      False,
        "key_level_detail":  "",
        "prev_week_high":    0.0,
        "prev_week_low":     0.0,
        "prev_day_high":     0.0,
        "prev_day_low":      0.0,
        "round_numbers":     [],
        "summary":           "S/R data unavailable",
    }

    try:
        if df is None or len(df) < 10 or current_price <= 0:
            return _FALLBACK

        # Normalise column names
        df_work = df.copy()
        df_work.columns = [c.lower().strip() for c in df_work.columns]
        if "close" not in df_work.columns:
            return _FALLBACK

        # ── Collect all raw levels (unsorted) ────────────────────────────────
        raw_resistance: list[dict[str, Any]] = []
        raw_support:    list[dict[str, Any]] = []

        prev_week_high = 0.0
        prev_week_low  = 0.0
        prev_day_high  = 0.0
        prev_day_low   = 0.0

        # ── SOURCE 1: Previous week high/low ─────────────────────────────────
        try:
            dt_series = _get_datetime_series(df_work)
            if dt_series is not None:
                # Ensure UTC
                if dt_series.dt.tz is None:
                    dt_series = dt_series.dt.tz_localize("UTC")
                else:
                    dt_series = dt_series.dt.tz_convert("UTC")

                now_utc = datetime.now(timezone.utc)
                # Start of current week (Monday)
                days_since_mon = now_utc.weekday()
                this_mon = (now_utc - timedelta(days=days_since_mon)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                prev_mon = this_mon - timedelta(weeks=1)
                prev_sun = this_mon  # exclusive end

                mask = (dt_series >= prev_mon) & (dt_series < prev_sun)
                pw = df_work[mask.values]
                if len(pw) >= 5 and "high" in pw.columns and "low" in pw.columns:
                    prev_week_high = float(pw["high"].max())
                    prev_week_low  = float(pw["low"].min())
            else:
                # Fallback: assume 1h candles, prev week = candles [-336:-168]
                if "high" in df_work.columns and "low" in df_work.columns and len(df_work) >= 336:
                    pw = df_work.iloc[-336:-168]
                    prev_week_high = float(pw["high"].max())
                    prev_week_low  = float(pw["low"].min())
                elif len(df_work) >= 20:
                    # Use older half of available data
                    half = max(5, len(df_work) // 4)
                    pw = df_work.iloc[-half * 2:-half]
                    if "high" in df_work.columns and "low" in df_work.columns:
                        prev_week_high = float(pw["high"].max())
                        prev_week_low  = float(pw["low"].min())
        except Exception:
            pass

        if prev_week_high > 0:
            lvl = _make_level(prev_week_high, "Prev Week High", "MAJOR",
                              current_price, ["prev_week"])
            (raw_resistance if prev_week_high > current_price else raw_support).append(lvl)
        if prev_week_low > 0:
            lvl = _make_level(prev_week_low, "Prev Week Low", "MAJOR",
                              current_price, ["prev_week"])
            (raw_resistance if prev_week_low > current_price else raw_support).append(lvl)

        # ── SOURCE 2: Previous day high/low ──────────────────────────────────
        try:
            dt_series = _get_datetime_series(df_work)
            if dt_series is not None:
                if dt_series.dt.tz is None:
                    dt_series = dt_series.dt.tz_localize("UTC")
                else:
                    dt_series = dt_series.dt.tz_convert("UTC")

                now_utc   = datetime.now(timezone.utc)
                today_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                yesterday = today_utc - timedelta(days=1)

                mask = (dt_series >= yesterday) & (dt_series < today_utc)
                pd_candles = df_work[mask.values]
                if len(pd_candles) >= 3 and "high" in pd_candles.columns and "low" in pd_candles.columns:
                    prev_day_high = float(pd_candles["high"].max())
                    prev_day_low  = float(pd_candles["low"].min())
            else:
                # Fallback: last 48-24 candles (1h assumption)
                if len(df_work) >= 48 and "high" in df_work.columns and "low" in df_work.columns:
                    yd = df_work.iloc[-48:-24]
                    prev_day_high = float(yd["high"].max())
                    prev_day_low  = float(yd["low"].min())
                elif len(df_work) >= 10 and "high" in df_work.columns and "low" in df_work.columns:
                    yd = df_work.iloc[-10:-2]
                    prev_day_high = float(yd["high"].max())
                    prev_day_low  = float(yd["low"].min())
        except Exception:
            pass

        if prev_day_high > 0:
            lvl = _make_level(prev_day_high, "Prev Day High", "MAJOR",
                              current_price, ["prev_day"])
            (raw_resistance if prev_day_high > current_price else raw_support).append(lvl)
        if prev_day_low > 0:
            lvl = _make_level(prev_day_low, "Prev Day Low", "MAJOR",
                              current_price, ["prev_day"])
            (raw_resistance if prev_day_low > current_price else raw_support).append(lvl)

        # ── SOURCE 3: Round numbers ───────────────────────────────────────────
        round_numbers: list[float] = []
        lo_bound = current_price - ROUND_NUM_RANGE
        hi_bound = current_price + ROUND_NUM_RANGE

        # Generate $50 grid
        base = round(current_price / ROUND_NUM_INTERVAL) * ROUND_NUM_INTERVAL
        candidate = base - ROUND_NUM_RANGE
        while candidate <= hi_bound:
            if lo_bound <= candidate <= hi_bound and abs(candidate - current_price) > 0.01:
                is_key = (candidate % ROUND_NUM_KEY_INTERVAL < 0.01 or
                          candidate % ROUND_NUM_KEY_INTERVAL > (ROUND_NUM_KEY_INTERVAL - 0.01))
                strength = "MAJOR" if is_key else "MODERATE"
                label = f"Key Round ${candidate:,.0f}" if is_key else f"Round ${candidate:,.0f}"
                lvl = _make_level(candidate, label, strength, current_price, ["round_number"])
                (raw_resistance if candidate > current_price else raw_support).append(lvl)
                round_numbers.append(candidate)
            candidate += ROUND_NUM_INTERVAL

        # ── SOURCE 4: Swing highs/lows (last 100 candles) ─────────────────────
        try:
            if "high" in df_work.columns and "low" in df_work.columns:
                sw_highs, sw_lows = _find_swing_highs_lows(df_work, SWING_LOOKBACK)

                # Top 3 swing highs above current price (closest first)
                res_swings = sorted(
                    [h for h in sw_highs if h > current_price],
                    key=lambda x: abs(x - current_price)
                )[:3]
                for sh in res_swings:
                    lvl = _make_level(sh, "Swing High", "MODERATE",
                                      current_price, ["swing"])
                    raw_resistance.append(lvl)

                # Top 3 swing lows below current price (closest first)
                sup_swings = sorted(
                    [lo for lo in sw_lows if lo < current_price],
                    key=lambda x: abs(x - current_price)
                )[:3]
                for sl in sup_swings:
                    lvl = _make_level(sl, "Swing Low", "MODERATE",
                                      current_price, ["swing"])
                    raw_support.append(lvl)
        except Exception:
            pass

        # ── SOURCE 5: EMA levels ──────────────────────────────────────────────
        try:
            for col, lbl in [("ema50", "EMA50"), ("ema200", "EMA200")]:
                if col in df_work.columns:
                    ema_val = float(df_work[col].iloc[-1])
                    if ema_val > 0:
                        dist_pct = abs(ema_val - current_price) / current_price * 100
                        strength = "IMMEDIATE" if dist_pct < 0.1 else "MODERATE"
                        lvl = _make_level(ema_val, lbl, strength,
                                          current_price, ["ema"])
                        (raw_resistance if ema_val > current_price else raw_support).append(lvl)
        except Exception:
            pass

        # ── Cluster detection ─────────────────────────────────────────────────
        resistance_merged = _merge_clusters(raw_resistance)
        support_merged    = _merge_clusters(raw_support)

        # Recalculate proximity after merge
        def _recalc_proximity(lvls: list[dict]) -> list[dict]:
            for lv in lvls:
                d_usd = abs(lv["price"] - current_price)
                d_pct = d_usd / max(current_price, 1.0) * 100.0
                lv["distance_usd"] = round(d_usd, 2)
                lv["distance_pct"] = round(d_pct, 4)
                lv["proximity"]    = _proximity_tag(d_pct)
            return lvls

        resistance_levels = _recalc_proximity(resistance_merged)
        support_levels    = _recalc_proximity(support_merged)

        # Sort: resistance ascending (nearest first), support descending (nearest first)
        resistance_levels.sort(key=lambda x: x["price"])
        support_levels.sort(key=lambda x: x["price"], reverse=True)

        # ── Nearest levels ────────────────────────────────────────────────────
        nearest_resistance = (
            min(resistance_levels, key=lambda x: x["distance_usd"])
            if resistance_levels else dict(_EMPTY_LEVEL)
        )
        nearest_support = (
            min(support_levels, key=lambda x: x["distance_usd"])
            if support_levels else dict(_EMPTY_LEVEL)
        )

        # ── at_key_level ──────────────────────────────────────────────────────
        at_key_level    = False
        key_level_detail = ""
        all_levels = resistance_levels + support_levels
        for lv in all_levels:
            if lv["strength"] in ("MAJOR", "STRONG") and lv["proximity"] == "IMMEDIATE":
                at_key_level     = True
                key_level_detail = f"{lv['label']} @ ${lv['price']:,.2f}"
                break

        # ── Summary string ────────────────────────────────────────────────────
        nr  = nearest_resistance
        ns  = nearest_support
        res_str = f"${nr['price']:,.2f} ({nr['label']})" if nr.get("price") else "—"
        sup_str = f"${ns['price']:,.2f} ({ns['label']})" if ns.get("price") else "—"
        summary = (
            f"Resistance: {res_str} | "
            f"Current: ${current_price:,.2f} | "
            f"Support: {sup_str}"
        )
        if at_key_level:
            summary = f"⭐ AT KEY LEVEL: {key_level_detail}  |  " + summary

        return {
            "resistance_levels":  resistance_levels,
            "support_levels":     support_levels,
            "nearest_resistance": nearest_resistance,
            "nearest_support":    nearest_support,
            "current_price":      current_price,
            "at_key_level":       at_key_level,
            "key_level_detail":   key_level_detail,
            "prev_week_high":     round(prev_week_high, 2),
            "prev_week_low":      round(prev_week_low, 2),
            "prev_day_high":      round(prev_day_high, 2),
            "prev_day_low":       round(prev_day_low, 2),
            "round_numbers":      sorted(round_numbers),
            "summary":            summary,
        }

    except Exception as _e:
        _FALLBACK["summary"] = f"S/R error: {_e}"
        return _FALLBACK


# ══════════════════════════════════════════════════════════════════════════════
#  Helper booleans
# ══════════════════════════════════════════════════════════════════════════════

def is_at_support(df: pd.DataFrame, current_price: float,
                  tolerance: float = 0.003) -> bool:
    """
    Return True if current_price is within `tolerance` (e.g. 0.003 = 0.3%)
    of any support level.
    """
    try:
        sr = get_sr_levels(df, current_price)
        for lv in sr.get("support_levels", []):
            if lv["distance_pct"] / 100.0 <= tolerance:
                return True
    except Exception:
        pass
    return False


def is_at_resistance(df: pd.DataFrame, current_price: float,
                     tolerance: float = 0.003) -> bool:
    """
    Return True if current_price is within `tolerance` (e.g. 0.003 = 0.3%)
    of any resistance level.
    """
    try:
        sr = get_sr_levels(df, current_price)
        for lv in sr.get("resistance_levels", []):
            if lv["distance_pct"] / 100.0 <= tolerance:
                return True
    except Exception:
        pass
    return False
