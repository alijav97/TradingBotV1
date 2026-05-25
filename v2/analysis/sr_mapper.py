"""
analysis/sr_mapper.py — Support/Resistance Auto-Mapper for TradingBotV2.

Automatically identifies key support and resistance levels from:
  1. Previous week high/low  (Mon-Sun UTC)
  2. Previous day high/low   (yesterday UTC)
  3. Round numbers           (configurable increments)
  4. Swing highs/lows        (last 100 candles)
  5. EMA levels              (EMA50 / EMA200)

Merges nearby levels into clusters and scores proximity to current price.
Works on any OHLCV DataFrame — not tied to any specific instrument.

Usage:
    from v2.analysis.sr_mapper import get_sr_levels, is_at_support, is_at_resistance
    sr = get_sr_levels(df, current_price)
    print(sr["summary"])
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CLUSTER_TOLERANCE_PCT   = 0.002   # levels within 0.2% of each other → merge
ROUND_NUM_DIVISORS      = [10, 25, 50, 100, 500, 1000]  # auto-selected by price
ROUND_NUM_RANGE_PCT     = 0.05    # scan ±5% from current price for round numbers
SWING_LOOKBACK          = 100     # candles to scan for swings
SWING_MIN_BARS          = 3       # bars either side to qualify as swing

# Proximity thresholds (percentage of current price)
PROX_IMMEDIATE = 0.001   # 0.1%
PROX_NEAR      = 0.005   # 0.5%
PROX_WATCH     = 0.010   # 1.0%

_EMPTY_LEVEL: dict[str, Any] = {
    "price":        0.0,
    "label":        "—",
    "strength":     "MODERATE",
    "proximity":    "DISTANT",
    "distance_abs": 999.0,
    "distance_pct": 99.0,
    "sources":      [],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_datetime_series(df: pd.DataFrame) -> pd.Series | None:
    """
    Try to extract a UTC datetime Series from the DataFrame.
    Checks DatetimeIndex first, then common column names.
    Returns a Series aligned with df.index, or None.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.to_series().reset_index(drop=True)

    for col in ("time", "date", "datetime", "timestamp", "open_time",
                "Datetime", "Time", "Date"):
        if col in df.columns:
            try:
                series = pd.to_datetime(df[col], utc=True, errors="coerce")
                if series.notna().sum() > 0:
                    return series.reset_index(drop=True)
            except (ValueError, TypeError) as exc:
                logger.debug("Could not parse datetime column %r: %s", col, exc)
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
    dist_abs = abs(price - current_price)
    dist_pct = dist_abs / max(current_price, 1e-9) * 100.0
    return {
        "price":        round(price, 5),
        "label":        label,
        "strength":     strength,
        "proximity":    _proximity_tag(dist_pct),
        "distance_abs": round(dist_abs, 5),
        "distance_pct": round(dist_pct, 4),
        "sources":      sources,
    }


def _auto_round_interval(current_price: float) -> tuple[float, float]:
    """
    Choose round-number step and key-level step appropriate for the
    instrument price magnitude so the mapper works on any asset.

    Returns (minor_interval, major_interval).
    """
    for divisor in sorted(ROUND_NUM_DIVISORS, reverse=True):
        if current_price / divisor >= 5:
            minor = divisor
            # Major = 2× minor, capped at the next divisor up
            idx = ROUND_NUM_DIVISORS.index(divisor)
            major = ROUND_NUM_DIVISORS[idx + 1] if idx + 1 < len(ROUND_NUM_DIVISORS) else divisor * 2
            return float(minor), float(major)
    return 1.0, 5.0


def _find_swing_highs_lows(df: pd.DataFrame,
                            lookback: int = SWING_LOOKBACK
                            ) -> tuple[list[float], list[float]]:
    """
    Detect local swing highs and lows in the last `lookback` candles.
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
        if len(highs) > 0:
            h = float(highs[i])
            if all(h > float(highs[i - j]) for j in range(1, bars + 1)) and \
               all(h > float(highs[i + j]) for j in range(1, bars + 1)):
                swing_highs.append(h)
        if len(lows) > 0:
            lo = float(lows[i])
            if all(lo < float(lows[i - j]) for j in range(1, bars + 1)) and \
               all(lo < float(lows[i + j]) for j in range(1, bars + 1)):
                swing_lows.append(lo)

    return swing_highs, swing_lows


def _merge_clusters(levels: list[dict[str, Any]],
                    current_price: float) -> list[dict[str, Any]]:
    """
    Merge levels within CLUSTER_TOLERANCE_PCT of each other into a single
    CLUSTER entry with upgraded strength.

    Uses percentage distance so the tolerance scales with instrument price.
    """
    if not levels:
        return levels

    tolerance = current_price * CLUSTER_TOLERANCE_PCT
    sorted_levels = sorted(levels, key=lambda x: x["price"])
    merged: list[dict[str, Any]] = []
    used = [False] * len(sorted_levels)

    for i, lvl in enumerate(sorted_levels):
        if used[i]:
            continue
        cluster = [lvl]
        for j in range(i + 1, len(sorted_levels)):
            if not used[j] and abs(sorted_levels[j]["price"] - lvl["price"]) <= tolerance:
                cluster.append(sorted_levels[j])
                used[j] = True
        used[i] = True

        if len(cluster) >= 2:
            avg_price    = sum(c["price"] for c in cluster) / len(cluster)
            all_labels   = [c["label"] for c in cluster]
            all_sources  = list({s for c in cluster for s in c["sources"]})
            has_major    = any(c["strength"] == "MAJOR" for c in cluster)
            merged_label = "Cluster: " + " + ".join(all_labels)
            if len(merged_label) > 60:
                merged_label = f"Cluster ({len(cluster)} levels @ {avg_price:,.4g})"
            merged.append({
                "price":        round(avg_price, 5),
                "label":        merged_label,
                "strength":     "MAJOR" if has_major else "STRONG",
                "proximity":    "DISTANT",     # recalculated below
                "distance_abs": 0.0,
                "distance_pct": 0.0,
                "sources":      all_sources,
            })
        else:
            merged.append(lvl)

    return merged


def _recalc_proximity(lvls: list[dict], current_price: float) -> list[dict]:
    for lv in lvls:
        d_abs = abs(lv["price"] - current_price)
        d_pct = d_abs / max(current_price, 1e-9) * 100.0
        lv["distance_abs"] = round(d_abs, 5)
        lv["distance_pct"] = round(d_pct, 4)
        lv["proximity"]    = _proximity_tag(d_pct)
    return lvls


# ══════════════════════════════════════════════════════════════════════════════
#  Main: get_sr_levels()
# ══════════════════════════════════════════════════════════════════════════════

def get_sr_levels(df: pd.DataFrame, current_price: float) -> dict[str, Any]:
    """
    Identify key support/resistance levels from 5 sources.

    Parameters
    ----------
    df            : OHLCV DataFrame. Expects columns: open, high, low, close.
                    Optional: volume, ema50, ema200. A datetime index or a
                    time/date column enables accurate prev-week/day detection.
    current_price : current market price for proximity calculations.

    Returns
    -------
    dict with keys:
        resistance_levels, support_levels, nearest_resistance, nearest_support,
        current_price, at_key_level, key_level_detail, prev_week_high,
        prev_week_low, prev_day_high, prev_day_low, round_numbers, summary
    """
    _FALLBACK: dict[str, Any] = {
        "resistance_levels":  [],
        "support_levels":     [],
        "nearest_resistance": dict(_EMPTY_LEVEL),
        "nearest_support":    dict(_EMPTY_LEVEL),
        "current_price":      current_price,
        "at_key_level":       False,
        "key_level_detail":   "",
        "prev_week_high":     0.0,
        "prev_week_low":      0.0,
        "prev_day_high":      0.0,
        "prev_day_low":       0.0,
        "round_numbers":      [],
        "summary":            "S/R data unavailable",
    }

    if df is None or len(df) < 10 or current_price <= 0:
        return _FALLBACK

    try:
        df_work = df.copy()
        df_work.columns = [c.lower().strip() for c in df_work.columns]
        if "close" not in df_work.columns:
            logger.warning("sr_mapper: DataFrame has no 'close' column")
            return _FALLBACK

        raw_resistance: list[dict[str, Any]] = []
        raw_support:    list[dict[str, Any]] = []

        prev_week_high = 0.0
        prev_week_low  = 0.0
        prev_day_high  = 0.0
        prev_day_low   = 0.0

        # ── SOURCE 1: Previous week high/low ──────────────────────────────────
        try:
            dt_series = _get_datetime_series(df_work)
            if dt_series is not None:
                if dt_series.dt.tz is None:
                    dt_series = dt_series.dt.tz_localize("UTC")
                else:
                    dt_series = dt_series.dt.tz_convert("UTC")

                now_utc        = datetime.now(timezone.utc)
                days_since_mon = now_utc.weekday()
                this_mon       = (now_utc - timedelta(days=days_since_mon)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                prev_mon = this_mon - timedelta(weeks=1)

                mask = (dt_series >= prev_mon) & (dt_series < this_mon)
                pw   = df_work[mask.values]
                if len(pw) >= 5 and "high" in pw.columns and "low" in pw.columns:
                    prev_week_high = float(pw["high"].max())
                    prev_week_low  = float(pw["low"].min())
            else:
                # Fallback: assume 1h candles, prev week = candles [-336:-168]
                if "high" in df_work.columns and "low" in df_work.columns:
                    if len(df_work) >= 336:
                        pw = df_work.iloc[-336:-168]
                        prev_week_high = float(pw["high"].max())
                        prev_week_low  = float(pw["low"].min())
                    elif len(df_work) >= 20:
                        half = max(5, len(df_work) // 4)
                        pw   = df_work.iloc[-half * 2:-half]
                        prev_week_high = float(pw["high"].max())
                        prev_week_low  = float(pw["low"].min())
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("sr_mapper: prev-week detection failed: %s", exc)

        if prev_week_high > 0:
            lvl = _make_level(prev_week_high, "Prev Week High", "MAJOR",
                              current_price, ["prev_week"])
            (raw_resistance if prev_week_high > current_price else raw_support).append(lvl)
        if prev_week_low > 0:
            lvl = _make_level(prev_week_low, "Prev Week Low", "MAJOR",
                              current_price, ["prev_week"])
            (raw_resistance if prev_week_low > current_price else raw_support).append(lvl)

        # ── SOURCE 2: Previous day high/low ───────────────────────────────────
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

                mask       = (dt_series >= yesterday) & (dt_series < today_utc)
                pd_candles = df_work[mask.values]
                if len(pd_candles) >= 3 and "high" in pd_candles.columns and "low" in pd_candles.columns:
                    prev_day_high = float(pd_candles["high"].max())
                    prev_day_low  = float(pd_candles["low"].min())
            else:
                if "high" in df_work.columns and "low" in df_work.columns:
                    if len(df_work) >= 48:
                        yd = df_work.iloc[-48:-24]
                        prev_day_high = float(yd["high"].max())
                        prev_day_low  = float(yd["low"].min())
                    elif len(df_work) >= 10:
                        yd = df_work.iloc[-10:-2]
                        prev_day_high = float(yd["high"].max())
                        prev_day_low  = float(yd["low"].min())
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("sr_mapper: prev-day detection failed: %s", exc)

        if prev_day_high > 0:
            lvl = _make_level(prev_day_high, "Prev Day High", "MAJOR",
                              current_price, ["prev_day"])
            (raw_resistance if prev_day_high > current_price else raw_support).append(lvl)
        if prev_day_low > 0:
            lvl = _make_level(prev_day_low, "Prev Day Low", "MAJOR",
                              current_price, ["prev_day"])
            (raw_resistance if prev_day_low > current_price else raw_support).append(lvl)

        # ── SOURCE 3: Round numbers ────────────────────────────────────────────
        round_numbers: list[float] = []
        minor_interval, major_interval = _auto_round_interval(current_price)
        lo_bound = current_price * (1 - ROUND_NUM_RANGE_PCT)
        hi_bound = current_price * (1 + ROUND_NUM_RANGE_PCT)

        base      = round(current_price / minor_interval) * minor_interval
        candidate = base - current_price * ROUND_NUM_RANGE_PCT * 2
        while candidate <= hi_bound + minor_interval:
            if lo_bound <= candidate <= hi_bound and abs(candidate - current_price) > 1e-9:
                is_key   = (candidate % major_interval < 1e-6 or
                            candidate % major_interval > (major_interval - 1e-6))
                strength = "MAJOR" if is_key else "MODERATE"
                label    = (f"Key Round {candidate:,.4g}" if is_key
                            else f"Round {candidate:,.4g}")
                lvl = _make_level(candidate, label, strength, current_price, ["round_number"])
                (raw_resistance if candidate > current_price else raw_support).append(lvl)
                round_numbers.append(candidate)
            candidate += minor_interval

        # ── SOURCE 4: Swing highs/lows (last 100 candles) ─────────────────────
        try:
            if "high" in df_work.columns and "low" in df_work.columns:
                sw_highs, sw_lows = _find_swing_highs_lows(df_work, SWING_LOOKBACK)

                res_swings = sorted(
                    [h for h in sw_highs if h > current_price],
                    key=lambda x: abs(x - current_price)
                )[:3]
                for sh in res_swings:
                    lvl = _make_level(sh, "Swing High", "MODERATE",
                                      current_price, ["swing"])
                    raw_resistance.append(lvl)

                sup_swings = sorted(
                    [lo for lo in sw_lows if lo < current_price],
                    key=lambda x: abs(x - current_price)
                )[:3]
                for sl in sup_swings:
                    lvl = _make_level(sl, "Swing Low", "MODERATE",
                                      current_price, ["swing"])
                    raw_support.append(lvl)
        except (KeyError, IndexError, ValueError) as exc:
            logger.debug("sr_mapper: swing detection failed: %s", exc)

        # ── SOURCE 5: EMA levels ───────────────────────────────────────────────
        try:
            for col, lbl in [("ema50", "EMA50"), ("ema200", "EMA200")]:
                if col in df_work.columns:
                    ema_val = float(df_work[col].iloc[-1])
                    if ema_val > 0 and not np.isnan(ema_val):
                        dist_pct = abs(ema_val - current_price) / current_price * 100
                        strength = "IMMEDIATE" if dist_pct < 0.1 else "MODERATE"
                        lvl      = _make_level(ema_val, lbl, strength,
                                               current_price, ["ema"])
                        (raw_resistance if ema_val > current_price else raw_support).append(lvl)
        except (KeyError, IndexError, ValueError) as exc:
            logger.debug("sr_mapper: EMA level extraction failed: %s", exc)

        # ── Cluster detection ──────────────────────────────────────────────────
        resistance_merged = _merge_clusters(raw_resistance, current_price)
        support_merged    = _merge_clusters(raw_support,    current_price)

        resistance_levels = _recalc_proximity(resistance_merged, current_price)
        support_levels    = _recalc_proximity(support_merged,    current_price)

        resistance_levels.sort(key=lambda x: x["price"])
        support_levels.sort(key=lambda x: x["price"], reverse=True)

        # ── Nearest levels ─────────────────────────────────────────────────────
        nearest_resistance = (
            min(resistance_levels, key=lambda x: x["distance_abs"])
            if resistance_levels else dict(_EMPTY_LEVEL)
        )
        nearest_support = (
            min(support_levels, key=lambda x: x["distance_abs"])
            if support_levels else dict(_EMPTY_LEVEL)
        )

        # ── at_key_level ───────────────────────────────────────────────────────
        at_key_level     = False
        key_level_detail = ""
        for lv in resistance_levels + support_levels:
            if lv["strength"] in ("MAJOR", "STRONG") and lv["proximity"] == "IMMEDIATE":
                at_key_level     = True
                key_level_detail = f"{lv['label']} @ {lv['price']:,.5g}"
                break

        # ── Summary string ─────────────────────────────────────────────────────
        nr      = nearest_resistance
        ns      = nearest_support
        res_str = f"{nr['price']:,.5g} ({nr['label']})" if nr.get("price") else "—"
        sup_str = f"{ns['price']:,.5g} ({ns['label']})" if ns.get("price") else "—"
        summary = (
            f"Resistance: {res_str} | "
            f"Current: {current_price:,.5g} | "
            f"Support: {sup_str}"
        )
        if at_key_level:
            summary = f"AT KEY LEVEL: {key_level_detail}  |  " + summary

        logger.debug("sr_mapper: %d resistance, %d support levels identified",
                     len(resistance_levels), len(support_levels))

        return {
            "resistance_levels":  resistance_levels,
            "support_levels":     support_levels,
            "nearest_resistance": nearest_resistance,
            "nearest_support":    nearest_support,
            "current_price":      current_price,
            "at_key_level":       at_key_level,
            "key_level_detail":   key_level_detail,
            "prev_week_high":     round(prev_week_high, 5),
            "prev_week_low":      round(prev_week_low, 5),
            "prev_day_high":      round(prev_day_high, 5),
            "prev_day_low":       round(prev_day_low, 5),
            "round_numbers":      sorted(round_numbers),
            "summary":            summary,
        }

    except Exception as exc:
        logger.exception("sr_mapper: unexpected error: %s", exc)
        _FALLBACK["summary"] = f"S/R error: {exc}"
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
    except Exception as exc:
        logger.warning("is_at_support failed: %s", exc)
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
    except Exception as exc:
        logger.warning("is_at_resistance failed: %s", exc)
    return False
