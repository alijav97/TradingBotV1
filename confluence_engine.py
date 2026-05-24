"""
confluence_engine.py
────────────────────
Upgrades signal quality by requiring MINIMUM 3 CONFLUENCES before a trade
signal is considered valid.

Usage:
    from confluence_engine import validate_signal, print_confluence_report
    result = validate_signal(signal_dict, df)
    print_confluence_report(result)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ── SMC integration ───────────────────────────────────────────────────────────
try:
    from smart_money import SmartMoneyAnalyzer as _SmartMoneyAnalyzer
    _SMC_OK = True
    # Check if enhanced get_smc_context is available
    _SMC_ENHANCED = hasattr(_SmartMoneyAnalyzer(), "get_smc_context")
except ImportError:
    _SMC_OK = False
    _SMC_ENHANCED = False

# ── MTF integration ───────────────────────────────────────────────────────────
try:
    from mtf_analyzer import MultiTimeframeAnalyzer as _MTFAnalyzer
    _MTF_OK = True
except ImportError:
    _MTF_OK = False

# ── DXY integration ───────────────────────────────────────────────────────────
try:
    from dxy_correlation import DXYCorrelation as _DXYCorrelation
    _DXY_OK = True
except ImportError:
    _DXY_OK = False

# ── S/R Mapper integration ────────────────────────────────────────────────────
try:
    from sr_mapper import get_sr_levels as _get_sr_levels
    _SR_OK = True
except ImportError:
    _SR_OK = False
    _get_sr_levels = None  # type: ignore[assignment]

# ── COT Analyzer integration ──────────────────────────────────────────────────
try:
    from cot_analyzer import fetch_cot_data as _fetch_cot_data, get_cot_signal as _get_cot_signal
    _COT_OK = True
except ImportError:
    _COT_OK = False
    def _fetch_cot_data(): return {"available": False}      # type: ignore[misc]
    def _get_cot_signal(d, c=None): return {"boost": 0.0, "aligned": False, "opposed": False, "bias": "NEUTRAL", "note": "unavailable", "available": False}  # type: ignore[misc]

# ── Liquidity Map integration ─────────────────────────────────────────────────
try:
    from liquidity_map import build_liquidity_map as _build_liquidity_map
    _LIQ_OK = True
except ImportError:
    _LIQ_OK = False
    def _build_liquidity_map(df, p): return {"available": False, "largest_cluster": None}  # type: ignore[misc]

# ── Indicators integration ────────────────────────────────────────────────────────
try:
    from indicators import get_all_indicators as _get_indicators
    _IND_OK = True
except Exception:
    _IND_OK = False
    def _get_indicators(df): return {}  # type: ignore[misc]

try:
    from ml_engine import get_ml_confidence_adjustment as _get_ml_adj
    _ML_OK = True
except Exception:
    _ML_OK = False
    def _get_ml_adj(*a, **kw): return {'adjustment': 0.0, 'available': False}  # type: ignore[misc]

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_CONFLUENCES     = 3          # hard minimum to pass
RSI_OVERSOLD        = 35
RSI_OVERBOUGHT      = 65
ATR_HIGH_MULT       = 1.5        # ATR > avg * this → high volatility
ATR_LOW_MULT        = 0.5        # ATR < avg * this → low volatility
STRUCTURE_TOLERANCE = 0.003      # 0.3% proximity to swing level

# ── Weighted confidence scores (sum ≤10 after cap) ────────────────────────────
# Each factor earns its weight when passing. Failing contributes 0 (not negative).
# Max raw sum = 11.5 but capped at 10.0.  Volatility acts as a bonus.
CONFLUENCE_WEIGHTS: dict[str, float] = {
    "HTF":        2.5,   # 25% — top-down D1/H4 alignment (most important)
    "SMC":        2.0,   # 20% — order blocks, FVG, liquidity sweeps
    "Trend":      1.5,   # 15% — price vs EMA200
    "Structure":  1.5,   # 15% — at key S/R level
    "Momentum":   1.0,   # 10% — RSI + MACD
    "DXY":        1.0,   # 10% — US Dollar Index inverse correlation
    "Candle":     0.5,   #  5% — confirmation candle pattern
    "Session":    0.5,   #  5% — high-probability trading window
    "Volatility": 0.5,   #  5% — suitable ATR (bonus; can push total to 10)
}


# ══════════════════════════════════════════════════════════════════════════════
#  ConfluenceChecker
# ══════════════════════════════════════════════════════════════════════════════

class ConfluenceChecker:
    """
    Runs six independent checks against a price DataFrame.
    Each check returns a plain dict describing what was found.

    Expected DataFrame columns (standard output from backtest.py / setup.py):
        open, high, low, close, volume, ema50, ema200, rsi, macd, macd_signal, atr
    """

    # ── 1. Trend ──────────────────────────────────────────────────────────────
    def check_trend(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Primary trend based on EMA200 position and EMA50/200 cross.

        Returns
        -------
        primary_trend : "bullish" | "bearish" | "sideways"
        strength      : "strong" | "weak"
        ema50         : current EMA50 value
        ema200        : current EMA200 value
        price         : current close
        detail        : human-readable explanation
        """
        row    = df.iloc[-1]
        price  = row["close"]
        ema50  = row.get("ema50",  float("nan"))
        ema200 = row.get("ema200", float("nan"))

        if math.isnan(ema50) or math.isnan(ema200):
            return {
                "primary_trend": "sideways",
                "strength":       "weak",
                "ema50":          ema50,
                "ema200":         ema200,
                "price":          price,
                "detail":         "EMA data unavailable",
            }

        above_ema200 = price > ema200
        golden_cross = ema50 > ema200          # bullish EMA alignment
        death_cross  = ema50 < ema200          # bearish EMA alignment

        if above_ema200 and golden_cross:
            trend    = "bullish"
            strength = "strong"
            detail   = f"Price ${price:.1f} above EMA200 ${ema200:.1f}; EMA50 above EMA200"
        elif above_ema200 and death_cross:
            trend    = "bullish"
            strength = "weak"
            detail   = f"Price ${price:.1f} above EMA200 ${ema200:.1f}; EMA50 below EMA200 (weakening)"
        elif not above_ema200 and death_cross:
            trend    = "bearish"
            strength = "strong"
            detail   = f"Price ${price:.1f} below EMA200 ${ema200:.1f}; EMA50 below EMA200"
        elif not above_ema200 and golden_cross:
            trend    = "bearish"
            strength = "weak"
            detail   = f"Price ${price:.1f} below EMA200 ${ema200:.1f}; EMA50 above EMA200 (recovering)"
        else:
            trend    = "sideways"
            strength = "weak"
            detail   = "Indeterminate trend"

        return {
            "primary_trend": trend,
            "strength":       strength,
            "ema50":          round(ema50, 2),
            "ema200":         round(ema200, 2),
            "price":          round(price, 2),
            "detail":         detail,
        }

    # ── 2. Momentum ───────────────────────────────────────────────────────────
    def check_momentum(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        RSI zone and MACD crossover.

        Returns
        -------
        momentum     : "bullish" | "bearish" | "neutral"
        rsi_value    : float
        rsi_zone     : "oversold" | "neutral" | "overbought"
        macd_signal  : "bullish_cross" | "bearish_cross" | "none"
        detail       : explanation string
        """
        last   = df.iloc[-1]
        prev   = df.iloc[-2] if len(df) >= 2 else last

        rsi    = last.get("rsi",         float("nan"))
        macd   = last.get("macd",        float("nan"))
        msig   = last.get("macd_signal", float("nan"))
        p_macd = prev.get("macd",        float("nan"))
        p_msig = prev.get("macd_signal", float("nan"))

        # RSI zone
        if math.isnan(rsi):
            rsi_zone = "neutral"
        elif rsi < RSI_OVERSOLD:
            rsi_zone = "oversold"
        elif rsi > RSI_OVERBOUGHT:
            rsi_zone = "overbought"
        else:
            rsi_zone = "neutral"

        # MACD cross (current bar crosses, previous bar did not)
        if not any(math.isnan(v) for v in [macd, msig, p_macd, p_msig]):
            if macd > msig and p_macd <= p_msig:
                macd_cross = "bullish_cross"
            elif macd < msig and p_macd >= p_msig:
                macd_cross = "bearish_cross"
            else:
                macd_cross = "none"
        else:
            macd_cross = "none"

        # Combined momentum bias
        bullish_signals = sum([
            rsi_zone == "oversold",
            macd_cross == "bullish_cross",
            (not math.isnan(macd) and not math.isnan(msig) and macd > msig),
        ])
        bearish_signals = sum([
            rsi_zone == "overbought",
            macd_cross == "bearish_cross",
            (not math.isnan(macd) and not math.isnan(msig) and macd < msig),
        ])

        if bullish_signals > bearish_signals:
            momentum = "bullish"
        elif bearish_signals > bullish_signals:
            momentum = "bearish"
        else:
            momentum = "neutral"

        rsi_str = f"{rsi:.1f}" if not math.isnan(rsi) else "N/A"
        detail  = f"RSI {rsi_str} ({rsi_zone}); MACD {macd_cross}"

        return {
            "momentum":    momentum,
            "rsi_value":   round(rsi, 1) if not math.isnan(rsi) else None,
            "rsi_zone":    rsi_zone,
            "macd_signal": macd_cross,
            "detail":      detail,
        }

    # ── 3. Structure ──────────────────────────────────────────────────────────
    def check_structure(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Swing high/low levels and dynamic EMA levels.

        Returns
        -------
        at_support      : bool
        at_resistance   : bool
        structure_type  : "support" | "resistance" | "both" | "none"
        nearest_level   : float or None
        detail          : explanation
        """
        price  = df.iloc[-1]["close"]
        ema50  = df.iloc[-1].get("ema50",  float("nan"))
        ema200 = df.iloc[-1].get("ema200", float("nan"))

        # Find swing highs and lows in the last 100 bars
        window = df.tail(100)
        highs  = []
        lows   = []
        for i in range(2, len(window) - 2):
            row    = window.iloc[i]
            p_hi   = window.iloc[i - 1]["high"]
            n_hi   = window.iloc[i + 1]["high"]
            p_lo   = window.iloc[i - 1]["low"]
            n_lo   = window.iloc[i + 1]["low"]
            if row["high"] > p_hi and row["high"] > n_hi:
                highs.append(row["high"])
            if row["low"] < p_lo and row["low"] < n_lo:
                lows.append(row["low"])

        # Last 3 significant swings
        swing_highs = sorted(highs)[-3:] if highs else []
        swing_lows  = sorted(lows)[:3]   if lows  else []

        # Dynamic EMA levels
        dynamic_supports    = []
        dynamic_resistances = []
        for level in [ema50, ema200]:
            if math.isnan(level):
                continue
            if level < price:
                dynamic_supports.append(level)
            else:
                dynamic_resistances.append(level)

        tol = price * STRUCTURE_TOLERANCE

        at_support    = any(abs(price - lv) <= tol for lv in swing_lows  + dynamic_supports)
        at_resistance = any(abs(price - lv) <= tol for lv in swing_highs + dynamic_resistances)

        if at_support and at_resistance:
            stype = "both"
        elif at_support:
            stype = "support"
        elif at_resistance:
            stype = "resistance"
        else:
            stype = "none"

        # Find nearest level for display
        all_levels   = swing_lows + swing_highs + dynamic_supports + dynamic_resistances
        nearest      = min(all_levels, key=lambda lv: abs(price - lv)) if all_levels else None

        detail_parts = []
        if at_support:
            detail_parts.append("at support")
        if at_resistance:
            detail_parts.append("at resistance")
        if not detail_parts:
            nearest_str = f"${nearest:.1f}" if nearest else "N/A"
            detail_parts.append(f"no key level nearby (nearest {nearest_str})")
        detail = "; ".join(detail_parts)
        if nearest and (at_support or at_resistance):
            detail += f" ${nearest:.1f}"

        return {
            "at_support":     at_support,
            "at_resistance":  at_resistance,
            "structure_type": stype,
            "nearest_level":  round(nearest, 2) if nearest else None,
            "swing_highs":    [round(h, 2) for h in swing_highs],
            "swing_lows":     [round(l, 2) for l in swing_lows],
            "detail":         detail,
        }

    # ── 4. Candle Pattern ─────────────────────────────────────────────────────
    def check_candle_pattern(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Detect the 10 supported candle patterns from the last 3 bars.

        Returns
        -------
        pattern_found     : bool
        pattern_name      : str
        pattern_direction : "bullish" | "bearish" | "neutral"
        detail            : explanation
        """
        if len(df) < 3:
            return {
                "pattern_found":     False,
                "pattern_name":      "none",
                "pattern_direction": "neutral",
                "detail":            "Insufficient candle data",
            }

        c0 = df.iloc[-1]   # current (most recent)
        c1 = df.iloc[-2]   # previous
        c2 = df.iloc[-3]   # two bars ago

        def _body(c):
            return abs(c["close"] - c["open"])

        def _range(c):
            return c["high"] - c["low"]

        def _upper_wick(c):
            return c["high"] - max(c["open"], c["close"])

        def _lower_wick(c):
            return min(c["open"], c["close"]) - c["low"]

        def _is_bull(c):
            return c["close"] > c["open"]

        def _is_bear(c):
            return c["close"] < c["open"]

        body0  = _body(c0)
        rng0   = _range(c0)
        body1  = _body(c1)
        rng1   = _range(c1)
        ratio0 = body0 / rng0 if rng0 > 0 else 0
        ratio1 = body1 / rng1 if rng1 > 0 else 0

        # ── Doji ──────────────────────────────────────────────────────────────
        if ratio0 < 0.1 and rng0 > 0:
            return {
                "pattern_found":     True,
                "pattern_name":      "Doji",
                "pattern_direction": "neutral",
                "detail":            "Open ≈ close — indecision",
            }

        # ── Hammer ────────────────────────────────────────────────────────────
        lw0 = _lower_wick(c0)
        uw0 = _upper_wick(c0)
        if lw0 >= body0 * 2 and uw0 <= body0 * 0.5 and body0 > 0:
            return {
                "pattern_found":     True,
                "pattern_name":      "Hammer",
                "pattern_direction": "bullish",
                "detail":            "Long lower wick, small body — bullish reversal",
            }

        # ── Shooting Star ─────────────────────────────────────────────────────
        if uw0 >= body0 * 2 and lw0 <= body0 * 0.5 and body0 > 0:
            return {
                "pattern_found":     True,
                "pattern_name":      "Shooting Star",
                "pattern_direction": "bearish",
                "detail":            "Long upper wick, small body — bearish reversal",
            }

        # ── Pinbar (wick ≥ 2× body, regardless of direction) ─────────────────
        max_wick = max(lw0, uw0)
        if max_wick >= body0 * 2 and body0 > 0:
            direction = "bullish" if lw0 > uw0 else "bearish"
            return {
                "pattern_found":     True,
                "pattern_name":      "Pinbar",
                "pattern_direction": direction,
                "detail":            f"Wick {max_wick:.1f} ≥ 2× body {body0:.1f}",
            }

        # ── Bullish Engulfing ─────────────────────────────────────────────────
        if (_is_bull(c0) and _is_bear(c1)
                and c0["open"] <= c1["close"]
                and c0["close"] >= c1["open"]):
            return {
                "pattern_found":     True,
                "pattern_name":      "Bullish Engulfing",
                "pattern_direction": "bullish",
                "detail":            "Green candle fully engulfs prior red candle",
            }

        # ── Bearish Engulfing ─────────────────────────────────────────────────
        if (_is_bear(c0) and _is_bull(c1)
                and c0["open"] >= c1["close"]
                and c0["close"] <= c1["open"]):
            return {
                "pattern_found":     True,
                "pattern_name":      "Bearish Engulfing",
                "pattern_direction": "bearish",
                "detail":            "Red candle fully engulfs prior green candle",
            }

        # ── Inside Bar ────────────────────────────────────────────────────────
        if c0["high"] <= c1["high"] and c0["low"] >= c1["low"]:
            return {
                "pattern_found":     True,
                "pattern_name":      "Inside Bar",
                "pattern_direction": "neutral",
                "detail":            "Current candle fully inside previous candle",
            }

        # ── Breakout Bar ──────────────────────────────────────────────────────
        prev_range_high = df.iloc[-6:-1]["high"].max() if len(df) >= 6 else c1["high"]
        prev_range_low  = df.iloc[-6:-1]["low"].min()  if len(df) >= 6 else c1["low"]
        if c0["close"] > prev_range_high:
            return {
                "pattern_found":     True,
                "pattern_name":      "Breakout Bar",
                "pattern_direction": "bullish",
                "detail":            f"Closes above 5-bar range high ${prev_range_high:.1f}",
            }
        if c0["close"] < prev_range_low:
            return {
                "pattern_found":     True,
                "pattern_name":      "Breakout Bar",
                "pattern_direction": "bearish",
                "detail":            f"Closes below 5-bar range low ${prev_range_low:.1f}",
            }

        # ── Morning Star (3-candle bullish reversal) ──────────────────────────
        if (_is_bear(c2)                       # large bearish candle
                and ratio1 < 0.3               # small middle candle
                and _is_bull(c0)               # large bullish candle
                and c0["close"] > (c2["open"] + c2["close"]) / 2):
            return {
                "pattern_found":     True,
                "pattern_name":      "Morning Star",
                "pattern_direction": "bullish",
                "detail":            "3-candle bullish reversal",
            }

        # ── Evening Star (3-candle bearish reversal) ──────────────────────────
        if (_is_bull(c2)                       # large bullish candle
                and ratio1 < 0.3               # small middle candle
                and _is_bear(c0)               # large bearish candle
                and c0["close"] < (c2["open"] + c2["close"]) / 2):
            return {
                "pattern_found":     True,
                "pattern_name":      "Evening Star",
                "pattern_direction": "bearish",
                "detail":            "3-candle bearish reversal",
            }

        return {
            "pattern_found":     False,
            "pattern_name":      "none",
            "pattern_direction": "neutral",
            "detail":            "No recognised pattern on last 3 candles",
        }

    # ── 5. Session ────────────────────────────────────────────────────────────
    def check_session(self, current_time: datetime | None = None) -> dict[str, Any]:
        """
        Classify the current trading session using UAE-accurate world_sessions
        (with UTC fallback if world_sessions is unavailable).
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Normalise to UTC for fallback
        if current_time.tzinfo is not None:
            utc_hour = current_time.utctimetuple().tm_hour
            utc_min  = current_time.utctimetuple().tm_min
        else:
            utc_hour = current_time.hour
            utc_min  = current_time.minute
        gst_hour = (utc_hour + 4) % 24

        # ── Try world_sessions (UAE-accurate) ─────────────────────────────────
        try:
            from world_sessions import get_active_sessions, get_session_summary_line
            _gst = timezone(timedelta(hours=4))
            _now_uae = current_time.astimezone(_gst)
            active = get_active_sessions(_now_uae)
            keys   = {s["key"] for s in active}

            if "london" in keys and "newyork" in keys:
                session   = "London/NY Overlap"
                high_prob = True
            elif "newyork" in keys:
                session   = "New York"
                high_prob = True
            elif "london" in keys:
                session   = "London"
                high_prob = True
            elif keys & {"tokyo", "hongkong", "shanghai"}:
                session   = "Asian"
                high_prob = False
            else:
                session   = "Off-hours"
                high_prob = False

            detail = (
                f"UTC {utc_hour:02d}:{utc_min:02d}  |  UAE {gst_hour:02d}:{utc_min:02d}  "
                f"→ {session} session  [{get_session_summary_line(_now_uae)}]"
            )
        except Exception:
            # Fallback: UTC-hour approximation
            t = utc_hour + utc_min / 60.0
            if 12.0 <= t < 15.0:
                session, high_prob = "London/NY Overlap", True
            elif 7.0 <= t < 12.0:
                session, high_prob = "London", True
            elif 13.0 <= t < 17.0:
                session, high_prob = "New York", True
            elif 0.0 <= t < 7.0:
                session, high_prob = "Asian", False
            else:
                session, high_prob = "Off-hours", False
            detail = (
                f"UTC {utc_hour:02d}:{utc_min:02d}  |  GST {gst_hour:02d}:{utc_min:02d}  "
                f"→ {session} session"
            )

        return {
            "session":                   session,
            "is_high_probability_window": high_prob,
            "utc_hour":                  utc_hour,
            "gst_hour":                  gst_hour,
            "detail":                    detail,
        }

    # ── 6. Volatility ─────────────────────────────────────────────────────────
    def check_volatility(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Compare current ATR to 50-period average ATR.

        Returns
        -------
        volatility           : "high" | "normal" | "low"
        atr_value            : current ATR
        avg_atr              : 50-period average ATR
        suitable_for_trading : bool  (False if extreme in either direction)
        detail               : explanation
        """
        atr_col = "atr"
        if atr_col not in df.columns or df[atr_col].isna().all():
            return {
                "volatility":            "normal",
                "atr_value":             None,
                "avg_atr":               None,
                "suitable_for_trading":  True,
                "detail":                "ATR column unavailable — assuming normal",
            }

        current_atr = df[atr_col].iloc[-1]
        avg_atr     = df[atr_col].tail(50).mean()

        if avg_atr == 0 or math.isnan(avg_atr):
            return {
                "volatility":            "normal",
                "atr_value":             round(current_atr, 2),
                "avg_atr":               None,
                "suitable_for_trading":  True,
                "detail":                "ATR average unavailable",
            }

        ratio = current_atr / avg_atr

        if ratio > ATR_HIGH_MULT:
            volatility = "high"
            suitable   = False
            detail     = (f"ATR {current_atr:.1f} > {ATR_HIGH_MULT}× avg {avg_atr:.1f} "
                          f"— too volatile ({ratio:.1f}×)")
        elif ratio < ATR_LOW_MULT:
            volatility = "low"
            suitable   = False
            detail     = (f"ATR {current_atr:.1f} < {ATR_LOW_MULT}× avg {avg_atr:.1f} "
                          f"— too quiet ({ratio:.1f}×)")
        else:
            volatility = "normal"
            suitable   = True
            detail     = (f"ATR {current_atr:.1f} vs avg {avg_atr:.1f} "
                          f"({ratio:.1f}× — normal)")

        return {
            "volatility":            volatility,
            "atr_value":             round(current_atr, 2),
            "avg_atr":               round(avg_atr, 2),
            "suitable_for_trading":  suitable,
            "detail":                detail,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  detect_rsi_divergence — standalone swing-based RSI divergence scanner
# ══════════════════════════════════════════════════════════════════════════════

def detect_rsi_divergence(df: pd.DataFrame, d1_bias: str = "neutral") -> dict:
    """
    Scan the last 50 candles for RSI divergence patterns.

    Supports four types:
      bullish          — price lower low, RSI higher low  (long signal)
      bearish          — price higher high, RSI lower high (short signal)
      hidden_bullish   — price higher low, RSI lower low  (uptrend continuation)
      hidden_bearish   — price lower high, RSI higher high (downtrend continuation)

    Parameters
    ----------
    df       : DataFrame with columns 'close', 'low', 'high', 'rsi'
    d1_bias  : "bullish"/"bearish"/"neutral" — used to gate hidden divergences

    Returns
    -------
    dict with keys: divergence_found, divergence_type, strength,
                    signal_direction, price_swing1, price_swing2,
                    rsi_swing1, rsi_swing2, bars_since_divergence,
                    note, confidence_boost
    """
    _NONE: dict = {
        "divergence_found":    False,
        "divergence_type":     None,
        "strength":            None,
        "signal_direction":    None,
        "price_swing1":        0.0,
        "price_swing2":        0.0,
        "rsi_swing1":          0.0,
        "rsi_swing2":          0.0,
        "bars_since_divergence": 0,
        "note":                "No divergence",
        "confidence_boost":    0.0,
    }
    try:
        if df is None or len(df) < 20:
            return _NONE

        # Normalise column names
        close_col = "close" if "close" in df.columns else "Close"
        low_col   = "low"   if "low"   in df.columns else "Low"
        high_col  = "high"  if "high"  in df.columns else "High"
        rsi_col   = "rsi"   if "rsi"   in df.columns else "RSI"

        if rsi_col not in df.columns:
            return _NONE

        window = df.tail(50).reset_index(drop=True)
        n      = len(window)

        # ── Find swing lows (local minima on price low) ───────────────────────
        swing_lows: list[tuple[int, float, float]] = []   # (idx, price_low, rsi)
        for i in range(2, n - 1):
            pl = float(window[low_col].iloc[i])
            if (pl < float(window[low_col].iloc[i - 1])
                    and pl < float(window[low_col].iloc[i + 1])
                    and pl < float(window[low_col].iloc[i - 2])):
                rsi_v = float(window[rsi_col].iloc[i])
                swing_lows.append((i, pl, rsi_v))

        # ── Find swing highs (local maxima on price high) ─────────────────────
        swing_highs: list[tuple[int, float, float]] = []  # (idx, price_high, rsi)
        for i in range(2, n - 1):
            ph = float(window[high_col].iloc[i])
            if (ph > float(window[high_col].iloc[i - 1])
                    and ph > float(window[high_col].iloc[i + 1])
                    and ph > float(window[high_col].iloc[i - 2])):
                rsi_v = float(window[rsi_col].iloc[i])
                swing_highs.append((i, ph, rsi_v))

        # Need at least 2 swings to compare
        if len(swing_lows) >= 2:
            # Use last two swing lows
            (i1, pl1, r1), (i2, pl2, r2) = swing_lows[-2], swing_lows[-1]
            bars_since = n - 1 - i2
            within_20  = i2 >= n - 20   # swing2 within last 20 candles

            # ── BULLISH DIVERGENCE ────────────────────────────────────────────
            if within_20 and pl2 < pl1 and r2 > r1 and r2 < 50:
                strength = "STRONG" if r2 < 35 else "MODERATE"
                boost    = 1.5 if strength == "STRONG" else 1.0
                pdiff    = round(pl2 - pl1, 2)   # negative
                rdiff    = round(r2 - r1, 2)     # positive
                return {
                    "divergence_found":      True,
                    "divergence_type":       "bullish",
                    "strength":              strength,
                    "signal_direction":      "long",
                    "price_swing1":          round(pl1, 2),
                    "price_swing2":          round(pl2, 2),
                    "rsi_swing1":            round(r1, 2),
                    "rsi_swing2":            round(r2, 2),
                    "bars_since_divergence": bars_since,
                    "note":                  (
                        f"Bullish div: price {pdiff:.1f} vs RSI +{rdiff:.1f}"
                    ),
                    "confidence_boost":      boost,
                }

            # ── HIDDEN BULLISH (trend continuation, only if D1 bullish) ───────
            if (within_20 and pl2 > pl1 and r2 < r1
                    and d1_bias.lower() == "bullish"):
                pdiff = round(pl2 - pl1, 2)   # positive
                rdiff = round(r2 - r1, 2)     # negative
                return {
                    "divergence_found":      True,
                    "divergence_type":       "hidden_bullish",
                    "strength":              "MODERATE",
                    "signal_direction":      "long",
                    "price_swing1":          round(pl1, 2),
                    "price_swing2":          round(pl2, 2),
                    "rsi_swing1":            round(r1, 2),
                    "rsi_swing2":            round(r2, 2),
                    "bars_since_divergence": bars_since,
                    "note":                  (
                        f"Hidden bullish div: price +{pdiff:.1f} vs RSI {rdiff:.1f} "
                        f"(uptrend continuation)"
                    ),
                    "confidence_boost":      1.0,
                }

        if len(swing_highs) >= 2:
            (i1, ph1, r1), (i2, ph2, r2) = swing_highs[-2], swing_highs[-1]
            bars_since = n - 1 - i2
            within_20  = i2 >= n - 20

            # ── BEARISH DIVERGENCE ────────────────────────────────────────────
            if within_20 and ph2 > ph1 and r2 < r1 and r2 > 50:
                strength = "STRONG" if r2 > 65 else "MODERATE"
                boost    = 1.5 if strength == "STRONG" else 1.0
                pdiff    = round(ph2 - ph1, 2)   # positive
                rdiff    = round(r2 - r1, 2)     # negative
                return {
                    "divergence_found":      True,
                    "divergence_type":       "bearish",
                    "strength":              strength,
                    "signal_direction":      "short",
                    "price_swing1":          round(ph1, 2),
                    "price_swing2":          round(ph2, 2),
                    "rsi_swing1":            round(r1, 2),
                    "rsi_swing2":            round(r2, 2),
                    "bars_since_divergence": bars_since,
                    "note":                  (
                        f"Bearish div: price +{pdiff:.1f} vs RSI {rdiff:.1f}"
                    ),
                    "confidence_boost":      boost,
                }

            # ── HIDDEN BEARISH (trend continuation, only if D1 bearish) ───────
            if (within_20 and ph2 < ph1 and r2 > r1
                    and d1_bias.lower() == "bearish"):
                pdiff = round(ph2 - ph1, 2)   # negative
                rdiff = round(r2 - r1, 2)     # positive
                return {
                    "divergence_found":      True,
                    "divergence_type":       "hidden_bearish",
                    "strength":              "MODERATE",
                    "signal_direction":      "short",
                    "price_swing1":          round(ph1, 2),
                    "price_swing2":          round(ph2, 2),
                    "rsi_swing1":            round(r1, 2),
                    "rsi_swing2":            round(r2, 2),
                    "bars_since_divergence": bars_since,
                    "note":                  (
                        f"Hidden bearish div: price {pdiff:.1f} vs RSI +{rdiff:.1f} "
                        f"(downtrend continuation)"
                    ),
                    "confidence_boost":      1.0,
                }

        return _NONE
    except Exception:
        return {"divergence_found": False, "confidence_boost": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
#  score_confluences  — v2: always wires SMC / MTF / DXY internally
# ══════════════════════════════════════════════════════════════════════════════

def score_confluences(
    df: pd.DataFrame,
    direction: str,
    current_time: datetime | None = None,
    symbol: str = "XAUUSD",
    # Legacy kwargs accepted but ignored (callers may still pass these)
    htf_bias: dict | None = None,
    dxy_ctx: dict | None = None,
    playbook: str = "unknown",
) -> dict[str, Any]:
    """
    Run all 10 confluence checks.  v2: SMC, MTF, and DXY are always called
    directly inside this function — no need to pass htf_bias or dxy_ctx.

    Parameters
    ----------
    df           : OHLCV + indicator DataFrame
    direction    : "long" or "short"
    current_time : datetime for session check (defaults to now UTC)
    symbol       : trading symbol (used for MTF yfinance fetch)

    Returns
    -------
    confidence         : float 0–10 (weighted score)
    weighted_score     : float (raw, before cap)
    detail_lines       : list[str]  one line per factor for display
    trade_valid        : bool  (weighted_score >= 5.0)
    confluences_met    : list[dict]  (backward-compat format)
    confluences_failed : list[dict]  (backward-compat format)
    confluence_score   : int  (net pass - fail, backward compat)
    passed_count       : int
    total_checks       : int
    check_weights_earned : dict
    weighted_confidence  : float  (alias for confidence)
    raw_checks         : dict
    """
    direction = direction.lower().strip()
    is_long   = direction in ("long", "buy")

    checker       = ConfluenceChecker()
    met    : list[dict] = []
    failed : list[dict] = []
    detail_lines  : list[str] = []
    weighted_raw  = 0.0
    check_weights_earned: dict[str, float] = {}

    # yfinance symbol mapping
    yf_symbol = {"XAUUSD": "GC=F", "XAGUSD": "SI=F"}.get(symbol.upper(), symbol)

    # ── FACTOR 1: HTF Alignment (weight 2.5) ─────────────────────────────────
    _htf_raw = None
    try:
        if _MTF_OK:
            _mta    = _MTFAnalyzer()
            _htf_raw = _mta.get_htf_bias(yf_symbol)
        elif htf_bias is not None:
            # Caller passed legacy param — wrap into expected shape
            _htf_raw = htf_bias
    except Exception:
        _htf_raw = None

    if _htf_raw:
        d1  = str(_htf_raw.get("d1_trend",     "ranging")).lower()
        h4  = str(_htf_raw.get("h4_trend",     "ranging")).lower()
        ov  = str(_htf_raw.get("overall_bias", "neutral")).lower()
        aligned = (
            (is_long  and d1 == "bullish" and h4 == "bullish") or
            (not is_long and d1 == "bearish" and h4 == "bearish")
        )
        partial = (
            (is_long  and (d1 == "bullish" or h4 == "bullish")) or
            (not is_long and (d1 == "bearish" or h4 == "bearish"))
        ) and not aligned
        if aligned:
            weighted_raw += 2.0
            check_weights_earned["HTF"] = 2.0
            lbl = f"HTF aligned D1:{d1} H4:{h4}  +2.0"
            detail_lines.append(f"✓ {lbl}")
            met.append({"check": "HTF", "result": "pass",
                        "detail": f"HTF aligned D1:{d1.upper()} H4:{h4.upper()} (STRONG bias)"})
        elif partial:
            weighted_raw += 0.8
            check_weights_earned["HTF"] = 0.8
            lbl = f"HTF partial D1:{d1} H4:{h4}  +0.8"
            detail_lines.append(f"~ {lbl}")
            met.append({"check": "HTF", "result": "pass",
                        "detail": f"HTF partial D1:{d1.upper()} H4:{h4.upper()} (WEAK bias)"})
        else:
            weighted_raw -= 0.5
            check_weights_earned["HTF"] = -0.5
            detail_lines.append(f"✗ HTF conflict D1:{d1} H4:{h4}  −0.5")
            failed.append({"check": "HTF", "result": "fail",
                           "detail": f"HTF NOT aligned D1:{d1.upper()} H4:{h4.upper()} — penalty"})
    else:
        check_weights_earned["HTF"] = 0.0
        detail_lines.append("✗ HTF — data unavailable  +0.0")
        failed.append({"check": "HTF", "result": "fail", "detail": "HTF data unavailable"})

    # ── FACTOR 2: Smart Money Concepts ───────────────────────────────────────
    smc_result        = None
    smc_context       = None
    _smc_context_store = None
    if _SMC_OK:
        try:
            _sma = _SmartMoneyAnalyzer()

            # ── Try enhanced get_smc_context first ────────────────────────────
            if _SMC_ENHANCED:
                try:
                    smc_context = _sma.get_smc_context(df, direction)
                    smc_result  = {
                        "active_order_block": smc_context["active_order_block"],
                        "fvg_nearby":         smc_context["fvg_nearby"],
                        "liquidity_swept":    smc_context["liquidity_swept"],
                        "structure_aligned":  smc_context["structure_aligned"],
                        "score":              smc_context["smc_score"],
                    }
                except Exception:
                    smc_context = None
                    smc_result  = None

            # ── Fallback to smc_score() ───────────────────────────────────────
            if smc_result is None:
                smc_result = _sma.smc_score(df, direction)

            ob_ok  = smc_result.get("active_order_block", False)
            fvg_ok = smc_result.get("fvg_nearby",        False)
            liq_ok = smc_result.get("liquidity_swept",   False)
            str_ok = smc_result.get("structure_aligned", False)
            smc_pts = smc_result.get("score", 0)

            # ── Scoring — use get_smc_context confidence_adjustment if available ──
            if smc_context is not None:
                grade      = smc_context.get("entry_quality", "D")
                conf_adj   = smc_context.get("confidence_adjustment", 0.0)
                grade_label = smc_context.get("entry_quality_label", f"Grade {grade}")
                reasons     = smc_context.get("entry_reasons", [])
                reason_str  = reasons[0][:50] if reasons else ""

                if conf_adj > 0:
                    weighted_raw += conf_adj * 2   # scale +1.0 adj → +2.0 weight
                    check_weights_earned["SMC"] = check_weights_earned.get("SMC", 0.0) + (conf_adj * 2)
                    detail_lines.append(
                        f"✓ SMC {grade_label[:40]}  +{conf_adj * 2:.1f}"
                    )
                elif conf_adj == 0.0:
                    check_weights_earned.setdefault("SMC", 0.0)
                    detail_lines.append(f"~ SMC {grade_label[:40]}   0.0")
                else:
                    check_weights_earned.setdefault("SMC", 0.0)
                    detail_lines.append(f"✗ SMC {grade_label[:40]}  {conf_adj * 2:.1f}")
            else:
                # Legacy scoring path
                if ob_ok:
                    weighted_raw += 2.0
                    check_weights_earned["SMC"] = check_weights_earned.get("SMC", 0.0) + 2.0
                    detail_lines.append("✓ Order block active         +2.0")
                else:
                    check_weights_earned.setdefault("SMC", 0.0)
                    detail_lines.append("✗ No active order block      +0.0")
                if fvg_ok:
                    weighted_raw += 0.3
                    check_weights_earned["SMC"] = check_weights_earned.get("SMC", 0.0) + 0.3
                    detail_lines.append("✓ Fair value gap nearby      +0.3")
                if liq_ok:
                    weighted_raw += 0.2
                    check_weights_earned["SMC"] = check_weights_earned.get("SMC", 0.0) + 0.2
                    detail_lines.append("✓ Liquidity sweep detected   +0.2")

            # SMC check for backward compat met/failed
            smc_passes = smc_pts >= 2
            _add_check(
                met, failed, smc_passes,
                name="SMC",
                agrees_label=(
                    f"SMC {smc_pts}/4 — OB={'✓' if ob_ok else '✗'} "
                    f"FVG={'✓' if fvg_ok else '✗'} "
                    f"Liq={'✓' if liq_ok else '✗'} "
                    f"Str={'✓' if str_ok else '✗'}"
                ),
                fails_label=(
                    f"SMC {smc_pts}/4 too low — OB={'✓' if ob_ok else '✗'} "
                    f"FVG={'✓' if fvg_ok else '✗'} "
                    f"Liq={'✓' if liq_ok else '✗'} "
                    f"Str={'✓' if str_ok else '✗'}"
                ),
            )

            # Store full smc_context in raw_checks for downstream use
            if smc_context is not None:
                _smc_context_store = smc_context

        except Exception as e:
            detail_lines.append(f"✗ SMC — error: {e}  +0.0")
            check_weights_earned.setdefault("SMC", 0.0)
            failed.append({"check": "SMC", "result": "fail", "detail": f"SMC error: {e}"})
    else:
        detail_lines.append("✗ SMC — module unavailable  +0.0")
        check_weights_earned.setdefault("SMC", 0.0)
        failed.append({"check": "SMC", "result": "fail", "detail": "SMC module unavailable"})

    # ── FACTOR 3: Trend vs EMA200 (weight 1.5) ────────────────────────────────
    trend_result = checker.check_trend(df)
    t_dir = trend_result["primary_trend"]
    t_agrees = (is_long and t_dir == "bullish") or (not is_long and t_dir == "bearish")
    if t_agrees:
        weighted_raw += 1.5
        check_weights_earned["Trend"] = 1.5
        detail_lines.append(f"✓ Trend {t_dir}  EMA200 aligned   +1.5")
        met.append({"check": "Trend", "result": "pass",
                    "detail": f"{t_dir.upper()} ({trend_result['detail']})"})
    else:
        check_weights_earned["Trend"] = 0.0
        detail_lines.append(f"✗ Trend {t_dir} — against {direction}  +0.0")
        failed.append({"check": "Trend", "result": "fail",
                       "detail": f"Trend is {t_dir.upper()} — opposes {direction}"})

    # ── FACTOR 4: Structure S/R (weight up to 2.0) ────────────────────────────
    structure = checker.check_structure(df)   # kept for raw_checks
    _sr_result: dict | None = None
    if _SR_OK and _get_sr_levels is not None:
        try:
            _cur_price  = float(df["close"].iloc[-1] if "close" in df.columns
                                else df["Close"].iloc[-1])
            _sr_result  = _get_sr_levels(df, _cur_price)
            _nr         = _sr_result["nearest_resistance"]
            _ns         = _sr_result["nearest_support"]
            _at_key     = _sr_result["at_key_level"]

            # LONG: at support?   SHORT: at resistance?
            if is_long:
                _at_sr = _ns["distance_pct"] < 0.5
                _lvl   = _ns
                _side  = "support"
            else:
                _at_sr = _nr["distance_pct"] < 0.5
                _lvl   = _nr
                _side  = "resistance"

            if _at_sr:
                _w = 2.0 if _at_key else 1.5
                # Liquidity cluster boost: +0.5 if near largest stop cluster
                if _LIQ_OK:
                    try:
                        _liq_boost = _build_liquidity_map(df, _cur_price)
                        _lc = _liq_boost.get("largest_cluster")
                        if _lc and _lc.get("distance_usd", 9999) < 8.0:
                            _w += 0.5
                            detail_lines.append(
                                f"  ↑ Liquidity cluster near ${_lc['price']:,.2f} "
                                f"(+$0.5 S/R boost)"
                            )
                    except Exception:
                        pass
                weighted_raw += _w
                check_weights_earned["Structure"] = _w
                detail_lines.append(
                    f"\u2713 At {_side}: {_lvl['label']} "
                    f"${_lvl['price']:,.2f}  +{_w:.1f}"
                )
                met.append({"check": "Structure", "result": "pass",
                            "detail": f"At {_side}: {_lvl['label']} ${_lvl['price']:,.2f}"})
            else:
                check_weights_earned["Structure"] = 0.0
                _nearest = _ns if is_long else _nr
                detail_lines.append(
                    f"\u2717 No key S/R nearby \u2014 nearest: "
                    f"{_nearest['label']} ${_nearest['price']:,.2f} "
                    f"({_nearest['distance_usd']:.1f} away)  +0.0"
                )
                failed.append({"check": "Structure", "result": "miss",
                               "detail": f"No {_side} within 0.5% — nearest {_nearest['label']}"})
        except Exception as _sre:
            _sr_result = None
            check_weights_earned.setdefault("Structure", 0.0)
            detail_lines.append(f"\u2717 S/R — error: {_sre}  +0.0")
            failed.append({"check": "Structure", "result": "miss",
                           "detail": f"S/R mapper error: {_sre}"})
    else:
        # Fallback: original check_structure()
        s_type   = structure["structure_type"]
        at_level = (
            (is_long  and s_type in ("support", "both")) or
            (not is_long and s_type in ("resistance", "both"))
        )
        if at_level:
            weighted_raw += 1.5
            check_weights_earned["Structure"] = 1.5
            lv = structure.get("nearest_level", "")
            detail_lines.append(f"\u2713 At S/R level ${lv}   +1.5")
            met.append({"check": "Structure", "result": "pass",
                        "detail": f"At {s_type} level \u2014 ${structure.get('nearest_level','')}"})
        else:
            check_weights_earned["Structure"] = 0.0
            detail_lines.append(f"\u2717 No key S/R level nearby    +0.0")
            if s_type == "none":
                failed.append({"check": "Structure", "result": "miss",
                               "detail": structure["detail"]})
            else:
                failed.append({"check": "Structure", "result": "fail",
                               "detail": f"At {s_type} but need {'support' if is_long else 'resistance'}"})

    # ── FACTOR 5: RSI Momentum (weight 1.0) ───────────────────────────────────
    momentum = checker.check_momentum(df)
    m_dir    = momentum["momentum"]
    rsi_val  = momentum.get("rsi_value") or 50.0
    rsi_strong  = (is_long and rsi_val < 35)  or (not is_long and rsi_val > 65)
    rsi_aligned = (is_long and rsi_val < 45)  or (not is_long and rsi_val > 55)
    if rsi_strong:
        weighted_raw += 1.0
        check_weights_earned["Momentum"] = 1.0
        detail_lines.append(f"✓ RSI strong {rsi_val:.1f}           +1.0")
    elif rsi_aligned:
        weighted_raw += 0.5
        check_weights_earned["Momentum"] = 0.5
        detail_lines.append(f"~ RSI aligned {rsi_val:.1f}          +0.5")
    else:
        check_weights_earned["Momentum"] = 0.0
        detail_lines.append(f"✗ RSI neutral {rsi_val:.1f}          +0.0")
    # Backward compat met/failed entry
    if m_dir == "neutral":
        met.append({"check": "Momentum", "result": "neutral", "detail": momentum["detail"]})
    else:
        m_agrees = (is_long and m_dir == "bullish") or (not is_long and m_dir == "bearish")
        _add_check(
            met, failed, m_agrees,
            name="Momentum",
            agrees_label=f"RSI {momentum['rsi_value']} ({momentum['rsi_zone']}) {momentum['macd_signal']}",
            fails_label=f"Momentum is {m_dir.upper()} — opposes {direction}",
        )

    # ── FACTOR 5B: RSI Divergence (bonus, weight up to +1.5) ─────────────────
    _div_result: dict = {"divergence_found": False, "confidence_boost": 0.0}
    try:
        # Pass D1 bias if available from HTF result
        _d1_bias = str((_htf_raw or {}).get("d1_trend", "neutral")).lower()
        _div_result = detect_rsi_divergence(df, d1_bias=_d1_bias)
        if _div_result.get("divergence_found"):
            div_type = _div_result["divergence_type"]
            div_dir  = _div_result.get("signal_direction")
            boost    = _div_result.get("confidence_boost", 0.0)
            strength = _div_result.get("strength", "MODERATE")
            div_label = div_type.replace("_", " ").title()

            if div_dir == direction:
                weighted_raw += boost
                check_weights_earned["RSI_Div"] = boost
                detail_lines.append(
                    f"✓ {div_label} divergence ({strength})  +{boost:.1f}"
                )
                met.append({
                    "check":  "RSI_Divergence",
                    "result": "pass",
                    "detail": _div_result["note"],
                })
            else:
                check_weights_earned["RSI_Div"] = 0.0
                detail_lines.append(
                    f"⚠ {div_label} divergence detected but opposes {direction}  0.0"
                )
                check_weights_earned["RSI_Div"] = 0.0
        else:
            check_weights_earned["RSI_Div"] = 0.0
    except Exception as _dive:
        check_weights_earned.setdefault("RSI_Div", 0.0)

    # ── FACTOR 6: Macro Context — DXY + US10Y Yields (weight 1.0) ──────────────
    _macro_ctx_live = None
    try:
        if _DXY_OK:
            from dxy_correlation import get_macro_context as _get_macro_ctx
            _macro_ctx_live = _get_macro_ctx(direction)
        elif dxy_ctx is not None:
            _macro_ctx_live = dxy_ctx   # legacy plain dxy_ctx fallback
    except Exception:
        _macro_ctx_live = dxy_ctx       # fall back to legacy param

    dxy_result = None
    if _macro_ctx_live and _macro_ctx_live.get("available"):
        try:
            _macro_confirmed = _macro_ctx_live.get("macro_confirmed", False)
            _macro_opposed   = _macro_ctx_live.get("macro_opposed",   False)
            _macro_summary   = _macro_ctx_live.get("summary", "")
            _macro_score     = _macro_ctx_live.get("macro_score", 0.0)
            _dxy_trend       = _macro_ctx_live.get("dxy_trend", "sideways")
            _dxy_rsi         = _macro_ctx_live.get("dxy_rsi", 50.0)

            if _macro_confirmed:
                weighted_raw += 1.0
                check_weights_earned["DXY"] = 1.0
                detail_lines.append(f"✓ {_macro_summary}  +1.0")
                met.append({"check": "DXY", "result": "pass",
                            "detail": _macro_summary})
            elif _macro_opposed:
                weighted_raw -= 0.5
                check_weights_earned["DXY"] = -0.5
                detail_lines.append(f"✗ {_macro_summary}  −0.5")
                failed.append({"check": "DXY", "result": "fail",
                               "detail": _macro_summary})
            else:
                check_weights_earned["DXY"] = 0.0
                detail_lines.append(f"→ {_macro_summary}  +0.0")
                met.append({"check": "DXY", "result": "neutral",
                            "detail": _macro_summary})

            dxy_result = {
                "dxy_trend":         _dxy_trend,
                "dxy_rsi":           _dxy_rsi,
                "momentum_strength": _macro_ctx_live.get("momentum_strength", "weak"),
                "macro_confirmed":   _macro_confirmed,
                "macro_opposed":     _macro_opposed,
                "macro_score":       _macro_score,
                "macro_summary":     _macro_summary,
            }
        except Exception as e:
            check_weights_earned.setdefault("DXY", 0.0)
            detail_lines.append(f"✗ Macro — error: {e}  +0.0")
            failed.append({"check": "DXY", "result": "fail", "detail": f"Macro error: {e}"})
    elif dxy_ctx is not None and dxy_ctx.get("available"):
        # Legacy plain dxy_ctx path (no yields data)
        try:
            _dxya     = _DXYCorrelation()
            dxy_trend = dxy_ctx.get("dxy_trend", "sideways")
            alignment = _dxya.dxy_gold_alignment(dxy_trend, direction)
            dxy_result = {
                "dxy_trend":         dxy_trend,
                "dxy_rsi":           dxy_ctx.get("dxy_rsi", 50.0),
                "momentum_strength": dxy_ctx.get("momentum_strength", "weak"),
                "aligned":           alignment["aligned"],
                "correlation_note":  alignment["correlation_note"],
                "conflict_severity": alignment["conflict_severity"],
            }
            if alignment["aligned"]:
                weighted_raw += 1.0
                check_weights_earned["DXY"] = 1.0
                detail_lines.append(f"✓ DXY aligned ({dxy_trend})      +1.0")
                met.append({"check": "DXY", "result": "pass",
                            "detail": f"DXY {dxy_trend.upper()} — {alignment['correlation_note']}"})
            else:
                check_weights_earned["DXY"] = 0.0
                detail_lines.append("✗ DXY conflict or ranging    +0.0")
                failed.append({"check": "DXY", "result": "fail",
                               "detail": f"DXY {dxy_trend.upper()} — {alignment['correlation_note']}"})
        except Exception as e:
            check_weights_earned.setdefault("DXY", 0.0)
            detail_lines.append(f"✗ DXY — error: {e}  +0.0")
            failed.append({"check": "DXY", "result": "fail", "detail": f"DXY error: {e}"})
    else:
        check_weights_earned.setdefault("DXY", 0.0)
        detail_lines.append("✗ Macro/DXY — unavailable    +0.0")
        met.append({"check": "DXY", "result": "neutral", "detail": "DXY/Macro data unavailable"})

    # ── FACTOR 7: Candle Pattern (weight 0.5) ─────────────────────────────────
    candle  = checker.check_candle_pattern(df)
    c_found = candle.get("pattern_found", False)
    c_dir   = str(candle.get("pattern_direction", "neutral")).lower()
    c_ok    = c_found and (c_dir in ("neutral",) or
              (is_long and c_dir == "bullish") or
              (not is_long and c_dir == "bearish"))
    if c_ok:
        weighted_raw += 0.5
        check_weights_earned["Candle"] = 0.5
        pname = candle.get("pattern_name", "Pattern")
        detail_lines.append(f"✓ {pname} candle              +0.5")
        met.append({"check": "Candle", "result": "pass",
                    "detail": f"{pname} ({candle.get('detail', '')})"})
    else:
        check_weights_earned["Candle"] = 0.0
        if c_found:
            detail_lines.append(f"✗ Candle {c_dir} opposes {direction}  +0.0")
            failed.append({"check": "Candle", "result": "fail",
                           "detail": f"{candle.get('pattern_name','—')} is {c_dir} — opposes {direction}"})
        else:
            detail_lines.append("✗ No confirming candle       +0.0")
            failed.append({"check": "Candle", "result": "miss",
                           "detail": "No pattern detected on last 3 candles"})

    # ── FACTOR 8: Session Quality (weight varies by session) ──────────────────
    session = checker.check_session(current_time)
    sess_nm = session.get("session", "Unknown")
    # Assign weight based on session quality
    if "Overlap" in sess_nm or "overlap" in sess_nm:
        _sess_weight = 1.5
    elif "London" in sess_nm or "New York" in sess_nm or "NewYork" in sess_nm:
        _sess_weight = 1.0
    elif "Asian" in sess_nm or "asian" in sess_nm:
        _sess_weight = 0.3
    else:  # Off-hours or Unknown
        _sess_weight = 0.0
    if _sess_weight > 0:
        weighted_raw += _sess_weight
        check_weights_earned["Session"] = _sess_weight
        detail_lines.append(f"✓ Session: {sess_nm}         +{_sess_weight}")
        met.append({"check": "Session", "result": "pass",
                    "detail": f"{sess_nm} (weight {_sess_weight})"})
    else:
        check_weights_earned["Session"] = 0.0
        detail_lines.append(f"✗ Session: {sess_nm}         +0.0")
        failed.append({"check": "Session", "result": "fail",
                       "detail": f"{sess_nm} — low-probability window"})

    # ── FACTOR 9: Volatility (weight 0.5, bonus) ─────────────────────────────
    vol = checker.check_volatility(df)
    if vol.get("suitable_for_trading", False):
        weighted_raw += 0.5
        check_weights_earned["Volatility"] = 0.5
        detail_lines.append(f"✓ {vol.get('volatility','normal').capitalize()} ATR — suitable  +0.5")
        met.append({"check": "Volatility", "result": "pass",
                    "detail": f"{vol.get('volatility','').capitalize()} ATR — suitable ({vol.get('detail','')})"})
    else:
        check_weights_earned["Volatility"] = 0.0
        detail_lines.append(f"✗ Volatility not suitable    +0.0")
        failed.append({"check": "Volatility", "result": "fail",
                       "detail": f"{vol.get('volatility','').capitalize()} ATR — not suitable"})

    # ── FACTOR 10: Volume (class-based weight) ────────────────────────────────
    _volume_result: dict = {}
    try:
        from volume_analyzer import check_volume_confluence
        vol_conf  = check_volume_confluence(df, direction, playbook)
        _vol_cls  = vol_conf.get("volume_class", "normal")
        _vol_rat  = vol_conf.get("volume_ratio", 1.0) or 1.0
        # Class-based weight: high=1.0, normal=0.5, low=0.0, very_low=-0.5
        if _vol_cls in ("high", "exceptional"):
            _vol_w = 1.0
        elif _vol_cls == "normal":
            _vol_w = 0.5
        elif _vol_cls == "low":
            _vol_w = 0.0
        else:  # very_low
            _vol_w = -0.5
        # Further penalise volume climax (exhaustion)
        if vol_conf.get("climax"):
            _vol_w -= 1.5
            detail_lines.append("✗ Volume climax — exhaustion  −1.5")
        weighted_raw += _vol_w
        for line in vol_conf.get("details", []):
            detail_lines.append(f"  Volume: {line}")
        check_weights_earned["Volume"] = round(_vol_w, 2)
        if _vol_w > 0:
            met.append({"check": "Volume", "result": "pass",
                         "detail": f"{_vol_cls} volume {_vol_rat:.2f}x  +{_vol_w}"})
        elif _vol_w == 0:
            met.append({"check": "Volume", "result": "neutral",
                         "detail": f"{_vol_cls} volume {_vol_rat:.2f}x  +0.0"})
        else:
            failed.append({"check": "Volume", "result": "fail",
                           "detail": f"{_vol_cls} volume {_vol_rat:.2f}x  {_vol_w}"})
        _volume_result = vol_conf
    except Exception as _ve:
        detail_lines.append(f"✗ Volume — error: {_ve}  +0.0")
        check_weights_earned["Volume"] = 0.0

    # ── FACTOR 11: COT (Commitment of Traders) (weight up to ±1.0) ───────────
    _cot_result: dict | None = None
    if _COT_OK:
        try:
            _cot_data   = _fetch_cot_data()
            _cot_sig    = _get_cot_signal(direction, _cot_data)
            _cot_boost  = float(_cot_sig.get("boost", 0.0))
            _cot_note   = _cot_sig.get("note", "")
            weighted_raw += _cot_boost
            check_weights_earned["COT"] = round(_cot_boost, 2)
            _sign = f"+{_cot_boost:.1f}" if _cot_boost >= 0 else f"{_cot_boost:.1f}"
            if _cot_sig.get("aligned"):
                detail_lines.append(f"\u2713 COT aligned — {_cot_note}  {_sign}")
                met.append({"check": "COT", "result": "pass", "detail": _cot_note})
            elif _cot_sig.get("opposed"):
                detail_lines.append(f"\u2717 COT opposes — {_cot_note}  {_sign}")
                failed.append({"check": "COT", "result": "fail", "detail": _cot_note})
            else:
                detail_lines.append(f"~ COT neutral — {_cot_note}  {_sign}")
                met.append({"check": "COT", "result": "neutral", "detail": _cot_note})
            _cot_result = _cot_sig
        except Exception as _cote:
            detail_lines.append(f"\u2717 COT — error: {_cote}  +0.0")
            check_weights_earned["COT"] = 0.0


    # -- FACTOR 12+: 14 Technical Indicators (combined weight: max +/-3.0) -------
    _ind_data: dict = {}
    if _IND_OK:
        try:
            inds = _get_indicators(df)
            _ind_data = inds
            ind_score = 0.0
            ind_lines_local: list[str] = []

            def _aligned(bias: str, dir_: str) -> bool:
                b = bias.lower()
                return ('bull' in b) if dir_ == 'long' else ('bear' in b)

            def _opposed_ind(bias: str, dir_: str) -> bool:
                b = bias.lower()
                return ('bear' in b) if dir_ == 'long' else ('bull' in b)

            al_i = inds.get('alligator', {})
            if al_i.get('sleeping'):
                ind_score -= 0.5
                ind_lines_local.append('\u26a0 Alligator SLEEPING  -0.5')
            elif _aligned(al_i.get('bias', 'neutral'), direction):
                w = 1.0 if 'EATING' in al_i.get('state', '') else 0.5
                ind_score += w
                ind_lines_local.append(f'\u2713 Alligator {al_i.get("state","")} +{w}')

            adx_i = inds.get('adx', {})
            if adx_i.get('trending') and _aligned(adx_i.get('bias', ''), direction):
                w = 0.8 if adx_i.get('strength') == 'STRONG' else 0.4
                ind_score += w
                ind_lines_local.append(f'\u2713 ADX {adx_i.get("adx",0):.0f} {adx_i.get("strength","")}  +{w}')
            elif not adx_i.get('trending'):
                ind_score -= 0.3
                ind_lines_local.append('~ ADX weak  -0.3')

            mac_i = inds.get('macd', {})
            if mac_i.get('bullish_cross') and direction == 'long':
                ind_score += 1.0
                ind_lines_local.append('\u2713 MACD bullish crossover  +1.0')
            elif mac_i.get('bearish_cross') and direction == 'short':
                ind_score += 1.0
                ind_lines_local.append('\u2713 MACD bearish crossover  +1.0')
            elif _aligned(mac_i.get('bias', ''), direction):
                ind_score += 0.5
                ind_lines_local.append('~ MACD aligned  +0.5')

            st_i = inds.get('stoch_rsi', {})
            if st_i.get('bullish_cross') and st_i.get('oversold') and direction == 'long':
                ind_score += 1.0
                ind_lines_local.append('\u2713 StochRSI oversold bullish cross  +1.0')
            elif st_i.get('bearish_cross') and st_i.get('overbought') and direction == 'short':
                ind_score += 1.0
                ind_lines_local.append('\u2713 StochRSI overbought bearish cross  +1.0')
            elif _aligned(st_i.get('bias', ''), direction):
                ind_score += 0.4
                ind_lines_local.append('~ StochRSI aligned  +0.4')

            ich_i = inds.get('ichimoku', {})
            if _aligned(ich_i.get('bias', ''), direction):
                w = 1.0 if 'strongly' in ich_i.get('bias', '') else 0.6
                ind_score += w
                ind_lines_local.append(f'\u2713 Ichimoku cloud aligned  +{w}')
            elif ich_i.get('in_cloud'):
                ind_lines_local.append('~ Ichimoku in cloud  0.0')

            vw_i = inds.get('vwap', {})
            if _aligned(vw_i.get('bias', ''), direction):
                ind_score += 0.5
                ind_lines_local.append('\u2713 VWAP aligned  +0.5')

            sq_i = inds.get('squeeze', {})
            if sq_i.get('squeeze_off') and _aligned(sq_i.get('bias', ''), direction):
                ind_score += 0.8
                ind_lines_local.append('\u2713 BB Squeeze fired  +0.8')
            elif sq_i.get('squeeze_on'):
                ind_score += 0.3
                ind_lines_local.append('~ Squeeze building  +0.3')

            su_i = inds.get('supertrend', {})
            if _aligned(su_i.get('bias', ''), direction):
                flipped_note = " (flipped)" if su_i.get('just_flipped') else ""
                w = 1.0 if su_i.get('just_flipped') else 0.7
                ind_score += w
                ind_lines_local.append(f'\u2713 Supertrend aligned{flipped_note}  +{w}')
            elif _opposed_ind(su_i.get('bias', ''), direction):
                ind_score -= 0.5
                ind_lines_local.append(f'\u2717 Supertrend opposes {direction}  -0.5')

            km_i = inds.get('kama', {})
            if _aligned(km_i.get('bias', ''), direction) and km_i.get('trending'):
                ind_score += 0.5
                ind_lines_local.append('\u2713 KAMA adaptive trend aligned  +0.5')

            kz_i = inds.get('killzones', {})
            if kz_i.get('in_killzone'):
                w = 1.0 if kz_i.get('high_quality') else 0.5
                zones = ', '.join(kz_i.get('active_zones', []))
                ind_score += w
                ind_lines_local.append(f'\u2713 ICT Kill Zone: {zones}  +{w}')

            wy_i = inds.get('wyckoff', {})
            if _aligned(wy_i.get('bias', ''), direction):
                ind_score += 0.8
                ind_lines_local.append(f'\u2713 Wyckoff {wy_i.get("phase","")}  +0.8')

            rr_i = inds.get('real_rate', {})
            if rr_i.get('available') and _aligned(rr_i.get('bias', ''), direction):
                w = 0.8 if 'strongly' in rr_i.get('bias', '') else 0.4
                ind_score += w
                ind_lines_local.append(f'\u2713 Real rate {rr_i.get("real_rate",0):.1f}%  +{w}')

            mc_i = inds.get('market_cipher', {})
            if mc_i.get('bullish_cross') and mc_i.get('oversold') and direction == 'long':
                ind_score += 1.0
                ind_lines_local.append('\u2713 Market Cipher oversold bullish cross  +1.0')
            elif mc_i.get('bearish_cross') and mc_i.get('overbought') and direction == 'short':
                ind_score += 1.0
                ind_lines_local.append('\u2713 Market Cipher overbought bearish cross  +1.0')
            elif _aligned(mc_i.get('bias', ''), direction):
                ind_score += 0.5
                ind_lines_local.append('~ Market Cipher aligned  +0.5')

            ob_i = inds.get('obv', {})
            if ob_i.get('divergence') == 'bullish_divergence' and direction == 'long':
                ind_score += 0.8
                ind_lines_local.append('\u2713 OBV bullish divergence  +0.8')
            elif ob_i.get('divergence') == 'bearish_divergence' and direction == 'short':
                ind_score += 0.8
                ind_lines_local.append('\u2713 OBV bearish divergence  +0.8')
            elif _aligned(ob_i.get('bias', ''), direction):
                ind_score += 0.4
                ind_lines_local.append('~ OBV aligned  +0.4')

            # Normalise: indicators contribute max +/-3.0
            ind_score_norm = min(max(ind_score, -1.5), 3.0)
            weighted_raw  += ind_score_norm
            for line in ind_lines_local:
                detail_lines.append(line)
            check_weights_earned['Indicators'] = round(ind_score_norm, 2)

        except Exception as _ind_e:
            detail_lines.append(f'~ Indicators unavailable: {str(_ind_e)[:50]}')
            _ind_data = {}

    # -- ML Confidence Adjustment (learns from paper trades) -------------------
    if _ML_OK:
        try:
            from datetime import datetime, timezone, timedelta
            _gst_zone  = timezone(timedelta(hours=4))
            _hour_uae  = datetime.now(_gst_zone).hour

            _sess_raw  = raw_checks.get("session") or {}
            _sess_name = (
                _sess_raw.get("session") if isinstance(_sess_raw, dict)
                else str(_sess_raw)
            )
            _reg_raw   = raw_checks.get("regime") or {}
            _reg_name  = (
                _reg_raw.get("regime") if isinstance(_reg_raw, dict)
                else str(_reg_raw)
            )

            ml_adj = _get_ml_adj(
                session    = str(_sess_name or "Unknown"),
                regime     = str(_reg_name  or "Unknown"),
                strategy   = str(symbol),
                confidence = float(weighted_raw),
                hour_uae   = int(_hour_uae),
            )

            if ml_adj.get("available") and ml_adj.get("adjustment", 0.0) != 0.0:
                weighted_raw += ml_adj["adjustment"]
                adj_sign = "+" if ml_adj["adjustment"] > 0 else ""
                detail_lines.append(
                    f"🤖 ML adj ({ml_adj.get('model_size',0)} trades): "
                    f"{adj_sign}{ml_adj['adjustment']:.1f}"
                )
                for reason in ml_adj.get("reasons", [])[:2]:
                    detail_lines.append(f"   → {reason}")
                raw_checks["ml_adjustment"] = ml_adj
        except Exception:
            pass

    # ── FINAL SCORE ──────────────────────────────────────────────────────────
    confidence  = min(10.0, round(weighted_raw, 1))
    trade_valid = weighted_raw >= 5.0

    passed_count = sum(1 for c in met    if c["result"] == "pass")
    failed_count = sum(1 for c in failed if c["result"] in ("fail", "miss"))
    total_checks = passed_count + failed_count + sum(1 for c in met if c["result"] == "neutral")
    conf_score   = passed_count - failed_count

    return {
        # ── new keys ──
        "confidence":           confidence,
        "weighted_score":       round(weighted_raw, 2),
        "detail_lines":         detail_lines,
        "trade_valid":          trade_valid,
        # ── backward-compat keys ──
        "confluences_met":      met,
        "confluences_failed":   failed,
        "confluence_score":     conf_score,
        "passed_count":         passed_count,
        "total_checks":         total_checks,
        "weighted_confidence":  confidence,
        "check_weights_earned": check_weights_earned,
        "raw_checks": {
            "trend":          trend_result,
            "momentum":       momentum,
            "structure":      structure,
            "candle":         candle,
            "session":        session,
            "volatility":     vol,
            "smc":            _smc_context_store or smc_result,
            "htf":            _htf_raw,
            "dxy":            dxy_result,
            "volume":         _volume_result,
            "rsi_divergence": _div_result,
            "sr_levels":      _sr_result,
            "cot":            _cot_result,
            "indicators":     _ind_data,
        },
        "volume": _volume_result,
        "smc_result": smc_result,
        "htf_result": _htf_raw,
        "dxy_result": dxy_result,
    }


def _add_check(
    met: list,
    failed: list,
    agrees: bool,
    *,
    name: str,
    agrees_label: str,
    fails_label: str,
) -> None:
    if agrees:
        met.append({"check": name, "result": "pass", "detail": agrees_label})
    else:
        failed.append({"check": name, "result": "fail", "detail": fails_label})


# ══════════════════════════════════════════════════════════════════════════════
#  validate_signal
# ══════════════════════════════════════════════════════════════════════════════

def validate_signal(
    signal: dict[str, Any],
    df: pd.DataFrame,
    current_time: datetime | None = None,
) -> dict[str, Any]:
    """
    Enhance a signal from the rules database with a full confluence assessment.

    Parameters
    ----------
    signal : dict — one item from morning_briefing.py signal list
    df     : OHLCV + indicator DataFrame
    current_time : optional datetime for session check

    Returns
    -------
    Enhanced signal dict containing all original keys plus:
        original_confidence  : float   (backtest-derived, 0–10)
        confluence_score     : int
        passed_count         : int
        total_checks         : int
        final_confidence     : float   (mean of original and confluence confidence)
        trade_valid          : bool
        reason_valid         : str or None
        reason_invalid       : str or None
        confluences_met      : list
        confluences_failed   : list
        raw_checks           : dict
    """
    direction  = str(signal.get("direction", "long")).lower()
    orig_conf  = float(signal.get("confidence", 5))

    result = score_confluences(df, direction, current_time)

    final_conf = round((orig_conf + result["confidence"]) / 2, 1)

    met_names  = [c["check"] for c in result["confluences_met"]    if c["result"] == "pass"]
    fail_names = [c["check"] for c in result["confluences_failed"] if c["result"] in ("fail", "miss")]

    if result["trade_valid"]:
        reason_valid   = f"Minimum {MIN_CONFLUENCES} confluences met: {', '.join(met_names)}"
        reason_invalid = None
    else:
        reason_valid   = None
        reason_invalid = (
            f"Only {result['passed_count']}/{MIN_CONFLUENCES} required confluences met. "
            f"Failed: {', '.join(fail_names)}"
        )

    enhanced = dict(signal)
    enhanced.update({
        "original_confidence":   orig_conf,
        "confluence_score":      result["confluence_score"],
        "passed_count":          result["passed_count"],
        "total_checks":          result["total_checks"],
        "confluence_confidence": result["confidence"],
        "final_confidence":      final_conf,
        "trade_valid":           result["trade_valid"],
        "reason_valid":          reason_valid,
        "reason_invalid":        reason_invalid,
        "confluences_met":       result["confluences_met"],
        "confluences_failed":    result["confluences_failed"],
        "raw_checks":            result["raw_checks"],
        "check_weights_earned":  result.get("check_weights_earned", {}),
    })
    return enhanced


# ══════════════════════════════════════════════════════════════════════════════
#  print_confluence_report
# ══════════════════════════════════════════════════════════════════════════════

def print_confluence_report(signal: dict[str, Any]) -> None:
    """
    Pretty-print a weighted confluence report for a validated signal.

    Expects the dict returned by validate_signal().
    """
    asset     = signal.get("asset",       "UNKNOWN").upper()
    direction = signal.get("direction",   "long").upper()
    tier      = signal.get("tier",        "?")
    tier_lbl  = signal.get("tier_label",  "")
    pattern   = signal.get("pattern_name", "—")
    is_xau    = "XAU" in asset or asset in ("GOLD", "XAUUSD")

    # Entry / SL / TP
    price = signal.get("entry", signal.get("entry_price", "—"))
    sl    = signal.get("stop_loss",   "—")
    tp    = signal.get("take_profit", "—")

    # Format distances — dollar amount for Gold, pips for FX
    rr_str = "—"
    try:
        p = float(price) if price not in ("—", None, "Market price") else None
        s = float(sl)    if sl    not in ("—", None, "As per pattern") else None
        t = float(tp)    if tp    not in ("—", None, "As per pattern") else None
        if p and s and t:
            risk   = abs(p - s)
            reward = abs(p - t)
            rr     = reward / risk if risk > 0 else 0
            rr_str = f"1:{rr:.1f}"
            if is_xau:
                sl = f"${s:,.2f}  (+${risk:,.2f})"
                tp = f"${t:,.2f}  (+${reward:,.2f})"
            else:
                pip  = 0.0001
                sl_p = round(risk  / pip)
                tp_p = round(reward / pip)
                sl   = f"{s:.5f}  ({sl_p:.0f} pips)"
                tp   = f"{t:.5f}  ({tp_p:.0f} pips)"
        elif p:
            price = f"${p:,.2f}"
    except (TypeError, ValueError):
        pass

    conf_score = signal.get("passed_count",          0)
    total      = signal.get("total_checks",          9)
    trade_ok   = signal.get("trade_valid",           False)
    met        = signal.get("confluences_met",    [])
    failed     = signal.get("confluences_failed", [])
    w_earned   = signal.get("check_weights_earned", {})

    # Recompute running total from w_earned in display order
    CHECK = "\u2713"   # ✓
    CROSS = "\u2717"   # ✗
    W     = 57

    print(f"\n  {'═' * W}")
    print(f"  SIGNAL: {asset} {direction}  [{tier}] {tier_lbl}")
    print(f"  {'═' * W}")
    print()
    print(f"  {'CHECK':<14}  {'DETAIL':<33}  {'WT':>4}")
    print(f"  {'-' * W}")

    check_order  = ["HTF", "SMC", "Trend", "Structure", "Momentum",
                    "DXY", "Candle", "Session", "Volatility"]
    all_checks   = {c["check"]: c for c in met + failed}
    running_wt   = 0.0

    for name in check_order:
        if name not in all_checks:
            continue
        c      = all_checks[name]
        earned = w_earned.get(name, 0.0)
        running_wt += earned

        if c["result"] == "pass":
            mark   = f"  {CHECK}"
            wt_str = f"+{earned:.1f}"
        elif c["result"] == "neutral":
            mark   = "  ~"
            wt_str = " 0.0"
        else:
            mark   = f"  {CROSS}"
            wt_str = " 0.0"

        detail = c["detail"][:33]
        print(f"  {mark} {name:<12}  {detail:<33}  {wt_str:>4}")

    weighted_conf = min(running_wt, 10.0)
    print(f"  {'-' * W}")
    print(f"  {'WEIGHTED CONFIDENCE':50s}  {weighted_conf:.1f}/10")
    print()

    if trade_ok:
        print(f"  TRADE VALID: YES  {CHECK}  ({conf_score}/{total} checks passed)")
        print(f"  {signal.get('reason_valid', '')}")
    else:
        print(f"  TRADE VALID: NO   {CROSS}  ({conf_score}/{total} checks passed)")
        print(f"  {signal.get('reason_invalid', '')}")

    print()
    print(f"  ENTRY       : {price}")
    print(f"  STOP LOSS   : {sl}")
    print(f"  TAKE PROFIT : {tp}")
    print(f"  RISK/REWARD : {rr_str}")
    print(f"  PATTERN     : {pattern}")
    print(f"  {'═' * W}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Quick self-test  (python confluence_engine.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, json

    RULES_FILE = os.path.join("data", "rules.json")
    HIST_CSV   = os.path.join("data", "historical_xauusd.csv")

    print("\n  Loading historical data & rules for self-test...\n")

    # ── Load & enrich price data ───────────────────────────────────────────
    try:
        raw = pd.read_csv(HIST_CSV, index_col=0)
        raw.columns = [c.lower() for c in raw.columns]

        # The CSV only has high/low/close/volume — synthesise open from prev close
        if "open" not in raw.columns:
            raw["open"] = raw["close"].shift(1).fillna(raw["close"])

        if "ema50" not in raw.columns:
            raw["ema50"]  = raw["close"].ewm(span=50,  adjust=False).mean()
        if "ema200" not in raw.columns:
            raw["ema200"] = raw["close"].ewm(span=200, adjust=False).mean()
        if "rsi" not in raw.columns:
            delta        = raw["close"].diff()
            gain         = delta.clip(lower=0).rolling(14).mean()
            loss         = (-delta.clip(upper=0)).rolling(14).mean()
            rs           = gain / loss.replace(0, np.nan)
            raw["rsi"]   = 100 - (100 / (1 + rs))
        if "atr" not in raw.columns:
            h_l          = raw["high"] - raw["low"]
            h_pc         = (raw["high"] - raw["close"].shift()).abs()
            l_pc         = (raw["low"]  - raw["close"].shift()).abs()
            raw["atr"]   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1).rolling(14).mean()
        if "macd" not in raw.columns:
            ema12              = raw["close"].ewm(span=12, adjust=False).mean()
            ema26              = raw["close"].ewm(span=26, adjust=False).mean()
            raw["macd"]        = ema12 - ema26
            raw["macd_signal"] = raw["macd"].ewm(span=9, adjust=False).mean()

        df = raw.dropna(subset=["close", "ema200", "rsi"]).copy()
        print(f"  Loaded {len(df)} candles  "
              f"({len(raw)} raw → {len(df)} after indicator warmup)\n")

    except FileNotFoundError:
        print("  Historical data not found. Run setup.py --refresh first.\n")
        raise SystemExit(1)

    # ── Pick best available rule ───────────────────────────────────────────
    try:
        with open(RULES_FILE) as f:
            rules = json.load(f)

        rule = None
        for tier_pref in ("A", "B", "C"):
            candidates = [r for r in rules if r.get("tier") == tier_pref]
            if candidates:
                rule = sorted(
                    candidates,
                    key=lambda r: r.get("backtest", {}).get("win_rate", 0),
                    reverse=True,
                )[0]
                break
        if rule is None:
            rule = rules[0]

        direction = str(rule.get("direction", "short")).lower()
        if direction not in ("long", "short", "buy", "sell"):
            direction = "short"

        signal = {
            "asset":         "XAUUSD",
            "direction":     direction,
            "confidence":    float(rule.get("confidence_score") or
                                   rule.get("backtest", {}).get("backtest_confidence", 5)),
            "tier":          rule.get("tier", "D"),
            "tier_label":    {
                "A": "STRONG SIGNAL",
                "B": "MODERATE SIGNAL",
                "C": "WEAK SIGNAL - paper trade only",
            }.get(rule.get("tier", "D"), "UNVERIFIED"),
            "pattern_name":  rule.get("name") or rule.get("pattern_name", "Unknown"),
            "entry":         str(round(df.iloc[-1]["close"], 2)),
            "stop_loss":     "—",
            "take_profit":   "—",
            "bt_win_rate":   rule.get("backtest", {}).get("win_rate", 0),
            "profit_factor": rule.get("backtest", {}).get("profit_factor", 0),
        }
        print(f"  Testing rule : [{rule.get('tier','?')}] "
              f"{signal['pattern_name']}  ({signal['direction']})\n")

    except (FileNotFoundError, IndexError, KeyError) as exc:
        print(f"  Could not load rules ({exc}). Using synthetic demo signal.\n")
        signal = {
            "asset":         "XAUUSD",
            "direction":     "short",
            "confidence":    7,
            "tier":          "B",
            "tier_label":    "MODERATE SIGNAL",
            "pattern_name":  "Demo Signal",
            "entry":         str(round(df.iloc[-1]["close"], 2)),
            "stop_loss":     "—",
            "take_profit":   "—",
            "bt_win_rate":   47.0,
            "profit_factor": 1.28,
        }

    validated = validate_signal(signal, df)
    print_confluence_report(validated)
