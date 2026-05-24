"""
mtf_analyzer.py
───────────────
Multi-Timeframe (MTF) Analysis for TradingBotV1.

Professional traders never trade H1 without knowing D1 and H4 bias first.
This module implements that top-down approach:

  D1  → Long-term directional bias (EMA200, key S/R levels)
  H4  → Intermediate structure (BOS / CHoCH from smart_money.py)
  H1  → Entry-level validation (main trading timeframe)

Classes:
    MultiTimeframeAnalyzer

Usage:
    from mtf_analyzer import MultiTimeframeAnalyzer
    mta    = MultiTimeframeAnalyzer()
    htf    = mta.get_htf_bias("GC=F")
    result = mta.validate_h1_with_htf(h1_signal, htf)
    levels = mta.get_key_htf_levels("GC=F")
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd

# ── SMC integration ────────────────────────────────────────────────────────────
try:
    from smart_money import SmartMoneyAnalyzer
    _SMC_OK = True
except ImportError:
    _SMC_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────
HTF_LEVEL_PROXIMITY  = 0.003   # 0.3% — "at" an HTF level
EQ_LEVEL_TOLERANCE   = 0.001   # 0.1% — equal high/low cluster
CACHE_TTL_SECONDS    = 3600    # 1 hour cache for HTF data per symbol

# yfinance interval → period mapping
_YF_PARAMS: dict[str, dict] = {
    "D1": {"interval": "1d",  "period": "200d"},
    "H4": {"interval": "4h",  "period": "60d"},
    "H1": {"interval": "1h",  "period": "10d"},
    "W1": {"interval": "1wk", "period": "2y"},
}

# ── In-memory cache ────────────────────────────────────────────────────────────
_CACHE: dict[str, dict] = {}   # key = "symbol_timeframe"


# ══════════════════════════════════════════════════════════════════════════════
#  Data fetcher
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_df(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for symbol/timeframe via yfinance.
    Uses in-memory cache (1 h TTL) to avoid repeated API calls.
    Returns enriched DataFrame (open/high/low/close/ema50/ema200/atr/rsi/macd)
    or None on failure.
    """
    cache_key = f"{symbol}_{timeframe}"
    cached    = _CACHE.get(cache_key)
    if cached and (time.time() - cached["ts"]) < CACHE_TTL_SECONDS:
        return cached["df"]

    params = _YF_PARAMS.get(timeframe)
    if params is None:
        return None

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df     = ticker.history(**params, auto_adjust=True)
        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        df = df.copy()

        if "open" not in df.columns:
            df["open"] = df["close"].shift(1).fillna(df["close"])

        # Indicators
        close = df["close"]
        df["ema50"]  = close.ewm(span=50,  adjust=False).mean()
        df["ema200"] = close.ewm(span=200, adjust=False).mean()

        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))

        prev_c = close.shift(1)
        tr     = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"]  - prev_c).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

        e12 = close.ewm(span=12, adjust=False).mean()
        e26 = close.ewm(span=26, adjust=False).mean()
        df["macd"]        = e12 - e26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        df = df.dropna(subset=["ema200", "rsi", "atr"]).copy()

        _CACHE[cache_key] = {"df": df, "ts": time.time()}
        return df

    except Exception:
        return None


def _enrich_existing(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators to an existing DataFrame that lacks them."""
    df = df.copy()
    if "open" not in df.columns:
        df["open"] = df["close"].shift(1).fillna(df["close"])

    close = df["close"]

    if "ema200" not in df.columns:
        df["ema50"]  = close.ewm(span=50,  adjust=False).mean()
        df["ema200"] = close.ewm(span=200, adjust=False).mean()

    if "rsi" not in df.columns:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))

    if "atr" not in df.columns:
        prev_c = close.shift(1)
        tr     = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"]  - prev_c).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    if "macd" not in df.columns:
        e12 = close.ewm(span=12, adjust=False).mean()
        e26 = close.ewm(span=26, adjust=False).mean()
        df["macd"]        = e12 - e26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    return df.dropna(subset=["ema200"])


# ══════════════════════════════════════════════════════════════════════════════
#  Swing level detector (lightweight, no SMC dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _swing_levels(df: pd.DataFrame, window: int = 5) -> tuple[list[float], list[float]]:
    """
    Return (swing_highs, swing_lows) — simple pivot detection.
    Each swing high/low is a price where the surrounding `window`
    bars are all lower/higher respectively.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(highs)
    sh, sl = [], []
    for i in range(window, n - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            sh.append(round(float(highs[i]), 2))
        if lows[i] == min(lows[i - window: i + window + 1]):
            sl.append(round(float(lows[i]), 2))
    return sh, sl


def _trend_from_ema(df: pd.DataFrame) -> str:
    """Return 'bullish' | 'bearish' | 'ranging' based on EMA200 and EMA50."""
    row    = df.iloc[-1]
    price  = float(row["close"])
    ema200 = float(row.get("ema200", float("nan")))
    ema50  = float(row.get("ema50",  float("nan")))

    if any(np.isnan(v) for v in [price, ema200, ema50]):
        return "ranging"

    gap_pct = abs(price - ema200) / ema200
    if gap_pct < 0.005:              # within 0.5% of EMA200 = ranging
        return "ranging"
    if price > ema200 and ema50 > ema200:
        return "bullish"
    if price < ema200 and ema50 < ema200:
        return "bearish"
    return "ranging"                  # price and EMA50 disagree


# ══════════════════════════════════════════════════════════════════════════════
#  MultiTimeframeAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class MultiTimeframeAnalyzer:
    """
    Top-down Multi-Timeframe analysis.

    All three public methods accept an optional pre-fetched DataFrame so that
    callers who already have live H1 data (morning_briefing.py) can skip
    the extra yfinance fetch.
    """

    # ══════════════════════════════════════════════════════════════════════════
    #  1. get_htf_bias
    # ══════════════════════════════════════════════════════════════════════════

    def get_htf_bias(
        self,
        symbol: str = "GC=F",
        d1_df:  pd.DataFrame | None = None,
        h4_df:  pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Compute Daily and 4-Hour bias for a symbol.

        Parameters
        ----------
        symbol : yfinance ticker (default "GC=F" = Gold)
        d1_df  : optional pre-fetched D1 DataFrame (skips yfinance call)
        h4_df  : optional pre-fetched H4 DataFrame (skips yfinance call)

        Returns
        -------
        d1_trend       : "bullish" | "bearish" | "ranging"
        d1_ema200      : float
        d1_price       : float
        d1_key_levels  : list[float]  — major S/R levels on D1
        h4_trend       : "bullish" | "bearish" | "ranging"
        h4_structure   : "trending" | "ranging" | "reversal"
        h4_bos         : float | None  — last Break of Structure price
        h4_choch       : float | None  — last Change of Character price
        overall_bias   : "bullish" | "bearish" | "neutral"
        bias_strength  : "strong" | "weak" | "conflicted"
        bias_detail    : str — plain English summary
        d1_available   : bool
        h4_available   : bool
        """
        # ── D1 analysis ───────────────────────────────────────────────────────
        d1_result = self._analyze_d1(symbol, d1_df)
        h4_result = self._analyze_h4(symbol, h4_df)

        # ── Combine biases ────────────────────────────────────────────────────
        d1_bias = d1_result["trend"]
        h4_bias = h4_result["trend"]

        d1_ok = d1_result["available"]
        h4_ok = h4_result["available"]

        if not d1_ok and not h4_ok:
            overall_bias  = "neutral"
            bias_strength = "conflicted"
            bias_detail   = "No HTF data available — using H1 only"

        elif not d1_ok:
            overall_bias  = h4_bias if h4_bias != "ranging" else "neutral"
            bias_strength = "weak"
            bias_detail   = f"D1 unavailable; H4 shows {h4_bias.upper()}"

        elif not h4_ok:
            overall_bias  = d1_bias if d1_bias != "ranging" else "neutral"
            bias_strength = "weak"
            bias_detail   = f"H4 unavailable; D1 shows {d1_bias.upper()}"

        else:
            # Both available — determine agreement
            if d1_bias == h4_bias and d1_bias != "ranging":
                overall_bias  = d1_bias
                bias_strength = "strong"
                bias_detail   = (
                    f"D1 {d1_bias.upper()} + H4 {h4_bias.upper()} aligned → "
                    f"STRONGLY {d1_bias.upper()}"
                )
            elif d1_bias == "ranging" and h4_bias != "ranging":
                overall_bias  = h4_bias
                bias_strength = "weak"
                bias_detail   = (
                    f"D1 ranging; H4 {h4_bias.upper()} — weak directional bias"
                )
            elif h4_bias == "ranging" and d1_bias != "ranging":
                overall_bias  = d1_bias
                bias_strength = "weak"
                bias_detail   = (
                    f"D1 {d1_bias.upper()}; H4 ranging — monitor for H4 confirmation"
                )
            elif d1_bias != h4_bias:
                # Conflicting — D1 wins but mark as conflicted
                overall_bias  = d1_bias if d1_bias != "ranging" else h4_bias
                bias_strength = "conflicted"
                bias_detail   = (
                    f"⚠️ CONFLICT: D1 {d1_bias.upper()} vs H4 {h4_bias.upper()} — "
                    f"reduce position size, wait for alignment"
                )
            else:
                overall_bias  = "neutral"
                bias_strength = "weak"
                bias_detail   = "Both timeframes ranging — avoid directional bias"

            # Upgrade to "strong" if CHoCH confirmed on H4 in same direction
            if h4_result.get("h4_choch") and bias_strength == "weak":
                if (overall_bias == "bullish" and h4_bias == "bullish") or \
                   (overall_bias == "bearish" and h4_bias == "bearish"):
                    bias_strength = "strong"
                    bias_detail  += " | H4 CHoCH confirms direction"

        return {
            # D1
            "d1_trend":      d1_bias,
            "d1_ema200":     d1_result["ema200"],
            "d1_price":      d1_result["price"],
            "d1_key_levels": d1_result["key_levels"],
            "d1_available":  d1_ok,
            # H4
            "h4_trend":      h4_bias,
            "h4_structure":  h4_result["structure"],
            "h4_bos":        h4_result.get("h4_bos"),
            "h4_choch":      h4_result.get("h4_choch"),
            "h4_available":  h4_ok,
            # Combined
            "overall_bias":  overall_bias,
            "bias_strength": bias_strength,
            "bias_detail":   bias_detail,
        }

    def _analyze_d1(self, symbol: str, df: pd.DataFrame | None) -> dict:
        """D1: EMA200 trend + key S/R levels."""
        if df is None:
            df = _fetch_df(symbol, "D1")
        if df is None or df.empty:
            return {
                "available": False, "trend": "ranging",
                "ema200": None, "price": None, "key_levels": [],
            }

        df    = _enrich_existing(df)
        trend = _trend_from_ema(df)
        row   = df.iloc[-1]
        price = round(float(row["close"]), 2)
        e200  = round(float(row.get("ema200", 0)), 2)

        # Key levels: swing highs/lows over last 100 D1 bars
        sh, sl   = _swing_levels(df.tail(100), window=5)
        # Keep levels within ±15% of current price
        all_lvls = sorted(
            {lvl for lvl in sh + sl
             if abs(lvl - price) / max(price, 1) < 0.15},
            reverse=True,
        )
        # Add EMA200 as a key level
        if e200 and e200 not in all_lvls:
            all_lvls.append(e200)
        all_lvls.sort(reverse=True)

        return {
            "available":  True,
            "trend":      trend,
            "ema200":     e200,
            "price":      price,
            "key_levels": all_lvls[:10],
        }

    def _analyze_h4(self, symbol: str, df: pd.DataFrame | None) -> dict:
        """H4: EMA trend + SMC market structure (BOS / CHoCH)."""
        if df is None:
            df = _fetch_df(symbol, "H4")
        if df is None or df.empty:
            return {
                "available": False, "trend": "ranging",
                "structure": "ranging", "h4_bos": None, "h4_choch": None,
            }

        df    = _enrich_existing(df)
        trend = _trend_from_ema(df)

        # SMC market structure
        h4_bos   = None
        h4_choch = None
        structure = "trending" if trend != "ranging" else "ranging"

        if _SMC_OK:
            try:
                sma  = SmartMoneyAnalyzer()
                ms   = sma.detect_market_structure(df)
                bias = ms.get("bias", "neutral")

                h4_bos   = ms.get("last_bos")
                h4_choch = ms.get("last_choch")

                if h4_choch:
                    structure = "reversal"
                elif ms.get("structure") == "trending_up" or ms.get("structure") == "trending_down":
                    structure = "trending"
                else:
                    structure = "ranging"

                # SMC bias can refine trend
                if bias in ("bullish", "bearish"):
                    trend = bias
            except Exception:
                pass

        return {
            "available": True,
            "trend":     trend,
            "structure": structure,
            "h4_bos":    h4_bos,
            "h4_choch":  h4_choch,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  2. validate_h1_with_htf
    # ══════════════════════════════════════════════════════════════════════════

    def validate_h1_with_htf(
        self,
        h1_signal: dict[str, Any],
        htf_bias:  dict[str, Any],
    ) -> dict[str, Any]:
        """
        Check whether an H1 signal aligns with D1 and H4 bias.

        Parameters
        ----------
        h1_signal : signal dict — must contain 'direction' ("long" | "short")
        htf_bias  : result from get_htf_bias()

        Returns
        -------
        aligned             : bool
        alignment_score     : int  0 | 1 | 2
                              2 = D1 + H4 both agree
                              1 = only one agrees
                              0 = neither agrees
        confidence_adjustment : int  +2 aligned, -2 against, 0 neutral
        note                : str — explanation
        d1_agrees           : bool
        h4_agrees           : bool
        """
        direction = str(h1_signal.get("direction", "")).lower().strip()
        is_long   = direction in ("long", "buy")
        is_short  = direction in ("short", "sell")

        d1_bias     = htf_bias.get("d1_trend",    "ranging")
        h4_bias     = htf_bias.get("h4_trend",    "ranging")
        overall     = htf_bias.get("overall_bias", "neutral")
        strength    = htf_bias.get("bias_strength","weak")

        def _agrees(bias: str) -> bool:
            if is_long  and bias == "bullish": return True
            if is_short and bias == "bearish": return True
            return False

        def _opposes(bias: str) -> bool:
            if is_long  and bias == "bearish": return True
            if is_short and bias == "bullish": return True
            return False

        d1_agrees  = _agrees(d1_bias)
        h4_agrees  = _agrees(h4_bias)
        d1_opposes = _opposes(d1_bias)
        h4_opposes = _opposes(h4_bias)

        alignment_score = int(d1_agrees) + int(h4_agrees)
        aligned         = alignment_score >= 1

        # Confidence adjustment
        if alignment_score == 2 and strength == "strong":
            conf_adj = +2
        elif alignment_score == 2:
            conf_adj = +1
        elif alignment_score == 1:
            conf_adj =  0
        elif d1_opposes and h4_opposes:
            conf_adj = -2
        elif d1_opposes or h4_opposes:
            conf_adj = -1
        else:
            conf_adj = 0   # both ranging — neutral

        # Note
        dir_label = "LONG" if is_long else "SHORT"
        parts = []
        if d1_agrees:
            parts.append(f"D1 {d1_bias.upper()} ✓")
        elif d1_opposes:
            parts.append(f"D1 {d1_bias.upper()} ✗ (opposes {dir_label})")
        else:
            parts.append(f"D1 ranging (neutral)")

        if h4_agrees:
            parts.append(f"H4 {h4_bias.upper()} ✓")
        elif h4_opposes:
            parts.append(f"H4 {h4_bias.upper()} ✗ (opposes {dir_label})")
        else:
            parts.append(f"H4 ranging (neutral)")

        if alignment_score == 2:
            verdict = f"ALIGNED ({strength.upper()} bias)"
        elif alignment_score == 1:
            verdict = "PARTIAL alignment"
        else:
            verdict = "NOT aligned — trade against HTF bias"

        note = f"{verdict} | " + " | ".join(parts)

        return {
            "aligned":                aligned,
            "alignment_score":        alignment_score,
            "confidence_adjustment":  conf_adj,
            "note":                   note,
            "d1_agrees":              d1_agrees,
            "h4_agrees":              h4_agrees,
            "d1_bias":                d1_bias,
            "h4_bias":                h4_bias,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  3. get_key_htf_levels
    # ══════════════════════════════════════════════════════════════════════════

    def get_key_htf_levels(
        self,
        symbol:   str = "GC=F",
        w1_df:    pd.DataFrame | None = None,
        d1_df:    pd.DataFrame | None = None,
        h4_df:    pd.DataFrame | None = None,
        h1_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Gather the key HTF price levels where institutional traders react.

        Fetches:
          W1 → weekly high/low
          D1 → daily high/low
          H4 → H4 swing high/swing low

        Returns
        -------
        weekly_high, weekly_low    : float | None
        daily_high, daily_low      : float | None
        h4_swing_high, h4_swing_low: float | None
        all_levels                 : sorted list of all key levels
        nearest_level              : closest level to h1_price
        nearest_distance_pct       : float — % distance to nearest level
        at_htf_level               : bool — True if within HTF_LEVEL_PROXIMITY
        """
        results: dict[str, Any] = {
            "weekly_high": None, "weekly_low": None,
            "daily_high":  None, "daily_low":  None,
            "h4_swing_high": None, "h4_swing_low": None,
            "all_levels": [],
            "nearest_level": None,
            "nearest_distance_pct": None,
            "at_htf_level": False,
        }

        # ── W1 ────────────────────────────────────────────────────────────────
        if w1_df is None:
            w1_df = _fetch_df(symbol, "W1")
        if w1_df is not None and not w1_df.empty:
            recent_w = w1_df.tail(4)   # last 4 weeks
            results["weekly_high"] = round(float(recent_w["high"].max()), 2)
            results["weekly_low"]  = round(float(recent_w["low"].min()),  2)

        # ── D1 ────────────────────────────────────────────────────────────────
        if d1_df is None:
            d1_df = _fetch_df(symbol, "D1")
        if d1_df is not None and not d1_df.empty:
            recent_d = d1_df.tail(5)   # last 5 trading days
            results["daily_high"] = round(float(recent_d["high"].max()), 2)
            results["daily_low"]  = round(float(recent_d["low"].min()),  2)

        # ── H4 ────────────────────────────────────────────────────────────────
        if h4_df is None:
            h4_df = _fetch_df(symbol, "H4")
        if h4_df is not None and not h4_df.empty:
            sh, sl = _swing_levels(h4_df.tail(40), window=3)
            if sh:
                results["h4_swing_high"] = sh[-1]   # most recent
            if sl:
                results["h4_swing_low"]  = sl[-1]

        # ── Compile all levels ────────────────────────────────────────────────
        all_lvls = sorted(
            {v for v in results.values()
             if isinstance(v, float) and v > 0},
        )
        results["all_levels"] = all_lvls

        # ── Nearest level to H1 price ─────────────────────────────────────────
        if h1_price and all_lvls:
            nearest = min(all_lvls, key=lambda lvl: abs(lvl - h1_price))
            dist    = abs(nearest - h1_price) / max(h1_price, 1e-9)
            results["nearest_level"]          = nearest
            results["nearest_distance_pct"]   = round(dist * 100, 3)
            results["at_htf_level"]            = dist <= HTF_LEVEL_PROXIMITY

        return results


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone HTF step for morning_briefing.py
# ══════════════════════════════════════════════════════════════════════════════

def get_htf_context(
    symbol:   str = "GC=F",
    h1_price: float | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper: run full MTF analysis and return a flat context dict
    suitable for passing into morning_briefing signal scan and print functions.

    Returns
    -------
    htf_bias      : full get_htf_bias() result
    htf_levels    : full get_key_htf_levels() result
    bias_line     : short formatted string for printing
    available     : True if at least one HTF dataset loaded
    """
    mta   = MultiTimeframeAnalyzer()
    bias  = mta.get_htf_bias(symbol)
    levels = mta.get_key_htf_levels(symbol, h1_price=h1_price)

    strength_map = {
        "strong":     "STRONGLY",
        "weak":       "WEAKLY",
        "conflicted": "CONFLICTED",
    }
    strength_word = strength_map.get(bias["bias_strength"], "")
    overall       = bias["overall_bias"].upper()
    d1            = bias["d1_trend"].upper()
    h4            = bias["h4_trend"].upper()

    if bias["bias_strength"] == "conflicted":
        bias_line = (
            f"D1 BIAS: {d1} | H4 BIAS: {h4} | "
            f"⚠️ CONFLICTED — wait for alignment or reduce size"
        )
    else:
        direction_guide = (
            "→ short only" if overall == "BEARISH"
            else "→ long only" if overall == "BULLISH"
            else "→ range trade"
        )
        bias_line = (
            f"D1 BIAS: {d1} | H4 BIAS: {h4} | "
            f"OVERALL: {strength_word} {overall} {direction_guide}"
        )

    return {
        "htf_bias":   bias,
        "htf_levels": levels,
        "bias_line":  bias_line,
        "available":  bias.get("d1_available") or bias.get("h4_available"),
    }


def print_htf_report(htf_ctx: dict) -> None:
    """Print a formatted MTF bias block to stdout."""
    bias   = htf_ctx.get("htf_bias",   {})
    levels = htf_ctx.get("htf_levels", {})
    W      = 62

    d1_trend  = str(bias.get("d1_trend",     "N/A")).upper()
    h4_trend  = str(bias.get("h4_trend",     "N/A")).upper()
    overall   = str(bias.get("overall_bias",  "N/A")).upper()
    strength  = str(bias.get("bias_strength", "N/A")).upper()
    detail    = str(bias.get("bias_detail",   ""))
    h4_struct = str(bias.get("h4_structure",  "N/A")).upper()
    h4_bos    = bias.get("h4_bos")
    h4_choch  = bias.get("h4_choch")

    trend_icon = {
        "BULLISH": "📈", "BEARISH": "📉", "RANGING": "↔️", "N/A": "—",
    }

    print("  ╔" + "═" * W + "╗")
    print(f"  ║{'  MULTI-TIMEFRAME BIAS  ':^{W}}║")
    print("  ╠" + "═" * W + "╣")
    print(f"  ║  {'D1 Bias':<16} {trend_icon.get(d1_trend,'—')} {d1_trend:<12}"
          f"  EMA200: ${bias.get('d1_ema200') or '—':<10}{'':{W-53}}║")
    print(f"  ║  {'H4 Bias':<16} {trend_icon.get(h4_trend,'—')} {h4_trend:<12}"
          f"  Structure: {h4_struct:<14}{'':{W-55}}║")

    if h4_bos:
        print(f"  ║  {'H4 BOS':<16} ${h4_bos:>12,.2f}{'':{W-31}}║")
    if h4_choch:
        print(f"  ║  {'H4 CHoCH ⚠️':<16} ${h4_choch:>12,.2f}  (reversal signal){'':{W-49}}║")

    print("  ╠" + "═" * W + "╣")
    print(f"  ║  OVERALL: {overall:<10}  Strength: {strength:<12}{'':{W-37}}║")

    # Detail (wrapped at W-4 chars)
    detail_chunks = [detail[i:i + W - 4] for i in range(0, len(detail), W - 4)]
    for chunk in detail_chunks[:2]:
        print(f"  ║  {chunk:<{W - 2}}║")

    # Key levels
    all_lvls = levels.get("all_levels", [])
    if all_lvls:
        lvl_str = "  ".join(f"${l:,.1f}" for l in all_lvls[:6])
        print("  ╠" + "═" * W + "╣")
        print(f"  ║  Key HTF Levels: {lvl_str:<{W-18}}║")

    at_lvl = levels.get("at_htf_level", False)
    if at_lvl:
        nearest = levels.get("nearest_level")
        dist    = levels.get("nearest_distance_pct", 0)
        print(f"  ║  ⭐ PRICE AT HTF LEVEL ${nearest:,.2f} ({dist:.2f}% away){'':{W-46}}║")

    print("  ╚" + "═" * W + "╝")


# ══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nMultiTimeframe Analyzer — Self Test")
    print("─" * 50)

    mta = MultiTimeframeAnalyzer()

    # Test with synthetic data (no yfinance needed)
    rng = np.random.default_rng(42)
    n   = 300

    def _make_df(n, seed_offset=0):
        c = 3300 + seed_offset + np.cumsum(rng.normal(-0.2, 8, n))  # slight downtrend
        df = pd.DataFrame({
            "open":   c - rng.uniform(0, 8, n),
            "high":   c + rng.uniform(2, 15, n),
            "low":    c - rng.uniform(2, 15, n),
            "close":  c,
            "volume": rng.integers(1000, 9000, n).astype(float),
        })
        return _enrich_existing(df)

    d1 = _make_df(300, seed_offset=0)
    h4 = _make_df(300, seed_offset=-50)   # slightly lower = bearish

    bias = mta.get_htf_bias("GC=F", d1_df=d1, h4_df=h4)
    print(f"D1 trend:      {bias['d1_trend']}")
    print(f"H4 trend:      {bias['h4_trend']}")
    print(f"Overall bias:  {bias['overall_bias']}")
    print(f"Strength:      {bias['bias_strength']}")
    print(f"Detail:        {bias['bias_detail']}")

    # H1 signal validation
    short_sig  = {"direction": "short", "confidence": 6.0}
    long_sig   = {"direction": "long",  "confidence": 6.0}

    for sig in [short_sig, long_sig]:
        result = mta.validate_h1_with_htf(sig, bias)
        print(f"\n  Signal {sig['direction'].upper():5}: aligned={result['aligned']} "
              f"score={result['alignment_score']}/2  adj={result['confidence_adjustment']:+d}")
        print(f"    {result['note']}")

    # Key levels
    h1_price = float(d1["close"].iloc[-1])
    levels   = mta.get_key_htf_levels("GC=F", d1_df=d1, h4_df=h4, h1_price=h1_price)
    print(f"\nKey HTF levels: {levels['all_levels'][:6]}")
    print(f"At HTF level:  {levels['at_htf_level']} (nearest: {levels['nearest_level']})")

    # Full context
    htf_ctx = {
        "htf_bias":   bias,
        "htf_levels": levels,
        "bias_line":  "TEST",
        "available":  True,
    }
    print()
    print_htf_report(htf_ctx)
    print("\nSelf-test PASSED ✓")
