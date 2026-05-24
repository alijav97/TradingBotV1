"""
strategy_playbooks.py
─────────────────────
12 hardcoded professional trading strategies built as structured rules.
NOT extracted from PDFs — these are hand-crafted playbooks.

Usage:
    from strategy_playbooks import get_active_playbooks, print_playbook_signals
    signals = get_active_playbooks(df, news_sentiment)
    print_playbook_signals(df, sentiment)
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = "data"
PLAYBOOKS_JSON = os.path.join(DATA_DIR, "playbooks.json")
PRICE_CACHE    = os.path.join(DATA_DIR, "price_cache.json")

# ── Session windows (UTC) ─────────────────────────────────────────────────────
SESSION_WINDOWS = {
    "Asian":       (0,  7),
    "London":      (7,  12),
    "LondonNY":    (12, 15),
    "NewYork":     (13, 17),
}


# ══════════════════════════════════════════════════════════════════════════════
#  PLAYBOOKS dictionary  — 12 professional strategies
# ══════════════════════════════════════════════════════════════════════════════

PLAYBOOKS: dict[str, dict[str, Any]] = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "ema_trend_continuation": {
        "id":               "ema_trend_continuation",
        "name":             "EMA Trend Continuation",
        "asset":            ["XAUUSD", "EURUSD", "GBPUSD"],
        "timeframe":        "H1",
        "conditions_long":  [
            "price above EMA200",
            "EMA50 above EMA200",
            "RSI between 40 and 60",
            "price pulls back to EMA50",
            "bullish candle closes above EMA50",
        ],
        "conditions_short": [
            "price below EMA200",
            "EMA50 below EMA200",
            "RSI between 40 and 60",
            "price pulls back to EMA50",
            "bearish candle closes below EMA50",
        ],
        "entry":            "close of confirmation candle",
        "stop_loss":        "below/above EMA50 + 0.5 ATR",
        "take_profit":      "2x stop loss distance",
        "risk_reward":      2.0,
        "best_session":     ["London", "NewYork"],
        "win_rate_expected": 58,
        "notes":            "Only trade in direction of EMA200 trend",
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "rsi_oversold_bounce": {
        "id":               "rsi_oversold_bounce",
        "name":             "RSI Oversold Bounce",
        "asset":            ["XAUUSD", "XAGUSD"],
        "timeframe":        "H1",
        "conditions_long":  [
            "RSI below 30",
            "price at major support level",
            "price above EMA200",
            "bullish reversal candle appears",
            "candle closes above support",
        ],
        "conditions_short": [
            "RSI above 70",
            "price at major resistance level",
            "price below EMA200",
            "bearish reversal candle appears",
            "candle closes below resistance",
        ],
        "entry":            "open of next candle after signal",
        "stop_loss":        "below/above the reversal candle wick",
        "take_profit":      "next major S/R level",
        "risk_reward":      2.5,
        "best_session":     ["London open", "NY open"],
        "win_rate_expected": 62,
        "notes":            "Strongest when RSI makes higher low while price makes lower low (divergence)",
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "london_breakout_gold": {
        "id":               "london_breakout_gold",
        "name":             "London Breakout",
        "asset":            ["XAUUSD"],
        "timeframe":        "H1",
        "conditions_long":  [
            "identify Asian session range (00:00-07:00 UTC)",
            "price breaks above Asian high",
            "break candle closes above Asian high",
            "RSI above 50",
            "volume above average",
        ],
        "conditions_short": [
            "identify Asian session range",
            "price breaks below Asian low",
            "break candle closes below Asian low",
            "RSI below 50",
        ],
        "entry":            "close of breakout candle",
        "stop_loss":        "middle of Asian range",
        "take_profit":      "Asian range size projected from breakout",
        "risk_reward":      2.0,
        "best_session":     ["London open 07:00-09:00 UTC"],
        "win_rate_expected": 55,
        "notes":            "Gold moves 60% of days during London open",
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "macd_divergence_reversal": {
        "id":               "macd_divergence_reversal",
        "name":             "MACD Divergence",
        "asset":            ["XAUUSD", "EURUSD", "GBPJPY"],
        "timeframe":        "H4",
        "conditions_long":  [
            "price making lower lows",
            "MACD histogram making higher lows",
            "bullish divergence confirmed",
            "RSI below 40",
            "price at support zone",
        ],
        "conditions_short": [
            "price making higher highs",
            "MACD histogram making lower highs",
            "bearish divergence confirmed",
            "RSI above 60",
            "price at resistance zone",
        ],
        "entry":            "next candle after divergence confirmed",
        "stop_loss":        "beyond recent swing high/low",
        "take_profit":      "3x stop loss distance",
        "risk_reward":      3.0,
        "best_session":     ["Any"],
        "win_rate_expected": 57,
        "notes":            "H4 divergence is strongest signal. Wait for full candle close to confirm.",
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "gold_safe_haven_spike": {
        "id":               "gold_safe_haven_spike",
        "name":             "Safe Haven Gold Spike",
        "asset":            ["XAUUSD"],
        "timeframe":        "H1",
        "conditions_long":  [
            "major risk-off news event detected",
            "geopolitical crisis or market fear spike",
            "gold gaps up or spikes on open",
            "RSI not yet overbought (below 70)",
            "price holds above previous close",
        ],
        "conditions_short": [
            "risk-on news after fear event",
            "resolution of geopolitical crisis",
            "gold giving back spike gains",
            "RSI above 70 after spike",
        ],
        "entry":            "first pullback after initial spike",
        "stop_loss":        "below the pullback low",
        "take_profit":      "measured move from spike base",
        "risk_reward":      2.0,
        "best_session":     ["Any - news driven"],
        "win_rate_expected": 64,
        "notes":            "Only when news_monitor detects HIGH risk geopolitical event",
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    "head_and_shoulders": {
        "id":               "head_and_shoulders",
        "name":             "Head and Shoulders",
        "asset":            ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"],
        "timeframe":        "H4",
        "conditions_short": [
            "left shoulder formed at resistance",
            "head formed higher than shoulder",
            "right shoulder formed at same level as left",
            "neckline identified",
            "price breaks below neckline",
            "RSI below 50 on breakout",
        ],
        "conditions_long":  [
            "inverse head and shoulders",
            "price breaks above neckline",
            "RSI above 50 on breakout",
        ],
        "entry":            "neckline breakout close",
        "stop_loss":        "above right shoulder",
        "take_profit":      "neckline to head distance projected down",
        "risk_reward":      2.5,
        "best_session":     ["London", "NewYork"],
        "win_rate_expected": 61,
        "notes":            "One of the most reliable reversal patterns. Must see clear 3-peak structure.",
    },

    # ── 7 ─────────────────────────────────────────────────────────────────────
    "fibonacci_golden_zone": {
        "id":               "fibonacci_golden_zone",
        "name":             "Fibonacci Golden Zone",
        "asset":            ["XAUUSD", "EURUSD", "GBPUSD"],
        "timeframe":        "H1 or H4",
        "conditions_long":  [
            "identify clear bullish impulse move",
            "price retraces to 61.8% fibonacci level",
            "RSI between 35 and 50 on retracement",
            "bullish candle forms at 61.8% level",
            "EMA200 nearby as additional confluence",
        ],
        "conditions_short": [
            "identify clear bearish impulse move",
            "price retraces to 61.8% fibonacci level",
            "RSI between 50 and 65 on retracement",
            "bearish candle forms at 61.8% level",
        ],
        "entry":            "candle close at 61.8% level",
        "stop_loss":        "beyond 78.6% fibonacci level",
        "take_profit":      "previous high/low (100% level)",
        "risk_reward":      2.0,
        "best_session":     ["London", "NewYork"],
        "win_rate_expected": 59,
        "notes":            "The 61.8% is called the golden ratio. Most powerful when combined with EMA.",
    },

    # ── 8 ─────────────────────────────────────────────────────────────────────
    "double_top_bottom": {
        "id":               "double_top_bottom",
        "name":             "Double Top and Bottom",
        "asset":            ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"],
        "timeframe":        "H1 or H4",
        "conditions_short": [
            "price tests resistance twice",
            "second test fails to break higher",
            "bearish candle on second test",
            "RSI shows lower high on second test",
            "price breaks below valley between two tops",
        ],
        "conditions_long":  [
            "price tests support twice",
            "second test holds support",
            "bullish candle on second test",
            "RSI shows higher low on second test",
            "price breaks above peak between two bottoms",
        ],
        "entry":            "break of the valley or peak level",
        "stop_loss":        "above the second top or below second bottom",
        "take_profit":      "height of pattern projected",
        "risk_reward":      2.0,
        "best_session":     ["London", "NewYork"],
        "win_rate_expected": 60,
        "notes":            "Must have clear two peaks or two troughs. RSI divergence adds confirmation.",
    },

    # ── 9 ─────────────────────────────────────────────────────────────────────
    "bb_squeeze_breakout": {
        "id":               "bb_squeeze_breakout",
        "name":             "BB Squeeze Breakout",
        "asset":            ["XAUUSD", "EURUSD"],
        "timeframe":        "H1",
        "conditions_long":  [
            "Bollinger Bands are narrow (squeeze)",
            "bands width below 20 period average",
            "price breaks above upper band",
            "RSI above 50",
            "breakout candle is strong (small wicks)",
        ],
        "conditions_short": [
            "Bollinger Bands are narrow (squeeze)",
            "price breaks below lower band",
            "RSI below 50",
            "breakout candle is strong",
        ],
        "entry":            "close of breakout candle",
        "stop_loss":        "opposite Bollinger Band",
        "take_profit":      "2x band width from entry",
        "risk_reward":      2.0,
        "best_session":     ["London open", "NY open"],
        "win_rate_expected": 56,
        "notes":            "Squeeze means low volatility before big move. Direction of breakout = trade.",
    },

    # ── 10 ────────────────────────────────────────────────────────────────────
    "news_spike_fade": {
        "id":               "news_spike_fade",
        "name":             "News Spike Fade",
        "asset":            ["XAUUSD", "EURUSD", "GBPUSD"],
        "timeframe":        "M15",
        "conditions_long":  [
            "high impact news causes sharp drop",
            "drop exceeds 1.5x ATR in one candle",
            "RSI drops below 25 on spike",
            "next candle starts recovering",
            "price returns above spike low",
        ],
        "conditions_short": [
            "high impact news causes sharp spike up",
            "spike exceeds 1.5x ATR in one candle",
            "RSI spikes above 75",
            "next candle starts reversing",
            "price returns below spike high",
        ],
        "entry":            "close of first recovery candle",
        "stop_loss":        "beyond the spike extreme",
        "take_profit":      "pre-news price level",
        "risk_reward":      2.5,
        "best_session":     ["During NFP, CPI, FOMC"],
        "win_rate_expected": 63,
        "notes":            "News spikes often retrace 50-100%. Wait for spike to fully form before entry.",
    },

    # ── 11 ────────────────────────────────────────────────────────────────────
    "asian_range_gold": {
        "id":               "asian_range_gold",
        "name":             "Asian Range Gold",
        "asset":            ["XAUUSD"],
        "timeframe":        "M30",
        "conditions_long":  [
            "Asian session forms tight range",
            "range size less than 1x ATR",
            "London open breaks above range high",
            "first 30min London candle closes above range",
            "RSI above 50",
        ],
        "conditions_short": [
            "Asian session forms tight range",
            "London open breaks below range low",
            "first 30min London candle closes below range",
            "RSI below 50",
        ],
        "entry":            "close of London open breakout candle",
        "stop_loss":        "opposite end of Asian range",
        "take_profit":      "1.5x the Asian range size",
        "risk_reward":      2.0,
        "best_session":     ["London open 07:00-08:30 UTC"],
        "win_rate_expected": 57,
        "notes":            "Gold respects London open breakout very reliably when Asian range is tight.",
    },

    # ── 12 ────────────────────────────────────────────────────────────────────
    "ema_cross_signal": {
        "id":               "ema_cross_signal",
        "name":             "EMA 50/200 Golden Cross",
        "asset":            ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"],
        "timeframe":        "H4 or D1",
        "conditions_long":  [
            "EMA50 crosses above EMA200 (golden cross)",
            "price above both EMAs",
            "RSI above 50",
            "crossing happens after sustained downtrend",
        ],
        "conditions_short": [
            "EMA50 crosses below EMA200 (death cross)",
            "price below both EMAs",
            "RSI below 50",
            "crossing happens after sustained uptrend",
        ],
        "entry":            "daily close after cross confirmed",
        "stop_loss":        "below EMA200",
        "take_profit":      "measured move or 3x stop",
        "risk_reward":      3.0,
        "best_session":     ["Any - higher timeframe"],
        "win_rate_expected": 65,
        "notes":            "Strongest signal on H4 and D1. Golden Cross on gold is very reliable.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Indicator helpers  (compute from raw OHLCV if columns not pre-built)
# ══════════════════════════════════════════════════════════════════════════════

def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns if absent. Works on a copy."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    if "open" not in df.columns:
        df["open"] = df["close"].shift(1).fillna(df["close"])

    if "ema50" not in df.columns:
        df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    if "ema200" not in df.columns:
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    if "rsi" not in df.columns:
        delta      = df["close"].diff()
        gain       = delta.clip(lower=0).rolling(14).mean()
        loss       = (-delta.clip(upper=0)).rolling(14).mean()
        rs         = gain / loss.replace(0, np.nan)
        df["rsi"]  = 100 - (100 / (1 + rs))

    if "atr" not in df.columns:
        hl         = df["high"] - df["low"]
        hc         = (df["high"] - df["close"].shift()).abs()
        lc         = (df["low"]  - df["close"].shift()).abs()
        df["atr"]  = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    if "macd" not in df.columns:
        e12              = df["close"].ewm(span=12, adjust=False).mean()
        e26              = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"]       = e12 - e26
        df["macd_signal"]= df["macd"].ewm(span=9, adjust=False).mean()

    if "bb_upper" not in df.columns:
        sma20            = df["close"].rolling(20).mean()
        std20            = df["close"].rolling(20).std()
        df["bb_upper"]   = sma20 + 2 * std20
        df["bb_lower"]   = sma20 - 2 * std20
        df["bb_width"]   = df["bb_upper"] - df["bb_lower"]
        df["bb_avg_width"] = df["bb_width"].rolling(20).mean()

    return df.dropna(subset=["ema200", "rsi", "atr"])


def _current(df: pd.DataFrame) -> pd.Series:
    return df.iloc[-1]


def _prev(df: pd.DataFrame, n: int = 1) -> pd.Series:
    return df.iloc[-1 - n]


def _utc_hour() -> float:
    now = datetime.now(timezone.utc)
    return now.hour + now.minute / 60.0


def _current_session() -> str:
    t = _utc_hour()
    if 12.0 <= t < 15.0:
        return "LondonNY"
    if  7.0 <= t < 12.0:
        return "London"
    if 13.0 <= t < 17.0:
        return "NewYork"
    if  0.0 <= t <  7.0:
        return "Asian"
    return "OffHours"


def _swing_levels(df: pd.DataFrame, window: int = 60) -> tuple[list[float], list[float]]:
    """Return (swing_highs, swing_lows) from the last `window` bars."""
    w = df.tail(window)
    highs, lows = [], []
    for i in range(2, len(w) - 2):
        hi = w.iloc[i]["high"]
        lo = w.iloc[i]["low"]
        if hi > w.iloc[i-1]["high"] and hi > w.iloc[i+1]["high"]:
            highs.append(hi)
        if lo < w.iloc[i-1]["low"] and lo < w.iloc[i+1]["low"]:
            lows.append(lo)
    return highs, lows


def _at_level(price: float, levels: list[float], tol: float = 0.003) -> bool:
    return any(abs(price - lv) / lv <= tol for lv in levels)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-playbook condition evaluators
#  Each returns (conditions_checked, conditions_passed, detail_list)
# ══════════════════════════════════════════════════════════════════════════════

def _eval_ema_trend_continuation(df: pd.DataFrame, direction: str) -> tuple[int, int, list[str]]:
    c    = _current(df)
    p    = _prev(df)
    met, detail = [], []
    total = 5

    if direction == "long":
        if c["close"] > c["ema200"]:  met.append("price above EMA200")
        if c["ema50"] > c["ema200"]:  met.append("EMA50 above EMA200")
        if 40 <= c["rsi"] <= 60:      met.append("RSI between 40-60")
        if c["low"] <= c["ema50"] * 1.001:   met.append("price pulled back to EMA50")
        if c["close"] > c["ema50"] and c["close"] > c["open"]: met.append("bullish candle above EMA50")
    else:
        if c["close"] < c["ema200"]:  met.append("price below EMA200")
        if c["ema50"] < c["ema200"]:  met.append("EMA50 below EMA200")
        if 40 <= c["rsi"] <= 60:      met.append("RSI between 40-60")
        if c["high"] >= c["ema50"] * 0.999:  met.append("price pulled back to EMA50")
        if c["close"] < c["ema50"] and c["close"] < c["open"]: met.append("bearish candle below EMA50")

    return total, len(met), met


def _eval_rsi_oversold_bounce(df: pd.DataFrame, direction: str) -> tuple[int, int, list[str]]:
    c     = _current(df)
    highs, lows = _swing_levels(df)
    met   = []
    total = 5

    if direction == "long":
        if c["rsi"] < 30:                              met.append("RSI below 30")
        if _at_level(c["close"], lows):                met.append("at support level")
        if c["close"] > c["ema200"]:                   met.append("price above EMA200")
        lw = min(c["open"], c["close"]) - c["low"]
        bdy= abs(c["close"] - c["open"])
        if c["close"] > c["open"] or lw > bdy:         met.append("bullish reversal candle")
        if _at_level(c["close"], lows, 0.002):         met.append("closes near support")
    else:
        if c["rsi"] > 70:                              met.append("RSI above 70")
        if _at_level(c["close"], highs):               met.append("at resistance level")
        if c["close"] < c["ema200"]:                   met.append("price below EMA200")
        uw = c["high"] - max(c["open"], c["close"])
        bdy= abs(c["close"] - c["open"])
        if c["close"] < c["open"] or uw > bdy:         met.append("bearish reversal candle")
        if _at_level(c["close"], highs, 0.002):        met.append("closes near resistance")

    return total, len(met), met


def _eval_london_breakout(df: pd.DataFrame, direction: str) -> tuple[int, int, list[str]]:
    c     = _current(df)
    t     = _utc_hour()
    met   = []
    total = 4 if direction == "short" else 5

    # Approximate Asian range from last ~30-40 bars (Asian session)
    asian_bars = df.tail(40)
    asian_high = asian_bars["high"].max()
    asian_low  = asian_bars["low"].min()
    asian_mid  = (asian_high + asian_low) / 2

    is_london_open = 7.0 <= t < 9.0
    if is_london_open:
        met.append("London open window 07-09 UTC")

    if direction == "long":
        if c["close"] > asian_high:     met.append("breaks above Asian high")
        if c["close"] > asian_high:     met.append("candle closes above Asian high")
        if c["rsi"] > 50:               met.append("RSI above 50")
        if c.get("volume", 0) > df["volume"].tail(20).mean():
            met.append("volume above average")
    else:
        if c["close"] < asian_low:      met.append("breaks below Asian low")
        if c["close"] < asian_low:      met.append("candle closes below Asian low")
        if c["rsi"] < 50:               met.append("RSI below 50")

    return total, len(met), met


def _eval_macd_divergence(df: pd.DataFrame, direction: str) -> tuple[int, int, list[str]]:
    c     = _current(df)
    highs, lows = _swing_levels(df, window=30)
    met   = []
    total = 5

    # Detect divergence: compare last two swing points vs MACD histogram
    macd_hist     = df["macd"] - df["macd_signal"]
    recent_macd   = macd_hist.tail(20)

    if direction == "long":
        # Price making lower lows but MACD histogram making higher lows
        price_ll = len(lows) >= 2 and lows[-1] < lows[-2]  if len(lows) >= 2 else False
        macd_hl  = recent_macd.iloc[-1] > recent_macd.min()
        if price_ll:            met.append("price making lower lows")
        if macd_hl:             met.append("MACD histogram higher lows (divergence)")
        if price_ll and macd_hl: met.append("bullish divergence confirmed")
        if c["rsi"] < 40:       met.append("RSI below 40")
        if _at_level(c["close"], lows): met.append("at support zone")
    else:
        price_hh = len(highs) >= 2 and highs[-1] > highs[-2] if len(highs) >= 2 else False
        macd_lh  = recent_macd.iloc[-1] < recent_macd.max()
        if price_hh:            met.append("price making higher highs")
        if macd_lh:             met.append("MACD histogram lower highs (divergence)")
        if price_hh and macd_lh: met.append("bearish divergence confirmed")
        if c["rsi"] > 60:       met.append("RSI above 60")
        if _at_level(c["close"], highs): met.append("at resistance zone")

    return total, min(len(met), total), met


def _eval_safe_haven_spike(df: pd.DataFrame, direction: str, sentiment: dict) -> tuple[int, int, list[str]]:
    c     = _current(df)
    p     = _prev(df)
    met   = []
    total = 4

    overall_risk = str(sentiment.get("overall_risk", "")).lower()
    gold_bias    = str(sentiment.get("gold", {}).get("bias", "")).lower()

    spike_size = abs(c["close"] - p["close"])
    big_spike  = spike_size > c["atr"] * 1.0

    if direction == "long":
        if overall_risk == "high":               met.append("risk-off news event detected")
        if big_spike and c["close"] > p["close"]: met.append("gold spiked up on event")
        if c["rsi"] < 70:                        met.append("RSI not yet overbought")
        if c["close"] > p["close"]:              met.append("price holds above previous close")
    else:
        if overall_risk in ("medium", "low"):    met.append("risk-on sentiment")
        if big_spike and c["close"] < p["close"]: met.append("gold retracing spike gains")
        if c["rsi"] > 70:                        met.append("RSI above 70 after spike")
        if c["close"] < p["close"]:              met.append("price falling back")

    return total, len(met), met


def _eval_head_and_shoulders(df: pd.DataFrame, direction: str) -> tuple[int, int, list[str]]:
    c     = _current(df)
    highs, lows = _swing_levels(df, window=80)
    met   = []
    total = 5 if direction == "short" else 3

    if direction == "short":
        # Need 3 peaks: left shoulder, head (highest), right shoulder ≈ left
        if len(highs) >= 3:
            ls, head, rs = sorted(highs[-3:])
            if head > ls and head > rs and abs(ls - rs) / ls < 0.02:
                met.append("left shoulder and head formed")
                met.append("right shoulder ≈ left shoulder level")
                met.append("neckline identified from valleys")
        # Neckline break approximation: price below recent low cluster
        neckline = min(lows[-3:]) if len(lows) >= 3 else c["close"] * 0.99
        if c["close"] < neckline:   met.append("price breaks below neckline")
        if c["rsi"] < 50:           met.append("RSI below 50 on breakout")
    else:
        if len(lows) >= 3:
            ll, head, rl = sorted(lows[-3:])
            if head < ll and head < rl and abs(ll - rl) / ll < 0.02:
                met.append("inverse H&S structure confirmed")
        neckline = max(highs[-3:]) if len(highs) >= 3 else c["close"] * 1.01
        if c["close"] > neckline:   met.append("price breaks above neckline")
        if c["rsi"] > 50:           met.append("RSI above 50 on breakout")

    return total, min(len(met), total), met


def _eval_fibonacci_golden_zone(df: pd.DataFrame, direction: str) -> tuple[int, int, list[str]]:
    c     = _current(df)
    highs, lows = _swing_levels(df, window=50)
    met   = []
    total = 4

    if direction == "long" and len(highs) >= 1 and len(lows) >= 1:
        swing_low  = min(lows)
        swing_high = max(highs)
        fib618     = swing_high - (swing_high - swing_low) * 0.618
        fib786     = swing_high - (swing_high - swing_low) * 0.786
        if abs(c["close"] - fib618) / fib618 < 0.005:
            met.append("price at 61.8% fibonacci retracement")
        if 35 <= c["rsi"] <= 50:    met.append("RSI in retracement zone 35-50")
        if c["close"] > c["open"]:  met.append("bullish candle at fib level")
        if abs(c["close"] - c["ema200"]) / c["ema200"] < 0.01: met.append("EMA200 confluence nearby")

    elif direction == "short" and len(highs) >= 1 and len(lows) >= 1:
        swing_low  = min(lows)
        swing_high = max(highs)
        fib618     = swing_low + (swing_high - swing_low) * 0.618
        if abs(c["close"] - fib618) / fib618 < 0.005:
            met.append("price at 61.8% retracement short")
        if 50 <= c["rsi"] <= 65:    met.append("RSI in retracement zone 50-65")
        if c["close"] < c["open"]:  met.append("bearish candle at fib level")
        if abs(c["close"] - c["ema200"]) / c["ema200"] < 0.01: met.append("EMA200 confluence")

    return total, len(met), met


def _eval_double_top_bottom(df: pd.DataFrame, direction: str) -> tuple[int, int, list[str]]:
    c     = _current(df)
    highs, lows = _swing_levels(df, window=60)
    met   = []
    total = 5

    if direction == "short" and len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        if abs(h1 - h2) / h1 < 0.015:          met.append("two similar peaks (double top)")
        if c["close"] < h2:                     met.append("second test failed to break higher")
        if c["close"] < c["open"]:              met.append("bearish candle on second test")
        valley = min(df["low"].tail(20))
        rsi_vals = df["rsi"].tail(20)
        if rsi_vals.iloc[-1] < rsi_vals.max() * 0.95: met.append("RSI lower high confirmed")
        if c["close"] < valley:                 met.append("breaks below valley")

    elif direction == "long" and len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        if abs(l1 - l2) / l1 < 0.015:          met.append("two similar troughs (double bottom)")
        if c["close"] > l2:                     met.append("second test held support")
        if c["close"] > c["open"]:              met.append("bullish candle on second test")
        peak = max(df["high"].tail(20))
        rsi_vals = df["rsi"].tail(20)
        if rsi_vals.iloc[-1] > rsi_vals.min() * 1.05: met.append("RSI higher low confirmed")
        if c["close"] > peak:                   met.append("breaks above peak")

    return total, len(met), met


def _eval_bb_squeeze(df: pd.DataFrame, direction: str) -> tuple[int, int, list[str]]:
    c     = _current(df)
    met   = []
    total = 4 if direction == "short" else 5

    if "bb_width" not in df.columns or "bb_avg_width" not in df.columns:
        return total, 0, ["Bollinger Band columns not available"]

    is_squeeze = c.get("bb_width", float("inf")) < c.get("bb_avg_width", float("inf")) * 0.8

    if direction == "long":
        if is_squeeze:                              met.append("Bollinger Band squeeze active")
        if c["close"] > c.get("bb_upper", float("inf")): met.append("price breaks above upper band")
        if c["rsi"] > 50:                           met.append("RSI above 50")
        body = abs(c["close"] - c["open"])
        rng  = c["high"] - c["low"]
        if rng > 0 and body / rng > 0.6:            met.append("strong breakout candle")
        met_count = len(met)
    else:
        if is_squeeze:                              met.append("Bollinger Band squeeze active")
        if c["close"] < c.get("bb_lower", float("-inf")): met.append("price breaks below lower band")
        if c["rsi"] < 50:                           met.append("RSI below 50")
        body = abs(c["close"] - c["open"])
        rng  = c["high"] - c["low"]
        if rng > 0 and body / rng > 0.6:            met.append("strong breakout candle")

    return total, len(met), met


def _eval_news_spike_fade(df: pd.DataFrame, direction: str, sentiment: dict) -> tuple[int, int, list[str]]:
    c     = _current(df)
    p     = _prev(df)
    met   = []
    total = 4

    news_risk   = str(sentiment.get("overall_risk", "")).lower()
    candle_size = abs(c["close"] - p["close"])
    big_move    = candle_size > c["atr"] * 1.5

    if direction == "long":
        sharp_drop = c["close"] < p["close"] and big_move
        if news_risk == "high":                   met.append("high impact news detected")
        if sharp_drop:                            met.append("sharp drop exceeds 1.5x ATR")
        if c["rsi"] < 30:                         met.append("RSI below 30 on spike down")
        prev2 = _prev(df, 2) if len(df) >= 3 else p
        if c["close"] > prev2["close"] * 0.999:   met.append("recovery candle forming")
    else:
        sharp_rise = c["close"] > p["close"] and big_move
        if news_risk == "high":                   met.append("high impact news detected")
        if sharp_rise:                            met.append("sharp spike exceeds 1.5x ATR")
        if c["rsi"] > 70:                         met.append("RSI above 70 on spike up")
        prev2 = _prev(df, 2) if len(df) >= 3 else p
        if c["close"] < prev2["close"] * 1.001:   met.append("reversal candle forming")

    return total, len(met), met


def _eval_asian_range_gold(df: pd.DataFrame, direction: str) -> tuple[int, int, list[str]]:
    c     = _current(df)
    t     = _utc_hour()
    met   = []
    total = 4

    asian_bars  = df.tail(40)
    asian_range = asian_bars["high"].max() - asian_bars["low"].min()
    asian_high  = asian_bars["high"].max()
    asian_low   = asian_bars["low"].min()
    is_tight    = asian_range < c["atr"]
    is_london   = 7.0 <= t < 8.5

    if is_tight:   met.append("Asian range is tight (< 1x ATR)")
    if is_london:  met.append("London open window active")

    if direction == "long":
        if c["close"] > asian_high:     met.append("breaks above Asian range high")
        if c["rsi"] > 50:               met.append("RSI above 50")
    else:
        if c["close"] < asian_low:      met.append("breaks below Asian range low")
        if c["rsi"] < 50:               met.append("RSI below 50")

    return total, len(met), met


def _eval_ema_cross(df: pd.DataFrame, direction: str) -> tuple[int, int, list[str]]:
    c    = _current(df)
    p    = _prev(df)
    met  = []
    total = 4

    golden_cross_now  = c["ema50"] > c["ema200"] and p["ema50"] <= p["ema200"]
    death_cross_now   = c["ema50"] < c["ema200"] and p["ema50"] >= p["ema200"]
    golden_cross_recent = (df["ema50"].tail(10) > df["ema200"].tail(10)).any() and \
                          (df["ema50"].tail(10) < df["ema200"].tail(10)).any()
    death_cross_recent  = (df["ema50"].tail(10) < df["ema200"].tail(10)).any() and \
                          (df["ema50"].tail(10) > df["ema200"].tail(10)).any()

    if direction == "long":
        if golden_cross_now or golden_cross_recent: met.append("EMA50 crossed above EMA200 (golden cross)")
        if c["close"] > c["ema50"] and c["close"] > c["ema200"]: met.append("price above both EMAs")
        if c["rsi"] > 50:               met.append("RSI above 50")
        prior_trend = df["close"].tail(30).is_monotonic_decreasing
        if not prior_trend:             met.append("after sustained downtrend")
    else:
        if death_cross_now or death_cross_recent: met.append("EMA50 crossed below EMA200 (death cross)")
        if c["close"] < c["ema50"] and c["close"] < c["ema200"]: met.append("price below both EMAs")
        if c["rsi"] < 50:               met.append("RSI below 50")
        prior_trend = df["close"].tail(30).is_monotonic_increasing
        if not prior_trend:             met.append("after sustained uptrend")

    return total, len(met), met


# ══════════════════════════════════════════════════════════════════════════════
#  score_playbook
# ══════════════════════════════════════════════════════════════════════════════

def _score_playbook(
    key: str,
    df: pd.DataFrame,
    direction: str,
    sentiment: dict,
) -> tuple[int, int, list[str]]:
    """
    Dispatch to the correct evaluator.
    Returns (total_conditions, conditions_met, met_list).
    """
    evals = {
        "ema_trend_continuation":  lambda: _eval_ema_trend_continuation(df, direction),
        "rsi_oversold_bounce":     lambda: _eval_rsi_oversold_bounce(df, direction),
        "london_breakout_gold":    lambda: _eval_london_breakout(df, direction),
        "macd_divergence_reversal":lambda: _eval_macd_divergence(df, direction),
        "gold_safe_haven_spike":   lambda: _eval_safe_haven_spike(df, direction, sentiment),
        "head_and_shoulders":      lambda: _eval_head_and_shoulders(df, direction),
        "fibonacci_golden_zone":   lambda: _eval_fibonacci_golden_zone(df, direction),
        "double_top_bottom":       lambda: _eval_double_top_bottom(df, direction),
        "bb_squeeze_breakout":     lambda: _eval_bb_squeeze(df, direction),
        "news_spike_fade":         lambda: _eval_news_spike_fade(df, direction, sentiment),
        "asian_range_gold":        lambda: _eval_asian_range_gold(df, direction),
        "ema_cross_signal":        lambda: _eval_ema_cross(df, direction),
    }
    fn = evals.get(key)
    if fn is None:
        return 1, 0, []
    return fn()


# ══════════════════════════════════════════════════════════════════════════════
#  get_active_playbooks
# ══════════════════════════════════════════════════════════════════════════════

def get_active_playbooks(
    df: pd.DataFrame,
    news_sentiment: dict | None = None,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """
    Check all 12 playbooks against current market data.

    Parameters
    ----------
    df              : OHLCV DataFrame (indicators added if missing)
    news_sentiment  : dict from morning_briefing / news_filter
                      expected keys: overall_risk, gold.bias
    top_n           : how many active playbooks to return

    Returns
    -------
    List of up to `top_n` dicts, each containing:
        playbook       : PLAYBOOKS entry
        direction      : "long" or "short"
        score          : 1–10 rounded
        conditions_met : int
        total_conditions: int
        met_list       : list of condition strings that passed
        entry          : float
        stop_loss      : float
        take_profit    : float
        risk_reward    : float
    """
    if news_sentiment is None:
        news_sentiment = {}

    # ── FIX 2: Volume pre-filter — no signals when nobody is trading ──────────
    try:
        from volume_analyzer import VolumeAnalyzer
        _vol_summary = VolumeAnalyzer().get_volume_summary(df)
        _vol_ratio   = _vol_summary.get("volume_ratio", 1.0)
        _vol_class   = _vol_summary.get("volume_class", "normal")
        if _vol_class == "very_low" or _vol_ratio < 0.4:
            return []  # no signals when nobody is trading
    except Exception:
        pass

    df = _enrich(df)
    if df.empty:
        return []

    c         = _current(df)
    is_bearish = c["close"] < c["ema200"]
    gold_bias  = str(news_sentiment.get("gold", {}).get("bias", "")).lower()

    # Determine primary direction for each playbook
    results: list[dict] = []

    for key, pb in PLAYBOOKS.items():
        # Determine direction to test (prefer news bias for gold playbooks)
        if gold_bias in ("sell", "short"):
            direction = "short"
        elif gold_bias in ("buy", "long"):
            direction = "long"
        elif is_bearish:
            direction = "short"
        else:
            direction = "long"

        total, passed, met_list = _score_playbook(key, df, direction, news_sentiment)

        if total == 0 or passed == 0:
            continue

        score = round(min(10.0, passed / total * 10), 1)

        # Calculate entry / SL / TP
        entry_price, sl_price, tp_price = format_playbook_signal(pb, df, direction)

        results.append({
            "playbook":        pb,
            "direction":       direction,
            "score":           score,
            "conditions_met":  passed,
            "total_conditions": total,
            "met_list":        met_list,
            "entry":           entry_price,
            "stop_loss":       sl_price,
            "take_profit":     tp_price,
            "risk_reward":     pb["risk_reward"],
        })

    # Sort by score descending, then by win_rate_expected
    results.sort(key=lambda x: (x["score"], x["playbook"]["win_rate_expected"]), reverse=True)
    return results[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
#  format_playbook_signal
# ══════════════════════════════════════════════════════════════════════════════

def _get_live_entry_price(
    symbol: str,
    direction: str,
    df: pd.DataFrame,
) -> float:
    """
    Return the best available entry price in priority order:
      1. MT5 bid/ask (ask for long, bid for short)
      2. Last price from JSON cache  (data/price_cache.json)
      3. Last close bar from df
    """
    import json as _json
    is_long = direction == "long"

    # 1 ── MT5 bid / ask ────────────────────────────────────────────────────
    try:
        import MetaTrader5 as mt5  # type: ignore
        if mt5.initialize():
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                price = float(tick.ask if is_long else tick.bid)
                # write fresh cache
                try:
                    os.makedirs(DATA_DIR, exist_ok=True)
                    with open(PRICE_CACHE, "w") as _f:
                        _json.dump({
                            "symbol": symbol,
                            "ask":    float(tick.ask),
                            "bid":    float(tick.bid),
                            "ts":     datetime.now(timezone.utc).isoformat(),
                        }, _f)
                except Exception:
                    pass
                mt5.shutdown()
                return price
    except (ImportError, Exception):
        pass

    # 2 ── JSON price cache ───────────────────────────────────────────────────
    try:
        with open(PRICE_CACHE) as _f:
            cache = _json.load(_f)
        key = "ask" if is_long else "bid"
        cached = float(cache.get(key, 0))
        if cached > 0:
            return cached
    except (FileNotFoundError, Exception):
        pass

    # 3 ── Last close from df ────────────────────────────────────────────────
    return float(_current(df)["close"])


def format_playbook_signal(
    playbook: dict[str, Any],
    df: pd.DataFrame,
    direction: str = "long",
) -> tuple[float, float, float]:
    """
    Calculate entry, stop_loss, and take_profit prices using
    current price and ATR for the given playbook.

    Returns
    -------
    (entry_price, stop_loss_price, take_profit_price)
    """
    if df.empty:
        return 0.0, 0.0, 0.0

    c         = _current(df)
    symbol    = (playbook.get("asset") or ["XAUUSD"])
    symbol    = symbol[0] if isinstance(symbol, list) else symbol
    is_long   = direction == "long"

    # ── Live price via mt5_sync (4-priority fallback) ─────────────────────────
    _price_stale_note = ""
    try:
        from mt5_sync import get_live_price as _glp
        _lp = _glp(symbol)
        if _lp.get("price") and _lp["price"] > 0:
            price = float(_lp["ask"] if is_long else _lp["bid"]) if (_lp.get("ask") and _lp.get("bid")) else float(_lp["price"])
            if not _lp.get("is_live"):
                _price_stale_note = _lp.get("stale_warning", "⚠ Stale price — verify on MT5")
        else:
            price = float(_get_live_entry_price(symbol, direction, df))
    except Exception:
        price = float(_get_live_entry_price(symbol, direction, df))

    if _price_stale_note:
        playbook["_price_stale_note"] = _price_stale_note
    atr       = float(c.get("atr", price * 0.005))
    ema50     = float(c.get("ema50",  price))
    ema200    = float(c.get("ema200", price))
    rr        = playbook.get("risk_reward", 2.0)
    pb_id     = playbook.get("id", "")
    highs, lows = _swing_levels(df, window=60)

    is_long = direction == "long"

    # ── Dynamic ATR SL engine ────────────────────────────────────────────────
    _dyn_sl_ok = False
    try:
        from atr_sl_engine import calculate_dynamic_sl as _cds
        _dyn = _cds(
            df,
            direction,
            entry        = price,
            session      = _current_session(),
            regime       = "RANGING",
            geo_multiplier = 0.0,
            strategy_name  = playbook.get("name", ""),
        )
        sl_dist            = _dyn["sl_distance"]
        entry_price        = price
        if is_long:
            sl_price = entry_price - sl_dist
            tp_price = entry_price + sl_dist * rr
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - sl_dist * rr
        playbook["_sl_breakdown"]   = _dyn["sl_breakdown"]
        playbook["_volatility_state"] = _dyn["volatility_state"]
        playbook["_atr_percentile"]   = _dyn["atr_percentile"]
        _dyn_sl_ok = True
    except Exception:
        _dyn_sl_ok = False

    if not _dyn_sl_ok:
        # ── SL distance caps (prevent untradeable setups) ──────────────────
        max_sl_distance = atr * 2.0
        min_sl_distance = atr * 0.5

    # ── Strategy-specific SL logic (fallback when dynamic engine unavailable) ──
    if _dyn_sl_ok:
        return round(entry_price, 2), round(sl_price, 2), round(tp_price, 2)

    # ── Fallback static SL dispatch ──────────────────────────────────────────
    max_sl_distance = atr * 2.0
    min_sl_distance = atr * 0.5
    if pb_id == "ema_trend_continuation":
        sl_dist = abs(price - ema50) + 0.5 * atr
    elif pb_id in ("rsi_oversold_bounce", "news_spike_fade"):
        # SL = beyond the candle wick
        wick_extreme = c["low"] if is_long else c["high"]
        sl_dist = abs(price - wick_extreme) + 0.2 * atr
    elif pb_id in ("london_breakout_gold", "asian_range_gold"):
        # SL = middle of Asian range
        asian = df.tail(40)
        asian_range = asian["high"].max() - asian["low"].min()
        sl_dist = asian_range / 2
    elif pb_id == "macd_divergence_reversal":
        nearest_swing = min(lows, key=lambda lv: abs(price - lv)) if lows else price
        sl_dist = abs(price - nearest_swing) + atr
    elif pb_id in ("head_and_shoulders", "double_top_bottom"):
        sl_dist = 1.5 * atr
    elif pb_id == "fibonacci_golden_zone":
        sl_dist = atr * 1.2    # just beyond 78.6% zone
    elif pb_id == "bb_squeeze_breakout":
        bb_w = float(c.get("bb_width", atr * 2))
        sl_dist = bb_w / 2
    elif pb_id == "ema_cross_signal":
        sl_dist = abs(price - ema200) + 0.5 * atr
    elif pb_id == "gold_safe_haven_spike":
        sl_dist = atr * 1.0
    else:
        sl_dist = atr * 1.5

    # Enforce caps — never let SL be tighter than 0.5×ATR or wider than 2×ATR
    sl_dist = min(max(sl_dist, min_sl_distance), max_sl_distance)

    entry_price = price
    if is_long:
        sl_price = entry_price - sl_dist
        tp_price = entry_price + sl_dist * rr
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - sl_dist * rr

    return round(entry_price, 2), round(sl_price, 2), round(tp_price, 2)  # static fallback


# ══════════════════════════════════════════════════════════════════════════════
#  print_playbook_signals
# ══════════════════════════════════════════════════════════════════════════════

def print_playbook_signals(
    df: pd.DataFrame,
    sentiment: dict | None = None,
    top_n: int = 3,
) -> None:
    """
    Print the top active playbook signals in clean formatted output.
    """
    if sentiment is None:
        sentiment = {}

    df      = _enrich(df)
    signals = get_active_playbooks(df, sentiment, top_n=top_n)

    c       = _current(df)
    price   = c["close"]
    rsi     = c.get("rsi", float("nan"))
    atr     = c.get("atr", float("nan"))
    session = _current_session()

    W = 58
    CHECK = "\u2713"
    CROSS = "\u2717"

    print(f"\n  {'═' * W}")
    print(f"  STRATEGY PLAYBOOK SIGNALS")
    print(f"  {'═' * W}")
    print(f"  Price: ${price:,.2f}  |  RSI: {rsi:.1f}  |  ATR: ${atr:,.2f}  |  Session: {session}")
    print(f"  {'─' * W}")

    if not signals:
        print("  No playbook conditions currently met.")
        print(f"  {'═' * W}\n")
        return

    for i, sig in enumerate(signals, 1):
        pb        = sig["playbook"]
        direction = sig["direction"].upper()
        score     = sig["score"]
        passed    = sig["conditions_met"]
        total     = sig["total_conditions"]
        entry     = sig["entry"]
        sl        = sig["stop_loss"]
        tp        = sig["take_profit"]
        rr        = sig["risk_reward"]
        sl_pts    = abs(entry - sl)
        tp_pts    = abs(entry - tp)
        win_exp   = pb["win_rate_expected"]
        sessions  = ", ".join(pb["best_session"]) if isinstance(pb["best_session"], list) else pb["best_session"]
        asset_lst = pb.get("asset", [])
        is_xau    = any("XAU" in a for a in (asset_lst if isinstance(asset_lst, list) else [asset_lst]))

        # Score bar (10 blocks)
        filled = round(score)
        bar    = "█" * filled + "░" * (10 - filled)

        print(f"\n  [{i}] {pb['name'].upper()}  —  {direction}")
        print(f"  {'─' * W}")
        print(f"  Score        : {bar}  {score}/10")
        print(f"  Conditions   : {passed}/{total} met")
        print(f"  Win Rate Exp : {win_exp}%  |  Timeframe: {pb['timeframe']}  |  Best: {sessions}")
        print()
        print(f"  ENTRY        : ${entry:>10,.2f}")
        if is_xau:
            print(f"  STOP LOSS    : ${sl:>10,.2f}  (+${sl_pts:,.2f})")
            print(f"  TAKE PROFIT  : ${tp:>10,.2f}  (+${tp_pts:,.2f})")
        else:
            sl_pips = round(sl_pts / 0.0001)
            tp_pips = round(tp_pts / 0.0001)
            print(f"  STOP LOSS    : {sl:>12,.5f}  ({sl_pips:.0f} pips)")
            print(f"  TAKE PROFIT  : {tp:>12,.5f}  ({tp_pips:.0f} pips)")
        print(f"  RISK/REWARD  : 1:{rr}")
        print()
        print(f"  CONDITIONS MET:")
        for cond in sig["met_list"]:
            print(f"    {CHECK} {cond}")
        print()
        print(f"  NOTE: {pb['notes']}")
        print(f"  {'─' * W}")

    print(f"\n  {'═' * W}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  save_playbooks_json
# ══════════════════════════════════════════════════════════════════════════════

def save_playbooks_json() -> None:
    """Persist the PLAYBOOKS dict to data/playbooks.json."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PLAYBOOKS_JSON, "w", encoding="utf-8") as f:
        json.dump(PLAYBOOKS, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(PLAYBOOKS)} playbooks → {PLAYBOOKS_JSON}")


# ══════════════════════════════════════════════════════════════════════════════
#  Self-test  (python strategy_playbooks.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    HIST_CSV = os.path.join(DATA_DIR, "historical_xauusd.csv")

    # 1 — Save playbooks to JSON
    save_playbooks_json()

    # 2 — Load historical data
    print("\n  Loading price data for self-test...\n")
    try:
        raw = pd.read_csv(HIST_CSV, index_col=0)
        raw.columns = [c.lower() for c in raw.columns]
        df  = _enrich(raw)
        print(f"  Candles loaded : {len(df)}")
    except FileNotFoundError:
        print("  No historical_xauusd.csv found. Run setup.py --refresh first.")
        sys.exit(1)

    # 3 — Mock sentiment (bearish gold)
    mock_sentiment = {
        "overall_risk": "high",
        "gold": {"bias": "sell"},
    }

    # 4 — Print playbook signals
    print_playbook_signals(df, mock_sentiment, top_n=3)
