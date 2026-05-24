"""
market_context.py — 3-layer market awareness module
=====================================================
Layer A : detect_gold_regime(df)          → regime + playbook hints + size multiplier
Layer B : score_news_sentiment(events)    → news_bias + news_score + trade_with_news
Layer C : get_pattern_win_rate(...)       → win_rate + confidence_boost
          save_trade_outcome(...)         → persist result to data/pattern_memory.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ── Storage ───────────────────────────────────────────────────────────────────
DATA_DIR            = os.path.join(os.path.dirname(__file__), "data")
PATTERN_MEMORY_FILE = os.path.join(DATA_DIR, "pattern_memory.json")


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_memory() -> list[dict]:
    _ensure_data_dir()
    if not os.path.exists(PATTERN_MEMORY_FILE):
        return []
    try:
        with open(PATTERN_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_memory(records: list[dict]) -> None:
    _ensure_data_dir()
    with open(PATTERN_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER A — Regime Detection
# ══════════════════════════════════════════════════════════════════════════════

REGIME_LABELS = {
    "TRENDING_STRONG":    "🟢 Strong Trend",
    "TRENDING_WEAK":      "🟡 Weak Trend",
    "RANGING":            "🔵 Ranging",
    "VOLATILE_EXPANDING": "🔴 Volatile / Expanding",
    "SQUEEZE_BUILDING":   "⚪ Squeeze Building",
}

# Which playbooks fit each regime
REGIME_PLAYBOOKS: dict[str, dict[str, list[str]]] = {
    "TRENDING_STRONG": {
        "best":  ["OB_Breakout", "SMC_Trend_Follow", "EMA_Pullback", "HTF_Continuation"],
        "avoid": ["Mean_Reversion", "Range_Trade", "Fade"],
    },
    "TRENDING_WEAK": {
        "best":  ["EMA_Pullback", "Structure_Break", "OB_Retest"],
        "avoid": ["Range_Trade", "Fade"],
    },
    "RANGING": {
        "best":  ["Range_Trade", "Mean_Reversion", "S/R_Bounce", "OB_Retest"],
        "avoid": ["OB_Breakout", "Breakout_Follow", "HTF_Continuation"],
    },
    "VOLATILE_EXPANDING": {
        "best":  ["News_Fade", "Spike_Reversal", "Volatility_Expansion"],
        "avoid": ["Tight_SL", "Range_Trade", "Mean_Reversion", "EMA_Pullback"],
    },
    "SQUEEZE_BUILDING": {
        "best":  ["Pre_Breakout", "Compression_Entry", "Limit_Order_Trap"],
        "avoid": ["Momentum_Follow", "HTF_Continuation"],
    },
}

# Position-size multipliers per regime
REGIME_SIZE_MULTIPLIER: dict[str, float] = {
    "TRENDING_STRONG":    1.2,
    "TRENDING_WEAK":      1.0,
    "RANGING":            0.8,
    "VOLATILE_EXPANDING": 0.5,
    "SQUEEZE_BUILDING":   0.6,
}


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"] if "high" in df.columns else df["High"]
    low  = df["low"]  if "low"  in df.columns else df["Low"]
    clos = df["close"] if "close" in df.columns else df["Close"]
    tr = pd.concat([
        high - low,
        (high - clos.shift(1)).abs(),
        (low  - clos.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def detect_gold_regime(df: pd.DataFrame) -> dict[str, Any]:
    """
    Classify the current gold market regime from OHLCV + indicator DataFrame.

    Returns
    -------
    regime                : str  one of the REGIME_LABELS keys
    regime_label          : str  human-readable label with emoji
    best_playbooks        : list[str]
    avoid_playbooks       : list[str]
    position_size_multiplier : float
    regime_note           : str  short explanation
    atr_now               : float
    atr_avg               : float
    ema50_slope           : float
    """
    try:
        close = df["close"] if "close" in df.columns else df["Close"]
        high  = df["high"]  if "high"  in df.columns else df["High"]
        low   = df["low"]   if "low"   in df.columns else df["Low"]

        atr_series = _atr(df, 14)
        atr_now    = float(atr_series.iloc[-1])
        atr_avg    = float(atr_series.iloc[-50:].mean()) if len(atr_series) >= 50 else atr_now

        ema50 = _ema(close, 50)
        slope = (float(ema50.iloc[-1]) - float(ema50.iloc[-10])) / 10.0 if len(ema50) >= 10 else 0.0

        lookback = min(20, len(close))
        price_range = float(high.iloc[-lookback:].max()) - float(low.iloc[-lookback:].min())

        # ── Classify ────────────────────────────────────────────────────────
        if atr_now > atr_avg * 1.8:
            regime = "VOLATILE_EXPANDING"
            note   = f"ATR {atr_now:.1f} is {atr_now/atr_avg:.1f}× avg — extreme volatility"
        elif atr_now < atr_avg * 0.6:
            regime = "SQUEEZE_BUILDING"
            note   = f"ATR {atr_now:.1f} is only {atr_now/atr_avg:.0%} of avg — compression"
        elif abs(slope) > 0.5 and atr_now > atr_avg * 1.2:
            regime = "TRENDING_STRONG"
            dir_s  = "UP" if slope > 0 else "DOWN"
            note   = f"EMA50 slope {slope:+.2f} ({dir_s}) with above-avg ATR — strong trend"
        elif abs(slope) > 0.1:
            regime = "TRENDING_WEAK"
            dir_s  = "UP" if slope > 0 else "DOWN"
            note   = f"EMA50 slope {slope:+.2f} ({dir_s}) — weak trend"
        else:
            regime = "RANGING"
            note   = (f"EMA50 slope {slope:+.2f} (flat), "
                      f"20-bar range ${price_range:,.0f} — ranging market")

        pb_info = REGIME_PLAYBOOKS[regime]
        return {
            "regime":                   regime,
            "regime_label":             REGIME_LABELS[regime],
            "best_playbooks":           pb_info["best"],
            "avoid_playbooks":          pb_info["avoid"],
            "position_size_multiplier": REGIME_SIZE_MULTIPLIER[regime],
            "regime_note":              note,
            "atr_now":                  round(atr_now, 2),
            "atr_avg":                  round(atr_avg, 2),
            "ema50_slope":              round(slope, 4),
        }

    except Exception as e:
        return {
            "regime":                   "RANGING",
            "regime_label":             REGIME_LABELS["RANGING"],
            "best_playbooks":           REGIME_PLAYBOOKS["RANGING"]["best"],
            "avoid_playbooks":          REGIME_PLAYBOOKS["RANGING"]["avoid"],
            "position_size_multiplier": 1.0,
            "regime_note":              f"Regime detection failed: {e}",
            "atr_now":                  0.0,
            "atr_avg":                  0.0,
            "ema50_slope":              0.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER B — News Sentiment Scoring
# ══════════════════════════════════════════════════════════════════════════════

# Expected event dict keys: type, result, forecast (optional), consensus (optional)
# type values: "NFP", "CPI", "FOMC", "GDP", "PMI", "GEOPOLITICAL", "other"
# result values: "beat", "miss", "inline", "hawkish", "dovish", "hike", "cut",
#                "better", "worse", "crisis", "escalation", "de-escalation"

_NEWS_RULES: list[tuple[str, str, str, float]] = [
    # (event_type, result_keyword, bias, score)
    ("NFP",         "beat",         "short",   1.5),
    ("NFP",         "miss",         "long",    1.5),
    ("CPI",         "higher",       "short",   1.0),
    ("CPI",         "hot",          "short",   1.0),
    ("CPI",         "hotter",       "short",   1.0),
    ("CPI",         "lower",        "long",    1.0),
    ("CPI",         "cooler",       "long",    1.0),
    ("FOMC",        "hike",         "short",   1.5),
    ("FOMC",        "hawkish",      "short",   1.5),
    ("FOMC",        "cut",          "long",    1.5),
    ("FOMC",        "dovish",       "long",    1.5),
    ("FOMC",        "hold",         "neutral", 0.0),
    ("GDP",         "beat",         "short",   0.5),
    ("GDP",         "miss",         "long",    0.5),
    ("PMI",         "beat",         "short",   0.3),
    ("PMI",         "miss",         "long",    0.3),
    ("GEOPOLITICAL","crisis",       "long",    1.0),
    ("GEOPOLITICAL","escalation",   "long",    0.8),
    ("GEOPOLITICAL","de-escalation","short",   0.5),
]


def score_news_sentiment(
    upcoming_events: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Score macro/geopolitical events and return a net news bias for gold.

    Parameters
    ----------
    upcoming_events : list of dicts, e.g.:
        [{"type": "NFP",  "result": "miss"},
         {"type": "FOMC", "result": "hawkish"},
         {"type": "GEOPOLITICAL", "result": "crisis"}]
        Pass None or [] if no events today.

    Returns
    -------
    news_bias      : "long" | "short" | "neutral"
    news_score     : float  (positive = bullish gold, negative = bearish)
    key_events     : list of str  (human-readable event descriptions)
    trade_with_news: bool  (True if |news_score| >= 1.0)
    """
    if not upcoming_events:
        return {
            "news_bias":       "neutral",
            "news_score":      0.0,
            "key_events":      [],
            "trade_with_news": False,
        }

    net_score  = 0.0
    key_events : list[str] = []

    for event in upcoming_events:
        etype  = str(event.get("type",   "")).upper()
        result = str(event.get("result", "")).lower()

        for rule_type, rule_result, bias, pts in _NEWS_RULES:
            if rule_type.upper() == etype and rule_result in result:
                adj = pts if bias == "long" else -pts if bias == "short" else 0.0
                net_score += adj
                sign  = "+" if adj > 0 else ""
                label = event.get("label", f"{etype} {result}")
                key_events.append(f"{label} → {'LONG' if adj>0 else 'SHORT' if adj<0 else 'NEUTRAL'} gold {sign}{pts}")
                break   # one rule per event

    if net_score > 0.3:
        bias = "long"
    elif net_score < -0.3:
        bias = "short"
    else:
        bias = "neutral"

    return {
        "news_bias":       bias,
        "news_score":      round(net_score, 2),
        "key_events":      key_events,
        "trade_with_news": abs(net_score) >= 1.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER C — Pattern Memory & Win Rate
# ══════════════════════════════════════════════════════════════════════════════

def get_pattern_win_rate(
    playbook_name: str,
    regime:        str,
    session:       str,
) -> dict[str, Any]:
    """
    Look up historical win rate for the given playbook+regime+session combo.

    Returns
    -------
    win_rate         : float  0.0–1.0  (NaN → 0.5 default)
    sample_size      : int
    avg_rr_achieved  : float
    confidence_boost : float  (+0.0 / +0.5 / +1.0)
    summary          : str
    """
    records = _load_memory()

    # Filter: same playbook AND same regime AND same session
    matches = [
        r for r in records
        if (r.get("playbook", "").lower() == playbook_name.lower()
            and r.get("regime",  "").upper() == regime.upper()
            and r.get("session", "").lower() == session.lower()
            and r.get("outcome") in ("win", "loss"))
    ]

    if not matches:
        return {
            "win_rate":        0.50,
            "sample_size":     0,
            "avg_rr_achieved": 0.0,
            "confidence_boost": 0.0,
            "summary": f"No history for {playbook_name} in {regime}/{session}",
        }

    wins       = sum(1 for r in matches if r.get("outcome") == "win")
    win_rate   = wins / len(matches)
    rr_values  = [float(r.get("rr_achieved", 0.0)) for r in matches
                  if r.get("rr_achieved") is not None]
    avg_rr     = float(np.mean(rr_values)) if rr_values else 0.0

    n = len(matches)
    if win_rate >= 0.75 and n >= 10:
        boost = 1.0
    elif win_rate >= 0.65 and n >= 10:
        boost = 0.5
    else:
        boost = 0.0

    return {
        "win_rate":         round(win_rate, 3),
        "sample_size":      n,
        "avg_rr_achieved":  round(avg_rr, 2),
        "confidence_boost": boost,
        "summary": (
            f"{playbook_name} in {regime}/{session}: "
            f"{wins}/{n} wins ({win_rate:.0%}), "
            f"avg R:R {avg_rr:.2f}  boost={boost:+.1f}"
        ),
    }


def save_trade_outcome(
    playbook:    str,
    regime:      str,
    session:     str,
    direction:   str,
    entry:       float,
    sl:          float,
    tp:          float,
    outcome:     str,         # "win" | "loss"
    rr_achieved: float | None = None,
    notes:       str          = "",
) -> None:
    """
    Append a completed trade to data/pattern_memory.json.
    outcome must be 'win' or 'loss'.
    """
    if outcome not in ("win", "loss"):
        raise ValueError(f"outcome must be 'win' or 'loss', got {outcome!r}")

    records = _load_memory()
    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)
    rr_plan = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0

    records.append({
        "date":        datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "playbook":    playbook,
        "regime":      regime,
        "session":     session,
        "direction":   direction.lower(),
        "entry":       entry,
        "sl":          sl,
        "tp":          tp,
        "rr_planned":  rr_plan,
        "rr_achieved": rr_achieved if rr_achieved is not None else (rr_plan if outcome == "win" else 0.0),
        "outcome":     outcome,
        "notes":       notes,
    })
    _save_memory(records)


# ══════════════════════════════════════════════════════════════════════════════
#  Convenience: build full market context in one call
# ══════════════════════════════════════════════════════════════════════════════

def get_full_market_context(
    df:               pd.DataFrame,
    upcoming_events:  list[dict] | None = None,
    playbook_name:    str                = "",
    session:          str                = "London",
) -> dict[str, Any]:
    """
    Run all 3 layers and return a single merged context dict.
    Used by morning_briefing.py and bot_chat.py.
    """
    regime_data = detect_gold_regime(df)
    news_data   = score_news_sentiment(upcoming_events)
    hist_data   = get_pattern_win_rate(
        playbook_name or "default",
        regime_data["regime"],
        session,
    ) if playbook_name else {
        "win_rate": 0.5, "sample_size": 0,
        "avg_rr_achieved": 0.0, "confidence_boost": 0.0, "summary": "N/A",
    }

    return {
        **regime_data,
        **{f"news_{k}": v for k, v in news_data.items()},
        "hist":               hist_data,
        "extra_confidence":   round(hist_data["confidence_boost"] + (
            abs(news_data["news_score"]) * 0.3 if news_data["trade_with_news"] else 0.0
        ), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Regime strategy configuration
# ══════════════════════════════════════════════════════════════════════════════

_REGIME_STRATEGY_CONFIGS: dict[str, dict] = {
    "TRENDING_STRONG": {
        "entry_filter":    "only_trend_following",
        "avoid_reversals": False,
        "sl_style":        "trail_with_ema50",
        "tp_style":        "let_it_run_no_tp2_cap",
        "max_signals":     2,
        "note":            "Strong trend — ride it, don't fade it",
    },
    "TRENDING_WEAK": {
        "entry_filter":    "trend_following_preferred",
        "avoid_reversals": False,
        "sl_style":        "standard_atr",
        "tp_style":        "standard_rr",
        "max_signals":     3,
        "note":            "Weak trend — standard approach",
    },
    "RANGING": {
        "entry_filter":    "sr_bounce_only",
        "avoid_reversals": False,
        "sl_style":        "tight_atr",
        "tp_style":        "early_tp1",
        "max_signals":     2,
        "note":            "Ranging — buy support sell resistance, take profits early",
    },
    "VOLATILE_EXPANDING": {
        "entry_filter":    "news_fade_only",
        "avoid_reversals": False,
        "sl_style":        "wide_atr_2x",
        "tp_style":        "tp1_only",
        "max_signals":     1,
        "note":            "High vol — news fade only, TP1 and done",
    },
    "SQUEEZE_BUILDING": {
        "entry_filter":    "breakout_only",
        "avoid_reversals": True,
        "sl_style":        "inside_squeeze",
        "tp_style":        "measured_move",
        "max_signals":     1,
        "note":            "Squeeze — wait for breakout direction, then ride it",
    },
}


def get_regime_strategy_config(regime: str) -> dict:
    """Return trading strategy configuration for the given regime."""
    return _REGIME_STRATEGY_CONFIGS.get(regime, _REGIME_STRATEGY_CONFIGS["TRENDING_WEAK"])


# ══════════════════════════════════════════════════════════════════════════════
#  Regime history tracker
# ══════════════════════════════════════════════════════════════════════════════

_REGIME_HISTORY_FILE = os.path.join(DATA_DIR, "regime_history.json")
_GST = timezone(timedelta(hours=4))  # UAE / Gulf Standard Time


def save_regime_snapshot(regime_data: dict, price: float = 0.0) -> None:
    """Append current regime reading to data/regime_history.json."""
    try:
        _ensure_data_dir()
        history: list[dict] = []
        if os.path.exists(_REGIME_HISTORY_FILE):
            try:
                with open(_REGIME_HISTORY_FILE, "r", encoding="utf-8") as fh:
                    history = json.load(fh)
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []

        snapshot = {
            "timestamp":  datetime.now(_GST).strftime("%Y-%m-%d %H:%M"),
            "regime":     regime_data.get("regime", "RANGING"),
            "atr_now":    regime_data.get("atr_now", 0.0),
            "atr_avg":    regime_data.get("atr_avg", 0.0),
            "ema50_slope":regime_data.get("ema50_slope", 0.0),
            "price":      round(price, 2),
        }
        history.append(snapshot)
        # Keep last 1000 entries to avoid unbounded growth
        history = history[-1000:]
        with open(_REGIME_HISTORY_FILE, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
    except Exception:
        pass


def get_regime_history(last_n: int = 24) -> list[dict]:
    """Load the last N regime snapshots from data/regime_history.json."""
    try:
        if not os.path.exists(_REGIME_HISTORY_FILE):
            return []
        with open(_REGIME_HISTORY_FILE, "r", encoding="utf-8") as fh:
            history = json.load(fh)
        if not isinstance(history, list):
            return []
        return history[-last_n:]
    except Exception:
        return []
