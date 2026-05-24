"""
ml_engine.py — ML Learning Layer for TradingBotV1
Learns from closed paper trades to improve signal confidence scoring.
"""
import json
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

try:
    from data_manager import get_path, save_json, load_json
    _USE_DM = True
except Exception:
    _USE_DM = False

GST             = timezone(timedelta(hours=4))
PAPER_FILE      = "data/paper_trades.json"
ML_MODEL_FILE   = "data/ml_model.json"
ML_INSIGHTS_FILE = "data/ml_insights.json"


def _get_ml_files(instrument: str = "XAUUSD") -> dict:
    """Return per-instrument file paths for ML model and paper trades."""
    safe = instrument.replace("/", "").upper()
    if _USE_DM:
        return {
            "model":    get_path(f"ml_model_{safe}.json"),
            "why":      get_path(f"ml_why_{safe}.json"),
            "paper":    get_path(f"paper_trades_{safe}.json"),
            "insights": get_path(f"ml_insights_{safe}.json"),
        }
    else:
        if safe == "XAUUSD":
            return {
                "model":    ML_MODEL_FILE,
                "why":      "ml_why_patterns.json",
                "paper":    PAPER_FILE,
                "insights": ML_INSIGHTS_FILE,
            }
        return {
            "model":    f"data/ml_model_{safe}.json",
            "why":      f"ml_why_patterns_{safe}.json",
            "paper":    f"data/paper_trades_{safe}.json",
            "insights": f"data/ml_insights_{safe}.json",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_hour(opened_at_str: str) -> int:
    """Parse hour from 'HH:MM AM/PM UAE' formatted string."""
    try:
        time_part = opened_at_str.split("|")[0].strip()
        dt = datetime.strptime(time_part, "%I:%M %p UAE")
        return dt.hour
    except Exception:
        return 12


def _parse_time_held(time_str: str) -> int:
    """Parse 'Xh Ym' or 'Ym' string into total minutes."""
    try:
        total = 0
        if "h" in time_str:
            h = int(time_str.split("h")[0].strip())
            total += h * 60
        if "m" in time_str:
            parts = time_str.replace("h", "").split("m")
            m_str = parts[0].strip()
            m = int(m_str) if m_str else 0
            total += m
        return total
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: load_training_data
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    """
    Load paper_trades.json and convert to DataFrame.
    Only uses CLOSED trades with WIN/LOSS outcome.
    Returns empty DataFrame if fewer than 5 qualifying trades.
    """
    try:
        with open(PAPER_FILE, encoding="utf-8") as f:
            trades = json.load(f)
    except Exception:
        return pd.DataFrame()

    closed = [
        t for t in trades
        if t.get("status") == "CLOSED"
        and t.get("outcome") in ("WIN", "LOSS")
    ]

    if len(closed) < 5:
        return pd.DataFrame()

    rows = []
    for t in closed:
        c = t.get("conditions_at_entry", {})
        rows.append({
            "outcome":          1 if t["outcome"] == "WIN" else 0,
            "direction":        1 if t.get("direction", "long") == "long" else 0,
            "pnl_pips":         float(t.get("pnl_pips", 0)),
            "time_held_mins":   _parse_time_held(t.get("time_held", "0m")),
            "confidence":       float(c.get("confidence", t.get("confidence", 0))),
            "session":          c.get("session", t.get("session", "Unknown")),
            "regime":           c.get("regime",  t.get("regime",  "Unknown")),
            "hour_uae":         int(c.get("hour_uae", _parse_hour(t.get("opened_at", "")))),
            "rsi":              float(c.get("rsi", 50)),
            "volume_ratio":     float(c.get("volume_ratio", 1.0)),
            "volume_class":     c.get("volume_class", "normal"),
            "checklist_passed": int(c.get("checklist_passed", 0)),
            "d1_bias":          c.get("d1_bias", "unknown"),
            "h4_bias":          c.get("h4_bias", "unknown"),
            "geo_risk":         c.get("geo_risk", "normal"),
            "fundamental":      c.get("fundamental_bias", "NEUTRAL"),
            "cot_bias":         c.get("cot_bias", "NEUTRAL"),
            "macro_bias":       c.get("macro_bias", "neutral"),
            "alligator":        c.get("alligator_state", ""),
            "macd":             c.get("macd_bias", ""),
            "stoch_k":          float(c.get("stoch_rsi_k", 50)),
            "ichimoku":         c.get("ichimoku_bias", ""),
            "supertrend":       c.get("supertrend_bias", ""),
            "adx":              float(c.get("adx", 0)),
            "adx_trending":     1 if c.get("adx_trending") else 0,
            "vwap_above":       1 if c.get("vwap_above") else 0,
            "squeeze_on":       1 if c.get("squeeze_on") else 0,
            "wyckoff":          c.get("wyckoff_phase", ""),
            "news_bias":        c.get("news_bias", "neutral"),
            "rr_ratio":         float(c.get("rr_ratio", 2.0)),
            "in_killzone":      1 if c.get("in_killzone") else 0,
            "smc_grade":        c.get("smc_grade", "D"),
            "spread":           float(c.get("spread", 0)),
            "is_counter_trend": 1 if c.get("is_counter_trend") else 0,
            "strategy":         t.get("strategy", "Unknown"),
            # ── enriched fields ────────────────────────────────────────
            "strategy_tags":        c.get("strategy_tags", []),
            "confluence_factors":   c.get("confluence_factors", []),
            "checklist_gates_passed": int(c.get("checklist_gates_passed",
                                             c.get("checklist_passed", 0))),
            "d1_trend":             c.get("d1_trend", c.get("d1_bias", "unknown")),
            "counter_trend":        1 if c.get("counter_trend", c.get("is_counter_trend", False)) else 0,
            "rsi_zone":             c.get("rsi_zone", "neutral"),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: train_pattern_model
# ─────────────────────────────────────────────────────────────────────────────

def train_pattern_model(df: pd.DataFrame) -> dict:
    """
    Learns which conditions lead to wins vs losses.
    Saves model to data/ml_model.json and returns it as a dict.
    """
    if df.empty or len(df) < 5:
        return {}

    model: dict = {}

    # Win rate by session
    session_wr: dict = {}
    for sess in df["session"].unique():
        subset = df[df["session"] == sess]
        if len(subset) >= 2:
            wr = subset["outcome"].mean() * 100
            session_wr[sess] = {
                "win_rate": round(float(wr), 1),
                "total":    int(len(subset)),
                "wins":     int(subset["outcome"].sum()),
            }
    model["session_win_rates"] = session_wr

    # Win rate by strategy
    strategy_wr: dict = {}
    for strat in df["strategy"].unique():
        subset = df[df["strategy"] == strat]
        if len(subset) >= 2:
            wr = subset["outcome"].mean() * 100
            strategy_wr[strat] = {
                "win_rate": round(float(wr), 1),
                "total":    int(len(subset)),
                "avg_pnl":  round(float(subset["pnl_pips"].mean()), 1),
            }
    model["strategy_win_rates"] = strategy_wr

    # Win rate by regime
    regime_wr: dict = {}
    for regime in df["regime"].unique():
        subset = df[df["regime"] == regime]
        if len(subset) >= 2:
            wr = subset["outcome"].mean() * 100
            regime_wr[regime] = {
                "win_rate": round(float(wr), 1),
                "total":    int(len(subset)),
            }
    model["regime_win_rates"] = regime_wr

    # Win rate by hour (UAE / GST time)
    hour_wr: dict = {}
    for hour in range(24):
        subset = df[df["hour_uae"] == hour]
        if len(subset) >= 2:
            wr = subset["outcome"].mean() * 100
            hour_wr[str(hour)] = {
                "win_rate": round(float(wr), 1),
                "total":    int(len(subset)),
            }
    model["hour_win_rates"] = hour_wr

    # Confidence threshold analysis
    conf_analysis: dict = {}
    for min_conf in [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
        subset = df[df["confidence"] >= min_conf]
        if len(subset) >= 3:
            wr = subset["outcome"].mean() * 100
            conf_analysis[str(min_conf)] = {
                "win_rate": round(float(wr), 1),
                "total":    int(len(subset)),
            }
    model["confidence_thresholds"] = conf_analysis

    # Best/worst session+regime combinations
    df = df.copy()
    df["combo"] = df["session"] + "_" + df["regime"]
    combo_wr: dict = {}
    for combo in df["combo"].unique():
        subset = df[df["combo"] == combo]
        if len(subset) >= 2:
            wr = subset["outcome"].mean() * 100
            combo_wr[combo] = {
                "win_rate": round(float(wr), 1),
                "total":    int(len(subset)),
            }
    model["combo_win_rates"] = combo_wr

    # Overall stats
    wins_df   = df[df["outcome"] == 1]
    losses_df = df[df["outcome"] == 0]
    model["overall"] = {
        "total_trades":  int(len(df)),
        "win_rate":      round(float(df["outcome"].mean() * 100), 1),
        "avg_win_pips":  round(float(wins_df["pnl_pips"].mean()), 1)   if len(wins_df)   > 0 else 0.0,
        "avg_loss_pips": round(float(losses_df["pnl_pips"].mean()), 1) if len(losses_df) > 0 else 0.0,
        "trained_at":    datetime.now(GST).isoformat(),
    }

    # Failure and success pattern analysis (WHY trades win/lose)
    losses_df2 = df[df["outcome"] == 0]
    wins_df2   = df[df["outcome"] == 1]
    failure_patterns: list = []
    success_patterns: list = []

    def _check_fail(mask, name: str, rec: str) -> None:
        sub      = df[mask]
        loss_sub = losses_df2[mask]
        if len(sub) >= 2:
            rate = len(loss_sub) / len(sub) * 100
            if rate > 60:
                failure_patterns.append({
                    "condition":      name,
                    "failure_rate":   round(float(rate), 1),
                    "count":          int(len(sub)),
                    "recommendation": rec,
                })

    def _check_win(mask, name: str, rec: str) -> None:
        sub     = df[mask]
        win_sub = wins_df2[mask]
        if len(sub) >= 2:
            rate = len(win_sub) / len(sub) * 100
            if rate > 60:
                success_patterns.append({
                    "condition":      name,
                    "win_rate":       round(float(rate), 1),
                    "count":          int(len(sub)),
                    "recommendation": rec,
                })

    if len(df) >= 5 and "volume_ratio" in df.columns:
        _check_fail(df["volume_ratio"] < 0.5,
            "volume_ratio < 0.5 (very low volume)",
            "Avoid trades with very low volume")
        _check_fail(df["checklist_passed"] == 0,
            "checklist_passed = 0 (failed all checks)",
            "Never trade 0/5 checklist signals")
        _check_fail(df["is_counter_trend"] == 1,
            "Counter-trend vs D1",
            "Avoid counter-trend trades")
        _check_fail(df["confidence"] < 5.0,
            "confidence < 5.0",
            "Minimum confidence 5.0 before trading")
        _check_win(df["adx"] > 25,
            "ADX > 25 (strong trend)",
            "Prioritize trades with ADX > 25")
        _check_win(df["in_killzone"] == 1,
            "ICT Kill Zone active",
            "Trade during ICT kill zones")
        _check_win(df["confidence"] >= 7.0,
            "Confidence >= 7.0",
            "High confidence signals are reliable")
        _check_win(df["is_counter_trend"] == 0,
            "Trend-aligned (not counter-trend)",
            "Trend-following signals outperform")

        # ── SMC grade patterns ────────────────────────────────────────
        if "smc_grade" in df.columns:
            _check_fail(df["smc_grade"] == "D",
                "D SMC grade",
                "Avoid D-grade SMC setups")
            _check_win(df["smc_grade"].isin(["A", "B"]),
                "A or B SMC grade",
                "Prioritise A/B SMC grade signals")

        # ── Regime patterns ─────────────────────────────────────────
        if "regime" in df.columns:
            _ranging_mask = df["regime"].str.lower().str.contains("rang", na=False)
            _trending_mask = df["regime"].str.lower().str.contains("trend", na=False)
            if _ranging_mask.any():
                _check_fail(_ranging_mask,
                    "Ranging/range market regime",
                    "Reduce position size or avoid in ranging markets")
            if _trending_mask.any():
                _check_win(_trending_mask,
                    "Trending market regime",
                    "Increase confidence in trending regimes")

        # ── Strategy-tag patterns ────────────────────────────────────
        if "strategy_tags" in df.columns:
            _no_tags_mask = df["strategy_tags"].apply(
                lambda x: len(x) == 0 if isinstance(x, list) else True)
            _multi_tags_mask = df["strategy_tags"].apply(
                lambda x: len(x) >= 3 if isinstance(x, list) else False)
            if _no_tags_mask.any():
                _check_fail(_no_tags_mask,
                    "No strategy votes (empty strategy_tags)",
                    "Require at least 1 strategy to vote before trading")
            if _multi_tags_mask.any():
                _check_win(_multi_tags_mask,
                    "3+ strategy confluence (multi_strategy_confluence)",
                    "Strong edge when 3 or more strategies agree")

    model["failure_patterns"] = failure_patterns
    model["success_patterns"] = success_patterns

    os.makedirs("data", exist_ok=True)
    with open(ML_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: get_ml_confidence_adjustment
# ─────────────────────────────────────────────────────────────────────────────

def get_ml_confidence_adjustment(
    session:    str,
    regime:     str,
    strategy:   str,
    confidence: float,
    hour_uae:   int,
) -> dict:
    """
    Load trained model and calculate confidence adjustment based on
    historical win-rate patterns for this session/regime/strategy/hour.
    Returns dict with 'adjustment' (float, clamped ±2.0) and reasons.
    """
    try:
        with open(ML_MODEL_FILE, encoding="utf-8") as f:
            model = json.load(f)
    except Exception:
        return {
            "adjustment": 0.0,
            "reason":     "No ML model yet",
            "available":  False,
        }

    overall_wr   = model.get("overall", {}).get("win_rate", 50.0)
    adjustments: list[float] = []
    reasons:     list[str]   = []

    # Session adjustment
    sess_data = model.get("session_win_rates", {}).get(session, {})
    if sess_data and sess_data.get("total", 0) >= 3:
        sess_wr = sess_data["win_rate"]
        if sess_wr > overall_wr + 15:
            adjustments.append(+0.5)
            reasons.append(f"Session {session} WR {sess_wr:.0f}% (strong)")
        elif sess_wr < overall_wr - 15:
            adjustments.append(-0.5)
            reasons.append(f"Session {session} WR {sess_wr:.0f}% (weak)")

    # Strategy adjustment
    strat_data = model.get("strategy_win_rates", {}).get(strategy, {})
    if strat_data and strat_data.get("total", 0) >= 3:
        strat_wr = strat_data["win_rate"]
        if strat_wr > 65:
            adjustments.append(+0.8)
            reasons.append(f"Strategy WR {strat_wr:.0f}% historically")
        elif strat_wr > 55:
            adjustments.append(+0.3)
            reasons.append(f"Strategy WR {strat_wr:.0f}%")
        elif strat_wr < 35:
            adjustments.append(-0.8)
            reasons.append(f"Strategy WR {strat_wr:.0f}% (poor)")
        elif strat_wr < 45:
            adjustments.append(-0.3)
            reasons.append(f"Strategy WR {strat_wr:.0f}%")

    # Regime adjustment
    regime_data = model.get("regime_win_rates", {}).get(regime, {})
    if regime_data and regime_data.get("total", 0) >= 3:
        regime_wr = regime_data["win_rate"]
        if regime_wr > overall_wr + 10:
            adjustments.append(+0.4)
            reasons.append(f"Regime {regime} WR {regime_wr:.0f}%")
        elif regime_wr < overall_wr - 10:
            adjustments.append(-0.4)
            reasons.append(f"Regime {regime} WR {regime_wr:.0f}%")

    # Hour adjustment
    hour_data = model.get("hour_win_rates", {}).get(str(hour_uae), {})
    if hour_data and hour_data.get("total", 0) >= 3:
        hour_wr = hour_data["win_rate"]
        if hour_wr > 70:
            adjustments.append(+0.3)
            reasons.append(f"Hour {hour_uae}:00 UAE WR {hour_wr:.0f}%")
        elif hour_wr < 35:
            adjustments.append(-0.3)
            reasons.append(f"Hour {hour_uae}:00 UAE WR {hour_wr:.0f}%")

    total_adj = float(sum(adjustments))
    total_adj = max(-2.0, min(2.0, total_adj))

    return {
        "adjustment":  round(total_adj, 2),
        "adjustments": adjustments,
        "reasons":     reasons,
        "available":   True,
        "model_size":  model.get("overall", {}).get("total_trades", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 4: generate_ml_insights
# ─────────────────────────────────────────────────────────────────────────────

def generate_ml_insights() -> dict:
    """
    Generate human-readable insights from the trained model.
    Saves to data/ml_insights.json and returns as dict.
    """
    try:
        with open(ML_MODEL_FILE, encoding="utf-8") as f:
            model = json.load(f)
    except Exception:
        return {"available": False}

    insights: list[str] = []
    overall_wr = model.get("overall", {}).get("win_rate", 50.0)

    # Best / worst session
    sess_wrs = model.get("session_win_rates", {})
    if sess_wrs:
        best_sess  = max(
            sess_wrs.items(),
            key=lambda x: x[1]["win_rate"] if x[1].get("total", 0) >= 3 else 0,
        )
        worst_sess = min(
            sess_wrs.items(),
            key=lambda x: x[1]["win_rate"] if x[1].get("total", 0) >= 3 else 100,
        )
        insights.append(
            f"Best session: {best_sess[0]} ({best_sess[1]['win_rate']}% WR)"
        )
        insights.append(
            f"Avoid session: {worst_sess[0]} ({worst_sess[1]['win_rate']}% WR)"
        )

    # Best strategy
    strat_wrs = model.get("strategy_win_rates", {})
    if strat_wrs:
        best_strat = max(
            strat_wrs.items(),
            key=lambda x: x[1]["win_rate"] if x[1].get("total", 0) >= 3 else 0,
        )
        insights.append(
            f"Most profitable strategy: {best_strat[0]} "
            f"({best_strat[1]['win_rate']}% WR)"
        )

    # Optimal confidence threshold
    conf_data = model.get("confidence_thresholds", {})
    if conf_data:
        best_conf = max(
            conf_data.items(),
            key=lambda x: x[1]["win_rate"] if x[1].get("total", 0) >= 3 else 0,
        )
        insights.append(
            f"Optimal min confidence: {best_conf[0]} "
            f"({best_conf[1]['win_rate']}% WR on {best_conf[1]['total']} trades)"
        )

    # Best hour range
    hour_wrs = model.get("hour_win_rates", {})
    if hour_wrs:
        best_hour = max(
            hour_wrs.items(),
            key=lambda x: x[1]["win_rate"] if x[1].get("total", 0) >= 3 else 0,
        )
        if best_hour[1].get("total", 0) >= 3:
            insights.append(
                f"Best hour: {best_hour[0]}:00 UAE "
                f"({best_hour[1]['win_rate']}% WR)"
            )

    # Best combo
    combo_wrs = model.get("combo_win_rates", {})
    if combo_wrs:
        best_combo = max(
            combo_wrs.items(),
            key=lambda x: x[1]["win_rate"] if x[1].get("total", 0) >= 3 else 0,
        )
        if best_combo[1].get("total", 0) >= 3:
            insights.append(
                f"Best combo: {best_combo[0]} "
                f"({best_combo[1]['win_rate']}% WR)"
            )

    # Failure and success pattern insights
    for fp in model.get("failure_patterns", [])[:3]:
        insights.append(
            f"\u26a0 AVOID: {fp['condition']} \u2192 "
            f"{fp['failure_rate']:.0f}% failure "
            f"({fp['count']} trades) \u2014 {fp['recommendation']}"
        )
    for sp in model.get("success_patterns", [])[:3]:
        insights.append(
            f"\u2705 TAKE: {sp['condition']} \u2192 "
            f"{sp['win_rate']:.0f}% WR "
            f"({sp['count']} trades) \u2014 {sp['recommendation']}"
        )

    result = {
        "available":    True,
        "overall_wr":   overall_wr,
        "total_trades": model.get("overall", {}).get("total_trades", 0),
        "insights":     insights,
        "trained_at":   model.get("overall", {}).get("trained_at", "Never"),
    }

    os.makedirs("data", exist_ok=True)
    with open(ML_INSIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 5: run_ml_training
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# CLASS: MLEngine  (predict_setup_quality + enhance_signal)
# ─────────────────────────────────────────────────────────────────────────────

class MLEngine:
    """Stateless ML engine wrapper for pre-trade quality scoring."""

    def __init__(self, instrument: str = "XAUUSD"):
        self.instrument = instrument
        self.files = _get_ml_files(instrument)

    def predict_setup_quality(self, conditions: dict) -> dict:
        """
        Takes a conditions dict (same structure as conditions_at_entry in
        paper_trades.json) and returns an ML-based quality assessment of the
        setup BEFORE taking the trade.
        """
        try:
            # Load why patterns for context
            why: dict = {}
            _why_file = self.files["model"]   # fall back to main model
            for _f in (self.files["why"], _why_file):
                if os.path.exists(_f):
                    try:
                        with open(_f, "r", encoding="utf-8") as _fh:
                            why = json.load(_fh)
                        break
                    except Exception:
                        pass

            if not why:
                return {
                    "ml_score": 50,
                    "ml_grade": "?",
                    "verdict":  "Not enough trade history yet",
                    "red_flags":   [],
                    "green_flags": [],
                }

            failure_patterns = why.get("failure_patterns", {})
            success_patterns = why.get("success_patterns", {})

            red_flags:   list = []
            green_flags: list = []
            score = 50  # baseline

            adx        = float(conditions.get("adx", 0) or 0)
            spread     = float(conditions.get("spread_pips", 0) or 0)
            in_kz      = bool(conditions.get("in_killzone", False))
            checklist  = int(conditions.get("checklist_score", 0) or 0)
            counter    = bool(conditions.get("counter_trend", False))
            confluence = float(conditions.get("confluence_score", 0) or 0)
            rsi_zone   = str(conditions.get("rsi_zone", "neutral"))
            direction  = str(conditions.get("direction", "LONG")).upper()

            # ── Failure checks ────────────────────────────────────────────
            if adx < 20:
                rate = failure_patterns.get("low_adx", {}).get("failure_rate", 0)
                red_flags.append(
                    f"ADX {adx:.1f} < 20 — historically {rate:.0f}% loss rate")
                score -= 15
            if spread > 3:
                rate = failure_patterns.get("high_spread", {}).get("failure_rate", 0)
                red_flags.append(
                    f"Spread {spread:.1f} pips > 3 — {rate:.0f}% loss rate")
                score -= 10
            if not in_kz:
                rate = failure_patterns.get("outside_killzone", {}).get("failure_rate", 0)
                red_flags.append(
                    f"Outside Kill Zone — {rate:.0f}% loss rate historically")
                score -= 15
            if counter:
                rate = failure_patterns.get("counter_trend_loss", {}).get("failure_rate", 0)
                red_flags.append(
                    f"Counter-trend trade — {rate:.0f}% loss rate historically")
                score -= 10
            if checklist <= 2:
                rate = failure_patterns.get("weak_checklist", {}).get("failure_rate", 0)
                red_flags.append(
                    f"Checklist only {checklist}/5 — {rate:.0f}% loss rate")
                score -= 10
            if confluence < 60:
                rate = failure_patterns.get("low_confluence", {}).get("failure_rate", 0)
                red_flags.append(
                    f"Confluence {confluence:.0f}% < 60 — {rate:.0f}% loss rate")
                score -= 10
            if rsi_zone == "overbought" and direction == "LONG":
                rate = failure_patterns.get("overbought_long", {}).get("failure_rate", 0)
                red_flags.append(
                    f"RSI overbought on LONG — {rate:.0f}% loss rate")
                score -= 5
            if rsi_zone == "oversold" and direction == "SHORT":
                rate = failure_patterns.get("oversold_short", {}).get("failure_rate", 0)
                red_flags.append(
                    f"RSI oversold on SHORT — {rate:.0f}% loss rate")
                score -= 5

            # ── Success checks ────────────────────────────────────────────
            if adx > 25:
                rate = success_patterns.get("strong_adx", {}).get("win_rate", 0)
                green_flags.append(
                    f"ADX {adx:.1f} > 25 — {rate:.0f}% win rate condition")
                score += 10
            if in_kz:
                rate = success_patterns.get("in_killzone_win", {}).get("win_rate", 0)
                green_flags.append(
                    f"Inside Kill Zone — {rate:.0f}% win rate condition")
                score += 15
            if checklist >= 4:
                rate = success_patterns.get("strong_checklist", {}).get("win_rate", 0)
                green_flags.append(
                    f"Checklist {checklist}/5 — {rate:.0f}% win rate condition")
                score += 10
            if not counter:
                rate = success_patterns.get("with_d1_trend", {}).get("win_rate", 0)
                green_flags.append(
                    f"Trend-aligned trade — {rate:.0f}% win rate condition")
                score += 10
            if confluence >= 75:
                rate = success_patterns.get("high_confluence", {}).get("win_rate", 0)
                green_flags.append(
                    f"Confluence {confluence:.0f}% >= 75 — {rate:.0f}% win rate")
                score += 10

            score = max(0, min(100, score))

            if score >= 75:
                grade   = "A"
                verdict = "STRONG SETUP — ML approves"
            elif score >= 60:
                grade   = "B"
                verdict = "GOOD SETUP — acceptable risk"
            elif score >= 45:
                grade   = "C"
                verdict = "MARGINAL — consider waiting"
            else:
                grade   = "D"
                verdict = "WEAK SETUP — ML advises skip"

            return {
                "ml_score":    score,
                "ml_grade":    grade,
                "verdict":     verdict,
                "red_flags":   red_flags,
                "green_flags": green_flags,
                "total_red":   len(red_flags),
                "total_green": len(green_flags),
            }

        except Exception as e:
            return {
                "ml_score":  50,
                "ml_grade":  "?",
                "verdict":   "ML assessment unavailable",
                "red_flags": [],
                "green_flags": [],
                "error":     str(e),
            }

    def enhance_signal(self, signal: dict) -> dict:
        """
        Takes a raw signal dict from confluence_engine.py and returns it
        enhanced with ML quality score, grade, verdict, red flags, and green
        flags.  Blends ML confidence with technical confluence score.
        """
        try:
            raw  = signal.get("raw_checks", {})
            inds = raw.get("indicators", {}) if isinstance(raw, dict) else {}

            conditions = {
                "adx":             signal.get("adx", raw.get("adx", 0) if isinstance(raw, dict) else 0),
                "adx_trending":    (float(signal.get("adx", 0) or 0)) > 25,
                "spread_pips":     signal.get("spread_usd", 0),
                "in_killzone":     inds.get("killzones", {}).get("in_killzone", False) if isinstance(inds, dict) else False,
                "killzone_name":   signal.get("session", "None"),
                "checklist_score": (signal.get("checklist_results") or {}).get("checks_passed", 0),
                "confluence_score": signal.get("confidence", 0),
                "counter_trend":   signal.get("counter_trend", False),
                "rsi":             signal.get("rsi", 50),
                "rsi_zone": (
                    "overbought" if (float(signal.get("rsi", 50) or 50)) > 70
                    else "oversold" if (float(signal.get("rsi", 50) or 50)) < 30
                    else "neutral"
                ),
                "direction":       signal.get("direction", "LONG"),
                "session":         signal.get("session", "Unknown"),
            }

            ml_result = self.predict_setup_quality(conditions)

            tech_conf         = float(signal.get("confidence", 50) or 50)
            ml_score          = float(ml_result.get("ml_score", 50))
            blended_confidence = round((tech_conf * 0.6) + (ml_score * 0.4), 1)

            signal["ml_score"]           = ml_score
            signal["ml_grade"]           = ml_result.get("ml_grade", "?")
            signal["ml_verdict"]         = ml_result.get("verdict", "")
            signal["ml_red_flags"]       = ml_result.get("red_flags", [])
            signal["ml_green_flags"]     = ml_result.get("green_flags", [])
            signal["blended_confidence"] = blended_confidence
            signal["ml_enhanced"]        = True
            return signal

        except Exception as e:
            signal["ml_enhanced"] = False
            signal["ml_error"]    = str(e)
            return signal


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 5: run_ml_training
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_training() -> str:
    """
    Main entry point — trains model on current paper trades and returns
    a human-readable summary string.
    """
    df = load_training_data()

    if df.empty:
        return (
            "Not enough data to train ML model yet.\n"
            "Need at least 5 closed paper trades.\n"
            "Paper trade more signals to build training data."
        )

    model    = train_pattern_model(df)
    insights = generate_ml_insights()

    overall = model.get("overall", {})
    insight_lines = "\n".join(
        f"  • {i}" for i in insights.get("insights", [])
    )
    return (
        f"ML MODEL TRAINED\n"
        f"Training data: {overall.get('total_trades', 0)} trades\n"
        f"Overall WR: {overall.get('win_rate', 0)}%\n"
        f"Avg win: +{overall.get('avg_win_pips', 0)} pips\n"
        f"Avg loss: {overall.get('avg_loss_pips', 0)} pips\n\n"
        f"Insights:\n"
        f"{insight_lines}"
    )
