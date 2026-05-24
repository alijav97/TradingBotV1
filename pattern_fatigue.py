"""
pattern_fatigue.py — Sequential Pattern Tracker & Strategy Fatigue Detector
============================================================================
Part 1 : SequenceTracker  — get_strategy_sequence, find_sequence_patterns, predict_next_outcome
Part 2 : check_strategy_fatigue  — 4 fatigue signals
Part 3 : detect_regime_shift     — ATR / EMA / RSI / session checks
Part 4 : backtest_sequence       — historical OHLCV scan for exact sequences
Part 5 : analyze_failed_trade    — 6 failure reasons + auto-rule updates
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_FILE       = os.path.join(BASE_DIR, "data", "trade_log.json")
JOURNAL_FILE         = os.path.join(BASE_DIR, "data", "trade_journal.json")
FAILED_ANALYSIS_FILE = os.path.join(BASE_DIR, "data", "logs", "failed_trade_analysis.json")
RULES_FILE           = os.path.join(BASE_DIR, "data", "rules.json")
GST                  = timezone(timedelta(hours=4))

# ── Debug logger (graceful fallback) ─────────────────────────────────────────
try:
    from debug_logger import log_info, log_error
except ImportError:
    def log_info(m: str) -> None: pass
    def log_error(**kw) -> None: pass


# ─────────────────────────────────────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_trade_log() -> list[dict]:
    if not os.path.exists(TRADE_LOG_FILE):
        return []
    try:
        with open(TRADE_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _load_journal() -> list[dict]:
    if not os.path.exists(JOURNAL_FILE):
        return []
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _load_rules() -> list[dict]:
    if not os.path.exists(RULES_FILE):
        return []
    try:
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_rules(rules: list[dict]) -> None:
    os.makedirs(os.path.dirname(RULES_FILE), exist_ok=True)
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)


def _save_failed_analysis(record: dict) -> None:
    os.makedirs(os.path.dirname(FAILED_ANALYSIS_FILE), exist_ok=True)
    records: list[dict] = []
    if os.path.exists(FAILED_ANALYSIS_FILE):
        try:
            with open(FAILED_ANALYSIS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                records = []
        except Exception:
            records = []
    records.append(record)
    with open(FAILED_ANALYSIS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)


def _parse_dt(s: Any) -> datetime | None:
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s)[:19], fmt)
        except ValueError:
            continue
    return None


def _current_session() -> str:
    now_utc = datetime.now(timezone.utc)
    h = now_utc.hour
    if 22 <= h or h < 7:
        return "Asian"
    if 7 <= h < 12:
        return "London"
    if 12 <= h < 17:
        return "New York"
    return "London"


# ─────────────────────────────────────────────────────────────────────────────
#  PART 1 — SequenceTracker
# ─────────────────────────────────────────────────────────────────────────────

class SequenceTracker:
    """Track win/loss sequences per strategy and predict next outcome risk."""

    # ── get_strategy_sequence ────────────────────────────────────────────────
    @staticmethod
    def get_strategy_sequence(playbook_name: str, last_n: int = 10) -> dict:
        """
        Load trade_log.json + trade_journal.json, filter by playbook_name,
        return the last `last_n` results as a W/L sequence.

        Returns:
            sequence      : list[str]  e.g. ["W","W","L","W"]
            current_streak: int
            streak_type   : "WIN" | "LOSS" | "NONE"
            total_trades  : int
            win_rate      : float  (0-1)
        """
        # ── Collect from trade_log ────────────────────────────────────────
        log_trades = _load_trade_log()
        log_results = []
        for t in log_trades:
            if t.get("status", "") == "closed":
                pn = (t.get("pattern_name") or "").strip().lower()
                if pn == playbook_name.strip().lower():
                    dt = _parse_dt(t.get("datetime", ""))
                    res = (t.get("result") or "").upper()
                    if res in ("WIN", "LOSS"):
                        log_results.append((dt, "W" if res == "WIN" else "L"))

        # ── Collect from journal ─────────────────────────────────────────
        journal = _load_journal()
        jrn_results = []
        for t in journal:
            pn = (t.get("pattern_name") or "").strip().lower()
            if pn == playbook_name.strip().lower():
                dt = _parse_dt(t.get("closed_at") or t.get("close_time", ""))
                out = (t.get("outcome") or "").lower()
                if out in ("win", "loss"):
                    jrn_results.append((dt, "W" if out == "win" else "L"))

        # ── Merge, deduplicate by datetime proximity, sort ───────────────
        combined = log_results + jrn_results
        combined_sorted = sorted(
            [r for r in combined if r[0] is not None],
            key=lambda x: x[0]
        )
        # Also include entries without datetimes (append at end)
        no_dt = [r for r in combined if r[0] is None]
        combined_sorted += no_dt

        sequence = [r[1] for r in combined_sorted][-last_n:]

        # ── Compute stats ────────────────────────────────────────────────
        total = len(sequence)
        win_rate = sequence.count("W") / total if total > 0 else 0.0

        current_streak = 0
        streak_type = "NONE"
        if sequence:
            last = sequence[-1]
            streak_type = "WIN" if last == "W" else "LOSS"
            for r in reversed(sequence):
                if r == last:
                    current_streak += 1
                else:
                    break

        return {
            "sequence": sequence,
            "current_streak": current_streak,
            "streak_type": streak_type,
            "total_trades": total,
            "win_rate": win_rate,
        }

    # ── find_sequence_patterns ───────────────────────────────────────────────
    @staticmethod
    def find_sequence_patterns(sequence: list[str]) -> list[dict]:
        """
        Scan for common patterns in a W/L sequence.

        Detected patterns:
          WWWWL  — 4-win streak ending in loss
          WWLL   — alternating 2-pair
          WLWL   — strict alternation
          WWWWWL — 5-win streak then loss
          LLW    — double-loss recovery
          WWLWWL — recurring 2-win / 1-loss cycle

        Returns list of pattern dicts with occurrence + P(L after N wins).
        """
        n = len(sequence)
        results = []

        patterns_to_check = [
            ("WWWWWL", 6),
            ("WWWWL",  5),
            ("WWLL",   4),
            ("WLWL",   4),
            ("LLW",    3),
            ("WWLWWL", 6),
        ]

        for pat_str, pat_len in patterns_to_check:
            pat = list(pat_str)
            count = 0
            last_pos = None
            for i in range(n - pat_len + 1):
                if sequence[i:i + pat_len] == pat:
                    count += 1
                    last_pos = i

            if count == 0:
                continue

            # Compute P(L after last W run)
            nw = pat_str.rstrip("L").count("W")
            follow_loss = 0
            follow_total = 0
            for i in range(n - nw):
                if sequence[i:i + nw] == ["W"] * nw:
                    follow_total += 1
                    if i + nw < n and sequence[i + nw] == "L":
                        follow_loss += 1
            p_loss = follow_loss / follow_total if follow_total > 0 else 0.0

            note = ""
            if p_loss > 0.70:
                note = f"⚠ HIGH — {p_loss:.0%} loss after {nw} wins historically"
            elif p_loss > 0.50:
                note = f"Moderate — {p_loss:.0%} loss probability"
            else:
                note = f"Low risk — {p_loss:.0%} loss probability"

            results.append({
                "pattern":                 pat_str,
                "occurrences":             count,
                "probability_of_L_after_N_wins": round(p_loss, 3),
                "last_occurred_at_index":  last_pos,
                "note":                    note,
            })

        return sorted(results, key=lambda x: x["probability_of_L_after_N_wins"], reverse=True)

    # ── predict_next_outcome ─────────────────────────────────────────────────
    @staticmethod
    def predict_next_outcome(playbook_name: str) -> dict:
        """
        Combine streak analysis + sequence patterns to predict next outcome.

        Returns:
            current_streak    : int
            streak_type       : str
            warning_level     : "low" | "medium" | "high" | "critical"
            predicted_outcome : "WIN" | "LOSS" | "UNCERTAIN"
            confidence        : float  (0-1)
            historical_pattern: str
            recommendation    : "skip" | "reduce_size" | "trade_normal"
            reason            : str
        """
        seq_data  = SequenceTracker.get_strategy_sequence(playbook_name)
        sequence  = seq_data["sequence"]
        streak    = seq_data["current_streak"]
        s_type    = seq_data["streak_type"]
        win_rate  = seq_data["win_rate"]
        patterns  = SequenceTracker.find_sequence_patterns(sequence)

        # ── Default values ────────────────────────────────────────────────
        warning_level     = "low"
        predicted_outcome = "UNCERTAIN"
        confidence        = 0.3
        hist_pattern      = "Not enough data"
        recommendation    = "trade_normal"
        reason            = "No notable pattern found"

        if s_type == "WIN":
            if streak >= 5:
                warning_level     = "critical"
                predicted_outcome = "LOSS"
                confidence        = 0.80
                recommendation    = "skip"
                reason            = f"5+ consecutive wins — historically high reversal risk"
            elif streak >= 4:
                warning_level     = "high"
                predicted_outcome = "LOSS"
                confidence        = 0.70
                recommendation    = "reduce_size"
                reason            = f"4-win streak — strategy fatigue likely"
            elif streak >= 3:
                warning_level     = "medium"
                predicted_outcome = "UNCERTAIN"
                confidence        = 0.55
                recommendation    = "reduce_size"
                reason            = f"3-win streak — monitor closely"

        elif s_type == "LOSS":
            if streak >= 3:
                warning_level     = "high"
                predicted_outcome = "LOSS"
                confidence        = 0.65
                recommendation    = "skip"
                reason            = f"3+ consecutive losses — regime or strategy mismatch"
            elif streak >= 2:
                warning_level     = "medium"
                predicted_outcome = "UNCERTAIN"
                confidence        = 0.50
                recommendation    = "reduce_size"
                reason            = f"2-loss streak — review setup quality"

        # ── Override with pattern analysis if stronger signal ─────────────
        if patterns:
            top = patterns[0]
            p_l = top["probability_of_L_after_N_wins"]
            if p_l > 0.70 and s_type == "WIN":
                hist_pattern = f"{top['pattern']} found {top['occurrences']}× — {p_l:.0%} loss follows"
                if warning_level not in ("critical",):
                    warning_level = "high"
                    recommendation = "reduce_size"
                    confidence = max(confidence, p_l)

        # ── Insufficient data fallback ────────────────────────────────────
        if len(sequence) < 4:
            warning_level     = "low"
            predicted_outcome = "UNCERTAIN"
            confidence        = 0.2
            recommendation    = "trade_normal"
            reason            = "Not enough trade history for this strategy"

        return {
            "current_streak":     streak,
            "streak_type":        s_type,
            "warning_level":      warning_level,
            "predicted_outcome":  predicted_outcome,
            "confidence":         round(confidence, 2),
            "historical_pattern": hist_pattern,
            "recommendation":     recommendation,
            "reason":             reason,
            "sequence":           sequence,
            "win_rate":           round(win_rate, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  PART 2 — check_strategy_fatigue
# ─────────────────────────────────────────────────────────────────────────────

def check_strategy_fatigue(
    playbook_name: str,
    df: pd.DataFrame,
    direction: str,
) -> dict:
    """
    4-signal fatigue detector.

    Signal 1: Win streak length  (>=3 moderate, >=4 high, >=5 critical)
    Signal 2: Diminishing RR     (last 5 closed pips trending downward)
    Signal 3: Time clustering    (4 wins within a 3-day window)
    Signal 4: Level exhaustion   (same price level hit 4+ times in log)

    Returns:
        fatigue_level        : "none" | "moderate" | "high" | "critical"
        recommendation       : str
        message              : str
        signals_triggered    : list[str]
        confidence_adjustment: float  (negative = reduce conf)
    """
    seq_data     = SequenceTracker.get_strategy_sequence(playbook_name, last_n=20)
    sequence     = seq_data["sequence"]
    streak       = seq_data["current_streak"]
    streak_type  = seq_data["streak_type"]

    signals_triggered: list[str] = []
    fatigue_score = 0

    # ── Signal 1: Win streak ─────────────────────────────────────────────
    if streak_type == "WIN":
        if streak >= 5:
            signals_triggered.append(f"WIN_STREAK_CRITICAL ({streak} wins)")
            fatigue_score += 3
        elif streak >= 4:
            signals_triggered.append(f"WIN_STREAK_HIGH ({streak} wins)")
            fatigue_score += 2
        elif streak >= 3:
            signals_triggered.append(f"WIN_STREAK_MODERATE ({streak} wins)")
            fatigue_score += 1

    # ── Signal 2: Diminishing returns (RR/pips shrinking) ────────────────
    log_trades = _load_trade_log()
    closed = [
        t for t in log_trades
        if t.get("status") == "closed"
        and (t.get("pattern_name") or "").strip().lower() == playbook_name.strip().lower()
        and t.get("result", "").upper() == "WIN"
    ]
    if len(closed) >= 5:
        recent_pips = [float(t.get("pips", 0)) for t in closed[-5:]]
        if len(recent_pips) == 5:
            diffs = [recent_pips[i+1] - recent_pips[i] for i in range(4)]
            if all(d < 0 for d in diffs):
                signals_triggered.append("DIMINISHING_RR (5 consecutive declining pip wins)")
                fatigue_score += 2

    # ── Signal 3: Time clustering (4 wins in 3-day window) ────────────────
    wins_with_dt = []
    for t in log_trades:
        if (t.get("status") == "closed"
                and (t.get("pattern_name") or "").strip().lower() == playbook_name.strip().lower()
                and t.get("result", "").upper() == "WIN"):
            dt = _parse_dt(t.get("datetime"))
            if dt:
                wins_with_dt.append(dt)

    wins_with_dt.sort()
    if len(wins_with_dt) >= 4:
        for i in range(len(wins_with_dt) - 3):
            window = wins_with_dt[i:i+4]
            span_days = (window[-1] - window[0]).total_seconds() / 86400
            if span_days <= 3.0:
                signals_triggered.append(f"TIME_CLUSTERING (4 wins in {span_days:.1f} days)")
                fatigue_score += 2
                break

    # ── Signal 4: Level exhaustion (same entry zone 4+ times) ─────────────
    all_entries = [
        float(t.get("entry", 0)) for t in log_trades
        if (t.get("pattern_name") or "").strip().lower() == playbook_name.strip().lower()
        and float(t.get("entry", 0)) > 0
    ]
    if all_entries:
        # Cluster entries within $5 zones
        level_counts: dict[int, int] = {}
        for e in all_entries:
            zone = int(e // 5) * 5
            level_counts[zone] = level_counts.get(zone, 0) + 1
        max_hits = max(level_counts.values())
        if max_hits >= 4:
            dominant_zone = max(level_counts, key=level_counts.get)
            signals_triggered.append(f"LEVEL_EXHAUSTION (${dominant_zone}–${dominant_zone+5} tested {max_hits}×)")
            fatigue_score += 2

    # ── Derive fatigue level ───────────────────────────────────────────────
    if fatigue_score >= 5:
        fatigue_level = "critical"
        recommendation = "SKIP this trade — strategy fatigue is critical"
        conf_adj      = -3.0
    elif fatigue_score >= 3:
        fatigue_level = "high"
        recommendation = "Trade at HALF position size only"
        conf_adj      = -1.5
    elif fatigue_score >= 1:
        fatigue_level = "moderate"
        recommendation = "Reduce position size slightly — monitor closely"
        conf_adj      = -0.5
    else:
        fatigue_level = "none"
        recommendation = "Strategy is fresh — normal position size"
        conf_adj      = 0.0

    # ── Build human message ───────────────────────────────────────────────
    if signals_triggered:
        sigs_str = " | ".join(signals_triggered)
        message = f"Fatigue signals: {sigs_str}"
    else:
        message = "No fatigue signals detected"

    return {
        "fatigue_level":         fatigue_level,
        "recommendation":        recommendation,
        "message":               message,
        "signals_triggered":     signals_triggered,
        "confidence_adjustment": conf_adj,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PART 3 — detect_regime_shift
# ─────────────────────────────────────────────────────────────────────────────

def detect_regime_shift(df: pd.DataFrame, last_trade_time: Any = None) -> dict:
    """
    Detect if market regime has changed since last winning trade.

    Check 1: ATR change   (atr_now / atr_then > 1.5 → expanded; < 0.67 → contracted)
    Check 2: EMA50 slope  (direction reversal = shift)
    Check 3: RSI regime   (was <40 trending, now >60 or vice versa)
    Check 4: Session match (current session vs winning trade sessions)

    Returns:
        regime_same     : bool
        changes_detected: list[str]
        risk_level      : "low" | "medium" | "high"
    """
    changes: list[str] = []

    if df is None or df.empty or len(df) < 55:
        return {"regime_same": True, "changes_detected": [], "risk_level": "low"}

    close = df["close"].astype(float).values if "close" in df.columns else None
    high  = df["high"].astype(float).values  if "high"  in df.columns else None
    low   = df["low"].astype(float).values   if "low"   in df.columns else None

    if close is None:
        return {"regime_same": True, "changes_detected": [], "risk_level": "low"}

    n = len(close)

    # ── Check 1: ATR ─────────────────────────────────────────────────────
    if high is not None and low is not None:
        tr = np.maximum(high - low, np.abs(high[1:] - close[:-1]) if n > 1 else high - low)
        if len(tr) >= 28:
            atr_then = float(np.mean(tr[-28:-14]))
            atr_now  = float(np.mean(tr[-14:]))
            if atr_then > 0:
                ratio = atr_now / atr_then
                if ratio > 1.5:
                    changes.append(f"ATR_EXPANDED (×{ratio:.2f} — volatility spike)")
                elif ratio < 0.67:
                    changes.append(f"ATR_CONTRACTED (×{ratio:.2f} — volatility dried up)")

    # ── Check 2: EMA50 slope ─────────────────────────────────────────────
    if n >= 55:
        alpha = 2 / 51
        ema = close[0]
        emas = []
        for c in close:
            ema = alpha * c + (1 - alpha) * ema
            emas.append(ema)
        emas = np.array(emas)
        slope_then = float(emas[-14] - emas[-28])
        slope_now  = float(emas[-1]  - emas[-14])
        if (slope_then > 0) != (slope_now > 0):
            dir_then = "UP" if slope_then > 0 else "DOWN"
            dir_now  = "UP" if slope_now  > 0 else "DOWN"
            changes.append(f"EMA50_SLOPE_FLIP ({dir_then}→{dir_now})")

    # ── Check 3: RSI regime ───────────────────────────────────────────────
    if n >= 30:
        deltas = np.diff(close[-30:])
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g  = float(np.mean(gains[-14:]))
        avg_l  = float(np.mean(losses[-14:]))
        rsi_now = 100 - (100 / (1 + avg_g / avg_l)) if avg_l > 0 else 100.0

        avg_g2 = float(np.mean(gains[:14]))
        avg_l2 = float(np.mean(losses[:14]))
        rsi_then = 100 - (100 / (1 + avg_g2 / avg_l2)) if avg_l2 > 0 else 100.0

        if rsi_then < 40 and rsi_now > 60:
            changes.append(f"RSI_REGIME_BULLISH_FLIP ({rsi_then:.0f}→{rsi_now:.0f})")
        elif rsi_then > 60 and rsi_now < 40:
            changes.append(f"RSI_REGIME_BEARISH_FLIP ({rsi_then:.0f}→{rsi_now:.0f})")

    # ── Check 4: Session match ────────────────────────────────────────────
    current_session = _current_session()
    log_trades = _load_trade_log()
    winning_sessions: dict[str, int] = {}
    for t in log_trades:
        if t.get("result", "").upper() == "WIN":
            s = t.get("session", "")
            if s:
                winning_sessions[s] = winning_sessions.get(s, 0) + 1

    if winning_sessions:
        best_session = max(winning_sessions, key=winning_sessions.get)
        if best_session != current_session and winning_sessions.get(best_session, 0) >= 3:
            changes.append(
                f"SESSION_MISMATCH (most wins in {best_session}, now {current_session})"
            )

    # ── Risk level ────────────────────────────────────────────────────────
    n_changes = len(changes)
    if n_changes == 0:
        risk_level = "low"
    elif n_changes == 1:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "regime_same":      n_changes == 0,
        "changes_detected": changes,
        "risk_level":       risk_level,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PART 4 — backtest_sequence
# ─────────────────────────────────────────────────────────────────────────────

def backtest_sequence(
    playbook_name: str,
    sequence_pattern: list[str],
    df_historical: pd.DataFrame,
) -> dict:
    """
    Scan historical trade_log for instances where the exact sequence_pattern
    appeared, then record the immediately following trade outcome.

    Also provides a regime breakdown of those following trades.

    Returns:
        matches_found     : int
        outcome_win       : int
        outcome_loss      : int
        win_rate          : float
        avg_loss_pips     : float
        avg_win_pips      : float
        best_case         : float  (max win pips)
        worst_case        : float  (max loss pips)
        recommendation    : str
        regime_breakdown  : dict
    """
    log_trades = _load_trade_log()
    filtered = [
        t for t in log_trades
        if (t.get("pattern_name") or "").strip().lower() == playbook_name.strip().lower()
        and t.get("status") == "closed"
        and t.get("result", "").upper() in ("WIN", "LOSS")
    ]
    filtered.sort(key=lambda t: _parse_dt(t.get("datetime")) or datetime.min)

    seq_len = len(sequence_pattern)
    matches_found = 0
    outcome_win   = 0
    outcome_loss  = 0
    win_pips_list: list[float] = []
    loss_pips_list: list[float] = []
    regime_breakdown: dict[str, int] = {}

    local_seq = [("W" if t.get("result", "").upper() == "WIN" else "L") for t in filtered]

    for i in range(len(local_seq) - seq_len):
        if local_seq[i:i + seq_len] == sequence_pattern:
            next_trade = filtered[i + seq_len]
            matches_found += 1
            res  = next_trade.get("result", "").upper()
            pips = float(next_trade.get("pips", 0) or 0)
            rgm  = next_trade.get("regime", "unknown")
            regime_breakdown[rgm] = regime_breakdown.get(rgm, 0) + 1
            if res == "WIN":
                outcome_win += 1
                win_pips_list.append(pips)
            else:
                outcome_loss += 1
                loss_pips_list.append(pips)

    if matches_found == 0:
        return {
            "matches_found":  0,
            "outcome_win":    0,
            "outcome_loss":   0,
            "win_rate":       0.0,
            "avg_win_pips":   0.0,
            "avg_loss_pips":  0.0,
            "best_case":      0.0,
            "worst_case":     0.0,
            "recommendation": "No historical matches — pattern is new, use caution",
            "regime_breakdown": {},
        }

    wr = outcome_win / matches_found

    recommendation = (
        "SKIP — historically loses after this sequence"   if wr < 0.35 else
        "REDUCE SIZE — marginal edge after this sequence" if wr < 0.50 else
        "NORMAL — good edge continues after this sequence"
    )

    return {
        "matches_found":   matches_found,
        "outcome_win":     outcome_win,
        "outcome_loss":    outcome_loss,
        "win_rate":        round(wr, 3),
        "avg_win_pips":    round(float(np.mean(win_pips_list))  if win_pips_list  else 0.0, 1),
        "avg_loss_pips":   round(float(np.mean(loss_pips_list)) if loss_pips_list else 0.0, 1),
        "best_case":       round(float(max(win_pips_list))      if win_pips_list  else 0.0, 1),
        "worst_case":      round(float(max(loss_pips_list))     if loss_pips_list else 0.0, 1),
        "recommendation":  recommendation,
        "regime_breakdown": regime_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PART 5 — analyze_failed_trade
# ─────────────────────────────────────────────────────────────────────────────

# Economic event keywords to flag as news spikes
_NEWS_KEYWORDS = [
    "non-farm", "nfp", "cpi", "fomc", "fed", "interest rate",
    "gdp", "unemployment", "pce", "payroll", "inflation",
]

_STRATEGY_RULE_THRESHOLD = 3   # how many identical failures before auto-updating rule


def analyze_failed_trade(trade_from_journal: dict) -> dict:
    """
    Diagnose why a trade failed (SL was hit).

    Reason 1 : News spike         — newsworthy event within 30 min of SL hit
    Reason 2 : Wrong regime       — strategy expects trending, market was ranging
    Reason 3 : SL too tight       — price reversed within 3 candles after SL hit
    Reason 4 : Fakeout / sweep    — spike through SL then immediate reversal
    Reason 5 : Trend exhaustion   — 4th/5th test of same level
    Reason 6 : Session change     — entered London session, SL hit in Asian session

    Saves result to data/logs/failed_trade_analysis.json.
    If same reason appears ≥3 times → auto-updates strategy rule.

    Returns:
        primary_reason  : str
        all_reasons     : list[str]
        what_to_do_next : str
        strategy_updated: bool
    """
    all_reasons:  list[str] = []
    explanations: dict[str, str] = {}

    symbol    = trade_from_journal.get("symbol", "XAUUSD")
    direction = (trade_from_journal.get("direction") or "").lower()
    entry     = float(trade_from_journal.get("entry", 0) or 0)
    sl        = float(trade_from_journal.get("stop_loss", trade_from_journal.get("sl", 0)) or 0)
    pnl       = float(trade_from_journal.get("pnl_usd", 0) or 0)
    open_time_raw  = trade_from_journal.get("open_time") or trade_from_journal.get("opened_at", "")
    close_time_raw = trade_from_journal.get("closed_at") or trade_from_journal.get("close_time", "")
    pattern   = (trade_from_journal.get("pattern_name") or "unknown").strip()
    regime    = (trade_from_journal.get("regime") or "").lower()
    session   = (trade_from_journal.get("session") or "").title()

    open_dt  = _parse_dt(open_time_raw)
    close_dt = _parse_dt(close_time_raw)

    # ── Reason 1: News spike ─────────────────────────────────────────────
    news_event_near = False
    try:
        from news_calendar import get_upcoming_events  # type: ignore
        events = get_upcoming_events()
        for ev in (events or []):
            ev_dt = _parse_dt(ev.get("datetime") or ev.get("time", ""))
            if ev_dt and close_dt:
                delta_min = abs((ev_dt - close_dt).total_seconds() / 60)
                ev_name = (ev.get("event") or ev.get("name", "")).lower()
                is_high_impact = any(kw in ev_name for kw in _NEWS_KEYWORDS)
                if delta_min <= 30 and is_high_impact:
                    news_event_near = True
                    all_reasons.append("NEWS_SPIKE")
                    explanations["NEWS_SPIKE"] = (
                        f"High-impact event '{ev.get('event','')}' was within "
                        f"{delta_min:.0f} min of SL hit"
                    )
                    break
    except Exception:
        pass

    # ── Reason 2: Wrong regime ───────────────────────────────────────────
    regime_mismatch = False
    trending_strategies = [
        "london breakout", "trend continuation", "ema pullback",
        "breakout", "momentum", "continuation"
    ]
    pat_lower = pattern.lower()
    expects_trend = any(kw in pat_lower for kw in trending_strategies)
    if expects_trend and regime in ("ranging", "range", "choppy", "sideways"):
        regime_mismatch = True
        all_reasons.append("WRONG_REGIME")
        explanations["WRONG_REGIME"] = (
            f"Strategy '{pattern}' expects trending, but regime was '{regime}'"
        )

    # ── Reason 3: SL too tight ───────────────────────────────────────────
    # We approximate: if pnl is very small negative vs full SL distance, SL was tight
    sl_tight = False
    if sl > 0 and entry > 0:
        sl_dist = abs(entry - sl)
        # If SL was < 50% of average ATR (proxy: < $8 for gold)
        if sl_dist < 8.0:
            sl_tight = True
            all_reasons.append("SL_TOO_TIGHT")
            explanations["SL_TOO_TIGHT"] = (
                f"SL distance was only ${sl_dist:.2f} — likely inside market noise"
            )

    # ── Reason 4: Fakeout / liquidity sweep ─────────────────────────────
    fakeout = False
    # Heuristic: loss much smaller than expected full-SL loss suggests a spike/sweep
    if sl > 0 and entry > 0 and pnl < 0:
        expected_full_loss = abs(entry - sl) * 0.01 * 100  # 0.01 lots estimate
        if abs(pnl) < expected_full_loss * 0.3 and abs(pnl) > 0:
            fakeout = True
            all_reasons.append("FAKEOUT_SWEEP")
            explanations["FAKEOUT_SWEEP"] = (
                f"Loss (${abs(pnl):.2f}) much smaller than full-SL loss "
                f"(${expected_full_loss:.2f}) — possible spike through SL"
            )

    # ── Reason 5: Trend exhaustion ───────────────────────────────────────
    exhaustion = False
    log_trades = _load_trade_log()
    # Count how many times this pattern was traded near same price zone
    zone = int(entry // 5) * 5 if entry else 0
    same_zone_count = sum(
        1 for t in log_trades
        if (t.get("pattern_name") or "").strip().lower() == pat_lower
        and abs(float(t.get("entry", 0) or 0) - entry) <= 5
    )
    if same_zone_count >= 4:
        exhaustion = True
        all_reasons.append("TREND_EXHAUSTION")
        explanations["TREND_EXHAUSTION"] = (
            f"This price zone (${zone}–${zone+5}) has been tested {same_zone_count}× — "
            "level is exhausted"
        )

    # ── Reason 6: Session change ─────────────────────────────────────────
    session_change = False
    if open_dt and close_dt:
        def _session_for_hour(h: int) -> str:
            if 22 <= h or h < 7:  return "Asian"
            if 7  <= h < 12:      return "London"
            return "New York"

        open_sess  = _session_for_hour(open_dt.hour)
        close_sess = _session_for_hour(close_dt.hour)
        if open_sess != close_sess:
            session_change = True
            all_reasons.append("SESSION_CHANGE")
            explanations["SESSION_CHANGE"] = (
                f"Entered in {open_sess}, SL hit in {close_sess} — "
                "different session dynamics"
            )

    # ── Reason 7: Volume mismatch ─────────────────────────────────────────
    vol_ratio = float(trade_from_journal.get("volume_ratio", 0) or 0)
    if vol_ratio > 0 and vol_ratio < 1.0:
        all_reasons.append("VOLUME_MISMATCH")
        explanations["VOLUME_MISMATCH"] = (
            f"Volume ratio {vol_ratio:.2f}x was below 1.0 — weak participation"
        )

    # ── Reason 8: Climax missed ───────────────────────────────────────────
    climax = trade_from_journal.get("climax_detected", False)
    if climax:
        all_reasons.append("CLIMAX_MISSED")
        explanations["CLIMAX_MISSED"] = (
            "Volume climax was present at entry — exhaustion signal missed"
        )

    # ── Determine primary reason ─────────────────────────────────────────
    priority = [
        "NEWS_SPIKE", "WRONG_REGIME", "SL_TOO_TIGHT",
        "FAKEOUT_SWEEP", "TREND_EXHAUSTION", "SESSION_CHANGE",
        "VOLUME_MISMATCH", "CLIMAX_MISSED",
    ]
    primary_reason = "UNKNOWN"
    for r in priority:
        if r in all_reasons:
            primary_reason = r
            break

    if not all_reasons:
        all_reasons = ["UNKNOWN"]
        explanations["UNKNOWN"] = "No specific pattern matched — review setup manually"

    # ── What to do next time ─────────────────────────────────────────────
    next_time_map = {
        "NEWS_SPIKE":       "Check the economic calendar before entry — avoid high-impact events within 1 hour",
        "WRONG_REGIME":     "Run regime check before entry — only take this strategy in trending conditions",
        "SL_TOO_TIGHT":     "Widen SL to at least 1 ATR — place below nearest swing structure",
        "FAKEOUT_SWEEP":    "Add liquidity sweep filter — wait for candle close beyond level before entry",
        "TREND_EXHAUSTION": "Skip trades at exhausted levels — look for fresh, untested structure",
        "SESSION_CHANGE":   "Close trade before session changes or move SL to breakeven at session close",
        "VOLUME_MISMATCH":  "Add minimum volume filter — only trade when volume ratio meets playbook threshold",
        "CLIMAX_MISSED":    "Skip signals within 3 candles of volume climax — add climax check to pre-entry routine",
        "UNKNOWN":          "Review trade chart manually and log observation",
    }
    what_to_do_next = next_time_map.get(primary_reason, "Review manually")

    # ── Save to log ───────────────────────────────────────────────────────
    record = {
        "saved_at":      datetime.now(GST).isoformat(),
        "pattern":       pattern,
        "symbol":        symbol,
        "direction":     direction,
        "entry":         entry,
        "sl":            sl,
        "pnl_usd":       pnl,
        "open_time":     str(open_time_raw),
        "close_time":    str(close_time_raw),
        "primary_reason":primary_reason,
        "all_reasons":   all_reasons,
        "explanations":  explanations,
        "what_to_do_next": what_to_do_next,
    }
    try:
        _save_failed_analysis(record)
        log_info(f"[pattern_fatigue] Failed trade analysis saved — reason: {primary_reason}")
    except Exception as exc:
        log_error(module="pattern_fatigue", error=str(exc), context="save_failed_analysis")

    # ── Auto-update strategy rule if same reason ≥ threshold ─────────────
    strategy_updated = False
    try:
        all_records: list[dict] = []
        if os.path.exists(FAILED_ANALYSIS_FILE):
            with open(FAILED_ANALYSIS_FILE, "r", encoding="utf-8") as f:
                all_records = json.load(f)

        # Count occurrences of primary_reason for this pattern
        same_reason_count = sum(
            1 for r in all_records
            if r.get("pattern", "").lower() == pat_lower
            and r.get("primary_reason") == primary_reason
        )

        if same_reason_count >= _STRATEGY_RULE_THRESHOLD:
            rules = _load_rules()
            rule_name = f"AUTO_{primary_reason}_{pat_lower.replace(' ','_').upper()}"
            existing_rule_names = [r.get("name", "") for r in rules]
            if rule_name not in existing_rule_names:
                new_rule = {
                    "name":        rule_name,
                    "description": (
                        f"Auto-added: Block '{pattern}' when reason '{primary_reason}' "
                        f"triggered {same_reason_count}× — {what_to_do_next}"
                    ),
                    "filter_type": primary_reason.lower(),
                    "pattern":     pattern,
                    "auto_added":  True,
                    "added_at":    datetime.now(GST).isoformat(),
                    "confidence":  0.0,
                }
                rules.append(new_rule)
                _save_rules(rules)
                strategy_updated = True
                log_info(
                    f"[pattern_fatigue] Auto-rule added: {rule_name} "
                    f"(triggered {same_reason_count}× for '{pattern}')"
                )
    except Exception as exc:
        log_error(module="pattern_fatigue", error=str(exc), context="auto_rule_update")

    return {
        "primary_reason":   primary_reason,
        "all_reasons":      all_reasons,
        "explanations":     explanations,
        "what_to_do_next":  what_to_do_next,
        "strategy_updated": strategy_updated,
    }
