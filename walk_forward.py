"""
walk_forward.py — Walk-Forward Optimization for TradingBotV1

Runs every Sunday automatically, analyses last 30 days of signal
performance and adjusts key parameters to maximise win rate.

Functions:
  run_walk_forward_optimization() -> dict
  check_if_sunday_run_needed()    -> bool
  get_wfo_summary()               -> str
"""

import json
import os
from datetime import datetime, timezone, timedelta

# ── Timezone ──────────────────────────────────────────────────────────────────
GST = timezone(timedelta(hours=4))   # UAE / Dubai time

DATA_DIR = "data"
WFO_HISTORY_PATH = os.path.join(DATA_DIR, "wfo_history.json")
SIGNAL_PERF_PATH = os.path.join(DATA_DIR, "signal_performance.json")

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from signal_tracker import get_signal_performance_report
    _ST_OK = True
except ImportError:
    _ST_OK = False
    def get_signal_performance_report():  # type: ignore[misc]
        return {"win_rate": 50.0, "total_signals": 0}

try:
    from pattern_fatigue import get_pattern_stats
    _PF_OK = True
except ImportError:
    _PF_OK = False
    def get_pattern_stats():  # type: ignore[misc]
        return {}

try:
    from settings import load_settings, save_settings
    _SETTINGS_OK = True
except ImportError:
    _SETTINGS_OK = False
    def load_settings() -> dict:   # type: ignore[misc]
        try:
            with open(os.path.join(DATA_DIR, "user_settings.json")) as f:
                return json.load(f)
        except Exception:
            return {}
    def save_settings(updates: dict) -> dict:  # type: ignore[misc]
        try:
            current = load_settings()
            current.update(updates)
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(os.path.join(DATA_DIR, "user_settings.json"), "w") as f:
                json.dump(current, f, indent=2)
            return current
        except Exception:
            return updates


# ══════════════════════════════════════════════════════════════════════════════
#  run_walk_forward_optimization
# ══════════════════════════════════════════════════════════════════════════════

def run_walk_forward_optimization() -> dict:
    """
    Analyse last 30 days of signal performance and adjust bot parameters
    to maximise win rate.

    Returns a report dict with keys:
      optimized, reason (if not optimized), run_at, signals_analyzed,
      overall_win_rate, changes_made, changes, playbook_stats,
      session_stats, weak_playbooks, strong_playbooks, best_session,
      worst_session, optimized_settings, next_run
    """

    # ── STEP 1 — Load recent performance data ─────────────────────────────────
    perf = get_signal_performance_report()

    try:
        with open(SIGNAL_PERF_PATH) as f:
            all_signals = json.load(f)

        cutoff = datetime.now(GST) - timedelta(days=30)
        recent = [
            s for s in all_signals
            if s.get("registered_at", "") > cutoff.isoformat()
        ]
    except Exception:
        recent = []

    if len(recent) < 10:
        return {
            "optimized": False,
            "reason": "Not enough data — need 10+ signals",
            "next_run": "Next Sunday",
        }

    # ── STEP 2 — Analyse performance by playbook ──────────────────────────────
    playbook_stats: dict = {}
    for sig in recent:
        pb = sig.get("playbook", "unknown")
        if pb not in playbook_stats:
            playbook_stats[pb] = {"wins": 0, "losses": 0, "total": 0}
        if sig.get("outcome") == "win":
            playbook_stats[pb]["wins"] += 1
        elif sig.get("outcome") in ("loss", "sl_hit"):
            playbook_stats[pb]["losses"] += 1
        playbook_stats[pb]["total"] += 1

    for pb, stats in playbook_stats.items():
        stats["win_rate"] = (
            stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0
        )

    # ── STEP 3 — Analyse performance by session ───────────────────────────────
    session_stats: dict = {}
    for sig in recent:
        sess = sig.get("session", "unknown")
        if sess not in session_stats:
            session_stats[sess] = {"wins": 0, "losses": 0, "total": 0}
        if sig.get("outcome") == "win":
            session_stats[sess]["wins"] += 1
        elif sig.get("outcome") in ("loss", "sl_hit"):
            session_stats[sess]["losses"] += 1
        session_stats[sess]["total"] += 1

    for sess, stats in session_stats.items():
        stats["win_rate"] = (
            stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0
        )

    # ── STEP 4 — Generate optimized parameters ────────────────────────────────
    current   = load_settings()
    optimized = current.copy()
    changes: list = []

    overall_wr = float(perf.get("win_rate", 50))

    # Adjust min_confidence based on overall win rate
    if overall_wr < 45:
        new_conf = min(float(current.get("min_confidence", 6.0)) + 0.5, 8.0)
        optimized["min_confidence"] = new_conf
        changes.append(
            f"min_confidence: {current.get('min_confidence', 6.0)} → {new_conf} "
            f"(win rate {overall_wr:.1f}% too low)"
        )
    elif overall_wr > 65:
        new_conf = max(float(current.get("min_confidence", 6.0)) - 0.3, 5.5)
        optimized["min_confidence"] = new_conf
        changes.append(
            f"min_confidence: {current.get('min_confidence', 6.0)} → {new_conf} "
            f"(win rate {overall_wr:.1f}% strong)"
        )

    # Flag underperforming / strong playbooks
    weak_playbooks = [
        pb for pb, stats in playbook_stats.items()
        if stats["total"] >= 3 and stats["win_rate"] < 40
    ]
    strong_playbooks = [
        pb for pb, stats in playbook_stats.items()
        if stats["total"] >= 3 and stats["win_rate"] > 65
    ]

    # Best / worst session (only count sessions with ≥3 trades)
    best_session = max(
        session_stats.items(),
        key=lambda x: x[1]["win_rate"] if x[1]["total"] >= 3 else 0,
        default=(None, {}),
    )[0]

    worst_session = min(
        session_stats.items(),
        key=lambda x: x[1]["win_rate"] if x[1]["total"] >= 3 else 100,
        default=(None, {}),
    )[0]

    # ── STEP 5 — Save optimized settings ──────────────────────────────────────
    if changes:
        save_settings(optimized)
        changes_made = True
    else:
        changes_made = False

    # Build report
    report = {
        "optimized":          True,
        "run_at":             datetime.now(GST).isoformat(),
        "signals_analyzed":   len(recent),
        "overall_win_rate":   overall_wr,
        "changes_made":       changes_made,
        "changes":            changes,
        "playbook_stats":     playbook_stats,
        "session_stats":      session_stats,
        "weak_playbooks":     weak_playbooks,
        "strong_playbooks":   strong_playbooks,
        "best_session":       best_session,
        "worst_session":      worst_session,
        "optimized_settings": optimized,
        "next_run":           "Next Sunday",
    }

    # Append to history (keep 52 weeks / 1 year)
    try:
        with open(WFO_HISTORY_PATH) as f:
            history = json.load(f)
    except Exception:
        history = []
    history.append(report)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(WFO_HISTORY_PATH, "w") as f:
        json.dump(history[-52:], f, indent=2)

    # Trigger ML retraining after optimization
    try:
        from ml_engine import run_ml_training as _wfo_ml_train
        ml_result = _wfo_ml_train()
        report["ml_retrain"] = ml_result
    except Exception:
        pass

    return report


# ══════════════════════════════════════════════════════════════════════════════
#  check_if_sunday_run_needed
# ══════════════════════════════════════════════════════════════════════════════

def check_if_sunday_run_needed() -> bool:
    """
    Returns True if:
      1. Today is Sunday (UAE / GST time), AND
      2. Last WFO run was more than 6 days ago (or never run).
    """
    now = datetime.now(GST)
    if now.weekday() != 6:          # 6 = Sunday
        return False

    try:
        with open(WFO_HISTORY_PATH) as f:
            history = json.load(f)
        if not history:
            return True
        last_run_str = history[-1].get("run_at", "")
        if not last_run_str:
            return True
        # Parse ISO timestamp (may or may not have tz info)
        try:
            last_run = datetime.fromisoformat(last_run_str)
        except Exception:
            return True
        # Normalise to UTC-aware for comparison
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=GST)
        age_days = (now - last_run).total_seconds() / 86400
        return age_days > 6
    except Exception:
        return True      # no history file → run now


# ══════════════════════════════════════════════════════════════════════════════
#  get_wfo_summary
# ══════════════════════════════════════════════════════════════════════════════

def get_wfo_summary() -> str:
    """
    Load the latest WFO history entry and return a formatted summary string.
    """
    try:
        with open(WFO_HISTORY_PATH) as f:
            history = json.load(f)
        if not history:
            return "📊 No optimization runs yet. Will run automatically next Sunday."
        latest = history[-1]
    except Exception:
        return "📊 No optimization runs yet. Will run automatically next Sunday."

    if not latest.get("optimized", True):
        reason = latest.get("reason", "unknown")
        return (
            f"📊 LAST OPTIMIZATION ATTEMPT: {latest.get('run_at', 'unknown')}\n"
            f"   Not optimized: {reason}\n"
            f"   Next run: Next Sunday"
        )

    run_at    = latest.get("run_at", "unknown")
    # Format date nicely if possible
    try:
        run_dt  = datetime.fromisoformat(run_at)
        run_str = run_dt.strftime("%A %d %B %Y %H:%M GST")
    except Exception:
        run_str = run_at

    n_sigs    = latest.get("signals_analyzed", 0)
    wr        = latest.get("overall_win_rate", 0.0)
    changed   = latest.get("changes_made", False)
    changes   = latest.get("changes", [])
    best_s    = latest.get("best_session")
    worst_s   = latest.get("worst_session")
    strong_pb = latest.get("strong_playbooks", [])
    weak_pb   = latest.get("weak_playbooks", [])
    sess_stats = latest.get("session_stats", {})

    lines = [
        f"📊 LAST OPTIMIZATION: {run_str}",
        f"   Signals analysed: {n_sigs}",
        f"   Win rate: {wr:.1f}%",
        f"   Changes made: {'Yes' if changed else 'No'}",
    ]

    if changed and changes:
        for c in changes:
            lines.append(f"   • {c}")

    # Best session with win rate
    if best_s and best_s in sess_stats:
        bs_wr = sess_stats[best_s].get("win_rate", 0.0)
        lines.append(f"   Best session: {best_s} ({bs_wr:.0f}% WR)")
    elif best_s:
        lines.append(f"   Best session: {best_s}")

    # Worst session with win rate
    if worst_s and worst_s in sess_stats:
        ws_wr = sess_stats[worst_s].get("win_rate", 0.0)
        lines.append(f"   Worst session: {worst_s} ({ws_wr:.0f}% WR)")
    elif worst_s:
        lines.append(f"   Worst session: {worst_s}")

    if strong_pb:
        lines.append(f"   Strong playbooks: {', '.join(strong_pb)}")
    if weak_pb:
        lines.append(f"   Weak playbooks: {', '.join(weak_pb)}")

    lines.append("   Next run: Next Sunday")
    return "\n".join(lines)
