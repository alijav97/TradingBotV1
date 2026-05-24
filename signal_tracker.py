"""
signal_tracker.py — Brain 2: Bot Signal Accuracy Tracker (TradingBotV1)
════════════════════════════════════════════════════════════════════════
Monitors every signal the bot generates — whether the user traded it
or not — by tracking live price against SL/TP levels.

Storage: data/signal_performance.json

Functions
---------
register_signal(signal)            → str  (signal_id)
update_signal_prices(current_price) → list (changed records)
get_signal_performance_report()    → dict
mark_user_traded(signal_id, bool)  → None
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.path.join(_BASE_DIR, "data")
_PERF_FILE = os.path.join(_DATA_DIR, "signal_performance.json")
_MEM_FILE  = os.path.join(_DATA_DIR, "pattern_memory.json")

os.makedirs(_DATA_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_perf() -> list[dict]:
    if not os.path.exists(_PERF_FILE):
        return []
    try:
        with open(_PERF_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _save_perf(records: list[dict]) -> None:
    with open(_PERF_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def _load_memory() -> list[dict]:
    if not os.path.exists(_MEM_FILE):
        return []
    try:
        with open(_MEM_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_hours(registered_at: str) -> float:
    try:
        dt = datetime.fromisoformat(registered_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  BRAIN 2 — Public API
# ══════════════════════════════════════════════════════════════════════════════

def register_signal(signal: dict) -> str:
    """
    Register a bot-generated signal for outcome tracking.

    Accepts the full signal dict from _handle_gold/_handle_signals/
    _step5_scan_signals and maps it to the internal tracking record.

    Returns the new signal_id (short uuid4 hex).
    """
    signal_id = uuid.uuid4().hex[:12]

    entry  = float(signal.get("entry",      0) or 0)
    sl     = float(signal.get("stop_loss",  0) or 0)
    tp     = float(signal.get("take_profit", 0) or 0)

    # Accept tp1/tp2 explicitly; otherwise split single TP into tp1=tp, tp2=tp
    tp1 = float(signal.get("tp1", tp) or tp)
    tp2 = float(signal.get("tp2", tp) or tp)

    sl_dist  = round(abs(entry - sl),  2) if entry and sl  else 0.0
    tp1_dist = round(abs(tp1  - entry), 2) if entry and tp1 else 0.0

    record: dict[str, Any] = {
        "signal_id":          signal_id,
        "registered_at":      _now_utc(),
        "playbook":           signal.get("pattern_name", "unknown"),
        "direction":          str(signal.get("direction", "")).lower(),
        "entry":              entry,
        "sl":                 sl,
        "tp1":                tp1,
        "tp2":                tp2,
        "sl_distance":        sl_dist,
        "tp1_distance":       tp1_dist,
        "confluence_score":   float(signal.get("confidence", 0) or 0),
        "session":            signal.get("session", ""),
        "regime":             signal.get("regime",  ""),
        "global_news_bias":   signal.get("global_news_bias", ""),
        "mtf_score":          int(signal.get("mtf_score", 0) or 0),
        "spread_at_signal":   float(signal.get("spread_at_signal", 0) or 0),
        "status":             "open",
        "outcome":            None,
        "outcome_pips":       None,
        "outcome_at":         None,
        "max_favorable":      0.0,
        "max_adverse":        0.0,
        "bars_to_outcome":    None,
        "user_traded":        None,
        "notes":              [],
    }

    records = _load_perf()
    records.append(record)
    _save_perf(records)
    return signal_id


def update_signal_prices(current_price: float) -> list[dict]:
    """
    Check all open signals against current price.
    Marks SL/TP hits, updates max_favorable/max_adverse, expires stale records.

    Returns list of records whose status changed this call.
    """
    if not current_price or current_price <= 0:
        return []

    records = _load_perf()
    changed: list[dict] = []

    for rec in records:
        if rec.get("status") != "open":
            continue

        entry    = float(rec.get("entry",  0) or 0)
        sl       = float(rec.get("sl",     0) or 0)
        tp1      = float(rec.get("tp1",    0) or 0)
        tp2      = float(rec.get("tp2",    0) or 0)
        sl_dist  = float(rec.get("sl_distance",  1) or 1)
        tp1_dist = float(rec.get("tp1_distance", 1) or 1)
        direction = str(rec.get("direction", "long")).lower()
        is_long   = direction == "long"

        if not entry:
            continue

        now_str = _now_utc()

        # ── Update max excursions ─────────────────────────────────────────────
        if is_long:
            favorable = current_price - entry
            adverse   = entry - current_price
        else:
            favorable = entry - current_price
            adverse   = current_price - entry

        if favorable > rec.get("max_favorable", 0):
            rec["max_favorable"] = round(favorable, 2)
        if adverse > rec.get("max_adverse", 0):
            rec["max_adverse"] = round(adverse, 2)

        # ── Expiry check (48 hours) ───────────────────────────────────────────
        if _age_hours(rec.get("registered_at", "")) >= 48:
            if is_long:
                pips = round((current_price - entry) / 0.1, 1)
            else:
                pips = round((entry - current_price) / 0.1, 1)
            rec["status"]        = "expired"
            rec["outcome"]       = "expired"
            rec["outcome_pips"]  = pips
            rec["outcome_at"]    = now_str
            rec["notes"].append(f"Expired at price {current_price:.2f} after 48h")
            changed.append(rec)
            continue

        # ── SL hit ────────────────────────────────────────────────────────────
        sl_hit = (is_long and current_price <= sl) or (not is_long and current_price >= sl)
        if sl_hit and sl:
            rec["status"]       = "sl_hit"
            rec["outcome"]      = "loss"
            rec["outcome_pips"] = round(-sl_dist / 0.1, 1)
            rec["outcome_at"]   = now_str
            changed.append(rec)
            continue

        # ── TP2 hit ───────────────────────────────────────────────────────────
        tp2_hit = tp2 and (
            (is_long  and current_price >= tp2) or
            (not is_long and current_price <= tp2)
        )
        if tp2_hit:
            tp2_dist = abs(tp2 - entry)
            rec["status"]       = "tp2_hit"
            rec["outcome"]      = "win"
            rec["outcome_pips"] = round(tp2_dist / 0.1, 1)
            rec["outcome_at"]   = now_str
            changed.append(rec)
            continue

        # ── TP1 hit ───────────────────────────────────────────────────────────
        tp1_hit = tp1 and (
            (is_long  and current_price >= tp1) or
            (not is_long and current_price <= tp1)
        )
        if tp1_hit and rec.get("status") == "open":
            rec["status"]       = "tp1_hit"
            rec["outcome"]      = "partial"
            rec["outcome_pips"] = round(tp1_dist / 0.1, 1)
            note = "TP1 reached — tracking TP2"
            if note not in rec["notes"]:
                rec["notes"].append(note)
            changed.append(rec)
            # Do NOT set outcome_at — keep monitoring for TP2

    if changed:
        _save_perf(records)

    return changed


def mark_user_traded(signal_id: str, traded: bool) -> None:
    """
    Set user_traded on a tracking record.
    Called from mt5_sync.auto_match_and_update() when a real trade is matched.
    """
    records = _load_perf()
    for rec in records:
        if rec.get("signal_id") == signal_id:
            rec["user_traded"] = traded
            _save_perf(records)
            return


def get_signal_performance_report() -> dict:
    """
    Aggregate all tracked signals into a performance report.

    Returns dict with keys:
        overall, by_playbook, by_session, by_regime,
        user_comparison, verdict
    """
    records = _load_perf()
    memory  = _load_memory()

    # Build real-trade RR map from pattern_memory keyed by playbook
    rr_map: dict[str, list[float]] = {}
    for m in memory:
        pb  = str(m.get("playbook", "")).strip()
        rr  = m.get("rr_achieved")
        if pb and rr is not None:
            rr_map.setdefault(pb, []).append(float(rr))

    resolved = [r for r in records if r.get("outcome") not in (None, "")]
    wins     = [r for r in resolved if r.get("outcome") == "win"]
    losses   = [r for r in resolved if r.get("outcome") == "loss"]
    partials = [r for r in resolved if r.get("outcome") == "partial"]

    total_signals = len(records)
    total_resolved = len(resolved)
    win_rate = round(len(wins) / total_resolved * 100, 1) if total_resolved else 0.0

    win_pips  = [abs(r.get("outcome_pips") or 0) for r in wins]
    loss_pips = [abs(r.get("outcome_pips") or 0) for r in losses]
    avg_win   = round(sum(win_pips)  / len(win_pips),  1) if win_pips  else 0.0
    avg_loss  = round(sum(loss_pips) / len(loss_pips), 1) if loss_pips else 0.0
    pf = round(
        (sum(win_pips) / sum(loss_pips)) if loss_pips and sum(loss_pips) > 0 else 0.0,
        2
    )

    # ── By playbook ───────────────────────────────────────────────────────────
    pb_stats: dict[str, dict] = {}
    for r in resolved:
        pb = str(r.get("playbook", "unknown")).strip() or "unknown"
        pb_stats.setdefault(pb, {"signals": 0, "wins": 0, "losses": 0,
                                  "pips": [], "real_rr": []})
        pb_stats[pb]["signals"] += 1
        if r.get("outcome") == "win":
            pb_stats[pb]["wins"] += 1
        elif r.get("outcome") == "loss":
            pb_stats[pb]["losses"] += 1
        if r.get("outcome_pips") is not None:
            pb_stats[pb]["pips"].append(float(r["outcome_pips"]))
        if pb in rr_map:
            pb_stats[pb]["real_rr"] = rr_map[pb]

    for pb, d in pb_stats.items():
        tot = d["wins"] + d["losses"]
        d["win_rate"] = round(d["wins"] / tot * 100, 1) if tot else 0.0
        d["avg_pips"] = round(sum(d["pips"]) / len(d["pips"]), 1) if d["pips"] else 0.0
        avg_rr = d["real_rr"]
        d["avg_real_rr"] = round(sum(avg_rr) / len(avg_rr), 2) if avg_rr else None
        del d["pips"], d["real_rr"]

    best_pb  = max(pb_stats, key=lambda k: pb_stats[k]["win_rate"], default="—")
    worst_pb = min(pb_stats, key=lambda k: pb_stats[k]["win_rate"], default="—")

    # ── By session ────────────────────────────────────────────────────────────
    sess_stats: dict[str, dict] = {}
    for r in resolved:
        s = str(r.get("session", "unknown")).strip() or "unknown"
        sess_stats.setdefault(s, {"total": 0, "wins": 0})
        sess_stats[s]["total"] += 1
        if r.get("outcome") == "win":
            sess_stats[s]["wins"] += 1
    for s, d in sess_stats.items():
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] else 0.0
    best_sess = max(sess_stats, key=lambda k: sess_stats[k]["win_rate"], default="—")

    # ── By regime ─────────────────────────────────────────────────────────────
    reg_stats: dict[str, dict] = {}
    for r in resolved:
        rg = str(r.get("regime", "unknown")).strip() or "unknown"
        reg_stats.setdefault(rg, {"total": 0, "wins": 0})
        reg_stats[rg]["total"] += 1
        if r.get("outcome") == "win":
            reg_stats[rg]["wins"] += 1
    for rg, d in reg_stats.items():
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] else 0.0

    # ── User comparison ───────────────────────────────────────────────────────
    user_took    = [r for r in resolved if r.get("user_traded") is True]
    user_skipped = [r for r in resolved if r.get("user_traded") is False]

    def _wr(lst: list[dict]) -> float:
        w = sum(1 for r in lst if r.get("outcome") == "win")
        return round(w / len(lst) * 100, 1) if lst else 0.0

    user_wr    = _wr(user_took)
    skipped_wr = _wr(user_skipped)

    if len(user_took) < 3:
        verdict = "⚪ Not enough real trades to compare yet (need 3+)"
    elif user_wr > win_rate + 5:
        verdict = "✅ You filter well — your trades outperform bot average"
    elif skipped_wr > user_wr + 5:
        verdict = "⚠ Bot is more accurate than your filtering — trust it more"
    else:
        verdict = "➡ Your filtering is neutral — no clear edge either way"

    return {
        "overall": {
            "total_signals":  total_signals,
            "resolved":       total_resolved,
            "open":           total_signals - total_resolved,
            "wins":           len(wins),
            "losses":         len(losses),
            "partials":       len(partials),
            "win_rate":       win_rate,
            "avg_win_pips":   avg_win,
            "avg_loss_pips":  avg_loss,
            "profit_factor":  pf,
            "best_playbook":  best_pb,
            "worst_playbook": worst_pb,
            "best_session":   best_sess,
        },
        "by_playbook": pb_stats,
        "by_session":  sess_stats,
        "by_regime":   reg_stats,
        "user_comparison": {
            "signals_user_took":    len(user_took),
            "user_win_rate":        user_wr,
            "signals_user_skipped": len(user_skipped),
            "skipped_win_rate":     skipped_wr,
            "overall_bot_win_rate": win_rate,
        },
        "verdict": verdict,
    }
