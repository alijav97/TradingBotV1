"""
debug_logger.py — TradingBotV1 Comprehensive Debug Logger
──────────────────────────────────────────────────────────
Records every decision the bot makes and why.

Creates three log files in data/logs/:
  bot_activity.log   — plain text, human-readable, step-by-step activity
  signal_detail.json — structured JSON, every signal with full breakdown
  errors.log         — every error, which module, what fallback was used

Usage:
    from debug_logger import (
        log_info, log_signal, log_rejected, log_error,
        log_session_start, log_session_end,
        log_playbook_check, log_confluence, log_checklist,
        save_signal_detail, get_logger,
    )
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ── Timezone ──────────────────────────────────────────────────────────────────
GST = timezone(timedelta(hours=4))

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

ACTIVITY_LOG = os.path.join(LOG_DIR, "bot_activity.log")
SIGNAL_JSON  = os.path.join(LOG_DIR, "signal_detail.json")
ERROR_LOG    = os.path.join(LOG_DIR, "errors.log")


# ══════════════════════════════════════════════════════════════════════════════
#  Custom GST formatter
# ══════════════════════════════════════════════════════════════════════════════

class _GSTFormatter(logging.Formatter):
    """Format log records with GST timestamp and level label."""

    LEVEL_LABELS = {
        "INFO":     "INFO",
        "WARNING":  "WARN",
        "ERROR":    "ERROR",
        "DEBUG":    "DEBUG",
        "CRITICAL": "CRITICAL",
        # Custom pseudo-levels stored in the record's 'label' attribute
    }

    def format(self, record: logging.LogRecord) -> str:
        ts    = datetime.now(GST).strftime("%Y-%m-%d %H:%M:%S GST")
        label = getattr(record, "label", self.LEVEL_LABELS.get(record.levelname, record.levelname))
        msg   = record.getMessage()
        return f"[{ts}] {label:<8} — {msg}"


# ══════════════════════════════════════════════════════════════════════════════
#  Logger setup — called once at import time
# ══════════════════════════════════════════════════════════════════════════════

def _make_file_handler(path: str, level: int = logging.DEBUG) -> logging.FileHandler:
    h = logging.FileHandler(path, encoding="utf-8")
    h.setLevel(level)
    h.setFormatter(_GSTFormatter())
    return h


# ── Activity logger (bot_activity.log) ───────────────────────────────────────
_activity_logger = logging.getLogger("bot.activity")
_activity_logger.setLevel(logging.DEBUG)
_activity_logger.propagate = False
if not _activity_logger.handlers:
    _activity_logger.addHandler(_make_file_handler(ACTIVITY_LOG))

# ── Error logger (errors.log) ─────────────────────────────────────────────────
_error_logger = logging.getLogger("bot.errors")
_error_logger.setLevel(logging.ERROR)
_error_logger.propagate = False
if not _error_logger.handlers:
    _error_logger.addHandler(_make_file_handler(ERROR_LOG))

# ── Session start time (for export filtering) ─────────────────────────────────
_SESSION_START: datetime = datetime.now(GST)


def get_logger() -> logging.Logger:
    """Return the activity logger for direct use."""
    return _activity_logger


# ══════════════════════════════════════════════════════════════════════════════
#  Core log helpers — use these everywhere
# ══════════════════════════════════════════════════════════════════════════════

def _emit(label: str, message: str) -> None:
    """Emit a log record with a custom label."""
    record = logging.LogRecord(
        name="bot.activity", level=logging.INFO,
        pathname="", lineno=0,
        msg=message, args=(), exc_info=None,
    )
    record.label = label
    _activity_logger.handle(record)


def log_info(message: str) -> None:
    """General informational message."""
    _emit("INFO", message)


def log_signal(message: str) -> None:
    """A signal was generated or accepted."""
    _emit("SIGNAL", message)


def log_rejected(message: str) -> None:
    """A signal was rejected at any stage."""
    _emit("REJECTED", message)


def log_error(
    module: str,
    function: str,
    error: str,
    fallback: str = "none",
    impact: str = "unknown",
) -> None:
    """
    Log an error with full context to errors.log AND bot_activity.log.

    Parameters
    ----------
    module   : module name (e.g. "smart_money")
    function : function name (e.g. "find_order_blocks")
    error    : str(e) from the exception
    fallback : what the bot did instead
    impact   : how this affects output quality
    """
    msg = (
        f"ERROR in {module} → {function}()\n"
        f"  Error:    {error}\n"
        f"  Fallback: {fallback}\n"
        f"  Impact:   {impact}"
    )
    _error_logger.error(msg)
    _emit("ERROR", f"{module}.{function}() — {error} | fallback: {fallback}")


# ══════════════════════════════════════════════════════════════════════════════
#  Session lifecycle
# ══════════════════════════════════════════════════════════════════════════════

def log_session_start(
    gold_price: float = 0.0,
    session: str = "—",
    d1_bias: str = "—",
    h4_bias: str = "—",
    dxy: str = "—",
    regime: str = "—",
    n_rules: int = 0,
    n_playbooks: int = 0,
) -> None:
    """Log the opening block of a new bot session."""
    global _SESSION_START
    _SESSION_START = datetime.now(GST)

    log_info("=" * 50)
    log_info("=== BOT SESSION STARTED ===")
    log_info(f"Time: {_SESSION_START.strftime('%Y-%m-%d %H:%M GST')} | Session: {session}")
    log_info(f"Gold price: ${gold_price:,.2f}")
    log_info(f"D1 bias: {d1_bias} | H4 bias: {h4_bias}")
    log_info(f"DXY: {dxy} | Regime: {regime}")
    log_info(f"Rules loaded: {n_rules} | Playbooks: {n_playbooks}")
    log_info("=" * 50)


def log_session_end(
    pb_checked: int = 0,
    pb_triggered: int = 0,
    confluence_rejections: int = 0,
    checklist_rejections: int = 0,
    sl_rejections: int = 0,
    conf_rejections: int = 0,
    valid_signals: int = 0,
) -> None:
    """Log the session summary block."""
    log_info("=== SESSION SUMMARY ===")
    log_info(f"Playbooks checked     : {pb_checked}")
    log_info(f"Playbooks triggered   : {pb_triggered}")
    log_info(f"Confluence rejections : {confluence_rejections}")
    log_info(f"Checklist rejections  : {checklist_rejections}")
    log_info(f"SL quality rejections : {sl_rejections}")
    log_info(f"Confidence rejections : {conf_rejections}")
    log_info(f"Valid signals shown   : {valid_signals}")
    log_info("=== END SESSION ===")
    log_info("=" * 50)


# ══════════════════════════════════════════════════════════════════════════════
#  Playbook check logging
# ══════════════════════════════════════════════════════════════════════════════

def log_playbook_check(
    name: str,
    number: int,
    conditions_met: int,
    conditions_total: int,
    conditions: list[dict] | None = None,
    triggered: bool = False,
    direction: str = "",
) -> None:
    """
    Log a detailed playbook condition check.

    Parameters
    ----------
    name             : playbook name
    number           : playbook index (1-based)
    conditions_met   : how many conditions passed
    conditions_total : total conditions checked
    conditions       : list of {"name": str, "passed": bool, "detail": str}
    triggered        : whether the playbook fired
    direction        : "long" or "short"
    """
    dir_str = f" [{direction.upper()}]" if direction else ""
    log_info(f"Checking Playbook {number} — {name}{dir_str}")
    log_info(f"Conditions met: {conditions_met}/{conditions_total}")

    if conditions:
        for cond in conditions:
            passed = cond.get("passed", False)
            cname  = cond.get("name", "condition")
            detail = cond.get("detail", "")
            status = "passed" if passed else "FAILED"
            detail_str = f": {detail}" if detail else ""
            log_info(f"  Condition {status}: {cname}{detail_str}")

    if triggered:
        log_signal(f"→ Playbook {name} triggered ({conditions_met}/{conditions_total} conditions)")
    else:
        log_rejected(f"→ Playbook {name} rejected — only {conditions_met}/{conditions_total} conditions")


# ══════════════════════════════════════════════════════════════════════════════
#  Confluence logging
# ══════════════════════════════════════════════════════════════════════════════

def log_confluence(
    asset: str,
    direction: str,
    checks: list[dict] | None = None,
    total_score: float = 0.0,
    raw_result: dict | None = None,
) -> None:
    """
    Log every confluence check result.

    checks format: [{"name": str, "score": float, "detail": str, "passed": bool}]
    """
    log_info(f"Running confluence for {asset} {direction.upper()}")

    if raw_result:
        # Use engine output directly
        met    = raw_result.get("confluences_met",    [])
        missed = raw_result.get("confluences_failed", [])
        for item in met:
            cname  = item.get("check", "?")
            score  = item.get("score", 0)
            detail = item.get("detail", "")
            log_info(f"  {cname} check: {detail} → +{score:.1f}")
        for item in missed:
            cname  = item.get("check", "?")
            detail = item.get("detail", "")
            log_info(f"  {cname} check: {detail} → +0.0")
        total = raw_result.get("confluence_score", total_score)
    elif checks:
        total = 0.0
        for chk in checks:
            cname  = chk.get("name", "?")
            score  = chk.get("score", 0.0)
            detail = chk.get("detail", "")
            total += score
            log_info(f"  {cname} check: {detail} → +{score:.1f}")
    else:
        total = total_score

    log_info(f"TOTAL CONFLUENCE SCORE: {total:.1f}/10")


# ══════════════════════════════════════════════════════════════════════════════
#  Checklist logging
# ══════════════════════════════════════════════════════════════════════════════

def log_checklist(
    signal_name: str,
    direction: str,
    checklist_result: dict,
) -> None:
    """
    Log every checklist check with pass/fail and detail.

    Parameters
    ----------
    signal_name      : playbook or pattern name
    direction        : "long" or "short"
    checklist_result : dict returned by validate_entry()
    """
    log_info(f"Running entry checklist — {signal_name} {direction.upper()}")

    check_results = checklist_result.get("check_results", {})
    check_names = {
        1: "Trend",
        2: "Confluence",
        3: "RR",
        4: "News",
        5: "Session",
        6: "SL Quality",
    }

    for idx in range(1, 7):
        r = check_results.get(idx)
        if r is None:
            continue
        passed = r.get("passed", False)
        cname  = check_names.get(idx, f"Check {idx}")
        detail = r.get("detail", "").split("\n")[0]  # first line only
        status = "PASSED" if passed else "FAILED"
        log_info(f"  Check {idx} {cname}: {status} — {detail}")

    n_passed = checklist_result.get("checks_passed", "?")
    total    = checklist_result.get("total_checks", 5)
    passed   = checklist_result.get("passed", False)

    if passed:
        log_signal(f"CHECKLIST: {n_passed}/{total} PASSED")
    else:
        reason = checklist_result.get("rejection_reason", "unknown")
        log_rejected(f"SIGNAL REJECTED — checklist failed")
        log_rejected(f"Reason: {reason}")


# ══════════════════════════════════════════════════════════════════════════════
#  Signal rejection logging
# ══════════════════════════════════════════════════════════════════════════════

def log_signal_rejection(
    signal_name: str,
    direction: str,
    reason: str,
    stage: str = "unknown",
    detail: str = "",
) -> None:
    """Log a clean rejection record."""
    log_rejected("SIGNAL REJECTED")
    log_rejected(f"  Signal:    {signal_name}")
    log_rejected(f"  Direction: {direction.upper()}")
    log_rejected(f"  Stage:     {stage}")
    log_rejected(f"  Reason:    {reason}")
    if detail:
        log_rejected(f"  Detail:    {detail}")
    log_rejected(f"  Action:    Signal dropped from output")


# ══════════════════════════════════════════════════════════════════════════════
#  signal_detail.json
# ══════════════════════════════════════════════════════════════════════════════

def _load_signal_log() -> list[dict]:
    if not os.path.exists(SIGNAL_JSON):
        return []
    try:
        with open(SIGNAL_JSON, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_signal_log(records: list[dict]) -> None:
    try:
        with open(SIGNAL_JSON, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
    except Exception:
        pass


def save_signal_detail(
    sig: dict,
    final_status: str,
    session: str = "—",
    gold_price: float = 0.0,
    d1_bias: str = "—",
    h4_bias: str = "—",
    dxy_status: str = "—",
    regime: str = "—",
    checklist_result: dict | None = None,
    confluence_result: dict | None = None,
    rejection_reason: str = "",
    rejection_stage: str = "",
    settings: dict | None = None,
    spread_usd=None,
) -> None:
    """
    Append a full signal record to signal_detail.json.
    Called for EVERY signal — passed or rejected.
    """
    ts    = datetime.now(GST).strftime("%Y-%m-%d %H:%M:%S")
    cfg   = settings or {}
    ck    = checklist_result or {}
    cr    = confluence_result or {}
    ck_checks = ck.get("check_results", {})

    entry  = float(sig.get("entry",     0) or 0)
    sl     = float(sig.get("stop_loss", 0) or 0)
    tp     = float(sig.get("take_profit",0) or 0)
    sl_dist = abs(entry - sl) if entry and sl else 0
    rr      = round(abs(entry - tp) / sl_dist, 2) if sl_dist else 0

    # Position sizing
    balance  = float(cfg.get("balance",  300))
    risk_pct = float(cfg.get("risk_pct",  10))
    leverage = float(cfg.get("leverage",  20))
    risk_usd = round(balance * risk_pct / 100, 2)
    lots     = round(risk_usd / (sl_dist * 100), 2) if sl_dist else 0.01
    reward_usd  = round(risk_usd * 3, 2)
    reward_pct  = round(reward_usd / balance * 100, 1)

    def _ck(idx: int) -> dict:
        r = ck_checks.get(idx, {})
        return {
            "passed": r.get("passed", None),
            "detail": r.get("detail", "—").split("\n")[0],
        }

    sl_q = ck.get("sl_quality", {}) or {}
    ck6  = {
        "passed": sl_q.get("passed", None),
        "detail": (sl_q.get("checks", {}).get("noise", {}).get("detail") or "not run"),
        "adjusted_sl": sl_q.get("adjusted_sl", sl),
    }

    # Confluence breakdown
    conf_met    = cr.get("confluences_met",    [])
    conf_missed = cr.get("confluences_failed", [])
    def _cscore(key: str) -> float:
        for item in conf_met:
            if key.lower() in str(item.get("check", "")).lower():
                return float(item.get("score", 0))
        return 0.0
    def _cpass(key: str) -> bool:
        return _cscore(key) > 0
    def _cdetail(key: str) -> str:
        for item in conf_met + conf_missed:
            if key.lower() in str(item.get("check", "")).lower():
                return str(item.get("detail", ""))
        return ""

    record = {
        "timestamp":   ts,
        "session":     session,
        "gold_price":  gold_price,
        "regime":      regime,
        "d1_bias":     d1_bias,
        "h4_bias":     h4_bias,
        "dxy_status":  dxy_status,

        "signal": {
            "playbook":        sig.get("pattern_name", sig.get("name", "?")),
            "source":          sig.get("source", "rules"),
            "direction":       sig.get("direction", "?"),
            "entry":           entry,
            "sl":              sl,
            "tp1":             tp,
            "tp2":             tp,
            "sl_distance":     round(sl_dist, 2),
            "rr":              rr,
        },

        "confluence": {
            "htf_aligned":     _cpass("HTF"),
            "htf_score":       _cscore("HTF"),
            "htf_detail":      _cdetail("HTF"),
            "order_block":     _cpass("SMC"),
            "ob_score":        _cscore("SMC"),
            "trend_aligned":   _cpass("Trend"),
            "trend_score":     _cscore("Trend"),
            "at_structure":    _cpass("Structure"),
            "structure_score": _cscore("Structure"),
            "rsi_score":       _cscore("Momentum"),
            "dxy_aligned":     _cpass("DXY"),
            "dxy_score":       _cscore("DXY"),
            "candle_pattern":  _cdetail("Candle"),
            "candle_score":    _cscore("Candle"),
            "session_quality": _cpass("Session"),
            "session_score":   _cscore("Session"),
            "total_confidence":float(sig.get("confidence", cr.get("confluence_score", 0))),
        },

        "checklist": {
            "check1_trend":      _ck(1),
            "check2_confluence": _ck(2),
            "check3_rr":         _ck(3),
            "check4_news":       _ck(4),
            "check5_session":    _ck(5),
            "check6_sl":         ck6,
        },

        "position": {
            "lots":       lots,
            "risk_usd":   risk_usd,
            "risk_pct":   risk_pct,
            "reward_usd": reward_usd,
            "reward_pct": reward_pct,
            "leverage":   int(leverage),
        },

        "volume": {
            "ratio":               sig.get("volume", {}).get("volume_ratio",  None),
            "class":               sig.get("volume", {}).get("volume_class",  None),
            "direction_confirmed": sig.get("volume", {}).get("confirmed",     None),
            "confirmation_score":  sig.get("volume", {}).get("confirmation_score", None),
            "climax_detected":     sig.get("volume", {}).get("climax",         False),
            "climax_type":         sig.get("volume", {}).get("climax_type",    None),
            "strategy_optimal":    sig.get("volume", {}).get("strategy_optimal", None),
            "volume_score":        sig.get("volume", {}).get("score",          0.0),
        },

        "spread": {
            "spread_usd": spread_usd,
            "spread_status": (
                "acceptable" if spread_usd and spread_usd <= 1.0
                else "warning" if spread_usd and spread_usd <= 2.0
                else "blocked" if spread_usd and spread_usd > 2.0
                else "unknown"
            ),
        },

        "outcome":          "pending",
        "final_status":     final_status.upper(),
        "rejection_reason": rejection_reason,
        "rejection_stage":  rejection_stage,
    }

    records = _load_signal_log()
    records.append(record)
    _save_signal_log(records)


# ══════════════════════════════════════════════════════════════════════════════
#  Export builder (Part 5)
# ══════════════════════════════════════════════════════════════════════════════

def build_export(
    account_name: str = "Pepperstone #51486884",
    settings: dict | None = None,
) -> tuple[str, str]:
    """
    Build a full session export and write to data/logs/export_*.txt.

    Returns (export_filepath, export_text).
    """
    cfg        = settings or {}
    balance    = cfg.get("balance",    300)
    risk_pct   = cfg.get("risk_pct",   10)
    leverage   = cfg.get("leverage",   20)
    implied_rr = cfg.get("implied_rr", 3)
    now        = datetime.now(GST)
    date_str   = now.strftime("%d %B %Y")
    time_str   = now.strftime("%H:%M GST")
    fname_ts   = now.strftime("%d%b_%H%M")

    SEP  = "═" * 44
    DASH = "─" * 44

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        SEP,
        "TRADINGBOTV1 — SESSION EXPORT",
        f"Date: {date_str}  |  Time: {time_str}",
        f"Account: {account_name}",
        f"Settings: ${balance} | {risk_pct}% risk | {leverage}x | 1:{implied_rr} RR",
        SEP,
        "",
    ]

    # ── bot_activity.log (this session only) ─────────────────────────────────
    lines += ["BOT ACTIVITY LOG", DASH]
    session_start_str = _SESSION_START.strftime("%Y-%m-%d %H:%M:%S GST")
    try:
        with open(ACTIVITY_LOG, encoding="utf-8") as f:
            all_lines = f.readlines()
        # Filter to lines after session start (approximate — include from last "SESSION STARTED")
        session_lines = []
        in_session = False
        for ln in all_lines:
            if "=== BOT SESSION STARTED ===" in ln:
                session_lines = [ln]   # reset — keep only last session
                in_session = True
            elif in_session:
                session_lines.append(ln)
        lines += [ln.rstrip() for ln in session_lines] if session_lines else ["(no activity logged yet)"]
    except FileNotFoundError:
        lines += ["(bot_activity.log not found)"]
    lines += ["", ""]

    # ── signal_detail.json ────────────────────────────────────────────────────
    lines += [SEP, "SIGNALS DETAIL", DASH]
    records = _load_signal_log()
    # Filter to this session (timestamp >= session start)
    sess_start_dt = _SESSION_START
    session_records = []
    for rec in records:
        try:
            rec_dt = datetime.strptime(rec["timestamp"], "%Y-%m-%d %H:%M:%S")
            rec_dt = rec_dt.replace(tzinfo=GST)
            if rec_dt >= sess_start_dt:
                session_records.append(rec)
        except Exception:
            pass

    if session_records:
        for i, rec in enumerate(session_records, 1):
            sig    = rec.get("signal", {})
            ck     = rec.get("checklist", {})
            pos    = rec.get("position", {})
            status = rec.get("final_status", "?")
            lines += [
                f"Signal {i}: {sig.get('playbook','?')} {sig.get('direction','?').upper()}",
                f"  Status    : {status}",
                f"  Entry     : ${sig.get('entry', 0):,.2f}  |  SL: ${sig.get('sl',0):,.2f}  |  TP: ${sig.get('tp1',0):,.2f}",
                f"  SL dist   : ${sig.get('sl_distance',0):,.2f}  |  RR: 1:{sig.get('rr',0)}",
                f"  Lots      : {pos.get('lots', 0.01)}  |  Risk: ${pos.get('risk_usd',0):.2f} ({pos.get('risk_pct',0):.0f}%)",
                f"  Confidence: {rec.get('confluence',{}).get('total_confidence', 0):.1f}/10",
                f"  Checklist : "
                + "  ".join(
                    f"C{k[-1]} {'✓' if (v or {}).get('passed') else '✗'}"
                    for k, v in ck.items()
                    if isinstance(v, dict)
                ),
            ]
            if rec.get("rejection_reason"):
                lines.append(f"  Rejected  : {rec['rejection_reason']}  (stage: {rec.get('rejection_stage','?')})")
            vol = rec.get("volume", {})
            if vol and vol.get("ratio") is not None:
                lines.append(f"  Volume    : {vol.get('ratio','?')}x avg | {vol.get('class','?')}")
                lines.append(f"  Vol Score : {vol.get('volume_score',0):.2f} | Confirmed: {vol.get('confirmation_score','?')}/3 | Climax: {'YES ⚠' if vol.get('climax_detected') else 'None'}")
                lines.append(f"  Optimal   : {vol.get('strategy_optimal','?')}")
            lines.append("")
    else:
        lines += ["(no signals logged this session)", ""]

    # ── errors.log (this session) ─────────────────────────────────────────────
    lines += [SEP, "ERRORS THIS SESSION", DASH]
    try:
        with open(ERROR_LOG, encoding="utf-8") as f:
            err_lines = f.readlines()
        session_errs = []
        for ln in err_lines:
            # Crude filter: keep lines timestamped after session start
            try:
                ts_part = ln[1:20]  # "[2026-05-17 22:14:33"
                ln_dt   = datetime.strptime(ts_part, "%Y-%m-%d %H:%M:%S")
                ln_dt   = ln_dt.replace(tzinfo=GST)
                if ln_dt >= sess_start_dt:
                    session_errs.append(ln.rstrip())
            except Exception:
                session_errs.append(ln.rstrip())  # keep if can't parse

        if session_errs:
            lines += session_errs
        else:
            lines += ["No errors this session. ✓"]
    except FileNotFoundError:
        lines += ["(errors.log not found)"]
    lines += ["", ""]

    # ── Summary ───────────────────────────────────────────────────────────────
    shown   = [r for r in session_records if r.get("final_status") == "SHOWN_TO_USER"]
    rejected= [r for r in session_records if r.get("final_status") == "REJECTED"]
    lines += [
        SEP,
        "SUMMARY",
        DASH,
        f"Signals scanned  : {len(session_records)}",
        f"Shown to user    : {len(shown)}",
        f"Rejected         : {len(rejected)}",
        f"Errors logged    : {len(session_errs) if 'session_errs' in dir() else 0}",
        "",
        "Copy this file and paste to Claude for full analysis and improvements.",
        SEP,
    ]

    export_text = "\n".join(lines)
    filename    = os.path.join(LOG_DIR, f"export_{fname_ts}.txt")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(export_text)
    except Exception as e:
        log_error("debug_logger", "build_export", str(e), "export file not saved")

    return filename, export_text
