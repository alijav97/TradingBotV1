"""
mt5_sync.py — MetaTrader 5 live data bridge for TradingBotV1
══════════════════════════════════════════════════════════════
Reads account info, open positions, and closed trade history
directly from the user's local MT5 terminal.

MT5 must be open and logged in for any function to return data.
All functions return safe empty values if MT5 is not available —
the bot continues to run without it.

NOT used for order execution.  Analyst read-only.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any

# ── Optional yfinance import ──────────────────────────────────────────────────
try:
    import yfinance as _yf
    _YF_OK = True
except ImportError:
    _yf = None   # type: ignore[assignment]
    _YF_OK = False

# ── Optional MT5 import ───────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
    _MT5_LIB = True   # alias kept for internal compatibility
except ImportError:
    mt5 = None           # type: ignore[assignment]
    _MT5_AVAILABLE = False
    _MT5_LIB = False
    print("[MT5] MetaTrader5 not available — using yfinance fallback")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
JOURNAL_FILE  = os.path.join(DATA_DIR, "trade_journal.json")
PRICE_CACHE   = os.path.join(DATA_DIR, "price_cache.json")
HIST_CSV      = os.path.join(BASE_DIR, "data", "historical_xauusd.csv")
_GST_TZ       = timezone(timedelta(hours=4))

os.makedirs(DATA_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Connection helpers
# ══════════════════════════════════════════════════════════════════════════════

def connect_mt5() -> tuple[bool, str | None]:
    """
    Attempt to initialise the MT5 terminal connection.
    Tries plain initialize() first, then common install paths.
    Returns (True, None) on success or (False, error_message) on failure.
    """
    if not _MT5_LIB:
        return False, "MetaTrader5 Python package not installed"

    # 1) Try basic init — works when MT5 is already running
    if mt5.initialize():
        info = mt5.account_info()
        if info is not None:
            _check_algo_trading()
            return True, None

    # 2) Try known install paths
    mt5_paths = [
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        r"C:\MT5\terminal64.exe",
        os.path.join(
            os.path.expandvars("%APPDATA%"),
            "MetaQuotes", "Terminal"
        ),
    ]
    for path in mt5_paths:
        if os.path.exists(path):
            try:
                if mt5.initialize(path=path):
                    info = mt5.account_info()
                    if info is not None:
                        _check_algo_trading()
                        return True, None
            except Exception:
                pass

    # 3) Shutdown + sleep + one last attempt
    try:
        mt5.shutdown()
    except Exception:
        pass
    import time
    time.sleep(2)
    if mt5.initialize():
        _check_algo_trading()
        return True, None

    err = mt5.last_error()
    return False, f"MT5 init failed: {err}"


def _check_algo_trading() -> None:
    """Warn (but don't block) if algo trading is disabled in MT5."""
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info and not terminal_info.trade_allowed:
            print(
                "WARNING: Algorithmic trading is disabled in MT5.\n"
                "Go to: Tools → Options → Expert Advisors → "
                "Allow algorithmic trading"
            )
    except Exception:
        pass


def is_connected() -> bool:
    """Quick non-raising check: is MT5 reachable right now?"""
    ok, _ = connect_mt5()
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — Account info
# ══════════════════════════════════════════════════════════════════════════════

def get_account_info() -> tuple[dict[str, Any] | None, str | None]:
    """
    Return (account_dict, None) on success or (None, error_str) on failure.

    Keys: balance, equity, margin_free, profit, account,
          currency, leverage, server, connected
    """
    import time
    connected, err = connect_mt5()
    if not connected:
        # Wait 3 s and retry once
        time.sleep(3)
        connected, err = connect_mt5()
    if not connected:
        return None, err
    try:
        info = mt5.account_info()
        if info is None:
            return None, "No account info returned"
        return {
            "balance":     round(info.balance,     2),
            "equity":      round(info.equity,      2),
            "margin_free": round(info.margin_free, 2),
            "profit":      round(info.profit,      2),
            "account":     info.login,
            "currency":    info.currency,
            "leverage":    info.leverage,
            "server":      getattr(info, "server", "—"),
            "connected":   True,
        }, None
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — Open positions
# ══════════════════════════════════════════════════════════════════════════════

def get_open_positions() -> list[dict[str, Any]]:
    """
    Return all currently open MT5 positions.
    Returns [] if MT5 is offline or no positions exist.
    """
    ok, _ = connect_mt5()
    if not ok:
        return []
    positions = mt5.positions_get()
    if not positions:
        return []

    result = []
    for p in positions:
        direction = "long" if p.type == 0 else "short"
        sl_dist   = abs(p.price_open - p.sl) if p.sl else 0.0
        tp_dist   = abs(p.tp - p.price_open) if p.tp else 0.0
        rr        = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0
        result.append({
            "ticket":        p.ticket,
            "symbol":        p.symbol,
            "direction":     direction,
            "lots":          p.volume,
            "entry":         round(p.price_open,    2),
            "sl":            round(p.sl,            2),
            "tp":            round(p.tp,            2),
            "current_price": round(p.price_current, 2),
            "pnl_usd":       round(p.profit,        2),
            "pnl_pips":      round(p.profit / (p.volume * 100), 2) if p.volume else 0.0,
            "rr_planned":    rr,
            "opened_at":     datetime.fromtimestamp(
                                 p.time, tz=timezone.utc
                             ).strftime("%Y-%m-%d %H:%M UTC"),
            "comment":       p.comment or "",
            "swap":          round(p.swap, 2),
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — Closed trade history
# ══════════════════════════════════════════════════════════════════════════════

def get_closed_trades(days_back: int = 30) -> list[dict[str, Any]]:
    """
    Return closed trades from MT5 history (exit deals only, entry == 1).
    Sorted newest first.  Returns [] if MT5 offline.
    """
    ok, _ = connect_mt5()
    if not ok:
        return []

    date_from = datetime.now(timezone.utc) - timedelta(days=days_back)
    date_to   = datetime.now(timezone.utc)

    deals = mt5.history_deals_get(date_from, date_to)
    if not deals:
        return []

    trades = []
    for d in deals:
        # entry == 1 means this deal is the closing leg
        if d.entry != 1:
            continue
        direction = "long"  if d.type == 0 else "short"
        outcome   = "win"   if d.profit > 0 else ("loss" if d.profit < 0 else "breakeven")
        trades.append({
            "ticket":      d.position_id,
            "symbol":      d.symbol,
            "direction":   direction,
            "lots":        d.volume,
            "entry_price": round(d.price, 2),   # closing price (exit)
            "close_price": round(d.price, 2),
            "pnl_usd":     round(d.profit, 2),
            "outcome":     outcome,
            "closed_at":   datetime.fromtimestamp(
                               d.time, tz=timezone.utc
                           ).strftime("%Y-%m-%d %H:%M"),
            "comment":     d.comment or "",
        })

    return sorted(trades, key=lambda x: x["closed_at"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PART 4 — Today P&L summary
# ══════════════════════════════════════════════════════════════════════════════

def get_today_pnl() -> dict[str, Any]:
    """
    Return today's closed-trade statistics.
    Safe to call even if MT5 offline (returns zeros).
    """
    trades = get_closed_trades(days_back=1)
    today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_trades = [t for t in trades if t["closed_at"].startswith(today)]

    total  = sum(t["pnl_usd"]    for t in today_trades)
    wins   = sum(1 for t in today_trades if t["outcome"] == "win")
    losses = sum(1 for t in today_trades if t["outcome"] == "loss")

    return {
        "pnl":    round(total, 2),
        "trades": len(today_trades),
        "wins":   wins,
        "losses": losses,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PART 5 — Journal sync
# ══════════════════════════════════════════════════════════════════════════════

def _load_journal() -> list[dict]:
    if not os.path.exists(JOURNAL_FILE):
        return []
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_journal(records: list[dict]) -> None:
    with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def sync_to_journal(days_back: int = 30) -> tuple[int, int]:
    """
    Pull MT5 closed-trade history and append new trades to
    data/trade_journal.json.  Skips trades already saved (by ticket).

    Returns (new_trades_added, total_journal_size).
    """
    journal          = _load_journal()
    existing_tickets = {str(t.get("ticket", "")) for t in journal}
    mt5_trades       = get_closed_trades(days_back)
    new_trades: list[dict] = []

    for trade in mt5_trades:
        if str(trade["ticket"]) not in existing_tickets:
            journal.append(trade)
            new_trades.append(trade)

    if new_trades:
        _save_journal(journal)

    return len(new_trades), len(journal)


# ══════════════════════════════════════════════════════════════════════════════
#  PART 5b — Track open trades against bot signals (Brain 1)
# ══════════════════════════════════════════════════════════════════════════════

_AUTO_FILTERS_FILE = os.path.join(DATA_DIR, "auto_filters.json")
_FAILED_TRADE_LOG  = os.path.join(DATA_DIR, "logs", "failed_trade_analysis.json")


def _load_auto_filters() -> list[dict]:
    if not os.path.exists(_AUTO_FILTERS_FILE):
        return []
    try:
        with open(_AUTO_FILTERS_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _save_auto_filters(filters: list[dict]) -> None:
    with open(_AUTO_FILTERS_FILE, "w", encoding="utf-8") as f:
        json.dump(filters, f, indent=2)


def _load_failed_trade_log() -> list[dict]:
    os.makedirs(os.path.dirname(_FAILED_TRADE_LOG), exist_ok=True)
    if not os.path.exists(_FAILED_TRADE_LOG):
        return []
    try:
        with open(_FAILED_TRADE_LOG, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _save_failed_trade_log(records: list[dict]) -> None:
    os.makedirs(os.path.dirname(_FAILED_TRADE_LOG), exist_ok=True)
    with open(_FAILED_TRADE_LOG, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def track_open_trades() -> list[dict]:
    """
    Brain 1 — Step 1: Track open MT5 positions against logged bot signals.

    For each open position:
      - Searches bot_signals_log.json for a matching signal
        (same symbol, direction, entry ±0.5%, within 30 min)
      - If matched: updates signal with status="in_trade",
        current_pnl, distance_to_sl_pct, distance_to_tp_pct,
        bars_open, last_checked (UTC)

    Returns list of currently tracked open trades (dicts with
    signal + position data merged).
    """
    positions = get_open_positions()
    if not positions:
        return []

    signals  = _load_json_list(_SIGNALS_LOG_FILE)
    now_utc  = datetime.now(timezone.utc)
    tracked: list[dict] = []

    for pos in positions:
        pos_entry = float(pos.get("entry", 0) or 0)
        if pos_entry <= 0:
            continue

        for sig in signals:
            if sig.get("symbol", "").upper() != pos["symbol"].upper():
                continue
            if sig.get("direction", "").lower() != pos["direction"].lower():
                continue

            sig_entry = float(sig.get("entry", 0) or 0)
            if sig_entry <= 0:
                continue
            if abs(sig_entry - pos_entry) / sig_entry * 100 > 0.5:
                continue

            try:
                gen_time = datetime.fromisoformat(
                    sig.get("generated_at", "1970-01-01T00:00:00")
                ).replace(tzinfo=timezone.utc)
                if abs((now_utc - gen_time).total_seconds()) > 1800:
                    continue
            except Exception:
                continue

            # ── Match found — enrich signal ──────────────────────────────
            sl  = float(sig.get("stop_loss",   pos_entry) or pos_entry)
            tp  = float(sig.get("take_profit", pos_entry) or pos_entry)
            cur = float(pos.get("current_price", pos_entry) or pos_entry)
            sl_dist = abs(pos_entry - sl)
            tp_dist = abs(tp - pos_entry)
            cur_dist_sl = abs(cur - sl)
            cur_dist_tp = abs(tp - cur)
            dist_sl_pct = round(cur_dist_sl / sl_dist * 100, 1) if sl_dist else 0.0
            dist_tp_pct = round(cur_dist_tp / tp_dist * 100, 1) if tp_dist else 0.0

            try:
                gen_ts   = gen_time.timestamp()
                now_ts   = now_utc.timestamp()
                bars_open = max(0, int((now_ts - gen_ts) / 3600))
            except Exception:
                bars_open = 0

            sig["status"]              = "in_trade"
            sig["current_pnl"]         = pos.get("pnl_usd", 0.0)
            sig["distance_to_sl_pct"]  = dist_sl_pct
            sig["distance_to_tp_pct"]  = dist_tp_pct
            sig["bars_open"]           = bars_open
            sig["last_checked"]        = now_utc.isoformat()

            tracked.append({**sig, **{"mt5_ticket": pos.get("ticket")}})
            break

    if tracked:
        with open(_SIGNALS_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(signals, f, indent=2)

    return tracked


# ══════════════════════════════════════════════════════════════════════════════
#  PART 6 — Auto-match MT5 trades to bot signals → pattern memory
# ══════════════════════════════════════════════════════════════════════════════

_PATTERN_MEMORY_FILE = os.path.join(DATA_DIR, "pattern_memory.json")
_SIGNALS_LOG_FILE    = os.path.join(DATA_DIR, "bot_signals_log.json")


def _load_json_list(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def log_bot_signal(signal: dict) -> None:
    """
    Persist a bot-generated signal so auto_match_and_update() can
    later link it to a real MT5 trade.

    Signal dict must have at least:
      symbol, direction, entry, stop_loss, take_profit,
      pattern_name, regime, session, generated_at (ISO str)
    """
    signals  = _load_json_list(_SIGNALS_LOG_FILE)
    sig_copy = dict(signal)
    if "generated_at" not in sig_copy:
        sig_copy["generated_at"] = datetime.now(timezone.utc).isoformat()
    signals.append(sig_copy)
    with open(_SIGNALS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2)


def auto_match_and_update(days_back: int = 2) -> list[dict]:
    """
    Compare recent MT5 closed trades against logged bot signals.
    A match requires:
      • Same symbol
      • Same direction
      • Entry price within 0.5 %
      • Trade opened within 30 min of signal generation

    For each match:
      • Saves the outcome to data/pattern_memory.json
      • Marks the signal as matched in bot_signals_log.json

    Returns list of notification dicts (one per new match).
    """
    closed   = get_closed_trades(days_back=days_back)
    signals  = _load_json_list(_SIGNALS_LOG_FILE)
    memory   = _load_json_list(_PATTERN_MEMORY_FILE)

    matched_tickets = {str(m.get("mt5_ticket", "")) for m in memory if m.get("mt5_ticket")}
    notifications   : list[dict] = []

    for trade in closed:
        if str(trade["ticket"]) in matched_tickets:
            continue   # already recorded

        try:
            trade_time = datetime.strptime(
                trade["closed_at"], "%Y-%m-%d %H:%M"
            ).replace(tzinfo=timezone.utc)
        except Exception:
            continue

        for sig in signals:
            if sig.get("matched"):
                continue
            if sig.get("symbol", "").upper() != trade["symbol"].upper():
                continue
            if sig.get("direction", "").lower() != trade["direction"].lower():
                continue

            sig_entry = float(sig.get("entry", 0) or 0)
            if sig_entry <= 0:
                continue
            price_diff_pct = abs(sig_entry - trade["entry_price"]) / sig_entry * 100
            if price_diff_pct > 0.5:
                continue

            try:
                gen_time = datetime.fromisoformat(
                    sig.get("generated_at", "1970-01-01T00:00:00")
                ).replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if abs((trade_time - gen_time).total_seconds()) > 1800:   # 30 min
                continue

            # ── It's a match ─────────────────────────────────────────────────
            outcome_rec = {
                "date":         trade["closed_at"],
                "playbook":     sig.get("pattern_name", "unknown"),
                "regime":       sig.get("regime",  "RANGING"),
                "session":      sig.get("session", "London"),
                "direction":    trade["direction"],
                "entry":        sig_entry,
                "sl":           float(sig.get("stop_loss",   0) or 0),
                "tp":           float(sig.get("take_profit", 0) or 0),
                "outcome":      trade["outcome"],
                "pnl_usd":      trade["pnl_usd"],
                "rr_achieved":  None,
                "mt5_ticket":   trade["ticket"],
                "auto_matched": True,
                "notes":        "Auto-matched from MT5",
            }
            sl_dist = abs(sig_entry - float(sig.get("stop_loss", sig_entry) or sig_entry))
            tp_dist = abs(float(sig.get("take_profit", sig_entry) or sig_entry) - sig_entry)
            if sl_dist > 0:
                outcome_rec["rr_achieved"] = round(tp_dist / sl_dist, 2)

            memory.append(outcome_rec)
            sig["matched"] = True

            # ── Wire learning: update rule confidence from this outcome ───────
            try:
                from learning import update_rule_confidence as _update_conf
                _update_conf(
                    outcome_rec["playbook"],
                    outcome_rec["outcome"].upper(),
                )
            except Exception:
                pass  # learning.py unavailable or rule not found — non-fatal

            # ── Brain 2: mark signal as user-traded ──────────────────────────
            try:
                from signal_tracker import mark_user_traded as _mark_traded
                _sig_id = sig.get("signal_id")
                if _sig_id:
                    _mark_traded(_sig_id, traded=True)
            except Exception:
                pass

            # ── Brain 1 — Post-trade automatic analysis ───────────────────────
            if trade["outcome"].lower() == "loss":
                # Step A: run pattern_fatigue analysis and save to log
                try:
                    from pattern_fatigue import analyze_failed_trade as _aft
                    _trade_rec = {
                        **outcome_rec,
                        "symbol":       trade["symbol"],
                        "pnl_usd":      trade["pnl_usd"],
                        "open_time":    trade.get("opened_at", ""),
                        "closed_at":    trade.get("closed_at", ""),
                        "entry":        sig_entry,
                        "stop_loss":    float(sig.get("stop_loss",   0) or 0),
                        "take_profit":  float(sig.get("take_profit", 0) or 0),
                        "pattern_name": sig.get("pattern_name", "unknown"),
                        "regime":       sig.get("regime", ""),
                        "session":      sig.get("session", ""),
                        "volume_ratio": float(sig.get("volume", {}).get("volume_ratio", 0) or 0),
                        "climax_detected": bool(sig.get("volume", {}).get("climax", False)),
                    }
                    _analysis = _aft(_trade_rec)
                    _primary  = _analysis.get("primary_reason", "UNKNOWN")
                    _playbook = outcome_rec["playbook"]

                    # Step B: save to failed_trade_analysis.json
                    _failed_log = _load_failed_trade_log()
                    _failed_log.append({
                        "saved_at":      datetime.now(timezone.utc).isoformat(),
                        "mt5_ticket":    trade["ticket"],
                        "playbook":      _playbook,
                        "primary_reason": _primary,
                        "all_reasons":   _analysis.get("all_reasons", []),
                        "what_to_do_next": _analysis.get("what_to_do_next", ""),
                        "pnl_usd":       trade["pnl_usd"],
                    })
                    _save_failed_trade_log(_failed_log)

                    # Step C: count same-reason failures for this pattern
                    _same = [
                        r for r in _failed_log
                        if r.get("playbook") == _playbook
                        and r.get("primary_reason") == _primary
                    ]
                    if len(_same) >= 3:
                        _afs = _load_auto_filters()
                        _existing = next(
                            (f for f in _afs
                             if f.get("pattern") == _playbook
                             and f.get("filter_reason") == _primary),
                            None,
                        )
                        if _existing:
                            _existing["times_triggered"] = len(_same)
                        else:
                            _afs.append({
                                "pattern":         _playbook,
                                "filter_reason":   _primary,
                                "added_at":        datetime.now(timezone.utc).isoformat(),
                                "times_triggered": len(_same),
                                "action":          "reduce_confidence_by_1",
                            })
                        _save_auto_filters(_afs)
                except Exception:
                    pass  # never block the match on analysis failure

            elif trade["outcome"].lower() == "win":
                # Check if live win rate > 60% over last 10 trades — boost rule
                try:
                    _mem_all  = _load_json_list(_PATTERN_MEMORY_FILE)
                    _pb_name  = outcome_rec["playbook"]
                    _last_10  = [
                        m for m in _mem_all
                        if m.get("playbook") == _pb_name
                    ][-10:]
                    if len(_last_10) >= 5:
                        _lw  = sum(1 for m in _last_10 if m.get("outcome") == "win")
                        _lwr = _lw / len(_last_10)
                        if _lwr > 0.60:
                            from learning import update_rule_confidence as _uc2
                            _uc2(_pb_name, "WIN")  # extra positive signal
                    # Annotate the just-appended memory entry
                    if memory and memory[-1].get("playbook") == _pb_name:
                        memory[-1]["positive_reinforcement"] = True
                except Exception:
                    pass

            pnl_str = f"+${trade['pnl_usd']:.2f}" if trade["pnl_usd"] >= 0 else f"-${abs(trade['pnl_usd']):.2f}"
            notifications.append({
                "symbol":    trade["symbol"],
                "direction": trade["direction"].upper(),
                "outcome":   trade["outcome"].upper(),
                "pnl":       pnl_str,
                "pattern":   sig.get("pattern_name", "?"),
                "close_price": trade["entry_price"],
            })
            break   # each trade matches at most one signal

    if notifications:
        with open(_PATTERN_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
        with open(_SIGNALS_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(signals, f, indent=2)

    return notifications


# ══════════════════════════════════════════════════════════════════════════════
#  PART 7 — Status string (for sidebar)
# ══════════════════════════════════════════════════════════════════════════════

def get_mt5_status_label() -> str:
    """Return a short human-readable MT5 connection status string."""
    if not _MT5_LIB:
        return "⚠️ mt5 package not installed"
    ok, err = connect_mt5()
    if ok:
        info = mt5.account_info()
        if info:
            return f"🟢 Connected — #{info.login}"
    return "🔴 Offline — open MT5 & log in"


# ══════════════════════════════════════════════════════════════════════════════
#  OHLCV data fetch with indicators
# ══════════════════════════════════════════════════════════════════════════════

def get_mt5_data(
    symbol:    str = "XAUUSD",
    timeframe: str = "H1",
    bars:      int = 500,
):
    """
    Fetch OHLCV bars from MT5 and add standard technical indicators.

    Returns a pandas DataFrame indexed by datetime (GST/UTC+4 display),
    or None if MT5 is offline.  Columns added: EMA50, EMA200, RSI, ATR,
    MACD, MACD_signal, BB_mid, BB_upper, BB_lower.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        print("pandas / numpy not installed")
        return None

    try:
        connected, err = connect_mt5()
        if not connected:
            print(f"MT5 not connected: {err}")
            return None

        tf_map = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
            "W1":  mt5.TIMEFRAME_W1,
        }
        tf    = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            print(f"MT5: no data returned for {symbol} {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        # Normalise volume column name
        df.rename(columns={"real_volume": "volume"}, inplace=True)
        if "volume" not in df.columns and "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]

        # ─ EMA 50 / 200 ──────────────────────────────────────────────────────
        df["EMA50"]  = df["close"].ewm(span=50,  adjust=False).mean()
        df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()

        # ─ RSI 14 ────────────────────────────────────────────────────────────
        delta    = df["close"].diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # ─ ATR 14 ────────────────────────────────────────────────────────────
        hl   = df["high"] - df["low"]
        hpc  = (df["high"] - df["close"].shift()).abs()
        lpc  = (df["low"]  - df["close"].shift()).abs()
        tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(com=13, adjust=False).mean()

        # ─ MACD 12/26/9 ──────────────────────────────────────────────────────
        ema12            = df["close"].ewm(span=12, adjust=False).mean()
        ema26            = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"]       = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # ─ Bollinger Bands 20,2 ──────────────────────────────────────────────
        mid              = df["close"].rolling(20).mean()
        std              = df["close"].rolling(20).std()
        df["BB_mid"]     = mid
        df["BB_upper"]   = mid + 2 * std
        df["BB_lower"]   = mid - 2 * std

        return df

    except Exception as exc:
        print(f"get_mt5_data error: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Live price helper
# ══════════════════════════════════════════════════════════════════════════════

def get_live_price(symbol: str = "XAUUSD") -> dict[str, Any]:
    """
    Return a live-price dict with 4-priority fallback:
      1. MT5 symbol_info_tick()         → source = "MT5"         (real broker feed)
      2. yfinance GC=F 1-min bar        → source = "yfinance_1m" (< 300s old)
      3. price_cache.json (< 5 min)     → source = "cache"
      4. historical CSV last row        → source = "CSV_STALE"   (always flagged stale)

    Returned dict keys:
      price, bid, ask, spread, source, timestamp_uae, age_seconds,
      is_live, stale_warning
    """
    def _ts() -> str:
        return datetime.now(_GST_TZ).strftime("%I:%M %p UAE | %A %d %B %Y")

    # ── Priority 1: MT5 live tick (real broker feed) ──────────────────────────
    if _MT5_LIB and mt5 is not None:
        try:
            connected, _err = connect_mt5()
            if connected:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None and tick.ask > 0 and tick.bid > 0:
                    spread = round(tick.ask - tick.bid, 2)
                    return {
                        "price":         round(tick.ask, 2),
                        "bid":           round(tick.bid, 2),
                        "ask":           round(tick.ask, 2),
                        "spread":        spread,
                        "source":        "MT5",
                        "timestamp_uae": _ts(),
                        "age_seconds":   0,
                        "is_live":       True,
                        "stale_warning": "",
                    }
        except Exception:
            pass

    # ── Priority 2: yfinance 1-minute bar (fallback if MT5 unavailable) ───────
    _YF_SYMBOLS_P2: dict = {
        "XAUUSD": "GC=F",  "NAS100": "NQ=F",    "US30":   "YM=F",
        "GBPUSD": "GBPUSD=X", "EURUSD": "EURUSD=X", "WTI":  "CL=F",
    }
    if _YF_OK and _yf is not None:
        try:
            import pandas as _pd_yf
            _yf_sym    = _YF_SYMBOLS_P2.get(symbol, "GC=F")
            ticker_obj = _yf.Ticker(_yf_sym)
            fast_df    = ticker_obj.history(interval="1m", period="1d",
                                            auto_adjust=True)
            if fast_df is not None and not fast_df.empty:
                last_price = float(fast_df["Close"].iloc[-1])
                last_time  = fast_df.index[-1]
                age_s = int((
                    _pd_yf.Timestamp.now(tz="UTC") -
                    last_time.tz_convert("UTC")
                ).total_seconds())
                # Accept any price ≤ 48 h old; mark stale if > 5 min
                _yf_live   = age_s < 300
                _yf_warn   = "" if _yf_live else f"⚠ yfinance price {age_s//60}m old"
                _yf_source = "yfinance_1m" if _yf_live else "yfinance_stale"
                if last_price > 0 and age_s < 172800:   # 48 hours max
                    return {
                        "price":         round(last_price, 2),
                        "bid":           round(last_price - 0.15, 2),
                        "ask":           round(last_price + 0.15, 2),
                        "spread":        0.30,
                        "source":        _yf_source,
                        "timestamp_uae": _ts(),
                        "age_seconds":   age_s,
                        "is_live":       _yf_live,
                        "stale_warning": _yf_warn,
                    }
        except Exception:
            pass

    # ── Priority 3: price_cache.json (< 5 minutes old) ───────────────────────
    try:
        if os.path.exists(PRICE_CACHE):
            with open(PRICE_CACHE, "r", encoding="utf-8") as fh:
                cache = json.load(fh)
            cached_price = float(cache.get("ask") or cache.get("price") or 0)
            cached_ts    = cache.get("ts") or cache.get("timestamp") or ""
            if cached_price > 0 and cached_ts:
                cache_dt    = datetime.fromisoformat(cached_ts)
                if cache_dt.tzinfo is None:
                    cache_dt = cache_dt.replace(tzinfo=_GST_TZ)
                now_aware   = datetime.now(_GST_TZ)
                age_s       = int((now_aware - cache_dt.astimezone(_GST_TZ)
                                   ).total_seconds())
                if age_s < 300:
                    _bid  = float(cache.get("bid") or round(cached_price - 0.15, 2))
                    return {
                        "price":         round(cached_price, 2),
                        "bid":           round(_bid, 2),
                        "ask":           round(cached_price, 2),
                        "spread":        round(cached_price - _bid, 2),
                        "source":        "cache",
                        "timestamp_uae": _ts(),
                        "age_seconds":   age_s,
                        "is_live":       False,
                        "stale_warning": f"Using cached price ({age_s}s old)",
                    }
    except Exception:
        pass

    # ── Priority 4: historical CSV last row (last resort, always stale) ───────
    try:
        if os.path.exists(HIST_CSV):
            import pandas as _pd_csv
            df_csv = _pd_csv.read_csv(HIST_CSV, nrows=None)
            if not df_csv.empty:
                last_price = float(df_csv.iloc[-1]["close"])
                return {
                    "price":         round(last_price, 2),
                    "bid":           round(last_price - 0.15, 2),
                    "ask":           round(last_price + 0.15, 2),
                    "spread":        0.30,
                    "source":        "CSV_STALE",
                    "timestamp_uae": _ts(),
                    "age_seconds":   99999,
                    "is_live":       False,
                    "stale_warning": "⚠ STALE — CSV data, verify on MT5",
                }
    except Exception:
        pass

    # ── Total failure ─────────────────────────────────────────────────────────
    return {
        "price":         0.0,
        "bid":           0.0,
        "ask":           0.0,
        "spread":        0.0,
        "source":        "unavailable",
        "timestamp_uae": _ts(),
        "age_seconds":   99999,
        "is_live":       False,
        "stale_warning": "⚠ No price source available",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-instrument price fetching
# ══════════════════════════════════════════════════════════════════════════════

INSTRUMENT_SYMBOLS: dict = {
    "XAUUSD": "XAUUSD",
    "NAS100": "NAS100",
    "US30":   "US30",
    "GBPUSD": "GBPUSD",
    "EURUSD": "EURUSD",
    "WTI":    "XTIUSD",   # Pepperstone symbol for WTI crude oil
}

_YF_SYMBOLS: dict = {
    "XAUUSD": "GC=F",
    "NAS100": "NQ=F",
    "US30":   "YM=F",
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "WTI":    "CL=F",
}


def get_price_for_instrument(instrument: str = "XAUUSD") -> float:
    """
    Get live price for any instrument.
    Uses yfinance on cloud (Linux), MT5 on Windows.
    Never raises — always returns a float.
    """
    import sys
    IS_WINDOWS = sys.platform.startswith("win")

    _YF_MAP_LOCAL = {
        "XAUUSD": "GC=F",
        "NAS100": "NQ=F",
        "US30":   "YM=F",
        "GBPUSD": "GBPUSD=X",
        "EURUSD": "EURUSD=X",
        "WTI":    "CL=F",
    }

    # ── Priority 1: MT5 live tick (Windows only) ──────────────────────────────
    if IS_WINDOWS:
        try:
            if _MT5_AVAILABLE and mt5 is not None:
                mt5_sym = INSTRUMENT_SYMBOLS.get(instrument, instrument)
                connected, _ = connect_mt5()
                if connected:
                    tick = mt5.symbol_info_tick(mt5_sym)
                    if tick is not None and tick.ask > 0 and tick.bid > 0:
                        return round((tick.bid + tick.ask) / 2, 5)
        except Exception:
            pass

    # ── Priority 2: yfinance (always works on cloud) ──────────────────────────
    try:
        if _YF_OK and _yf is not None:
            yf_sym     = _YF_MAP_LOCAL.get(instrument, "GC=F")
            ticker_obj = _yf.Ticker(yf_sym)
            data       = ticker_obj.history(period="1d", interval="1m",
                                            auto_adjust=True)
            if data is not None and not data.empty:
                price = float(data["Close"].iloc[-1])
                if price > 0:
                    return round(price, 5)
    except Exception as e:
        print(f"[Price] {instrument}: {e}")

    return 0.0


def get_live_price(instrument: str = "XAUUSD") -> float:  # type: ignore[misc]
    """Convenience alias for get_price_for_instrument()."""
    return get_price_for_instrument(instrument)

