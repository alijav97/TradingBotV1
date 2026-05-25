"""
auto_trader.py — TradingBotV1 Automated Paper Trader
─────────────────────────────────────────────────────
Runs a background loop that:
  1. Checks market conditions each minute
  2. Opens paper trades when signal confidence ≥ min_confidence
  3. Monitors open trades for SL/TP hits
  4. Feeds closed trade results back to ML engine
  5. Respects kill-zone windows and daily trade limits

All trades are PAPER trades — no live order execution.
Adaptive ATR-based SL/TP uses live yfinance data at signal time.

Usage:
    python auto_trader.py           # runs loop forever
    python auto_trader.py --once    # single tick then exit
"""

from __future__ import annotations

import json
import os
import time
import sys
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any

# ── Timezone ───────────────────────────────────────────────────────────────────
GST = timezone(timedelta(hours=4))           # UAE / Gulf Standard Time

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
ACTIVITY_FILE = os.path.join(DATA_DIR, "auto_trader_activity.json")
os.makedirs(DATA_DIR, exist_ok=True)

try:
    from data_manager import get_path, save_json, load_json
    _USE_DM = True
except Exception:
    _USE_DM = False

STATE_FILE = (
    get_path("auto_trader_state.json")
    if _USE_DM
    else os.path.join(DATA_DIR, "auto_trader_state.json")
)

# ─────────────────────────────────────────────────────────────────────────────
#  CHANGE 1 — Final INSTRUMENTS config
# ─────────────────────────────────────────────────────────────────────────────
INSTRUMENTS: dict[str, dict] = {
    "XAUUSD": {
        "session_start":      0,
        "session_end":        24,
        "kill_zones":         [(8, 11), (13, 16)],
        "max_trades_per_day": 5,
        "min_confidence":     70,
        "trades_file":        "paper_trades.json",
        # Adaptive ATR settings
        "atr_sl_mult":        1.5,
        "atr_tp_mult":        4.5,
        "atr_period":         14,
        "min_sl_pct":         0.004,
        "max_sl_pct":         0.008,
        "max_hold_bars":      12,
        # Account risk info (for display)
        "avg_account_risk":   "8-10%",
        "avg_account_profit": "24-30%",
        "priority":           1,
        "grade":              "A",
    },
    "WTI": {
        "session_start":      0,
        "session_end":        24,
        "kill_zones":         [(13, 16)],
        "max_trades_per_day": 5,
        "min_confidence":     70,
        "trades_file":        "data/paper_trades_WTI.json",
        "atr_sl_mult":        1.5,
        "atr_tp_mult":        4.5,
        "atr_period":         14,
        "min_sl_pct":         0.006,
        "max_sl_pct":         0.012,
        "max_hold_bars":      24,
        "avg_account_risk":   "9-12%",
        "avg_account_profit": "27-36%",
        "priority":           2,
        "grade":              "B",
    },
    "US30": {
        "session_start":      0,
        "session_end":        24,
        "kill_zones":         [(13, 16)],
        "max_trades_per_day": 5,
        "min_confidence":     70,
        "trades_file":        "data/paper_trades_US30.json",
        "atr_sl_mult":        2.0,
        "atr_tp_mult":        6.0,
        "atr_period":         14,
        "min_sl_pct":         0.004,
        "max_sl_pct":         0.010,
        "max_hold_bars":      24,
        "avg_account_risk":   "9-10%",
        "avg_account_profit": "27-30%",
        "priority":           3,
        "grade":              "B",
    },
    "NAS100": {
        "session_start":      0,
        "session_end":        24,
        "kill_zones":         [(13, 16)],
        "max_trades_per_day": 5,
        "min_confidence":     65,
        "trades_file":        "data/paper_trades_NAS100.json",
        "atr_sl_mult":        1.2,
        "atr_tp_mult":        3.6,
        "atr_period":         14,
        "min_sl_pct":         0.004,
        "max_sl_pct":         0.008,
        "max_hold_bars":      24,
        "avg_account_risk":   "4-8%",
        "avg_account_profit": "12-24%",
        "priority":           4,
        "grade":              "B",
    },
    "GBPUSD": {
        "session_start":      0,
        "session_end":        24,
        "kill_zones":         [(8, 11)],
        "max_trades_per_day": 5,
        "min_confidence":     70,
        "trades_file":        "data/paper_trades_GBPUSD.json",
        "atr_sl_mult":        1.5,
        "atr_tp_mult":        4.5,
        "atr_period":         14,
        "min_sl_pct":         0.002,
        "max_sl_pct":         0.005,
        "max_hold_bars":      48,
        "avg_account_risk":   "3-5%",
        "avg_account_profit": "9-15%",
        "priority":           5,
        "grade":              "C",
    },
    "EURUSD": {
        "session_start":      0,
        "session_end":        24,
        "kill_zones":         [(7, 10)],
        "max_trades_per_day": 5,
        "min_confidence":     70,
        "trades_file":        "data/paper_trades_EURUSD.json",
        "atr_sl_mult":        1.5,
        "atr_tp_mult":        4.5,
        "atr_period":         14,
        "min_sl_pct":         0.002,
        "max_sl_pct":         0.005,
        "max_hold_bars":      48,
        "avg_account_risk":   "3-5%",
        "avg_account_profit": "9-15%",
        "priority":           6,
        "grade":              "C",
    },
}

ACCOUNT_BALANCE: float = 1000.0
LEVERAGE:        int   = 10
RR_RATIO:        float = 3.0
RISK_PER_TRADE:  float = 0.02   # 2% risk per trade

NEWS_BLACKOUT: dict[str, list] = {
    "GBPUSD": [(1, 10, 2), (2, 10, 2)],
    "EURUSD": [(1, 10, 2), (3, 13, 2)],
    "WTI":    [(2, 18, 2)],
    "NAS100": [(4, 14, 2)],
    "US30":   [(4, 14, 2)],
    "XAUUSD": [(4, 14, 2)],
}

# yfinance ticker map
_YF_MAP: dict[str, str] = {
    "XAUUSD": "GC=F",
    "NAS100": "NQ=F",
    "US30":   "YM=F",
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "WTI":    "CL=F",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers — state / trades / logging
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load auto-trader runtime state from disk."""
    try:
        if _USE_DM:
            data = load_json("auto_trader_state.json", default={})
            if data:
                return data
        else:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
    except Exception:
        pass
    return {
        "enabled":      False,
        "running":      False,
        "trades_today": {},
        "last_tick":    None,
        "open_trades":  [],
        "activity_log": [],
        "started_at":   None,
    }


def save_state(state: dict) -> None:
    """Persist state to disk."""
    try:
        if _USE_DM:
            save_json("auto_trader_state.json", state)
        else:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print(f"[AutoTrader] State save error: {e}")


def _trades_path(instr: str) -> str:
    cfg  = INSTRUMENTS[instr]
    path = cfg["trades_file"]
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def load_trades(instr: str) -> list:
    """Load paper trades for an instrument."""
    try:
        filename = os.path.basename(
            INSTRUMENTS[instr]["trades_file"])
        if _USE_DM:
            return load_json(filename, default=[]) or []
        else:
            path = _trades_path(instr)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
    except Exception:
        pass
    return []


def save_trades(instr: str, trades: list) -> None:
    try:
        filename = os.path.basename(
            INSTRUMENTS[instr]["trades_file"])
        if _USE_DM:
            save_json(filename, trades)
        else:
            path = _trades_path(instr)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(trades, f, indent=2, default=str)
    except Exception as e:
        print(f"[Trades] Save error: {e}")


def log_activity(state: dict, message: str, category: str = "INFO") -> None:
    """Append a timestamped entry to the activity log (in-memory + file)."""
    entry = {
        "ts":       datetime.now(GST).strftime("%Y-%m-%d %H:%M:%S"),
        "category": category,
        "message":  message,
    }
    log = state.setdefault("activity_log", [])
    log.append(entry)
    # Keep last 500 entries in-memory
    if len(log) > 500:
        state["activity_log"] = log[-500:]
    # Also append to the persistent activity file
    try:
        existing: list = []
        if os.path.exists(ACTIVITY_FILE):
            with open(ACTIVITY_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.append(entry)
        if len(existing) > 2000:
            existing = existing[-2000:]
        with open(ACTIVITY_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)
    except Exception:
        pass


def in_kill_zone(instr: str) -> bool:
    """Return True if current UAE hour is inside any kill zone."""
    cfg   = INSTRUMENTS.get(instr, {})
    now_h = datetime.now(GST).hour
    for (start, end) in cfg.get("kill_zones", []):
        if start <= now_h < end:
            return True
    return False


def in_session(instr: str) -> bool:
    """Return True if current UAE hour is within the trading session."""
    cfg   = INSTRUMENTS.get(instr, {})
    now_h = datetime.now(GST).hour
    return cfg.get("session_start", 0) <= now_h < cfg.get("session_end", 24)


def in_news_blackout(instr: str) -> bool:
    """Return True if we're within 2 h of a major news event (stub)."""
    # NEWS_BLACKOUT entries are (weekday, hour, buffer_hours)
    # weekday: 0=Mon … 4=Fri
    now    = datetime.now(GST)
    now_wd = now.weekday()
    now_h  = now.hour
    for (wd, h, buf) in NEWS_BLACKOUT.get(instr, []):
        if now_wd == wd and abs(now_h - h) <= buf:
            return True
    return False


def trades_today_count(state: dict, instr: str) -> int:
    today = datetime.now(GST).strftime("%Y-%m-%d")
    key   = f"{instr}_{today}"
    return state.get("trades_today", {}).get(key, 0)


def increment_trades_today(state: dict, instr: str) -> None:
    today = datetime.now(GST).strftime("%Y-%m-%d")
    key   = f"{instr}_{today}"
    td    = state.setdefault("trades_today", {})
    td[key] = td.get(key, 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
#  CHANGE 2 — Adaptive ATR SL/TP calculator
# ─────────────────────────────────────────────────────────────────────────────

def calculate_adaptive_sl_tp(
        instr: str,
        entry_price: float,
        direction: str) -> tuple:
    """
    Calculate adaptive SL/TP based on current ATR at time of signal.
    Returns (sl_price, tp_price, sl_pct, tp_pct).

    Uses live yfinance data (4h for XAUUSD, 1h for others).
    Falls back to midpoint of [min_sl_pct, max_sl_pct] on any error.
    """
    try:
        import yfinance as yf
        import numpy as np

        cfg      = INSTRUMENTS[instr]
        interval = "4h" if instr == "XAUUSD" else "1h"

        tk   = yf.Ticker(_YF_MAP[instr])
        hist = tk.history(period="5d", interval=interval)

        if hist.empty or len(hist) < 14:
            # Fallback to midpoint
            sl_pct = (cfg["min_sl_pct"] + cfg["max_sl_pct"]) / 2
        else:
            atr_period = cfg.get("atr_period", 14)
            highs      = hist["High"].values
            lows       = hist["Low"].values
            closes     = hist["Close"].values

            atr_vals: list[float] = []
            for i in range(1, len(hist)):
                tr = max(
                    highs[i]  - lows[i],
                    abs(highs[i]  - closes[i - 1]),
                    abs(lows[i]   - closes[i - 1]),
                )
                atr_vals.append(tr)

            if len(atr_vals) >= atr_period:
                current_atr = float(np.mean(atr_vals[-atr_period:]))
            else:
                current_atr = float(np.mean(atr_vals))

            # ATR as % of price
            atr_pct = current_atr / entry_price

            # Adaptive SL — clamp to [min, max]
            sl_pct = atr_pct * cfg["atr_sl_mult"]
            sl_pct = max(cfg["min_sl_pct"], min(cfg["max_sl_pct"], sl_pct))

        # TP = 3× SL (strict 1:3 RR)
        tp_pct = sl_pct * RR_RATIO

        # Calculate prices
        if direction.upper() == "LONG":
            sl_price = round(entry_price * (1 - sl_pct), 5)
            tp_price = round(entry_price * (1 + tp_pct), 5)
        else:
            sl_price = round(entry_price * (1 + sl_pct), 5)
            tp_price = round(entry_price * (1 - tp_pct), 5)

        return (sl_price, tp_price, sl_pct, tp_pct)

    except Exception as e:
        # Fallback to midpoint
        cfg    = INSTRUMENTS[instr]
        sl_pct = (cfg["min_sl_pct"] + cfg["max_sl_pct"]) / 2
        tp_pct = sl_pct * RR_RATIO
        if direction.upper() == "LONG":
            sl_price = round(entry_price * (1 - sl_pct), 5)
            tp_price = round(entry_price * (1 + tp_pct), 5)
        else:
            sl_price = round(entry_price * (1 + sl_pct), 5)
            tp_price = round(entry_price * (1 - tp_pct), 5)
        return (sl_price, tp_price, sl_pct, tp_pct)


# ─────────────────────────────────────────────────────────────────────────────
#  CHANGE 3 — open_trade() with adaptive ATR SL/TP
# ─────────────────────────────────────────────────────────────────────────────

def open_trade(
        state:     dict,
        instr:     str,
        direction: str,
        price:     float,
        conf:      float,
        strategy:  str = "AUTO",
        reason:    str = "") -> dict | None:
    """
    Open a paper trade with adaptive ATR-based SL/TP.
    Returns the trade dict, or None if blocked.
    """
    cfg       = INSTRUMENTS[instr]

    # ── ONE TRADE AT A TIME PER INSTRUMENT ──────────────────────────────────
    _all_trades = load_trades(instr)
    if any(t.get("status") == "OPEN" for t in _all_trades):
        log_activity(state,
                     f"{instr}: trade already open, waiting for close",
                     "BLOCKED")
        return None
    # ───────────────────────────────────────────────────────────────────────

    # ── Daily COMPLETED trade limit (counts closed trades, not opens) ──────────
    trade_num = trades_today_count(state, instr) + 1

    if trade_num > cfg["max_trades_per_day"]:
        log_activity(state,
                     f"⛔ {instr} max trades/day reached ({cfg['max_trades_per_day']})",
                     "BLOCKED")
        return None

    if not in_session(instr):
        log_activity(state, f"⛔ {instr} outside session hours", "BLOCKED")
        return None

    if in_news_blackout(instr):
        log_activity(state, f"⛔ {instr} news blackout active", "BLOCKED")
        return None

    if conf < cfg["min_confidence"]:
        log_activity(state,
                     f"⛔ {instr} confidence {conf:.0f}% < {cfg['min_confidence']}%",
                     "BLOCKED")
        return None

    # ── CHANGE 3 core: adaptive ATR-based SL/TP ──────────────────────────────
    sl, tp, sl_pct, tp_pct = calculate_adaptive_sl_tp(instr, price, direction)

    # Calculate account impact
    account_risk_pct   = round(sl_pct   * LEVERAGE * 100, 1)
    account_profit_pct = round(tp_pct   * LEVERAGE * 100, 1)

    now = datetime.now(GST)
    trade: dict[str, Any] = {
        "id":                  f"{instr}_{now.strftime('%Y%m%d%H%M%S')}",
        "instrument":          instr,
        "direction":           direction.upper(),
        "entry_price":         price,
        "sl":                  sl,
        "tp":                  tp,
        "sl_pct":              sl_pct,
        "tp_pct":              tp_pct,
        "account_risk_pct":    account_risk_pct,
        "account_profit_pct":  account_profit_pct,
        "adaptive_atr_sl":     True,
        "confidence":          conf,
        "strategy":            strategy,
        "reason":              reason,
        "in_kill_zone":        in_kill_zone(instr),
        "status":              "OPEN",
        "opened_at":           now.isoformat(),
        "closed_at":           None,
        "outcome":             None,
        "pnl_pct":             0.0,
        "auto_trade":          True,
        "grade":               cfg.get("grade", "?"),
        "priority":            cfg.get("priority", 0),
    }

    # Persist
    trades = load_trades(instr)
    trades.append(trade)
    save_trades(instr, trades)

    # Note: increment_trades_today is called when this trade CLOSES (in close_trade)
    # so max_trades_per_day counts completed trades, not simultaneous opens.

    # Add to open-trades watch list
    state.setdefault("open_trades", []).append({
        "id":         trade["id"],
        "instr":      instr,
        "sl":         sl,
        "tp":         tp,
        "direction":  direction.upper(),
        "entry":      price,
        "opened_at":  now.isoformat(),
        "max_hold_bars": cfg.get("max_hold_bars", 24),
        "bars_held":  0,
    })

    # ── CHANGE 3 log with account impact ────────────────────────────────────
    log_activity(
        state,
        f"✅ OPENED | "
        f"{direction.upper()} {instr} @ {price} | "
        f"SL:{sl} (-{account_risk_pct}% acc) | "
        f"TP:{tp} (+{account_profit_pct}% acc) | "
        f"Conf:{conf:.0f}% | "
        f"{'🎯 KZ' if in_kill_zone(instr) else '⚠️'}",
        "TRADE",
    )

    save_state(state)
    return trade


def close_trade(
        state:   dict,
        trade_id: str,
        outcome: str,
        close_price: float) -> None:
    """Close an open paper trade and record outcome."""
    now = datetime.now(GST)

    # Find in open_trades watch list
    state["open_trades"] = [
        t for t in state.get("open_trades", [])
        if t.get("id") != trade_id
    ]

    # Update trades file — find the trade across all instruments
    for instr in INSTRUMENTS:
        trades = load_trades(instr)
        updated = False
        for t in trades:
            if t.get("id") == trade_id and t.get("status") == "OPEN":
                t["status"]     = "CLOSED"
                t["outcome"]    = outcome
                t["closed_at"]  = now.isoformat()
                t["close_price"] = close_price
                if outcome == "TP_HIT":
                    t["pnl_pct"] = round(t.get("tp_pct", 0) * 100, 2)
                elif outcome == "SL_HIT":
                    t["pnl_pct"] = -round(t.get("sl_pct", 0) * 100, 2)
                else:
                    t["pnl_pct"] = 0.0
                updated = True
                log_activity(
                    state,
                    f"🔒 CLOSED {t['direction']} {instr} @ {close_price} | "
                    f"{outcome} | P&L: {t['pnl_pct']:+.2f}% on trade",
                    "TRADE",
                )
                # Count this completed trade toward the daily limit
                increment_trades_today(state, instr)
                break
        if updated:
            save_trades(instr, trades)
            break

    save_state(state)


def monitor_open_trades(state: dict) -> None:
    """
    Check each open paper trade against a fresh price.
    Closes trades that have hit SL, TP, or max_hold_bars.
    """
    open_list = state.get("open_trades", [])
    if not open_list:
        return

    try:
        import yfinance as yf
    except ImportError:
        return

    for ot in list(open_list):
        instr = ot["instr"]
        try:
            tk   = yf.Ticker(_YF_MAP[instr])
            hist = tk.history(period="1d", interval="1m")
            if hist.empty:
                continue
            price = float(hist["Close"].iloc[-1])
        except Exception:
            continue

        ot["bars_held"] = ot.get("bars_held", 0) + 1
        direction = ot["direction"]
        sl, tp    = ot["sl"], ot["tp"]

        outcome = None
        if direction == "LONG":
            if price <= sl:
                outcome = "SL_HIT"
            elif price >= tp:
                outcome = "TP_HIT"
        else:
            if price >= sl:
                outcome = "SL_HIT"
            elif price <= tp:
                outcome = "TP_HIT"

        if outcome is None and ot["bars_held"] >= ot.get("max_hold_bars", 24):
            outcome = "TIMEOUT"

        if outcome:
            close_trade(state, ot["id"], outcome, price)


# ─────────────────────────────────────────────────────────────────────────────
#  CHANGE 4 — feed_ml() with priority/grade logging
# ─────────────────────────────────────────────────────────────────────────────

def feed_ml(state: dict) -> None:
    """
    Feed closed paper trades to the ML engine.
    Logs instrument priority and grade for each batch.
    """
    try:
        from learning import update_pattern_memory as _upm
        _ML_AVAILABLE = True
    except ImportError:
        _ML_AVAILABLE = False

    for instr in sorted(INSTRUMENTS, key=lambda x: INSTRUMENTS[x].get("priority", 99)):
        grade    = INSTRUMENTS[instr].get("grade", "?")
        priority = INSTRUMENTS[instr].get("priority", 0)

        trades = load_trades(instr)
        closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("auto_trade")]
        total  = len(closed)

        if total == 0:
            continue

        if _ML_AVAILABLE:
            try:
                for t in closed:
                    _upm(
                        pattern_name=t.get("strategy", "AUTO"),
                        outcome="WIN" if t.get("outcome") == "TP_HIT" else "LOSS",
                        instrument=instr,
                    )
            except Exception as e:
                log_activity(state, f"⚠️ ML feed error for {instr}: {e}", "ML")

        # ── CHANGE 4 log ──────────────────────────────────────────────────────
        log_activity(
            state,
            f"🧠 ML learning: {instr} "
            f"(Priority {priority}, Grade {grade}) | "
            f"{total} closed trades",
            "ML",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  CHANGE 5 — get_daily_summary()
# ─────────────────────────────────────────────────────────────────────────────

def get_daily_summary() -> dict:
    """
    Generate daily P&L summary across all instruments.
    Returns a dict with per-instrument stats and totals.
    """
    try:
        state = load_state()
        today = datetime.now(GST).strftime("%Y-%m-%d")

        summary: dict[str, Any] = {
            "date":               today,
            "instruments":        {},
            "total_trades_today": 0,
            "total_wins":         0,
            "total_losses":       0,
            "total_pnl_pct":      0.0,
            "best_trade":         None,
            "worst_trade":        None,
        }

        best_pnl  = float("-inf")
        worst_pnl = float("inf")

        for instr in INSTRUMENTS:
            try:
                trades = load_trades(instr)
                today_closed = [
                    t for t in trades
                    if t.get("auto_trade")
                    and t.get("status") == "CLOSED"
                    and today in str(t.get("closed_at", ""))
                ]

                wins   = [t for t in today_closed if t.get("outcome") == "TP_HIT"]
                losses = [t for t in today_closed if t.get("outcome") == "SL_HIT"]

                instr_pnl = sum(t.get("pnl_pct", 0) for t in today_closed)

                summary["instruments"][instr] = {
                    "trades":  len(today_closed),
                    "wins":    len(wins),
                    "losses":  len(losses),
                    "pnl_pct": round(instr_pnl, 2),
                    "grade":   INSTRUMENTS[instr].get("grade", "?"),
                    "priority": INSTRUMENTS[instr].get("priority", 0),
                }

                summary["total_trades_today"] += len(today_closed)
                summary["total_wins"]         += len(wins)
                summary["total_losses"]        += len(losses)
                summary["total_pnl_pct"]       += instr_pnl

                # Track best / worst individual trade
                for t in today_closed:
                    pnl = t.get("pnl_pct", 0)
                    if pnl > best_pnl:
                        best_pnl = pnl
                        summary["best_trade"] = {
                            "instr":     instr,
                            "direction": t.get("direction"),
                            "outcome":   t.get("outcome"),
                            "pnl_pct":  round(pnl, 2),
                        }
                    if pnl < worst_pnl:
                        worst_pnl = pnl
                        summary["worst_trade"] = {
                            "instr":     instr,
                            "direction": t.get("direction"),
                            "outcome":   t.get("outcome"),
                            "pnl_pct":  round(pnl, 2),
                        }

            except Exception:
                continue

        summary["total_pnl_pct"] = round(summary["total_pnl_pct"], 2)
        return summary

    except Exception as e:
        return {"error": str(e)}


def get_at_status() -> dict:
    """Return a lightweight auto-trader status dict for the chat UI."""
    state   = load_state()
    today   = datetime.now(GST).strftime("%Y-%m-%d")
    running = state.get("running", False)
    enabled = state.get("enabled", False)

    trades_counts = {}
    for instr in INSTRUMENTS:
        key = f"{instr}_{today}"
        trades_counts[instr] = state.get("trades_today", {}).get(key, 0)

    return {
        "enabled":       enabled,
        "running":       running,
        "started_at":    state.get("started_at"),
        "last_tick":     state.get("last_tick"),
        "open_trades":   len(state.get("open_trades", [])),
        "trades_today":  trades_counts,
        "activity_last": state.get("activity_log", [{}])[-1] if state.get("activity_log") else {},
    }


def get_status() -> dict:
    """Alias for get_at_status() — used by sidebar."""
    state = load_state()
    today = datetime.now(GST).strftime("%Y-%m-%d")

    instr_data = {}
    total_pnl  = 0.0
    total_trades = 0
    for instr in INSTRUMENTS:
        key   = f"{instr}_{today}"
        done  = state.get("trades_today", {}).get(key, 0)
        trades = load_trades(instr)
        open_t = any(
            t.get("status") == "OPEN" and t.get("auto_trade")
            for t in trades
        )
        today_closed = [
            t for t in trades
            if t.get("auto_trade") and t.get("status") == "CLOSED"
            and today in str(t.get("closed_at", ""))
        ]
        wins_today   = [t for t in today_closed if t.get("outcome") == "TP_HIT"]
        losses_today = [t for t in today_closed if t.get("outcome") == "SL_HIT"]
        pnl = sum(t.get("pnl_pct", 0) for t in today_closed)
        total_pnl    += pnl
        total_trades += done
        instr_data[instr] = {
            "trades_today":  done,
            "has_open":      open_t,
            "daily_pnl":     round(pnl, 2),
            "wins_today":    len(wins_today),
            "losses_today":  len(losses_today),
        }

    last_scan = state.get("last_tick")
    if last_scan:
        try:
            _dt = datetime.fromisoformat(last_scan)
            last_scan = _dt.strftime("%H:%M GST")
        except Exception:
            pass

    return {
        "enabled":       state.get("enabled", False),
        "running":       state.get("running", False),
        "total_trades":  total_trades,
        "daily_pnl":     round(total_pnl, 2),
        "instruments":   instr_data,
        "last_scan":     last_scan or "Never",
        "open_trades":   len(state.get("open_trades", [])),
    }


def start_auto_trader() -> None:
    """Enable the auto trader (sets enabled=True in state)."""
    state = load_state()
    state["enabled"]    = True
    state["started_at"] = datetime.now(GST).isoformat()
    log_activity(state, "▶️ Auto trader STARTED via UI", "SYSTEM")
    save_state(state)


def stop_auto_trader() -> None:
    """Disable the auto trader (sets enabled=False in state)."""
    state = load_state()
    state["enabled"] = False
    state["running"] = False
    log_activity(state, "⏹️ Auto trader STOPPED via UI", "SYSTEM")
    save_state(state)


# ─────────────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_ohlcv(instr: str):
    """Fetch recent OHLCV data via yfinance for signal evaluation."""
    try:
        import yfinance as yf
        import pandas as pd
        ticker   = _YF_MAP[instr]
        interval = "1h" if instr not in ("XAUUSD",) else "1h"
        hist = yf.Ticker(ticker).history(period="30d", interval=interval)
        if hist.empty or len(hist) < 50:
            return None
        hist.columns = [c.lower() for c in hist.columns]
        return hist
    except Exception as e:
        return None


def _tick(state: dict) -> None:
    """One evaluation cycle across all instruments — scans signals and opens trades."""
    state["last_tick"] = datetime.now(GST).isoformat()

    # ── 1. Monitor open positions first ──────────────────────────────────────
    monitor_open_trades(state)

    # ── 2. Scan each instrument for new signals ───────────────────────────────
    try:
        from strategy_playbooks import get_active_playbooks, INSTRUMENT_PRIMARY
        _pb_available = True
    except ImportError:
        _pb_available = False

    for instr in sorted(INSTRUMENTS, key=lambda x: INSTRUMENTS[x].get("priority", 99)):
        cfg = INSTRUMENTS[instr]

        # Skip if already have an open trade for this instrument
        _existing = load_trades(instr)
        if any(t.get("status") == "OPEN" for t in _existing):
            log_activity(state, f"⏭️ {instr}: skip scan — trade already open", "SCAN")
            continue

        # Skip if daily limit reached
        if trades_today_count(state, instr) >= cfg["max_trades_per_day"]:
            log_activity(state, f"⏭️ {instr}: skip scan — daily limit reached", "SCAN")
            continue

        # Fetch OHLCV data
        df = _fetch_ohlcv(instr)
        if df is None:
            log_activity(state, f"⚠️ {instr}: no OHLCV data, skipping", "SCAN")
            continue

        # Get live price
        try:
            live_price = float(df["close"].iloc[-1])
        except Exception:
            log_activity(state, f"⚠️ {instr}: could not read price, skipping", "SCAN")
            continue

        # Run playbook signals
        signals: list[dict] = []
        if _pb_available:
            try:
                signals = get_active_playbooks(
                    df,
                    news_sentiment={},
                    top_n=3,
                    instrument=instr,
                )
            except Exception as e:
                log_activity(state, f"⚠️ {instr}: playbook error — {e}", "SCAN")

        # Log what we see
        if signals:
            best = signals[0]
            pb_name  = best["playbook"].get("name", "?")
            pb_score = best["score"]
            pb_dir   = best["direction"]
            # Score is 0-10; convert to 0-100 for confidence threshold comparison
            conf_pct = pb_score * 10
            log_activity(
                state,
                f"🔍 {instr} scan | best={pb_name} | score={pb_score}/10 "
                f"({conf_pct:.0f}%) | dir={pb_dir} | price={live_price}",
                "SCAN",
            )
        else:
            log_activity(state, f"🔍 {instr}: no signals this cycle", "SCAN")
            continue

        # Evaluate top signal
        best     = signals[0]
        pb_score = best["score"]
        pb_dir   = best["direction"]
        conf_pct = pb_score * 10   # scale 0-10 → 0-100 to match min_confidence
        pb_name  = best["playbook"].get("id", best["playbook"].get("name", "AUTO"))

        # Check conditions_met (checklist >= 3/5)
        conds_met   = best.get("conditions_met", 0)
        total_conds = best.get("total_conditions", 1)
        checklist_ok = conds_met >= 3

        # Threshold: confidence >= 70 AND checklist >= 3
        if conf_pct >= cfg["min_confidence"] and checklist_ok:
            log_activity(
                state,
                f"✅ {instr}: signal QUALIFIES — conf={conf_pct:.0f}% "
                f"checklist={conds_met}/{total_conds} | opening trade",
                "SIGNAL",
            )
            open_trade(
                state     = state,
                instr     = instr,
                direction = pb_dir.upper(),
                price     = live_price,
                conf      = conf_pct,
                strategy  = pb_name,
                reason    = f"Playbook score {pb_score}/10, {conds_met}/{total_conds} conditions",
            )
        else:
            log_activity(
                state,
                f"⛔ {instr}: signal below threshold — "
                f"conf={conf_pct:.0f}% (need {cfg['min_confidence']}%) "
                f"checklist={conds_met}/{total_conds} (need >=3)",
                "SCAN",
            )

    save_state(state)



def run_loop(once: bool = False) -> None:
    """Main auto-trader loop. Runs every 60 s unless once=True."""
    state = load_state()
    state["running"]    = True
    state["enabled"]    = True
    state["started_at"] = datetime.now(GST).isoformat()
    save_state(state)

    log_activity(state,
                 f"🚀 Auto trader started | "
                 f"Instruments: {list(INSTRUMENTS.keys())} | "
                 f"Account: ${ACCOUNT_BALANCE} × {LEVERAGE}x",
                 "SYSTEM")

    try:
        while True:
            _tick(state)
            if once:
                break
            time.sleep(60)
    except KeyboardInterrupt:
        log_activity(state, "🛑 Auto trader stopped by user", "SYSTEM")
    finally:
        state["running"] = False
        save_state(state)


if __name__ == "__main__":
    once = "--once" in sys.argv
    run_loop(once=once)
