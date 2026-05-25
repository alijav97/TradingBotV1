"""
risk/trade_manager.py — Partial TP + trailing SL management for TradingBotV2.

Ported from V1 trade_manager.py.
Fixed: get_current_risk_profile now reads from SQLite journal (not JSON files).
Core math functions (partial TP, trailing SL, RoR) unchanged — verified solid.

Public API:
  calculate_partial_tp_plan(signal, df)  → TP management plan dict
  get_trailing_sl(entry, direction, current_price, atr, tp1_hit, initial_sl) → dict
  calculate_risk_of_ruin(win_rate, risk_pct, rr_ratio, ...) → dict
  get_current_risk_profile(journal) → dict
"""
from __future__ import annotations

import logging
from random import random
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from v2.journal.sqlite_journal import Journal

logger = logging.getLogger(__name__)


# ── Partial TP plan ───────────────────────────────────────────────────────────

def calculate_partial_tp_plan(signal: dict, df: Any) -> dict[str, Any]:
    """
    Build a two-stage TP management plan for a signal.
    TP1 at 1:2 (close 50%), TP2 at 1:3 (close rest with trail).
    """
    try:
        entry      = float(signal.get("entry", 0) or 0)
        sl         = float(signal.get("stop_loss", 0) or 0)
        direction  = str(signal.get("direction", "long")).lower().strip()
        total_lots = float(signal.get("lots", 0.01) or 0.01)
        is_long    = direction in ("long", "buy")

        if entry <= 0 or sl <= 0:
            return _empty_plan(signal)

        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            return _empty_plan(signal)

        atr = _read_atr(df)

        if is_long:
            tp1_price = round(entry + sl_dist * 2.0, 5)
            tp2_price = round(entry + sl_dist * 3.0, 5)
        else:
            tp1_price = round(entry - sl_dist * 2.0, 5)
            tp2_price = round(entry - sl_dist * 3.0, 5)

        tp1_lots = max(0.01, round(total_lots * 0.5, 2))
        tp2_lots = max(0.01, round(total_lots - tp1_lots, 2))

        # Instrument-agnostic profit estimate — caller should override with real pip values
        _factor    = 100.0
        tp1_profit = round(sl_dist * 2.0 * tp1_lots * _factor, 2)
        tp2_profit = round(sl_dist * 3.0 * tp2_lots * _factor, 2)
        total_risk = round(sl_dist * total_lots * _factor, 2)
        trail_step = round(atr * 0.5, 5)

        steps = [
            f"1. Enter {total_lots:.2f} lots at {entry}",
            f"2. Initial SL at {sl}",
            f"3. TP1 {tp1_price}: close {tp1_lots:.2f} lots (+${tp1_profit:,.2f})",
            f"4. Move SL to breakeven ({entry})",
            f"5. Trail remaining {tp2_lots:.2f} lots by 1× ATR ({trail_step})",
            f"6. Final target {tp2_price} or trail until stopped",
        ]

        return {
            "entry":            entry,
            "direction":        direction,
            "total_lots":       total_lots,
            "tp1_price":        tp1_price,
            "tp1_lots":         tp1_lots,
            "tp1_lots_pct":     50,
            "tp1_rr":           2.0,
            "tp1_profit_usd":   tp1_profit,
            "tp1_action":       "Close 50% + move SL to breakeven",
            "tp2_price":        tp2_price,
            "tp2_lots":         tp2_lots,
            "tp2_lots_pct":     50,
            "tp2_rr":           3.0,
            "tp2_profit_usd":   tp2_profit,
            "tp2_action":       "Close remaining 50%",
            "initial_sl":       sl,
            "breakeven_sl":     entry,
            "trail_step_usd":   trail_step,
            "trail_atr_mult":   1.0,
            "sl_distance":      sl_dist,
            "atr":              atr,
            "total_risk_usd":   total_risk,
            "best_case_usd":    round(tp1_profit + tp2_profit, 2),
            "worst_case_usd":   total_risk,
            "management_steps": steps,
            "valid":            True,
        }

    except Exception as exc:
        logger.error("calculate_partial_tp_plan error: %s", exc)
        return _empty_plan(signal)


# ── Trailing SL ───────────────────────────────────────────────────────────────

def get_trailing_sl(
    entry: float,
    direction: str,
    current_price: float,
    atr: float,
    tp1_hit: bool,
    initial_sl: float,
) -> dict[str, Any]:
    """
    Return the current trailing SL price.
    Before TP1: original SL unchanged.
    After TP1: trail 1× ATR behind price, never past breakeven.
    """
    try:
        is_long = direction.lower().strip() in ("long", "buy")

        if not tp1_hit:
            return {"sl": round(initial_sl, 5), "trailing_active": False,
                    "trail_distance": 0.0, "note": "Waiting for TP1 before trailing"}

        if is_long:
            trail_sl = max(current_price - atr, entry)
            return {"sl": round(trail_sl, 5), "trailing_active": True,
                    "trail_distance": round(atr, 5),
                    "note": f"Trailing {current_price} − {atr:.5f} ATR = {trail_sl:.5f}"}
        else:
            trail_sl = min(current_price + atr, entry)
            return {"sl": round(trail_sl, 5), "trailing_active": True,
                    "trail_distance": round(atr, 5),
                    "note": f"Trailing {current_price} + {atr:.5f} ATR = {trail_sl:.5f}"}

    except Exception:
        return {"sl": round(initial_sl, 5), "trailing_active": False,
                "trail_distance": 0.0, "note": "Trail error — using initial SL"}


# ── Risk of Ruin ──────────────────────────────────────────────────────────────

def calculate_risk_of_ruin(
    win_rate: float,
    risk_pct: float,
    rr_ratio: float,
    num_trades: int = 100,
    simulations: int = 1000,
) -> dict[str, Any]:
    """Monte Carlo Risk-of-Ruin analysis."""
    try:
        win_prob         = max(0.0, min(1.0, win_rate / 100.0))
        loss_prob        = 1.0 - win_prob
        risk_per_trade   = max(0.001, risk_pct / 100.0)
        reward_per_trade = risk_per_trade * rr_ratio

        ruin_count      = 0
        final_balances: list[float] = []
        drawdown_list:  list[float] = []

        for _ in range(simulations):
            balance, peak, ruined = 1.0, 1.0, False
            for _ in range(num_trades):
                balance *= (1.0 + reward_per_trade) if random() < win_prob else (1.0 - risk_per_trade)
                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak if peak > 0 else 0.0
                if balance <= 0.1:
                    ruined = True
                    ruin_count += 1
                    break
            final_balances.append(balance)
            drawdown_list.append(drawdown)

        ruin_prob  = ruin_count / simulations * 100.0
        avg_bal    = sum(final_balances) / len(final_balances)
        med_bal    = sorted(final_balances)[len(final_balances) // 2]
        avg_dd     = sum(drawdown_list) / len(drawdown_list) * 100.0
        ev         = (win_prob * reward_per_trade) - (loss_prob * risk_per_trade)

        consec = 0
        tmp    = 1.0
        while tmp > 0.1 and consec < 10_000:
            tmp *= (1.0 - risk_per_trade)
            consec += 1

        if   ruin_prob < 5:  rating, rec = "SAFE",     "Good risk management — continue"
        elif ruin_prob < 15: rating, rec = "MODERATE", "Acceptable risk — monitor drawdown"
        elif ruin_prob < 30: rating, rec = "HIGH",     "Reduce position size or improve win rate"
        else:                rating, rec = "DANGER",   "High ruin risk — reduce risk per trade NOW"

        return {
            "ruin_probability":           round(ruin_prob, 2),
            "risk_rating":                rating,
            "recommendation":             rec,
            "win_rate":                   win_rate,
            "risk_pct":                   risk_pct,
            "rr_ratio":                   rr_ratio,
            "ev_per_trade":               round(ev, 6),
            "ev_positive":                ev > 0,
            "avg_final_balance_pct":      round(avg_bal * 100, 1),
            "median_final_balance_pct":   round(med_bal * 100, 1),
            "avg_max_drawdown_pct":       round(avg_dd, 2),
            "consecutive_losses_to_ruin": consec,
            "simulations_run":            simulations,
            "trades_simulated":           num_trades,
            "summary": (
                f"RoR {ruin_prob:.1f}% [{rating}] | "
                f"EV {'+' if ev >= 0 else ''}{ev:.4f}/trade | "
                f"Avg DD {avg_dd:.1f}%"
            ),
        }

    except Exception as exc:
        return {"ruin_probability": 0.0, "risk_rating": "UNKNOWN",
                "recommendation": str(exc), "ev_per_trade": 0.0,
                "ev_positive": False, "summary": f"Error: {exc}"}


def get_current_risk_profile(journal: "Journal | None" = None) -> dict[str, Any]:
    """
    Build live RoR profile using current settings and SQLite journal data.
    Falls back to 50% win rate if fewer than 5 closed trades available.
    """
    from v2.settings import ACCOUNT_BALANCE, RISK_PER_TRADE_PCT, MIN_RR_RATIO

    win_rate = 50.0
    if journal is not None:
        try:
            stats    = journal.get_stats(days=90)
            if stats.get("trades", 0) >= 5:
                win_rate = float(stats.get("win_rate", 50.0))
        except Exception:
            pass

    ror = calculate_risk_of_ruin(
        win_rate=win_rate,
        risk_pct=RISK_PER_TRADE_PCT,
        rr_ratio=MIN_RR_RATIO,
    )
    ror["current_balance"]    = ACCOUNT_BALANCE
    ror["risk_per_trade_usd"] = round(ACCOUNT_BALANCE * RISK_PER_TRADE_PCT / 100.0, 2)

    # Find safe risk %
    safe = RISK_PER_TRADE_PCT
    while safe > 0.5:
        if calculate_risk_of_ruin(win_rate, safe, MIN_RR_RATIO, 100, 200)["ruin_probability"] < 5.0:
            break
        safe = round(safe - 0.5, 1)
    ror["recommended_risk_pct"] = safe
    ror["recommended_risk_usd"] = round(ACCOUNT_BALANCE * safe / 100.0, 2)
    return ror


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_atr(df: Any) -> float:
    fallback = 20.0
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return fallback
        for col in ("atr", "ATR"):
            if col in df.columns:
                v = float(df[col].iloc[-1])
                return v if v > 0 else fallback
    except Exception:
        pass
    return fallback


def _empty_plan(signal: dict) -> dict[str, Any]:
    return {
        "entry": float(signal.get("entry", 0) or 0),
        "direction": str(signal.get("direction", "long")),
        "total_lots": 0.01, "tp1_price": 0.0, "tp2_price": 0.0,
        "tp1_lots": 0.01, "tp2_lots": 0.01, "initial_sl": float(signal.get("stop_loss", 0) or 0),
        "sl_distance": 0.0, "atr": 20.0, "total_risk_usd": 0.0,
        "best_case_usd": 0.0, "worst_case_usd": 0.0,
        "management_steps": ["(invalid signal)"], "valid": False,
    }
