"""
trade_manager.py
════════════════
Partial TP + trailing SL management for TradingBotV1.

  - calculate_partial_tp_plan(signal, df)  → full TP management plan dict
  - get_trailing_sl(entry, direction, current_price, atr, tp1_hit, initial_sl) → dict
  - format_trade_instructions(plan)        → MT5-ready instruction string
  - calculate_risk_of_ruin(...)            → Monte Carlo RoR analysis dict
  - get_current_risk_profile()             → live risk profile from settings + history
  - format_ror_report(ror)                 → formatted risk-of-ruin report string
"""

from __future__ import annotations

import json
import os
from random import random
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def calculate_partial_tp_plan(signal: dict, df: Any) -> dict[str, Any]:
    """
    Build a full two-stage TP management plan for a signal.

    Parameters
    ----------
    signal : dict — must contain 'entry', 'stop_loss', 'direction';
                    optionally 'lots'
    df     : OHLCV DataFrame — used to read the latest ATR value

    Returns
    -------
    Flat dict with entry/TP/SL/risk details + management_steps list.
    Falls back gracefully on any bad input.
    """
    try:
        # ── Extract signal fields ──────────────────────────────────────────
        entry     = float(signal.get("entry",     0) or 0)
        sl        = float(signal.get("stop_loss", 0) or 0)
        direction = str(signal.get("direction",   "long")).lower().strip()
        total_lots = float(signal.get("lots", 0.01) or 0.01)
        is_long    = direction in ("long", "buy")

        if entry <= 0 or sl <= 0:
            return _empty_plan(signal)

        sl_distance = abs(entry - sl)
        if sl_distance <= 0:
            return _empty_plan(signal)

        # ── ATR from DataFrame ─────────────────────────────────────────────
        atr = _read_atr(df)

        # ── TP levels ─────────────────────────────────────────────────────
        if is_long:
            tp1_price = round(entry + sl_distance * 2.0, 2)
            tp2_price = round(entry + sl_distance * 3.0, 2)
        else:
            tp1_price = round(entry - sl_distance * 2.0, 2)
            tp2_price = round(entry - sl_distance * 3.0, 2)

        # ── Lot split ─────────────────────────────────────────────────────
        tp1_lots = max(0.01, round(total_lots * 0.5, 2))
        tp2_lots = max(0.01, round(total_lots - tp1_lots, 2))
        # Re-enforce total (rounding may drift by 0.01)
        if round(tp1_lots + tp2_lots, 2) != round(total_lots, 2):
            tp2_lots = max(0.01, round(total_lots - tp1_lots, 2))

        # ── Profit estimates (XAU: 1 lot = $100/point on standard account) ─
        _lot_factor = 100.0          # $100 per lot per $1 move on XAUUSD
        tp1_profit  = round(sl_distance * 2.0 * tp1_lots * _lot_factor, 2)
        tp2_profit  = round(sl_distance * 3.0 * tp2_lots * _lot_factor, 2)
        total_risk  = round(sl_distance       * total_lots * _lot_factor, 2)

        # ── Trail step ────────────────────────────────────────────────────
        trail_step = round(atr * 0.5, 2)

        # ── Risk/reward summary ───────────────────────────────────────────
        best_case  = round(tp1_profit + tp2_profit, 2)
        worst_case = total_risk    # signed as positive loss amount

        # ── Management steps (human-readable) ────────────────────────────
        steps = [
            f"1. Enter {total_lots:.2f} lots at ${entry:,.2f}",
            f"2. Initial SL at ${sl:,.2f}",
            f"3. When price hits ${tp1_price:,.2f}: close {tp1_lots:.2f} lots (+${tp1_profit:,.2f})",
            f"4. Move SL to breakeven (${entry:,.2f})",
            f"5. Trail remaining {tp2_lots:.2f} lots with 1× ATR (${trail_step:.2f} steps)",
            f"6. Final target ${tp2_price:,.2f} or trail until stopped",
        ]

        return {
            # Core trade info
            "entry":            entry,
            "direction":        direction,
            "total_lots":       total_lots,
            # TP1
            "tp1_price":        tp1_price,
            "tp1_lots":         tp1_lots,
            "tp1_lots_pct":     50,
            "tp1_rr":           2.0,
            "tp1_profit_usd":   tp1_profit,
            "tp1_action":       "Close 50% + move SL to breakeven",
            # TP2
            "tp2_price":        tp2_price,
            "tp2_lots":         tp2_lots,
            "tp2_lots_pct":     50,
            "tp2_rr":           3.0,
            "tp2_profit_usd":   tp2_profit,
            "tp2_action":       "Close remaining 50%",
            # SL management
            "initial_sl":       sl,
            "breakeven_sl":     entry,
            "trail_step_usd":   trail_step,
            "trail_atr_mult":   1.0,
            "sl_distance":      sl_distance,
            "atr":              atr,
            # Risk/reward
            "total_risk_usd":   worst_case,
            "total_reward_tp1": tp1_profit,
            "total_reward_tp2": tp2_profit,
            "best_case_usd":    best_case,
            "worst_case_usd":   worst_case,
            # Steps
            "management_steps": steps,
            "valid":            True,
        }

    except Exception:
        return _empty_plan(signal)


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

    Called every 60 s (or on tick) while a trade is open.
    Before TP1 is hit the original SL is returned unchanged.
    After TP1 is hit the SL trails 1× ATR behind price, but
    never crosses breakeven (entry).

    Returns
    -------
    {
      "sl":              float,
      "trailing_active": bool,
      "trail_distance":  float,   # 0 when not active
      "note":            str,
    }
    """
    try:
        direction = str(direction).lower().strip()
        is_long   = direction in ("long", "buy")

        if not tp1_hit:
            return {
                "sl":              round(initial_sl, 2),
                "trailing_active": False,
                "trail_distance":  0.0,
                "note":            "Waiting for TP1 before trailing",
            }

        if is_long:
            trail_sl = current_price - atr * 1.0
            trail_sl = max(trail_sl, entry)          # never below breakeven
            return {
                "sl":              round(trail_sl, 2),
                "trailing_active": True,
                "trail_distance":  round(atr * 1.0, 2),
                "note": (
                    f"Trailing: ${current_price:,.2f} − ${atr:.2f} ATR "
                    f"= ${trail_sl:,.2f}"
                ),
            }
        else:
            trail_sl = current_price + atr * 1.0
            trail_sl = min(trail_sl, entry)          # never above breakeven
            return {
                "sl":              round(trail_sl, 2),
                "trailing_active": True,
                "trail_distance":  round(atr * 1.0, 2),
                "note": (
                    f"Trailing: ${current_price:,.2f} + ${atr:.2f} ATR "
                    f"= ${trail_sl:,.2f}"
                ),
            }

    except Exception:
        return {
            "sl":              round(initial_sl, 2),
            "trailing_active": False,
            "trail_distance":  0.0,
            "note":            "Trail calculation error — using initial SL",
        }


def format_trade_instructions(plan: dict) -> str:
    """
    Return a clean MT5-style trade management instruction block.

    Designed to be embedded in a trade card (inside a ```code``` block).
    Never raises.
    """
    try:
        entry         = plan["entry"]
        direction     = str(plan.get("direction", "long")).upper()
        total_lots    = plan["total_lots"]
        initial_sl    = plan["initial_sl"]
        sl_distance   = plan["sl_distance"]
        tp1_price     = plan["tp1_price"]
        tp1_lots      = plan["tp1_lots"]
        tp1_profit    = plan["tp1_profit_usd"]
        tp2_price     = plan["tp2_price"]
        tp2_lots      = plan["tp2_lots"]
        tp2_profit    = plan["tp2_profit_usd"]
        trail_step    = plan["trail_step_usd"]
        best_case     = plan["best_case_usd"]
        worst_case    = plan["worst_case_usd"]

        lines = [
            "═══ TRADE MANAGEMENT PLAN ═══",
            "",
            "  ENTRY:",
            f"  → {direction} {total_lots:.2f} lots at ${entry:,.2f}",
            f"  → Initial SL: ${initial_sl:,.2f}  (−${sl_distance:.2f})",
            "",
            f"  STEP 1 — When price hits ${tp1_price:,.2f}:",
            f"  → Close {tp1_lots:.2f} lots  (+${tp1_profit:,.2f})",
            f"  → Move SL to breakeven (${entry:,.2f})",
            f"  → Risk is now ZERO",
            "",
            f"  STEP 2 — Trail remaining {tp2_lots:.2f} lots:",
            f"  → Trail SL 1× ATR (${trail_step:.2f}) behind price",
            f"  → Target ${tp2_price:,.2f}  (+${tp2_profit:,.2f})",
            f"  → Or let trail stop you for partial profit",
            "",
            f"  BEST CASE:  +${best_case:,.2f}  (both TPs hit)",
            f"  WORST CASE: −${worst_case:,.2f}  (SL before TP1)",
        ]
        return "\n".join(lines)

    except Exception:
        return "═══ TRADE MANAGEMENT PLAN ═══\n  (plan data unavailable)"


# ══════════════════════════════════════════════════════════════════════════════
#  Risk of Ruin  (Monte Carlo)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_risk_of_ruin(
    win_rate: float,
    risk_pct: float,
    rr_ratio: float,
    num_trades: int = 100,
    simulations: int = 1000,
) -> dict[str, Any]:
    """
    Monte Carlo Risk-of-Ruin analysis.

    Parameters
    ----------
    win_rate    : historical win % (0–100)
    risk_pct    : risk per trade as % of balance (e.g. 10 = 10%)
    rr_ratio    : reward-to-risk ratio (e.g. 3.0 = 1:3)
    num_trades  : trades to simulate per run
    simulations : number of Monte Carlo runs

    Returns
    -------
    Dict with ruin_probability, risk_rating, ev_per_trade, balance projections,
    drawdown metrics, recommendation, and summary line.
    """
    try:
        win_prob          = max(0.0, min(1.0, win_rate / 100.0))
        loss_prob         = 1.0 - win_prob
        risk_per_trade    = max(0.001, risk_pct / 100.0)
        reward_per_trade  = risk_per_trade * rr_ratio

        ruin_count      = 0
        final_balances: list[float] = []
        drawdown_list:  list[float] = []

        for _ in range(simulations):
            balance = 1.0
            peak    = 1.0
            ruined  = False

            for _ in range(num_trades):
                if random() < win_prob:
                    balance *= (1.0 + reward_per_trade)
                else:
                    balance *= (1.0 - risk_per_trade)

                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak if peak > 0 else 0.0

                if balance <= 0.1:          # 90 % loss = ruin threshold
                    ruined = True
                    ruin_count += 1
                    break

            final_balances.append(balance)
            drawdown_list.append(drawdown)

        # ── Statistics ────────────────────────────────────────────────────────
        ruin_probability        = ruin_count / simulations * 100.0
        avg_final_balance       = sum(final_balances) / len(final_balances)
        sorted_bal              = sorted(final_balances)
        median_final_balance    = sorted_bal[len(sorted_bal) // 2]
        avg_max_drawdown        = sum(drawdown_list) / len(drawdown_list) * 100.0

        # Expected value per trade (fractional)
        ev = (win_prob * reward_per_trade) - (loss_prob * risk_per_trade)

        # Consecutive losses that produce a 90 % drawdown
        consec_ruin = 0
        temp = 1.0
        while temp > 0.1 and consec_ruin < 10_000:
            temp *= (1.0 - risk_per_trade)
            consec_ruin += 1

        # ── Risk rating ───────────────────────────────────────────────────────
        if ruin_probability < 5:
            risk_rating    = "SAFE"
            rating_color   = "green"
            recommendation = "Good risk management — continue"
        elif ruin_probability < 15:
            risk_rating    = "MODERATE"
            rating_color   = "yellow"
            recommendation = "Acceptable risk — monitor drawdown"
        elif ruin_probability < 30:
            risk_rating    = "HIGH"
            rating_color   = "orange"
            recommendation = "Reduce position size or improve win rate"
        else:
            risk_rating    = "DANGER"
            rating_color   = "red"
            recommendation = "High ruin risk — reduce risk per trade NOW"

        summary = (
            f"RoR {ruin_probability:.1f}% [{risk_rating}] | "
            f"EV {'+' if ev >= 0 else ''}{ev:.4f}/trade | "
            f"Avg drawdown {avg_max_drawdown:.1f}%"
        )

        return {
            "ruin_probability":           round(ruin_probability, 2),
            "risk_rating":                risk_rating,
            "rating_color":               rating_color,
            "recommendation":             recommendation,
            "win_rate":                   win_rate,
            "risk_pct":                   risk_pct,
            "rr_ratio":                   rr_ratio,
            "ev_per_trade":               round(ev, 6),
            "ev_positive":                ev > 0,
            "avg_final_balance_pct":      round(avg_final_balance * 100, 1),
            "median_final_balance_pct":   round(median_final_balance * 100, 1),
            "avg_max_drawdown_pct":       round(avg_max_drawdown, 2),
            "consecutive_losses_to_ruin": consec_ruin,
            "simulations_run":            simulations,
            "trades_simulated":           num_trades,
            "summary":                    summary,
        }

    except Exception as exc:
        return {
            "ruin_probability": 0.0,
            "risk_rating":      "UNKNOWN",
            "rating_color":     "grey",
            "recommendation":   f"RoR calculation error: {exc}",
            "win_rate":         win_rate,
            "risk_pct":         risk_pct,
            "rr_ratio":         rr_ratio,
            "ev_per_trade":     0.0,
            "ev_positive":      False,
            "avg_final_balance_pct":      100.0,
            "median_final_balance_pct":   100.0,
            "avg_max_drawdown_pct":       0.0,
            "consecutive_losses_to_ruin": 0,
            "simulations_run":            simulations,
            "trades_simulated":           num_trades,
            "summary":          f"Error: {exc}",
        }


def get_current_risk_profile() -> dict[str, Any]:
    """
    Build a live risk-of-ruin profile using current settings and bot history.

    Blends win rates from signal_performance.json and pattern_memory.json
    (falls back to 50 % if fewer than 5 resolved trades are available).

    Adds account-specific fields and a safe-risk-% recommendation.
    """
    try:
        from settings import load_settings
        settings = load_settings()
    except Exception:
        settings = {}

    balance  = float(settings.get("balance",  300.0))
    risk_pct = float(settings.get("risk_pct", 10.0))
    rr_ratio = float(settings.get("min_rr",   3.0))

    # ── Win rate: signal_performance.json ─────────────────────────────────────
    win_rate = 50.0
    try:
        _sp_path = os.path.join("data", "signal_performance.json")
        with open(_sp_path) as _f:
            signals = json.load(_f)
        resolved = [s for s in signals if s.get("outcome") in ("win", "loss", "partial")]
        if len(resolved) >= 5:
            wins     = len([s for s in resolved if s.get("outcome") == "win"])
            win_rate = wins / len(resolved) * 100.0
    except Exception:
        pass

    # ── Blend with pattern_memory.json ────────────────────────────────────────
    try:
        _pm_path = os.path.join("data", "pattern_memory.json")
        with open(_pm_path) as _f:
            memory = json.load(_f)
        if len(memory) >= 5:
            mem_wins = len([m for m in memory if m.get("outcome") == "WIN"])
            mem_wr   = mem_wins / len(memory) * 100.0
            win_rate = (win_rate + mem_wr) / 2.0
    except Exception:
        pass

    # ── Core RoR calculation ──────────────────────────────────────────────────
    ror = calculate_risk_of_ruin(
        win_rate   = win_rate,
        risk_pct   = risk_pct,
        rr_ratio   = rr_ratio,
        num_trades = 100,
        simulations= 1000,
    )

    # ── Account-specific additions ────────────────────────────────────────────
    ror["current_balance"]     = balance
    ror["risk_per_trade_usd"]  = round(balance * risk_pct / 100.0, 2)
    ror["ruin_threshold_usd"]  = round(balance * 0.1, 2)
    ror["trades_to_ruin_worst"]= ror["consecutive_losses_to_ruin"]

    # ── Find safe risk % (≤ 5 % ruin probability) ────────────────────────────
    safe_risk = risk_pct
    while safe_risk > 1.0:
        test = calculate_risk_of_ruin(win_rate, safe_risk, rr_ratio, 100, 200)
        if test["ruin_probability"] < 5.0:
            break
        safe_risk = round(safe_risk - 0.5, 1)

    ror["recommended_risk_pct"] = safe_risk
    ror["recommended_risk_usd"] = round(balance * safe_risk / 100.0, 2)

    return ror


def format_ror_report(ror: dict) -> str:
    """Return a formatted Risk-of-Ruin analysis report string."""
    try:
        ev_sign   = "+" if ror.get("ev_positive") else ""
        has_usd   = "risk_per_trade_usd" in ror
        usd_str   = f" (${ror['risk_per_trade_usd']:.2f})" if has_usd else ""
        safe_usd  = (
            f" (${ror['recommended_risk_usd']:.2f})"
            if "recommended_risk_usd" in ror
            else ""
        )

        lines = [
            "═══ RISK OF RUIN ANALYSIS ═══",
            "",
            f"  RATING: {ror['risk_rating']}",
            f"  Ruin probability: {ror['ruin_probability']:.1f}%",
            f"  ({ror['simulations_run']} simulations × {ror['trades_simulated']} trades)",
            "",
            "  YOUR SETTINGS:",
            f"  Win rate:     {ror['win_rate']:.1f}%",
            f"  Risk/trade:   {ror['risk_pct']}%{usd_str}",
            f"  Reward ratio: 1:{ror['rr_ratio']}",
            f"  EV/trade:     {ev_sign}{ror['ev_per_trade']:.4f}",
            "",
            "  RISK METRICS:",
            f"  Avg final balance: {ror['avg_final_balance_pct']:.0f}% of start",
            f"  Avg max drawdown:  {ror['avg_max_drawdown_pct']:.1f}%",
            f"  Losses to ruin:    {ror['consecutive_losses_to_ruin']} in a row",
            "",
            "  RECOMMENDATION:",
            f"  {ror['recommendation']}",
            f"  Safe risk %: {ror.get('recommended_risk_pct', ror['risk_pct'])}%{safe_usd}",
            "  ════════════════════════════",
        ]
        return "\n".join(lines)

    except Exception as exc:
        return f"═══ RISK OF RUIN ANALYSIS ═══\n  (report unavailable: {exc})"


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _read_atr(df: Any) -> float:
    """Extract latest ATR from DataFrame; return safe fallback on error."""
    _FALLBACK = 20.0
    try:
        if df is None:
            return _FALLBACK
        import pandas as pd
        if not isinstance(df, pd.DataFrame) or df.empty:
            return _FALLBACK
        for col in ("atr", "ATR", "Atr"):
            if col in df.columns:
                val = float(df[col].iloc[-1])
                return val if val > 0 else _FALLBACK
        return _FALLBACK
    except Exception:
        return 20.0


def _empty_plan(signal: dict) -> dict[str, Any]:
    """Return a safe empty plan when inputs are invalid."""
    return {
        "entry":            float(signal.get("entry", 0) or 0),
        "direction":        str(signal.get("direction", "long")),
        "total_lots":       float(signal.get("lots", 0.01) or 0.01),
        "tp1_price":        0.0,
        "tp1_lots":         0.01,
        "tp1_lots_pct":     50,
        "tp1_rr":           2.0,
        "tp1_profit_usd":   0.0,
        "tp1_action":       "Close 50% + move SL to breakeven",
        "tp2_price":        0.0,
        "tp2_lots":         0.01,
        "tp2_lots_pct":     50,
        "tp2_rr":           3.0,
        "tp2_profit_usd":   0.0,
        "tp2_action":       "Close remaining 50%",
        "initial_sl":       float(signal.get("stop_loss", 0) or 0),
        "breakeven_sl":     float(signal.get("entry", 0) or 0),
        "trail_step_usd":   10.0,
        "trail_atr_mult":   1.0,
        "sl_distance":      0.0,
        "atr":              20.0,
        "total_risk_usd":   0.0,
        "total_reward_tp1": 0.0,
        "total_reward_tp2": 0.0,
        "best_case_usd":    0.0,
        "worst_case_usd":   0.0,
        "management_steps": ["(invalid signal — missing entry or SL)"],
        "valid":            False,
    }
