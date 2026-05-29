"""
btc_research/backtest/engine.py — Bar-by-bar BTC backtest simulation.

Rules:
  - One trade open at a time (never double up on BTC)
  - Entries only during kill-zone (13:00-17:00 UTC) via morning-range breakout
  - Long and short both evaluated each kill-zone bar; first signal wins
  - TP1 hit: close 50% position, move SL to breakeven
  - TP2 hit: close remaining 50% position
  - SL hit after TP1: SL_AFTER_TP1 (we already banked partial profit)
  - MAX_HOLD: force-close at bar close after 96 H1 bars (4 days)
  - Compounding: risk 3% of current balance per trade (lot size grows/shrinks)

Usage:
    from btc_research.backtest.engine import run
    results = run(df_btc, df_gold, df_nas)
"""
from __future__ import annotations

from datetime import timezone
import pandas as pd

from btc_research.settings import (
    STARTING_BALANCE, RISK_PCT, TP1_RR, TP2_RR,
    MAX_HOLD_BARS, KZ_START_UTC, KZ_END_UTC, MIN_CONFLUENCE_SCORE,
)
from btc_research.strategy.confluence import score_bar


def _lot_size(balance: float, entry: float, sl: float) -> float:
    """
    BTC units to trade.
    risk_amount = balance * RISK_PCT
    btc_units   = risk_amount / sl_distance_usd
    """
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0.001
    units = (balance * RISK_PCT) / sl_dist
    return round(max(units, 0.001), 6)


def run(
    df_btc:  pd.DataFrame,
    df_gold: pd.DataFrame,
    df_nas:  pd.DataFrame,
    verbose: bool = False,
) -> dict:
    """
    Run the backtest on 2 years of H1 data.

    Returns:
        trades  : list[dict]   all closed trades
        balance : float        final account balance
    """
    if df_btc.empty:
        return {"trades": [], "balance": STARTING_BALANCE}

    # Work on copies and ensure UTC-aware timestamps
    df_btc  = df_btc.copy().reset_index(drop=True)
    df_gold = df_gold.copy() if not df_gold.empty else pd.DataFrame()
    df_nas  = df_nas.copy()  if not df_nas.empty  else pd.DataFrame()

    for df in [df_btc, df_gold, df_nas]:
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)

    balance:    float        = STARTING_BALANCE
    open_trade: dict | None  = None
    trades:     list[dict]   = []
    n          = len(df_btc)

    for i in range(n):
        row       = df_btc.iloc[i]
        bar_time  = row["time"]
        bar_close = float(row["close"])
        bar_high  = float(row["high"])
        bar_low   = float(row["low"])

        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)
        utc_hour = bar_time.hour

        # ── Monitor open trade ────────────────────────────────────────────────
        if open_trade:
            t         = open_trade
            is_long   = t["direction"] == "long"
            t["bars_held"] = t.get("bars_held", 0) + 1
            bars_held  = t["bars_held"]

            sl       = t["sl"]
            tp1      = t["tp1"]
            tp2      = t["tp2"]
            tp1_hit  = t.get("tp1_hit", False)
            entry    = t["entry"]
            lots     = t["lots"]

            closed      = False
            exit_price  = None
            exit_reason = None

            # Max hold
            if bars_held >= MAX_HOLD_BARS:
                exit_price  = bar_close
                exit_reason = "MAX_HOLD"
                closed = True

            # SL check
            elif (is_long and bar_low <= sl) or (not is_long and bar_high >= sl):
                exit_price  = sl
                exit_reason = "SL_AFTER_TP1" if tp1_hit else "SL"
                closed = True

            # TP2 check (only active after TP1)
            elif tp1_hit:
                if (is_long and bar_high >= tp2) or (not is_long and bar_low <= tp2):
                    exit_price  = tp2
                    exit_reason = "TP2"
                    closed = True

            # TP1 check
            elif not tp1_hit:
                if (is_long and bar_high >= tp1) or (not is_long and bar_low <= tp1):
                    t["tp1_hit"] = True
                    t["sl"]      = entry              # SL to breakeven
                    # Book 50% at TP1
                    partial_pnl      = (tp1 - entry if is_long else entry - tp1) * lots * 0.5
                    t["tp1_pnl"]     = round(partial_pnl, 2)
                    balance         += partial_pnl
                    if verbose:
                        print(f"    TP1  {t['id'][:8]} {t['direction'].upper():5s} "
                              f"@ {tp1:.2f}  partial_pnl=${partial_pnl:+.2f}  "
                              f"SL->BE={entry:.2f}  bal=${balance:.2f}")

            # Close trade on SL / TP2 / MAX_HOLD
            if closed and exit_price is not None:
                price_diff     = (exit_price - entry) if is_long else (entry - exit_price)
                remaining_lots = lots * (0.5 if tp1_hit else 1.0)
                residual_pnl   = price_diff * remaining_lots
                total_pnl      = residual_pnl + t.get("tp1_pnl", 0.0)

                sl_dist_orig   = abs(entry - t["original_sl"])
                r_multiple     = price_diff / sl_dist_orig if sl_dist_orig > 0 else 0.0

                balance += residual_pnl
                balance  = max(balance, 1.0)

                closed_trade = {
                    "id":            t["id"],
                    "symbol":        "BTCUSD",
                    "direction":     t["direction"],
                    "entry":         round(entry, 2),
                    "exit":          round(exit_price, 2),
                    "sl":            round(t["original_sl"], 2),
                    "tp1":           round(tp1, 2),
                    "tp2":           round(tp2, 2),
                    "lots":          lots,
                    "pnl_usd":       round(total_pnl, 2),
                    "r_multiple":    round(r_multiple, 2),
                    "bars_held":     bars_held,
                    "open_time":     t["open_time"],
                    "close_time":    bar_time.isoformat(),
                    "exit_reason":   exit_reason,
                    "score":         t.get("score", 0.0),
                    "factors":       t.get("factors", {}),
                    "balance_after": round(balance, 2),
                }
                trades.append(closed_trade)
                open_trade = None

                if verbose:
                    print(f"  CLOSE {closed_trade['id'][:8]} "
                          f"{t['direction'].upper():5s} @ {exit_price:.2f}  "
                          f"PnL=${total_pnl:+.2f}  R={r_multiple:+.2f}  "
                          f"[{exit_reason}]  bal=${balance:.2f}")

        # ── Look for new entry ─────────────────────────────────────────────────
        if open_trade is None and KZ_START_UTC <= utc_hour < KZ_END_UTC:
            # Slice df_btc up to current bar for the confluence engine
            df_window = df_btc.iloc[: i + 1]

            for direction in ("long", "short"):
                res = score_bar(
                    bar_time  = bar_time,
                    bar_close = bar_close,
                    direction = direction,
                    df_btc    = df_window,
                    df_gold   = df_gold,
                    df_nas    = df_nas,
                )

                if not res["signal"]:
                    continue

                entry   = res["entry"]
                sl      = res["sl"]
                tp1     = res["tp1"]
                tp2     = res["tp2"]
                lots    = _lot_size(balance, entry, sl)
                trade_id = f"BTC-{len(trades) + 1:04d}"

                open_trade = {
                    "id":          trade_id,
                    "symbol":      "BTCUSD",
                    "direction":   direction,
                    "entry":       entry,
                    "sl":          sl,
                    "original_sl": sl,
                    "tp1":         tp1,
                    "tp2":         tp2,
                    "lots":        lots,
                    "tp1_hit":     False,
                    "tp1_pnl":     0.0,
                    "bars_held":   0,
                    "open_time":   bar_time.isoformat(),
                    "score":       res["score"],
                    "factors":     res["factors"],
                }

                if verbose:
                    print(f"  OPEN  {trade_id} {direction.upper():5s} "
                          f"@ {entry:.2f}  SL={sl:.2f}  TP1={tp1:.2f}  TP2={tp2:.2f}  "
                          f"lots={lots:.4f}  score={res['score']:.1f}  bal=${balance:.2f}")
                break   # only one direction per bar

    # ── Force-close anything still open at end of data ────────────────────────
    if open_trade:
        t         = open_trade
        last_row  = df_btc.iloc[-1]
        exit_p    = float(last_row["close"])
        entry     = t["entry"]
        is_long   = t["direction"] == "long"
        lots      = t["lots"]
        tp1_hit   = t.get("tp1_hit", False)
        rem_lots  = lots * (0.5 if tp1_hit else 1.0)
        price_diff = (exit_p - entry) if is_long else (entry - exit_p)
        total_pnl  = price_diff * rem_lots + t.get("tp1_pnl", 0.0)
        sl_dist    = abs(entry - t["original_sl"])
        r_multiple = price_diff / sl_dist if sl_dist > 0 else 0.0
        balance   += price_diff * rem_lots
        balance    = max(balance, 1.0)
        trades.append({
            "id":            t["id"],
            "symbol":        "BTCUSD",
            "direction":     t["direction"],
            "entry":         round(entry, 2),
            "exit":          round(exit_p, 2),
            "sl":            round(t["original_sl"], 2),
            "tp1":           round(t["tp1"], 2),
            "tp2":           round(t["tp2"], 2),
            "lots":          lots,
            "pnl_usd":       round(total_pnl, 2),
            "r_multiple":    round(r_multiple, 2),
            "bars_held":     t.get("bars_held", 0),
            "open_time":     t["open_time"],
            "close_time":    pd.to_datetime(last_row["time"]).isoformat(),
            "exit_reason":   "END_OF_DATA",
            "score":         t.get("score", 0.0),
            "factors":       {},
            "balance_after": round(balance, 2),
        })

    return {"trades": trades, "balance": round(balance, 2)}
