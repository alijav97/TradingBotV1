"""
btc_research/backtest/optimizer.py — Parameter optimizer for BTC strategies.

Runs a grid search over key parameters for the top-performing strategies
(Volatility Breakout + Morning Range Breakout) to find the combination that:
  - Produces enough trades (target: 4-8 per month = ~96-192 over 2 years)
  - Maintains acceptable win rate (>= 42%)
  - Maximises total PnL and avg R

For Volatility Breakout the key levers are:
  - atr_multiplier  : how large a bar must be relative to ATR (1.0 = any, 2.0 = very large)
  - close_zone      : how far close must be from bar extreme (0.30 = top/bottom 30%)
  - session_hours   : which UTC hours to trade

For Morning Range Breakout:
  - range_bars      : how many bars define the reference range (4-12)
  - session_hours   : which UTC hours to trade

Usage:
    from btc_research.backtest.optimizer import run_optimizer
    run_optimizer(df_btc, df_gold, df_nas)
"""
from __future__ import annotations

from datetime import timezone
import pandas as pd
import itertools

from btc_research.settings import (
    STARTING_BALANCE, RISK_PCT, TP1_RR, TP2_RR, MAX_HOLD_BARS,
)
from btc_research.strategies.volatility_breakout import VolatilityBreakout
from btc_research.strategies.morning_range       import MorningRangeBreakout
from btc_research.factors.gold_factor            import compute_gold_factor
from btc_research.factors.nasdaq_factor          import compute_nasdaq_factor


# ── Target constraints ────────────────────────────────────────────────────────
MIN_TRADES_2YR   = 80     # minimum trades over 2 years (~3.3/month)
TARGET_TRADES_2YR = 120   # ideal: ~5/month
MAX_TRADES_2YR   = 300    # too many = low quality
MIN_WIN_RATE     = 42.0   # minimum acceptable WR%
MIN_AVG_R        = 0.20   # minimum acceptable avg R per trade
MAX_DRAWDOWN     = 30.0   # maximum acceptable drawdown%


def _lot_size(balance: float, entry: float, sl: float) -> float:
    sl_dist = abs(entry - sl)
    return round(max((balance * RISK_PCT) / sl_dist, 0.001), 6) if sl_dist > 0 else 0.001


def _im_score(df_btc, df_gold, df_nas, bar_time, direction):
    """Inter-market score (gold + nas, direction-adjusted)."""
    if df_gold.empty or df_nas.empty:
        return 0.0
    gold_f = compute_gold_factor(df_gold, bar_time)
    nas_f  = compute_nasdaq_factor(df_nas, bar_time)
    d_mult = 1.0 if direction.lower() in ("long", "buy") else -1.0
    return gold_f["score"] * 0.5 * d_mult + nas_f["score"] * 0.5 * d_mult


def _run_strategy(
    strategy,
    df_btc:        pd.DataFrame,
    df_gold:       pd.DataFrame,
    df_nas:        pd.DataFrame,
    allowed_hours: set[int],
    use_im_filter: bool = True,
    im_threshold:  float = -1.0,
) -> dict:
    """Run a strategy simulation and return summary stats."""
    df_btc = df_btc.copy().reset_index(drop=True)
    for df in [df_btc, df_gold, df_nas]:
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)

    balance    = STARTING_BALANCE
    open_trade = None
    trades     = []
    n          = len(df_btc)

    for i in range(n):
        row       = df_btc.iloc[i]
        bar_time  = row["time"]
        bar_close = float(row["close"])
        bar_high  = float(row["high"])
        bar_low   = float(row["low"])

        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        # Monitor open trade
        if open_trade:
            t       = open_trade
            is_long = t["direction"] == "long"
            t["bars_held"] = t.get("bars_held", 0) + 1
            bh = t["bars_held"]
            closed, exit_price, exit_reason = False, None, None

            if bh >= MAX_HOLD_BARS:
                exit_price, exit_reason, closed = bar_close, "MAX_HOLD", True
            elif (is_long and bar_low <= t["sl"]) or (not is_long and bar_high >= t["sl"]):
                exit_price = t["sl"]
                exit_reason = "SL_AFTER_TP1" if t.get("tp1_hit") else "SL"
                closed = True
            elif t.get("tp1_hit"):
                if (is_long and bar_high >= t["tp2"]) or (not is_long and bar_low <= t["tp2"]):
                    exit_price, exit_reason, closed = t["tp2"], "TP2", True
            else:
                if (is_long and bar_high >= t["tp1"]) or (not is_long and bar_low <= t["tp1"]):
                    t["tp1_hit"] = True
                    t["sl"] = t["entry"]
                    partial = (t["tp1"] - t["entry"] if is_long else t["entry"] - t["tp1"]) * t["lots"] * 0.5
                    t["tp1_pnl"] = round(partial, 2)
                    balance += partial

            if closed and exit_price is not None:
                pd_   = (exit_price - t["entry"]) if is_long else (t["entry"] - exit_price)
                rem   = t["lots"] * (0.5 if t.get("tp1_hit") else 1.0)
                pnl   = pd_ * rem + t.get("tp1_pnl", 0.0)
                sld   = abs(t["entry"] - t["original_sl"])
                rr    = pd_ / sld if sld > 0 else 0.0
                balance = max(balance + pd_ * rem, 1.0)
                trades.append({
                    "pnl_usd": round(pnl, 2),
                    "r_multiple": round(rr, 2),
                    "exit_reason": exit_reason,
                    "bars_held": bh,
                    "balance_after": round(balance, 2),
                })
                open_trade = None

        # New entry
        if open_trade is None and bar_time.hour in allowed_hours:
            df_window = df_btc.iloc[:i + 1]
            for direction in ("long", "short"):
                sig = strategy.generate_signal(df_window, bar_time, direction)
                if not sig.get("signal"):
                    continue
                entry, sl = sig["entry"], sig["sl"]
                if sl <= 0 or abs(entry - sl) <= 0:
                    continue

                # Optional IM filter
                if use_im_filter:
                    ims = _im_score(df_btc, df_gold, df_nas, bar_time, direction)
                    if ims <= im_threshold:
                        continue

                sl_dist = abs(entry - sl)
                is_long = direction == "long"
                tp1 = entry + TP1_RR * sl_dist if is_long else entry - TP1_RR * sl_dist
                tp2 = entry + TP2_RR * sl_dist if is_long else entry - TP2_RR * sl_dist
                lots = _lot_size(balance, entry, sl)
                open_trade = {
                    "direction": direction, "entry": entry,
                    "sl": sl, "original_sl": sl,
                    "tp1": tp1, "tp2": tp2, "lots": lots,
                    "tp1_hit": False, "tp1_pnl": 0.0, "bars_held": 0,
                }
                break

    if not trades:
        return {"trades": 0, "wr": 0.0, "avg_r": 0.0,
                "total_pnl": 0.0, "max_dd": 0.0, "expectancy": 0.0,
                "tp2_rate": 0.0}

    df_t   = pd.DataFrame(trades)
    total  = len(df_t)
    wins   = (df_t["pnl_usd"] > 0).sum()
    wr     = wins / total * 100
    avg_r  = df_t["r_multiple"].mean()
    total_pnl = df_t["pnl_usd"].sum()
    avg_win  = df_t[df_t["pnl_usd"] > 0]["pnl_usd"].mean() if wins > 0 else 0
    avg_loss = df_t[df_t["pnl_usd"] <= 0]["pnl_usd"].mean() if (total - wins) > 0 else 0
    expectancy = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_loss)
    tp2_rate = (df_t["exit_reason"] == "TP2").sum() / total * 100

    # Drawdown
    equity = [STARTING_BALANCE] + df_t["balance_after"].tolist()
    peak, max_dd = STARTING_BALANCE, 0.0
    for b in equity:
        if b > peak:
            peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": total, "wr": round(wr, 1), "avg_r": round(avg_r, 2),
        "total_pnl": round(total_pnl, 2), "max_dd": round(max_dd, 1),
        "expectancy": round(expectancy, 2), "tp2_rate": round(tp2_rate, 1),
    }


# ── Session window presets to test ────────────────────────────────────────────
SESSION_WINDOWS = {
    "02-04 UTC (2h)":   set(range(2, 4)),
    "01-05 UTC (4h)":   set(range(1, 5)),
    "00-05 UTC (5h)":   set(range(0, 5)),
    "02-06 UTC (4h)":   set(range(2, 6)),
    "00-06 UTC (6h)":   set(range(0, 6)),
    "21-04 UTC (7h)":   set(range(21, 24)) | set(range(0, 4)),
    "22-05 UTC (7h)":   set(range(22, 24)) | set(range(0, 5)),
    "00-08 UTC (8h)":   set(range(0, 8)),
}


def optimize_volatility_breakout(
    df_btc: pd.DataFrame,
    df_gold: pd.DataFrame,
    df_nas: pd.DataFrame,
) -> None:
    """Grid search over ATR multiplier × close zone × session window."""

    atr_multipliers = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]
    close_zones     = [0.30, 0.35, 0.40, 0.45, 0.50]

    print("\n" + "=" * 95)
    print("VOLATILITY BREAKOUT — PARAMETER OPTIMIZATION")
    print(f"Goal: {MIN_TRADES_2YR}-{MAX_TRADES_2YR} trades  |  WR>={MIN_WIN_RATE}%  |  AvgR>={MIN_AVG_R}  |  MaxDD<={MAX_DRAWDOWN}%")
    print("=" * 95)

    results = []
    total_combos = len(atr_multipliers) * len(close_zones) * len(SESSION_WINDOWS)
    done = 0

    for (atr_mult, cz, (session_name, hours)) in itertools.product(
        atr_multipliers, close_zones, SESSION_WINDOWS.items()
    ):
        strat  = VolatilityBreakout(atr_multiplier=atr_mult, close_zone=cz)
        stats  = _run_strategy(strat, df_btc, df_gold, df_nas, hours,
                               use_im_filter=True)
        done += 1
        print(f"  {done}/{total_combos} ...", end="\r")

        results.append({
            "atr_mult":    atr_mult,
            "close_zone":  cz,
            "session":     session_name,
            "**trades**":  stats["trades"],
            "wr":          stats["wr"],
            "avg_r":       stats["avg_r"],
            "total_pnl":   stats["total_pnl"],
            "max_dd":      stats["max_dd"],
            "tp2_rate":    stats["tp2_rate"],
            "expectancy":  stats["expectancy"],
        })

    print(" " * 50, end="\r")

    df_r = pd.DataFrame(results)

    # Filter to valid combinations
    valid = df_r[
        (df_r["**trades**"] >= MIN_TRADES_2YR) &
        (df_r["**trades**"] <= MAX_TRADES_2YR) &
        (df_r["wr"]          >= MIN_WIN_RATE) &
        (df_r["avg_r"]       >= MIN_AVG_R) &
        (df_r["max_dd"]      <= MAX_DRAWDOWN)
    ].sort_values("total_pnl", ascending=False)

    if valid.empty:
        print("\nNo combinations met all constraints. Relaxing WR floor to 38%...")
        valid = df_r[
            (df_r["**trades**"] >= MIN_TRADES_2YR) &
            (df_r["**trades**"] <= MAX_TRADES_2YR) &
            (df_r["avg_r"]      >= 0.10) &
            (df_r["max_dd"]     <= MAX_DRAWDOWN)
        ].sort_values("total_pnl", ascending=False)

    print(f"\nTop 15 combinations (from {len(valid)} passing filters):\n")
    print(f"  {'ATR':>5} {'CZ':>5} {'Session':<20} {'Trades':>7} {'WR%':>7} "
          f"{'AvgR':>7} {'Total PnL':>12} {'MaxDD':>7} {'TP2%':>6} {'Expect':>9}")
    print(f"  {'-'*90}")

    for _, row in valid.head(15).iterrows():
        trades_per_mo = row["**trades**"] / 24
        print(f"  {row['atr_mult']:>5.1f} {row['close_zone']:>5.2f} "
              f"{row['session']:<20} "
              f"{int(row['**trades**']):>5} ({trades_per_mo:.1f}/mo) "
              f"{row['wr']:>6.1f}% "
              f"{row['avg_r']:>+6.2f}R "
              f"${row['total_pnl']:>+10.2f} "
              f"{row['max_dd']:>6.1f}% "
              f"{row['tp2_rate']:>5.1f}% "
              f"${row['expectancy']:>+7.2f}")

    if not valid.empty:
        best = valid.iloc[0]
        print(f"\n  *** BEST COMBINATION ***")
        print(f"  ATR multiplier : {best['atr_mult']}×")
        print(f"  Close zone     : top/bottom {best['close_zone']*100:.0f}% of bar")
        print(f"  Session        : {best['session']}")
        print(f"  Result         : {int(best['**trades**'])} trades "
              f"({best['**trades**']/24:.1f}/month)  "
              f"WR={best['wr']}%  AvgR={best['avg_r']:+.2f}  "
              f"PnL=${best['total_pnl']:+,.2f}  "
              f"MaxDD={best['max_dd']}%")


def optimize_morning_range(
    df_btc: pd.DataFrame,
    df_gold: pd.DataFrame,
    df_nas: pd.DataFrame,
) -> None:
    """Grid search over range_bars × session window."""

    range_bars_options = [3, 4, 5, 6, 8, 10, 12]

    print("\n" + "=" * 95)
    print("MORNING RANGE BREAKOUT — PARAMETER OPTIMIZATION")
    print(f"Goal: {MIN_TRADES_2YR}-{MAX_TRADES_2YR} trades  |  WR>={MIN_WIN_RATE}%  |  AvgR>={MIN_AVG_R}  |  MaxDD<={MAX_DRAWDOWN}%")
    print("=" * 95)

    results = []
    total_combos = len(range_bars_options) * len(SESSION_WINDOWS)
    done = 0

    for (rb, (session_name, hours)) in itertools.product(
        range_bars_options, SESSION_WINDOWS.items()
    ):
        strat  = MorningRangeBreakout(range_bars=rb)
        stats  = _run_strategy(strat, df_btc, df_gold, df_nas, hours,
                               use_im_filter=True)
        done += 1
        print(f"  {done}/{total_combos} ...", end="\r")

        results.append({
            "range_bars":  rb,
            "session":     session_name,
            "**trades**":  stats["trades"],
            "wr":          stats["wr"],
            "avg_r":       stats["avg_r"],
            "total_pnl":   stats["total_pnl"],
            "max_dd":      stats["max_dd"],
            "tp2_rate":    stats["tp2_rate"],
            "expectancy":  stats["expectancy"],
        })

    print(" " * 50, end="\r")

    df_r = pd.DataFrame(results)

    valid = df_r[
        (df_r["**trades**"] >= MIN_TRADES_2YR) &
        (df_r["**trades**"] <= MAX_TRADES_2YR) &
        (df_r["wr"]          >= MIN_WIN_RATE) &
        (df_r["avg_r"]       >= MIN_AVG_R) &
        (df_r["max_dd"]      <= MAX_DRAWDOWN)
    ].sort_values("total_pnl", ascending=False)

    if valid.empty:
        valid = df_r[
            (df_r["**trades**"] >= MIN_TRADES_2YR) &
            (df_r["avg_r"]      >= 0.10)
        ].sort_values("total_pnl", ascending=False)

    print(f"\nTop 15 combinations (from {len(valid)} passing filters):\n")
    print(f"  {'RangeBars':>10} {'Session':<20} {'Trades':>7} {'WR%':>7} "
          f"{'AvgR':>7} {'Total PnL':>12} {'MaxDD':>7} {'TP2%':>6} {'Expect':>9}")
    print(f"  {'-'*85}")

    for _, row in valid.head(15).iterrows():
        trades_per_mo = row["**trades**"] / 24
        print(f"  {int(row['range_bars']):>10} "
              f"{row['session']:<20} "
              f"{int(row['**trades**']):>5} ({trades_per_mo:.1f}/mo) "
              f"{row['wr']:>6.1f}% "
              f"{row['avg_r']:>+6.2f}R "
              f"${row['total_pnl']:>+10.2f} "
              f"{row['max_dd']:>6.1f}% "
              f"{row['tp2_rate']:>5.1f}% "
              f"${row['expectancy']:>+7.2f}")

    if not valid.empty:
        best = valid.iloc[0]
        print(f"\n  *** BEST COMBINATION ***")
        print(f"  Range bars : {int(best['range_bars'])} bars")
        print(f"  Session    : {best['session']}")
        print(f"  Result     : {int(best['**trades**'])} trades "
              f"({best['**trades**']/24:.1f}/month)  "
              f"WR={best['wr']}%  AvgR={best['avg_r']:+.2f}  "
              f"PnL=${best['total_pnl']:+,.2f}  "
              f"MaxDD={best['max_dd']}%")


def run_optimizer(
    df_btc:  pd.DataFrame,
    df_gold: pd.DataFrame,
    df_nas:  pd.DataFrame,
    strategy: str = "both",   # "volatility", "morning_range", or "both"
) -> None:
    """
    Run parameter optimization for top strategies.

    Args:
        strategy : which strategy to optimize
                   "volatility"    — Volatility Breakout only
                   "morning_range" — Morning Range Breakout only
                   "both"          — both (default)
    """
    if strategy in ("volatility", "both"):
        optimize_volatility_breakout(df_btc, df_gold, df_nas)

    if strategy in ("morning_range", "both"):
        optimize_morning_range(df_btc, df_gold, df_nas)

    print("\n" + "=" * 95)
    print("OPTIMIZATION COMPLETE")
    print("Update btc_research/settings.py with the best parameters, then re-run:")
    print("  python btc_research/run_backtest.py --compare")
    print("=" * 95)
