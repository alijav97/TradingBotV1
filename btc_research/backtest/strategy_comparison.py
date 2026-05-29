"""
btc_research/backtest/strategy_comparison.py — Compare all BTC strategies head-to-head.

Tests every strategy with:
  A. Pure signal only (no inter-market filter)
  B. Signal + inter-market filter (Gold + NAS100)

This tells us:
  1. Which strategy entry logic is best for BTC
  2. Whether adding Gold/NAS filter improves each strategy
  3. Which combination to use for the live BTC bot

All strategies use IDENTICAL:
  - Session window  (from settings: KZ_START_UTC to KZ_END_UTC)
  - Exit logic      (TP1=1:2R → 50% close + BE, TP2=1:5R, SL, MAX_HOLD=96h)
  - Risk per trade  (3% of compounding balance)
  - Starting balance ($10,000)

Usage:
    from btc_research.backtest.strategy_comparison import run_comparison
    run_comparison(df_btc, df_gold, df_nas)
"""
from __future__ import annotations

from datetime import timezone
import pandas as pd

from btc_research.settings import (
    STARTING_BALANCE, RISK_PCT, TP1_RR, TP2_RR, MAX_HOLD_BARS,
    KZ_START_UTC, KZ_END_UTC,
)
from btc_research.strategies.base              import BTCStrategy
from btc_research.strategies.morning_range     import MorningRangeBreakout
from btc_research.strategies.ema_trend         import EMATrendFollow
from btc_research.strategies.rsi_reversion     import RSIMeanReversion
from btc_research.strategies.volatility_breakout import VolatilityBreakout
from btc_research.strategies.swing_level       import SwingLevelBreak
from btc_research.factors.gold_factor          import compute_gold_factor
from btc_research.factors.nasdaq_factor        import compute_nasdaq_factor
from btc_research.factors.btc_momentum        import compute_btc_momentum


# All strategies to compare
ALL_STRATEGIES: list[BTCStrategy] = [
    MorningRangeBreakout(range_bars=6),
    EMATrendFollow(),
    RSIMeanReversion(),
    VolatilityBreakout(),
    SwingLevelBreak(),
]


def _lot_size(balance: float, entry: float, sl: float) -> float:
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0.001
    return round(max((balance * RISK_PCT) / sl_dist, 0.001), 6)


def _intermarket_score(
    df_btc:    pd.DataFrame,
    df_gold:   pd.DataFrame,
    df_nas:    pd.DataFrame,
    bar_time:  pd.Timestamp,
    direction: str,
) -> float:
    """
    Combined inter-market score (Gold + NAS) adjusted for direction.
    Returns a float: positive = factors support direction, negative = oppose.
    Threshold: score > 0 = allow, score <= -1 = block.
    """
    gold_f  = compute_gold_factor(df_gold, bar_time)
    nas_f   = compute_nasdaq_factor(df_nas, bar_time)
    d_mult  = 1.0 if direction.lower() in ("long", "buy") else -1.0
    return gold_f["score"] * 0.5 * d_mult + nas_f["score"] * 0.5 * d_mult


def _simulate(
    strategy:           BTCStrategy,
    df_btc:             pd.DataFrame,
    df_gold:            pd.DataFrame,
    df_nas:             pd.DataFrame,
    use_intermarket:    bool = True,
    intermarket_thresh: float = -1.0,  # block if IM score <= this
    allowed_hours:      set[int] | None = None,
) -> list[dict]:
    """
    Run a single strategy simulation over the full dataset.

    Args:
        strategy           : the BTCStrategy to test
        df_btc/gold/nas    : price data
        use_intermarket    : whether to apply Gold/NAS filter
        intermarket_thresh : minimum IM score to allow a trade
        allowed_hours      : set of UTC hours to trade in (None = use KZ settings)

    Returns list of closed trade dicts.
    """
    if allowed_hours is None:
        allowed_hours = set(range(KZ_START_UTC, KZ_END_UTC))

    df_btc  = df_btc.copy().reset_index(drop=True)
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

        # ── Monitor open trade ────────────────────────────────────────────────
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
                    t["sl"] = t["entry"]   # move SL to breakeven
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
                    "id":            t["id"],
                    "strategy":      strategy.name,
                    "direction":     t["direction"],
                    "entry":         t["entry"],
                    "exit":          round(exit_price, 2),
                    "sl":            t["original_sl"],
                    "pnl_usd":       round(pnl, 2),
                    "r_multiple":    round(rr, 2),
                    "bars_held":     bh,
                    "exit_reason":   exit_reason,
                    "score":         t.get("im_score", 0.0),
                    "open_hour_utc": t["open_hour"],
                    "open_time":     t["open_time"],
                    "balance_after": round(balance, 2),
                    "signal_reason": t.get("signal_reason", ""),
                    "used_filter":   use_intermarket,
                })
                open_trade = None

        # ── New entry ──────────────────────────────────────────────────────────
        if open_trade is None and bar_time.hour in allowed_hours:
            df_window = df_btc.iloc[: i + 1]

            for direction in ("long", "short"):
                # Strategy signal
                sig = strategy.generate_signal(df_window, bar_time, direction)
                if not sig.get("signal"):
                    continue

                entry = sig["entry"]
                sl    = sig["sl"]
                if sl <= 0 or abs(entry - sl) <= 0:
                    continue

                # Inter-market filter (optional)
                im_score = 0.0
                if use_intermarket and not df_gold.empty and not df_nas.empty:
                    im_score = _intermarket_score(df_btc, df_gold, df_nas, bar_time, direction)
                    if im_score <= intermarket_thresh:
                        continue   # inter-market factors oppose this trade

                sl_dist = abs(entry - sl)
                tp1 = entry + TP1_RR * sl_dist if direction == "long" else entry - TP1_RR * sl_dist
                tp2 = entry + TP2_RR * sl_dist if direction == "long" else entry - TP2_RR * sl_dist
                lots = _lot_size(balance, entry, sl)
                trade_id = f"{strategy.name[:3].upper()}-{len(trades)+1:04d}"

                open_trade = {
                    "id":           trade_id,
                    "direction":    direction,
                    "entry":        entry,
                    "sl":           sl,
                    "original_sl":  sl,
                    "tp1":          tp1,
                    "tp2":          tp2,
                    "lots":         lots,
                    "tp1_hit":      False,
                    "tp1_pnl":      0.0,
                    "bars_held":    0,
                    "open_hour":    bar_time.hour,
                    "open_time":    bar_time.isoformat(),
                    "im_score":     im_score,
                    "signal_reason": sig.get("reason", ""),
                }
                break   # one direction per bar

    return trades


def _stats(trades: list[dict], balance_start: float = STARTING_BALANCE) -> dict:
    """Compute summary statistics for a list of trades."""
    if not trades:
        return {
            "trades": 0, "wr": 0.0, "avg_r": 0.0,
            "avg_pnl": 0.0, "total_pnl": 0.0,
            "max_dd": 0.0, "expectancy": 0.0,
        }
    df = pd.DataFrame(trades)
    wins   = (df["pnl_usd"] > 0).sum()
    total  = len(df)
    wr     = wins / total * 100
    avg_r  = df["r_multiple"].mean()
    avg_pnl = df["pnl_usd"].mean()
    total_pnl = df["pnl_usd"].sum()
    wins_pnl = df[df["pnl_usd"] > 0]["pnl_usd"].mean() if wins > 0 else 0
    loss_pnl = df[df["pnl_usd"] <= 0]["pnl_usd"].mean() if (total - wins) > 0 else 0
    expectancy = (wr/100 * wins_pnl) + ((1 - wr/100) * loss_pnl)

    # Drawdown
    balances = [balance_start] + df["balance_after"].tolist()
    peak = balance_start
    max_dd = 0.0
    for b in balances:
        if b > peak:
            peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        "trades":     total,
        "wr":         round(wr, 1),
        "avg_r":      round(avg_r, 2),
        "avg_pnl":    round(avg_pnl, 2),
        "total_pnl":  round(total_pnl, 2),
        "max_dd":     round(max_dd, 1),
        "expectancy": round(expectancy, 2),
    }


def run_comparison(
    df_btc:          pd.DataFrame,
    df_gold:         pd.DataFrame,
    df_nas:          pd.DataFrame,
    strategies:      list[BTCStrategy] | None = None,
    allowed_hours:   set[int] | None = None,
) -> None:
    """
    Test all strategies with and without inter-market filter.
    Print a ranked comparison table.
    """
    if strategies is None:
        strategies = ALL_STRATEGIES
    if allowed_hours is None:
        allowed_hours = set(range(KZ_START_UTC, KZ_END_UTC))

    SEP  = "=" * 90
    LINE = "-" * 90

    print()
    print(SEP)
    print("BTC STRATEGY COMPARISON")
    print(f"Session : UTC {min(allowed_hours):02d}:00 - {max(allowed_hours)+1:02d}:00")
    print(f"Risk    : {RISK_PCT*100:.0f}% per trade  |  TP1={TP1_RR:.0f}R  TP2={TP2_RR:.0f}R  MaxHold={MAX_HOLD_BARS}h")
    print(SEP)

    all_results: list[dict] = []

    for strat in strategies:
        print(f"\nTesting: {strat.name}...", end=" ", flush=True)

        # A. Without inter-market filter
        trades_raw = _simulate(strat, df_btc, df_gold, df_nas,
                               use_intermarket=False,
                               allowed_hours=allowed_hours)
        stats_raw  = _stats(trades_raw)

        # B. With inter-market filter
        trades_fil = _simulate(strat, df_btc, df_gold, df_nas,
                               use_intermarket=True,
                               allowed_hours=allowed_hours)
        stats_fil  = _stats(trades_fil)

        print(f"done  ({stats_raw['trades']} raw / {stats_fil['trades']} filtered trades)")

        all_results.append({
            "strategy":     strat.name,
            "raw":          stats_raw,
            "filtered":     stats_fil,
            "raw_trades":   trades_raw,
            "filtered_trades": trades_fil,
        })

    # ── Print comparison tables ───────────────────────────────────────────────
    for label, key, trades_key in [
        ("WITHOUT inter-market filter (pure strategy signal)", "raw",      "raw_trades"),
        ("WITH inter-market filter (Gold + NAS100)",           "filtered", "filtered_trades"),
    ]:
        print()
        print(f"{'─'*90}")
        print(f"  {label}")
        print(f"{'─'*90}")
        print(f"  {'Strategy':<28} {'Trades':>7} {'WR%':>7} {'Avg R':>7} "
              f"{'Avg PnL':>10} {'Total PnL':>12} {'MaxDD':>8} {'Expect':>9}")
        print(f"  {LINE}")

        # Sort by Total PnL descending
        sorted_r = sorted(all_results, key=lambda x: x[key]["total_pnl"], reverse=True)

        for rank, r in enumerate(sorted_r, 1):
            s   = r[key]
            pfx = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
            print(f"  {pfx} {r['strategy']:<26} "
                  f"{s['trades']:>7} "
                  f"{s['wr']:>6.1f}% "
                  f"{s['avg_r']:>+6.2f}R "
                  f"${s['avg_pnl']:>+8.2f} "
                  f"${s['total_pnl']:>+10.2f} "
                  f"{s['max_dd']:>7.1f}% "
                  f"${s['expectancy']:>+7.2f}")

    # ── Overall winner ────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("STRATEGY WINNER SUMMARY")
    print(SEP)

    def _rank(key):
        return sorted(all_results, key=lambda x: x[key]["total_pnl"], reverse=True)

    raw_winner = _rank("raw")[0]
    fil_winner = _rank("filtered")[0]

    print(f"  Best pure strategy  : {raw_winner['strategy']}")
    print(f"    Trades={raw_winner['raw']['trades']}  "
          f"WR={raw_winner['raw']['wr']}%  "
          f"Avg R={raw_winner['raw']['avg_r']:+.2f}  "
          f"Total PnL=${raw_winner['raw']['total_pnl']:+,.2f}")
    print()
    print(f"  Best with IM filter : {fil_winner['strategy']}")
    print(f"    Trades={fil_winner['filtered']['trades']}  "
          f"WR={fil_winner['filtered']['wr']}%  "
          f"Avg R={fil_winner['filtered']['avg_r']:+.2f}  "
          f"Total PnL=${fil_winner['filtered']['total_pnl']:+,.2f}")

    # Does the IM filter help?
    print()
    print("  DOES INTER-MARKET FILTER ADD EDGE?")
    print(f"  {'Strategy':<28} {'Raw PnL':>12} {'Filtered PnL':>14} {'Delta':>10} {'Worth it?':>10}")
    print(f"  {'-'*75}")
    for r in all_results:
        raw_pnl = r["raw"]["total_pnl"]
        fil_pnl = r["filtered"]["total_pnl"]
        delta   = fil_pnl - raw_pnl
        worth   = "YES ++" if delta > 500 else ("YES" if delta > 0 else "NO")
        print(f"  {r['strategy']:<28} ${raw_pnl:>+10,.2f} ${fil_pnl:>+12,.2f} "
              f"${delta:>+8,.2f} {worth:>10}")

    # Per-strategy exit breakdown
    print()
    print(SEP)
    print("EXIT REASON BREAKDOWN (filtered trades)")
    print(SEP)
    for r in all_results:
        t_list = r["filtered_trades"]
        if not t_list:
            continue
        df_t = pd.DataFrame(t_list)
        exits = df_t["exit_reason"].value_counts()
        total = len(df_t)
        parts = [f"{reason}:{count}({count/total*100:.0f}%)"
                 for reason, count in exits.items()]
        print(f"  {r['strategy']:<28}: {' | '.join(parts)}")

    print(SEP)
    print()
    print("NEXT STEPS:")
    print(f"  1. Use '{fil_winner['strategy']}' as the primary strategy")
    print(f"  2. Update btc_research/settings.py KZ_START/END if you changed session")
    print(f"  3. Run full backtest: python btc_research/run_backtest.py")
    print(f"  4. Tune MIN_CONFLUENCE_SCORE based on score quartile results")
    print(SEP)
