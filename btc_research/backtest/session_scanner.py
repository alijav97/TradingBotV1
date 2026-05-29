"""
btc_research/backtest/session_scanner.py — Find the optimal trading session for BTC.

Instead of assuming 13:00-17:00 UTC is best (that's the WTI kill-zone),
this scanner runs the strategy on EVERY hour of the day and shows which
hours/sessions actually produce the best results for BTC.

Output:
  - Hour-by-hour WR and expectancy heat map
  - Session-block comparison (Asia / EU / US Open / US Mid / Asia Night)
  - Recommendation: best session window to use

Usage:
    from btc_research.backtest.session_scanner import run_session_scan
    run_session_scan(df_btc, df_gold, df_nas)
"""
from __future__ import annotations

from datetime import timezone
from typing import Callable

import pandas as pd

from btc_research.settings import (
    STARTING_BALANCE, RISK_PCT, TP1_RR, TP2_RR, MAX_HOLD_BARS,
)
from btc_research.strategy.confluence import score_bar as _score_bar_with_gate
from btc_research.factors.gold_factor   import compute_gold_factor
from btc_research.factors.nasdaq_factor import compute_nasdaq_factor
from btc_research.factors.btc_momentum  import compute_btc_momentum


# ── Pre-defined session blocks to compare ────────────────────────────────────
SESSION_BLOCKS = {
    "Asia Night  (00-04 UTC)": (0,  4),
    "Asia Open   (04-08 UTC)": (4,  8),
    "EU Session  (08-12 UTC)": (8,  12),
    "EU/US Cross (12-14 UTC)": (12, 14),
    "US Open     (13-17 UTC)": (13, 17),
    "US Mid      (17-21 UTC)": (17, 21),
    "US Late     (21-24 UTC)": (21, 24),
}


def _lot_size(balance: float, entry: float, sl: float) -> float:
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0.001
    return round(max((balance * RISK_PCT) / sl_dist, 0.001), 6)


def _score_bar_no_gate(
    bar_time:  pd.Timestamp,
    bar_close: float,
    direction: str,
    df_btc:    pd.DataFrame,
    df_gold:   pd.DataFrame,
    df_nas:    pd.DataFrame,
    min_score: float = 3.0,
) -> dict:
    """
    Same as score_bar() in confluence.py but WITHOUT the time-of-day gate.
    Still requires morning range and breakout condition.
    The 'morning range' here is defined as the 5-hour window BEFORE the entry bar.
    """
    from btc_research.strategy.confluence import _morning_range as _mr

    result = {
        "signal": False, "score": 0.0,
        "entry": bar_close, "sl": 0.0, "tp1": 0.0, "tp2": 0.0,
        "factors": {}, "blocked_by": "",
    }

    if bar_time.tzinfo is None:
        bar_time = bar_time.replace(tzinfo=timezone.utc)

    # ── Reference range: 5 hours before current bar ───────────────────────────
    # For session scanning, we define the "range" as the 5 H1 bars preceding
    # the entry bar (instead of fixed 08-13 UTC window).
    # This allows us to evaluate any hour fairly.
    if len(df_btc) < 8:
        return result

    recent = df_btc.tail(8)
    range_bars = recent.iloc[:-1]   # all but current bar
    if len(range_bars) < 3:
        result["blocked_by"] = "not enough range bars"
        return result

    mr_high = float(range_bars["high"].max())
    mr_low  = float(range_bars["low"].min())
    is_long = direction.lower() in ("long", "buy")

    if is_long and bar_close <= mr_high:
        result["blocked_by"] = f"no breakout: close {bar_close:.2f} <= range_high {mr_high:.2f}"
        return result
    if not is_long and bar_close >= mr_low:
        result["blocked_by"] = f"no breakout: close {bar_close:.2f} >= range_low {mr_low:.2f}"
        return result

    entry   = bar_close
    sl      = mr_low  if is_long else mr_high
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return result

    tp1 = entry + TP1_RR * sl_dist if is_long else entry - TP1_RR * sl_dist
    tp2 = entry + TP2_RR * sl_dist if is_long else entry - TP2_RR * sl_dist

    # Factor scoring
    btc_f  = compute_btc_momentum(df_btc, bar_time, direction)
    gold_f = compute_gold_factor(df_gold, bar_time)
    nas_f  = compute_nasdaq_factor(df_nas, bar_time)

    d_mult = 1.0 if is_long else -1.0
    total_score = (
        btc_f["score"]  * 1.0         +
        gold_f["score"] * 0.5 * d_mult +
        nas_f["score"]  * 0.5 * d_mult
    )

    result.update({
        "score": round(total_score, 2),
        "entry": round(entry, 2),
        "sl":    round(sl, 2),
        "tp1":   round(tp1, 2),
        "tp2":   round(tp2, 2),
        "factors": {
            "btc": btc_f, "gold": gold_f, "nas": nas_f,
            "mr_high": mr_high, "mr_low": mr_low,
        },
    })

    if total_score < min_score:
        result["blocked_by"] = f"score {total_score:.1f} < {min_score}"
        return result

    result["signal"] = True
    return result


def _simulate_trades(
    df_btc:    pd.DataFrame,
    df_gold:   pd.DataFrame,
    df_nas:    pd.DataFrame,
    allowed_hours: set[int],
    min_score: float = 3.0,
    verbose:   bool  = False,
) -> list[dict]:
    """
    Run a simulation restricted to allowed_hours (set of UTC hours 0-23).
    Returns list of closed trades.
    """
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
                    t["sl"]      = t["entry"]
                    partial = (t["tp1"] - t["entry"] if is_long else t["entry"] - t["tp1"]) * t["lots"] * 0.5
                    t["tp1_pnl"] = round(partial, 2)
                    balance += partial

            if closed and exit_price is not None:
                pd_ = (exit_price - t["entry"]) if is_long else (t["entry"] - exit_price)
                rem = t["lots"] * (0.5 if t.get("tp1_hit") else 1.0)
                pnl = pd_ * rem + t.get("tp1_pnl", 0.0)
                sld = abs(t["entry"] - t["original_sl"])
                rr  = pd_ / sld if sld > 0 else 0.0
                balance = max(balance + pd_ * rem, 1.0)
                trades.append({
                    "id": t["id"], "direction": t["direction"],
                    "entry": t["entry"], "exit": round(exit_price, 2),
                    "pnl_usd": round(pnl, 2), "r_multiple": round(rr, 2),
                    "bars_held": bh, "exit_reason": exit_reason,
                    "score": t.get("score", 0),
                    "open_hour_utc": t["open_hour_utc"],
                    "open_time": t["open_time"],
                    "balance_after": round(balance, 2),
                })
                open_trade = None

        # New entry — only in allowed hours
        if open_trade is None and bar_time.hour in allowed_hours:
            df_window = df_btc.iloc[: i + 1]
            for direction in ("long", "short"):
                res = _score_bar_no_gate(
                    bar_time, bar_close, direction,
                    df_window, df_gold, df_nas, min_score,
                )
                if not res["signal"]:
                    continue
                lots     = _lot_size(balance, res["entry"], res["sl"])
                trade_id = f"BTC-{len(trades)+1:04d}"
                open_trade = {
                    "id": trade_id, "direction": direction,
                    "entry": res["entry"], "sl": res["sl"],
                    "original_sl": res["sl"],
                    "tp1": res["tp1"], "tp2": res["tp2"],
                    "lots": lots, "tp1_hit": False, "tp1_pnl": 0.0,
                    "bars_held": 0,
                    "open_hour_utc": bar_time.hour,
                    "open_time": bar_time.isoformat(),
                    "score": res["score"],
                }
                break

    return trades


# ── Public API ────────────────────────────────────────────────────────────────

def run_session_scan(
    df_btc:    pd.DataFrame,
    df_gold:   pd.DataFrame,
    df_nas:    pd.DataFrame,
    min_score: float = 3.0,
) -> None:
    """
    Scan all 24 hours + all session blocks and print a full comparison table.
    This answers: "Which session gives the best BTC trades?"
    """
    SEP  = "=" * 72
    LINE = "-" * 72

    print()
    print(SEP)
    print("BTC SESSION ANALYSIS — Which hours give the best trades?")
    print(SEP)
    print("Running simulations for each hour and session block...")
    print("(This may take a few minutes — testing 24 individual hours)")
    print()

    # ── 1. Hour-by-hour analysis ───────────────────────────────────────────────
    hour_results: list[dict] = []
    for h in range(24):
        trades = _simulate_trades(df_btc, df_gold, df_nas, {h}, min_score)
        if not trades:
            hour_results.append({
                "hour": h, "trades": 0, "wr": 0.0,
                "avg_pnl": 0.0, "total_pnl": 0.0, "avg_r": 0.0,
            })
            continue
        df_t = pd.DataFrame(trades)
        wr   = (df_t["pnl_usd"] > 0).mean() * 100
        hour_results.append({
            "hour":      h,
            "trades":    len(df_t),
            "wr":        round(wr, 1),
            "avg_pnl":   round(df_t["pnl_usd"].mean(), 2),
            "total_pnl": round(df_t["pnl_usd"].sum(), 2),
            "avg_r":     round(df_t["r_multiple"].mean(), 2),
        })
        print(f"  Hour {h:02d}:00 UTC  done  ({len(df_t)} trades)", end="\r")

    print(" " * 60, end="\r")  # clear progress line

    # Print heat map
    df_h = pd.DataFrame(hour_results)
    print(f"{'UTC Hour':<12} {'Trades':>7} {'Win%':>7} {'Avg R':>7} "
          f"{'Avg PnL':>10} {'Total PnL':>12}")
    print(LINE)

    sessions_map = {
        range(0,  4): "Asia Night ",
        range(4,  8): "Asia Open  ",
        range(8,  12): "EU Session ",
        range(12, 14): "EU/US Cross",
        range(13, 17): "US Open    ",
        range(17, 21): "US Mid     ",
        range(21, 24): "US Late    ",
    }

    def _session_label(hour: int) -> str:
        for r, label in sessions_map.items():
            if hour in r:
                return label
        return "           "

    for _, row in df_h.iterrows():
        h = int(row["hour"])
        bar = "#" * min(int(row["wr"] / 5), 20) if row["trades"] > 0 else ""
        lbl = _session_label(h)
        print(f"  {h:02d}:00 UTC "
              f"[{lbl}] "
              f"{int(row['trades']):>5} "
              f"{row['wr']:>6.1f}% "
              f"{row['avg_r']:>+6.2f}R "
              f"${row['avg_pnl']:>+8.2f} "
              f"${row['total_pnl']:>+10.2f}  {bar}")

    # ── 2. Session block comparison ────────────────────────────────────────────
    print()
    print(SEP)
    print("SESSION BLOCK COMPARISON")
    print(SEP)
    print(f"{'Session':<25} {'Trades':>7} {'Win%':>7} {'Avg R':>7} "
          f"{'Avg PnL':>10} {'Total PnL':>12}")
    print(LINE)

    block_results = []
    for name, (start, end) in SESSION_BLOCKS.items():
        hours  = set(range(start, end))
        trades = _simulate_trades(df_btc, df_gold, df_nas, hours, min_score)
        if not trades:
            block_results.append({"session": name, "trades": 0})
            print(f"  {name:<25} {'0':>7} {'—':>7}")
            continue
        df_t = pd.DataFrame(trades)
        wr   = (df_t["pnl_usd"] > 0).mean() * 100
        rec  = {
            "session":   name,
            "trades":    len(df_t),
            "wr":        round(wr, 1),
            "avg_pnl":   round(df_t["pnl_usd"].mean(), 2),
            "total_pnl": round(df_t["pnl_usd"].sum(), 2),
            "avg_r":     round(df_t["r_multiple"].mean(), 2),
        }
        block_results.append(rec)
        print(f"  {name:<25} {len(df_t):>7} {wr:>6.1f}% "
              f"{rec['avg_r']:>+6.2f}R "
              f"${rec['avg_pnl']:>+8.2f} "
              f"${rec['total_pnl']:>+10.2f}")

    # ── 3. Best session recommendation ────────────────────────────────────────
    valid = [b for b in block_results if b.get("trades", 0) >= 10]
    if valid:
        best_wr  = max(valid, key=lambda x: x["wr"])
        best_pnl = max(valid, key=lambda x: x["total_pnl"])
        best_r   = max(valid, key=lambda x: x["avg_r"])
        print()
        print(SEP)
        print("RECOMMENDATION")
        print(SEP)
        print(f"  Best WR        : {best_wr['session'].strip():<25} "
              f"({best_wr['wr']:.1f}% WR, {best_wr['trades']} trades)")
        print(f"  Best Total PnL : {best_pnl['session'].strip():<25} "
              f"(${best_pnl['total_pnl']:+,.2f}, {best_pnl['trades']} trades)")
        print(f"  Best Avg R     : {best_r['session'].strip():<25} "
              f"({best_r['avg_r']:+.2f}R per trade)")
        print()
        print("  Use the session with best WR AND decent trade count.")
        print("  Then update KZ_START_UTC / KZ_END_UTC in btc_research/settings.py")
        print("  and re-run: python btc_research/run_backtest.py")
    print(SEP)


# ── Strategy-aware session scanner ───────────────────────────────────────────

def run_combined_session_scan(
    df_btc:    pd.DataFrame,
    df_gold:   pd.DataFrame,
    df_nas:    pd.DataFrame,
    strategy:  "BTCStrategy | None" = None,
) -> None:
    """
    Scan all 24 hours + session blocks using any BTCStrategy (default: CombinedStrategy).
    Unlike run_session_scan() which uses the old confluence engine, this function uses
    the actual strategy classes — so results reflect the REAL combined strategy behaviour.

    For CombinedStrategy, selective IM filtering is applied automatically
    (only Volatility Breakout sub-strategy gets the Gold/NAS gate).

    Usage:
        from btc_research.backtest.session_scanner import run_combined_session_scan
        run_combined_session_scan(df_btc, df_gold, df_nas)
    """
    from btc_research.strategies.combined import CombinedStrategy, _FILTER_STRATEGIES
    from btc_research.backtest.strategy_comparison import _simulate, _stats

    if strategy is None:
        strategy = CombinedStrategy(atr_multiplier=1.2, close_zone=0.45, range_bars=6)

    is_combined = "Combined" in strategy.name
    filter_names = _FILTER_STRATEGIES if is_combined else None

    SEP  = "=" * 72
    LINE = "-" * 72

    print()
    print(SEP)
    print(f"SESSION SCAN — {strategy.name}")
    if is_combined:
        print("  IM filter: Volatility sub-strategy only (Morning Range + Swing = unfiltered)")
    print(SEP)
    print("Testing all 24 hours and session blocks... (may take a few minutes)")
    print()

    # ── 1. Hour-by-hour ───────────────────────────────────────────────────────
    hour_results: list[dict] = []
    for h in range(24):
        trades = _simulate(
            strategy, df_btc, df_gold, df_nas,
            use_intermarket=False,
            filter_strategy_names=filter_names,
            allowed_hours={h},
        )
        if not trades:
            hour_results.append({
                "hour": h, "trades": 0, "wr": 0.0,
                "avg_pnl": 0.0, "total_pnl": 0.0, "avg_r": 0.0,
            })
            print(f"  Hour {h:02d}:00 UTC  — 0 trades", end="\r")
            continue
        df_t = pd.DataFrame(trades)
        wr   = (df_t["pnl_usd"] > 0).mean() * 100
        hour_results.append({
            "hour":      h,
            "trades":    len(df_t),
            "wr":        round(wr, 1),
            "avg_pnl":   round(df_t["pnl_usd"].mean(), 2),
            "total_pnl": round(df_t["pnl_usd"].sum(), 2),
            "avg_r":     round(df_t["r_multiple"].mean(), 2),
        })
        print(f"  Hour {h:02d}:00 UTC  done  ({len(df_t)} trades)", end="\r")

    print(" " * 65, end="\r")

    # Heat map
    sessions_map = {
        range(0,  4):  "Asia Night ",
        range(4,  8):  "Asia Open  ",
        range(8,  12): "EU Session ",
        range(12, 14): "EU/US Cross",
        range(13, 17): "US Open    ",
        range(17, 21): "US Mid     ",
        range(21, 24): "US Late    ",
    }

    def _session_label(hour: int) -> str:
        for r, label in sessions_map.items():
            if hour in r:
                return label
        return "           "

    print(f"{'UTC Hour':<12} {'Session':<14} {'Trades':>7} {'Win%':>7} "
          f"{'Avg R':>7} {'Avg PnL':>10} {'Total PnL':>12}")
    print(LINE)
    for row in hour_results:
        h   = int(row["hour"])
        lbl = _session_label(h)
        bar = "#" * min(int(row["wr"] / 5), 20) if row["trades"] > 0 else "—"
        print(f"  {h:02d}:00 UTC  [{lbl}] "
              f"{int(row['trades']):>5} "
              f"{row['wr']:>6.1f}% "
              f"{row['avg_r']:>+6.2f}R "
              f"${row['avg_pnl']:>+8.2f} "
              f"${row['total_pnl']:>+10.2f}  {bar}")

    # ── 2. Session block comparison ───────────────────────────────────────────
    print()
    print(SEP)
    print("SESSION BLOCK COMPARISON")
    print(SEP)
    print(f"{'Session':<25} {'Trades':>7} {'WR%':>7} {'Avg R':>7} "
          f"{'Avg PnL':>10} {'Total PnL':>12} {'MaxDD':>8}")
    print(LINE)

    block_results = []
    for name, (start, end) in SESSION_BLOCKS.items():
        hours  = set(range(start, end))
        trades = _simulate(
            strategy, df_btc, df_gold, df_nas,
            use_intermarket=False,
            filter_strategy_names=filter_names,
            allowed_hours=hours,
        )
        if not trades:
            block_results.append({"session": name, "trades": 0,
                                   "wr": 0.0, "total_pnl": 0.0, "avg_r": 0.0})
            print(f"  {name:<25} {'0':>7}")
            continue
        st = _stats(trades)
        block_results.append({"session": name, **st})
        print(f"  {name:<25} {st['trades']:>7} {st['wr']:>6.1f}% "
              f"{st['avg_r']:>+6.2f}R "
              f"${st['avg_pnl']:>+8.2f} "
              f"${st['total_pnl']:>+10.2f} "
              f"{st['max_dd']:>7.1f}%")

    # ── 3. Recommendation ────────────────────────────────────────────────────
    valid = [b for b in block_results if b.get("trades", 0) >= 10]
    if valid:
        best_wr  = max(valid, key=lambda x: x["wr"])
        best_pnl = max(valid, key=lambda x: x["total_pnl"])
        best_r   = max(valid, key=lambda x: x["avg_r"])
        # Composite score: normalise each metric and sum
        max_wr   = max(b["wr"] for b in valid) or 1
        max_pnl  = max(b["total_pnl"] for b in valid) or 1
        max_r    = max(b["avg_r"] for b in valid) or 1
        for b in valid:
            b["composite"] = (b["wr"] / max_wr * 0.4 +
                              b["total_pnl"] / max_pnl * 0.4 +
                              b["avg_r"] / max_r * 0.2)
        best_overall = max(valid, key=lambda x: x["composite"])

        print()
        print(SEP)
        print("RECOMMENDATION")
        print(SEP)
        print(f"  Best WR        : {best_wr['session'].strip():<25} "
              f"({best_wr['wr']:.1f}% WR, {best_wr['trades']} trades)")
        print(f"  Best Total PnL : {best_pnl['session'].strip():<25} "
              f"(${best_pnl['total_pnl']:+,.2f}, {best_pnl['trades']} trades)")
        print(f"  Best Avg R     : {best_r['session'].strip():<25} "
              f"({best_r['avg_r']:+.2f}R per trade)")
        print()
        print(f"  ★ BEST OVERALL (WR 40% + PnL 40% + AvgR 20%) :")
        print(f"      {best_overall['session'].strip()}")
        print(f"      {best_overall['trades']} trades  |  "
              f"WR={best_overall['wr']:.1f}%  |  "
              f"Avg R={best_overall['avg_r']:+.2f}  |  "
              f"Total PnL=${best_overall['total_pnl']:+,.2f}  |  "
              f"MaxDD={best_overall.get('max_dd', 0):.1f}%")
        print()
        print("  Update KZ_START_UTC / KZ_END_UTC in btc_research/settings.py")
        print("  then re-run: python btc_research/run_backtest.py --compare")
    print(SEP)


# ── Per-strategy individual session scanner ──────────────────────────────────

def run_per_strategy_scan(
    df_btc:     pd.DataFrame,
    df_gold:    pd.DataFrame,
    df_nas:     pd.DataFrame,
    strategies: "list[BTCStrategy] | None" = None,
) -> None:
    """
    Run the session scanner for EACH strategy independently.
    Shows which session window is optimal for each strategy, so we can:
      1. See if strategies share the same best session (can stay in Combined)
      2. Identify strategies that belong in a completely different session
      3. Decide which strategies to keep in Combined and which to run separately

    Output:
      - Per-strategy: best session block + hour by WR, AvgR, PnL
      - Summary table: one row per strategy with their optimal session
      - Recommendation: which strategies are session-compatible
    """
    from btc_research.strategies.combined import CombinedStrategy, _FILTER_STRATEGIES
    from btc_research.backtest.strategy_comparison import _simulate, _stats
    from btc_research.backtest.strategy_comparison import ALL_STRATEGIES

    if strategies is None:
        strategies = ALL_STRATEGIES

    SEP  = "=" * 80
    LINE = "-" * 80

    print()
    print(SEP)
    print("PER-STRATEGY SESSION SCANNER")
    print("Finding the optimal trading session for each strategy independently")
    print(SEP)
    print(f"Testing {len(strategies)} strategies × 7 session blocks each...")
    print("(This will take several minutes)\n")

    summary: list[dict] = []

    for strat in strategies:
        is_combined = "Combined" in strat.name
        filter_names = _FILTER_STRATEGIES if is_combined else None

        print(f"\n{'─'*80}")
        print(f"  STRATEGY: {strat.name}")
        print(f"{'─'*80}")
        print(f"  {'Session':<25} {'Trades':>7} {'WR%':>7} {'Avg R':>8} "
              f"{'Total PnL':>12} {'MaxDD':>8}")
        print(f"  {LINE}")

        block_results = []
        for name, (start, end) in SESSION_BLOCKS.items():
            hours  = set(range(start, end))
            trades = _simulate(
                strat, df_btc, df_gold, df_nas,
                use_intermarket=not is_combined,
                filter_strategy_names=filter_names,
                allowed_hours=hours,
            )
            if not trades:
                block_results.append({"session": name, "trades": 0,
                                       "wr": 0.0, "total_pnl": 0.0,
                                       "avg_r": 0.0, "max_dd": 0.0})
                print(f"  {name:<25} {'0':>7}")
                continue
            st = _stats(trades)
            block_results.append({"session": name, **st})
            # Highlight the row if it looks good
            flag = " ◄" if st["wr"] >= 45 and st["total_pnl"] > 0 and st["max_dd"] < 25 else ""
            print(f"  {name:<25} {st['trades']:>7} {st['wr']:>6.1f}% "
                  f"{st['avg_r']:>+7.2f}R "
                  f"${st['total_pnl']:>+10,.2f} "
                  f"{st['max_dd']:>7.1f}%{flag}")

        # Find best session for this strategy
        valid = [b for b in block_results if b.get("trades", 0) >= 10]
        if not valid:
            print(f"\n  ⚠  No sessions with ≥10 trades. Strategy may not work on H1.")
            summary.append({
                "strategy": strat.name,
                "best_session": "N/A",
                "best_wr": 0.0,
                "best_pnl": 0.0,
                "best_r": 0.0,
                "best_dd": 0.0,
                "best_trades": 0,
            })
            continue

        # Score: 40% WR + 40% PnL + 20% AvgR (normalised)
        max_wr  = max(b["wr"] for b in valid) or 1
        max_pnl = max(b["total_pnl"] for b in valid) or 1
        max_r   = max(b["avg_r"] for b in valid) or 1
        for b in valid:
            b["score"] = (max(b["wr"], 0) / max_wr * 0.4 +
                          max(b["total_pnl"], 0) / max(max_pnl, 1) * 0.4 +
                          max(b["avg_r"], 0) / max(max_r, 1) * 0.2)
        best = max(valid, key=lambda x: x["score"])

        print(f"\n  ★ Best session : {best['session'].strip()}")
        print(f"    {best['trades']} trades | WR={best['wr']:.1f}% | "
              f"AvgR={best['avg_r']:+.2f}R | "
              f"PnL=${best['total_pnl']:+,.2f} | "
              f"MaxDD={best['max_dd']:.1f}%")

        summary.append({
            "strategy":     strat.name,
            "best_session": best["session"].strip(),
            "best_wr":      best["wr"],
            "best_pnl":     best["total_pnl"],
            "best_r":       best["avg_r"],
            "best_dd":      best["max_dd"],
            "best_trades":  best["trades"],
        })

    # ── Summary comparison table ──────────────────────────────────────────────
    print()
    print(SEP)
    print("SUMMARY — OPTIMAL SESSION PER STRATEGY")
    print(SEP)
    print(f"  {'Strategy':<28} {'Best Session':<25} {'Trades':>7} "
          f"{'WR%':>7} {'Avg R':>8} {'PnL':>12} {'MaxDD':>8}")
    print(f"  {LINE}")
    for s in sorted(summary, key=lambda x: x["best_pnl"], reverse=True):
        print(f"  {s['strategy']:<28} {s['best_session']:<25} "
              f"{s['best_trades']:>7} "
              f"{s['best_wr']:>6.1f}% "
              f"{s['best_r']:>+7.2f}R "
              f"${s['best_pnl']:>+10,.2f} "
              f"{s['best_dd']:>7.1f}%")

    # ── Session compatibility analysis ───────────────────────────────────────
    print()
    print(SEP)
    print("SESSION COMPATIBILITY — Which strategies can share the same window?")
    print(SEP)

    # Group strategies by their best session
    from collections import defaultdict
    by_session: dict = defaultdict(list)
    for s in summary:
        if s["best_session"] != "N/A":
            by_session[s["best_session"]].append(s["strategy"])

    for session, strat_names in sorted(by_session.items(),
                                        key=lambda x: len(x[1]), reverse=True):
        print(f"  {session:<25}: {', '.join(strat_names)}")

    print()
    print("RECOMMENDATION:")
    print("  Strategies sharing the same best session → keep together in Combined")
    print("  Strategies with DIFFERENT best sessions  → run as separate bots OR")
    print("  widen the Combined session window to cover both optimal windows")
    print()
    print("  Next: update settings.py with the best combined session window,")
    print("  then re-run: python btc_research/run_backtest.py --compare")
    print(SEP)
