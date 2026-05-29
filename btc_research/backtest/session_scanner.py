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
