"""
btc_research/eth_bot/backtest/eth_vbswing_streaks.py — losing-streak forensics.

User question (before risking real $500 at 10%): "What are the chances I hit the
maximum losing streak? What were the CONDITIONS when it happened historically?
How long are the streaks and how likely is each, at 10% risk?"

This script answers it three ways, all from the REAL VBSwing-on-ETH trade history
(best config from eth_vbswing.py: BTC hours [1,2,3,8], BTC 3/2/3 sizing — note the
win/LOSS SEQUENCE is identical for any risk %, since sizing never changes which
trades are taken or whether each hits SL vs TP):

  PART A — HISTORICAL: every losing streak that actually occurred, its length,
           dates, and the drawdown it WOULD cause at 10% flat risk. Full per-trade
           detail for the single worst streak (ADX, direction, sub-strategy, R).

  PART B — CONDITIONS: for the worst streaks, what the market was doing — average
           ADX, long/short mix, which sub-strategy fired, and ETH's net move across
           the streak window (chop vs trend).

  PART C — PROBABILITY: bootstrap 6- and 12-month paths from the real loss pool and
           report P(max losing streak >= k) plus the equity drawdown each k costs at
           10% risk. This is the honest "how bad, how likely" table.

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.eth_vbswing_streaks
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btc_research.eth_bot.backtest.eth_vbswing import (
    _load_eth, _indicators, simulate, VBSwingStrategy, STARTING_BALANCE,
)

# Best config from eth_vbswing.py PASS 1.
HOURS      = [1, 2, 3, 8]
RISK_TUPLE = (0.03, 0.02, 0.03)     # 3/2/3 — sequence is risk-independent anyway
LIVE_RISK  = 0.10                    # the aggressive setting being considered
HORIZONS   = [6, 12]                 # months
N_PATHS    = 100_000
SEED       = 7
MIN_SHOW   = 4                       # list historical streaks of this length or more


def _bar(c: str = "-", w: int = 92) -> str:
    return c * w


def _streaks(trades: list[dict]) -> list[dict]:
    """Find consecutive runs of losing trades (pnl_usd <= 0, matching WR/MaxCL)."""
    out, cur = [], []
    for t in trades:
        if t["pnl_usd"] <= 0:
            cur.append(t)
        else:
            if cur:
                out.append(cur); cur = []
    if cur:
        out.append(cur)
    return [_streak_info(s) for s in out]


def _streak_info(s: list[dict]) -> dict:
    rs = [float(t["r_multiple"]) for t in s]
    dd = np.prod([1.0 + LIVE_RISK * r for r in rs]) - 1.0     # equity impact at 10%
    return dict(
        length=len(s),
        start=str(s[0]["open_time"])[:16],
        end=str(s[-1]["open_time"])[:16],
        avg_adx=np.mean([float(t["adx_at_entry"]) for t in s]),
        n_long=sum(1 for t in s if t["direction"] == "long"),
        n_short=sum(1 for t in s if t["direction"] == "short"),
        strats=[t.get("strategy_used", "?") for t in s],
        dd10=dd * 100,
        trades=s,
    )


def main() -> None:
    df = _load_eth()
    atr, ema200, adx = _indicators(df)
    strat = VBSwingStrategy()
    trades = simulate(df, atr, ema200, adx, strat, HOURS, RISK_TUPLE)

    n = len(trades)
    losers = [t for t in trades if t["pnl_usd"] <= 0]
    wr = (1 - len(losers) / n) * 100
    is_loss = np.array([t["pnl_usd"] <= 0 for t in trades], dtype=bool)
    tpm = n / max((pd.Timestamp(trades[-1]["open_time"])
                   - pd.Timestamp(trades[0]["open_time"])).days / 30.44, 1.0)

    print()
    print(_bar("="))
    print("  ETH VBSwing — LOSING-STREAK FORENSICS  (BTC hours [1,2,3,8], one-position)")
    print(f"  {n} trades {str(trades[0]['open_time'])[:10]}→{str(trades[-1]['open_time'])[:10]} "
          f"| WR {wr:.1f}% | {len(losers)} losers | ~{tpm:.1f} trades/mo")
    print(_bar("="))

    streaks = _streaks(trades)
    streaks.sort(key=lambda x: x["length"], reverse=True)
    max_len = streaks[0]["length"]

    # ---- count by length -----------------------------------------------------
    from collections import Counter
    by_len = Counter(s["length"] for s in streaks)
    print()
    print("  PART A — every losing streak that HISTORICALLY occurred")
    print(f"  {'streak len':>11} {'#times':>7} {'~drawdown@10% (all -1R)':>26}")
    print(_bar("-", 50))
    for L in sorted(by_len, reverse=True):
        dd = (0.9 ** L - 1) * 100
        print(f"  {L:>11} {by_len[L]:>7}        {dd:>+9.1f}%")
    print(_bar("-", 50))

    # ---- list the longer streaks with dates + real 10% drawdown --------------
    print()
    print(f"  Streaks of length >= {MIN_SHOW} (with REAL R, actual 10%-risk equity hit):")
    print(f"  {'len':>4} {'start':<17} {'end':<17} {'avgADX':>7} {'L/S':>6} {'dd@10%':>8}")
    print(_bar("-", 70))
    for s in streaks:
        if s["length"] < MIN_SHOW:
            continue
        print(f"  {s['length']:>4} {s['start']:<17} {s['end']:<17} "
              f"{s['avg_adx']:>7.1f} {s['n_long']}L/{s['n_short']}S {s['dd10']:>+7.1f}%")
    print(_bar("-", 70))

    # ---- PART B: the WORST streak, trade by trade + conditions ----------------
    worst = streaks[0]
    print()
    print(_bar("="))
    print(f"  PART B — THE WORST STREAK: {worst['length']} losers in a row")
    print(f"  {worst['start']}  →  {worst['end']}   "
          f"(real 10%-risk equity hit: {worst['dd10']:+.1f}%)")
    print(_bar("="))
    sc = Counter(worst["strats"])
    print(f"  Conditions: avg ADX {worst['avg_adx']:.1f} | "
          f"{worst['n_long']} long / {worst['n_short']} short | "
          f"sub-strategy: {dict(sc)}")
    # ETH regime across the streak window
    t0 = pd.Timestamp(worst['trades'][0]['open_time'])
    t1 = pd.Timestamp(worst['trades'][-1]['open_time'])
    win = df[(df.index >= t0) & (df.index <= t1)]
    if len(win) > 1:
        net = (win['close'].iloc[-1] / win['close'].iloc[0] - 1) * 100
        rng = (win['high'].max() / win['low'].min() - 1) * 100
        span_days = (t1 - t0).days
        print(f"  ETH over the {span_days}-day window: net move {net:+.1f}% inside a "
              f"{rng:.1f}% range  ->  {'CHOP (whipsaw)' if abs(net) < rng/2 else 'TREND against entries'}")
    print(_bar("-", 92))
    print(f"  {'#':>2} {'time':<17} {'dir':<6} {'sub-strategy':<22} {'ADX':>5} {'R':>6}")
    for i, t in enumerate(worst["trades"], 1):
        print(f"  {i:>2} {str(t['open_time'])[:16]:<17} {t['direction']:<6} "
              f"{str(t.get('strategy_used','?')):<22} {float(t['adx_at_entry']):>5.1f} "
              f"{float(t['r_multiple']):>+6.2f}")
    print(_bar("="))

    # ---- PART C: bootstrap probability of each max-streak length --------------
    rng = np.random.default_rng(SEED)
    print()
    print(_bar("="))
    print("  PART C — PROBABILITY of hitting a max losing streak (bootstrap, real loss rate)")
    print(f"  loss rate {len(losers)/n*100:.1f}% | {N_PATHS:,} simulated paths per horizon")
    print(_bar("="))
    for months in HORIZONS:
        L = max(int(round(tpm * months)), 1)
        draws = rng.choice(is_loss, size=(N_PATHS, L), replace=True)
        run = np.zeros(N_PATHS, dtype=int)
        maxrun = np.zeros(N_PATHS, dtype=int)
        for j in range(L):
            run = (run + 1) * draws[:, j]
            maxrun = np.maximum(maxrun, run)
        print()
        print(f"  {months}-MONTH RUN ({L} trades):  median max-streak {int(np.median(maxrun))}, "
              f"95th-pctile {int(np.percentile(maxrun,95))}, worst seen {int(maxrun.max())}")
        print(f"    {'streak >= k':>12} {'P(happens)':>11} {'drawdown@10% (all -1R)':>24}")
        print(_bar("-", 54))
        for k in range(4, max_len + 2):
            p = (maxrun >= k).mean() * 100
            dd = (0.9 ** k - 1) * 100
            star = "  <- historical max" if k == max_len else ""
            print(f"    {k:>12} {p:>10.1f}% {dd:>+18.1f}%{star}")
    print(_bar("="))
    print("  Read: 'P(happens)' = chance you see AT LEAST one streak that long in the")
    print("  window. Cross-reference the drawdown column with what YOU can stomach. At 10%")
    print("  a -1R loss is -10% of balance; streaks compound. The historical max already")
    print("  happened once in real data — treat it as a when, not an if.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
