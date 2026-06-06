"""
btc_research/eth_bot/backtest/regime_complement.py

Find an ANTI-CORRELATED complement to the S4 breakout portfolio.

Idea: our 8 S4 slots are all trend/breakout. In choppy months they ALL fail
together (clean SL hits). A failed breakout is itself a signal that the market
is mean-reverting -- so a mean-reversion / reversal strategy should WIN exactly
when S4 loses. run_backtest.py already logged 25 strategies to
backtest_trades.csv, so we can test this directly without a rerun.

This script:
  1. Builds the S4 monthly R series and flags S4's LOSING months.
  2. For every NON-S4 (strategy, all-hours-pooled) candidate, measures:
       - standalone quality (N, WR, AvgR, TotR)
       - correlation of its monthly R with S4's monthly R   (want NEGATIVE)
       - its performance IN S4's losing months              (want POSITIVE)
  3. Ranks candidates so we can pick the best regime complement.

The winner is a strategy that is (a) at least break-even standalone,
(b) anti-correlated with S4, and (c) profitable in S4's bad months.

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.regime_complement
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR = Path(__file__).parent
DATA = _DIR / "data"
CSV  = DATA / "backtest_trades.csv"

# -- Parameters ----------------------------------------------------------------
START_YEAR = 2023

# S4 slot set (strategy, hour, require_btc_aligned) -- excluded from candidates
S4_SLOTS = [
    ("rsi_50",    2,  True),
    ("macd_adx",  6,  True),
    ("rsi_ema",  10,  True),
    ("ema_cross", 5,  False),
    ("rsi_50",    7,  False),
    ("keltner",  14,  False),
    ("ema_cross",14,  False),
    ("keltner",  15,  False),
]

# Strategy families that are mean-reversion / reversal (natural complements)
MR_TYPES = {"rsi_rev", "stoch", "pin_bar", "engulfing"}

MIN_TRADES = 10   # ignore candidates with too few trades to judge


def _bar(c: str = "-", w: int = 100) -> str:
    return c * w


def _s4_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in S4_SLOTS:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return mask


def main() -> None:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first")
        sys.exit(1)

    df = pd.read_csv(CSV, parse_dates=["entry_time"])
    df = df[df["entry_time"].dt.year >= START_YEAR].copy()
    df["month"] = df["entry_time"].dt.to_period("M").astype(str)
    all_months = sorted(df["month"].unique())

    # ---- S4 monthly R + bad months ------------------------------------------
    s4 = df[_s4_mask(df)]
    s4_monthly = (s4.groupby("month")["r_achieved"].sum()
                    .reindex(all_months, fill_value=0.0))
    bad_months = set(s4_monthly[s4_monthly < 0].index)

    print()
    print(_bar("="))
    print(f"  ETH BOT -- REGIME COMPLEMENT SEARCH  (anti-correlated leg, {START_YEAR}+)")
    print(f"  S4 slots: {s4['strategy'].count()} trades  "
          f"|  WR={ (s4['outcome']=='win').mean()*100:.1f}%  "
          f"|  TotR={s4['r_achieved'].sum():+.1f}")
    print(_bar("="))

    # ---- S4 monthly table ----------------------------------------------------
    print()
    print("  S4 MONTHLY R  (BAD = losing month we want a complement for)")
    print(_bar())
    print(f"  {'Month':<9}  {'S4 TotR':>8}   flag")
    print(_bar())
    for m in all_months:
        flag = "  <-- BAD" if m in bad_months else ""
        print(f"  {m:<9}  {s4_monthly[m]:>+8.2f}{flag}")
    print(_bar())
    print(f"  Bad months: {len(bad_months)} of {len(all_months)}  "
          f"(S4 R lost in these months: {s4_monthly[s4_monthly < 0].sum():+.1f})")

    # ---- Candidate scan ------------------------------------------------------
    print()
    print(_bar("="))
    print("  CANDIDATE COMPLEMENTS  (every non-S4 strategy, all hours pooled)")
    print("  want: corrS4 NEGATIVE, badAvgR POSITIVE, badTotR high")
    print(_bar())
    print(f"  {'strategy':<12} {'MR':>3} {'N':>4} {'WR%':>6} {'AvgR':>6} {'TotR':>7}  "
          f"{'corrS4':>7}  {'badN':>4} {'badWR%':>6} {'badAvgR':>7} {'badTotR':>8}")
    print(_bar())

    s4_strat_hours = {(s, h) for s, h, _ in S4_SLOTS}
    rows = []
    for strat, g_all in df.groupby("strategy"):
        # exclude the exact S4 (strategy,hour) slots from the candidate's trades
        g = g_all[~g_all.apply(lambda r: (r["strategy"], r["hour_utc"]) in s4_strat_hours, axis=1)]
        if len(g) < MIN_TRADES:
            continue

        monthly = (g.groupby("month")["r_achieved"].sum()
                     .reindex(all_months, fill_value=0.0))
        # correlation needs variance on both sides
        corr = monthly.corr(s4_monthly) if monthly.std() > 0 else float("nan")

        bm = g[g["month"].isin(bad_months)]
        bm_n   = len(bm)
        bm_wr  = (bm["outcome"] == "win").mean() * 100 if bm_n else 0.0
        bm_avg = bm["r_achieved"].mean() if bm_n else 0.0
        bm_tot = bm["r_achieved"].sum() if bm_n else 0.0

        rows.append({
            "strat":  strat,
            "mr":     "Y" if strat in MR_TYPES else "",
            "n":      len(g),
            "wr":     (g["outcome"] == "win").mean() * 100,
            "avgr":   g["r_achieved"].mean(),
            "totr":   g["r_achieved"].sum(),
            "corr":   corr,
            "bm_n":   bm_n,
            "bm_wr":  bm_wr,
            "bm_avg": bm_avg,
            "bm_tot": bm_tot,
        })

    # rank by performance in S4's bad months (the whole point)
    rows.sort(key=lambda r: r["bm_tot"], reverse=True)
    for r in rows:
        corr_str = f"{r['corr']:>+7.2f}" if pd.notna(r["corr"]) else "   n/a"
        print(f"  {r['strat']:<12} {r['mr']:>3} {r['n']:>4} {r['wr']:>5.1f}% "
              f"{r['avgr']:>+6.2f} {r['totr']:>+7.1f}  {corr_str}  "
              f"{r['bm_n']:>4} {r['bm_wr']:>5.1f}% {r['bm_avg']:>+7.2f} {r['bm_tot']:>+8.1f}")

    print(_bar())

    # ---- Best-hour drill-down for top MR candidates --------------------------
    print()
    print(_bar("="))
    print("  BEST HOURS for top mean-reversion candidates  (badTotR by hour)")
    print(_bar())
    mr_rows = [r for r in rows if r["mr"] == "Y"]
    mr_rows.sort(key=lambda r: r["bm_tot"], reverse=True)
    for r in mr_rows[:3]:
        strat = r["strat"]
        g = df[(df["strategy"] == strat)]
        print(f"\n  {strat}  (overall AvgR {r['avgr']:+.2f}, corrS4 "
              f"{r['corr']:+.2f}, badTotR {r['bm_tot']:+.1f})")
        print(f"    {'hr':>3} {'N':>4} {'WR%':>6} {'AvgR':>6} {'TotR':>7}  "
              f"{'badN':>4} {'badAvgR':>7} {'badTotR':>8}")
        for hour, gh in g.groupby("hour_utc"):
            if len(gh) < 5:
                continue
            bm = gh[gh["month"].isin(bad_months)]
            bt = bm["r_achieved"].sum() if len(bm) else 0.0
            ba = bm["r_achieved"].mean() if len(bm) else 0.0
            print(f"    {int(hour):>3} {len(gh):>4} "
                  f"{(gh['outcome']=='win').mean()*100:>5.1f}% "
                  f"{gh['r_achieved'].mean():>+6.2f} {gh['r_achieved'].sum():>+7.1f}  "
                  f"{len(bm):>4} {ba:>+7.2f} {bt:>+8.1f}")

    print()
    print(_bar("="))
    print("  Read: a good complement has corrS4 < 0 (anti-correlated), badAvgR > 0,")
    print("  and decent standalone AvgR. Next: add it to S4 and re-run monthly_compound.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
