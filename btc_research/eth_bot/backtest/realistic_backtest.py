"""
btc_research/eth_bot/backtest/realistic_backtest.py

REALISTIC one-position-at-a-time backtest -- matches how the LIVE bot trades.

Every other backtest here (monthly_compound, run_backtest, the sweeps) assumes
each (strategy, hour) slot holds an INDEPENDENT, CONCURRENT position. The live
ETH bot cannot do that: it holds ONE ETH position at a time (paper_trader
"one-trade rule"). While a trade is open it skips every kill-zone scan until the
trade closes. So the concurrent backtests are upper bounds we can't execute.

This script simulates the REAL constraint:
  * candidate signals from backtest_trades.csv are walked in entry_time order
  * a trade is TAKEN only if no position is currently open
    (entry_time >= the last taken trade's exit_time)
  * otherwise the signal is missed (position busy), exactly like live
  * risk: Config D (2/3/5%), Oct x0.5, monthly -10% CB, equity throttle x0.20
    when >=15% below peak

It compares slot sets (pure S4 vs S4+complement) under this rule, then prints a
month-by-month table for each. THIS is the number that reflects live behaviour.

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.realistic_backtest
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR = Path(__file__).parent
CSV  = _DIR / "data" / "backtest_trades.csv"

# -- Fixed sim parameters (must match live signal_engine) ----------------------
STARTING_BALANCE = 500.0
START_YEAR       = 2023
OCT_RISK_FACTOR  = 0.5
CB_THRESHOLD     = -0.10
RISK_EARLY, RISK_TRANS, RISK_STRONG = 0.020, 0.030, 0.050  # Config D
DD_THROTTLE_TRIGGER = -0.15
DD_THROTTLE_FACTOR  = 0.20

# -- Slot sets -----------------------------------------------------------------
S4_SLOTS = [
    ("rsi_50",    2,  True), ("macd_adx",  6,  True), ("rsi_ema", 10, True),
    ("ema_cross", 5,  False), ("rsi_50",    7,  False), ("keltner", 14, False),
    ("ema_cross",14,  False), ("keltner",  15,  False),
]
COMPLEMENT_9 = [
    ("engulfing",16,False), ("engulfing", 1,False), ("pin_bar",  2,False),
    ("engulfing", 8,False), ("engulfing", 2,False), ("engulfing",5,False),
    ("pin_bar",  19,False), ("pin_bar",  13,False), ("pin_bar", 11,False),
]


def _bar(c: str = "-", w: int = 92) -> str:
    return c * w


def _risk_pct(adx: float) -> float:
    if adx >= 40:
        return RISK_STRONG
    if adx <= 25:
        return RISK_EARLY
    return RISK_TRANS


def _select(df: pd.DataFrame, slots) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in slots:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return df[mask].sort_values("entry_time").reset_index(drop=True)


def _simulate_onepos(cand: pd.DataFrame) -> dict:
    """One-position-at-a-time compound sim. cand sorted by entry_time."""
    balance = STARTING_BALANCE
    peak = balance
    last_exit = pd.Timestamp.min
    records = []

    cur_month = None
    month_start_bal = balance
    month_halted = False
    n_taken = n_missed_busy = n_throttled = n_halted = 0

    for _, t in cand.iterrows():
        et = t["entry_time"]
        xt = t["exit_time"]
        mk = (et.year, et.month)
        if mk != cur_month:
            cur_month = mk
            month_start_bal = balance
            month_halted = False

        # position busy? -> miss it (this is the live one-trade rule)
        if et < last_exit:
            n_missed_busy += 1
            continue
        if month_halted:
            n_halted += 1
            continue

        rp = _risk_pct(float(t["adx"]))
        if et.month == 10:
            rp *= OCT_RISK_FACTOR
        if (balance - peak) / peak <= DD_THROTTLE_TRIGGER:
            rp *= DD_THROTTLE_FACTOR
            n_throttled += 1

        r_val = float(t["r_achieved"])
        pnl = balance * rp * r_val
        balance += pnl
        peak = max(peak, balance)
        last_exit = xt
        n_taken += 1

        records.append({
            "entry_time": et, "strategy": t["strategy"],
            "hour_utc": int(t["hour_utc"]), "outcome": t["outcome"],
            "r": r_val, "pnl": pnl, "balance": balance,
        })

        if (balance - month_start_bal) / month_start_bal <= CB_THRESHOLD:
            month_halted = True

    sim = pd.DataFrame(records)
    bser = sim["balance"]
    peakser = bser.cummax()
    max_dd = ((bser - peakser) / peakser * 100).min()

    max_cl = cur = 0
    for o in sim["outcome"]:
        cur = cur + 1 if o == "loss" else 0
        max_cl = max(max_cl, cur)

    days = (sim["entry_time"].max() - sim["entry_time"].min()).days or 1
    n_years = days / 365.25
    final_bal = bser.iloc[-1]
    cagr = ((final_bal / STARTING_BALANCE) ** (1 / n_years) - 1) * 100
    wins = (sim["outcome"] == "win").sum()

    return {
        "sim": sim, "n_taken": n_taken, "n_missed": n_missed_busy,
        "n_throttled": n_throttled, "n_halted": n_halted,
        "wr": wins / n_taken * 100 if n_taken else 0.0,
        "tot_r": sim["r"].sum(), "final": final_bal, "cagr": cagr,
        "maxdd": max_dd, "maxcl": max_cl, "n_years": n_years,
    }


def _monthly(sim: pd.DataFrame) -> None:
    sim = sim.copy()
    sim["month"] = sim["entry_time"].dt.to_period("M")
    print(f"  {'Month':<9} {'N':>3} {'W':>3} {'WR%':>6} {'TotR':>7} "
          f"{'P&L $':>10} {'EndBal':>11}")
    print(_bar())
    prev_year = None
    for m, g in sim.groupby("month", sort=True):
        if prev_year and m.year != prev_year:
            print(_bar("."))
        n = len(g); w = (g["outcome"] == "win").sum()
        print(f"  {str(m):<9} {n:>3} {w:>3} {w/n*100:>5.1f}% "
              f"{g['r'].sum():>+7.2f} {g['pnl'].sum():>+10.2f} "
              f"{g['balance'].iloc[-1]:>11,.2f}")
        prev_year = m.year


def main() -> None:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first")
        sys.exit(1)
    df = pd.read_csv(CSV, parse_dates=["entry_time", "exit_time"])
    # Normalise to tz-naive so chronological comparisons (et < last_exit) work
    # regardless of whether the CSV stored tz-aware or tz-naive timestamps.
    for _col in ("entry_time", "exit_time"):
        if df[_col].dt.tz is not None:
            df[_col] = df[_col].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[df["entry_time"].dt.year >= START_YEAR].reset_index(drop=True)

    sets = [
        ("PURE S4",           S4_SLOTS),
        ("S4 + 9 COMPLEMENT", S4_SLOTS + COMPLEMENT_9),
    ]
    results = [(name, _simulate_onepos(_select(df, slots))) for name, slots in sets]

    print()
    print(_bar("="))
    print(f"  ETH BOT -- REALISTIC ONE-POSITION BACKTEST  ({START_YEAR}+)")
    print(f"  ONE ETH trade at a time (live one-trade rule). Start ${STARTING_BALANCE:.0f}.")
    print(f"  Config D 2/3/5% | Oct x0.5 | monthly CB -10% | throttle x0.2 @ -15% peak")
    print(_bar("="))
    print(f"  {'slot set':<20} {'taken':>6} {'missed':>7} {'WR%':>6} {'TotR':>7} "
          f"{'finalBal':>11} {'CAGR%':>8} {'MaxDD%':>8} {'MaxCL':>6}")
    print(_bar())
    for name, r in results:
        print(f"  {name:<20} {r['n_taken']:>6} {r['n_missed']:>7} {r['wr']:>5.1f}% "
              f"{r['tot_r']:>+7.1f} {r['final']:>11,.0f} {r['cagr']:>+7.1f}% "
              f"{r['maxdd']:>+7.1f}% {r['maxcl']:>6}")
    print(_bar())
    print("  'missed' = signals skipped because a position was already open"
          " (the live reality).")

    for name, r in results:
        print()
        print(_bar("="))
        print(f"  MONTHLY -- {name}   "
              f"(final ${r['final']:,.0f}, CAGR {r['cagr']:+.1f}%, "
              f"MaxDD {r['maxdd']:.1f}%, CL {r['maxcl']}, "
              f"throttled {r['n_throttled']}, CBskip {r['n_halted']})")
        print(_bar())
        _monthly(r["sim"])
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
