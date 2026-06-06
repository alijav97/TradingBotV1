"""
btc_research/eth_bot/backtest/risk_sweep.py

DIAGNOSTIC: how much does LOWER base risk tame the drawdown?

We learned slot-trimming makes drawdown WORSE (less diversification, MR is
positively correlated with S4). The drawdown is a regime problem (e.g. the
mid-2023 multi-month bleed), not a slot-quality problem. Before adding an
equity-curve circuit breaker, this script checks the simplest lever: just turn
risk down.

It runs the SAME compound sim as monthly_compound.py for two slot sets:
    * PURE S4            (8 trend/breakout slots)
    * S4 + 9 COMPLEMENT  (full engulfing/pin_bar MR add)
across several ADX-split risk profiles, and prints one comparison matrix.

NOTE: Total R and Max-consecutive-losses are ~invariant to risk size (they are
outcome-sequence properties). Risk scaling moves the DRAWDOWN % and the final
balance / CAGR. The circuit breaker can shift the sequence slightly because it
fires at different points when swings are smaller.

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.risk_sweep
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR = Path(__file__).parent
CSV  = _DIR / "data" / "backtest_trades.csv"

# -- Fixed sim parameters ------------------------------------------------------
STARTING_BALANCE = 500.0
START_YEAR       = 2023
OCT_RISK_FACTOR  = 0.5
CB_THRESHOLD     = -0.10   # monthly circuit breaker

# -- Slot sets -----------------------------------------------------------------
# (strategy, hour, require_btc_aligned)
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

COMPLEMENT_9 = [
    ("engulfing", 16, False),
    ("engulfing",  1, False),
    ("pin_bar",    2, False),
    ("engulfing",  8, False),
    ("engulfing",  2, False),
    ("engulfing",  5, False),
    ("pin_bar",   19, False),
    ("pin_bar",   13, False),
    ("pin_bar",   11, False),
]

# -- Risk profiles: (label, early<=25, transition 25-40, strong>=40) -----------
RISK_PROFILES = [
    ("2/3/5  (Config D)", 0.020, 0.030, 0.050),
    ("1.5/2.5/4",         0.015, 0.025, 0.040),
    ("1.5/2/3",           0.015, 0.020, 0.030),
    ("1/1.5/2",           0.010, 0.015, 0.020),
    ("1/1/1  (flat)",     0.010, 0.010, 0.010),
]


def _bar(c: str = "-", w: int = 92) -> str:
    return c * w


def _load() -> pd.DataFrame:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first")
        sys.exit(1)
    df = pd.read_csv(CSV, parse_dates=["entry_time"])
    df = df[df["entry_time"].dt.year >= START_YEAR].reset_index(drop=True)
    return df


def _select(df: pd.DataFrame, slots) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in slots:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return df[mask].sort_values("entry_time").reset_index(drop=True)


def _risk_pct(adx: float, early: float, trans: float, strong: float) -> float:
    if adx >= 40:
        return strong
    if adx <= 25:
        return early
    return trans


def _simulate(ok: pd.DataFrame, early: float, trans: float, strong: float) -> dict:
    balance = STARTING_BALANCE
    bals, outcomes = [], []

    cur_month = None
    month_start_bal = balance
    month_halted = False
    n_halted = 0

    for _, t in ok.iterrows():
        et = t["entry_time"]
        mk = (et.year, et.month)
        if mk != cur_month:
            cur_month = mk
            month_start_bal = balance
            month_halted = False
        if month_halted:
            n_halted += 1
            continue

        adx = float(t["adx"])
        r_val = float(t["r_achieved"])
        rp = _risk_pct(adx, early, trans, strong)
        if et.month == 10:
            rp *= OCT_RISK_FACTOR

        pnl = balance * rp * r_val
        balance += pnl
        bals.append(balance)
        outcomes.append(t["outcome"])

        mo_pct = (balance - month_start_bal) / month_start_bal
        if mo_pct <= CB_THRESHOLD:
            month_halted = True

    bser = pd.Series(bals)
    peak = bser.cummax()
    max_dd = ((bser - peak) / peak * 100).min()

    max_cl = cur = 0
    for o in outcomes:
        cur = cur + 1 if o == "loss" else 0
        max_cl = max(max_cl, cur)

    n = len(outcomes)
    wins = sum(1 for o in outcomes if o == "win")

    final_bal = bals[-1] if bals else STARTING_BALANCE
    days = (ok["entry_time"].max() - ok["entry_time"].min()).days or 1
    n_years = days / 365.25
    cagr = ((final_bal / STARTING_BALANCE) ** (1 / n_years) - 1) * 100

    return {
        "n": n,
        "wr": wins / n * 100 if n else 0.0,
        "final": final_bal,
        "cagr": cagr,
        "maxdd": max_dd,
        "maxcl": max_cl,
        "skipped": n_halted,
    }


def main() -> None:
    df = _load()

    sets = [
        ("PURE S4",          _select(df, S4_SLOTS)),
        ("S4 + 9 COMPLEMENT", _select(df, S4_SLOTS + COMPLEMENT_9)),
    ]

    print()
    print(_bar("="))
    print(f"  ETH BOT -- RISK SWEEP  (does lower risk tame the drawdown?)  {START_YEAR}+")
    print(f"  Start ${STARTING_BALANCE:.0f} | Oct x{OCT_RISK_FACTOR} | monthly CB {CB_THRESHOLD*100:.0f}%")
    print(f"  NOTE: MaxCL (loss streak) is ~risk-invariant; risk moves MaxDD% + balance.")
    print(_bar("="))

    for set_name, ok in sets:
        # total R is fixed for the set (risk-invariant), report once
        tot_r = ok["r_achieved"].sum()
        base_wr = (ok["outcome"] == "win").mean() * 100
        print()
        print(_bar("="))
        print(f"  {set_name}   (N={len(ok)} pooled, WR={base_wr:.1f}%, TotR={tot_r:+.1f})")
        print(_bar())
        print(f"  {'risk profile':<20} {'finalBal':>12} {'CAGR%':>8} "
              f"{'MaxDD%':>8} {'MaxCL':>6} {'CBskip':>7}")
        print(_bar())
        for label, e, tr, st in RISK_PROFILES:
            r = _simulate(ok, e, tr, st)
            print(f"  {label:<20} {r['final']:>12,.0f} {r['cagr']:>+7.1f}% "
                  f"{r['maxdd']:>+7.1f}% {r['maxcl']:>6} {r['skipped']:>7}")
        print(_bar())

    print()
    print(_bar("="))
    print("  READ: find the risk profile where MaxDD% drops under your tolerance.")
    print("  If MaxCL stays high everywhere -> risk scaling can't fix the streak;")
    print("  that needs an equity-curve (regime) circuit breaker instead.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
