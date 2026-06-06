"""
btc_research/eth_bot/backtest/realistic_risk_sweep.py

The TP1 50/50 fix corrected S4+9 to the HONEST $6,115 -- but it also revealed
the strategy now sits at MaxCL 9 / MaxDD -21.6%, far inside the user's hard
limits (CL < 16, DD > -42%). We were throttling risk to tame a drawdown that,
under honest accounting, isn't threatening. So we have spare RISK BUDGET.

This sweep spends that budget: it runs the REALISTIC one-position sim (live
one-trade rule, on the corrected backtest_trades.csv) for the S4+9 set across
bigger base-risk profiles and looser equity throttles, and flags every config
that still keeps MaxCL < 16 AND MaxDD > -42%. The best such row = the most we
can compound toward $20k without breaking the risk constraints.

This is legitimate (sizing into proven-spare budget), NOT curve-fitting slots.
Higher risk scales finalBal AND drawdown together -- the limits are the guardrail.

== USAGE ==
  # run AFTER run_backtest.py has regenerated trades with the TP1 fix
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.realistic_risk_sweep
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
CB_THRESHOLD     = -0.10

# -- User hard limits ----------------------------------------------------------
MAX_CL_LIMIT = 16      # must stay strictly below
MAX_DD_LIMIT = -42.0   # must stay above (less negative)

# -- Slot set: S4 + 9 complement (the realistic winner) ------------------------
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

# -- Risk profiles to sweep: (label, early<=25, trans 25-40, strong>=40) -------
RISK_PROFILES = [
    ("2/3/5  (current)", 0.020, 0.030, 0.050),
    ("3/4/6",            0.030, 0.040, 0.060),
    ("3/5/8",            0.030, 0.050, 0.080),
    ("4/6/10",           0.040, 0.060, 0.100),
    ("5/8/12",           0.050, 0.080, 0.120),
]
# -- Equity-throttle variants: (label, trigger, factor) -- factor 1.0 = OFF ----
THROTTLES = [
    ("thr -15% x0.20 (current)", -0.15, 0.20),
    ("thr -15% x0.50",           -0.15, 0.50),
    ("thr -20% x0.50",           -0.20, 0.50),
    ("thr -25% x0.50",           -0.25, 0.50),
    ("throttle OFF",             -0.99, 1.00),
]


def _bar(c: str = "-", w: int = 100) -> str:
    return c * w


def _risk_pct(adx: float, e: float, tr: float, st: float) -> float:
    if adx >= 40:
        return st
    if adx <= 25:
        return e
    return tr


def _select(df: pd.DataFrame, slots) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in slots:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return df[mask].sort_values("entry_time").reset_index(drop=True)


def _simulate(cand: pd.DataFrame, e: float, tr: float, st: float,
              dd_trig: float, dd_fac: float) -> dict:
    """One-position-at-a-time sim with configurable risk + throttle."""
    balance = STARTING_BALANCE
    peak = balance
    last_exit = pd.Timestamp.min
    bals, outcomes = [], []

    cur_month = None
    month_start_bal = balance
    month_halted = False

    for _, t in cand.iterrows():
        et = t["entry_time"]
        mk = (et.year, et.month)
        if mk != cur_month:
            cur_month = mk
            month_start_bal = balance
            month_halted = False
        if et < last_exit:
            continue
        if month_halted:
            continue

        rp = _risk_pct(float(t["adx"]), e, tr, st)
        if et.month == 10:
            rp *= OCT_RISK_FACTOR
        if (balance - peak) / peak <= dd_trig:
            rp *= dd_fac

        balance += balance * rp * float(t["r_achieved"])
        peak = max(peak, balance)
        last_exit = t["exit_time"]
        bals.append(balance)
        outcomes.append(t["outcome"])

        if (balance - month_start_bal) / month_start_bal <= CB_THRESHOLD:
            month_halted = True

    if not bals:
        return {"final": STARTING_BALANCE, "cagr": 0, "maxdd": 0, "maxcl": 0, "wr": 0}
    bser = pd.Series(bals)
    max_dd = ((bser - bser.cummax()) / bser.cummax() * 100).min()
    max_cl = cur = 0
    for o in outcomes:
        cur = cur + 1 if o == "loss" else 0
        max_cl = max(max_cl, cur)
    days = (cand["entry_time"].max() - cand["entry_time"].min()).days or 1
    cagr = ((bals[-1] / STARTING_BALANCE) ** (365.25 / days) - 1) * 100
    wins = sum(1 for o in outcomes if o == "win")
    return {"final": bals[-1], "cagr": cagr, "maxdd": max_dd,
            "maxcl": max_cl, "wr": wins / len(outcomes) * 100}


def main() -> None:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first")
        sys.exit(1)
    df = pd.read_csv(CSV, parse_dates=["entry_time", "exit_time"])
    for c in ("entry_time", "exit_time"):
        if df[c].dt.tz is not None:
            df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[df["entry_time"].dt.year >= START_YEAR].reset_index(drop=True)
    cand = _select(df, S4_SLOTS + COMPLEMENT_9)

    print()
    print(_bar("="))
    print(f"  ETH BOT -- REALISTIC RISK SWEEP  (S4+9, one-position, TP1-fixed)  {START_YEAR}+")
    print(f"  Spend the spare risk budget. Guardrail: MaxCL < {MAX_CL_LIMIT} AND "
          f"MaxDD > {MAX_DD_LIMIT:.0f}%.")
    print(_bar("="))
    print(f"  {'risk profile':<18} {'throttle':<26} {'finalBal':>11} {'CAGR%':>8} "
          f"{'WR%':>6} {'MaxDD%':>8} {'MaxCL':>6}")
    print(_bar())

    best = None
    for rlabel, e, tr, st in RISK_PROFILES:
        for tlabel, dtrig, dfac in THROTTLES:
            r = _simulate(cand, e, tr, st, dtrig, dfac)
            ok = (r["maxcl"] < MAX_CL_LIMIT) and (r["maxdd"] > MAX_DD_LIMIT)
            flag = ""
            if ok:
                flag = "  OK"
                if best is None or r["final"] > best[2]["final"]:
                    best = (rlabel, tlabel, r)
            else:
                flag = "  <breach>"
            print(f"  {rlabel:<18} {tlabel:<26} {r['final']:>11,.0f} "
                  f"{r['cagr']:>+7.1f}% {r['wr']:>5.1f}% {r['maxdd']:>+7.1f}% "
                  f"{r['maxcl']:>6}{flag}")
        print(_bar("."))
    print(_bar())
    if best:
        rl, tl, r = best
        print(f"  BEST WITHIN LIMITS: {rl} | {tl}")
        print(f"     finalBal ${r['final']:,.0f} | CAGR {r['cagr']:+.1f}% | "
              f"WR {r['wr']:.1f}% | MaxDD {r['maxdd']:.1f}% | MaxCL {r['maxcl']}")
        if r["final"] >= 20000:
            print("     >>> clears the $20k target within your risk limits.")
        else:
            print(f"     >>> ${r['final']:,.0f} of the $20k target; gap needs exit")
            print("         optimization or new edge on top.")
    else:
        print("  No config stayed within limits -- current sizing is near the ceiling.")
    print(_bar("="))
    print()
    print("  NOTE: higher risk scales BOTH return and drawdown; the OK flag means")
    print("  the historical path stayed inside your limits, not that live will.")
    print("  Live tail risk (a worse-than-backtest streak) is larger at high risk.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
