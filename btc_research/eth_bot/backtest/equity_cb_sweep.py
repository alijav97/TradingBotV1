"""
btc_research/eth_bot/backtest/equity_cb_sweep.py

Can we unlock S4+9-COMPLEMENT's edge AT Config D (2/3/5%) while taming the
-42% drawdown?  risk_sweep.py proved:
  * lowering risk shrinks DD but kills return (and can't fix the loss streak),
  * the monthly -10% CB can't catch a multi-month regime bleed (mid-2023):
    the complement skipped 160 trades to it and STILL hit -42%.

This script keeps Config D sizing and adds an EQUITY-CURVE THROTTLE: as the
account falls below its running peak, risk is scaled DOWN (so we starve the
regime bleed) and restored as equity recovers toward a new high. Two modes:

  STEP : discrete ladder -> in drawdown >= T, multiply risk by F (until new peak)
  HALT : when drawdown from peak >= T, halt new entries for the rest of the month
         (peak-based version of the monthly CB; auto-resumes next month)

Both run ON TOP of the existing monthly -10% CB. Baseline row = no equity breaker.

Reference points to beat (from risk_sweep.py):
  Pure S4   @ Config D : $14,165 | -30.0% | MaxCL 7
  S4+9 comp @ Config D : $943,368 | -42.2% | MaxCL 16   (no equity breaker)

Goal: keep finalBal well above $14k while MaxDD < ~30%. (MaxCL is mostly
risk-/halt-invariant -- expect it to stay ~15.)

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.equity_cb_sweep
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR = Path(__file__).parent
CSV  = _DIR / "data" / "backtest_trades.csv"

# -- Fixed sim parameters (Config D) -------------------------------------------
STARTING_BALANCE = 500.0
START_YEAR       = 2023
OCT_RISK_FACTOR  = 0.5
CB_THRESHOLD     = -0.10   # existing monthly circuit breaker (month-start based)
RISK_EARLY, RISK_TRANS, RISK_STRONG = 0.020, 0.030, 0.050  # 2/3/5

# -- Slot set: S4 + 9 complement ----------------------------------------------
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


def _bar(c: str = "-", w: int = 96) -> str:
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


def _simulate(ok: pd.DataFrame, mode: str, trig: float, factor: float) -> dict:
    """
    mode: "none" | "step" | "halt"
      trig   = drawdown-from-peak threshold (e.g. 0.15 = 15% below peak)
      factor = risk multiplier when throttled (step mode only)
    """
    balance = STARTING_BALANCE
    peak = balance
    bals, outcomes = [], []

    cur_month = None
    month_start_bal = balance
    month_halted = False        # monthly -10% CB
    peak_halt_month = False      # equity-curve HALT (rest of month)
    n_skipped = 0

    for _, t in ok.iterrows():
        et = t["entry_time"]
        mk = (et.year, et.month)
        if mk != cur_month:
            cur_month = mk
            month_start_bal = balance
            month_halted = False
            peak_halt_month = False

        if month_halted or peak_halt_month:
            n_skipped += 1
            continue

        dd_from_peak = (balance - peak) / peak  # <= 0

        rp = _risk_pct(float(t["adx"]))
        if et.month == 10:
            rp *= OCT_RISK_FACTOR

        # equity-curve STEP throttle
        if mode == "step" and dd_from_peak <= -trig:
            rp *= factor

        pnl = balance * rp * float(t["r_achieved"])
        balance += pnl
        peak = max(peak, balance)
        bals.append(balance)
        outcomes.append(t["outcome"])

        # monthly -10% CB (month-start based)
        if (balance - month_start_bal) / month_start_bal <= CB_THRESHOLD:
            month_halted = True

        # equity-curve HALT (peak based, rest of month)
        if mode == "halt" and (balance - peak) / peak <= -trig:
            peak_halt_month = True

    bser = pd.Series(bals)
    pk = bser.cummax()
    max_dd = ((bser - pk) / pk * 100).min()

    max_cl = cur = 0
    for o in outcomes:
        cur = cur + 1 if o == "loss" else 0
        max_cl = max(max_cl, cur)

    days = (ok["entry_time"].max() - ok["entry_time"].min()).days or 1
    n_years = days / 365.25
    final_bal = bals[-1] if bals else STARTING_BALANCE
    cagr = ((final_bal / STARTING_BALANCE) ** (1 / n_years) - 1) * 100

    return {
        "final": final_bal, "cagr": cagr, "maxdd": max_dd,
        "maxcl": max_cl, "skipped": n_skipped, "n": len(outcomes),
    }


def main() -> None:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first")
        sys.exit(1)
    df = pd.read_csv(CSV, parse_dates=["entry_time"])
    df = df[df["entry_time"].dt.year >= START_YEAR].reset_index(drop=True)
    ok = _select(df, S4_SLOTS + COMPLEMENT_9)

    print()
    print(_bar("="))
    print(f"  ETH BOT -- EQUITY-CURVE CIRCUIT BREAKER SWEEP  (S4+9 @ Config D, {START_YEAR}+)")
    print(f"  Keep 2/3/5% sizing; throttle as equity falls below its peak.")
    print(f"  Beat: pure-S4 ${14165:,} / -30.0% / CL7   |   un-broken S4+9 $943,368 / -42.2% / CL16")
    print(_bar("="))

    # configs to test
    configs = [("none", 0.0, 1.0, "BASELINE (monthly CB only)")]
    for trig in (0.12, 0.15, 0.20, 0.25):
        for fac in (0.50, 0.25):
            configs.append(("step", trig, fac,
                            f"STEP  dd>={int(trig*100)}% -> risk x{fac:g}"))
    for trig in (0.12, 0.15, 0.20):
        configs.append(("halt", trig, 1.0,
                        f"HALT  dd>={int(trig*100)}% -> stop month"))

    print(f"  {'config':<34} {'finalBal':>12} {'CAGR%':>8} "
          f"{'MaxDD%':>8} {'MaxCL':>6} {'skip':>5}")
    print(_bar())
    for mode, trig, fac, label in configs:
        r = _simulate(ok, mode, trig, fac)
        flag = ""
        if r["maxdd"] > -30.0 and r["final"] > 14165:
            flag = "  <== beats pure-S4 on BOTH"
        print(f"  {label:<34} {r['final']:>12,.0f} {r['cagr']:>+7.1f}% "
              f"{r['maxdd']:>+7.1f}% {r['maxcl']:>6} {r['skipped']:>5}{flag}")
    print(_bar())
    print()
    print(_bar("="))
    print("  READ: want a row with MaxDD% better than -30 AND finalBal above $14,165")
    print("  (pure-S4 @ Config D). That row = complement's edge unlocked safely.")
    print("  MaxCL ~15 expected; decide if that's tolerable vs pure-S4's 7.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
