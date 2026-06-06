"""
btc_research/eth_bot/backtest/backtest_optimised.py

Validate the proposed optimisations AGAINST the real historical trade data
BEFORE changing the live strategy. Each fix is applied incrementally so we
can see its individual contribution to CAGR / MaxDD / consecutive losses.

FIXES UNDER TEST (from deep_dive.py findings):
  Fix 1  drop macd_adx[H15]   -- net -$188 overall, -$13,812 in 2026 (fallback noise)
  Fix 2  October risk -50%    -- October avg return = -3.7% (only negative month)
  Fix 3  monthly circuit-breaker  -- halt the month after it draws down -10%
                                     (kills the 8-loss -$45k cluster of 2026)
  Fix 4  add rsi_50[H07]       -- was skipped for MaxDD; re-test WITH circuit breaker
                                  (circuit breaker may bring its DD back under -35%)

NOTE: the ema_cross ATR-contraction filter (the 5th idea) needs new indicator
columns -> it requires a full run_backtest.py rerun and is handled separately.

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.backtest_optimised
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# -- Paths ---------------------------------------------------------------------
_BACKTEST_DIR = Path(__file__).parent
DATA_DIR      = _BACKTEST_DIR / "data"
TRADES_CSV    = DATA_DIR / "backtest_trades.csv"

# -- Parameters ----------------------------------------------------------------
STARTING_BALANCE = 500.0
ADX_SPLIT_EARLY_MAX  = 25
ADX_SPLIT_STRONG_MIN = 40
RISK_EARLY      = 0.02
RISK_TRANSITION = 0.03
RISK_STRONG     = 0.05

OCT_RISK_FACTOR = 0.5      # Fix 2: halve risk in October
CB_THRESHOLD    = -0.10    # Fix 3: halt month after -10% month drawdown

# Base 8-slot set
BASE_SLOTS = [
    ("rsi_50",   2,  True),
    ("macd_adx", 6,  True),
    ("rsi_ema",  10, True),
    ("ema_cross",  5,  False),
    ("keltner",   14,  False),
    ("ema_cross", 14,  False),
    ("keltner",   15,  False),
    ("macd_adx",  15,  False),   # <- Fix 1 removes this
]
RSI50_H7 = ("rsi_50", 7, False)  # <- Fix 4 adds this


def _bar(char: str = "-", width: int = 88) -> str:
    return char * width


def _risk_pct(adx: float) -> float:
    if adx >= ADX_SPLIT_STRONG_MIN:
        return RISK_STRONG
    elif adx <= ADX_SPLIT_EARLY_MAX:
        return RISK_EARLY
    else:
        return RISK_TRANSITION


def _filter_trades(df: pd.DataFrame, slots: list) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in slots:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return df[mask].sort_values("entry_time").reset_index(drop=True)


def _simulate(trades: pd.DataFrame,
              oct_cut: bool = False,
              circuit_breaker: bool = False) -> dict:
    """Run compound simulation with optional October cut + circuit breaker.

    Returns dict with full stats + the per-trade balance series.
    """
    balance  = STARTING_BALANCE
    balances = []
    sim_rows = []

    cur_month       = None
    month_start_bal = balance
    month_halted    = False
    n_halted        = 0   # trades skipped by circuit breaker

    for _, t in trades.iterrows():
        et        = t["entry_time"]
        month_key = (et.year, et.month)

        # New month -> reset circuit breaker state
        if month_key != cur_month:
            cur_month       = month_key
            month_start_bal = balance
            month_halted    = False

        # Circuit breaker: skip remaining trades this month
        if circuit_breaker and month_halted:
            n_halted += 1
            continue

        adx = float(t["adx"])
        rv  = float(t["r_achieved"])
        rp  = _risk_pct(adx)

        # October risk reduction
        if oct_cut and et.month == 10:
            rp *= OCT_RISK_FACTOR

        pnl      = balance * rp * rv
        balance += pnl
        balances.append(balance)

        sim_rows.append({
            "entry_time": et, "year": et.year, "month": et.month,
            "strategy": t["strategy"], "hour_utc": int(t["hour_utc"]),
            "adx": round(adx, 1), "outcome": t["outcome"],
            "r_achieved": round(rv, 3), "risk_pct": rp * 100,
            "pnl_usd": round(pnl, 2), "balance": round(balance, 2),
        })

        # Update circuit breaker after the trade
        if circuit_breaker:
            mo_pct = (balance - month_start_bal) / month_start_bal
            if mo_pct <= CB_THRESHOLD:
                month_halted = True

    if not sim_rows:
        return {}

    sim = pd.DataFrame(sim_rows)
    bser = pd.Series(balances)
    peak = bser.cummax()
    max_dd = ((bser - peak) / peak * 100).min()

    # consecutive losses
    max_cl = cl = 0
    for o in sim["outcome"]:
        cl = cl + 1 if o == "loss" else 0
        max_cl = max(max_cl, cl)

    n_years = (sim["entry_time"].max() - sim["entry_time"].min()).days / 365.25
    final   = balance
    cagr    = ((final / STARTING_BALANCE) ** (1 / n_years) - 1) * 100
    wins    = (sim["outcome"] == "win").sum()

    return {
        "sim":        sim,
        "n":          len(sim),
        "n_halted":   n_halted,
        "wr":         wins / len(sim) * 100,
        "final":      final,
        "cagr":       cagr,
        "proj5":      STARTING_BALANCE * ((1 + cagr / 100) ** 5),
        "max_dd":     max_dd,
        "max_cl":     max_cl,
        "n_years":    n_years,
    }


# ==============================================================================
def main() -> None:
    if not TRADES_CSV.exists():
        print(f"ERROR: {TRADES_CSV} not found -- run run_backtest.py first")
        sys.exit(1)

    df = pd.read_csv(TRADES_CSV, parse_dates=["entry_time"])

    slots_no_m15 = [s for s in BASE_SLOTS if not (s[0] == "macd_adx" and s[1] == 15)]
    slots_plus_h7 = slots_no_m15 + [RSI50_H7]

    # Scenarios (incremental)
    scenarios = [
        ("S0 baseline (8-slot)",      BASE_SLOTS,    False, False),
        ("S1 + drop macd_adx[15]",    slots_no_m15,  False, False),
        ("S2 + October risk -50%",    slots_no_m15,  True,  False),
        ("S3 + circuit breaker -10%", slots_no_m15,  True,  True),
        ("S4 + add rsi_50[H07]",      slots_plus_h7, True,  True),
    ]

    print()
    print(_bar("="))
    print("  ETH BOT -- OPTIMISATION VALIDATION (incremental fixes vs baseline)")
    print(f"  $500 start | Config D risk | TP1=2R/TP2=4R | tested on real trade data")
    print(_bar("="))
    print()
    print(f"  {'Scenario':<28}  {'N':>4}  {'WR%':>6}  {'CAGR':>8}  "
          f"{'5yr proj':>11}  {'MaxDD':>7}  {'MaxCL':>5}  {'Halt':>5}")
    print(_bar())

    results = {}
    for label, slots, oct_cut, cb in scenarios:
        trades = _filter_trades(df, slots)
        res    = _simulate(trades, oct_cut=oct_cut, circuit_breaker=cb)
        results[label] = res
        if not res:
            print(f"  {label:<28}  (no trades)")
            continue
        print(f"  {label:<28}  {res['n']:>4}  {res['wr']:>5.1f}%  "
              f"{res['cagr']:>+7.1f}%  ${res['proj5']:>10,.0f}  "
              f"{res['max_dd']:>+6.1f}%  {res['max_cl']:>5}  {res['n_halted']:>5}")

    print(_bar())
    print("  N=trades taken | MaxCL=max consecutive losses | Halt=trades skipped by breaker")

    # -- Pick the best scenario (highest 5yr proj with MaxDD >= -35%) ----------
    valid = {k: v for k, v in results.items()
             if v and v["max_dd"] >= -35.0}
    if valid:
        best_label = max(valid, key=lambda k: valid[k]["proj5"])
    else:
        # fall back to least-bad MaxDD
        best_label = max(results, key=lambda k: results[k]["proj5"] if results[k] else 0)

    best = results[best_label]
    print()
    print(_bar("="))
    print(f"  RECOMMENDED SCENARIO: {best_label}")
    print(_bar())
    print(f"  Trades             : {best['n']}  ({best['n']/(best['n_years']*12):.1f}/month)")
    print(f"  Win rate           : {best['wr']:.1f}%")
    print(f"  CAGR               : {best['cagr']:+.1f}%")
    print(f"  Final balance      : ${best['final']:,.2f}")
    print(f"  5yr projection     : ${best['proj5']:,.0f}  (from $500)")
    print(f"  Max drawdown       : {best['max_dd']:+.1f}%")
    print(f"  Max consec losses  : {best['max_cl']}")
    print(f"  Trades skipped (CB): {best['n_halted']}")

    base = results["S0 baseline (8-slot)"]
    print()
    print(f"  IMPROVEMENT vs baseline:")
    print(f"    CAGR    : {base['cagr']:+.1f}% -> {best['cagr']:+.1f}%  "
          f"({best['cagr']-base['cagr']:+.1f} pts)")
    print(f"    5yr     : ${base['proj5']:,.0f} -> ${best['proj5']:,.0f}  "
          f"({(best['proj5']/base['proj5']-1)*100:+.0f}%)")
    print(f"    MaxDD   : {base['max_dd']:+.1f}% -> {best['max_dd']:+.1f}%  "
          f"({best['max_dd']-base['max_dd']:+.1f} pts)")
    print(f"    MaxCL   : {base['max_cl']} -> {best['max_cl']}")

    # -- Monthly breakdown of the best scenario -------------------------------
    print()
    print(_bar("="))
    print(f"  MONTHLY BREAKDOWN -- {best_label}")
    print(_bar())
    print(f"  {'Month':<9}  {'N':>3}  {'WR%':>6}  {'P&L $':>9}  "
          f"{'EndBal':>11}  {'MoRtn':>7}")
    print(_bar())

    sim = best["sim"]
    sim["month_label"] = sim["entry_time"].dt.to_period("M").astype(str)
    prev_year = None
    for ml, grp in sim.groupby("month_label", sort=True):
        yr = int(ml[:4])
        if prev_year and yr != prev_year:
            print(_bar("."))
        n   = len(grp)
        wr  = (grp["outcome"] == "win").sum() / n * 100
        pnl = grp["pnl_usd"].sum()
        eb  = grp["balance"].iloc[-1]
        sb  = eb - pnl
        rtn = pnl / sb * 100 if sb > 0 else 0
        print(f"  {ml:<9}  {n:>3}  {wr:>5.1f}%  {pnl:>+9.2f}  "
              f"{eb:>11,.2f}  {rtn:>+6.1f}%")
        prev_year = yr

    # -- Save the best scenario trade log -------------------------------------
    out = DATA_DIR / "optimised_trades.csv"
    sim.drop(columns=["month_label"], errors="ignore").to_csv(out, index=False)
    print(_bar("="))
    print(f"  Saved best-scenario trade log -> {out.name}")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
