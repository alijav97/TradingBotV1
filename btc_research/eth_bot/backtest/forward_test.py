"""
btc_research/eth_bot/backtest/forward_test.py — OUT-OF-SAMPLE 2026 forward run.

The S4 / Tier-B / slot config was tuned on PRE-2026 history. This script replays
ONLY the chosen forward window (default: 2026 year-to-date) as a fresh $500
account, exactly the way the LIVE bot would trade it:

  * PURE S4 slots only (the live hour-routing set — NO complement engulfing/pin_bar,
    because the live bot does not trade those). Set INCLUDE_COMPLEMENT=True to also
    print the S4+9 reference line.
  * ONE position at a time (live one-trade rule): a signal is taken only if no
    position is open (entry_time >= last taken trade's exit_time), else missed.
  * BTC same-bar alignment on H02/H06/H10 (S4 btc_req=True) — already encoded in the
    CSV's btc_aligned column, matching the live _btc_aligned() filter.
  * Tier B risk 3/4/6% by ADX, October x0.5, monthly -10% circuit breaker,
    equity throttle x0.50 when >=25% below the running peak.
  * Peak + circuit-breaker state START FRESH at the forward window (peak = $500),
    because this models opening a brand-new $500 account on 2026-01-01.

This is the genuine out-of-sample number: how $500 would have compounded on data
the strategy never saw during tuning.

== USAGE (on the VPS, with FRESH data through today) ==
  cd C:\\Temp\\TradingBotV1
  # 1. refresh the H1 cache (collect_data skips files that already exist):
  del btc_research\\eth_bot\\backtest\\data\\ETHUSD_H1.csv
  del btc_research\\eth_bot\\backtest\\data\\BTCUSD_H1.csv
  # 2. re-collect + regenerate trades + run the forward test:
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.collect_data
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.run_backtest
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.forward_test
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR = Path(__file__).parent
CSV  = _DIR / "data" / "backtest_trades.csv"

# -- Forward window ------------------------------------------------------------
FORWARD_START = pd.Timestamp("2026-01-01")
FORWARD_END   = pd.Timestamp("2027-01-01")   # exclusive upper bound (all of 2026)

# -- Fixed sim parameters (must match live signal_engine) ----------------------
STARTING_BALANCE = 500.0
OCT_RISK_FACTOR  = 0.5
CB_THRESHOLD     = -0.10
RISK_EARLY, RISK_TRANS, RISK_STRONG = 0.030, 0.040, 0.060  # Tier B (locked live)
DD_THROTTLE_TRIGGER = -0.25
DD_THROTTLE_FACTOR  = 0.50

INCLUDE_COMPLEMENT = False   # live trades PURE S4; flip True for an S4+9 reference

# -- Slot sets (identical to realistic_backtest) -------------------------------
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
    """One-position-at-a-time compound sim. cand sorted by entry_time.
    Peak + circuit-breaker state start fresh (models a new $500 account)."""
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

        if et < last_exit:          # position busy -> miss (live one-trade rule)
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

    if not records:
        return {"sim": pd.DataFrame(), "n_taken": 0, "n_missed": n_missed_busy,
                "n_throttled": 0, "n_halted": 0, "wr": 0.0, "tot_r": 0.0,
                "final": STARTING_BALANCE, "ret_pct": 0.0, "maxdd": 0.0, "maxcl": 0}

    sim = pd.DataFrame(records)
    bser = sim["balance"]
    peakser = bser.cummax()
    max_dd = ((bser - peakser) / peakser * 100).min()

    max_cl = cur = 0
    for o in sim["outcome"]:
        cur = cur + 1 if o == "loss" else 0
        max_cl = max(max_cl, cur)

    final_bal = bser.iloc[-1]
    ret_pct = (final_bal / STARTING_BALANCE - 1) * 100
    wins = (sim["outcome"] == "win").sum()

    return {
        "sim": sim, "n_taken": n_taken, "n_missed": n_missed_busy,
        "n_throttled": n_throttled, "n_halted": n_halted,
        "wr": wins / n_taken * 100 if n_taken else 0.0,
        "tot_r": sim["r"].sum(), "final": final_bal, "ret_pct": ret_pct,
        "maxdd": max_dd, "maxcl": max_cl,
    }


def _monthly(sim: pd.DataFrame) -> None:
    if sim.empty:
        print("  (no trades taken in window)")
        return
    sim = sim.copy()
    sim["month"] = sim["entry_time"].dt.to_period("M")
    print(f"  {'Month':<9} {'N':>3} {'W':>3} {'WR%':>6} {'TotR':>7} "
          f"{'P&L $':>10} {'EndBal':>11}")
    print(_bar())
    for m, g in sim.groupby("month", sort=True):
        n = len(g); w = (g["outcome"] == "win").sum()
        print(f"  {str(m):<9} {n:>3} {w:>3} {w/n*100:>5.1f}% "
              f"{g['r'].sum():>+7.2f} {g['pnl'].sum():>+10.2f} "
              f"{g['balance'].iloc[-1]:>11,.2f}")


def main() -> None:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first")
        sys.exit(1)
    df = pd.read_csv(CSV, parse_dates=["entry_time", "exit_time"])
    for _col in ("entry_time", "exit_time"):
        if df[_col].dt.tz is not None:
            df[_col] = df[_col].dt.tz_convert("UTC").dt.tz_localize(None)

    full_min, full_max = df["entry_time"].min(), df["entry_time"].max()
    df = df[(df["entry_time"] >= FORWARD_START) & (df["entry_time"] < FORWARD_END)]
    df = df.reset_index(drop=True)

    print()
    print(_bar("="))
    print(f"  ETH BOT -- OUT-OF-SAMPLE FORWARD RUN  "
          f"({FORWARD_START.date()} -> {FORWARD_END.date()}, exclusive)")
    print(f"  Fresh ${STARTING_BALANCE:.0f} account | one position at a time "
          f"| Tier B 3/4/6% | Oct x0.5 | CB -10% | throttle x0.5 @ -25% peak")
    print(_bar("="))

    if df.empty:
        print(f"  NO TRADES in the forward window.")
        print(f"  CSV trade range: {full_min} -> {full_max}")
        print(f"  => Your backtest_trades.csv has no 2026 data. Refresh the H1 cache")
        print(f"     (delete ETHUSD_H1.csv + BTCUSD_H1.csv), re-run collect_data,")
        print(f"     then run_backtest, then this script again.")
        print(_bar("="))
        return

    sets = [("PURE S4 (live config)", S4_SLOTS)]
    if INCLUDE_COMPLEMENT:
        sets.append(("S4 + 9 COMPLEMENT (ref)", S4_SLOTS + COMPLEMENT_9))
    results = [(name, _simulate_onepos(_select(df, slots))) for name, slots in sets]

    print(f"  {'slot set':<24} {'taken':>6} {'missed':>7} {'WR%':>6} {'TotR':>7} "
          f"{'finalBal':>11} {'Return%':>9} {'MaxDD%':>8} {'MaxCL':>6}")
    print(_bar())
    for name, r in results:
        print(f"  {name:<24} {r['n_taken']:>6} {r['n_missed']:>7} {r['wr']:>5.1f}% "
              f"{r['tot_r']:>+7.1f} {r['final']:>11,.2f} {r['ret_pct']:>+8.1f}% "
              f"{r['maxdd']:>+7.1f}% {r['maxcl']:>6}")
    print(_bar())
    print("  'missed' = signals skipped because a position was already open (live reality).")
    print("  NOTE: <6 months of 2026 = small sample. Read totR / WR, not annualised CAGR.")

    for name, r in results:
        print()
        print(_bar("="))
        print(f"  MONTHLY -- {name}   "
              f"(final ${r['final']:,.2f}, return {r['ret_pct']:+.1f}%, "
              f"MaxDD {r['maxdd']:.1f}%, CL {r['maxcl']}, "
              f"throttled {r['n_throttled']}, CBskip {r['n_halted']})")
        print(_bar())
        _monthly(r["sim"])
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
