"""
btc_research/eth_bot/backtest/exit_sweep.py

EXIT / TRADE-MANAGEMENT SWEEP  (capture more of the runner)

The entry-quality study proved entries are well-timed (winners take little heat,
median MFE ~3.6R) but we only realise ~+0.30R/trade -- the gap is in the EXIT.
This sweep re-simulates every live-taken trade (S4+9, one-position) bar-by-bar
under different exit rules and re-compounds under the locked Tier B risk model,
to see whether a looser trail / higher TP2 / longer hold banks more of that run
WITHOUT breaking the hard risk limits.

Three exit knobs are swept (TP1 stays at 2R -- that's the proven scale-out):
  * TP2_RR          final-target R for the runner half      (current 4.0)
  * TRAIL_ATR_MULT  trailing stop width after TP1, x ATR    (current 2.0; 99=off)
  * MAX_HOLD_BARS   force-close after N H1 bars             (current 96)

Because a changed exit moves each trade's exit_time, the one-position rule is
re-applied per config (a longer hold can block a later signal). Compounding uses
the live Tier B model: 3/4/6 ADX risk, Oct x0.5, monthly -10% CB, -25% x0.50
equity throttle. Configs are flagged OK only if MaxCL < 16 AND MaxDD > -42%.

NO lookahead: exits are walked strictly on bars after the entry bar.

== USAGE ==
  # run AFTER run_backtest.py (TP1-fixed trades) and with ETHUSD_H1.csv present
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.exit_sweep
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR    = Path(__file__).parent
CSV     = _DIR / "data" / "backtest_trades.csv"
ETH_CSV = _DIR / "data" / "ETHUSD_H1.csv"

# -- Fixed sim params ----------------------------------------------------------
START_YEAR       = 2023
STARTING_BALANCE = 500.0
TP1_RR           = 2.0          # scale-out point (50% off, SL->breakeven) -- fixed
OCT_RISK_FACTOR  = 0.5
CB_THRESHOLD     = -0.10

# -- Locked Tier B risk model --------------------------------------------------
RISK_EARLY, RISK_TRANS, RISK_STRONG = 0.030, 0.040, 0.060
THROTTLE_TRIGGER = -0.25
THROTTLE_FACTOR  = 0.50

# -- Hard limits ---------------------------------------------------------------
MAX_CL_LIMIT = 16
MAX_DD_LIMIT = -42.0

# -- Baseline exit (current live config) ---------------------------------------
BASE_TP2, BASE_TRAIL, BASE_HOLD = 4.0, 2.0, 96

# -- Sweep grids ---------------------------------------------------------------
TP2_GRID   = [3.0, 4.0, 5.0, 6.0, 8.0]
TRAIL_GRID = [1.5, 2.0, 3.0, 4.0, 99.0]   # 99 ~ trail OFF (breakeven stop only)
HOLD_GRID  = [48, 96, 144, 240]

# -- Slot set ------------------------------------------------------------------
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


def _bar(c: str = "-", w: int = 100) -> str:
    return c * w


def _risk_pct(adx: float) -> float:
    if adx >= 40:
        return RISK_STRONG
    if adx <= 25:
        return RISK_EARLY
    return RISK_TRANS


# ── Data ─────────────────────────────────────────────────────────────────────
def _load_trades() -> pd.DataFrame:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first")
        sys.exit(1)
    df = pd.read_csv(CSV, parse_dates=["entry_time", "exit_time"])
    for c in ("entry_time", "exit_time"):
        if df[c].dt.tz is not None:
            df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
    return df[df["entry_time"].dt.year >= START_YEAR].reset_index(drop=True)


def _load_eth() -> pd.DataFrame:
    if not ETH_CSV.exists():
        print(f"ERROR: {ETH_CSV} not found -- run collect_data.py first")
        sys.exit(1)
    df = pd.read_csv(ETH_CSV, parse_dates=["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("time").reset_index(drop=True)


def _select(df: pd.DataFrame, slots) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in slots:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return df[mask].sort_values("entry_time").reset_index(drop=True)


# ── Trade-path simulation with configurable exit ─────────────────────────────
def _simulate(highs, lows, closes, ei, entry, sl_dist, is_long, atr,
              tp2_rr, trail_mult, max_hold):
    """Return (realized_R, exit_idx) for the trade entered at bar ei."""
    sgn = 1.0 if is_long else -1.0
    tp1 = entry + sl_dist * TP1_RR * sgn
    tp2 = entry + sl_dist * tp2_rr * sgn
    cur_sl = entry - sl_dist * sgn
    tp1_hit = False
    realized = 0.0
    n = len(closes)
    end = min(ei + max_hold + 1, n)
    for j in range(ei + 1, end):
        bh, bl, bc = highs[j], lows[j], closes[j]
        if tp1_hit and atr > 0:
            trail = trail_mult * atr
            if is_long:
                cur_sl = max(cur_sl, bc - trail)
            else:
                cur_sl = min(cur_sl, bc + trail)
        hit_tp2 = (bh >= tp2) if is_long else (bl <= tp2)
        if hit_tp2:
            if not tp1_hit:
                realized += 0.5 * TP1_RR
            realized += 0.5 * tp2_rr
            return realized, j
        hit_sl = (bl <= cur_sl) if is_long else (bh >= cur_sl)
        if hit_sl:
            sl_r = (cur_sl - entry) / sl_dist * sgn
            realized += (0.5 if tp1_hit else 1.0) * sl_r
            return realized, j
        if not tp1_hit:
            hit_tp1 = (bh >= tp1) if is_long else (bl <= tp1)
            if hit_tp1:
                tp1_hit = True
                realized += 0.5 * TP1_RR
                cur_sl = entry
    last_j = min(ei + max_hold, n - 1)
    mark_r = (closes[last_j] - entry) / sl_dist * sgn
    realized += (0.5 if tp1_hit else 1.0) * mark_r
    return realized, last_j


def _run_config(cand, eth_time, highs, lows, closes, t2i,
                tp2_rr, trail_mult, max_hold) -> dict:
    """Re-apply one-position rule with this exit, then Tier B compounding."""
    balance = STARTING_BALANCE
    peak = balance
    last_exit = pd.Timestamp.min
    cur_month = None
    month_start_bal = balance
    month_halted = False
    bals, outcomes = [], []

    for _, t in cand.iterrows():
        et = t["entry_time"]
        if et < last_exit:
            continue
        ei = t2i.get(et)
        if ei is None:
            continue
        mk = (et.year, et.month)
        if mk != cur_month:
            cur_month = mk
            month_start_bal = balance
            month_halted = False
        if month_halted:
            continue

        entry   = float(t["entry"]); sl_dist = float(t["sl_dist"])
        if sl_dist <= 0:
            continue
        is_long = (t["direction"] == "long")
        atr = float(t["atr"]) if not pd.isna(t["atr"]) else 0.0

        r, exit_idx = _simulate(highs, lows, closes, ei, entry, sl_dist,
                                is_long, atr, tp2_rr, trail_mult, max_hold)
        last_exit = eth_time[exit_idx]

        rp = _risk_pct(float(t["adx"]))
        if et.month == 10:
            rp *= OCT_RISK_FACTOR
        if (balance - peak) / peak <= THROTTLE_TRIGGER:
            rp *= THROTTLE_FACTOR

        balance += balance * rp * r
        peak = max(peak, balance)
        bals.append(balance)
        outcomes.append(r)

        if (balance - month_start_bal) / month_start_bal <= CB_THRESHOLD:
            month_halted = True

    if not bals:
        return {"final": STARTING_BALANCE, "maxdd": 0, "maxcl": 0,
                "totr": 0, "wr": 0, "n": 0}
    bser = pd.Series(bals)
    max_dd = ((bser - bser.cummax()) / bser.cummax() * 100).min()
    max_cl = cur = 0
    for r in outcomes:
        cur = cur + 1 if r <= 0 else 0
        max_cl = max(max_cl, cur)
    wins = sum(1 for r in outcomes if r > 0)
    return {"final": bals[-1], "maxdd": max_dd, "maxcl": max_cl,
            "totr": sum(outcomes), "wr": wins / len(outcomes) * 100,
            "n": len(outcomes)}


def main() -> None:
    trades = _load_trades()
    eth = _load_eth()
    cand = _select(trades, S4_SLOTS + COMPLEMENT_9)

    eth_time = eth["time"].to_numpy()
    highs = eth["high"].to_numpy()
    lows  = eth["low"].to_numpy()
    closes = eth["close"].to_numpy()
    t2i = {t: i for i, t in enumerate(eth["time"])}

    print()
    print(_bar("="))
    print(f"  ETH BOT -- EXIT / TRADE-MANAGEMENT SWEEP  (S4+9, one-position, Tier B)  {START_YEAR}+")
    print(f"  TP1 fixed 2R | sweep TP2_RR x TRAIL_ATR x MAX_HOLD | guardrail CL<{MAX_CL_LIMIT}, DD>{MAX_DD_LIMIT:.0f}%")
    print(_bar("="))

    # baseline first
    base = _run_config(cand, eth_time, highs, lows, closes, t2i,
                       BASE_TP2, BASE_TRAIL, BASE_HOLD)
    print(f"  BASELINE  TP2={BASE_TP2}  trail={BASE_TRAIL}  hold={BASE_HOLD}")
    print(f"     finalBal ${base['final']:,.0f} | totR {base['totr']:+.1f} | "
          f"WR {base['wr']:.1f}% | MaxDD {base['maxdd']:.1f}% | MaxCL {base['maxcl']} | N {base['n']}")
    print(_bar())

    results = []
    for tp2 in TP2_GRID:
        for tr in TRAIL_GRID:
            for hold in HOLD_GRID:
                r = _run_config(cand, eth_time, highs, lows, closes, t2i, tp2, tr, hold)
                ok = (r["maxcl"] < MAX_CL_LIMIT) and (r["maxdd"] > MAX_DD_LIMIT)
                r.update({"tp2": tp2, "trail": tr, "hold": hold, "ok": ok})
                results.append(r)

    within = [r for r in results if r["ok"]]
    within.sort(key=lambda r: r["final"], reverse=True)

    print(f"  TOP CONFIGS WITHIN LIMITS (of {len(within)}/{len(results)} that pass):")
    print(f"  {'TP2':>5} {'trail':>6} {'hold':>5} {'finalBal':>11} {'totR':>8} "
          f"{'WR%':>6} {'MaxDD%':>8} {'MaxCL':>6} {'vsBase':>9}")
    print(_bar())
    for r in within[:18]:
        trail_s = "off" if r["trail"] >= 99 else f"{r['trail']:.1f}"
        delta = (r["final"] / base["final"] - 1) * 100 if base["final"] else 0
        star = "  <== current" if (r["tp2"] == BASE_TP2 and r["trail"] == BASE_TRAIL
                                    and r["hold"] == BASE_HOLD) else ""
        print(f"  {r['tp2']:>5.1f} {trail_s:>6} {r['hold']:>5} {r['final']:>11,.0f} "
              f"{r['totr']:>+8.1f} {r['wr']:>5.1f}% {r['maxdd']:>+7.1f}% "
              f"{r['maxcl']:>6} {delta:>+8.1f}%{star}")
    print(_bar())

    if within:
        b = within[0]
        trail_s = "off" if b["trail"] >= 99 else f"{b['trail']:.1f}"
        lift = (b["final"] / base["final"] - 1) * 100 if base["final"] else 0
        print(f"  BEST EXIT WITHIN LIMITS: TP2={b['tp2']}  trail={trail_s}  hold={b['hold']}")
        print(f"     finalBal ${b['final']:,.0f}  (vs baseline ${base['final']:,.0f}, "
              f"{lift:+.1f}%) | MaxDD {b['maxdd']:.1f}% | MaxCL {b['maxcl']}")
        if b["final"] <= base["final"] * 1.05:
            print("     -> within 5% of baseline: current exit is already near-optimal; "
                  "no real edge in retuning.")
        else:
            print("     -> a meaningful lift. Validate it isn't a single-config fluke "
                  "(check neighbours rank high too) before locking it in.")
    print(_bar())
    print("  NOTE: higher TP2 / looser trail bank more of the runner but let some give")
    print("  back to the stop; the within-limits flag is the guardrail, not a guarantee.")
    print(_bar())
    print()


if __name__ == "__main__":
    main()
