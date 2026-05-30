"""
btc_research/analysis_btc2_temporal.py — Bot 2 temporal performance breakdown.

Final config: EMA200 + ADX-split risk + hours [1,2,3,8] UTC
  ADX-split: 3% at ADX<=25, 2% at ADX 25-40, 3% at ADX>=40

Breakdowns:
  1. By YEAR HALF  (H1/H2)
  2. By QUARTER    (Q1-Q4 each year)
  3. By MONTH      (full 24-month grid)
  4. Consistency metrics: % profitable quarters, % profitable months, streak analysis

Run:
    python btc_research/analysis_btc2_temporal.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(str(_ROOT))

import pandas as pd
import numpy as np
import btc_research.settings as cfg
from btc_research.data.fetcher import fetch_all
from btc_research.btc_bot_2.strategy.vb_swing_combined import VBSwingStrategy
from btc_research.btc_bot_2 import settings as b2cfg

# ── Data ──────────────────────────────────────────────────────────────────────
print("Fetching 2yr data...")
data   = fetch_all(use_cache=True, force_refresh=False)
df_btc = data.get(cfg.BTC_SYMBOL, pd.DataFrame())

if df_btc.empty:
    print("ERROR: No BTC data.")
    sys.exit(1)

if "time" in df_btc.columns:
    df_btc = df_btc.set_index(pd.to_datetime(df_btc["time"], utc=True)).drop(columns=["time"])
elif not isinstance(df_btc.index, pd.DatetimeIndex):
    df_btc.index = pd.to_datetime(df_btc.index, utc=True)

print(f"Loaded {len(df_btc):,} H1 bars  ({df_btc.index[0].date()} → {df_btc.index[-1].date()})\n")

# ── Indicators ────────────────────────────────────────────────────────────────
_c = df_btc["close"].astype(float)
_h = df_btc["high"].astype(float)
_l = df_btc["low"].astype(float)
_tr = pd.concat([_h - _l, (_h - _c.shift(1)).abs(), (_l - _c.shift(1)).abs()], axis=1).max(axis=1)
ATR_ARR = _tr.rolling(14).mean().bfill().values
EMA200  = _c.ewm(span=200, adjust=False).mean().values
TS      = df_btc.index
H_ARR   = _h.values
L_ARR   = _l.values
C_ARR   = _c.values


def _calc_adx(period: int = 14) -> np.ndarray:
    sp  = 2 * period - 1
    hd  = _h.diff(); ld = _l.diff()
    pdm = hd.where((hd > 0) & (hd > -ld), 0.0)
    mdm = (-ld).where((-ld > 0) & (-ld > hd), 0.0)
    aw  = _tr.ewm(span=sp, adjust=False).mean()
    pw  = pdm.ewm(span=sp, adjust=False).mean()
    mw  = mdm.ewm(span=sp, adjust=False).mean()
    pdi = 100 * pw / aw; mdi = 100 * mw / aw
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, float("nan"))
    return dx.ewm(span=sp, adjust=False).mean().fillna(0).values


ADX_ARR = _calc_adx(14)


# ── Simulator (final Bot 2 config) ────────────────────────────────────────────
def simulate_final() -> list[dict]:
    """Run the definitive Bot 2 config: EMA200 + ADX-split risk + [1,2,3,8] UTC."""
    strat   = VBSwingStrategy(atr_multiplier=b2cfg.VB_ATR_MULTIPLIER, close_zone=b2cfg.VB_CLOSE_ZONE)
    balance = float(b2cfg.STARTING_BALANCE)
    trades: list[dict] = []
    open_t  = None

    for i in range(220, len(df_btc)):
        bar_time = TS[i]
        hr   = bar_time.hour
        bh_  = float(H_ARR[i])
        bl_  = float(L_ARR[i])
        bc_  = float(C_ARR[i])

        # ── Manage open trade ────────────────────────────────────────────────
        if open_t is not None:
            long_   = open_t["direction"] == "long"
            entry_  = open_t["entry"]
            sl_dist = abs(entry_ - open_t["orig_sl"])
            risk_u  = open_t["risk_usd"]
            age     = i - open_t["open_idx"]

            if not open_t.get("tp1_hit"):
                hit_sl  = (bl_ <= open_t["sl"]) if long_ else (bh_ >= open_t["sl"])
                hit_tp1 = (bh_ >= open_t["tp1"]) if long_ else (bl_ <= open_t["tp1"])
                if hit_sl:
                    pnl = -risk_u; r_ = -1.0; ex = "SL"
                elif hit_tp1:
                    pnl = risk_u * open_t["tp1_rr"]; r_ = open_t["tp1_rr"]
                    open_t["tp1_hit"]    = True
                    open_t["sl"]         = entry_
                    open_t["trail_peak"] = bh_ if long_ else bl_
                    balance += pnl
                    open_t["pnl_running"] = round(pnl, 2)
                    open_t["r_running"]   = r_
                    continue
                elif age >= b2cfg.MAX_HOLD_BARS:
                    pnl = (bc_ - entry_ if long_ else entry_ - bc_) * open_t["lots"]
                    r_  = pnl / risk_u if risk_u else 0; ex = "MAX_HOLD"
                else:
                    continue
            else:
                atr_now = float(ATR_ARR[min(i, len(ATR_ARR) - 1)])
                if long_:
                    if bh_ > open_t["trail_peak"]: open_t["trail_peak"] = bh_
                    open_t["sl"] = max(open_t["trail_peak"] - b2cfg.TRAIL_ATR_MULT * atr_now, entry_)
                else:
                    if bl_ < open_t["trail_peak"]: open_t["trail_peak"] = bl_
                    open_t["sl"] = min(open_t["trail_peak"] + b2cfg.TRAIL_ATR_MULT * atr_now, entry_)
                new_sl  = open_t["sl"]
                hit_sl2 = (bl_ <= new_sl) if long_ else (bh_ >= new_sl)
                if hit_sl2:
                    dist_run = abs(new_sl - entry_)
                    r_  = dist_run / sl_dist if sl_dist else 0
                    pnl = risk_u * r_; ex = "TRAIL_SL"
                elif age >= b2cfg.MAX_HOLD_BARS:
                    dist_run = abs(bc_ - entry_)
                    r_  = dist_run / sl_dist if sl_dist else 0
                    pnl = risk_u * r_; ex = "MAX_HOLD"
                else:
                    continue

            balance += pnl
            open_t["pnl_usd"]       = round(open_t.get("pnl_running", 0) + pnl, 2)
            open_t["r_multiple"]    = round(open_t.get("r_running", 0) + r_, 2)
            open_t["balance_after"] = round(balance, 2)
            open_t["exit_reason"]   = ex
            trades.append(open_t)
            open_t = None

        if open_t is not None:
            continue

        if hr not in b2cfg.KZ_HOURS:
            continue

        adx_now    = float(ADX_ARR[i])
        ema200_now = float(EMA200[i])

        if adx_now < b2cfg.ADX_THRESHOLD:
            continue

        # EMA200 filter
        win = df_btc.iloc[max(0, i - 220):i + 1]

        for direction in ("long", "short"):
            if direction == "long"  and bc_ < ema200_now: continue
            if direction == "short" and bc_ > ema200_now: continue

            sig = strat.generate_signal(win, bar_time, direction)
            if not sig.get("signal"):
                continue

            sl_d = abs(float(sig["entry"]) - float(sig["sl"]))
            if sl_d <= 0:
                continue

            # ADX-split risk
            if adx_now >= b2cfg.ADX_SPLIT_STRONG_MIN:
                risk_pct = b2cfg.RISK_PCT_STRONG
            elif adx_now <= b2cfg.ADX_SPLIT_EARLY_MAX:
                risk_pct = b2cfg.RISK_PCT_EARLY_TREND
            else:
                risk_pct = b2cfg.RISK_PCT_TRANSITION

            ru   = round(balance * risk_pct, 2)
            lots = ru / sl_d
            tp1r = sig.get("tp1_rr", b2cfg.TP1_RR)
            tp2r = sig.get("tp2_rr", b2cfg.TP2_RR)
            tp1  = float(sig["entry"]) + tp1r * sl_d if direction == "long" else float(sig["entry"]) - tp1r * sl_d
            tp2  = float(sig["entry"]) + tp2r * sl_d if direction == "long" else float(sig["entry"]) - tp2r * sl_d

            open_t = {
                "open_time":     bar_time,
                "open_idx":      i,
                "direction":     direction,
                "entry":         float(sig["entry"]),
                "sl":            float(sig["sl"]),
                "orig_sl":       float(sig["sl"]),
                "tp1":           tp1, "tp2": tp2,
                "tp1_rr":        tp1r, "tp2_rr": tp2r,
                "lots":          lots, "risk_usd": ru, "risk_pct": risk_pct,
                "signal_reason": sig.get("reason", ""),
                "strategy_used": sig.get("strategy_used", ""),
                "trail_peak":    0.0, "tp1_hit": False,
                "pnl_running":   0.0, "r_running": 0.0, "pnl_usd": 0.0,
                "adx_at_entry":  round(adx_now, 1),
                "atr_at_entry":  round(float(ATR_ARR[i]), 0),
                "above_ema200":  bc_ > ema200_now,
                "hour":          hr,
                "exit_reason":   "",
                "balance_after": 0.0,
                "r_multiple":    0.0,
            }
            break

    return trades


# ── Stats helpers ─────────────────────────────────────────────────────────────
def _stats(trades: list[dict], running_start_bal: float | None = None) -> dict:
    """Compute stats. running_start_bal allows correct MaxDD for sub-periods."""
    if not trades:
        start = running_start_bal if running_start_bal is not None else b2cfg.STARTING_BALANCE
        return {"n": 0, "wr": 0.0, "avgr": 0.0, "pnl": 0.0,
                "start_bal": start, "end_bal": start, "maxdd": 0.0,
                "vb": 0, "sl_": 0, "longs": 0, "shorts": 0, "profitable": False}
    wins  = [t for t in trades if t["pnl_usd"] > 0]
    start = running_start_bal if running_start_bal is not None else b2cfg.STARTING_BALANCE
    bals  = [start] + [t["balance_after"] for t in trades]
    peak  = start; maxdd = 0.0
    for b in bals:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100
        if dd > maxdd: maxdd = dd
    return {
        "n":          len(trades),
        "wr":         round(len(wins) / len(trades) * 100, 1),
        "avgr":       round(sum(t["r_multiple"] for t in trades) / len(trades), 2),
        "pnl":        round(sum(t["pnl_usd"] for t in trades), 2),
        "start_bal":  round(start, 2),
        "end_bal":    round(bals[-1], 2),
        "maxdd":      round(maxdd, 1),
        "vb":         len([t for t in trades if t.get("strategy_used") == "Volatility Breakout"]),
        "sl_":        len([t for t in trades if t.get("strategy_used") == "Swing Level Break"]),
        "longs":      len([t for t in trades if t["direction"] == "long"]),
        "shorts":     len([t for t in trades if t["direction"] == "short"]),
        "profitable": sum(t["pnl_usd"] for t in trades) > 0,
    }


def _grade(s: dict) -> str:
    if s["n"] == 0:
        return "— no trades"
    if s["n"] < 5:
        return "⚠  few trades"
    if s["wr"] >= 60 and s["avgr"] >= 1.0:
        return "🟢 EXCELLENT"
    if s["wr"] >= 50 and s["avgr"] >= 0.5:
        return "🟢 STRONG"
    if s["wr"] >= 45 and s["avgr"] >= 0.3:
        return "🟡 GOOD"
    if s["wr"] >= 40 and s["avgr"] >= 0.0:
        return "🟡 MARGINAL"
    if s["pnl"] < 0:
        return "🔴 LOSING"
    return "🟠 WEAK"


# ── Run final config ──────────────────────────────────────────────────────────
print("Running final Bot 2 config (EMA200 + ADX-split + [1,2,3,8] UTC)...")
trades = simulate_final()
total  = _stats(trades)
print(f"Total: {total['n']} trades | WR={total['wr']}% | AvgR={total['avgr']:+.2f}R | "
      f"PnL=${total['pnl']:+,.2f} | MaxDD={total['maxdd']}%\n")

W = 112

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — YEAR HALVES
# ─────────────────────────────────────────────────────────────────────────────
print("=" * W)
print("  SECTION 1 — YEAR HALVES  (H1 = Jan-Jun  |  H2 = Jul-Dec)")
print("=" * W)
print(f"  {'Period':>10}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>10}  {'Start $':>9}  "
      f"{'End $':>9}  {'MaxDD':>7}  {'VB':>3}  {'SL':>3}  {'L/S':>7}  Grade")
print("  " + "-" * 105)

half_groups: dict[str, list[dict]] = defaultdict(list)
for t in trades:
    dt   = t["open_time"]
    half = f"H1 {dt.year}" if dt.month <= 6 else f"H2 {dt.year}"
    half_groups[half].append(t)

# Sort halves chronologically
half_order = sorted(half_groups.keys(), key=lambda x: (int(x.split()[1]), 0 if x.startswith("H1") else 1))

running_bal = b2cfg.STARTING_BALANCE
half_stats  = {}
for half in half_order:
    ts_ = half_groups[half]
    s   = _stats(ts_, running_bal)
    half_stats[half] = s
    print(f"  {half:>10}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+9,.2f}  ${s['start_bal']:>8,.2f}  ${s['end_bal']:>8,.2f}  "
          f"{s['maxdd']:>6.1f}%  {s['vb']:>3}  {s['sl_']:>3}  "
          f"{s['longs']}L/{s['shorts']}S  {_grade(s)}")
    running_bal = s["end_bal"]

print("  " + "-" * 105)
s = total
print(f"  {'TOTAL':>10}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
      f"${s['pnl']:>+9,.2f}  ${b2cfg.STARTING_BALANCE:>8,.2f}  ${s['end_bal']:>8,.2f}  "
      f"{s['maxdd']:>6.1f}%  {s['vb']:>3}  {s['sl_']:>3}  {s['longs']}L/{s['shorts']}S")

# Half consistency
profitable_halves = sum(1 for s in half_stats.values() if s["profitable"] and s["n"] > 0)
total_halves      = sum(1 for s in half_stats.values() if s["n"] > 0)
print(f"\n  Profitable halves: {profitable_halves}/{total_halves}  "
      f"({profitable_halves/total_halves*100:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — QUARTERS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * W)
print("  SECTION 2 — QUARTERLY BREAKDOWN")
print("=" * W)
print(f"  {'Quarter':>8}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>10}  {'Start $':>9}  "
      f"{'End $':>9}  {'MaxDD':>7}  {'VB':>3}  {'SL':>3}  {'L/S':>7}  Grade")
print("  " + "-" * 105)

quarter_map = {1: "Q1", 2: "Q1", 3: "Q1",
               4: "Q2", 5: "Q2", 6: "Q2",
               7: "Q3", 8: "Q3", 9: "Q3",
               10: "Q4", 11: "Q4", 12: "Q4"}

qtr_groups: dict[str, list[dict]] = defaultdict(list)
for t in trades:
    dt  = t["open_time"]
    key = f"{quarter_map[dt.month]} {dt.year}"
    qtr_groups[key].append(t)

qtr_order = sorted(qtr_groups.keys(), key=lambda x: (int(x.split()[1]), x.split()[0]))

running_bal = b2cfg.STARTING_BALANCE
qtr_stats   = {}
for qtr in qtr_order:
    ts_ = qtr_groups[qtr]
    s   = _stats(ts_, running_bal)
    qtr_stats[qtr] = s
    print(f"  {qtr:>8}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+9,.2f}  ${s['start_bal']:>8,.2f}  ${s['end_bal']:>8,.2f}  "
          f"{s['maxdd']:>6.1f}%  {s['vb']:>3}  {s['sl_']:>3}  "
          f"{s['longs']}L/{s['shorts']}S  {_grade(s)}")
    running_bal = s["end_bal"]

print("  " + "-" * 105)
s = total
print(f"  {'TOTAL':>8}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
      f"${s['pnl']:>+9,.2f}  ${b2cfg.STARTING_BALANCE:>8,.2f}  ${s['end_bal']:>8,.2f}  "
      f"{s['maxdd']:>6.1f}%")

# Quarter consistency
profitable_qtrs = sum(1 for s in qtr_stats.values() if s["profitable"] and s["n"] > 0)
total_qtrs      = sum(1 for s in qtr_stats.values() if s["n"] > 0)
best_qtr        = max(qtr_stats.items(), key=lambda x: x[1]["pnl"] if x[1]["n"] > 0 else -9999)
worst_qtr       = min(qtr_stats.items(), key=lambda x: x[1]["pnl"] if x[1]["n"] > 0 else 9999)
print(f"\n  Profitable quarters: {profitable_qtrs}/{total_qtrs}  "
      f"({profitable_qtrs/total_qtrs*100:.0f}%)")
print(f"  Best quarter : {best_qtr[0]}  PnL=${best_qtr[1]['pnl']:+,.2f}  WR={best_qtr[1]['wr']}%")
print(f"  Worst quarter: {worst_qtr[0]}  PnL=${worst_qtr[1]['pnl']:+,.2f}  WR={worst_qtr[1]['wr']}%")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MONTHLY BREAKDOWN (full grid)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * W)
print("  SECTION 3 — MONTHLY BREAKDOWN (24-month grid)")
print("=" * W)
print(f"  {'Month':>7}  {'#':>3}  {'WR%':>5}  {'AvgR':>6}  {'PnL $':>9}  {'Start $':>9}  "
      f"{'End $':>9}  {'MaxDD':>6}  {'Exit breakdown':20}  {'L/S':>6}  Grade")
print("  " + "-" * 107)

monthly: dict[str, list[dict]] = defaultdict(list)
for t in trades:
    ym = str(t["open_time"])[:7]
    monthly[ym].append(t)

# Build full month grid (fill months with 0 trades)
from dateutil.relativedelta import relativedelta as _rd
import datetime as _dt

start_ym = str(trades[0]["open_time"])[:7]
end_ym   = str(trades[-1]["open_time"])[:7]
cur      = _dt.datetime.strptime(start_ym, "%Y-%m")
end_dt   = _dt.datetime.strptime(end_ym,   "%Y-%m")
all_months = []
while cur <= end_dt:
    all_months.append(cur.strftime("%Y-%m"))
    cur += _rd(months=1)

running_bal = b2cfg.STARTING_BALANCE
month_stats: dict[str, dict] = {}
prev_qtr    = ""
for ym in all_months:
    ts_  = monthly.get(ym, [])
    s    = _stats(ts_, running_bal)
    month_stats[ym] = s

    # Quarter separator
    mo   = int(ym[5:7])
    qtr  = quarter_map[mo]
    yr   = ym[:4]
    this_qtr = f"{qtr} {yr}"
    if this_qtr != prev_qtr:
        print(f"  {'─'*105}")
        prev_qtr = this_qtr

    # Exit breakdown
    if ts_:
        sl_    = len([t for t in ts_ if t["exit_reason"] == "SL"])
        tp_    = len([t for t in ts_ if t["exit_reason"] == "TRAIL_SL"])
        mh_    = len([t for t in ts_ if t["exit_reason"] == "MAX_HOLD"])
        ex_str = f"SL={sl_} TR={tp_} MH={mh_}"
    else:
        ex_str = "—"

    pnl_tag = f"${s['pnl']:>+8,.2f}" if s["n"] > 0 else "        —"
    ls_str  = f"{s['longs']}L/{s['shorts']}S" if s["n"] > 0 else "—"
    grade   = _grade(s) if s["n"] > 0 else ""

    print(f"  {ym:>7}  {s['n']:>3}  {s['wr']:>4.1f}%  {s['avgr']:>+5.2f}R  "
          f"{pnl_tag}  ${s['start_bal']:>8,.2f}  ${s['end_bal']:>8,.2f}  "
          f"{s['maxdd']:>5.1f}%  {ex_str:20}  {ls_str:>6}  {grade}")
    running_bal = s["end_bal"]

print(f"  {'─'*105}")
s = total
print(f"  {'TOTAL':>7}  {s['n']:>3}  {s['wr']:>4.1f}%  {s['avgr']:>+5.2f}R  "
      f"${s['pnl']:>+8,.2f}  ${b2cfg.STARTING_BALANCE:>8,.2f}  ${s['end_bal']:>8,.2f}  "
      f"{s['maxdd']:>5.1f}%")

# Monthly consistency
profitable_months = sum(1 for s in month_stats.values() if s["profitable"] and s["n"] > 0)
total_months_w    = sum(1 for s in month_stats.values() if s["n"] > 0)
best_mo           = max(month_stats.items(), key=lambda x: x[1]["pnl"] if x[1]["n"] > 0 else -9999)
worst_mo          = min(month_stats.items(), key=lambda x: x[1]["pnl"] if x[1]["n"] > 0 else 9999)
print(f"\n  Profitable months  : {profitable_months}/{total_months_w}  "
      f"({profitable_months/total_months_w*100:.0f}%)")
print(f"  Best month  : {best_mo[0]}  PnL=${best_mo[1]['pnl']:+,.2f}  WR={best_mo[1]['wr']}%  n={best_mo[1]['n']}")
print(f"  Worst month : {worst_mo[0]}  PnL=${worst_mo[1]['pnl']:+,.2f}  WR={worst_mo[1]['wr']}%  n={worst_mo[1]['n']}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — STREAK & CONSISTENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * W)
print("  SECTION 4 — STREAK & CONSISTENCY ANALYSIS")
print("=" * W)

# Win/loss streak on individual trades
max_win_streak = max_loss_streak = 0
cur_win = cur_loss = 0
for t in trades:
    if t["pnl_usd"] > 0:
        cur_win  += 1; cur_loss = 0
    else:
        cur_loss += 1; cur_win  = 0
    max_win_streak  = max(max_win_streak,  cur_win)
    max_loss_streak = max(max_loss_streak, cur_loss)

# Monthly win/loss streaks
month_pnl_seq = [month_stats[ym]["pnl"] for ym in all_months if month_stats[ym]["n"] > 0]
mw_str = ml_str = 0
cmw = cml = 0
for p in month_pnl_seq:
    if p > 0:
        cmw += 1; cml = 0
    else:
        cml += 1; cmw = 0
    mw_str = max(mw_str, cmw)
    ml_str = max(ml_str, cml)

# Avg monthly PnL and std dev
monthly_pnls = [month_stats[ym]["pnl"] for ym in all_months if month_stats[ym]["n"] > 0]
avg_mo_pnl   = np.mean(monthly_pnls)
std_mo_pnl   = np.std(monthly_pnls)
sharpe_mo    = avg_mo_pnl / std_mo_pnl if std_mo_pnl > 0 else 0

# Profit factor
gross_wins   = sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0)
gross_losses = abs(sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0))
pf           = gross_wins / gross_losses if gross_losses > 0 else float("inf")

# Recovery factor
recovery = total["pnl"] / (total["maxdd"] / 100 * b2cfg.STARTING_BALANCE) if total["maxdd"] > 0 else float("inf")

# Calmar (annual return / max DD) — 2yr data, annualise PnL
annual_return_pct = (total["pnl"] / b2cfg.STARTING_BALANCE) * 100 / 2
calmar = annual_return_pct / total["maxdd"] if total["maxdd"] > 0 else float("inf")

print(f"  Trade streaks:")
print(f"    Max consecutive wins  : {max_win_streak}")
print(f"    Max consecutive losses: {max_loss_streak}")
print()
print(f"  Monthly streaks (active months only):")
print(f"    Max consecutive profitable months: {mw_str}")
print(f"    Max consecutive losing months    : {ml_str}")
print()
print(f"  Monthly PnL distribution:")
print(f"    Avg monthly PnL : ${avg_mo_pnl:>+8,.2f}")
print(f"    Std dev         : ${std_mo_pnl:>8,.2f}")
print(f"    Monthly Sharpe  : {sharpe_mo:>+.2f}  (>1.0 = good, >1.5 = very good)")
print()
print(f"  Risk-adjusted metrics:")
print(f"    Profit Factor   : {pf:.2f}  (>1.5 = good, >2.0 = very good)")
print(f"    Recovery Factor : {recovery:.2f}  (PnL / MaxDD$)")
print(f"    Calmar Ratio    : {calmar:.2f}  (annualised return% / MaxDD%)")
print(f"    Gross Wins      : ${gross_wins:>+9,.2f}")
print(f"    Gross Losses    : $-{gross_losses:>8,.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — YEAR-ON-YEAR COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * W)
print("  SECTION 5 — YEAR-ON-YEAR COMPARISON")
print("  Note: 2024 = partial (May-Dec), 2025 = full year, 2026 = partial (Jan-May)")
print("=" * W)
print(f"  {'Year':>6}  {'Months':>6}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>10}  "
      f"{'Start $':>9}  {'End $':>9}  {'MaxDD':>7}  Grade")
print("  " + "-" * 93)

year_groups: dict[str, list[dict]] = defaultdict(list)
for t in trades:
    year_groups[str(t["open_time"].year)].append(t)

running_bal = b2cfg.STARTING_BALANCE
for yr in sorted(year_groups.keys()):
    ts_ = year_groups[yr]
    s   = _stats(ts_, running_bal)
    mo_count = len(set(str(t["open_time"])[:7] for t in ts_))
    note = "(partial)" if yr in ("2024", "2026") else "(full)"
    print(f"  {yr:>6}  {mo_count:>4} mo  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+9,.2f}  ${s['start_bal']:>8,.2f}  ${s['end_bal']:>8,.2f}  "
          f"{s['maxdd']:>6.1f}%  {_grade(s)}  {note}")
    running_bal = s["end_bal"]

print()
print("=" * W)
print("  SUMMARY vs BOT 1")
print("=" * W)
print(f"  {'Metric':25s}  {'Bot 2 (Final)':>18}  {'Bot 1 (Ref)':>12}  {'Delta':>10}")
print("  " + "-" * 72)
comparisons = [
    ("Trades",               str(total["n"]),              "223",          ""),
    ("Win Rate",             f"{total['wr']}%",            "43.0%",        f"{total['wr']-43.0:+.1f}%"),
    ("Avg R",                f"{total['avgr']:+.2f}R",     "+0.47R",       f"{total['avgr']-0.47:+.2f}R"),
    ("Total PnL",            f"${total['pnl']:+,.2f}",     "+$23,733",     f"${total['pnl']-23733:+,.2f}"),
    ("Max Drawdown",         f"{total['maxdd']}%",         "16.1%",        f"{total['maxdd']-16.1:+.1f}%"),
    ("Profit Factor",        f"{pf:.2f}",                  "—",            ""),
    ("Monthly Sharpe",       f"{sharpe_mo:.2f}",           "—",            ""),
    ("Calmar Ratio",         f"{calmar:.2f}",              "—",            ""),
    ("Profitable months",    f"{profitable_months}/{total_months_w}",  "—", ""),
    ("Profitable quarters",  f"{profitable_qtrs}/{total_qtrs}",        "—", ""),
]
for name, b2_val, b1_val, delta in comparisons:
    print(f"  {name:25s}  {b2_val:>18}  {b1_val:>12}  {delta:>10}")
print()
