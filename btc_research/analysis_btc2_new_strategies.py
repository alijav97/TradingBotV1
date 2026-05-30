"""
btc_research/analysis_btc2_new_strategies.py — New strategy candidates for Bot 2.

Tests two new entry strategies vs the current VB + Swing Level baseline:
  1. Swing Break Retest (SBR) — enter on pullback/retest of broken swing level
     "The second breakout after retest is usually the real move"
     Key advantage: tighter SL (retest bar extreme, not prior swing structure)
  2. Inside Bar Breakout (IBB) — enter when compression inside bar(s) break out
     "Compression → explosive move; Asia Night = natural inside bar session"

Tested combinations (all with EMA200 + ADX-split risk + [1,2,3,8] UTC):
  0  Baseline      : VB + Swing Level  (current final config = $20,354)
  1  SBR only      : VB + SBR
  2  IBB only      : VB + IBB
  3  Add SBR       : VB + SL + SBR    (SBR as 3rd strategy)
  4  Add IBB       : VB + SL + IBB    (IBB as 3rd strategy)
  5  SBR + IBB     : VB + SBR + IBB   (replace SL entirely)
  6  All 4         : VB + SL + SBR + IBB

Per-strategy breakdown + monthly breakdown for the best combo.

Run:
    python btc_research/analysis_btc2_new_strategies.py
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
from btc_research.btc_bot_2 import settings as b2cfg
from btc_research.strategies.base                  import BTCStrategy
from btc_research.strategies.volatility_breakout   import VolatilityBreakout
from btc_research.strategies.swing_level           import SwingLevelBreak
from btc_research.strategies.swing_break_retest    import SwingBreakRetest
from btc_research.strategies.inside_bar_breakout   import InsideBarBreakout

# ── Data ──────────────────────────────────────────────────────────────────────
print("Fetching 2yr data...")
data   = fetch_all(use_cache=True, force_refresh=False)
df_btc = data.get(cfg.BTC_SYMBOL, pd.DataFrame())
if df_btc.empty:
    print("ERROR: No BTC data."); sys.exit(1)

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


# ── Simulator ─────────────────────────────────────────────────────────────────
def simulate(strategy_list: list[BTCStrategy]) -> list[dict]:
    """
    Final Bot 2 config: EMA200 + ADX-split risk + [1,2,3,8] UTC.
    Strategies are tried IN ORDER — first signal wins.
    """
    balance = float(b2cfg.STARTING_BALANCE)
    trades: list[dict] = []
    open_t = None

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
                    r_ = dist_run / sl_dist if sl_dist else 0
                    pnl = risk_u * r_; ex = "TRAIL_SL"
                elif age >= b2cfg.MAX_HOLD_BARS:
                    dist_run = abs(bc_ - entry_)
                    r_ = dist_run / sl_dist if sl_dist else 0
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

        win = df_btc.iloc[max(0, i - 220):i + 1]

        for direction in ("long", "short"):
            if direction == "long"  and bc_ < ema200_now: continue
            if direction == "short" and bc_ > ema200_now: continue

            # Try each strategy in order — first signal wins
            sig = None
            for strat in strategy_list:
                result = strat.generate_signal(win, bar_time, direction)
                if result.get("signal"):
                    sig = result
                    sig["strategy_used"] = strat.name
                    sig["tp1_rr"]        = result.get("tp1_rr", b2cfg.TP1_RR)
                    sig["tp2_rr"]        = result.get("tp2_rr", b2cfg.TP2_RR)
                    break

            if sig is None:
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
            tp1r = sig["tp1_rr"]
            tp2r = sig["tp2_rr"]
            tp1  = float(sig["entry"]) + tp1r * sl_d if direction == "long" else float(sig["entry"]) - tp1r * sl_d
            tp2  = float(sig["entry"]) + tp2r * sl_d if direction == "long" else float(sig["entry"]) - tp2r * sl_d

            open_t = {
                "open_time":     bar_time, "open_idx": i,
                "direction":     direction,
                "entry":         float(sig["entry"]),
                "sl":            float(sig["sl"]), "orig_sl": float(sig["sl"]),
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
                "exit_reason":   "", "balance_after": 0.0, "r_multiple": 0.0,
            }
            break   # one trade per bar

    return trades


def _stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "avgr": 0.0, "pnl": 0.0,
                "final_bal": b2cfg.STARTING_BALANCE, "maxdd": 0.0}
    wins  = [t for t in trades if t["pnl_usd"] > 0]
    bals  = [b2cfg.STARTING_BALANCE] + [t["balance_after"] for t in trades]
    peak  = b2cfg.STARTING_BALANCE; maxdd = 0.0
    for b in bals:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100
        if dd > maxdd: maxdd = dd
    return {
        "n":         len(trades),
        "wr":        round(len(wins) / len(trades) * 100, 1),
        "avgr":      round(sum(t["r_multiple"] for t in trades) / len(trades), 2),
        "pnl":       round(sum(t["pnl_usd"] for t in trades), 2),
        "final_bal": round(bals[-1], 2),
        "maxdd":     round(maxdd, 1),
    }


# ── Strategy instances ─────────────────────────────────────────────────────────
VB  = VolatilityBreakout(atr_multiplier=b2cfg.VB_ATR_MULTIPLIER, close_zone=b2cfg.VB_CLOSE_ZONE)
SL  = SwingLevelBreak()
SBR = SwingBreakRetest()
IBB = InsideBarBreakout()

combos = [
    ("0  Baseline  VB + SL      (current)",  [VB, SL]),
    ("1  VB + SBR",                          [VB, SBR]),
    ("2  VB + IBB",                          [VB, IBB]),
    ("3  VB + SL + SBR",                     [VB, SL, SBR]),
    ("4  VB + SL + IBB",                     [VB, SL, IBB]),
    ("5  VB + SBR + IBB  (replace SL)",      [VB, SBR, IBB]),
    ("6  VB + SL + SBR + IBB  (all 4)",      [VB, SL, SBR, IBB]),
]

W = 108
print("Running 7 strategy combinations on 2yr data...")
print("Note: EMA200 + ADX-split + [1,2,3,8] UTC kept fixed for all combos.\n")

results = {}
for label, strat_list in combos:
    t = simulate(strat_list)
    s = _stats(t)
    results[label] = (t, s)
    names = " > ".join(st.name.split(" ")[0] for st in strat_list)
    print(f"  {label:45s}  n={s['n']:3d}  WR={s['wr']:5.1f}%  "
          f"AvgR={s['avgr']:+.2f}R  PnL=${s['pnl']:+,.2f}  MaxDD={s['maxdd']:.1f}%")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * W)
print("  BOT 2 — NEW STRATEGY COMPARISON  |  [01,02,03,08] UTC  |  EMA200 + ADX-split")
print(f"  2yr data: {df_btc.index[0].date()} → {df_btc.index[-1].date()}")
print("=" * W)
print(f"  {'Combo':45s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>10}  "
      f"{'Bal $':>9}  {'MaxDD':>7}")
print("  " + "-" * 96)

base_pnl = results[list(results.keys())[0]][1]["pnl"]
best_pnl = max(s["pnl"] for _, s in results.values())
best_wr  = max(s["wr"]  for _, (_, s) in results.items() if s["n"] >= 10)

for label, (trades, s) in results.items():
    delta  = s["pnl"] - base_pnl
    tag    = ""
    if s["pnl"] == best_pnl and s["n"] >= 10:
        tag = " ◄◄◄ BEST PnL"
    elif s["wr"] == best_wr and s["n"] >= 10:
        tag = " ◄ best WR"
    d_str  = f"  ({delta:+,.0f})" if delta != 0 else "  (baseline)"
    print(f"  {label:45s}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+9,.2f}{d_str}  ${s['final_bal']:>8,.2f}  "
          f"{s['maxdd']:>6.1f}%{tag}")

print("=" * W)

# ── Per-strategy breakdown for every combo ─────────────────────────────────────
print()
print("=" * W)
print("  PER-STRATEGY CONTRIBUTION  (how many trades each strategy fired per combo)")
print("=" * W)
strat_names = ["Volatility Breakout", "Swing Level Break", "Swing Break Retest",
               "Inside Bar Breakout"]
header = f"  {'Combo':45s}  {'Total':>5}  " + "  ".join(f"{n[:8]:>8}" for n in strat_names)
print(header)
print("  " + "-" * 95)

for label, (trades, s) in results.items():
    counts = {n: 0 for n in strat_names}
    for t in trades:
        su = t.get("strategy_used", "")
        if su in counts:
            counts[su] += 1
    row = f"  {label:45s}  {s['n']:>5}  "
    for n in strat_names:
        c    = counts[n]
        pct  = f"{c/s['n']*100:.0f}%" if s["n"] > 0 else "—"
        row += f"  {c:3d}({pct:>4})"
    print(row)


# ── SL distance analysis: current SL vs new tighter SL ─────────────────────────
print()
print("=" * W)
print("  SL DISTANCE ANALYSIS  (avg SL dist in ATR — lower = better R per trade)")
print("=" * W)
print(f"  {'Strategy':30s}  {'Trades':>6}  {'Avg SL (ATR)':>12}  {'WR%':>6}  "
      f"{'AvgR':>6}  {'PnL':>10}")
print("  " + "-" * 78)

# Collect per-strategy SL stats from the "all 4" combo to have all strategies
trades_all4 = results[list(results.keys())[-1]][0]

for strat_name in strat_names:
    ts_ = [t for t in trades_all4 if t.get("strategy_used") == strat_name]
    if not ts_:
        continue
    sl_atrs = []
    for t in ts_:
        sl_d = abs(t["entry"] - t["orig_sl"])
        atr_ = t["atr_at_entry"]
        if atr_ > 0:
            sl_atrs.append(sl_d / atr_)
    wins_ = [t for t in ts_ if t["pnl_usd"] > 0]
    wr_   = len(wins_) / len(ts_) * 100 if ts_ else 0
    avgr_ = sum(t["r_multiple"] for t in ts_) / len(ts_) if ts_ else 0
    pnl_  = sum(t["pnl_usd"] for t in ts_)
    avg_sl_atr = np.mean(sl_atrs) if sl_atrs else 0
    print(f"  {strat_name:30s}  {len(ts_):>6}  {avg_sl_atr:>11.2f}×  "
          f"{wr_:>5.1f}%  {avgr_:>+5.2f}R  ${pnl_:>+9,.2f}")


# ── Best combo: monthly breakdown ──────────────────────────────────────────────
eligible = [(lbl, t, s) for lbl, (t, s) in results.items() if s["n"] >= 15]
if eligible:
    best_lbl, best_trades, best_s = max(eligible, key=lambda x: x[2]["pnl"])

    print()
    print("=" * W)
    print(f"  MONTHLY BREAKDOWN — Best combo: {best_lbl}")
    print("=" * W)
    print(f"  {'Month':>7}  {'#':>3}  {'WR%':>5}  {'AvgR':>6}  {'PnL$':>9}  "
          f"{'Bal$':>9}  {'VB':>3}  {'SL':>3}  {'SBR':>4}  {'IBB':>4}  {'L/S':>6}  Grade")
    print("  " + "-" * 95)

    monthly = defaultdict(list)
    for t in best_trades:
        monthly[str(t["open_time"])[:7]].append(t)

    def _month_grade(s: dict) -> str:
        if s["n"] == 0: return ""
        if s["n"] < 4: return "⚠ few"
        if s["wr"] >= 60 and s["avgr"] >= 1.0: return "🟢 EXCELLENT"
        if s["wr"] >= 50 and s["avgr"] >= 0.5: return "🟢 STRONG"
        if s["wr"] >= 45: return "🟡 GOOD"
        if s["pnl"] < 0:  return "🔴 LOSING"
        return "🟠 WEAK"

    for ym in sorted(monthly.keys()):
        ts_  = monthly[ym]
        s2   = _stats(ts_)
        vb_  = len([t for t in ts_ if t.get("strategy_used") == "Volatility Breakout"])
        sl_  = len([t for t in ts_ if t.get("strategy_used") == "Swing Level Break"])
        sbr_ = len([t for t in ts_ if t.get("strategy_used") == "Swing Break Retest"])
        ibb_ = len([t for t in ts_ if t.get("strategy_used") == "Inside Bar Breakout"])
        lo_  = len([t for t in ts_ if t["direction"] == "long"])
        sh_  = len([t for t in ts_ if t["direction"] == "short"])
        print(f"  {ym:>7}  {s2['n']:>3}  {s2['wr']:>4.1f}%  {s2['avgr']:>+5.2f}R  "
              f"${s2['pnl']:>+8,.2f}  ${s2['final_bal']:>8,.2f}  "
              f"{vb_:>3}  {sl_:>3}  {sbr_:>4}  {ibb_:>4}  "
              f"{lo_}L/{sh_}S  {_month_grade(s2)}")

    s_tot = _stats(best_trades)
    print("  " + "-" * 95)
    print(f"  {'TOTAL':>7}  {s_tot['n']:>3}  {s_tot['wr']:>4.1f}%  {s_tot['avgr']:>+5.2f}R  "
          f"${s_tot['pnl']:>+8,.2f}  ${s_tot['final_bal']:>8,.2f}")


# ── Final recommendation ───────────────────────────────────────────────────────
print()
print("=" * W)
print("  RECOMMENDATION")
print("=" * W)
print(f"  Current baseline (VB + SL)  : $+20,354  |  176 trades  |  50.6% WR")
print()

sorted_results = sorted(
    [(lbl, s) for lbl, (_, s) in results.items() if s["n"] >= 10],
    key=lambda x: x[1]["pnl"], reverse=True
)
for i, (lbl, s) in enumerate(sorted_results[:4]):
    delta = s["pnl"] - base_pnl
    flag  = "  *** BEST ***" if i == 0 else ""
    print(f"  #{i+1} {lbl:45s}  PnL=${s['pnl']:+,.2f}  ({delta:+,.0f} vs baseline)"
          f"  WR={s['wr']}%  AvgR={s['avgr']:+.2f}R  MaxDD={s['maxdd']}%{flag}")

print()
print("  Interpretation guide:")
print("  - Higher PnL + similar/lower MaxDD = clear upgrade")
print("  - Higher WR with fewer trades = quality > quantity trade-off")
print("  - If SBR/IBB adds meaningful PnL above baseline → add to Bot 2")
print()
