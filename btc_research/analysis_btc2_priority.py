"""
btc_research/analysis_btc2_priority.py — Priority order comparison + negative streak analysis.

Answers two questions:
  1. Max consecutive NEGATIVE TRADE DAYS over 2yr
     (a "negative trade day" = any calendar day where net closed PnL < 0)

  2. Does flipping priority (Swing first, VB fallback) improve results?
     Current : VB --> SwingLevelBreakV2
     Flipped : SwingLevelBreakV2 --> VB

Run:
    python btc_research/analysis_btc2_priority.py
"""
from __future__ import annotations

import sys, os
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
from btc_research.strategies.volatility_breakout import VolatilityBreakout
from btc_research.strategies.swing_level_v2 import SwingLevelBreakV2

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

print(f"Loaded {len(df_btc):,} bars  ({df_btc.index[0].date()} to {df_btc.index[-1].date()})\n")

# ── Indicators ────────────────────────────────────────────────────────────────
_c = df_btc["close"].astype(float)
_h = df_btc["high"].astype(float)
_l = df_btc["low"].astype(float)
_tr = pd.concat([_h-_l, (_h-_c.shift(1)).abs(), (_l-_c.shift(1)).abs()], axis=1).max(axis=1)

ATR_ARR = _tr.rolling(14).mean().bfill().values
EMA200  = _c.ewm(span=200, adjust=False).mean().values
TS      = df_btc.index
H_ARR   = _h.values
L_ARR   = _l.values
C_ARR   = _c.values

_sp = 2*14-1
_hd = _h.diff(); _ld = _l.diff()
_pdm = _hd.where((_hd>0)&(_hd>-_ld), 0.0)
_mdm = (-_ld).where((-_ld>0)&(-_ld>_hd), 0.0)
_aw = _tr.ewm(span=_sp,adjust=False).mean()
_pw = _pdm.ewm(span=_sp,adjust=False).mean()
_mw = _mdm.ewm(span=_sp,adjust=False).mean()
_pdi = 100*_pw/_aw; _mdi = 100*_mw/_aw
_dx  = 100*(_pdi-_mdi).abs()/(_pdi+_mdi).replace(0,float("nan"))
ADX_ARR = _dx.ewm(span=_sp,adjust=False).mean().fillna(0).values


# ── Core simulator (accepts strategy list in priority order) ───────────────────
def simulate(strategies: list, label: str = "") -> list[dict]:
    """
    strategies = list of strategy objects tried IN ORDER.
    First one that fires wins. Records which strategy fired and entry_type.
    """
    balance = float(b2cfg.STARTING_BALANCE)
    trades: list[dict] = []
    open_t = None

    for i in range(220, len(df_btc)):
        bar_time = TS[i]
        hr  = bar_time.hour
        bh_ = float(H_ARR[i])
        bl_ = float(L_ARR[i])
        bc_ = float(C_ARR[i])
        atr_now = float(ATR_ARR[i])

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
                    pnl=-risk_u; r_=-1.0; ex="SL"
                elif hit_tp1:
                    pnl=risk_u*open_t["tp1_rr"]; r_=open_t["tp1_rr"]
                    open_t["tp1_hit"]=True; open_t["sl"]=entry_
                    open_t["trail_peak"]=bh_ if long_ else bl_
                    balance+=pnl; open_t["pnl_running"]=round(pnl,2); open_t["r_running"]=r_
                    continue
                elif age >= b2cfg.MAX_HOLD_BARS:
                    pnl=(bc_-entry_ if long_ else entry_-bc_)*open_t["lots"]; r_=pnl/risk_u if risk_u else 0; ex="MAX_HOLD"
                else:
                    continue
            else:
                if long_:
                    if bh_>open_t["trail_peak"]: open_t["trail_peak"]=bh_
                    open_t["sl"]=max(open_t["trail_peak"]-b2cfg.TRAIL_ATR_MULT*atr_now, entry_)
                else:
                    if bl_<open_t["trail_peak"]: open_t["trail_peak"]=bl_
                    open_t["sl"]=min(open_t["trail_peak"]+b2cfg.TRAIL_ATR_MULT*atr_now, entry_)
                hit_sl2=(bl_<=open_t["sl"]) if long_ else (bh_>=open_t["sl"])
                if hit_sl2:
                    dist_run=abs(open_t["sl"]-entry_); r_=dist_run/sl_dist if sl_dist else 0; pnl=risk_u*r_; ex="TRAIL_SL"
                elif age>=b2cfg.MAX_HOLD_BARS:
                    dist_run=abs(bc_-entry_); r_=dist_run/sl_dist if sl_dist else 0; pnl=risk_u*r_; ex="MAX_HOLD"
                else:
                    continue

            balance+=pnl
            open_t["pnl_usd"]       = round(open_t.get("pnl_running",0)+pnl, 2)
            open_t["r_multiple"]    = round(open_t.get("r_running",0)+r_, 2)
            open_t["balance_after"] = round(balance, 2)
            open_t["exit_reason"]   = ex
            open_t["close_date"]    = str(bar_time)[:10]
            trades.append(open_t)
            open_t = None

        if open_t is not None:
            continue

        if hr not in b2cfg.KZ_HOURS:
            continue

        adx_v   = float(ADX_ARR[i])
        ema200v = float(EMA200[i])
        if adx_v < b2cfg.ADX_THRESHOLD:
            continue

        win = df_btc.iloc[max(0,i-220):i+1]

        for direction in ("long", "short"):
            if direction=="long"  and bc_<ema200v: continue
            if direction=="short" and bc_>ema200v: continue

            fired_sig   = None
            fired_strat = None

            for strat in strategies:
                sig = strat.generate_signal(win, bar_time, direction)
                if sig.get("signal"):
                    fired_sig   = sig
                    fired_strat = strat.name
                    break

            if fired_sig is None:
                continue

            sl_d = abs(float(fired_sig["entry"]) - float(fired_sig["sl"]))
            if sl_d <= 0:
                continue

            if adx_v >= b2cfg.ADX_SPLIT_STRONG_MIN:
                risk_pct = b2cfg.RISK_PCT_STRONG
            elif adx_v <= b2cfg.ADX_SPLIT_EARLY_MAX:
                risk_pct = b2cfg.RISK_PCT_EARLY_TREND
            else:
                risk_pct = b2cfg.RISK_PCT_TRANSITION

            ru   = round(balance * risk_pct, 2)
            lots = ru / sl_d
            tp1r = float(fired_sig.get("tp1_rr", b2cfg.TP1_RR))
            tp2r = float(fired_sig.get("tp2_rr", b2cfg.TP2_RR))
            tp1  = float(fired_sig["entry"]) + tp1r*sl_d if direction=="long" else float(fired_sig["entry"]) - tp1r*sl_d
            tp2  = float(fired_sig["entry"]) + tp2r*sl_d if direction=="long" else float(fired_sig["entry"]) - tp2r*sl_d

            open_t = {
                "open_time":     bar_time,
                "open_date":     str(bar_time)[:10],
                "close_date":    None,
                "open_idx":      i,
                "direction":     direction,
                "entry":         float(fired_sig["entry"]),
                "sl":            float(fired_sig["sl"]),
                "orig_sl":       float(fired_sig["sl"]),
                "tp1":           tp1, "tp2": tp2,
                "tp1_rr":        tp1r, "tp2_rr": tp2r,
                "lots":          lots, "risk_usd": ru,
                "risk_pct_used": risk_pct,
                "strategy_used": fired_strat,
                "entry_type":    fired_sig.get("entry_type", ""),
                "trail_peak":    0.0, "tp1_hit": False,
                "pnl_running":   0.0, "r_running": 0.0,
                "pnl_usd":       0.0,
                "adx_at_entry":  round(adx_v, 1),
                "sl_atr_mult":   round(sl_d/atr_now, 2) if atr_now>0 else 0,
            }
            break

    return trades


def _stats(trades):
    if not trades:
        return {"n":0,"wr":0.0,"avgr":0.0,"pnl":0.0,"final_bal":b2cfg.STARTING_BALANCE,"maxdd":0.0,"pf":0.0}
    wins   = [t for t in trades if t["pnl_usd"]>0]
    losses = [t for t in trades if t["pnl_usd"]<0]
    wr     = len(wins)/len(trades)*100
    avgr   = sum(t["r_multiple"] for t in trades)/len(trades)
    pnl    = sum(t["pnl_usd"] for t in trades)
    gross_w = sum(t["pnl_usd"] for t in wins)
    gross_l = abs(sum(t["pnl_usd"] for t in losses))
    pf     = gross_w/gross_l if gross_l>0 else float("inf")
    bals   = [b2cfg.STARTING_BALANCE]+[t["balance_after"] for t in trades]
    peak   = b2cfg.STARTING_BALANCE; maxdd=0.0
    for b in bals:
        if b>peak: peak=b
        dd=(peak-b)/peak*100
        if dd>maxdd: maxdd=dd
    return {"n":len(trades),"wr":round(wr,1),"avgr":round(avgr,2),"pnl":round(pnl,2),
            "final_bal":round(bals[-1],2),"maxdd":round(maxdd,1),"pf":round(pf,2)}


def neg_streak_analysis(trades: list[dict], label: str) -> None:
    """
    Compute and print consecutive negative trade day analysis.

    A 'trade day' = any calendar day where at least one trade CLOSED.
    Net PnL for that day = sum of all trades that closed on that day.
    Consecutive negative days = longest run of days (with closed trades)
    where net PnL was negative back-to-back.
    """
    W = 100
    print("=" * W)
    print(f"  CONSECUTIVE NEGATIVE TRADE DAYS — {label}")
    print("=" * W)

    # Group closed PnL by close date
    daily_pnl: dict[str, float] = defaultdict(float)
    daily_trades: dict[str, int] = defaultdict(int)
    for t in trades:
        cd = t.get("close_date") or str(t["open_time"])[:10]
        daily_pnl[cd]    += t["pnl_usd"]
        daily_trades[cd] += 1

    sorted_days = sorted(daily_pnl.keys())
    n_days      = len(sorted_days)

    neg_days      = [d for d in sorted_days if daily_pnl[d] < 0]
    pos_days      = [d for d in sorted_days if daily_pnl[d] >= 0]

    # Consecutive negative streak
    max_consec_neg = 0
    cur_consec_neg = 0
    worst_streak_start = None
    worst_streak_end   = None
    cur_start = None

    for d in sorted_days:
        if daily_pnl[d] < 0:
            if cur_consec_neg == 0:
                cur_start = d
            cur_consec_neg += 1
            if cur_consec_neg > max_consec_neg:
                max_consec_neg    = cur_consec_neg
                worst_streak_start = cur_start
                worst_streak_end   = d
        else:
            cur_consec_neg = 0
            cur_start = None

    # Worst single day
    worst_day     = min(sorted_days, key=lambda d: daily_pnl[d])
    worst_day_pnl = daily_pnl[worst_day]
    best_day      = max(sorted_days, key=lambda d: daily_pnl[d])
    best_day_pnl  = daily_pnl[best_day]

    print(f"  Total trade days    : {n_days}")
    print(f"  Positive days       : {len(pos_days)}  ({len(pos_days)/n_days*100:.0f}%)")
    print(f"  Negative days       : {len(neg_days)}  ({len(neg_days)/n_days*100:.0f}%)")
    print()
    print(f"  Max consecutive NEGATIVE days : {max_consec_neg}", end="")
    if worst_streak_start:
        print(f"  ({worst_streak_start} to {worst_streak_end})")
    else:
        print()
    print(f"  Max consecutive POSITIVE days : ", end="")

    # Consecutive positive streak
    max_consec_pos = 0
    cur_consec_pos = 0
    for d in sorted_days:
        if daily_pnl[d] >= 0:
            cur_consec_pos += 1
            max_consec_pos = max(max_consec_pos, cur_consec_pos)
        else:
            cur_consec_pos = 0
    print(max_consec_pos)

    print()
    print(f"  Worst single day    : {worst_day}  PnL=${worst_day_pnl:+,.2f}  ({daily_trades[worst_day]} trade(s))")
    print(f"  Best single day     : {best_day}   PnL=${best_day_pnl:+,.2f}  ({daily_trades[best_day]} trade(s))")
    print()

    # Show all negative days (if few enough)
    if len(neg_days) <= 30:
        print(f"  All negative trade days ({len(neg_days)}):")
        for d in neg_days:
            n_t = daily_trades[d]
            print(f"    {d}  PnL=${daily_pnl[d]:+,.2f}  ({n_t} trade(s))")
    else:
        print(f"  5 worst trade days:")
        worst5 = sorted(sorted_days, key=lambda d: daily_pnl[d])[:5]
        for d in worst5:
            print(f"    {d}  PnL=${daily_pnl[d]:+,.2f}  ({daily_trades[d]} trade(s))")
    print()


# ── Build strategies ──────────────────────────────────────────────────────────
vb_strat  = VolatilityBreakout(
    atr_multiplier = b2cfg.VB_ATR_MULTIPLIER,
    close_zone     = b2cfg.VB_CLOSE_ZONE,
)
slv2_strat = SwingLevelBreakV2(
    entry_mode  = b2cfg.SWING_ENTRY_MODE,   # "both"
    max_sl_atr  = b2cfg.SWING_MAX_SL_ATR,   # 2.0
)

# ── Run both priority orders ───────────────────────────────────────────────────
print("Running: CURRENT order  (VB first, then SwingV2)...")
trades_vb_first = simulate([vb_strat, slv2_strat], label="VB -> SwingV2")

print("Running: FLIPPED order  (SwingV2 first, then VB)...")
trades_sl_first = simulate([slv2_strat, vb_strat], label="SwingV2 -> VB")

s_vb  = _stats(trades_vb_first)
s_sl  = _stats(trades_sl_first)

W = 100
SEP = "=" * W

# ── Section 1: Head-to-head comparison ────────────────────────────────────────
print()
print(SEP)
print("  SECTION 1: PRIORITY ORDER COMPARISON")
print(SEP)
print(f"  {'Order':35s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>12}  {'MaxDD%':>7}  {'PF':>5}")
print("  " + "-" * 80)
for label, s in [
    ("CURRENT: VB first -> SwingV2", s_vb),
    ("FLIPPED: SwingV2 first -> VB", s_sl),
]:
    print(f"  {label:35s}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+11,.2f}  {s['maxdd']:>6.1f}%  {s['pf']:>4.2f}")
print()

# ── Section 2: Per-strategy breakdown for each order ─────────────────────────
print(SEP)
print("  SECTION 2: STRATEGY BREAKDOWN — CURRENT (VB first)")
print(SEP)
print(f"  {'Source':30s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>11}")
print("  " + "-" * 58)
for name, tlist in [
    ("Volatility Breakout",        [t for t in trades_vb_first if "Volatility" in t.get("strategy_used","")]),
    ("Swing Level Break v2 (all)", [t for t in trades_vb_first if "Swing" in t.get("strategy_used","")]),
    ("  -> Break entries",         [t for t in trades_vb_first if t.get("entry_type")=="break"]),
    ("  -> Retest entries",        [t for t in trades_vb_first if t.get("entry_type")=="retest"]),
]:
    if not tlist:
        print(f"  {name:30s}  no trades"); continue
    s2 = _stats(tlist)
    print(f"  {name:30s}  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  ${s2['pnl']:>+10,.2f}")

print()
print(SEP)
print("  SECTION 2b: STRATEGY BREAKDOWN — FLIPPED (SwingV2 first)")
print(SEP)
print(f"  {'Source':30s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>11}")
print("  " + "-" * 58)
for name, tlist in [
    ("Volatility Breakout",        [t for t in trades_sl_first if "Volatility" in t.get("strategy_used","")]),
    ("Swing Level Break v2 (all)", [t for t in trades_sl_first if "Swing" in t.get("strategy_used","")]),
    ("  -> Break entries",         [t for t in trades_sl_first if t.get("entry_type")=="break"]),
    ("  -> Retest entries",        [t for t in trades_sl_first if t.get("entry_type")=="retest"]),
]:
    if not tlist:
        print(f"  {name:30s}  no trades"); continue
    s2 = _stats(tlist)
    print(f"  {name:30s}  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  ${s2['pnl']:>+10,.2f}")
print()

# ── Section 3: Consecutive negative trade day analysis ────────────────────────
neg_streak_analysis(trades_vb_first, "CURRENT (VB first)")
neg_streak_analysis(trades_sl_first, "FLIPPED (SwingV2 first)")

# ── Section 4: Monthly comparison ─────────────────────────────────────────────
print(SEP)
print("  SECTION 4: MONTHLY PnL — CURRENT vs FLIPPED")
print(SEP)
print(f"  {'Month':>7}  {'CURRENT $':>12}  {'FLIPPED $':>12}  {'Diff $':>10}  Winner")
print("  " + "-" * 58)

monthly_vb = defaultdict(list)
monthly_sl = defaultdict(list)
for t in trades_vb_first:
    monthly_vb[str(t["open_time"])[:7]].append(t)
for t in trades_sl_first:
    monthly_sl[str(t["open_time"])[:7]].append(t)

all_months = sorted(set(list(monthly_vb.keys()) + list(monthly_sl.keys())))
total_diff = 0
for ym in all_months:
    pnl_vb = _stats(monthly_vb[ym])["pnl"] if monthly_vb[ym] else 0
    pnl_sl = _stats(monthly_sl[ym])["pnl"] if monthly_sl[ym] else 0
    diff   = pnl_sl - pnl_vb
    total_diff += diff
    winner = "SWING-FIRST" if pnl_sl > pnl_vb else ("TIE" if pnl_sl == pnl_vb else "VB-FIRST")
    print(f"  {ym:>7}  ${pnl_vb:>+11,.2f}  ${pnl_sl:>+11,.2f}  ${diff:>+9,.2f}  {winner}")

print("  " + "-" * 58)
print(f"  {'TOTAL':>7}  ${s_vb['pnl']:>+11,.2f}  ${s_sl['pnl']:>+11,.2f}  ${total_diff:>+9,.2f}")
print()

# ── Final recommendation ───────────────────────────────────────────────────────
print(SEP)
print("  VERDICT")
print(SEP)
winner_pnl = "SWING-FIRST" if s_sl["pnl"] > s_vb["pnl"] else "VB-FIRST"
winner_wr  = "SWING-FIRST" if s_sl["wr"]  > s_vb["wr"]  else "VB-FIRST"
winner_dd  = "VB-FIRST"    if s_vb["maxdd"] < s_sl["maxdd"] else "SWING-FIRST"

print(f"  PnL winner  : {winner_pnl}  (${s_vb['pnl']:+,.0f} vs ${s_sl['pnl']:+,.0f})")
print(f"  WR winner   : {winner_wr}  ({s_vb['wr']}% vs {s_sl['wr']}%)")
print(f"  MaxDD winner: {winner_dd}  ({s_vb['maxdd']}% vs {s_sl['maxdd']}%)")
print()
pnl_diff_pct = (s_sl["pnl"] - s_vb["pnl"]) / s_vb["pnl"] * 100 if s_vb["pnl"] > 0 else 0
if s_sl["pnl"] > s_vb["pnl"]:
    print(f"  RECOMMENDATION: Switch to SWING-FIRST order")
    print(f"  Gain: +${s_sl['pnl'] - s_vb['pnl']:,.0f} (+{pnl_diff_pct:.1f}%) more PnL")
    print(f"  Update VBSwingStrategy: put SwingLevelBreakV2 before VolatilityBreakout")
else:
    print(f"  RECOMMENDATION: Keep CURRENT order (VB first)")
    print(f"  VB-first leads by ${s_vb['pnl'] - s_sl['pnl']:,.0f} in PnL")
print(SEP)
