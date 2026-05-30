"""
btc_research/analysis_btc2_swing_v2_temporal.py
Temporal breakdown (monthly / quarterly / half-year) for every SwingLevelBreak
v2 mode so we can make a final, informed decision.

For each mode we show:
  • Half-year table   (H1/H2 per year)
  • Quarterly table   (Q1-Q4 per year)
  • Monthly table     (full 24-month grid)
  • Consistency block (profitable periods, streaks, Sharpe, Profit-Factor)

Modes tested (all with VB first, then the SL variant, EMA200+ADX-split+[1,2,3,8] UTC):
  0  v1 original           SL = prior swing structure  (4.37×ATR)
  1  break_capped 2×ATR    first break, SL ≤ 2×ATR
  2  break_capped 1.5×ATR  first break, SL ≤ 1.5×ATR
  3  retest only            retest entry, SL = bar extreme
  4  retest_preferred 2×    retest→break_capped(2×)
  5  retest_preferred 1.5×  retest→break_capped(1.5×)
  6  both 2×ATR             retest OR break_capped(2×)

Run:
    python btc_research/analysis_btc2_swing_v2_temporal.py
"""
from __future__ import annotations

import sys, os
from pathlib import Path
from collections import defaultdict
import datetime as _dt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(str(_ROOT))

import pandas as pd
import numpy as np
import btc_research.settings as cfg
from btc_research.data.fetcher import fetch_all
from btc_research.btc_bot_2 import settings as b2cfg
from btc_research.strategies.volatility_breakout import VolatilityBreakout
from btc_research.strategies.swing_level         import SwingLevelBreak
from btc_research.strategies.swing_level_v2      import SwingLevelBreakV2

try:
    from dateutil.relativedelta import relativedelta as _rd
except ImportError:
    class _rd:
        def __init__(self, months=0): self.months = months
        def __radd__(self, other):
            m = other.month - 1 + self.months
            return other.replace(year=other.year + m//12, month=m%12+1)

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

print(f"Loaded {len(df_btc):,} bars  ({df_btc.index[0].date()} → {df_btc.index[-1].date()})\n")

# ── Indicators ─────────────────────────────────────────────────────────────────
_c = df_btc["close"].astype(float)
_h = df_btc["high"].astype(float)
_l = df_btc["low"].astype(float)
_tr = pd.concat([_h-_l,(_h-_c.shift(1)).abs(),(_l-_c.shift(1)).abs()],axis=1).max(axis=1)
ATR_ARR = _tr.rolling(14).mean().bfill().values
EMA200  = _c.ewm(span=200,adjust=False).mean().values
TS=df_btc.index; H_ARR=_h.values; L_ARR=_l.values; C_ARR=_c.values

def _calc_adx(p=14):
    sp=2*p-1; hd=_h.diff(); ld=_l.diff()
    pdm=hd.where((hd>0)&(hd>-ld),0.0); mdm=(-ld).where((-ld>0)&(-ld>hd),0.0)
    aw=_tr.ewm(span=sp,adjust=False).mean()
    pdi=100*pdm.ewm(span=sp,adjust=False).mean()/aw
    mdi=100*mdm.ewm(span=sp,adjust=False).mean()/aw
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,float("nan"))
    return dx.ewm(span=sp,adjust=False).mean().fillna(0).values

ADX_ARR = _calc_adx()

# ── Simulator ──────────────────────────────────────────────────────────────────
def simulate(strat_list) -> list[dict]:
    balance = float(b2cfg.STARTING_BALANCE)
    trades: list[dict] = []
    open_t = None

    for i in range(220, len(df_btc)):
        bar_time=TS[i]; hr=bar_time.hour
        bh_=float(H_ARR[i]); bl_=float(L_ARR[i]); bc_=float(C_ARR[i])

        if open_t is not None:
            long_=open_t["direction"]=="long"; entry_=open_t["entry"]
            sl_dist=abs(entry_-open_t["orig_sl"]); risk_u=open_t["risk_usd"]
            age=i-open_t["open_idx"]

            if not open_t.get("tp1_hit"):
                hit_sl =(bl_<=open_t["sl"]) if long_ else (bh_>=open_t["sl"])
                hit_tp1=(bh_>=open_t["tp1"]) if long_ else (bl_<=open_t["tp1"])
                if hit_sl:
                    pnl=-risk_u; r_=-1.0; ex="SL"
                elif hit_tp1:
                    pnl=risk_u*open_t["tp1_rr"]; r_=open_t["tp1_rr"]
                    open_t["tp1_hit"]=True; open_t["sl"]=entry_
                    open_t["trail_peak"]=bh_ if long_ else bl_
                    balance+=pnl; open_t["pnl_running"]=round(pnl,2); open_t["r_running"]=r_
                    continue
                elif age>=b2cfg.MAX_HOLD_BARS:
                    pnl=(bc_-entry_ if long_ else entry_-bc_)*open_t["lots"]
                    r_=pnl/risk_u if risk_u else 0; ex="MAX_HOLD"
                else: continue
            else:
                atr_n=float(ATR_ARR[min(i,len(ATR_ARR)-1)])
                if long_:
                    if bh_>open_t["trail_peak"]: open_t["trail_peak"]=bh_
                    open_t["sl"]=max(open_t["trail_peak"]-b2cfg.TRAIL_ATR_MULT*atr_n,entry_)
                else:
                    if bl_<open_t["trail_peak"]: open_t["trail_peak"]=bl_
                    open_t["sl"]=min(open_t["trail_peak"]+b2cfg.TRAIL_ATR_MULT*atr_n,entry_)
                new_sl=open_t["sl"]
                hit_sl2=(bl_<=new_sl) if long_ else (bh_>=new_sl)
                if hit_sl2:
                    dist_run=abs(new_sl-entry_); r_=dist_run/sl_dist if sl_dist else 0
                    pnl=risk_u*r_; ex="TRAIL_SL"
                elif age>=b2cfg.MAX_HOLD_BARS:
                    dist_run=abs(bc_-entry_); r_=dist_run/sl_dist if sl_dist else 0
                    pnl=risk_u*r_; ex="MAX_HOLD"
                else: continue

            balance+=pnl
            open_t["pnl_usd"]      =round(open_t.get("pnl_running",0)+pnl,2)
            open_t["r_multiple"]   =round(open_t.get("r_running",0)+r_,2)
            open_t["balance_after"]=round(balance,2); open_t["exit_reason"]=ex
            trades.append(open_t); open_t=None

        if open_t is not None: continue
        if hr not in b2cfg.KZ_HOURS: continue
        adx_now=float(ADX_ARR[i]); ema200_now=float(EMA200[i])
        if adx_now<b2cfg.ADX_THRESHOLD: continue
        win=df_btc.iloc[max(0,i-220):i+1]

        for direction in ("long","short"):
            if direction=="long"  and bc_<ema200_now: continue
            if direction=="short" and bc_>ema200_now: continue
            sig=None
            for strat in strat_list:
                r=strat.generate_signal(win,bar_time,direction)
                if r.get("signal"):
                    sig=r; sig["strategy_used"]=strat.name
                    sig["tp1_rr"]=r.get("tp1_rr",b2cfg.TP1_RR)
                    sig["tp2_rr"]=r.get("tp2_rr",b2cfg.TP2_RR)
                    break
            if sig is None: continue
            sl_d=abs(float(sig["entry"])-float(sig["sl"]))
            if sl_d<=0: continue
            if adx_now>=b2cfg.ADX_SPLIT_STRONG_MIN: rp=b2cfg.RISK_PCT_STRONG
            elif adx_now<=b2cfg.ADX_SPLIT_EARLY_MAX: rp=b2cfg.RISK_PCT_EARLY_TREND
            else: rp=b2cfg.RISK_PCT_TRANSITION
            ru=round(balance*rp,2); lots=ru/sl_d
            tp1r=sig["tp1_rr"]; tp2r=sig["tp2_rr"]
            tp1=(float(sig["entry"])+tp1r*sl_d if direction=="long" else float(sig["entry"])-tp1r*sl_d)
            tp2=(float(sig["entry"])+tp2r*sl_d if direction=="long" else float(sig["entry"])-tp2r*sl_d)
            open_t={
                "open_time":bar_time,"open_idx":i,"direction":direction,
                "entry":float(sig["entry"]),"sl":float(sig["sl"]),"orig_sl":float(sig["sl"]),
                "tp1":tp1,"tp2":tp2,"tp1_rr":tp1r,"tp2_rr":tp2r,
                "lots":lots,"risk_usd":ru,"risk_pct":rp,
                "strategy_used":sig.get("strategy_used",""),
                "entry_type":sig.get("entry_type","break"),
                "trail_peak":0.0,"tp1_hit":False,
                "pnl_running":0.0,"r_running":0.0,"pnl_usd":0.0,
                "adx_at_entry":round(adx_now,1),
                "atr_at_entry":round(float(ATR_ARR[i]),0),
                "above_ema200":bc_>ema200_now,"hour":hr,
                "exit_reason":"","balance_after":0.0,"r_multiple":0.0,
            }
            break
    return trades


# ── Stats helpers ─────────────────────────────────────────────────────────────
QTR_MAP = {1:"Q1",2:"Q1",3:"Q1",4:"Q2",5:"Q2",6:"Q2",
           7:"Q3",8:"Q3",9:"Q3",10:"Q4",11:"Q4",12:"Q4"}

def _stats(trades, start_bal=None):
    sb = start_bal if start_bal is not None else b2cfg.STARTING_BALANCE
    if not trades:
        return {"n":0,"wr":0.0,"avgr":0.0,"pnl":0.0,
                "end_bal":sb,"maxdd":0.0,"profitable":False}
    wins=[t for t in trades if t["pnl_usd"]>0]
    bals=[sb]+[t["balance_after"] for t in trades]
    peak=sb; maxdd=0.0
    for b in bals:
        if b>peak: peak=b
        dd=(peak-b)/peak*100
        if dd>maxdd: maxdd=dd
    pnl=round(sum(t["pnl_usd"] for t in trades),2)
    return {
        "n":len(trades),
        "wr":round(len(wins)/len(trades)*100,1),
        "avgr":round(sum(t["r_multiple"] for t in trades)/len(trades),2),
        "pnl":pnl,
        "start_bal":round(sb,2),
        "end_bal":round(bals[-1],2),
        "maxdd":round(maxdd,1),
        "profitable": pnl > 0,
    }

def _grade(s):
    if s["n"]==0: return "  —"
    if s["n"]<4:  return "  ⚠ few"
    if s["wr"]>=60 and s["avgr"]>=1.0: return "  🟢 EXCEL"
    if s["wr"]>=50 and s["avgr"]>=0.5: return "  🟢 STRONG"
    if s["wr"]>=45 and s["avgr"]>=0.0: return "  🟡 GOOD"
    if s["pnl"]<0: return "  🔴 LOSS"
    return "  🟠 WEAK"

def _all_months(trades):
    if not trades: return []
    start=_dt.datetime.strptime(str(trades[0]["open_time"])[:7],"%Y-%m")
    end  =_dt.datetime.strptime(str(trades[-1]["open_time"])[:7],"%Y-%m")
    months=[]; cur=start
    while cur<=end:
        months.append(cur.strftime("%Y-%m"))
        try: cur=cur+_rd(months=1)
        except: break
    return months


# ── Run all modes ─────────────────────────────────────────────────────────────
VB = VolatilityBreakout(atr_multiplier=b2cfg.VB_ATR_MULTIPLIER, close_zone=b2cfg.VB_CLOSE_ZONE)

MODES = [
    ("0  v1 original",             [VB, SwingLevelBreak()]),
    ("1  break_cap 2×ATR",         [VB, SwingLevelBreakV2(entry_mode="break_capped",      max_sl_atr=2.0)]),
    ("2  break_cap 1.5×ATR",       [VB, SwingLevelBreakV2(entry_mode="break_capped",      max_sl_atr=1.5)]),
    ("3  retest only",             [VB, SwingLevelBreakV2(entry_mode="retest")]),
    ("4  retest_pref 2×ATR",       [VB, SwingLevelBreakV2(entry_mode="retest_preferred",  max_sl_atr=2.0)]),
    ("5  retest_pref 1.5×ATR",     [VB, SwingLevelBreakV2(entry_mode="retest_preferred",  max_sl_atr=1.5)]),
    ("6  both 2×ATR",              [VB, SwingLevelBreakV2(entry_mode="both",               max_sl_atr=2.0)]),
]

print("Running all 7 modes (this takes 3-5 min)...")
all_trades = {}
for label, strats in MODES:
    t = simulate(strats)
    all_trades[label] = t
    s = _stats(t)
    print(f"  {label:28s}  n={s['n']:3d}  WR={s['wr']:5.1f}%  "
          f"AvgR={s['avgr']:+.2f}R  PnL=${s['pnl']:+,.2f}  MaxDD={s['maxdd']:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
W = 120

def _print_section(title):
    print(); print("="*W); print(f"  {title}"); print("="*W)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HALF-YEAR COMPARISON ACROSS ALL MODES
# ══════════════════════════════════════════════════════════════════════════════
_print_section("SECTION 1 — YEAR-HALF BREAKDOWN (all modes side-by-side)")

# Collect half keys
all_halves = sorted(set(
    ("H1" if t["open_time"].month<=6 else "H2") + " " + str(t["open_time"].year)
    for trades in all_trades.values() for t in trades
), key=lambda x: (int(x.split()[1]), 0 if x.startswith("H1") else 1))

# Header
h_labels = [lbl[:18] for lbl in [m[0] for m in MODES]]
print(f"\n  {'Half':>8} │ " + " │ ".join(f"{'PnL':>9}  {'WR':>5}" for _ in h_labels))
print(f"  {'':>8} │ " + " │ ".join(f"{lb:>15}" for lb in h_labels))
print("  " + "─"*10 + "┼" + ("─"*17+"┼")*(len(MODES)-1) + "─"*17)

for half in all_halves:
    row = f"  {half:>8} │ "
    cells = []
    for label, _ in MODES:
        trades = all_trades[label]
        ts_ = [t for t in trades
               if (("H1" if t["open_time"].month<=6 else "H2")
                   + " " + str(t["open_time"].year)) == half]
        s = _stats(ts_)
        if s["n"]==0:
            cells.append(f"{'—':>9}  {'—':>5}")
        else:
            wr_str = f"{s['wr']:4.1f}%"
            cells.append(f"${s['pnl']:>+8,.0f}  {wr_str:>5}")
    row += " │ ".join(cells)
    print(row)

# Totals row
print("  " + "─"*10 + "┼" + ("─"*17+"┼")*(len(MODES)-1) + "─"*17)
row = f"  {'TOTAL':>8} │ "
cells = []
for label, _ in MODES:
    s = _stats(all_trades[label])
    cells.append(f"${s['pnl']:>+8,.0f}  {s['wr']:>4.1f}%")
row += " │ ".join(cells)
print(row)

# Profitable halves summary
print()
print(f"  {'Metric':28s} │ " + " │ ".join(f"  {lbl[:15]:15s}" for lbl, _ in MODES))
print("  " + "─"*30 + "┼" + ("─"*18+"┼")*(len(MODES)-1) + "─"*18)

for metric_name, fn in [
    ("Profitable halves",  lambda trades: f"{sum(1 for h in all_halves if _stats([t for t in trades if (('H1' if t['open_time'].month<=6 else 'H2')+' '+str(t['open_time'].year))==h and t]).get('profitable',False))}/{sum(1 for h in all_halves if any((('H1' if t['open_time'].month<=6 else 'H2')+' '+str(t['open_time'].year))==h for t in trades))}"),
    ("Max DD %",           lambda trades: f"{_stats(trades)['maxdd']:.1f}%"),
    ("Total trades",       lambda trades: str(_stats(trades)['n'])),
]:
    row = f"  {metric_name:28s} │ "
    cells = [f"  {fn(all_trades[label]):>15s}" for label, _ in MODES]
    row += " │ ".join(cells)
    print(row)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — QUARTERLY COMPARISON ACROSS ALL MODES
# ══════════════════════════════════════════════════════════════════════════════
_print_section("SECTION 2 — QUARTERLY BREAKDOWN (all modes side-by-side)")

all_qtrs = sorted(set(
    QTR_MAP[t["open_time"].month] + " " + str(t["open_time"].year)
    for trades in all_trades.values() for t in trades
), key=lambda x: (int(x.split()[1]), x.split()[0]))

print(f"\n  {'Quarter':>8} │ " + " │ ".join(f"{'PnL':>9}  {'WR':>5}" for _ in MODES))
print(f"  {'':>8} │ " + " │ ".join(f"{lb[:15]:>15}" for lb, _ in MODES))
print("  " + "─"*10 + "┼" + ("─"*17+"┼")*(len(MODES)-1) + "─"*17)

for qtr in all_qtrs:
    row = f"  {qtr:>8} │ "
    cells = []
    for label, _ in MODES:
        trades = all_trades[label]
        ts_ = [t for t in trades
               if QTR_MAP[t["open_time"].month]+" "+str(t["open_time"].year) == qtr]
        s = _stats(ts_)
        if s["n"]==0:
            cells.append(f"{'—':>9}  {'—':>5}")
        else:
            sign = "✅" if s["profitable"] else "❌"
            cells.append(f"${s['pnl']:>+8,.0f}  {s['wr']:>4.1f}%")
    row += " │ ".join(cells)
    # flag losing quarters
    any_loss = any(_stats([t for t in all_trades[label]
                           if QTR_MAP[t["open_time"].month]+" "+str(t["open_time"].year)==qtr])["pnl"]<0
                   for label,_ in MODES)
    print(row + ("  ← loss" if any_loss else ""))

print("  " + "─"*10 + "┼" + ("─"*17+"┼")*(len(MODES)-1) + "─"*17)
row = f"  {'TOTAL':>8} │ "
cells = [f"${_stats(all_trades[label])['pnl']:>+8,.0f}  {_stats(all_trades[label])['wr']:>4.1f}%" for label,_ in MODES]
row += " │ ".join(cells)
print(row)

# Profitable quarters
print()
print(f"  Profitable quarters out of {len(all_qtrs)}:")
for label, _ in MODES:
    prof = sum(1 for q in all_qtrs
               if _stats([t for t in all_trades[label]
                          if QTR_MAP[t["open_time"].month]+" "+str(t["open_time"].year)==q])["profitable"])
    active = sum(1 for q in all_qtrs
                 if any(QTR_MAP[t["open_time"].month]+" "+str(t["open_time"].year)==q
                        for t in all_trades[label]))
    pct = prof/active*100 if active else 0
    print(f"    {label:28s}  {prof}/{active}  ({pct:.0f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FULL MONTHLY GRID (each mode gets its own table)
# ══════════════════════════════════════════════════════════════════════════════
_print_section("SECTION 3 — MONTHLY BREAKDOWN (one table per mode)")

all_month_keys = _all_months(sorted(
    [t for trades in all_trades.values() for t in trades],
    key=lambda t: t["open_time"]
))

for label, _ in MODES:
    trades = all_trades[label]
    s_tot  = _stats(trades)

    print()
    print(f"  ── {label}  │  Total: n={s_tot['n']}  WR={s_tot['wr']}%  "
          f"AvgR={s_tot['avgr']:+.2f}R  PnL=${s_tot['pnl']:+,.2f}  "
          f"MaxDD={s_tot['maxdd']}% ──")
    print(f"  {'Month':>7}  {'#':>3}  {'WR':>5}  {'AvgR':>6}  "
          f"{'PnL$':>9}  {'Bal$':>9}  {'MaxDD':>6}  Grade")
    print("  " + "─"*70)

    monthly = defaultdict(list)
    for t in trades:
        monthly[str(t["open_time"])[:7]].append(t)

    running_bal = b2cfg.STARTING_BALANCE
    prev_qtr = ""
    for ym in all_month_keys:
        ts_  = monthly.get(ym, [])
        s    = _stats(ts_, running_bal)
        mo   = int(ym[5:7])
        this_qtr = QTR_MAP[mo] + " " + ym[:4]
        if this_qtr != prev_qtr:
            print(f"  {'─'*70}")
            prev_qtr = this_qtr

        if ts_:
            pnl_str = f"${s['pnl']:>+8,.0f}"
            bal_str = f"${s['end_bal']:>8,.0f}"
            dd_str  = f"{s['maxdd']:>5.1f}%"
            wr_str  = f"{s['wr']:>4.1f}%"
            avgr    = f"{s['avgr']:>+5.2f}R"
            grade   = _grade(s)
            print(f"  {ym:>7}  {s['n']:>3}  {wr_str}  {avgr}  "
                  f"{pnl_str}  {bal_str}  {dd_str}{grade}")
            running_bal = s["end_bal"]
        else:
            print(f"  {ym:>7}  {'0':>3}  {'—':>5}  {'—':>6}  "
                  f"{'—':>9}  ${running_bal:>8,.0f}  {'—':>6}")

    print(f"  {'─'*70}")
    print(f"  {'TOTAL':>7}  {s_tot['n']:>3}  {s_tot['wr']:>4.1f}%  "
          f"{s_tot['avgr']:>+5.2f}R  ${s_tot['pnl']:>+8,.2f}  "
          f"${s_tot['end_bal']:>8,.2f}  {s_tot['maxdd']:>5.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CONSISTENCY SCORECARD (all modes)
# ══════════════════════════════════════════════════════════════════════════════
_print_section("SECTION 4 — CONSISTENCY SCORECARD")

print(f"\n  {'Metric':35s} │ " + " │ ".join(f"{lbl[:15]:>15}" for lbl, _ in MODES))
print("  " + "─"*37 + "┼" + ("─"*17+"┼")*(len(MODES)-1) + "─"*17)

def _consistency(trades):
    if not trades: return {}
    monthly = defaultdict(list)
    for t in trades:
        monthly[str(t["open_time"])[:7]].append(t)
    months = sorted(monthly.keys())
    all_m = [sum(t["pnl_usd"] for t in monthly[m]) for m in months]

    # streaks
    mw=ml=cmw=cml=0
    for p in all_m:
        if p>0: cmw+=1; cml=0
        else: cml+=1; cmw=0
        mw=max(mw,cmw); ml=max(ml,cml)

    wins_   = [t for t in trades if t["pnl_usd"]>0]
    losses_ = [t for t in trades if t["pnl_usd"]<=0]
    gw = sum(t["pnl_usd"] for t in wins_)
    gl = abs(sum(t["pnl_usd"] for t in losses_))
    pf = gw/gl if gl>0 else float("inf")

    avg_mo = np.mean(all_m); std_mo = np.std(all_m)
    sharpe = avg_mo/std_mo if std_mo>0 else 0

    s = _stats(trades)
    ann_ret_pct = (s["pnl"]/b2cfg.STARTING_BALANCE)*100/2
    calmar = ann_ret_pct/s["maxdd"] if s["maxdd"]>0 else float("inf")

    # trade streaks
    tw=tl=ctw=ctl=0
    for t in trades:
        if t["pnl_usd"]>0: ctw+=1; ctl=0
        else: ctl+=1; ctw=0
        tw=max(tw,ctw); tl=max(tl,ctl)

    prof_m = sum(1 for p in all_m if p>0)
    prof_q = sum(1 for q in all_qtrs if _stats([t for t in trades if QTR_MAP[t["open_time"].month]+" "+str(t["open_time"].year)==q])["profitable"])
    act_q  = sum(1 for q in all_qtrs if any(QTR_MAP[t["open_time"].month]+" "+str(t["open_time"].year)==q for t in trades))

    return {
        "n_mo": len(months), "prof_mo": prof_m,
        "prof_q": prof_q, "act_q": act_q,
        "max_win_mo": mw, "max_loss_mo": ml,
        "max_win_tr": tw, "max_loss_tr": tl,
        "pf": pf, "sharpe": sharpe, "calmar": calmar,
        "pnl": s["pnl"], "maxdd": s["maxdd"], "wr": s["wr"],
    }

consy = {label: _consistency(all_trades[label]) for label, _ in MODES}

metrics = [
    ("Total PnL",            lambda c: f"${c['pnl']:>+9,.0f}"),
    ("Win Rate",             lambda c: f"{c['wr']:>5.1f}%"),
    ("Max Drawdown",         lambda c: f"{c['maxdd']:>5.1f}%"),
    ("Profitable months",    lambda c: f"{c['prof_mo']}/{c['n_mo']} ({c['prof_mo']/c['n_mo']*100:.0f}%)"),
    ("Profitable quarters",  lambda c: f"{c['prof_q']}/{c['act_q']} ({c['prof_q']/c['act_q']*100:.0f}%)" if c['act_q']>0 else "—"),
    ("Max consec. profit mo",lambda c: str(c["max_win_mo"])),
    ("Max consec. losing mo",lambda c: str(c["max_loss_mo"])),
    ("Max consec. win trades",lambda c: str(c["max_win_tr"])),
    ("Max consec. loss trades",lambda c: str(c["max_loss_tr"])),
    ("Profit Factor",        lambda c: f"{c['pf']:.2f}"),
    ("Monthly Sharpe",       lambda c: f"{c['sharpe']:.2f}"),
    ("Calmar Ratio",         lambda c: f"{c['calmar']:.1f}"),
]

for m_name, fn in metrics:
    row = f"  {m_name:35s} │ "
    cells = []
    for label, _ in MODES:
        c = consy[label]
        if not c: cells.append(f"{'—':>15}"); continue
        try: cells.append(f"{fn(c):>15}")
        except: cells.append(f"{'—':>15}")
    row += " │ ".join(cells)
    print(row)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FINAL DECISION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
_print_section("SECTION 5 — DECISION MATRIX  (score each mode on key criteria)")

print(f"""
  Scoring (higher = better):
    PnL rank            : 7=highest, 1=lowest
    WR rank             : 7=highest, 1=lowest  (live-trading comfort)
    MaxDD rank          : 7=lowest DD, 1=highest DD
    Profitable months % : raw %
    Profitable qtrs %   : raw %
    Profit Factor       : raw value
    Monthly Sharpe      : raw value
""")

print(f"  {'Mode':28s}  {'PnL':>10}  {'WR':>6}  {'MaxDD':>7}  "
      f"{'ProfMo':>7}  {'ProfQ':>6}  {'PF':>5}  {'Sharpe':>6}  {'Score':>6}")
print("  " + "─"*96)

pnls    = [_stats(all_trades[label])["pnl"] for label,_ in MODES]
wrs     = [_stats(all_trades[label])["wr"]  for label,_ in MODES]
dds     = [_stats(all_trades[label])["maxdd"] for label,_ in MODES]
pnl_order = sorted(range(len(pnls)), key=lambda i: pnls[i])
wr_order  = sorted(range(len(wrs)),  key=lambda i: wrs[i])
dd_order  = sorted(range(len(dds)),  key=lambda i: dds[i], reverse=True)  # lower DD = better rank

for idx, (label, _) in enumerate(MODES):
    c = consy[label]
    if not c: continue
    pnl_rank = pnl_order.index(idx) + 1
    wr_rank  = wr_order.index(idx)  + 1
    dd_rank  = dd_order.index(idx)  + 1
    # Weighted score: PnL 30%, MaxDD 25%, WR 20%, Prof months 15%, PF 10%
    score = (pnl_rank*0.30 + dd_rank*0.25 + wr_rank*0.20
             + c["prof_mo"]/c["n_mo"]*7*0.15
             + min(c["pf"],5)/5*7*0.10)
    s = _stats(all_trades[label])
    prof_mo_pct = f"{c['prof_mo']/c['n_mo']*100:.0f}%"
    prof_q_pct  = f"{c['prof_q']}/{c['act_q']}"
    print(f"  {label:28s}  ${s['pnl']:>+9,.0f}  {s['wr']:>5.1f}%  {s['maxdd']:>6.1f}%  "
          f"{prof_mo_pct:>7}  {prof_q_pct:>6}  {c['pf']:>5.2f}  {c['sharpe']:>6.2f}  {score:>6.2f}")

print()
print("  Score weights: PnL 30% | MaxDD 25% | WR 20% | Profitable months 15% | PF 10%")
print("  Higher score = better overall balance of return AND risk")
print()
