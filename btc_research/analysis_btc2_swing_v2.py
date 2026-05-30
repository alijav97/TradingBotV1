"""
btc_research/analysis_btc2_swing_v2.py — SwingLevelBreak v2 mode comparison.

Tests every entry mode of SwingLevelBreakV2 combined with VB,
keeping EMA200 + ADX-split risk + [1,2,3,8] UTC fixed.

Modes under test:
  0  v1 original       : first break, SL = prior swing structure (4.42×ATR avg)
  1  break_capped      : first break, SL capped at 2×ATR
  2  retest only       : retest entry, SL = bar extreme (~0.6×ATR)
  3  retest_preferred  : retest if available, else break_capped ← expected best
  4  both              : retest OR break whichever fires (retest priority on same bar)

For the best mode: full breakdown — monthly, per-entry-type, SL distance analysis.

Run:
    python btc_research/analysis_btc2_swing_v2.py
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
from btc_research.strategies.swing_level         import SwingLevelBreak
from btc_research.strategies.swing_level_v2      import SwingLevelBreakV2

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

# ── Indicators ─────────────────────────────────────────────────────────────────
_c = df_btc["close"].astype(float)
_h = df_btc["high"].astype(float)
_l = df_btc["low"].astype(float)
_tr = pd.concat([_h-_l, (_h-_c.shift(1)).abs(), (_l-_c.shift(1)).abs()], axis=1).max(axis=1)
ATR_ARR = _tr.rolling(14).mean().bfill().values
EMA200  = _c.ewm(span=200, adjust=False).mean().values
TS = df_btc.index; H_ARR = _h.values; L_ARR = _l.values; C_ARR = _c.values

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
        bar_time = TS[i]
        hr = bar_time.hour
        bh_ = float(H_ARR[i]); bl_ = float(L_ARR[i]); bc_ = float(C_ARR[i])

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
                    pnl=(bc_-entry_ if long_ else entry_-bc_)*open_t["lots"]
                    r_=pnl/risk_u if risk_u else 0; ex="MAX_HOLD"
                else:
                    continue
            else:
                atr_now=float(ATR_ARR[min(i,len(ATR_ARR)-1)])
                if long_:
                    if bh_>open_t["trail_peak"]: open_t["trail_peak"]=bh_
                    open_t["sl"]=max(open_t["trail_peak"]-b2cfg.TRAIL_ATR_MULT*atr_now,entry_)
                else:
                    if bl_<open_t["trail_peak"]: open_t["trail_peak"]=bl_
                    open_t["sl"]=min(open_t["trail_peak"]+b2cfg.TRAIL_ATR_MULT*atr_now,entry_)
                new_sl=open_t["sl"]
                hit_sl2=(bl_<=new_sl) if long_ else (bh_>=new_sl)
                if hit_sl2:
                    dist_run=abs(new_sl-entry_); r_=dist_run/sl_dist if sl_dist else 0
                    pnl=risk_u*r_; ex="TRAIL_SL"
                elif age>=b2cfg.MAX_HOLD_BARS:
                    dist_run=abs(bc_-entry_); r_=dist_run/sl_dist if sl_dist else 0
                    pnl=risk_u*r_; ex="MAX_HOLD"
                else:
                    continue

            balance+=pnl
            open_t["pnl_usd"]      =round(open_t.get("pnl_running",0)+pnl,2)
            open_t["r_multiple"]   =round(open_t.get("r_running",0)+r_,2)
            open_t["balance_after"]=round(balance,2)
            open_t["exit_reason"]  =ex
            trades.append(open_t); open_t=None

        if open_t is not None: continue
        if hr not in b2cfg.KZ_HOURS: continue

        adx_now=float(ADX_ARR[i]); ema200_now=float(EMA200[i])
        if adx_now < b2cfg.ADX_THRESHOLD: continue

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

            if adx_now>=b2cfg.ADX_SPLIT_STRONG_MIN: risk_pct=b2cfg.RISK_PCT_STRONG
            elif adx_now<=b2cfg.ADX_SPLIT_EARLY_MAX: risk_pct=b2cfg.RISK_PCT_EARLY_TREND
            else: risk_pct=b2cfg.RISK_PCT_TRANSITION

            ru=round(balance*risk_pct,2); lots=ru/sl_d
            tp1r=sig["tp1_rr"]; tp2r=sig["tp2_rr"]
            tp1=(float(sig["entry"])+tp1r*sl_d if direction=="long"
                 else float(sig["entry"])-tp1r*sl_d)
            tp2=(float(sig["entry"])+tp2r*sl_d if direction=="long"
                 else float(sig["entry"])-tp2r*sl_d)

            open_t={
                "open_time":bar_time,"open_idx":i,"direction":direction,
                "entry":float(sig["entry"]),"sl":float(sig["sl"]),"orig_sl":float(sig["sl"]),
                "tp1":tp1,"tp2":tp2,"tp1_rr":tp1r,"tp2_rr":tp2r,
                "lots":lots,"risk_usd":ru,"risk_pct":risk_pct,
                "signal_reason":sig.get("reason",""),
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


def _stats(trades, start_bal=None):
    sb = start_bal if start_bal is not None else b2cfg.STARTING_BALANCE
    if not trades:
        return {"n":0,"wr":0.0,"avgr":0.0,"pnl":0.0,"final_bal":sb,"maxdd":0.0}
    wins=[t for t in trades if t["pnl_usd"]>0]
    bals=[sb]+[t["balance_after"] for t in trades]
    peak=sb; maxdd=0.0
    for b in bals:
        if b>peak: peak=b
        dd=(peak-b)/peak*100
        if dd>maxdd: maxdd=dd
    return {
        "n":len(trades),"wr":round(len(wins)/len(trades)*100,1),
        "avgr":round(sum(t["r_multiple"] for t in trades)/len(trades),2),
        "pnl":round(sum(t["pnl_usd"] for t in trades),2),
        "final_bal":round(bals[-1],2),"maxdd":round(maxdd,1),
    }


# ── Strategy instances ─────────────────────────────────────────────────────────
VB = VolatilityBreakout(atr_multiplier=b2cfg.VB_ATR_MULTIPLIER, close_zone=b2cfg.VB_CLOSE_ZONE)

combos = [
    ("0  v1 original  (SL=prior struct)",   [VB, SwingLevelBreak()]),
    ("1  break_capped (SL≤2×ATR)",          [VB, SwingLevelBreakV2(entry_mode="break_capped",   max_sl_atr=2.0)]),
    ("2  break_capped (SL≤1.5×ATR)",        [VB, SwingLevelBreakV2(entry_mode="break_capped",   max_sl_atr=1.5)]),
    ("3  retest only",                       [VB, SwingLevelBreakV2(entry_mode="retest")]),
    ("4  retest_preferred (retest→break)",  [VB, SwingLevelBreakV2(entry_mode="retest_preferred", max_sl_atr=2.0)]),
    ("5  retest_preferred (cap=1.5×ATR)",   [VB, SwingLevelBreakV2(entry_mode="retest_preferred", max_sl_atr=1.5)]),
    ("6  both  (retest priority)",           [VB, SwingLevelBreakV2(entry_mode="both",            max_sl_atr=2.0)]),
]

W = 108
print("Running 7 SwingLevelBreak mode variants...")
print("Fixed: EMA200 + ADX-split + [1,2,3,8] UTC\n")

results = {}
for label, strat_list in combos:
    t = simulate(strat_list)
    s = _stats(t)
    results[label] = (t, s)
    print(f"  {label:42s}  n={s['n']:3d}  WR={s['wr']:5.1f}%  "
          f"AvgR={s['avgr']:+.2f}R  PnL=${s['pnl']:+,.2f}  MaxDD={s['maxdd']:.1f}%")


# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("="*W)
print("  SWINGLEVELBREAK v2 MODE COMPARISON  |  [01,02,03,08] UTC  |  EMA200 + ADX-split")
print(f"  Baseline reference: VB+SL original = $+19,392 | WR=51.4% | MaxDD=13.1%")
print("="*W)
print(f"  {'Mode':42s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>10}  "
      f"{'Delta':>8}  {'Bal $':>9}  {'MaxDD':>7}")
print("  "+"-"*100)

base_pnl = results[list(results.keys())[0]][1]["pnl"]
best_pnl = max(s["pnl"] for _,(t,s) in results.items() if s["n"]>=10)

for label, (trades, s) in results.items():
    delta = s["pnl"] - base_pnl
    tag   = " ◄◄◄ BEST" if s["pnl"]==best_pnl and s["n"]>=10 else ""
    d_str = f"{delta:>+8,.0f}" if delta != 0 else "baseline"
    print(f"  {label:42s}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+9,.2f}  {d_str}  ${s['final_bal']:>8,.2f}  "
          f"{s['maxdd']:>6.1f}%{tag}")
print("="*W)


# ── Entry type breakdown for v2 modes ─────────────────────────────────────────
print()
print("="*W)
print("  ENTRY TYPE BREAKDOWN — how many 'break' vs 'retest' entries fired")
print("="*W)
print(f"  {'Mode':42s}  {'Total':>5}  {'VB':>4}  {'SL-break':>8}  {'SL-retest':>9}  "
      f"{'Retest WR':>9}  {'Break WR':>8}")
print("  "+"-"*92)

for label, (trades, s) in results.items():
    vb_t   = [t for t in trades if "Volatility" in t.get("strategy_used","")]
    br_t   = [t for t in trades if t.get("entry_type")=="break"  and "Swing" in t.get("strategy_used","")]
    rt_t   = [t for t in trades if t.get("entry_type")=="retest" and "Swing" in t.get("strategy_used","")]

    rt_wr  = f"{len([t for t in rt_t if t['pnl_usd']>0])/len(rt_t)*100:.1f}%" if rt_t else "—"
    br_wr  = f"{len([t for t in br_t if t['pnl_usd']>0])/len(br_t)*100:.1f}%" if br_t else "—"
    print(f"  {label:42s}  {s['n']:>5}  {len(vb_t):>4}  {len(br_t):>8}  {len(rt_t):>9}  "
          f"{rt_wr:>9}  {br_wr:>8}")


# ── SL distance analysis for each mode ────────────────────────────────────────
print()
print("="*W)
print("  SL DISTANCE IN ATR  (lower = better R per trade)")
print("  Original v1 avg: VB=1.34×ATR  SL-break=4.42×ATR")
print("="*W)
print(f"  {'Mode':42s}  {'VB SL':>6}  {'SL-break':>8}  {'SL-retest':>9}  "
      f"{'Combined':>8}  {'AvgR':>6}")
print("  "+"-"*88)

for label, (trades, s) in results.items():
    def avg_sl_atr(tlist):
        vals = [abs(t["entry"]-t["orig_sl"]) / t["atr_at_entry"]
                for t in tlist if t.get("atr_at_entry",0)>0]
        return f"{np.mean(vals):.2f}×" if vals else "—"

    vb_t = [t for t in trades if "Volatility" in t.get("strategy_used","")]
    br_t = [t for t in trades if t.get("entry_type")=="break"  and "Swing" in t.get("strategy_used","")]
    rt_t = [t for t in trades if t.get("entry_type")=="retest" and "Swing" in t.get("strategy_used","")]
    all_sl_vals = [abs(t["entry"]-t["orig_sl"])/t["atr_at_entry"]
                   for t in trades if t.get("atr_at_entry",0)>0]
    combined = f"{np.mean(all_sl_vals):.2f}×" if all_sl_vals else "—"

    print(f"  {label:42s}  {avg_sl_atr(vb_t):>6}  {avg_sl_atr(br_t):>8}  "
          f"{avg_sl_atr(rt_t):>9}  {combined:>8}  {s['avgr']:>+5.2f}R")


# ── Best mode: full monthly breakdown ─────────────────────────────────────────
eligible = [(lbl,t,s) for lbl,(t,s) in results.items() if s["n"]>=10]
if eligible:
    best_lbl, best_trades, best_s = max(eligible, key=lambda x: x[2]["pnl"])
    print()
    print("="*W)
    print(f"  MONTHLY BREAKDOWN — Best: {best_lbl}")
    print("="*W)
    print(f"  {'Month':>7}  {'#':>3}  {'WR%':>5}  {'AvgR':>6}  {'PnL$':>9}  "
          f"{'Bal$':>9}  {'VB':>3}  {'Break':>5}  {'Retest':>6}  {'L/S':>6}")
    print("  "+"-"*88)

    monthly = defaultdict(list)
    for t in best_trades:
        monthly[str(t["open_time"])[:7]].append(t)

    for ym in sorted(monthly.keys()):
        ts_  = monthly[ym]
        s2   = _stats(ts_)
        vb_  = len([t for t in ts_ if "Volatility" in t.get("strategy_used","")])
        br_  = len([t for t in ts_ if t.get("entry_type")=="break"])
        rt_  = len([t for t in ts_ if t.get("entry_type")=="retest"])
        lo_  = len([t for t in ts_ if t["direction"]=="long"])
        sh_  = len([t for t in ts_ if t["direction"]=="short"])
        tag  = "  <<HIGH WR" if s2["wr"]>=60 else ("  !WEAK" if s2["pnl"]<0 else "")
        print(f"  {ym:>7}  {s2['n']:>3}  {s2['wr']:>4.1f}%  {s2['avgr']:>+5.2f}R  "
              f"${s2['pnl']:>+8,.2f}  ${s2['final_bal']:>8,.2f}  "
              f"{vb_:>3}  {br_:>5}  {rt_:>6}  {lo_}L/{sh_}S{tag}")

    total_s = _stats(best_trades)
    print("  "+"-"*88)
    print(f"  {'TOTAL':>7}  {total_s['n']:>3}  {total_s['wr']:>4.1f}%  "
          f"{total_s['avgr']:>+5.2f}R  ${total_s['pnl']:>+8,.2f}  "
          f"${total_s['final_bal']:>8,.2f}")

    # Retest vs break stats within best mode
    all_rt = [t for t in best_trades if t.get("entry_type")=="retest"]
    all_br = [t for t in best_trades if t.get("entry_type")=="break" and "Swing" in t.get("strategy_used","")]
    all_vb = [t for t in best_trades if "Volatility" in t.get("strategy_used","")]

    print()
    print("="*W)
    print(f"  ENTRY TYPE QUALITY — {best_lbl}")
    print("="*W)
    print(f"  {'Type':20s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>10}  "
          f"{'Avg SL':>8}  {'MaxDD':>7}")
    print("  "+"-"*70)
    for name_, ts_ in [("Volatility Breakout", all_vb),
                        ("SL Break (first)",    all_br),
                        ("SL Retest",           all_rt)]:
        if not ts_: continue
        s2  = _stats(ts_)
        sl_vals = [abs(t["entry"]-t["orig_sl"])/t["atr_at_entry"]
                   for t in ts_ if t.get("atr_at_entry",0)>0]
        avg_sl = f"{np.mean(sl_vals):.2f}×ATR" if sl_vals else "—"
        print(f"  {name_:20s}  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  "
              f"${s2['pnl']:>+9,.2f}  {avg_sl:>8}  {s2['maxdd']:>6.1f}%")


# ── Final recommendation ───────────────────────────────────────────────────────
print()
print("="*W)
print("  RECOMMENDATION  — Which mode should Bot 2 use?")
print("="*W)
print(f"  Original v1 (baseline): $+19,392 | WR=51.4% | MaxDD=13.1%")
print()
ranked = sorted(eligible, key=lambda x: x[2]["pnl"], reverse=True)
for i,(lbl,t,s) in enumerate(ranked[:5]):
    delta = s["pnl"] - base_pnl
    mark  = "  ★ RECOMMENDED" if i==0 else ""
    print(f"  #{i+1} {lbl:42s}  ${s['pnl']:+,.2f}  ({delta:+,.0f})  "
          f"WR={s['wr']}%  AvgR={s['avgr']:+.2f}R  MaxDD={s['maxdd']}%{mark}")
print()
