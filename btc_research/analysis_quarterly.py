"""
btc_research/analysis_quarterly.py

Runs Version D (EMA200 + FlippedRisk) and breaks results down by quarter.
Gives a clear picture of how the strategy performed in each 3-month period.

Run: python btc_research/analysis_quarterly.py
"""
import sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(str(_ROOT))

import btc_research.settings as cfg
from btc_research.data.fetcher import fetch_all
from btc_research.strategies.combined import CombinedStrategy
import pandas as pd
import numpy as np
from collections import defaultdict

data   = fetch_all(use_cache=True, force_refresh=False)
df_btc = data.get(cfg.BTC_SYMBOL, pd.DataFrame())
if df_btc.empty:
    print("ERROR: No BTC data."); sys.exit(1)

if "time" in df_btc.columns:
    df_btc = df_btc.set_index(pd.to_datetime(df_btc["time"], utc=True)).drop(columns=["time"])
elif not isinstance(df_btc.index, pd.DatetimeIndex):
    df_btc.index = pd.to_datetime(df_btc.index, utc=True)

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_adx(df, period=14):
    h=df["high"].astype(float); l=df["low"].astype(float); c=df["close"].astype(float)
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    hd=h.diff(); ld=l.diff()
    pdm=hd.where((hd>0)&(hd>-ld),0.0); mdm=(-ld).where((-ld>0)&(-ld>hd),0.0)
    sp=2*period-1
    aw=tr.ewm(span=sp,adjust=False).mean()
    pw=pdm.ewm(span=sp,adjust=False).mean(); mw=mdm.ewm(span=sp,adjust=False).mean()
    pdi=100*pw/aw; mdi=100*mw/aw
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,float("nan"))
    return dx.ewm(span=sp,adjust=False).mean().fillna(0).values

_c      = df_btc["close"].astype(float)
_h      = df_btc["high"].astype(float)
_l      = df_btc["low"].astype(float)
_tr     = pd.concat([_h-_l,(_h-_c.shift(1)).abs(),(_l-_c.shift(1)).abs()],axis=1).max(axis=1)
ATR_ARR = _tr.rolling(14).mean().bfill().values
ADX_ARR = calc_adx(df_btc, 14)
EMA200  = _c.ewm(span=200, adjust=False).mean().values
EMA50   = _c.ewm(span=50,  adjust=False).mean().values
EMA50_SLOPE = pd.Series(EMA50).diff(8).fillna(0).values
TS      = df_btc.index
H_ARR   = _h.values; L_ARR   = _l.values; C_ARR   = _c.values

# ── Version D simulator ───────────────────────────────────────────────────────
def simulate_D(strategy):
    balance = float(cfg.STARTING_BALANCE)
    trades  = []
    open_t  = None

    for i in range(220, len(df_btc)):
        bar_time = TS[i]; hr = bar_time.hour
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
                    balance+=pnl; open_t["pnl_running"]=round(pnl,2)
                    open_t["r_running"]=r_; continue
                elif age >= cfg.MAX_HOLD_BARS:
                    pnl=(bc_-entry_ if long_ else entry_-bc_)*open_t["lots"]
                    r_=pnl/risk_u if risk_u else 0; ex="MAX_HOLD"
                else:
                    continue
            else:
                atr_now = float(ATR_ARR[min(i,len(ATR_ARR)-1)])
                if long_:
                    if bh_>open_t["trail_peak"]: open_t["trail_peak"]=bh_
                    open_t["sl"]=max(open_t["trail_peak"]-2.0*atr_now, entry_)
                else:
                    if bl_<open_t["trail_peak"]: open_t["trail_peak"]=bl_
                    open_t["sl"]=min(open_t["trail_peak"]+2.0*atr_now, entry_)
                new_sl=open_t["sl"]
                if (long_ and bl_<=new_sl) or (not long_ and bh_>=new_sl):
                    dist_run=abs(new_sl-entry_); r_=dist_run/sl_dist if sl_dist else 0
                    pnl=risk_u*r_; ex="TRAIL_SL"
                elif age >= cfg.MAX_HOLD_BARS:
                    dist_run=abs(bc_-entry_); r_=dist_run/sl_dist if sl_dist else 0
                    pnl=risk_u*r_; ex="MAX_HOLD"
                else:
                    continue

            balance+=pnl
            open_t["pnl_usd"]       = round(open_t.get("pnl_running",0)+pnl, 2)
            open_t["r_multiple"]    = round(open_t.get("r_running",0)+r_, 2)
            open_t["balance_after"] = round(balance, 2)
            open_t["exit_reason"]   = ex
            trades.append(open_t)
            open_t = None

        if open_t is not None: continue

        ks=cfg.KZ_START_UTC; ke=cfg.KZ_END_UTC
        in_kz=(ks<ke and ks<=hr<ke) or (ks>ke and (hr>=ks or hr<ke))
        if not in_kz: continue

        adx_now    = float(ADX_ARR[i])
        if adx_now < 20: continue
        ema200_now = float(EMA200[i])
        bc_        = float(C_ARR[i])

        win = df_btc.iloc[max(0,i-220):i+1]
        for direction in ("long","short"):
            if direction=="long"  and bc_ < ema200_now: continue
            if direction=="short" and bc_ > ema200_now: continue

            sig = strategy.generate_signal(win, bar_time, direction)
            if sig.get("signal"):
                sl_d=abs(float(sig["entry"])-float(sig["sl"]))
                if sl_d<=0: continue
                rp = 0.03 if adx_now <= 28 else cfg.RISK_PCT
                ru=round(balance*rp,2); lots=ru/sl_d
                tp1r=sig.get("tp1_rr",cfg.TP1_RR); tp2r=sig.get("tp2_rr",cfg.TP2_RR)
                tp1=float(sig["entry"])+tp1r*sl_d if direction=="long" else float(sig["entry"])-tp1r*sl_d
                tp2=float(sig["entry"])+tp2r*sl_d if direction=="long" else float(sig["entry"])-tp2r*sl_d
                open_t={
                    "open_time":bar_time,"open_idx":i,"direction":direction,
                    "entry":float(sig["entry"]),"sl":float(sig["sl"]),"orig_sl":float(sig["sl"]),
                    "tp1":tp1,"tp2":tp2,"tp1_rr":tp1r,"tp2_rr":tp2r,
                    "lots":lots,"risk_usd":ru,
                    "signal_reason":sig.get("reason",""),
                    "trail_peak":0.0,"tp1_hit":False,"pnl_running":0.0,"r_running":0.0,"pnl_usd":0.0,
                    "adx_at_entry":round(adx_now,1),
                }
                break
    return trades

# ── Run ───────────────────────────────────────────────────────────────────────
strat  = CombinedStrategy(atr_multiplier=1.2, close_zone=0.45, range_bars=6)
print("Running Version D simulation...")
trades = simulate_D(strat)
print(f"Total trades: {len(trades)}\n")

# ── Quarter helper ────────────────────────────────────────────────────────────
def get_quarter(ts):
    m = pd.Timestamp(ts).month
    y = pd.Timestamp(ts).year
    q = (m - 1) // 3 + 1
    return f"{y}-Q{q}"

def quarter_label(qstr):
    y, q = qstr.split("-Q")
    months = {1:"Jan-Mar", 2:"Apr-Jun", 3:"Jul-Sep", 4:"Oct-Dec"}
    return f"{y} Q{q} ({months[int(q)]})"

# ── Group by quarter ──────────────────────────────────────────────────────────
quarterly = defaultdict(list)
for t in trades:
    quarterly[get_quarter(t["open_time"])].append(t)

# ── BTC price range by quarter ────────────────────────────────────────────────
def btc_range_for_quarter(qstr):
    y, q = qstr.split("-Q"); q=int(q); y=int(y)
    start_m = (q-1)*3+1; end_m = q*3
    try:
        mask = (pd.to_datetime(df_btc.index).year == y) & \
               (pd.to_datetime(df_btc.index).month >= start_m) & \
               (pd.to_datetime(df_btc.index).month <= end_m)
        sub = df_btc[mask]
        if sub.empty: return "N/A"
        lo = int(sub["low"].astype(float).min()); hi = int(sub["high"].astype(float).max())
        return f"${lo//1000}K-${hi//1000}K"
    except: return "N/A"

# ── Print quarterly summary ───────────────────────────────────────────────────
W = 130
print("="*W)
print("  VERSION D — QUARTERLY BACKTEST  |  $500 start, 2%/3% risk, compounding")
print("  Strategy: EMA200 trend filter + Flipped risk (3% ADX 20-28, 2% ADX>28)")
print("  Session : 21:00-00:00 UTC  |  Trail 2.0xATR after TP1  |  ADX>20 gate")
print("="*W)

bal_start_q = float(cfg.STARTING_BALANCE)

print(f"\n  {'Quarter':<22}  {'#':>3}  {'WR%':>5}  {'AvgR':>6}  {'PnL$':>9}  "
      f"{'StartBal':>9}  {'EndBal':>9}  {'Qtr%':>7}  {'MaxDD':>6}  "
      f"{'L/S':>7}  {'LWR':>4}  {'SWR':>4}  {'BTC Range':>14}")
print(f"  {'-'*125}")

all_qtrs = sorted(quarterly.keys())
overall_peak = float(cfg.STARTING_BALANCE)
overall_max_dd = 0.0
profitable_qtrs = 0

for qstr in all_qtrs:
    ts_     = quarterly[qstr]
    wins    = [t for t in ts_ if t["pnl_usd"]>0]
    wr      = round(len(wins)/len(ts_)*100, 1)
    avgr    = round(sum(t["r_multiple"] for t in ts_)/len(ts_), 2)
    pnl     = round(sum(t["pnl_usd"] for t in ts_), 2)
    end_bal = round(ts_[-1]["balance_after"], 2)
    qtr_pct = round((end_bal/bal_start_q - 1)*100, 1)

    longs  = [t for t in ts_ if t["direction"]=="long"]
    shorts = [t for t in ts_ if t["direction"]=="short"]
    lwr    = round(len([t for t in longs  if t["pnl_usd"]>0])/len(longs)*100)  if longs  else 0
    swr    = round(len([t for t in shorts if t["pnl_usd"]>0])/len(shorts)*100) if shorts else 0
    ls     = f"{len(longs)}L/{len(shorts)}S"

    # Max DD within quarter
    peak_q = bal_start_q; max_dd_q = 0.0
    for t in ts_:
        b = t["balance_after"]
        if b > peak_q: peak_q = b
        dd = (peak_q-b)/peak_q*100
        if dd > max_dd_q: max_dd_q = dd
        if b > overall_peak: overall_peak = b
        dd_overall = (overall_peak-b)/overall_peak*100
        if dd_overall > overall_max_dd: overall_max_dd = dd_overall

    btc_rng = btc_range_for_quarter(qstr)
    tag = "  ✓" if pnl > 0 else "  ✗ LOSS"
    if pnl > 0: profitable_qtrs += 1

    print(f"  {quarter_label(qstr):<22}  {len(ts_):>3}  {wr:>4.1f}%  {avgr:>+5.2f}R"
          f"  {pnl:>+9.2f}  {bal_start_q:>9.2f}  {end_bal:>9.2f}"
          f"  {qtr_pct:>+6.1f}%  {max_dd_q:>5.1f}%"
          f"  {ls:>7}  {lwr:>3}%  {swr:>3}%  {btc_rng:>14}{tag}")

    bal_start_q = end_bal

print(f"  {'-'*125}")

# Totals
total   = len(trades)
wins_all= [t for t in trades if t["pnl_usd"]>0]
total_pnl = sum(t["pnl_usd"] for t in trades)
final_bal = trades[-1]["balance_after"]
total_ret = round((final_bal/float(cfg.STARTING_BALANCE)-1)*100, 1)

print(f"\n  {'OVERALL 2-YEAR TOTAL':<22}  {total:>3}  "
      f"{len(wins_all)/total*100:>4.1f}%  "
      f"{sum(t['r_multiple'] for t in trades)/total:>+5.2f}R  "
      f"{total_pnl:>+9.2f}  {float(cfg.STARTING_BALANCE):>9.2f}  "
      f"{final_bal:>9.2f}  {total_ret:>+6.1f}%  {overall_max_dd:>5.1f}%")

print(f"\n  Profitable quarters : {profitable_qtrs} / {len(all_qtrs)}")
print(f"  Losing quarters     : {len(all_qtrs)-profitable_qtrs} / {len(all_qtrs)}")

# ── Monthly breakdown inside each quarter ─────────────────────────────────────
print()
print("="*W)
print("  MONTHLY DETAIL INSIDE EACH QUARTER")
print("="*W)

monthly = defaultdict(list)
for t in trades:
    ym = str(t["open_time"])[:7]
    monthly[ym].append(t)

bal_m = float(cfg.STARTING_BALANCE)
current_q = None
for ym in sorted(monthly.keys()):
    ts_  = monthly[ym]
    qstr = get_quarter(ts_[0]["open_time"])
    if qstr != current_q:
        current_q = qstr
        print(f"\n  ── {quarter_label(qstr)} ──────────────────────────────────────")
        print(f"  {'Month':>7}  {'#':>3}  {'WR%':>5}  {'AvgR':>6}  {'PnL$':>9}  "
              f"{'Bal$':>10}  {'L/S':>7}  {'LWR':>4}  {'SWR':>4}")
        print(f"  {'-'*75}")

    wins  = [t for t in ts_ if t["pnl_usd"]>0]
    wr    = round(len(wins)/len(ts_)*100, 1)
    avgr  = round(sum(t["r_multiple"] for t in ts_)/len(ts_), 2)
    pnl   = round(sum(t["pnl_usd"] for t in ts_), 2)
    bal   = round(ts_[-1]["balance_after"], 2)
    longs  = [t for t in ts_ if t["direction"]=="long"]
    shorts = [t for t in ts_ if t["direction"]=="short"]
    lwr    = round(len([t for t in longs  if t["pnl_usd"]>0])/len(longs)*100)  if longs  else 0
    swr    = round(len([t for t in shorts if t["pnl_usd"]>0])/len(shorts)*100) if shorts else 0
    ls     = f"{len(longs)}L/{len(shorts)}S"
    tag    = "  <<< HIGH WR" if wr >= 55 else ("  !WEAK" if wr < 35 else "")
    print(f"  {ym:>7}  {len(ts_):>3}  {wr:>4.1f}%  {avgr:>+5.2f}R  {pnl:>+9.2f}"
          f"  {bal:>10.2f}  {ls:>7}  {lwr:>3}%  {swr:>3}%{tag}")

# ── Quarter-by-quarter equity narrative ───────────────────────────────────────
print()
print("="*W)
print("  EQUITY CURVE NARRATIVE — How $500 grew quarter by quarter")
print("="*W)

bal = float(cfg.STARTING_BALANCE)
print(f"\n  Start : ${bal:,.2f}")
for qstr in all_qtrs:
    ts_   = quarterly[qstr]
    pnl   = sum(t["pnl_usd"] for t in ts_)
    end_b = ts_[-1]["balance_after"]
    pct   = (end_b/bal-1)*100
    bar_len = max(0, min(40, int(pct/5)))
    bar   = ("█" * bar_len) if pct >= 0 else ("▓" * min(40, int(abs(pct)/5)))
    sign  = "+" if pct >= 0 else ""
    print(f"  {quarter_label(qstr):<22} : ${bal:>8,.2f} → ${end_b:>9,.2f}  "
          f"({sign}{pct:.1f}%)  {bar}")
    bal = end_b

print(f"\n  End   : ${bal:,.2f}  (×{bal/float(cfg.STARTING_BALANCE):.1f} in 2 years)\n")
