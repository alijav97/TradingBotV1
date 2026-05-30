"""
btc_research/analysis_btc2_daily.py — Daily trade frequency + day-of-week analysis.

Questions answered:
  1. How many trades per day should I expect?
  2. Which days of the week are best/worst?
  3. Weekend vs weekday performance

Run:
    python btc_research/analysis_btc2_daily.py
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
_c  = df_btc["close"].astype(float)
_h  = df_btc["high"].astype(float)
_l  = df_btc["low"].astype(float)
_tr = pd.concat([_h-_l, (_h-_c.shift(1)).abs(), (_l-_c.shift(1)).abs()], axis=1).max(axis=1)

ATR_ARR = _tr.rolling(14).mean().bfill().values
EMA200  = _c.ewm(span=200, adjust=False).mean().values
TS      = df_btc.index
H_ARR, L_ARR, C_ARR = _h.values, _l.values, _c.values

_sp  = 2*14-1
_hd  = _h.diff(); _ld = _l.diff()
_pdm = _hd.where((_hd>0)&(_hd>-_ld), 0.0)
_mdm = (-_ld).where((-_ld>0)&(-_ld>_hd), 0.0)
_aw  = _tr.ewm(span=_sp, adjust=False).mean()
_pw  = _pdm.ewm(span=_sp, adjust=False).mean()
_mw  = _mdm.ewm(span=_sp, adjust=False).mean()
_pdi = 100*_pw/_aw; _mdi = 100*_mw/_aw
_dx  = 100*(_pdi-_mdi).abs()/(_pdi+_mdi).replace(0, float("nan"))
ADX_ARR = _dx.ewm(span=_sp, adjust=False).mean().fillna(0).values

# ── Simulator ─────────────────────────────────────────────────────────────────
def simulate(strategies):
    balance = float(b2cfg.STARTING_BALANCE)
    trades, open_t = [], None

    for i in range(220, len(df_btc)):
        bar_time = TS[i]
        hr  = bar_time.hour
        bh_ = float(H_ARR[i]); bl_ = float(L_ARR[i]); bc_ = float(C_ARR[i])
        atr_now = float(ATR_ARR[i])

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
                    open_t.update({"tp1_hit":True,"sl":entry_,"trail_peak":bh_ if long_ else bl_,
                                   "pnl_running":round(pnl,2),"r_running":r_})
                    balance+=pnl; continue
                elif age>=b2cfg.MAX_HOLD_BARS:
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
            open_t.update({"pnl_usd":round(open_t.get("pnl_running",0)+pnl,2),
                            "r_multiple":round(open_t.get("r_running",0)+r_,2),
                            "balance_after":round(balance,2),"exit_reason":ex})
            trades.append(open_t)
            open_t = None

        if open_t is not None: continue
        if hr not in b2cfg.KZ_HOURS: continue

        adx_v = float(ADX_ARR[i]); ema200v = float(EMA200[i])
        if adx_v < b2cfg.ADX_THRESHOLD: continue

        win = df_btc.iloc[max(0,i-220):i+1]

        for direction in ("long","short"):
            if direction=="long"  and bc_<ema200v: continue
            if direction=="short" and bc_>ema200v: continue

            fired_sig=None; fired_strat=None
            for strat in strategies:
                sig=strat.generate_signal(win, bar_time, direction)
                if sig.get("signal"):
                    fired_sig=sig; fired_strat=strat.name; break

            if fired_sig is None: continue
            sl_d=abs(float(fired_sig["entry"])-float(fired_sig["sl"]))
            if sl_d<=0: continue

            if adx_v>=b2cfg.ADX_SPLIT_STRONG_MIN: risk_pct=b2cfg.RISK_PCT_STRONG
            elif adx_v<=b2cfg.ADX_SPLIT_EARLY_MAX: risk_pct=b2cfg.RISK_PCT_EARLY_TREND
            else: risk_pct=b2cfg.RISK_PCT_TRANSITION

            ru=round(balance*risk_pct,2); lots=ru/sl_d
            tp1r=float(fired_sig.get("tp1_rr",b2cfg.TP1_RR)); tp2r=float(fired_sig.get("tp2_rr",b2cfg.TP2_RR))
            tp1=float(fired_sig["entry"])+tp1r*sl_d if direction=="long" else float(fired_sig["entry"])-tp1r*sl_d
            tp2=float(fired_sig["entry"])+tp2r*sl_d if direction=="long" else float(fired_sig["entry"])-tp2r*sl_d

            open_t = {
                "open_time":bar_time, "open_date":str(bar_time)[:10],
                "open_dow":bar_time.strftime("%A"),    # full day name
                "open_dow_num":bar_time.weekday(),     # 0=Mon 6=Sun
                "open_hour_utc":bar_time.hour,
                "open_idx":i, "direction":direction,
                "entry":float(fired_sig["entry"]),"sl":float(fired_sig["sl"]),"orig_sl":float(fired_sig["sl"]),
                "tp1":tp1,"tp2":tp2,"tp1_rr":tp1r,"tp2_rr":tp2r,
                "lots":lots,"risk_usd":ru,"risk_pct_used":risk_pct,
                "strategy_used":fired_strat,"entry_type":fired_sig.get("entry_type",""),
                "trail_peak":0.0,"tp1_hit":False,"pnl_running":0.0,"r_running":0.0,"pnl_usd":0.0,
                "adx_at_entry":round(adx_v,1),
            }
            break

    return trades


def _stats(trades):
    if not trades:
        return {"n":0,"wr":0.0,"avgr":0.0,"pnl":0.0,"maxdd":0.0,"pf":0.0}
    wins=[t for t in trades if t["pnl_usd"]>0]; losses=[t for t in trades if t["pnl_usd"]<0]
    wr=len(wins)/len(trades)*100; avgr=sum(t["r_multiple"] for t in trades)/len(trades)
    pnl=sum(t["pnl_usd"] for t in trades)
    gw=sum(t["pnl_usd"] for t in wins); gl=abs(sum(t["pnl_usd"] for t in losses))
    pf=gw/gl if gl>0 else float("inf")
    bals=[b2cfg.STARTING_BALANCE]+[t["balance_after"] for t in trades]
    peak=b2cfg.STARTING_BALANCE; maxdd=0.0
    for b in bals:
        if b>peak: peak=b
        dd=(peak-b)/peak*100
        if dd>maxdd: maxdd=dd
    return {"n":len(trades),"wr":round(wr,1),"avgr":round(avgr,2),"pnl":round(pnl,2),"maxdd":round(maxdd,1),"pf":round(pf,2)}


# ── Run ────────────────────────────────────────────────────────────────────────
slv2 = SwingLevelBreakV2(entry_mode=b2cfg.SWING_ENTRY_MODE, max_sl_atr=b2cfg.SWING_MAX_SL_ATR)
vb   = VolatilityBreakout(atr_multiplier=b2cfg.VB_ATR_MULTIPLIER, close_zone=b2cfg.VB_CLOSE_ZONE)

print("Running final config (SwingV2 first, VB fallback)...")
trades = simulate([slv2, vb])
s      = _stats(trades)

W = 100
SEP = "=" * W

# ── Total calendar days in backtest ───────────────────────────────────────────
date_range = pd.date_range(df_btc.index[220].date(), df_btc.index[-1].date(), freq="D")
total_days = len(date_range)
trading_days_with_entry = len(set(t["open_date"] for t in trades))

# ── Section 1: Trade frequency ─────────────────────────────────────────────────
print()
print(SEP)
print("  SECTION 1: HOW MANY TRADES PER DAY?")
print(SEP)

daily_counts = defaultdict(int)
for t in trades:
    daily_counts[t["open_date"]] += 1

zero_days   = total_days - trading_days_with_entry
one_trade   = sum(1 for v in daily_counts.values() if v == 1)
multi_trade = sum(1 for v in daily_counts.values() if v > 1)
max_in_day  = max(daily_counts.values()) if daily_counts else 0
avg_on_active = s["n"] / trading_days_with_entry if trading_days_with_entry else 0
avg_overall   = s["n"] / total_days

print(f"  Backtest period          : {df_btc.index[220].date()} to {df_btc.index[-1].date()}")
print(f"  Total calendar days      : {total_days}")
print(f"  Days with trades         : {trading_days_with_entry}  ({trading_days_with_entry/total_days*100:.0f}% of days)")
print(f"  Days with NO trade       : {zero_days}  ({zero_days/total_days*100:.0f}% of days)")
print()
print(f"  Average per calendar day : {avg_overall:.2f} trades")
print(f"  Average on active days   : {avg_on_active:.2f} trades")
print(f"  Max trades in one day    : {max_in_day}")
print()
print(f"  Days with exactly 1 trade: {one_trade}  ({one_trade/total_days*100:.0f}% of all days)")
print(f"  Days with 2+ trades      : {multi_trade}  ({multi_trade/total_days*100:.0f}% of all days)")
print()

# Show distribution
print(f"  Trade count distribution:")
for n in range(0, max_in_day+1):
    if n == 0:
        count = zero_days
    else:
        count = sum(1 for v in daily_counts.values() if v == n)
    bar   = "#" * count
    pct   = count/total_days*100
    print(f"    {n} trade(s): {count:4d} days ({pct:4.1f}%)  {bar[:60]}")

# ── Section 2: KZ hour breakdown ──────────────────────────────────────────────
print()
print(SEP)
print("  SECTION 2: PERFORMANCE BY KILL-ZONE HOUR (UTC)")
print(SEP)
print(f"  {'Hour (UTC)':15s}  {'UAE time':10s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>11}  {'MaxDD%':>7}")
print("  " + "-" * 70)

uae_map = {1:"05:00", 2:"06:00", 3:"07:00", 8:"12:00"}
total_kz_trades = 0

for hr in sorted(b2cfg.KZ_HOURS):
    ht = [t for t in trades if t["open_hour_utc"] == hr]
    total_kz_trades += len(ht)
    if not ht:
        print(f"  {hr:02d}:00 UTC          {uae_map.get(hr,'?'):10s}  no trades")
        continue
    s2 = _stats(ht)
    session = "Asia Night" if hr in [1,2,3] else "EU Open"
    print(f"  {hr:02d}:00 UTC ({session:10s})  {uae_map.get(hr,'?'):10s}  {s2['n']:>4}  {s2['wr']:>5.1f}%  "
          f"{s2['avgr']:>+5.2f}R  ${s2['pnl']:>+10,.2f}  {s2['maxdd']:>6.1f}%")

print()

# ── Section 3: Day-of-week breakdown ──────────────────────────────────────────
print(SEP)
print("  SECTION 3: PERFORMANCE BY DAY OF WEEK")
print(SEP)
print(f"  {'Day':12s}  {'Type':10s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>11}  {'MaxDD%':>7}  {'PnL/trade':>10}")
print("  " + "-" * 82)

DOW_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

best_wr_day = None; best_wr = 0
best_pnl_day = None; best_pnl = float("-inf")
worst_wr_day = None; worst_wr = 100
worst_pnl_day = None; worst_pnl = float("inf")

dow_results = {}
for dow_num, dow_name in enumerate(DOW_NAMES):
    dt = [t for t in trades if t["open_dow_num"] == dow_num]
    day_type = "Weekend" if dow_num >= 5 else "Weekday"
    if not dt:
        print(f"  {dow_name:12s}  {day_type:10s}  no trades")
        dow_results[dow_name] = {"n":0,"wr":0,"avgr":0,"pnl":0,"pf":0}
        continue
    s2 = _stats(dt)
    pnl_per_trade = s2["pnl"] / s2["n"] if s2["n"] > 0 else 0
    dow_results[dow_name] = s2
    tag = ""
    if s2["wr"] == max(_stats([t for t in trades if t["open_dow_num"]==d])["wr"]
                       for d in range(7) if any(t["open_dow_num"]==d for t in trades)):
        tag += " <- best WR"

    print(f"  {dow_name:12s}  {day_type:10s}  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  "
          f"${s2['pnl']:>+10,.2f}  {s2['maxdd']:>6.1f}%  ${pnl_per_trade:>+9,.2f}")

    if s2["n"] >= 5:
        if s2["wr"] > best_wr:
            best_wr = s2["wr"]; best_wr_day = dow_name
        if s2["wr"] < worst_wr:
            worst_wr = s2["wr"]; worst_wr_day = dow_name
        if s2["pnl"] > best_pnl:
            best_pnl = s2["pnl"]; best_pnl_day = dow_name
        if s2["pnl"] < worst_pnl:
            worst_pnl = s2["pnl"]; worst_pnl_day = dow_name

print()

# ── Section 4: Weekday vs Weekend ─────────────────────────────────────────────
print(SEP)
print("  SECTION 4: WEEKDAY vs WEEKEND")
print(SEP)

weekday_t  = [t for t in trades if t["open_dow_num"] < 5]
weekend_t  = [t for t in trades if t["open_dow_num"] >= 5]
s_wd = _stats(weekday_t); s_we = _stats(weekend_t)

print(f"  {'':12s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>11}  {'PF':>5}  {'MaxDD%':>7}")
print("  " + "-" * 60)
for label, s2 in [("WEEKDAYS (Mon-Fri)", s_wd), ("WEEKENDS (Sat-Sun)", s_we)]:
    if s2["n"] == 0:
        print(f"  {label:20s}  no trades"); continue
    print(f"  {label:20s}  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  "
          f"${s2['pnl']:>+10,.2f}  {s2['pf']:>4.2f}  {s2['maxdd']:>6.1f}%")

wd_pct = len(weekday_t)/len(trades)*100 if trades else 0
we_pct = len(weekend_t)/len(trades)*100 if trades else 0
print()
print(f"  Trade split: {wd_pct:.0f}% weekday | {we_pct:.0f}% weekend")
print()

# ── Section 5: Best and worst combinations ────────────────────────────────────
print(SEP)
print("  SECTION 5: BEST & WORST DAY+HOUR COMBINATIONS (min 5 trades)")
print(SEP)
print(f"  {'Day':12s}  {'Hour':8s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>11}")
print("  " + "-" * 56)

combos = []
for dow_num, dow_name in enumerate(DOW_NAMES):
    for hr in b2cfg.KZ_HOURS:
        ct = [t for t in trades if t["open_dow_num"]==dow_num and t["open_hour_utc"]==hr]
        if len(ct) >= 5:
            s2 = _stats(ct)
            combos.append((dow_name, hr, s2))

# Sort by AvgR descending
combos_sorted = sorted(combos, key=lambda x: x[2]["wr"], reverse=True)

print("  Top 5 by Win Rate:")
for dow_name, hr, s2 in combos_sorted[:5]:
    uae = uae_map.get(hr,"?")
    print(f"  {dow_name:12s}  {hr:02d}:00 UTC  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  ${s2['pnl']:>+10,.2f}")

print()
print("  Bottom 5 by Win Rate:")
for dow_name, hr, s2 in combos_sorted[-5:]:
    uae = uae_map.get(hr,"?")
    print(f"  {dow_name:12s}  {hr:02d}:00 UTC  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  ${s2['pnl']:>+10,.2f}")

print()

# ── Section 6: Summary ────────────────────────────────────────────────────────
print(SEP)
print("  SUMMARY")
print(SEP)
print(f"  Total trades over 2yr   : {s['n']}")
print(f"  Per calendar day        : {avg_overall:.2f} avg  (expect ~0 or 1 per day)")
print(f"  Days you actually trade : {trading_days_with_entry}/{total_days} ({trading_days_with_entry/total_days*100:.0f}%)")
print(f"  On active days          : {avg_on_active:.2f} trades avg (almost always just 1)")
print()
print(f"  Best performing day     : {best_wr_day}  ({dow_results[best_wr_day]['wr']}% WR  {dow_results[best_wr_day]['avgr']:+.2f}R)")
print(f"  Worst performing day    : {worst_wr_day}  ({dow_results[worst_wr_day]['wr']}% WR  {dow_results[worst_wr_day]['avgr']:+.2f}R)")
print()
print(f"  Best KZ hour            : check Section 2 above")
print(f"  Weekend trading         : {'YES — BTC trades 24/7, weekends included' if weekend_t else 'No weekend trades found'}")
if weekend_t:
    s_we_final = _stats(weekend_t)
    s_wd_final = _stats(weekday_t)
    better = "WEEKENDS" if s_we_final["wr"] > s_wd_final["wr"] else "WEEKDAYS"
    print(f"  Weekend vs Weekday WR   : {s_we_final['wr']}% vs {s_wd_final['wr']}%  -> {better} win on WR")
print(SEP)
