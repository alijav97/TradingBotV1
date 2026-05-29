"""
btc_research/analysis_weak_months.py

Deep-dive on weak months (WR < 35% OR negative PnL) from Version D simulation.
Shows trade-by-trade with RSI, EMA50 slope, and direction context.
Then tests Version E: consecutive-loss pause + EMA50 slope filter.

Run: python btc_research/analysis_weak_months.py
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

# Ensure datetime index
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
    adx=dx.ewm(span=sp,adjust=False).mean().fillna(0)
    return adx.values, pdi.fillna(0).values, mdi.fillna(0).values

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = (-delta).where(delta < 0, 0.0)
    avg_g = gain.ewm(com=period-1, adjust=False).mean()
    avg_l = loss.ewm(com=period-1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, float("nan"))
    return (100 - 100/(1+rs)).fillna(50).values

_c      = df_btc["close"].astype(float)
_h      = df_btc["high"].astype(float)
_l      = df_btc["low"].astype(float)
_tr     = pd.concat([_h-_l,(_h-_c.shift(1)).abs(),(_l-_c.shift(1)).abs()],axis=1).max(axis=1)
ATR_ARR = _tr.rolling(14).mean().bfill().values
ADX_ARR, PDI_ARR, MDI_ARR = calc_adx(df_btc, 14)
EMA50   = _c.ewm(span=50,  adjust=False).mean().values
EMA200  = _c.ewm(span=200, adjust=False).mean().values
RSI_ARR = calc_rsi(_c, 14)
# EMA50 slope: difference over last 8 bars (positive = rising, negative = falling)
EMA50_SLOPE = pd.Series(EMA50).diff(8).fillna(0).values
TS      = df_btc.index
H_ARR   = _h.values
L_ARR   = _l.values
C_ARR   = _c.values

# ── Simulator (Version D: EMA200 filter + flipped risk) ───────────────────────
def simulate(strategy, use_ema200=True, flipped_risk=True,
             ema50_slope_filter=False, consec_loss_pause=0):
    """
    ema50_slope_filter : only take longs if EMA50 slope > 0, only shorts if slope < 0
    consec_loss_pause  : after N consecutive losses, skip next N bars (0 = off)
    """
    balance    = float(cfg.STARTING_BALANCE)
    trades     = []
    open_t     = None
    consec_loss = 0
    pause_until = -1

    for i in range(220, len(df_btc)):
        bar_time = TS[i]; hr = bar_time.hour
        bh_ = float(H_ARR[i]); bl_ = float(L_ARR[i]); bc_ = float(C_ARR[i])

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
                    balance+=pnl; open_t["pnl_running"]=round(pnl,2); open_t["r_running"]=r_; continue
                elif age >= cfg.MAX_HOLD_BARS:
                    pnl=(bc_-entry_ if long_ else entry_-bc_)*open_t["lots"]
                    r_=pnl/risk_u if risk_u else 0; ex="MAX_HOLD"
                else:
                    continue
            else:
                atr_now = float(ATR_ARR[min(i,len(ATR_ARR)-1)])
                if long_:
                    if bh_ > open_t["trail_peak"]: open_t["trail_peak"] = bh_
                    open_t["sl"] = max(open_t["trail_peak"] - 2.0*atr_now, entry_)
                else:
                    if bl_ < open_t["trail_peak"]: open_t["trail_peak"] = bl_
                    open_t["sl"] = min(open_t["trail_peak"] + 2.0*atr_now, entry_)
                new_sl = open_t["sl"]
                if (long_ and bl_<=new_sl) or (not long_ and bh_>=new_sl):
                    dist_run=abs(new_sl-entry_); r_=dist_run/sl_dist if sl_dist else 0
                    pnl=risk_u*r_; ex="TRAIL_SL"
                elif age >= cfg.MAX_HOLD_BARS:
                    dist_run=abs(bc_-entry_); r_=dist_run/sl_dist if sl_dist else 0
                    pnl=risk_u*r_; ex="MAX_HOLD"
                else:
                    continue

            balance += pnl
            is_win = pnl > 0
            if is_win:
                consec_loss = 0
            else:
                consec_loss += 1
                if consec_loss_pause > 0 and consec_loss >= consec_loss_pause:
                    pause_until = i + consec_loss_pause * 24   # pause N trading days
                    consec_loss = 0

            open_t["pnl_usd"]       = round(open_t.get("pnl_running",0)+pnl, 2)
            open_t["r_multiple"]    = round(open_t.get("r_running",0)+r_, 2)
            open_t["balance_after"] = round(balance, 2)
            open_t["exit_reason"]   = ex
            trades.append(open_t)
            open_t = None

        if open_t is not None: continue

        # ── Entry filters ────────────────────────────────────────────────────
        ks=cfg.KZ_START_UTC; ke=cfg.KZ_END_UTC
        in_kz=(ks<ke and ks<=hr<ke) or (ks>ke and (hr>=ks or hr<ke))
        if not in_kz: continue

        adx_now = float(ADX_ARR[i])
        if adx_now < 20: continue

        if consec_loss_pause > 0 and i < pause_until: continue

        ema200_now = float(EMA200[i])
        ema50_now  = float(EMA50[i])
        slope_now  = float(EMA50_SLOPE[i])
        rsi_now    = float(RSI_ARR[i])

        win = df_btc.iloc[max(0, i-220):i+1]
        for direction in ("long", "short"):
            if use_ema200:
                if direction == "long"  and bc_ < ema200_now: continue
                if direction == "short" and bc_ > ema200_now: continue

            if ema50_slope_filter:
                if direction == "long"  and slope_now <= 0: continue
                if direction == "short" and slope_now >= 0: continue

            sig = strategy.generate_signal(win, bar_time, direction)
            if sig.get("signal"):
                sl_d = abs(float(sig["entry"])-float(sig["sl"]))
                if sl_d <= 0: continue
                rp = 0.03 if adx_now <= 28 else cfg.RISK_PCT   # flipped risk
                ru = round(balance*rp, 2); lots=ru/sl_d
                tp1r=sig.get("tp1_rr",cfg.TP1_RR); tp2r=sig.get("tp2_rr",cfg.TP2_RR)
                tp1=float(sig["entry"])+tp1r*sl_d if direction=="long" else float(sig["entry"])-tp1r*sl_d
                tp2=float(sig["entry"])+tp2r*sl_d if direction=="long" else float(sig["entry"])-tp2r*sl_d
                open_t = {
                    "open_time": bar_time, "open_idx": i, "direction": direction,
                    "entry": float(sig["entry"]), "sl": float(sig["sl"]), "orig_sl": float(sig["sl"]),
                    "tp1": tp1, "tp2": tp2, "tp1_rr": tp1r, "tp2_rr": tp2r,
                    "lots": lots, "risk_usd": ru,
                    "signal_reason": sig.get("reason",""),
                    "trail_peak": 0.0, "tp1_hit": False, "pnl_running": 0.0, "r_running": 0.0, "pnl_usd": 0.0,
                    "adx":   round(adx_now, 1),
                    "pdi":   round(float(PDI_ARR[i]), 1),
                    "mdi":   round(float(MDI_ARR[i]), 1),
                    "rsi":   round(rsi_now, 1),
                    "slope": round(slope_now, 0),
                    "above_ema200": bc_ > ema200_now,
                    "above_ema50":  bc_ > ema50_now,
                    "risk_pct":     rp,
                }
                break
    return trades

def get_stats(trades):
    if not trades:
        return {"trades":0,"wr":0.0,"avg_r":0.0,"pnl":0.0,
                "final":float(cfg.STARTING_BALANCE),"max_dd":0.0,"pct":0.0}
    total=len(trades); wins=[t for t in trades if t["pnl_usd"]>0]
    pnl=sum(t["pnl_usd"] for t in trades)
    avg_r=sum(t["r_multiple"] for t in trades)/total
    bal=float(cfg.STARTING_BALANCE); peak=bal; max_dd=0.0
    for t in trades:
        bal=t["balance_after"]
        if bal>peak: peak=bal
        dd=(peak-bal)/peak*100
        if dd>max_dd: max_dd=dd
    return {"trades":total,"wr":round(len(wins)/total*100,1),"avg_r":round(avg_r,2),
            "pnl":round(pnl,2),"final":round(trades[-1]["balance_after"],2),
            "max_dd":round(max_dd,1),
            "pct":round((trades[-1]["balance_after"]/float(cfg.STARTING_BALANCE)-1)*100,1)}

strat = CombinedStrategy(atr_multiplier=1.2, close_zone=0.45, range_bars=6)

print("Running simulations...")
print("  [D] Baseline (EMA200 + FlippedRisk)...")
tD = simulate(strat, use_ema200=True, flipped_risk=True,
              ema50_slope_filter=False, consec_loss_pause=0)

print("  [E1] +EMA50 slope filter...")
tE1 = simulate(strat, use_ema200=True, flipped_risk=True,
               ema50_slope_filter=True, consec_loss_pause=0)

print("  [E2] +Consecutive loss pause (pause after 2 losses)...")
tE2 = simulate(strat, use_ema200=True, flipped_risk=True,
               ema50_slope_filter=False, consec_loss_pause=2)

print("  [E3] +Both filters combined...")
tE3 = simulate(strat, use_ema200=True, flipped_risk=True,
               ema50_slope_filter=True, consec_loss_pause=2)

sD=get_stats(tD); sE1=get_stats(tE1); sE2=get_stats(tE2); sE3=get_stats(tE3)

W = 115
print()
print("="*W)
print("  COMPARISON — Version D vs Weak-Month Fixes")
print("="*W)
print(f"  {'Version':<50}  {'Trades':>6}  {'WR%':>5}  {'AvgR':>6}  {'PnL$':>10}  {'Final$':>10}  {'Return':>8}  {'MaxDD':>6}")
print(f"  {'-'*111}")
all_s = [(tD,sD,"D  Baseline (EMA200+FlippedRisk)"),
         (tE1,sE1,"E1 +EMA50 slope filter"),
         (tE2,sE2,"E2 +Consec-loss pause (2 losses)"),
         (tE3,sE3,"E3 +EMA50 slope + Consec-loss pause")]
best_pnl = max(s["pnl"] for _,s,_ in all_s)
for t_,s,lbl in all_s:
    mk = "  <- BEST" if s["pnl"]==best_pnl else ""
    print(f"  {lbl:<50}  {s['trades']:>6}  {s['wr']:>4.1f}%  {s['avg_r']:>+5.2f}R"
          f"  {s['pnl']:>+10.2f}  {s['final']:>10.2f}  {s['pct']:>+7.1f}%  {s['max_dd']:>5.1f}%{mk}")

print()
for t_,s,lbl in all_s[1:]:
    diff_pnl = s["pnl"] - sD["pnl"]
    diff_dd  = s["max_dd"] - sD["max_dd"]
    diff_tr  = s["trades"] - sD["trades"]
    print(f"  D→{lbl[:2]}: trades {sD['trades']}{diff_tr:+d}  PnL {sD['pnl']:+,.0f}→{s['pnl']:+,.0f} ({diff_pnl:+,.0f})  MaxDD {sD['max_dd']}%→{s['max_dd']}% ({diff_dd:+.1f}pp)")

# ── Deep-dive: weak months in Version D ───────────────────────────────────────
monthly_D = defaultdict(list)
for t in tD:
    ym = str(t["open_time"])[:7]
    monthly_D[ym].append(t)

weak_months = []
for ym in sorted(monthly_D.keys()):
    ts_ = monthly_D[ym]
    wr  = len([t for t in ts_ if t["pnl_usd"]>0])/len(ts_)*100
    pnl = sum(t["pnl_usd"] for t in ts_)
    if wr < 35 or pnl < 0:
        weak_months.append(ym)

print()
print("="*W)
print("  WEAK MONTH DEEP-DIVE (WR<35% or negative PnL) — Version D trades")
print("="*W)

for ym in weak_months:
    ts_ = monthly_D[ym]
    wins = [t for t in ts_ if t["pnl_usd"]>0]
    wr   = round(len(wins)/len(ts_)*100,1)
    pnl  = round(sum(t["pnl_usd"] for t in ts_),2)
    avgr = round(sum(t["r_multiple"] for t in ts_)/len(ts_),2)
    longs = [t for t in ts_ if t["direction"]=="long"]
    shorts= [t for t in ts_ if t["direction"]=="short"]

    print(f"\n  {'='*95}")
    print(f"  {ym}  |  WR={wr}%  AvgR={avgr:+.2f}R  PnL=${pnl:+,.2f}  "
          f"Longs={len(longs)}  Shorts={len(shorts)}")
    print(f"  {'='*95}")
    print(f"  {'#':>3}  {'Date':>10}  {'Dir':>5}  {'Entry$':>9}  {'ADX':>5}  "
          f"{'PDI':>5}  {'MDI':>5}  {'RSI':>5}  {'EMA50slp':>9}  "
          f">EMA50  >EMA200  {'Reason':>22}  {'Exit':>10}  {'R':>6}  {'PnL$':>8}")
    print(f"  {'-'*110}")
    for idx2, t in enumerate(ts_, 1):
        mk   = "W" if t["pnl_usd"]>0 else "L"
        e50  = "YES" if t.get("above_ema50")  else "no"
        e200 = "YES" if t.get("above_ema200") else "no"
        slope_str = f"{t.get('slope',0):+.0f}"
        print(f"  {idx2:>3}  {str(t['open_time'])[:10]:>10}  {t['direction'].upper():>5}"
              f"  {t['entry']:>9,.0f}  {t['adx']:>5.1f}"
              f"  {t['pdi']:>5.1f}  {t['mdi']:>5.1f}  {t['rsi']:>5.1f}"
              f"  {slope_str:>9}  {e50:>6}  {e200:>7}"
              f"  {t.get('signal_reason','')[:22]:>22}"
              f"  {t['exit_reason']:>10}  {t['r_multiple']:>+5.2f}R  ${t['pnl_usd']:>+7.2f} {mk}")

    # Pattern analysis
    sl_exits   = [t for t in ts_ if t["exit_reason"]=="SL"]
    trail_exits= [t for t in ts_ if t["exit_reason"]=="TRAIL_SL"]
    avg_adx    = sum(t["adx"] for t in ts_)/len(ts_)
    avg_rsi    = sum(t["rsi"] for t in ts_)/len(ts_)
    shorts_pct = len(shorts)/len(ts_)*100
    print(f"\n  Pattern: SL={len(sl_exits)}  TRAIL_SL={len(trail_exits)}  "
          f"AvgADX={avg_adx:.1f}  AvgRSI={avg_rsi:.1f}  "
          f"Shorts={shorts_pct:.0f}% of trades")

    # Check: would EMA50 slope filter have skipped any of these trades?
    slope_blocked = [t for t in ts_ if
                     (t["direction"]=="long"  and t.get("slope",0) <= 0) or
                     (t["direction"]=="short" and t.get("slope",0) >= 0)]
    slope_kept    = [t for t in ts_ if t not in slope_blocked]
    if slope_blocked:
        blocked_pnl = sum(t["pnl_usd"] for t in slope_blocked)
        kept_pnl    = sum(t["pnl_usd"] for t in slope_kept)
        blocked_wr  = len([t for t in slope_blocked if t["pnl_usd"]>0])/len(slope_blocked)*100
        print(f"  EMA50 slope filter would block {len(slope_blocked)} trades "
              f"(WR={blocked_wr:.0f}%, PnL=${blocked_pnl:+.2f}) "
              f"→ kept {len(slope_kept)} trades (PnL=${kept_pnl:+.2f})")

# ── Monthly comparison D vs E-versions ────────────────────────────────────────
print()
print("="*W)
print("  MONTHLY COMPARISON — D vs best E version (focus on weak months)")
print("="*W)

monthly_E1=defaultdict(list); monthly_E2=defaultdict(list); monthly_E3=defaultdict(list)
for t in tE1: monthly_E1[str(t["open_time"])[:7]].append(t)
for t in tE2: monthly_E2[str(t["open_time"])[:7]].append(t)
for t in tE3: monthly_E3[str(t["open_time"])[:7]].append(t)

all_months = sorted(set(list(monthly_D.keys())+list(monthly_E3.keys())))
print(f"  {'Month':>7}  {'D:WR%':>6}  {'D:PnL$':>8}  {'E1:WR%':>7}  {'E1:PnL$':>9}  "
      f"{'E2:WR%':>7}  {'E2:PnL$':>9}  {'E3:WR%':>7}  {'E3:PnL$':>9}")
print(f"  {'-'*90}")
for ym in all_months:
    def m_stats(monthly):
        ts_=monthly.get(ym,[])
        if not ts_: return 0.0, 0.0
        return round(len([t for t in ts_ if t["pnl_usd"]>0])/len(ts_)*100,1), round(sum(t["pnl_usd"] for t in ts_),2)
    dwr,dpnl   = m_stats(monthly_D)
    e1wr,e1pnl = m_stats(monthly_E1)
    e2wr,e2pnl = m_stats(monthly_E2)
    e3wr,e3pnl = m_stats(monthly_E3)
    weak_flag = "  !" if ym in weak_months else ""
    print(f"  {ym:>7}  {dwr:>5.1f}%  {dpnl:>+8.2f}  {e1wr:>6.1f}%  {e1pnl:>+9.2f}  "
          f"{e2wr:>6.1f}%  {e2pnl:>+9.2f}  {e3wr:>6.1f}%  {e3pnl:>+9.2f}{weak_flag}")
print()
