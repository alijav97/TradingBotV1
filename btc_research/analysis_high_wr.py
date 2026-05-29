"""
Deep-dive analysis of high-WR months (WR >= 55%).
Run: python btc_research/analysis_high_wr.py
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
    print("ERROR: No BTC data.")
    sys.exit(1)

def calc_adx(df, period=14):
    h = df["high"].astype(float); l = df["low"].astype(float); c = df["close"].astype(float)
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    h_diff = h.diff(); l_diff = l.diff()
    plus_dm  = h_diff.where((h_diff > 0) & (h_diff > -l_diff), 0.0)
    minus_dm = (-l_diff).where((-l_diff > 0) & (-l_diff > h_diff), 0.0)
    span = 2 * period - 1
    atr_w = tr.ewm(span=span, adjust=False).mean()
    pdm_w = plus_dm.ewm(span=span, adjust=False).mean()
    mdm_w = minus_dm.ewm(span=span, adjust=False).mean()
    pdi = 100 * pdm_w / atr_w
    mdi = 100 * mdm_w / atr_w
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, float("nan"))
    return dx.ewm(span=span, adjust=False).mean().fillna(0).values

def simulate(strategy, df_btc, trail_atr_mult=2.0, adx_threshold=20):
    h   = df_btc["high"].astype(float).values
    l   = df_btc["low"].astype(float).values
    c   = df_btc["close"].astype(float).values
    ts  = pd.to_datetime(df_btc.index)

    tr_s = pd.concat([
        df_btc["high"].astype(float) - df_btc["low"].astype(float),
        (df_btc["high"].astype(float) - df_btc["close"].astype(float).shift(1)).abs(),
        (df_btc["low"].astype(float)  - df_btc["close"].astype(float).shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_arr = tr_s.rolling(14).mean().bfill().values
    adx_arr = calc_adx(df_btc, 14)
    ema50   = df_btc["close"].astype(float).ewm(span=50,  adjust=False).mean().values
    ema200  = df_btc["close"].astype(float).ewm(span=200, adjust=False).mean().values

    balance = float(cfg.STARTING_BALANCE)
    trades  = []
    open_t  = None

    for i in range(220, len(df_btc)):
        bar_time = ts[i]
        hr  = bar_time.hour
        bh_ = float(h[i]); bl_ = float(l[i]); bc_ = float(c[i])

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
                    pnl = risk_u * open_t["tp1_rr"]; r_ = open_t["tp1_rr"]; ex = "TP1_partial"
                    open_t["tp1_hit"]    = True
                    open_t["sl"]         = entry_
                    open_t["trail_peak"] = bh_ if long_ else bl_
                    balance += pnl
                    open_t["pnl_running"] = round(pnl, 2)
                    open_t["r_running"]   = r_
                    continue
                elif age >= cfg.MAX_HOLD_BARS:
                    pnl = (bc_ - entry_ if long_ else entry_ - bc_) * open_t["lots"]
                    r_  = pnl / risk_u if risk_u else 0; ex = "MAX_HOLD"
                else:
                    continue
            else:
                atr_now = float(atr_arr[min(i, len(atr_arr)-1)])
                if long_:
                    if bh_ > open_t["trail_peak"]: open_t["trail_peak"] = bh_
                    open_t["sl"] = max(open_t["trail_peak"] - trail_atr_mult * atr_now, entry_)
                else:
                    if bl_ < open_t["trail_peak"]: open_t["trail_peak"] = bl_
                    open_t["sl"] = min(open_t["trail_peak"] + trail_atr_mult * atr_now, entry_)
                new_sl = open_t["sl"]
                hit_sl = (bl_ <= new_sl) if long_ else (bh_ >= new_sl)
                if hit_sl:
                    dist_run = abs(new_sl - entry_)
                    r_ = dist_run / sl_dist if sl_dist else 0
                    pnl = risk_u * r_; ex = "TRAIL_SL"
                elif age >= cfg.MAX_HOLD_BARS:
                    dist_run = abs(bc_ - entry_)
                    r_ = dist_run / sl_dist if sl_dist else 0
                    pnl = risk_u * r_; ex = "MAX_HOLD"
                else:
                    continue

            balance += pnl
            open_t["pnl_usd"]       = round(open_t.get("pnl_running", 0) + pnl, 2)
            open_t["r_multiple"]    = round(open_t.get("r_running",   0) + r_,  2)
            open_t["balance_after"] = round(balance, 2)
            open_t["exit_reason"]   = ex
            trades.append(open_t)
            open_t = None

        if open_t is not None:
            continue

        ks = cfg.KZ_START_UTC; ke = cfg.KZ_END_UTC
        in_kz = (ks < ke and ks <= hr < ke) or (ks > ke and (hr >= ks or hr < ke))
        if not in_kz: continue
        if adx_threshold > 0 and float(adx_arr[i]) < adx_threshold: continue

        win = df_btc.iloc[max(0, i-220):i+1]
        for direction in ("long", "short"):
            sig = strategy.generate_signal(win, bar_time, direction)
            if sig.get("signal"):
                sl_d = abs(float(sig["entry"]) - float(sig["sl"]))
                if sl_d <= 0: continue
                ru   = round(balance * cfg.RISK_PCT, 2)
                lots = ru / sl_d
                tp1r = sig.get("tp1_rr", cfg.TP1_RR)
                tp2r = sig.get("tp2_rr", cfg.TP2_RR)
                tp1  = float(sig["entry"]) + tp1r*sl_d if direction=="long" else float(sig["entry"]) - tp1r*sl_d
                tp2  = float(sig["entry"]) + tp2r*sl_d if direction=="long" else float(sig["entry"]) - tp2r*sl_d
                open_t = {
                    "open_time": bar_time, "open_idx": i,
                    "direction": direction,
                    "entry":   float(sig["entry"]),
                    "sl":      float(sig["sl"]),
                    "orig_sl": float(sig["sl"]),
                    "tp1": tp1, "tp2": tp2, "tp1_rr": tp1r, "tp2_rr": tp2r,
                    "lots": lots, "risk_usd": ru,
                    "signal_reason": sig.get("reason", ""),
                    "trail_peak": 0.0, "tp1_hit": False,
                    "pnl_running": 0.0, "r_running": 0.0,
                    "adx_at_entry":  round(float(adx_arr[i]), 1),
                    "atr_at_entry":  round(float(atr_arr[i]), 0),
                    "above_ema50":   bc_ > float(ema50[i]),
                    "above_ema200":  bc_ > float(ema200[i]),
                    "pnl_usd": 0.0,
                }
                break
    return trades

# ── Run ───────────────────────────────────────────────────────────────────────
strat  = CombinedStrategy(atr_multiplier=1.2, close_zone=0.45, range_bars=6)
trades = simulate(strat, df_btc)

# ── Monthly stats ─────────────────────────────────────────────────────────────
monthly = defaultdict(list)
for t in trades:
    ym = str(t["open_time"])[:7]
    monthly[ym].append(t)

W = 125
print("=" * W)
print("MONTHLY BREAKDOWN  |  Trail 2.0xATR + ADX>20  |  $500 start, 2% risk compounding")
print("=" * W)
print(f"{'Month':>7}  {'#':>3}  {'WR%':>5}  {'AvgR':>6}  {'PnL$':>8}  {'Bal$':>8}  {'AvgADX':>7}  {'AvgATR$':>8}  {'BTC Range':>14}  {'L/S':>7}  {'LWR':>4}  {'SWR':>4}")
print("-" * W)

high_wr_data = []
for ym in sorted(monthly.keys()):
    ts_ = monthly[ym]
    wins  = [t for t in ts_ if t["pnl_usd"] > 0]
    wr    = round(len(wins)/len(ts_)*100, 1)
    avgr  = round(sum(t["r_multiple"] for t in ts_)/len(ts_), 2)
    pnl   = round(sum(t["pnl_usd"] for t in ts_), 2)
    bal   = round(ts_[-1]["balance_after"], 2)
    adxv  = round(sum(t["adx_at_entry"] for t in ts_)/len(ts_), 1)
    atrv  = round(sum(t["atr_at_entry"] for t in ts_)/len(ts_), 0)
    longs  = [t for t in ts_ if t["direction"]=="long"]
    shorts = [t for t in ts_ if t["direction"]=="short"]
    lwr    = round(len([t for t in longs  if t["pnl_usd"]>0])/len(longs)*100)  if longs  else 0
    swr    = round(len([t for t in shorts if t["pnl_usd"]>0])/len(shorts)*100) if shorts else 0
    ls     = f"{len(longs)}L/{len(shorts)}S"

    try:
        btc_m = df_btc[pd.to_datetime(df_btc.index).strftime("%Y-%m") == ym]
        lo_ = int(btc_m["low"].astype(float).min())
        hi_ = int(btc_m["high"].astype(float).max())
        pr  = f"${lo_//1000}K-${hi_//1000}K"
    except Exception:
        pr = "N/A"

    tag = "  <<< HIGH WR" if wr >= 55 else ("  !WEAK" if wr < 35 else "")
    print(f"{ym:>7}  {len(ts_):>3}  {wr:>4.1f}%  {avgr:>+5.2f}R  {pnl:>+8.2f}  {bal:>8.2f}"
          f"  {adxv:>7.1f}  {atrv:>8.0f}  {pr:>14}  {ls:>7}  {lwr:>3}%  {swr:>3}%{tag}")

    if wr >= 55:
        high_wr_data.append({
            "ym": ym, "trades": ts_, "wr": wr, "avgr": avgr,
            "pnl": pnl, "adx": adxv, "atr": atrv, "pr": pr,
            "longs": longs, "shorts": shorts, "lwr": lwr, "swr": swr
        })

total = len(trades)
wt    = [t for t in trades if t["pnl_usd"] > 0]
print("-" * W)
print(f"{'TOTAL':>7}  {total:>3}  {len(wt)/total*100:>4.1f}%"
      f"  {sum(t['r_multiple'] for t in trades)/total:>+5.2f}R"
      f"  {sum(t['pnl_usd'] for t in trades):>+8.2f}  {trades[-1]['balance_after']:>8.2f}")

# ── Deep-dive each high-WR month ──────────────────────────────────────────────
print()
print("=" * W)
print("DEEP-DIVE: HIGH-WR MONTHS (WR >= 55%)")
print("=" * W)

for m in high_wr_data:
    ym = m["ym"]; ts_ = m["trades"]
    print(f"\n{'='*80}")
    print(f"  {ym}  |  WR={m['wr']}%  AvgR={m['avgr']:+.2f}R  PnL=${m['pnl']:+,.2f}"
          f"  AvgADX={m['adx']}  AvgATR=${m['atr']:,.0f}  BTC={m['pr']}")
    print(f"  Longs={len(m['longs'])} ({m['lwr']}% WR)   Shorts={len(m['shorts'])} ({m['swr']}% WR)")
    print(f"  {'#':>3}  {'Date':>10}  {'Dir':>5}  {'Entry$':>9}  {'ADX':>5}  {'ATR$':>7}"
          f"  {'>EMA50':>6}  {'>EMA200':>7}  {'Reason':>28}  {'Exit':>11}  {'R':>6}  {'PnL$':>8}")
    print(f"  {'-'*108}")

    exits = defaultdict(int)
    for idx2, t in enumerate(ts_, 1):
        sub  = t.get("signal_reason","")[:28]
        mk   = "W" if t["pnl_usd"] > 0 else "L"
        e50  = "YES" if t.get("above_ema50")  else "no"
        e200 = "YES" if t.get("above_ema200") else "no"
        print(f"  {idx2:>3}  {str(t['open_time'])[:10]:>10}  {t['direction'].upper():>5}"
              f"  {t['entry']:>9,.0f}  {t['adx_at_entry']:>5.1f}"
              f"  {t['atr_at_entry']:>7,.0f}  {e50:>6}  {e200:>7}  {sub:>28}"
              f"  {t['exit_reason']:>11}  {t['r_multiple']:>+5.2f}R  ${t['pnl_usd']:>+7.2f} {mk}")
        exits[t["exit_reason"]] += 1

    print(f"\n  Exits: " + "  |  ".join(f"{k}:{v}" for k,v in sorted(exits.items())))
    adx_vals = sorted([t["adx_at_entry"] for t in ts_])
    atr_vals = sorted([t["atr_at_entry"] for t in ts_])
    print(f"  ADX at entry: min={adx_vals[0]:.1f}  median={adx_vals[len(adx_vals)//2]:.1f}  max={adx_vals[-1]:.1f}")
    print(f"  ATR at entry: min=${atr_vals[0]:,.0f}  median=${atr_vals[len(atr_vals)//2]:,.0f}  max=${atr_vals[-1]:,.0f}")
    a50 = [t for t in ts_ if t.get("above_ema50")]
    a50w= [t for t in a50 if t["pnl_usd"]>0]
    print(f"  Above EMA50 : {len(a50)}/{len(ts_)} trades  WR={len(a50w)/len(a50)*100:.0f}%" if a50 else "  Above EMA50 : 0 trades")

# ── Cross-month pattern summary ────────────────────────────────────────────────
print()
print("=" * W)
print("CROSS-MONTH PATTERN SUMMARY — What separates high-WR from low-WR months?")
print("=" * W)

if high_wr_data:
    all_hw = [t for m in high_wr_data for t in m["trades"]]
    wins_hw = [t for t in all_hw if t["pnl_usd"] > 0]

    low_wr_months_trades = []
    for ym in sorted(monthly.keys()):
        ts_ = monthly[ym]
        if not ts_: continue
        wr_ = len([t for t in ts_ if t["pnl_usd"]>0])/len(ts_)*100
        if wr_ < 40:
            low_wr_months_trades.extend(ts_)

    print(f"\n  HIGH-WR months ({len(high_wr_data)} months, {len(all_hw)} trades):")
    print(f"    WR={len(wins_hw)/len(all_hw)*100:.1f}%  AvgR={sum(t['r_multiple'] for t in all_hw)/len(all_hw):+.2f}R"
          f"  AvgADX={sum(t['adx_at_entry'] for t in all_hw)/len(all_hw):.1f}"
          f"  AvgATR=${sum(t['atr_at_entry'] for t in all_hw)/len(all_hw):,.0f}")
    lhw = [t for t in all_hw if t["direction"]=="long"]
    shw = [t for t in all_hw if t["direction"]=="short"]
    print(f"    Longs={len(lhw)} ({len([t for t in lhw if t['pnl_usd']>0])/len(lhw)*100:.0f}% WR)"
          f"  Shorts={len(shw)} ({len([t for t in shw if t['pnl_usd']>0])/len(shw)*100:.0f}% WR)" if lhw and shw else "")
    exits_hw = defaultdict(int)
    for t in all_hw: exits_hw[t["exit_reason"]] += 1
    print(f"    Exits: " + "  |  ".join(f"{k}:{v}" for k,v in sorted(exits_hw.items())))

    if low_wr_months_trades:
        wins_lw = [t for t in low_wr_months_trades if t["pnl_usd"]>0]
        print(f"\n  LOW-WR months (<40% WR, {len(low_wr_months_trades)} trades):")
        print(f"    WR={len(wins_lw)/len(low_wr_months_trades)*100:.1f}%"
              f"  AvgR={sum(t['r_multiple'] for t in low_wr_months_trades)/len(low_wr_months_trades):+.2f}R"
              f"  AvgADX={sum(t['adx_at_entry'] for t in low_wr_months_trades)/len(low_wr_months_trades):.1f}"
              f"  AvgATR=${sum(t['atr_at_entry'] for t in low_wr_months_trades)/len(low_wr_months_trades):,.0f}")
        llw = [t for t in low_wr_months_trades if t["direction"]=="long"]
        slw = [t for t in low_wr_months_trades if t["direction"]=="short"]
        print(f"    Longs={len(llw)} ({len([t for t in llw if t['pnl_usd']>0])/len(llw)*100:.0f}% WR)"
              f"  Shorts={len(slw)} ({len([t for t in slw if t['pnl_usd']>0])/len(slw)*100:.0f}% WR)" if llw and slw else "")

    # EMA200 alignment across ALL trades
    print()
    above200_all = [t for t in trades if t.get("above_ema200")]
    below200_all = [t for t in trades if not t.get("above_ema200")]
    a200w = [t for t in above200_all if t["pnl_usd"]>0]
    b200w = [t for t in below200_all if t["pnl_usd"]>0]
    print(f"  EMA200 alignment (ALL 2yr trades):")
    print(f"    Above EMA200: {len(above200_all)} trades  WR={len(a200w)/len(above200_all)*100:.0f}%  AvgR={sum(t['r_multiple'] for t in above200_all)/len(above200_all):+.2f}R" if above200_all else "")
    print(f"    Below EMA200: {len(below200_all)} trades  WR={len(b200w)/len(below200_all)*100:.0f}%  AvgR={sum(t['r_multiple'] for t in below200_all)/len(below200_all):+.2f}R" if below200_all else "")

print()
