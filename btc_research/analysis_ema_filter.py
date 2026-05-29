"""
btc_research/analysis_ema_filter.py
Compares 3 versions of the strategy:
  A) Baseline   : Trail 2.0xATR + ADX>20
  B) EMA200     : A + only longs above EMA200, only shorts below EMA200
  C) Dynamic    : B + risk 3% when ADX>28, else 2%

Run from project root: python btc_research/analysis_ema_filter.py
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
    print("ERROR: No BTC data. Make sure MT5 is running.")
    sys.exit(1)

# Ensure datetime index (fetcher returns a "time" column, not index)
if "time" in df_btc.columns:
    df_btc = df_btc.set_index(pd.to_datetime(df_btc["time"], utc=True)).drop(columns=["time"])
elif not isinstance(df_btc.index, pd.DatetimeIndex):
    df_btc.index = pd.to_datetime(df_btc.index, utc=True)

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_adx(df, period=14):
    h = df["high"].astype(float); l = df["low"].astype(float); c = df["close"].astype(float)
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    hd = h.diff(); ld = l.diff()
    pdm = hd.where((hd > 0) & (hd > -ld), 0.0)
    mdm = (-ld).where((-ld > 0) & (-ld > hd), 0.0)
    sp  = 2 * period - 1
    aw  = tr.ewm(span=sp, adjust=False).mean()
    pw  = pdm.ewm(span=sp, adjust=False).mean()
    mw  = mdm.ewm(span=sp, adjust=False).mean()
    pdi = 100 * pw / aw; mdi = 100 * mw / aw
    dx  = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, float("nan"))
    return dx.ewm(span=sp, adjust=False).mean().fillna(0).values

# Pre-compute all indicator arrays once
_c      = df_btc["close"].astype(float)
_h      = df_btc["high"].astype(float)
_l      = df_btc["low"].astype(float)
_tr     = pd.concat([_h-_l,(_h-_c.shift(1)).abs(),(_l-_c.shift(1)).abs()],axis=1).max(axis=1)
ATR_ARR = _tr.rolling(14).mean().bfill().values
ADX_ARR = calc_adx(df_btc, 14)
EMA50   = _c.ewm(span=50,  adjust=False).mean().values
EMA200  = _c.ewm(span=200, adjust=False).mean().values
TS      = df_btc.index
H_ARR   = _h.values
L_ARR   = _l.values
C_ARR   = _c.values

# ── Core simulator ────────────────────────────────────────────────────────────
def simulate(strategy,
             trail_atr_mult  = 2.0,
             adx_threshold   = 20,
             use_ema200_filter = False,
             dynamic_risk    = False):
    """
    use_ema200_filter : skip longs when price < EMA200, skip shorts when price > EMA200
    dynamic_risk      : risk 3% if ADX > 28, else risk 2%
    """
    balance = float(cfg.STARTING_BALANCE)
    trades  = []
    open_t  = None

    for i in range(220, len(df_btc)):
        bar_time = TS[i]
        hr  = bar_time.hour
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
                elif age >= cfg.MAX_HOLD_BARS:
                    pnl = (bc_ - entry_ if long_ else entry_ - bc_) * open_t["lots"]
                    r_  = pnl / risk_u if risk_u else 0; ex = "MAX_HOLD"
                else:
                    continue
            else:
                atr_now = float(ATR_ARR[min(i, len(ATR_ARR)-1)])
                if long_:
                    if bh_ > open_t["trail_peak"]: open_t["trail_peak"] = bh_
                    open_t["sl"] = max(open_t["trail_peak"] - trail_atr_mult * atr_now, entry_)
                else:
                    if bl_ < open_t["trail_peak"]: open_t["trail_peak"] = bl_
                    open_t["sl"] = min(open_t["trail_peak"] + trail_atr_mult * atr_now, entry_)
                new_sl = open_t["sl"]
                if (long_ and bl_ <= new_sl) or (not long_ and bh_ >= new_sl):
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

        # ── Entry filters ─────────────────────────────────────────────────────
        ks = cfg.KZ_START_UTC; ke = cfg.KZ_END_UTC
        in_kz = (ks < ke and ks <= hr < ke) or (ks > ke and (hr >= ks or hr < ke))
        if not in_kz: continue

        adx_now  = float(ADX_ARR[i])
        if adx_now < adx_threshold: continue

        ema200_now = float(EMA200[i])

        # ── Signal generation ─────────────────────────────────────────────────
        win = df_btc.iloc[max(0, i-220):i+1]
        for direction in ("long", "short"):
            # EMA200 trend filter
            if use_ema200_filter:
                if direction == "long"  and bc_ < ema200_now: continue
                if direction == "short" and bc_ > ema200_now: continue

            sig = strategy.generate_signal(win, bar_time, direction)
            if sig.get("signal"):
                sl_d = abs(float(sig["entry"]) - float(sig["sl"]))
                if sl_d <= 0: continue

                # Dynamic risk
                risk_pct = (0.03 if (dynamic_risk and adx_now > 28) else cfg.RISK_PCT)
                ru   = round(balance * risk_pct, 2)
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
                    "pnl_running": 0.0, "r_running": 0.0, "pnl_usd": 0.0,
                    "adx_at_entry":    round(adx_now, 1),
                    "atr_at_entry":    round(float(ATR_ARR[i]), 0),
                    "above_ema200":    bc_ > ema200_now,
                    "above_ema50":     bc_ > float(EMA50[i]),
                    "risk_pct_used":   risk_pct,
                }
                break
    return trades

# ── Stats helper ───────────────────────────────────────────────────────────────
def stats(trades):
    if not trades:
        return {"trades": 0, "wr": 0.0, "avg_r": 0.0, "pnl": 0.0,
                "final": float(cfg.STARTING_BALANCE), "max_dd": 0.0, "pct": 0.0}
    total = len(trades)
    wins  = [t for t in trades if t["pnl_usd"] > 0]
    pnl   = sum(t["pnl_usd"] for t in trades)
    avg_r = sum(t["r_multiple"] for t in trades) / total

    # Max drawdown
    bal = float(cfg.STARTING_BALANCE)
    peak = bal; max_dd = 0.0
    for t in trades:
        bal = t["balance_after"]
        if bal > peak: peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd: max_dd = dd

    return {
        "trades": total,
        "wr":     round(len(wins)/total*100, 1),
        "avg_r":  round(avg_r, 2),
        "pnl":    round(pnl, 2),
        "final":  round(trades[-1]["balance_after"], 2),
        "max_dd": round(max_dd, 1),
        "pct":    round((trades[-1]["balance_after"]/cfg.STARTING_BALANCE - 1)*100, 1),
    }

# ── Monthly breakdown helper ──────────────────────────────────────────────────
def monthly_breakdown(trades, label):
    if not trades:
        print(f"\n  No trades for {label}")
        return
    monthly = defaultdict(list)
    for t in trades:
        ym = str(t["open_time"])[:7]
        monthly[ym].append(t)

    print(f"\n{'='*110}")
    print(f"  MONTHLY BREAKDOWN — {label}")
    print(f"{'='*110}")
    print(f"  {'Month':>7}  {'#':>3}  {'WR%':>5}  {'AvgR':>6}  {'PnL$':>8}  {'Bal$':>9}  {'AvgADX':>7}  {'L/S':>7}  {'LWR':>4}  {'SWR':>4}")
    print(f"  {'-'*95}")

    running_best  = 0
    peak_bal      = float(cfg.STARTING_BALANCE)
    for ym in sorted(monthly.keys()):
        ts_   = monthly[ym]
        wins  = [t for t in ts_ if t["pnl_usd"] > 0]
        wr    = round(len(wins)/len(ts_)*100, 1)
        avgr  = round(sum(t["r_multiple"] for t in ts_)/len(ts_), 2)
        pnl   = round(sum(t["pnl_usd"] for t in ts_), 2)
        bal   = round(ts_[-1]["balance_after"], 2)
        adxv  = round(sum(t["adx_at_entry"] for t in ts_)/len(ts_), 1)
        longs  = [t for t in ts_ if t["direction"]=="long"]
        shorts = [t for t in ts_ if t["direction"]=="short"]
        lwr    = round(len([t for t in longs  if t["pnl_usd"]>0])/len(longs)*100)  if longs  else 0
        swr    = round(len([t for t in shorts if t["pnl_usd"]>0])/len(shorts)*100) if shorts else 0
        ls     = f"{len(longs)}L/{len(shorts)}S"
        tag    = "  <<< HIGH WR" if wr >= 55 else ("  !WEAK" if wr < 35 else "")
        print(f"  {ym:>7}  {len(ts_):>3}  {wr:>4.1f}%  {avgr:>+5.2f}R  {pnl:>+8.2f}  {bal:>9.2f}"
              f"  {adxv:>7.1f}  {ls:>7}  {lwr:>3}%  {swr:>3}%{tag}")

    total = len(trades)
    wt    = [t for t in trades if t["pnl_usd"] > 0]
    print(f"  {'-'*95}")
    print(f"  {'TOTAL':>7}  {total:>3}  {len(wt)/total*100:>4.1f}%"
          f"  {sum(t['r_multiple'] for t in trades)/total:>+5.2f}R"
          f"  {sum(t['pnl_usd'] for t in trades):>+8.2f}  {trades[-1]['balance_after']:>9.2f}")

# ── Run all three versions ────────────────────────────────────────────────────
strat = CombinedStrategy(atr_multiplier=1.2, close_zone=0.45, range_bars=6)

print("Running simulations — please wait...")
print("  [A] Baseline (Trail 2.0xATR + ADX>20)...")
trades_A = simulate(strat, trail_atr_mult=2.0, adx_threshold=20,
                    use_ema200_filter=False, dynamic_risk=False)

print("  [B] + EMA200 filter (longs above EMA200 only, shorts below only)...")
trades_B = simulate(strat, trail_atr_mult=2.0, adx_threshold=20,
                    use_ema200_filter=True, dynamic_risk=False)

print("  [C] + EMA200 filter + Dynamic risk (3% when ADX>28, 2% otherwise)...")
trades_C = simulate(strat, trail_atr_mult=2.0, adx_threshold=20,
                    use_ema200_filter=True, dynamic_risk=True)

sA = stats(trades_A)
sB = stats(trades_B)
sC = stats(trades_C)

# ── Head-to-head summary ──────────────────────────────────────────────────────
W = 110
print()
print("=" * W)
print("  HEAD-TO-HEAD COMPARISON  |  $500 start, 2% base risk, compounding")
print("=" * W)
print(f"  {'Version':<42}  {'Trades':>6}  {'WR%':>5}  {'AvgR':>6}  {'PnL$':>8}  {'Final$':>8}  {'Return%':>8}  {'MaxDD%':>7}")
print(f"  {'-'*(W-2)}")

rows = [
    ("A  Baseline  (Trail+ADX>20)",           sA),
    ("B  +EMA200 filter",                      sB),
    ("C  +EMA200 filter +Dynamic risk",        sC),
]
best_pnl = max(s["pnl"] for _, s in rows)
for label, s in rows:
    tag = "  ← BEST PnL" if s["pnl"] == best_pnl else ""
    print(f"  {label:<42}  {s['trades']:>6}  {s['wr']:>4.1f}%  {s['avg_r']:>+5.2f}R"
          f"  {s['pnl']:>+8.2f}  {s['final']:>8.2f}  {s['pct']:>+7.1f}%  {s['max_dd']:>6.1f}%{tag}")

print()
print("  IMPROVEMENT A→B (EMA200 filter effect):")
if sA["trades"] > 0 and sB["trades"] > 0:
    print(f"    Trades : {sA['trades']} → {sB['trades']} ({sB['trades']-sA['trades']:+d} fewer trades after filter)")
    print(f"    WR     : {sA['wr']}% → {sB['wr']}%  ({sB['wr']-sA['wr']:+.1f}pp)")
    print(f"    AvgR   : {sA['avg_r']:+.2f}R → {sB['avg_r']:+.2f}R  ({sB['avg_r']-sA['avg_r']:+.2f}R)")
    print(f"    PnL    : ${sA['pnl']:+,.2f} → ${sB['pnl']:+,.2f}  ({sB['pnl']-sA['pnl']:+.2f})")
    print(f"    MaxDD  : {sA['max_dd']}% → {sB['max_dd']}%  ({sB['max_dd']-sA['max_dd']:+.1f}pp)")

print()
print("  IMPROVEMENT B→C (Dynamic risk effect):")
if sB["trades"] > 0 and sC["trades"] > 0:
    print(f"    PnL    : ${sB['pnl']:+,.2f} → ${sC['pnl']:+,.2f}  ({sC['pnl']-sB['pnl']:+.2f})")
    print(f"    MaxDD  : {sB['max_dd']}% → {sC['max_dd']}%  ({sC['max_dd']-sB['max_dd']:+.1f}pp)")
    print(f"    Return : {sB['pct']:+.1f}% → {sC['pct']:+.1f}%")

# ── EMA200 alignment stats across all trades ──────────────────────────────────
print()
print("=" * W)
print("  EMA200 ALIGNMENT STATS (from Baseline trades)")
print("=" * W)
above = [t for t in trades_A if t.get("above_ema200")]
below = [t for t in trades_A if not t.get("above_ema200")]
aw    = [t for t in above if t["pnl_usd"]>0]
bw    = [t for t in below if t["pnl_usd"]>0]
print(f"  Above EMA200 : {len(above)} trades  WR={len(aw)/len(above)*100:.1f}%  AvgR={sum(t['r_multiple'] for t in above)/len(above):+.2f}R  PnL=${sum(t['pnl_usd'] for t in above):+,.2f}" if above else "")
print(f"  Below EMA200 : {len(below)} trades  WR={len(bw)/len(below)*100:.1f}%  AvgR={sum(t['r_multiple'] for t in below)/len(below):+.2f}R  PnL=${sum(t['pnl_usd'] for t in below):+,.2f}" if below else "")

# Long/Short breakdown for above/below EMA200
if above:
    al = [t for t in above if t["direction"]=="long"]
    as_ = [t for t in above if t["direction"]=="short"]
    print(f"    ↳ Above : Longs={len(al)} ({len([t for t in al if t['pnl_usd']>0])/len(al)*100:.0f}% WR)"
          f"  Shorts={len(as_)} ({len([t for t in as_ if t['pnl_usd']>0])/len(as_)*100:.0f}% WR)" if al and as_
          else f"    ↳ Above : Longs={len(al)}  Shorts={len(as_)}")
if below:
    bl2 = [t for t in below if t["direction"]=="long"]
    bs2 = [t for t in below if t["direction"]=="short"]
    print(f"    ↳ Below : Longs={len(bl2)} ({len([t for t in bl2 if t['pnl_usd']>0])/len(bl2)*100:.0f}% WR)"
          f"  Shorts={len(bs2)} ({len([t for t in bs2 if t['pnl_usd']>0])/len(bs2)*100:.0f}% WR)" if bl2 and bs2
          else f"    ↳ Below : Longs={len(bl2)}  Shorts={len(bs2)}")

# ── Dynamic risk breakdown ─────────────────────────────────────────────────────
if trades_C:
    high_adx = [t for t in trades_C if t.get("risk_pct_used", 0) > 0.02]
    norm_adx  = [t for t in trades_C if t.get("risk_pct_used", 0.02) <= 0.02]
    print()
    print("=" * W)
    print("  DYNAMIC RISK BREAKDOWN (Version C)")
    print("=" * W)
    if high_adx:
        hw = [t for t in high_adx if t["pnl_usd"]>0]
        print(f"  ADX>28 (3% risk) : {len(high_adx)} trades  WR={len(hw)/len(high_adx)*100:.1f}%"
              f"  AvgR={sum(t['r_multiple'] for t in high_adx)/len(high_adx):+.2f}R"
              f"  PnL=${sum(t['pnl_usd'] for t in high_adx):+,.2f}")
    if norm_adx:
        nw = [t for t in norm_adx if t["pnl_usd"]>0]
        print(f"  ADX 20-28 (2%)   : {len(norm_adx)} trades  WR={len(nw)/len(norm_adx)*100:.1f}%"
              f"  AvgR={sum(t['r_multiple'] for t in norm_adx)/len(norm_adx):+.2f}R"
              f"  PnL=${sum(t['pnl_usd'] for t in norm_adx):+,.2f}")

# ── Monthly breakdown for the best version ────────────────────────────────────
best_trades = trades_C if sC["pnl"] >= sB["pnl"] else trades_B
best_label  = "C (EMA200 + Dynamic Risk)" if sC["pnl"] >= sB["pnl"] else "B (EMA200 Filter)"
monthly_breakdown(best_trades, f"BEST VERSION — {best_label}")

# ── Save best trades to CSV ───────────────────────────────────────────────────
out = Path("btc_research/data/trades_ema_filter.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(best_trades).to_csv(out, index=False)
print(f"\nBest-version trades saved -> {out}")
print("Open in Excel to review individual trades.\n")
