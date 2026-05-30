"""
btc_research/run_backtest_btc2.py — BTC Bot 2 filter comparison backtest.

Tests 7 filter combinations on 2yr data to find the optimal setup for
the Asia Night session (02:00-04:00 UTC) using VB + Swing Level strategy.

Filters tested (matching what Bot 1 uses, applied to Bot 2's session):
  A  Baseline      : VB + Swing Level, 02-04 UTC, ADX >= 20 only
  B  + EMA200      : only longs above EMA200, shorts below EMA200
  C  + ADX strict  : ADX >= 25 (tighter trend filter — Asia session may need this)
  D  B + ADX strict: EMA200 + ADX >= 25
  E  B + Flip Risk : EMA200 + flipped risk (3% ADX 20-28, 2% ADX >28)
  F  D + Flip Risk : EMA200 + ADX 25 + flipped risk  (closest to Bot 1 full set)
  G  B + ADX 20    : EMA200 + ADX >= 20  (Bot 1's exact filter set — baseline comparison)

Why test ADX 25 separately?
  Asia Night bars are often part of a sustained overnight trend rather than a
  fresh breakout. A slightly higher ADX threshold may filter out choppy
  consolidation entries. Baseline uses 20 (same as Bot 1) for fair comparison.

Run:
    python btc_research/run_backtest_btc2.py

Output:
  - Summary comparison table (all 7 variants)
  - Monthly breakdown for the best variant
  - Per-strategy breakdown (VB vs Swing Level contribution)
  - Filter recommendation for Bot 2 live deployment
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
    print("ERROR: No BTC data. Make sure MT5 is running (or use cached data).")
    sys.exit(1)

if "time" in df_btc.columns:
    df_btc = df_btc.set_index(pd.to_datetime(df_btc["time"], utc=True)).drop(columns=["time"])
elif not isinstance(df_btc.index, pd.DatetimeIndex):
    df_btc.index = pd.to_datetime(df_btc.index, utc=True)

print(f"Loaded {len(df_btc):,} H1 bars  ({df_btc.index[0].date()} → {df_btc.index[-1].date()})")

# ── Pre-compute all indicators once ───────────────────────────────────────────
_c = df_btc["close"].astype(float)
_h = df_btc["high"].astype(float)
_l = df_btc["low"].astype(float)

_tr     = pd.concat([_h - _l, (_h - _c.shift(1)).abs(), (_l - _c.shift(1)).abs()], axis=1).max(axis=1)
ATR_ARR = _tr.rolling(14).mean().bfill().values
EMA200  = _c.ewm(span=200, adjust=False).mean().values
EMA50   = _c.ewm(span=50,  adjust=False).mean().values
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
    pdi = 100 * pw / aw
    mdi = 100 * mw / aw
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, float("nan"))
    return dx.ewm(span=sp, adjust=False).mean().fillna(0).values


ADX_ARR = _calc_adx(14)

# ── Kill-zone: 02:00-04:00 UTC ────────────────────────────────────────────────
KZ_START = b2cfg.KZ_START_UTC   # 2
KZ_END   = b2cfg.KZ_END_UTC     # 4


# ── Core simulator ─────────────────────────────────────────────────────────────
def simulate(
    strategy,
    adx_threshold:    float = 20.0,
    use_ema200:       bool  = False,
    flipped_risk:     bool  = False,
    trail_atr_mult:   float = 2.0,
    label:            str   = "",
) -> list[dict]:
    """
    Bar-by-bar simulation for Bot 2 session (02-04 UTC).

    adx_threshold  : minimum ADX to take a trade
    use_ema200     : skip longs below EMA200 / shorts above EMA200
    flipped_risk   : 3% risk if ADX 20-28 (early trend), 2% if ADX > 28
    trail_atr_mult : trailing SL after TP1 = this × ATR
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
                    pnl = -risk_u
                    r_  = -1.0
                    ex  = "SL"
                elif hit_tp1:
                    pnl = risk_u * open_t["tp1_rr"]
                    r_  = open_t["tp1_rr"]
                    open_t["tp1_hit"]    = True
                    open_t["sl"]         = entry_
                    open_t["trail_peak"] = bh_ if long_ else bl_
                    balance += pnl
                    open_t["pnl_running"] = round(pnl, 2)
                    open_t["r_running"]   = r_
                    continue
                elif age >= b2cfg.MAX_HOLD_BARS:
                    pnl = (bc_ - entry_ if long_ else entry_ - bc_) * open_t["lots"]
                    r_  = pnl / risk_u if risk_u else 0
                    ex  = "MAX_HOLD"
                else:
                    continue
            else:
                # Trailing SL after TP1
                atr_now = float(ATR_ARR[min(i, len(ATR_ARR) - 1)])
                if long_:
                    if bh_ > open_t["trail_peak"]:
                        open_t["trail_peak"] = bh_
                    open_t["sl"] = max(open_t["trail_peak"] - trail_atr_mult * atr_now, entry_)
                else:
                    if bl_ < open_t["trail_peak"]:
                        open_t["trail_peak"] = bl_
                    open_t["sl"] = min(open_t["trail_peak"] + trail_atr_mult * atr_now, entry_)

                new_sl  = open_t["sl"]
                hit_sl2 = (bl_ <= new_sl) if long_ else (bh_ >= new_sl)

                if hit_sl2:
                    dist_run = abs(new_sl - entry_)
                    r_  = dist_run / sl_dist if sl_dist else 0
                    pnl = risk_u * r_
                    ex  = "TRAIL_SL"
                elif age >= b2cfg.MAX_HOLD_BARS:
                    dist_run = abs(bc_ - entry_)
                    r_  = dist_run / sl_dist if sl_dist else 0
                    pnl = risk_u * r_
                    ex  = "MAX_HOLD"
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

        # ── Entry gate: final hours [1,2,3,8] UTC ────────────────────────────
        if hr not in b2cfg.KZ_HOURS:
            continue

        adx_now    = float(ADX_ARR[i])
        ema200_now = float(EMA200[i])

        if adx_now < adx_threshold:
            continue

        win = df_btc.iloc[max(0, i - 220):i + 1]

        for direction in ("long", "short"):
            # EMA200 filter
            if use_ema200:
                if direction == "long"  and bc_ < ema200_now: continue
                if direction == "short" and bc_ > ema200_now: continue

            sig = strategy.generate_signal(win, bar_time, direction)
            if not sig.get("signal"):
                continue

            sl_d = abs(float(sig["entry"]) - float(sig["sl"]))
            if sl_d <= 0:
                continue

            # Risk sizing
            if flipped_risk:
                risk_pct = b2cfg.RISK_PCT_EARLY_TREND if adx_now <= b2cfg.ADX_EARLY_TREND_MAX else b2cfg.RISK_PCT
            else:
                risk_pct = b2cfg.RISK_PCT

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
                "tp1":           tp1,
                "tp2":           tp2,
                "tp1_rr":        tp1r,
                "tp2_rr":        tp2r,
                "lots":          lots,
                "risk_usd":      ru,
                "risk_pct_used": risk_pct,
                "signal_reason": sig.get("reason", ""),
                "strategy_used": sig.get("strategy_used", ""),
                "trail_peak":    0.0,
                "tp1_hit":       False,
                "pnl_running":   0.0,
                "r_running":     0.0,
                "pnl_usd":       0.0,
                "adx_at_entry":  round(adx_now, 1),
                "atr_at_entry":  round(float(ATR_ARR[i]), 0),
                "above_ema200":  bc_ > ema200_now,
                "above_ema50":   bc_ > float(EMA50[i]),
            }
            break   # one trade per bar

    return trades


def _stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0, "avgr": 0, "pnl": 0, "final_bal": b2cfg.STARTING_BALANCE, "maxdd": 0}
    wins   = [t for t in trades if t["pnl_usd"] > 0]
    wr     = len(wins) / len(trades) * 100
    avgr   = sum(t["r_multiple"] for t in trades) / len(trades)
    pnl    = sum(t["pnl_usd"] for t in trades)
    bals   = [b2cfg.STARTING_BALANCE] + [t["balance_after"] for t in trades]
    peak   = b2cfg.STARTING_BALANCE
    maxdd  = 0.0
    for b in bals:
        if b > peak:
            peak = b
        dd = (peak - b) / peak * 100
        if dd > maxdd:
            maxdd = dd
    return {
        "n":         len(trades),
        "wr":        round(wr, 1),
        "avgr":      round(avgr, 2),
        "pnl":       round(pnl, 2),
        "final_bal": round(bals[-1], 2),
        "maxdd":     round(maxdd, 1),
    }


# ── Run all variants ───────────────────────────────────────────────────────────
strat = VBSwingStrategy(
    atr_multiplier=b2cfg.VB_ATR_MULTIPLIER,
    close_zone=b2cfg.VB_CLOSE_ZONE,
)

print("\nRunning 7 filter variants on 2yr data (02:00-04:00 UTC)...")
print("This may take 1-2 minutes...\n")

variants = [
    # label,                   adx_thresh, ema200, flip_risk
    ("A  Baseline (ADX>=20)",         20,  False,  False),
    ("B  + EMA200",                   20,  True,   False),
    ("C  + ADX>=25",                  25,  False,  False),
    ("D  EMA200 + ADX>=25",           25,  True,   False),
    ("E  EMA200 + Flip Risk",         20,  True,   True),
    ("F  EMA200+ADX25+Flip Risk",     25,  True,   True),
    ("G  EMA200 + ADX>=20 (Bot1 set)",20,  True,   False),
]

results = {}
for label, adx_t, ema200, flip in variants:
    t = simulate(strat, adx_threshold=adx_t, use_ema200=ema200, flipped_risk=flip, label=label)
    results[label] = (t, _stats(t))
    print(f"  {label:35s}  trades={_stats(t)['n']:3d}  WR={_stats(t)['wr']:5.1f}%  "
          f"AvgR={_stats(t)['avgr']:+.2f}R  PnL=${_stats(t)['pnl']:+,.2f}  "
          f"MaxDD={_stats(t)['maxdd']:.1f}%")


# ── Summary table ─────────────────────────────────────────────────────────────
W = 100
print()
print("=" * W)
print("  BTC BOT 2 — FILTER COMPARISON  |  Asia Night 02:00-04:00 UTC  |  VB + Swing Level")
print(f"  2yr data: {df_btc.index[0].date()} → {df_btc.index[-1].date()}")
print("=" * W)
print(f"  {'Variant':38s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>9}  {'Bal $':>8}  {'MaxDD%':>7}")
print("  " + "-" * 82)

best_pnl  = max(s["pnl"]  for _, s in results.values())
best_wr   = max(s["wr"]   for _, s in results.values())

for label, (trades, s) in results.items():
    wr_tag  = " ← best WR"  if s["wr"]  == best_wr  and s["n"] > 5 else ""
    pnl_tag = " ← best PnL" if s["pnl"] == best_pnl and s["n"] > 5 else ""
    tag     = wr_tag or pnl_tag
    print(f"  {label:38s}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+8,.2f}  ${s['final_bal']:>7,.2f}  {s['maxdd']:>6.1f}%{tag}")

print("=" * W)


# ── Bot 1 reference (for comparison context) ──────────────────────────────────
print()
print("  FOR REFERENCE — Bot 1 (21-24 UTC, Combined 3-Strategy, Version D):")
print("  223 trades | WR = 43.0% | AvgR = +0.47R | PnL = +$23,733 | MaxDD = 16.1%")
print("  Bot 2 has fewer trades (2 bars/night vs 3 bars/night) but higher WR potential.")


# ── Per-strategy breakdown for best variant ────────────────────────────────────
# Pick best variant by PnL (with at least 10 trades)
eligible = [(lbl, t, s) for lbl, (t, s) in results.items() if s["n"] >= 10]
if eligible:
    best_lbl, best_trades, best_s = max(eligible, key=lambda x: x[2]["pnl"])
    print()
    print("=" * W)
    print(f"  PER-STRATEGY BREAKDOWN — Best variant: {best_lbl}")
    print("=" * W)

    vb_trades  = [t for t in best_trades if t.get("strategy_used") == "Volatility Breakout"]
    sl_trades  = [t for t in best_trades if t.get("strategy_used") == "Swing Level Break"]

    for name, tlist in [("Volatility Breakout", vb_trades), ("Swing Level Break", sl_trades)]:
        if not tlist:
            print(f"  {name}: no trades")
            continue
        s2 = _stats(tlist)
        print(f"  {name:22s}  trades={s2['n']:3d}  WR={s2['wr']:5.1f}%  "
              f"AvgR={s2['avgr']:+.2f}R  PnL=${s2['pnl']:+,.2f}  MaxDD={s2['maxdd']:.1f}%")

    # Long vs Short
    print()
    print(f"  LONG vs SHORT — {best_lbl}")
    longs  = [t for t in best_trades if t["direction"] == "long"]
    shorts = [t for t in best_trades if t["direction"] == "short"]
    for name, tlist in [("LONG", longs), ("SHORT", shorts)]:
        if not tlist:
            continue
        s2 = _stats(tlist)
        print(f"  {name:6s}  trades={s2['n']:3d}  WR={s2['wr']:5.1f}%  "
              f"AvgR={s2['avgr']:+.2f}R  PnL=${s2['pnl']:+,.2f}")


# ── Monthly breakdown for best variant ────────────────────────────────────────
if eligible:
    print()
    print("=" * W)
    print(f"  MONTHLY BREAKDOWN — {best_lbl}")
    print("=" * W)
    print(f"  {'Month':>7}  {'#':>3}  {'WR%':>5}  {'AvgR':>6}  {'PnL$':>8}  "
          f"{'Bal$':>8}  {'VB':>3}  {'SL':>3}  {'L/S':>6}")
    print("  " + "-" * 66)

    monthly = defaultdict(list)
    for t in best_trades:
        ym = str(t["open_time"])[:7]
        monthly[ym].append(t)

    for ym in sorted(monthly.keys()):
        ts_ = monthly[ym]
        s2  = _stats(ts_)
        vb_ = len([t for t in ts_ if t.get("strategy_used") == "Volatility Breakout"])
        sl_ = len([t for t in ts_ if t.get("strategy_used") == "Swing Level Break"])
        lo_ = len([t for t in ts_ if t["direction"] == "long"])
        sh_ = len([t for t in ts_ if t["direction"] == "short"])
        tag = "  <<< HIGH WR" if s2["wr"] >= 55 else ("  !WEAK" if s2["wr"] < 35 else "")
        print(f"  {ym:>7}  {s2['n']:>3}  {s2['wr']:>4.1f}%  {s2['avgr']:>+5.2f}R  "
              f"${s2['pnl']:>+7,.2f}  ${s2['final_bal']:>7,.2f}  {vb_:>3}  {sl_:>3}  "
              f"{lo_}L/{sh_}S{tag}")

    total_s = _stats(best_trades)
    print("  " + "-" * 66)
    print(f"  {'TOTAL':>7}  {total_s['n']:>3}  {total_s['wr']:>4.1f}%  "
          f"{total_s['avgr']:>+5.2f}R  ${total_s['pnl']:>+7,.2f}  "
          f"${total_s['final_bal']:>7,.2f}")


# ── ADX distribution analysis ─────────────────────────────────────────────────
if eligible:
    print()
    print("=" * W)
    print(f"  ADX DISTRIBUTION AT ENTRY — {best_lbl}")
    print("  (Helps decide if ADX threshold or flipped risk is optimal)")
    print("=" * W)

    adx_buckets = {"20-25": [], "25-30": [], "30-40": [], "40+": []}
    for t in best_trades:
        adx_v = t["adx_at_entry"]
        if adx_v < 25:
            adx_buckets["20-25"].append(t)
        elif adx_v < 30:
            adx_buckets["25-30"].append(t)
        elif adx_v < 40:
            adx_buckets["30-40"].append(t)
        else:
            adx_buckets["40+"].append(t)

    print(f"  {'ADX Range':>10}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>9}")
    print("  " + "-" * 45)
    for bucket, tlist in adx_buckets.items():
        if not tlist:
            continue
        s2 = _stats(tlist)
        print(f"  {bucket:>10}  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  ${s2['pnl']:>+8,.2f}")


# ── Final recommendation ──────────────────────────────────────────────────────
print()
print("=" * W)
print("  RECOMMENDATION")
print("=" * W)

if eligible:
    rec = max(eligible, key=lambda x: x[2]["pnl"])
    rec_label, rec_trades, rec_s = rec
    wr_vs_b1  = rec_s["wr"]  - 43.0
    pnl_ratio = rec_s["pnl"] / 23733 * 100

    print(f"  Best variant : {rec_label}")
    print(f"  Trades       : {rec_s['n']}  (vs Bot 1: 223 — fewer bars per night, expected)")
    print(f"  Win Rate     : {rec_s['wr']}%  ({'+' if wr_vs_b1 >= 0 else ''}{wr_vs_b1:.1f}% vs Bot 1's 43.0%)")
    print(f"  Avg R        : {rec_s['avgr']:+.2f}R")
    print(f"  Total PnL    : ${rec_s['pnl']:+,.2f}  ({pnl_ratio:.0f}% of Bot 1's $23,733)")
    print(f"  Max Drawdown : {rec_s['maxdd']}%")
    print()
    print("  Next steps:")
    print("  1. Update btc_bot_2/settings.py with the best filter configuration")
    print("  2. Run scan_day_btc2.py on recent dates to validate signal quality")
    print("  3. Build btc_bot_2 live infrastructure (mirrors btc_bot_1)")

print()
