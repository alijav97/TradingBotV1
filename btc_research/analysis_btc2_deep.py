"""
btc_research/analysis_btc2_deep.py — BTC Bot 2 deep-dive analysis.

Three questions this answers:

  1. RISK/REWARD — Are the current TP levels (VB=9R, SL=5R) right for Asia Night?
     Tests 5 TP configurations: current vs scalp vs moderate vs extended vs asymmetric.

  2. ADX TREATMENT — ADX 25-30 was the WORST zone (+0.36R, 37.5% WR) while ADX 40+
     was the BEST (+0.96R, 53.6% WR). The flipped risk (size up at 20-28) may be wrong
     for this session. Tests: normal risk / flipped / skip-25-30 / ADX-based sizing.

  3. HIGH WR MONTHS — What drove June 2024 (100% WR), November 2024 (75%), April 2025
     (66.7%)? Are there seasonal filters we can apply to avoid the weak months?

Run:
    python btc_research/analysis_btc2_deep.py
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
from btc_research.strategies.volatility_breakout import VolatilityBreakout
from btc_research.strategies.swing_level import SwingLevelBreak

# ── Data ──────────────────────────────────────────────────────────────────────
print("Fetching 2yr data...")
data   = fetch_all(use_cache=True, force_refresh=False)
df_btc = data.get(cfg.BTC_SYMBOL, pd.DataFrame())
if df_btc.empty:
    print("ERROR: No BTC data.")
    sys.exit(1)

if "time" in df_btc.columns:
    df_btc = df_btc.set_index(pd.to_datetime(df_btc["time"], utc=True)).drop(columns=["time"])
elif not isinstance(df_btc.index, pd.DatetimeIndex):
    df_btc.index = pd.to_datetime(df_btc.index, utc=True)

print(f"Loaded {len(df_btc):,} bars  ({df_btc.index[0].date()} → {df_btc.index[-1].date()})\n")

# ── Pre-compute indicators ────────────────────────────────────────────────────
_c = df_btc["close"].astype(float)
_h = df_btc["high"].astype(float)
_l = df_btc["low"].astype(float)
_tr = pd.concat([_h - _l, (_h - _c.shift(1)).abs(), (_l - _c.shift(1)).abs()], axis=1).max(axis=1)
ATR_ARR = _tr.rolling(14).mean().bfill().values
EMA200  = _c.ewm(span=200, adjust=False).mean().values
EMA50   = _c.ewm(span=50,  adjust=False).mean().values
EMA20   = _c.ewm(span=20,  adjust=False).mean().values
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
KZ_START = b2cfg.KZ_START_UTC
KZ_END   = b2cfg.KZ_END_UTC


# ── Simulator (parametric) ────────────────────────────────────────────────────
def simulate(
    strategy,
    adx_threshold: float = 20.0,
    use_ema200:    bool  = True,
    risk_mode:     str   = "flat",     # flat | flipped | normal | adx_split
    tp1_override:  float | None = None,
    tp2_override:  float | None = None,
    skip_adx_band: tuple | None = None,  # e.g. (25, 32) — skip this ADX range
    trail_mult:    float = 2.0,
) -> list[dict]:
    """
    risk_mode:
      flat      — always 2% (baseline)
      flipped   — 3% ADX 20-28, 2% ADX >28  (what we tested in run_backtest_btc2)
      normal    — 2% ADX 20-28, 3% ADX >28  (Bot 1's original dynamic — size up strong trend)
      adx_split — 3% ADX 20-25, 2% ADX 25-40, 3% ADX 40+  (custom for Bot 2)

    skip_adx_band — skip trades where ADX falls in this range (e.g. the 25-30 dead zone)
    tp1/tp2 override — force fixed TP multiples for ALL strategies (ignore per-strategy defaults)
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

        # ── Manage open trade ───────────────────────────────────────────────
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
                    open_t["sl"] = max(open_t["trail_peak"] - trail_mult * atr_now, entry_)
                else:
                    if bl_ < open_t["trail_peak"]: open_t["trail_peak"] = bl_
                    open_t["sl"] = min(open_t["trail_peak"] + trail_mult * atr_now, entry_)
                new_sl = open_t["sl"]
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
            open_t["r_multiple"]    = round(open_t.get("r_running",   0) + r_,  2)
            open_t["balance_after"] = round(balance, 2)
            open_t["exit_reason"]   = ex
            trades.append(open_t)
            open_t = None

        if open_t is not None:
            continue

        # ── Entry gate ──────────────────────────────────────────────────────
        if not (KZ_START <= hr < KZ_END):
            continue

        adx_now    = float(ADX_ARR[i])
        ema200_now = float(EMA200[i])

        if adx_now < adx_threshold:
            continue

        # Skip ADX band if specified
        if skip_adx_band and skip_adx_band[0] <= adx_now < skip_adx_band[1]:
            continue

        win = df_btc.iloc[max(0, i - 220):i + 1]

        for direction in ("long", "short"):
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
            if risk_mode == "flipped":
                rp = 0.03 if adx_now <= 28 else 0.02
            elif risk_mode == "normal":
                rp = 0.03 if adx_now > 28 else 0.02
            elif risk_mode == "adx_split":
                rp = 0.03 if adx_now <= 25 else (0.02 if adx_now <= 40 else 0.03)
            else:
                rp = 0.02   # flat

            ru   = round(balance * rp, 2)
            lots = ru / sl_d

            # TP levels — use override if provided, else strategy's own defaults
            tp1r = tp1_override if tp1_override is not None else sig.get("tp1_rr", b2cfg.TP1_RR)
            tp2r = tp2_override if tp2_override is not None else sig.get("tp2_rr", b2cfg.TP2_RR)
            tp1  = float(sig["entry"]) + tp1r * sl_d if direction == "long" else float(sig["entry"]) - tp1r * sl_d
            tp2  = float(sig["entry"]) + tp2r * sl_d if direction == "long" else float(sig["entry"]) - tp2r * sl_d

            open_t = {
                "open_time":     bar_time,
                "open_idx":      i,
                "direction":     direction,
                "entry":         float(sig["entry"]),
                "sl":            float(sig["sl"]),
                "orig_sl":       float(sig["sl"]),
                "tp1":           tp1, "tp2": tp2,
                "tp1_rr":        tp1r, "tp2_rr": tp2r,
                "lots":          lots, "risk_usd": ru, "risk_pct": rp,
                "signal_reason": sig.get("reason", ""),
                "strategy_used": sig.get("strategy_used", ""),
                "trail_peak":    0.0, "tp1_hit": False,
                "pnl_running":   0.0, "r_running": 0.0, "pnl_usd": 0.0,
                "adx_at_entry":  round(adx_now, 1),
                "atr_at_entry":  round(float(ATR_ARR[i]), 0),
                "above_ema200":  bc_ > ema200_now,
                "above_ema50":   bc_ > float(EMA50[i]),
                "above_ema20":   bc_ > float(EMA20[i]),
            }
            break

    return trades


def _stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "avgr": 0.0, "pnl": 0.0, "final_bal": b2cfg.STARTING_BALANCE, "maxdd": 0.0}
    wins  = [t for t in trades if t["pnl_usd"] > 0]
    wr    = len(wins) / len(trades) * 100
    avgr  = sum(t["r_multiple"] for t in trades) / len(trades)
    pnl   = sum(t["pnl_usd"]   for t in trades)
    bals  = [b2cfg.STARTING_BALANCE] + [t["balance_after"] for t in trades]
    peak  = b2cfg.STARTING_BALANCE; maxdd = 0.0
    for b in bals:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100
        if dd > maxdd: maxdd = dd
    return {"n": len(trades), "wr": round(wr, 1), "avgr": round(avgr, 2),
            "pnl": round(pnl, 2), "final_bal": round(bals[-1], 2), "maxdd": round(maxdd, 1)}


strat = VBSwingStrategy(atr_multiplier=b2cfg.VB_ATR_MULTIPLIER, close_zone=b2cfg.VB_CLOSE_ZONE)
W = 105

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TP/RR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * W)
print("  SECTION 1 — RISK/REWARD (TP) COMPARISON")
print("  Base config: EMA200 + ADX>=20 + Flat 2% risk")
print("  All use trailing SL 2×ATR after TP1")
print("=" * W)
print(f"  {'Config':40s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>9}  {'Bal $':>8}  {'MaxDD':>7}")
print("  " + "-" * 87)

tp_variants = [
    # label,                              tp1,  tp2
    ("Current  (VB=2R/9R, SL=1.5R/5R)",  None, None),   # use strategy defaults
    ("Scalp    (TP1=1.5R, TP2=3R)",       1.5,  3.0),
    ("Moderate (TP1=1.5R, TP2=4R)",       1.5,  4.0),
    ("Bot1-same(TP1=2R,   TP2=5R)",       2.0,  5.0),
    ("Extended (TP1=2R,   TP2=7R)",       2.0,  7.0),
    ("Runner   (TP1=1.5R, TP2=6R)",       1.5,  6.0),
]

tp_results = {}
for label, tp1, tp2 in tp_variants:
    t = simulate(strat, use_ema200=True, risk_mode="flat", tp1_override=tp1, tp2_override=tp2)
    s = _stats(t)
    tp_results[label] = (t, s)
    print(f"  {label:40s}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+8,.2f}  ${s['final_bal']:>7,.2f}  {s['maxdd']:>6.1f}%")

print()
# Best TP config
best_tp = max(tp_results.items(), key=lambda x: x[1][1]["pnl"] if x[1][1]["n"] >= 10 else -9999)
print(f"  >>> Best by PnL: {best_tp[0]}")
best_tp_trades = best_tp[1][0]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ADX TREATMENT
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * W)
print("  SECTION 2 — ADX TREATMENT COMPARISON")
print("  All use EMA200 + best TP from Section 1")
print("=" * W)

# Use best TP from Section 1 for all ADX tests
best_tp1_val = tp_variants[[l for l,_,_ in tp_variants].index(best_tp[0])][1]
best_tp2_val = tp_variants[[l for l,_,_ in tp_variants].index(best_tp[0])][2]

print(f"  {'Config':45s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>9}  {'Bal $':>8}  {'MaxDD':>7}")
print("  " + "-" * 92)

adx_variants = [
    # label,                              adx_thresh, skip_band,   risk_mode
    ("Flat risk, ADX>=20",                20,         None,        "flat"),
    ("Flat risk, ADX>=25",                25,         None,        "flat"),
    ("Flat risk, ADX>=30",                30,         None,        "flat"),
    ("Flat risk, ADX>=20 skip 25-30",     20,         (25, 30),    "flat"),
    ("Flat risk, ADX>=20 skip 25-35",     20,         (25, 35),    "flat"),
    ("Flipped  (3% @20-28, 2% @28+)",     20,         None,        "flipped"),
    ("Normal   (2% @20-28, 3% @28+)",     20,         None,        "normal"),
    ("ADX-split(3%@20-25, 2%@25-40,3%@40+)", 20,      None,        "adx_split"),
    ("Normal + skip 25-30",               20,         (25, 30),    "normal"),
    ("ADX-split + skip 25-30",            20,         (25, 30),    "adx_split"),
]

adx_results = {}
for label, adx_t, skip, rmode in adx_variants:
    t = simulate(strat, adx_threshold=adx_t, use_ema200=True, risk_mode=rmode,
                 tp1_override=best_tp1_val, tp2_override=best_tp2_val,
                 skip_adx_band=skip)
    s = _stats(t)
    adx_results[label] = (t, s)
    print(f"  {label:45s}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+8,.2f}  ${s['final_bal']:>7,.2f}  {s['maxdd']:>6.1f}%")

print()
best_adx = max(adx_results.items(), key=lambda x: x[1][1]["pnl"] if x[1][1]["n"] >= 10 else -9999)
print(f"  >>> Best by PnL: {best_adx[0]}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HIGH WR MONTH DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════════
# Use baseline Variant E trades (EMA200 + Flip Risk) for month analysis
print()
print("=" * W)
print("  SECTION 3 — HIGH WR vs WEAK MONTH DEEP-DIVE")
print("  Using Variant E (EMA200 + Flip Risk) from run_backtest_btc2.py")
print("  Question: What market conditions drove HIGH WR months?")
print("=" * W)

base_trades = simulate(strat, adx_threshold=20, use_ema200=True, risk_mode="flipped",
                       tp1_override=None, tp2_override=None)

monthly = defaultdict(list)
for t in base_trades:
    ym = str(t["open_time"])[:7]
    monthly[ym].append(t)

print(f"\n  {'Month':>7}  {'#':>3}  {'WR%':>5}  {'AvgR':>6}  {'PnL$':>8}  "
      f"{'L/S':>6}  {'VB/SL':>6}  {'AvgADX':>7}  {'AvgATR$':>8}  {'AbvEMA200':>9}  {'AbvEMA50':>8}  Tag")
print("  " + "-" * 113)

high_wr_months = []
weak_months    = []

for ym in sorted(monthly.keys()):
    ts_  = monthly[ym]
    s2   = _stats(ts_)
    lo_  = len([t for t in ts_ if t["direction"] == "long"])
    sh_  = len([t for t in ts_ if t["direction"] == "short"])
    vb_  = len([t for t in ts_ if t.get("strategy_used") == "Volatility Breakout"])
    sl_  = len([t for t in ts_ if t.get("strategy_used") == "Swing Level Break"])
    adxv = round(sum(t["adx_at_entry"] for t in ts_) / len(ts_), 1)
    atrv = round(sum(t["atr_at_entry"] for t in ts_) / len(ts_), 0)
    pct_above_200 = round(len([t for t in ts_ if t.get("above_ema200")]) / len(ts_) * 100)
    pct_above_50  = round(len([t for t in ts_ if t.get("above_ema50")])  / len(ts_) * 100)

    tag = " <<< HIGH WR" if s2["wr"] >= 55 else (" !WEAK" if s2["wr"] < 35 else "")
    print(f"  {ym:>7}  {s2['n']:>3}  {s2['wr']:>4.1f}%  {s2['avgr']:>+5.2f}R  ${s2['pnl']:>+7,.2f}  "
          f"{lo_}L/{sh_}S  {vb_}VB/{sl_}SL  {adxv:>7.1f}  ${atrv:>7,.0f}  "
          f"{pct_above_200:>8}%  {pct_above_50:>7}%{tag}")

    if s2["wr"] >= 55 and s2["n"] >= 3:
        high_wr_months.append({"ym": ym, "trades": ts_, "stats": s2,
                                "lo": lo_, "sh": sh_, "adxv": adxv, "atrv": atrv,
                                "pct200": pct_above_200, "pct50": pct_above_50})
    elif s2["wr"] < 35 and s2["n"] >= 3:
        weak_months.append({"ym": ym, "trades": ts_, "stats": s2,
                            "lo": lo_, "sh": sh_, "adxv": adxv, "atrv": atrv,
                            "pct200": pct_above_200, "pct50": pct_above_50})


# ── Aggregate comparison ──────────────────────────────────────────────────────
print()
print("=" * W)
print("  WHAT SEPARATES HIGH WR FROM WEAK MONTHS?")
print("=" * W)

all_hw = [t for m in high_wr_months for t in m["trades"]]
all_wk = [t for m in weak_months    for t in m["trades"]]

if all_hw and all_wk:
    hw_s = _stats(all_hw)
    wk_s = _stats(all_wk)
    print(f"\n  HIGH WR months ({len(high_wr_months)} months, {hw_s['n']} trades):")
    print(f"    WR={hw_s['wr']}%  AvgR={hw_s['avgr']:+.2f}R  PnL=${hw_s['pnl']:+,.2f}")
    hw_lo  = len([t for t in all_hw if t["direction"] == "long"])
    hw_sh  = len([t for t in all_hw if t["direction"] == "short"])
    hw_200 = round(len([t for t in all_hw if t.get("above_ema200")]) / len(all_hw) * 100)
    hw_50  = round(len([t for t in all_hw if t.get("above_ema50")])  / len(all_hw) * 100)
    hw_adx = round(sum(t["adx_at_entry"] for t in all_hw) / len(all_hw), 1)
    hw_atr = round(sum(t["atr_at_entry"] for t in all_hw) / len(all_hw), 0)
    hw_vb  = len([t for t in all_hw if t.get("strategy_used") == "Volatility Breakout"])
    hw_sl  = len([t for t in all_hw if t.get("strategy_used") == "Swing Level Break"])
    print(f"    Direction: {hw_lo}L / {hw_sh}S  (EMA200 alignment: {hw_200}%  EMA50: {hw_50}%)")
    print(f"    Avg ADX: {hw_adx}  Avg ATR: ${hw_atr:,.0f}")
    print(f"    VB: {hw_vb}  SL: {hw_sl}")

    print(f"\n  WEAK months ({len(weak_months)} months, {wk_s['n']} trades):")
    print(f"    WR={wk_s['wr']}%  AvgR={wk_s['avgr']:+.2f}R  PnL=${wk_s['pnl']:+,.2f}")
    wk_lo  = len([t for t in all_wk if t["direction"] == "long"])
    wk_sh  = len([t for t in all_wk if t["direction"] == "short"])
    wk_200 = round(len([t for t in all_wk if t.get("above_ema200")]) / len(all_wk) * 100)
    wk_50  = round(len([t for t in all_wk if t.get("above_ema50")])  / len(all_wk) * 100)
    wk_adx = round(sum(t["adx_at_entry"] for t in all_wk) / len(all_wk), 1)
    wk_atr = round(sum(t["atr_at_entry"] for t in all_wk) / len(all_wk), 0)
    wk_vb  = len([t for t in all_wk if t.get("strategy_used") == "Volatility Breakout"])
    wk_sl  = len([t for t in all_wk if t.get("strategy_used") == "Swing Level Break"])
    print(f"    Direction: {wk_lo}L / {wk_sh}S  (EMA200 alignment: {wk_200}%  EMA50: {wk_50}%)")
    print(f"    Avg ADX: {wk_adx}  Avg ATR: ${wk_atr:,.0f}")
    print(f"    VB: {wk_vb}  SL: {wk_sl}")

    print()
    print("  KEY DIFFERENCES (High WR vs Weak):")
    print(f"    ADX at entry:  {hw_adx:.1f} (HIGH) vs {wk_adx:.1f} (WEAK)  — diff: {hw_adx-wk_adx:+.1f}")
    print(f"    ATR at entry:  ${hw_atr:,.0f} (HIGH) vs ${wk_atr:,.0f} (WEAK)  — diff: ${hw_atr-wk_atr:+,.0f}")
    print(f"    Above EMA200:  {hw_200}% (HIGH) vs {wk_200}% (WEAK)")
    print(f"    Above EMA50:   {hw_50}% (HIGH) vs {wk_50}% (WEAK)")
    dir_consistency_hw = max(hw_lo, hw_sh) / len(all_hw) * 100
    dir_consistency_wk = max(wk_lo, wk_sh) / len(all_wk) * 100
    print(f"    Dir consistency: {dir_consistency_hw:.0f}% one-sided (HIGH) vs {dir_consistency_wk:.0f}% (WEAK)")
    print(f"    Interpretation: " + (
        "Higher ATR + directional bias = HIGH WR months"
        if hw_atr > wk_atr and dir_consistency_hw > dir_consistency_wk
        else "Check individual months below for patterns"
    ))


# ── Individual high WR month details ─────────────────────────────────────────
print()
print("=" * W)
print("  HIGH WR MONTH INDIVIDUAL ANALYSIS")
print("=" * W)

for m in high_wr_months:
    ym   = m["ym"]
    ts_  = m["trades"]
    s2   = m["stats"]
    print(f"\n  {ym}  WR={s2['wr']}%  AvgR={s2['avgr']:+.2f}R  PnL=${s2['pnl']:+,.2f}  "
          f"ADX={m['adxv']}  ATR=${m['atrv']:,.0f}  {m['lo']}L/{m['sh']}S  EMA200:{m['pct200']}%  EMA50:{m['pct50']}%")
    exits = defaultdict(int)
    for t in ts_:
        exits[t["exit_reason"]] += 1
        mk = "W" if t["pnl_usd"] > 0 else "L"
        print(f"    {str(t['open_time'])[:13]}  {t['direction'].upper():5s}  "
              f"${t['entry']:>9,.0f}  ADX={t['adx_at_entry']:>5.1f}  "
              f"ATR=${t['atr_at_entry']:>6,.0f}  {t.get('strategy_used',''):22s}  "
              f"{t['exit_reason']:12s}  {t['r_multiple']:>+5.2f}R  {mk}")
    print(f"    Exits: " + "  ".join(f"{k}:{v}" for k, v in sorted(exits.items())))


# ── Weak month pattern ────────────────────────────────────────────────────────
print()
print("=" * W)
print("  WEAK MONTH COMMON PATTERNS")
print("=" * W)
for m in weak_months:
    s2 = m["stats"]
    vb_ = len([t for t in m["trades"] if t.get("strategy_used") == "Volatility Breakout"])
    sl_ = len([t for t in m["trades"] if t.get("strategy_used") == "Swing Level Break"])
    print(f"  {m['ym']}  WR={s2['wr']}%  ADX={m['adxv']}  ATR=${m['atrv']:,.0f}  "
          f"{m['lo']}L/{m['sh']}S  VB:{vb_}  SL:{sl_}  EMA200:{m['pct200']}%")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COMBINED BEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * W)
print("  SECTION 4 — COMBINED BEST: Optimal TP + Optimal ADX + Optimal Risk")
print("  Takes winner from each section and combines them")
print("=" * W)

# Get best TP and best ADX configs from above
best_tp1 = best_tp1_val
best_tp2 = best_tp2_val
best_adx_lbl, (best_adx_trades, best_adx_s) = best_adx

# Extract params from best ADX label
skip_band = None
if "skip 25-30" in best_adx_lbl:
    skip_band = (25, 30)
elif "skip 25-35" in best_adx_lbl:
    skip_band = (25, 35)

adx_thresh = 30 if "ADX>=30" in best_adx_lbl else (25 if "ADX>=25" in best_adx_lbl else 20)

risk_m = "flat"
if "Normal" in best_adx_lbl or "normal" in best_adx_lbl:
    risk_m = "normal"
elif "Flipped" in best_adx_lbl or "flipped" in best_adx_lbl:
    risk_m = "flipped"
elif "adx_split" in best_adx_lbl or "ADX-split" in best_adx_lbl:
    risk_m = "adx_split"

combos = [
    ("Best TP + Best ADX (from above)",       best_tp1, best_tp2, adx_thresh, skip_band, risk_m),
    ("Best TP + EMA200 + ADX20 + Flat",        best_tp1, best_tp2, 20,         None,      "flat"),
    ("Best TP + EMA200 + ADX20 + Normal risk", best_tp1, best_tp2, 20,         None,      "normal"),
    ("Best TP + EMA200 + ADX20 + ADX-split",   best_tp1, best_tp2, 20,         None,      "adx_split"),
    ("Best TP + EMA200 + skip25-30 + Normal",  best_tp1, best_tp2, 20,         (25, 30),  "normal"),
]

print(f"  {'Config':48s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>9}  {'Bal $':>8}  {'MaxDD':>7}")
print("  " + "-" * 95)

combo_results = {}
for label, tp1, tp2, adx_t, skip, rmode in combos:
    t = simulate(strat, adx_threshold=adx_t, use_ema200=True, risk_mode=rmode,
                 tp1_override=tp1, tp2_override=tp2, skip_adx_band=skip)
    s = _stats(t)
    combo_results[label] = (t, s)
    print(f"  {label:48s}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+8,.2f}  ${s['final_bal']:>7,.2f}  {s['maxdd']:>6.1f}%")

print()

# ── Final recommendation ──────────────────────────────────────────────────────
all_results = {**tp_results, **adx_results, **combo_results}
eligible = [(lbl, t, s) for lbl, (t, s) in all_results.items() if s["n"] >= 15]
if eligible:
    champion = max(eligible, key=lambda x: x[2]["pnl"])
    c_lbl, c_trades, c_s = champion

    print("=" * W)
    print("  FINAL RECOMMENDATION FOR BTC BOT 2")
    print("=" * W)
    print(f"  Best config found: {c_lbl}")
    print(f"  Trades:       {c_s['n']}")
    print(f"  Win Rate:     {c_s['wr']}%  (Bot 1 baseline: 43.0%)")
    print(f"  Avg R:        {c_s['avgr']:+.2f}R  (Bot 1 baseline: +0.47R)")
    print(f"  Total PnL:    ${c_s['pnl']:+,.2f}")
    print(f"  Max Drawdown: {c_s['maxdd']}%")
    print()
    print("  Update btc_bot_2/settings.py with these parameters,")
    print("  then build the live bot infrastructure.")

print()
