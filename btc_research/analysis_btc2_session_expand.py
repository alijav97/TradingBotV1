"""
btc_research/analysis_btc2_session_expand.py — BTC Bot 2 session & risk expansion.

Core problem: Bot 2 only fires 118 trades in 2yr (2 bars/night, 02-04 UTC).
Bot 1 fires 223 trades with worse per-trade quality but 2x the compounding.

This script answers:
  1. WHICH HOURS?   Scan every UTC hour 00-08 individually — find all productive hours.
                    Can we expand the kill-zone beyond 02-04 UTC?

  2. RISK SCALING?  At what risk% does Bot 2 reach $20k+ PnL?
                    Show the PnL vs MaxDD tradeoff curve.

  3. H4 TREND FILTER? High WR months were directional (strong trend at H4/D1 level).
                    Does adding a H4 EMA alignment boost WR enough to justify 4%+ risk?

  4. COMBINED TARGET: What combination hits $20k PnL with MaxDD under 25%?

Run:
    python btc_research/analysis_btc2_session_expand.py
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
ATR_ARR  = _tr.rolling(14).mean().bfill().values
EMA200   = _c.ewm(span=200, adjust=False).mean().values
EMA50    = _c.ewm(span=50,  adjust=False).mean().values
# H4 EMA — use EMA20 on H1 bars as proxy for H4 EMA5 trend (4*5=20)
EMA20_H1 = _c.ewm(span=20,  adjust=False).mean().values
# D1 proxy — EMA24 on H1 bars ≈ D1 EMA1 (very short D1), use EMA96 as ~D1 EMA4
EMA96    = _c.ewm(span=96,  adjust=False).mean().values

TS    = df_btc.index
H_ARR = _h.values
L_ARR = _l.values
C_ARR = _c.values


def _calc_adx(period: int = 14) -> np.ndarray:
    sp  = 2 * period - 1
    hd  = _h.diff(); ld = _l.diff()
    pdm = hd.where((hd > 0) & (hd > -ld), 0.0)
    mdm = (-ld).where((-ld > 0) & (-ld > hd), 0.0)
    aw  = _tr.ewm(span=sp, adjust=False).mean()
    pw  = pdm.ewm(span=sp, adjust=False).mean()
    mw  = mdm.ewm(span=sp, adjust=False).mean()
    pdi = 100 * pw / aw; mdi = 100 * mw / aw
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, float("nan"))
    return dx.ewm(span=sp, adjust=False).mean().fillna(0).values


ADX_ARR = _calc_adx(14)


def _stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "avgr": 0.0, "pnl": 0.0,
                "final_bal": b2cfg.STARTING_BALANCE, "maxdd": 0.0}
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


def simulate(
    kz_hours:       list[int],
    adx_threshold:  float = 20.0,
    use_ema200:     bool  = True,
    risk_pct:       float = 0.02,
    risk_mode:      str   = "flat",    # flat | adx_split | normal
    use_h4_filter:  bool  = False,     # H4 trend must align with trade direction
    use_ema96:      bool  = False,     # D1-proxy EMA96 must align
    tp1_rr:         float = 2.0,
    tp2_rr:         float = 5.0,
    trail_mult:     float = 2.0,
) -> list[dict]:
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
        if hr not in kz_hours:
            continue

        adx_now    = float(ADX_ARR[i])
        ema200_now = float(EMA200[i])
        ema20_now  = float(EMA20_H1[i])
        ema96_now  = float(EMA96[i])

        if adx_now < adx_threshold:
            continue

        win = df_btc.iloc[max(0, i - 220):i + 1]

        for direction in ("long", "short"):
            # EMA200 filter
            if use_ema200:
                if direction == "long"  and bc_ < ema200_now: continue
                if direction == "short" and bc_ > ema200_now: continue
            # H4 EMA filter (EMA20 on H1 as H4 trend proxy)
            if use_h4_filter:
                if direction == "long"  and bc_ < ema20_now: continue
                if direction == "short" and bc_ > ema20_now: continue
            # D1 EMA96 filter
            if use_ema96:
                if direction == "long"  and bc_ < ema96_now: continue
                if direction == "short" and bc_ > ema96_now: continue

            sig = strat.generate_signal(win, bar_time, direction)
            if not sig.get("signal"):
                continue

            sl_d = abs(float(sig["entry"]) - float(sig["sl"]))
            if sl_d <= 0:
                continue

            # Risk sizing
            if risk_mode == "adx_split":
                rp = 0.03 if adx_now <= 25 else (0.02 if adx_now <= 40 else 0.03)
            elif risk_mode == "normal":
                rp = 0.03 if adx_now > 28 else 0.02
            else:
                rp = risk_pct

            ru   = round(balance * rp, 2)
            lots = ru / sl_d
            tp1  = float(sig["entry"]) + tp1_rr * sl_d if direction == "long" else float(sig["entry"]) - tp1_rr * sl_d
            tp2  = float(sig["entry"]) + tp2_rr * sl_d if direction == "long" else float(sig["entry"]) - tp2_rr * sl_d

            open_t = {
                "open_time":     bar_time, "open_idx": i,
                "direction":     direction,
                "entry":         float(sig["entry"]),
                "sl":            float(sig["sl"]), "orig_sl": float(sig["sl"]),
                "tp1":           tp1, "tp2": tp2,
                "tp1_rr":        tp1_rr, "tp2_rr": tp2_rr,
                "lots":          lots, "risk_usd": ru, "risk_pct": rp,
                "signal_reason": sig.get("reason", ""),
                "strategy_used": sig.get("strategy_used", ""),
                "trail_peak":    0.0, "tp1_hit": False,
                "pnl_running":   0.0, "r_running": 0.0, "pnl_usd": 0.0,
                "adx_at_entry":  round(adx_now, 1),
                "atr_at_entry":  round(float(ATR_ARR[i]), 0),
                "above_ema200":  bc_ > ema200_now,
                "hour":          hr,
            }
            break

    return trades


W = 108

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HOUR-BY-HOUR SCAN (00-08 UTC)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * W)
print("  SECTION 1 — SINGLE-HOUR SCAN  |  VB + Swing Level  |  EMA200 + ADX>=20")
print("  Which Asia Night hours are genuinely worth trading?")
print("=" * W)
print(f"  {'Hour':>6}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>9}  {'MaxDD':>7}  Grade")
print("  " + "-" * 65)

hour_results = {}
for hr in range(0, 9):
    t = simulate([hr], adx_threshold=20, use_ema200=True, risk_mode="flat",
                 risk_pct=0.02, tp1_rr=2.0, tp2_rr=5.0)
    s = _stats(t)
    hour_results[hr] = (t, s)
    grade = ""
    if s["n"] >= 5:
        if s["wr"] >= 50 and s["avgr"] > 0.4:
            grade = "✅ STRONG"
        elif s["wr"] >= 45 and s["avgr"] > 0.3:
            grade = "✅ GOOD"
        elif s["wr"] >= 40 and s["avgr"] > 0.0:
            grade = "⚠️  MARGINAL"
        elif s["n"] < 8:
            grade = "⚠️  FEW TRADES"
        else:
            grade = "❌ AVOID"
    else:
        grade = "— insufficient data"
    print(f"  {hr:02d}:00    {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+8,.2f}  {s['maxdd']:>6.1f}%  {grade}")

# Best hours by AvgR (minimum 5 trades, positive expectancy)
good_hours = sorted(
    [(hr, t, s) for hr, (t, s) in hour_results.items() if s["n"] >= 5 and s["avgr"] > 0.2],
    key=lambda x: x[2]["avgr"], reverse=True
)
print()
print(f"  Top hours by AvgR (min 5 trades, positive expectancy):")
for hr, _, s in good_hours[:6]:
    print(f"    {hr:02d}:00 UTC  trades={s['n']}  WR={s['wr']}%  AvgR={s['avgr']:+.2f}R  PnL=${s['pnl']:+,.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXPANDED KILL-ZONE COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * W)
print("  SECTION 2 — EXPANDED KILL-ZONE  |  Combining productive hours")
print("  Config: EMA200 + ADX>=20 + ADX-split risk + TP1=2R/TP2=5R")
print("=" * W)
print(f"  {'Session':35s}  {'Hours':>18}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>9}  {'MaxDD':>7}")
print("  " + "-" * 94)

# Build candidate hour sets from good hours
top_hrs = [hr for hr, _, _ in good_hours[:6]]
print(f"  (Good hours identified: {sorted(top_hrs)})")
print()

kz_combos = [
    ("Bot 2 original",              [2, 3]),
    ("Add 1hr before",              [1, 2, 3]),
    ("Add 1hr after",               [2, 3, 4]),
    ("1-4 UTC (4 bars)",            [1, 2, 3, 4]),
    ("0-4 UTC (4 bars incl 00)",    [0, 1, 2, 3]),
    ("2-6 UTC (4 bars)",            [2, 3, 4, 5]),
    ("Best 3 from scan",            sorted(top_hrs[:3])),
    ("Best 4 from scan",            sorted(top_hrs[:4])),
    ("Best 5 from scan",            sorted(top_hrs[:5])),
    ("Full Asia 0-8 UTC",           list(range(0, 8))),
]

expanded_results = {}
for label, hours in kz_combos:
    t = simulate(hours, adx_threshold=20, use_ema200=True, risk_mode="adx_split",
                 tp1_rr=2.0, tp2_rr=5.0)
    s = _stats(t)
    expanded_results[label] = (t, s, hours)
    hrs_str = ",".join(f"{h:02d}" for h in sorted(hours))
    print(f"  {label:35s}  [{hrs_str}]  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+8,.2f}  {s['maxdd']:>6.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RISK SCALING CURVE
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * W)
print("  SECTION 3 — RISK % SCALING CURVE")
print("  Using BEST expanded session from Section 2 + ADX-split risk (overrides flat %)")
print("  Shows PnL vs MaxDD tradeoff — find the sweet spot for $20k target")
print("=" * W)

# Use best expanded session by PnL
best_exp = max(expanded_results.items(), key=lambda x: x[1][1]["pnl"] if x[1][1]["n"] >= 10 else -9999)
best_exp_label, (_, _, best_hours) = best_exp[0], best_exp[1]
print(f"  Session used: {best_exp_label}  hours={sorted(best_hours)}")
print()
print(f"  {'Risk %':>8}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>10}  {'Bal $':>9}  {'MaxDD':>7}  Target?")
print("  " + "-" * 78)

for risk_pct in [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08]:
    t = simulate(best_hours, adx_threshold=20, use_ema200=True,
                 risk_mode="flat", risk_pct=risk_pct,
                 tp1_rr=2.0, tp2_rr=5.0)
    s = _stats(t)
    target = "✅ $20k+" if s["pnl"] >= 20000 and s["maxdd"] <= 30 else (
             "⚠️  $20k+ but high DD" if s["pnl"] >= 20000 else "")
    print(f"  {risk_pct*100:>7.1f}%  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+9,.2f}  ${s['final_bal']:>8,.2f}  {s['maxdd']:>6.1f}%  {target}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — H4 / D1 TREND FILTER
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * W)
print("  SECTION 4 — TREND ALIGNMENT FILTER (H4 + D1 proxy)")
print("  High WR months were directional. Does adding a higher-TF trend filter help?")
print("  H4 proxy = EMA20 on H1 bars (aligns with H4 short-term trend)")
print("  D1 proxy = EMA96 on H1 bars (aligns with daily trend direction)")
print("=" * W)
print(f"  {'Config':50s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>9}  {'MaxDD':>7}")
print("  " + "-" * 93)

trend_configs = [
    ("Baseline (EMA200 only)",                         False, False),
    ("+ H4 EMA20 aligned",                             True,  False),
    ("+ D1 EMA96 aligned",                             False, True),
    ("+ H4 EMA20 + D1 EMA96 both aligned",             True,  True),
]

for label, h4, d1 in trend_configs:
    t = simulate(best_hours, adx_threshold=20, use_ema200=True,
                 risk_mode="adx_split", tp1_rr=2.0, tp2_rr=5.0,
                 use_h4_filter=h4, use_ema96=d1)
    s = _stats(t)
    print(f"  {label:50s}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+8,.2f}  {s['maxdd']:>6.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BEST COMBINED: TARGET $20k+
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * W)
print("  SECTION 5 — HUNT FOR $20k+  |  Expanded hours + optimal risk + best filters")
print("=" * W)
print(f"  {'Config':55s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL $':>10}  {'MaxDD':>7}  OK?")
print("  " + "-" * 103)

# Test promising combinations targeting $20k+
hunt_configs = []
for hours in [best_hours, sorted(top_hrs[:4]), sorted(top_hrs[:5]), list(range(0,8))]:
    for risk_pct in [0.03, 0.04, 0.05]:
        for h4, d1 in [(False, False), (True, False), (False, True)]:
            hunt_configs.append((hours, risk_pct, h4, d1))

seen = set()
for hours, rp, h4, d1 in hunt_configs:
    key = (tuple(sorted(hours)), rp, h4, d1)
    if key in seen: continue
    seen.add(key)

    t = simulate(hours, adx_threshold=20, use_ema200=True,
                 risk_mode="flat", risk_pct=rp,
                 use_h4_filter=h4, use_ema96=d1,
                 tp1_rr=2.0, tp2_rr=5.0)
    s = _stats(t)

    if s["pnl"] < 8000 or s["n"] < 20:
        continue   # skip low PnL / low trade count configs

    hrs_str  = ",".join(f"{h:02d}" for h in sorted(hours))
    h4_tag   = "+H4" if h4 else ""
    d1_tag   = "+D1" if d1 else ""
    label    = f"[{hrs_str}] risk={rp*100:.0f}%{h4_tag}{d1_tag}"
    ok_flag  = "✅" if s["pnl"] >= 20000 and s["maxdd"] <= 30 else (
               "⚠️ DD" if s["pnl"] >= 20000 else "")

    print(f"  {label:55s}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avgr']:>+5.2f}R  "
          f"${s['pnl']:>+9,.2f}  {s['maxdd']:>6.1f}%  {ok_flag}")

print()
print("  Legend: ✅ = hits $20k+ with MaxDD ≤ 30%  |  ⚠️ DD = hits $20k but DD too high")
print()
print("=" * W)
print("  SUMMARY")
print("=" * W)
print("  Bot 1 reference: 223 trades | WR=43% | AvgR=+0.47R | PnL=+$23,733 | MaxDD=16.1%")
print()
print("  The path to $20k+ for Bot 2 requires one of:")
print("    A) More hours — expand beyond 02-04 UTC to include other productive Asia hours")
print("    B) Higher risk — increase from 2% to 4-5% (but MaxDD scales proportionally)")
print("    C) Both A + B — wider session at moderate risk increase")
print("    D) A new filter that boosts WR above 55% — then higher risk is safer")
print()
