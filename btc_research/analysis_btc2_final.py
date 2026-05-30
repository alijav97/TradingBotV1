"""
btc_research/analysis_btc2_final.py — BTC Bot 2 FINAL config full backtest + temporal analysis.

Uses the LOCKED IN final config:
  Strategy  : VB + Swing Level Break v2  (mode="both", max_sl_atr=2.0)
  Hours     : [1, 2, 3, 8] UTC
  Filters   : EMA200 + ADX >= 20
  Risk      : ADX-split (3% ADX<=25, 2% ADX 25-40, 3% ADX>=40)
  TP1 / TP2 : 2R / 5R  |  Trailing SL: 2×ATR after TP1
  Max hold  : 96 bars

Sections:
  1. Overall performance summary
  2. Per-strategy + per-entry-type breakdown (VB | SLv2-break | SLv2-retest)
  3. ADX zone breakdown (20-25 | 25-40 | 40+)
  4. Year-half breakdown (5 halves)
  5. Quarterly breakdown (10 quarters)
  6. Monthly breakdown (24 months)
  7. Consistency scorecard

Run:
    python btc_research/analysis_btc2_final.py
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

print(f"Loaded {len(df_btc):,} bars  ({df_btc.index[0].date()} to {df_btc.index[-1].date()})")

# ── Pre-compute indicators ────────────────────────────────────────────────────
_c = df_btc["close"].astype(float)
_h = df_btc["high"].astype(float)
_l = df_btc["low"].astype(float)

_tr     = pd.concat([_h - _l, (_h - _c.shift(1)).abs(), (_l - _c.shift(1)).abs()], axis=1).max(axis=1)
ATR_ARR = _tr.rolling(14).mean().bfill().values
EMA200  = _c.ewm(span=200, adjust=False).mean().values
TS      = df_btc.index
H_ARR   = _h.values
L_ARR   = _l.values
C_ARR   = _c.values

_sp     = 2 * 14 - 1
_hd     = _h.diff(); _ld = _l.diff()
_pdm    = _hd.where((_hd > 0) & (_hd > -_ld), 0.0)
_mdm    = (-_ld).where((-_ld > 0) & (-_ld > _hd), 0.0)
_aw     = _tr.ewm(span=_sp, adjust=False).mean()
_pw     = _pdm.ewm(span=_sp, adjust=False).mean()
_mw     = _mdm.ewm(span=_sp, adjust=False).mean()
_pdi    = 100 * _pw / _aw
_mdi    = 100 * _mw / _aw
_dx     = 100 * (_pdi - _mdi).abs() / (_pdi + _mdi).replace(0, float("nan"))
ADX_ARR = _dx.ewm(span=_sp, adjust=False).mean().fillna(0).values


# ── Simulator (final config only) ─────────────────────────────────────────────
def simulate_final(strategy) -> list[dict]:
    """
    Bar-by-bar simulation: EMA200 + ADX>=20 + ADX-split risk + KZ hours [1,2,3,8].
    Records entry_type (break | retest) from SwingLevelBreakV2 signals.
    """
    balance = float(b2cfg.STARTING_BALANCE)
    trades: list[dict] = []
    open_t = None

    for i in range(220, len(df_btc)):
        bar_time = TS[i]
        hr       = bar_time.hour
        bh_      = float(H_ARR[i])
        bl_      = float(L_ARR[i])
        bc_      = float(C_ARR[i])
        atr_now  = float(ATR_ARR[i])

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
                    pnl = -risk_u;  r_ = -1.0;  ex = "SL"
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
                if long_:
                    if bh_ > open_t["trail_peak"]:
                        open_t["trail_peak"] = bh_
                    open_t["sl"] = max(open_t["trail_peak"] - b2cfg.TRAIL_ATR_MULT * atr_now, entry_)
                else:
                    if bl_ < open_t["trail_peak"]:
                        open_t["trail_peak"] = bl_
                    open_t["sl"] = min(open_t["trail_peak"] + b2cfg.TRAIL_ATR_MULT * atr_now, entry_)

                hit_sl2 = (bl_ <= open_t["sl"]) if long_ else (bh_ >= open_t["sl"])

                if hit_sl2:
                    dist_run = abs(open_t["sl"] - entry_)
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

        # ── Entry gate ───────────────────────────────────────────────────────
        if hr not in b2cfg.KZ_HOURS:
            continue

        adx_v  = float(ADX_ARR[i])
        ema200v = float(EMA200[i])

        if adx_v < b2cfg.ADX_THRESHOLD:
            continue

        win = df_btc.iloc[max(0, i - 220):i + 1]

        for direction in ("long", "short"):
            # EMA200 filter
            if direction == "long"  and bc_ < ema200v: continue
            if direction == "short" and bc_ > ema200v: continue

            sig = strategy.generate_signal(win, bar_time, direction)
            if not sig.get("signal"):
                continue

            sl_d = abs(float(sig["entry"]) - float(sig["sl"]))
            if sl_d <= 0:
                continue

            # ADX-split risk
            if adx_v >= b2cfg.ADX_SPLIT_STRONG_MIN:
                risk_pct = b2cfg.RISK_PCT_STRONG
            elif adx_v <= b2cfg.ADX_SPLIT_EARLY_MAX:
                risk_pct = b2cfg.RISK_PCT_EARLY_TREND
            else:
                risk_pct = b2cfg.RISK_PCT_TRANSITION

            ru   = round(balance * risk_pct, 2)
            lots = ru / sl_d
            tp1r = float(sig.get("tp1_rr", b2cfg.TP1_RR))
            tp2r = float(sig.get("tp2_rr", b2cfg.TP2_RR))
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
                "strategy_used": sig.get("strategy_used", ""),
                "entry_type":    sig.get("entry_type", ""),   # "break" | "retest" | ""
                "signal_reason": sig.get("reason", ""),
                "trail_peak":    0.0,
                "tp1_hit":       False,
                "pnl_running":   0.0,
                "r_running":     0.0,
                "pnl_usd":       0.0,
                "adx_at_entry":  round(adx_v, 1),
                "atr_at_entry":  round(atr_now, 0),
                "sl_atr_mult":   round(sl_d / atr_now, 2) if atr_now > 0 else 0,
            }
            break

    return trades


def _stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "avgr": 0.0, "pnl": 0.0,
                "final_bal": b2cfg.STARTING_BALANCE, "maxdd": 0.0, "pf": 0.0}
    wins   = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] < 0]
    wr     = len(wins) / len(trades) * 100
    avgr   = sum(t["r_multiple"] for t in trades) / len(trades)
    pnl    = sum(t["pnl_usd"] for t in trades)
    gross_win  = sum(t["pnl_usd"] for t in wins)
    gross_loss = abs(sum(t["pnl_usd"] for t in losses))
    pf     = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    bals   = [b2cfg.STARTING_BALANCE] + [t["balance_after"] for t in trades]
    peak   = b2cfg.STARTING_BALANCE
    maxdd  = 0.0
    for b in bals:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100
        if dd > maxdd: maxdd = dd
    return {
        "n":         len(trades),
        "wr":        round(wr, 1),
        "avgr":      round(avgr, 2),
        "pnl":       round(pnl, 2),
        "final_bal": round(bals[-1], 2),
        "maxdd":     round(maxdd, 1),
        "pf":        round(pf, 2),
    }


# ── Run final config ───────────────────────────────────────────────────────────
strat = VBSwingStrategy(
    atr_multiplier   = b2cfg.VB_ATR_MULTIPLIER,
    close_zone       = b2cfg.VB_CLOSE_ZONE,
    swing_entry_mode = b2cfg.SWING_ENTRY_MODE,   # "both"
    swing_max_sl_atr = b2cfg.SWING_MAX_SL_ATR,   # 2.0
)

print("\nRunning FINAL config backtest...")
print(f"  Strategy  : VB + Swing Level Break v2  [mode='{b2cfg.SWING_ENTRY_MODE}' SL_cap={b2cfg.SWING_MAX_SL_ATR}xATR]")
print(f"  Hours     : {b2cfg.KZ_HOURS} UTC")
print(f"  Filters   : EMA200 + ADX>={b2cfg.ADX_THRESHOLD}")
print(f"  Risk      : ADX-split ({int(b2cfg.RISK_PCT_EARLY_TREND*100)}% ADX<={b2cfg.ADX_SPLIT_EARLY_MAX} | "
      f"{int(b2cfg.RISK_PCT_TRANSITION*100)}% ADX {b2cfg.ADX_SPLIT_EARLY_MAX}-{b2cfg.ADX_SPLIT_STRONG_MIN} | "
      f"{int(b2cfg.RISK_PCT_STRONG*100)}% ADX>={b2cfg.ADX_SPLIT_STRONG_MIN})")
print(f"  TP1/TP2   : {b2cfg.TP1_RR}R / {b2cfg.TP2_RR}R  |  Trail: {b2cfg.TRAIL_ATR_MULT}xATR after TP1")
print()

trades = simulate_final(strat)
s      = _stats(trades)

W = 100
SEP = "=" * W

# ── Section 1: Overall summary ─────────────────────────────────────────────────
print(SEP)
print("  SECTION 1: OVERALL PERFORMANCE")
print(f"  BTC Bot 2 FINAL  |  VB + SwingLevelBreak v2 [both 2xATR]  |  {df_btc.index[0].date()} to {df_btc.index[-1].date()}")
print(SEP)
print(f"  Trades     : {s['n']}")
print(f"  Win Rate   : {s['wr']}%")
print(f"  Avg R      : {s['avgr']:+.2f}R")
print(f"  Total PnL  : ${s['pnl']:+,.2f}  (started from ${b2cfg.STARTING_BALANCE:,.0f})")
print(f"  Final Bal  : ${s['final_bal']:+,.2f}")
print(f"  Max DD     : {s['maxdd']}%")
print(f"  Profit Fac : {s['pf']:.2f}")
print()

# ── Section 2: Per-strategy + per-entry-type breakdown ────────────────────────
print(SEP)
print("  SECTION 2: STRATEGY + ENTRY TYPE BREAKDOWN")
print(SEP)

vb_trades     = [t for t in trades if "Volatility"   in t.get("strategy_used", "")]
sl_break      = [t for t in trades if t.get("entry_type") == "break"]
sl_retest     = [t for t in trades if t.get("entry_type") == "retest"]
sl_all        = [t for t in trades if "Swing"        in t.get("strategy_used", "")]

print(f"  {'Source':30s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>10}  {'MaxDD%':>7}")
print("  " + "-" * 72)

for name, tlist in [
    ("Volatility Breakout",           vb_trades),
    ("Swing Level Break v2 (ALL)",    sl_all),
    ("  -> Break entries",             sl_break),
    ("  -> Retest entries",           sl_retest),
]:
    if not tlist:
        print(f"  {name:30s}  no trades")
        continue
    s2 = _stats(tlist)
    print(f"  {name:30s}  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  "
          f"${s2['pnl']:>+9,.2f}  {s2['maxdd']:>6.1f}%")

# Long vs Short
print()
longs  = [t for t in trades if t["direction"] == "long"]
shorts = [t for t in trades if t["direction"] == "short"]
print(f"  {'LONG':30s}  {_stats(longs)['n']:>4}  {_stats(longs)['wr']:>5.1f}%  "
      f"{_stats(longs)['avgr']:>+5.2f}R  ${_stats(longs)['pnl']:>+9,.2f}")
print(f"  {'SHORT':30s}  {_stats(shorts)['n']:>4}  {_stats(shorts)['wr']:>5.1f}%  "
      f"{_stats(shorts)['avgr']:>+5.2f}R  ${_stats(shorts)['pnl']:>+9,.2f}")

# SL distance analysis
if trades:
    sl_mults = [t["sl_atr_mult"] for t in trades if t.get("sl_atr_mult", 0) > 0]
    if sl_mults:
        print()
        print(f"  SL distance analysis (× ATR):")
        print(f"    All trades : avg={np.mean(sl_mults):.2f}x  median={np.median(sl_mults):.2f}x  "
              f"min={np.min(sl_mults):.2f}x  max={np.max(sl_mults):.2f}x")
        vb_sl   = [t["sl_atr_mult"] for t in vb_trades  if t.get("sl_atr_mult", 0) > 0]
        sl_b_sl = [t["sl_atr_mult"] for t in sl_break   if t.get("sl_atr_mult", 0) > 0]
        sl_r_sl = [t["sl_atr_mult"] for t in sl_retest  if t.get("sl_atr_mult", 0) > 0]
        if vb_sl:   print(f"    VB         : avg={np.mean(vb_sl):.2f}x")
        if sl_b_sl: print(f"    SL break   : avg={np.mean(sl_b_sl):.2f}x")
        if sl_r_sl: print(f"    SL retest  : avg={np.mean(sl_r_sl):.2f}x")

print()

# ── Section 3: ADX zone breakdown ─────────────────────────────────────────────
print(SEP)
print("  SECTION 3: ADX ZONE BREAKDOWN  (validates ADX-split risk logic)")
print(SEP)
print(f"  {'ADX Zone':15s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>10}  Risk%  Description")
print("  " + "-" * 68)

adx_zones = [
    ("ADX 20-25",  [t for t in trades if 20 <= t["adx_at_entry"] < 25],  "3%  early trend"),
    ("ADX 25-40",  [t for t in trades if 25 <= t["adx_at_entry"] < 40],  "2%  dead zone"),
    ("ADX 40+",    [t for t in trades if t["adx_at_entry"] >= 40],        "3%  strong trend"),
]
for zone_name, tlist, desc in adx_zones:
    if not tlist:
        print(f"  {zone_name:15s}  no trades")
        continue
    s2 = _stats(tlist)
    print(f"  {zone_name:15s}  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  "
          f"${s2['pnl']:>+9,.2f}  {desc}")

print()

# ── Section 4: Year-half breakdown ────────────────────────────────────────────
print(SEP)
print("  SECTION 4: YEAR-HALF BREAKDOWN")
print(SEP)
print(f"  {'Period':12s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>10}  {'Bal$':>9}  {'MaxDD%':>7}")
print("  " + "-" * 72)

def _half_label(ts):
    return f"{ts.year}-H{'1' if ts.month <= 6 else '2'}"

halves = defaultdict(list)
for t in trades:
    halves[_half_label(t["open_time"])].append(t)

all_halves_profitable = True
for hl in sorted(halves.keys()):
    ts_ = halves[hl]
    s2  = _stats(ts_)
    tag = "  LOSS" if s2["pnl"] < 0 else ""
    if s2["pnl"] < 0:
        all_halves_profitable = False
    print(f"  {hl:12s}  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  "
          f"${s2['pnl']:>+9,.2f}  ${s2['final_bal']:>8,.2f}  {s2['maxdd']:>6.1f}%{tag}")

n_halves = len(halves)
prof_halves = sum(1 for ts_ in halves.values() if _stats(ts_)["pnl"] > 0)
print("  " + "-" * 72)
print(f"  Profitable halves: {prof_halves}/{n_halves}  {'(100%) ALL PROFITABLE' if all_halves_profitable else ''}")
print()

# ── Section 5: Quarterly breakdown ────────────────────────────────────────────
print(SEP)
print("  SECTION 5: QUARTERLY BREAKDOWN")
print(SEP)
print(f"  {'Quarter':10s}  {'#':>4}  {'WR%':>6}  {'AvgR':>6}  {'PnL$':>10}  {'Bal$':>9}  {'MaxDD%':>7}")
print("  " + "-" * 70)

def _quarter_label(ts):
    q = (ts.month - 1) // 3 + 1
    return f"{ts.year}-Q{q}"

quarters = defaultdict(list)
for t in trades:
    quarters[_quarter_label(t["open_time"])].append(t)

all_qtrs_profitable = True
for ql in sorted(quarters.keys()):
    ts_ = quarters[ql]
    s2  = _stats(ts_)
    tag = "  LOSS" if s2["pnl"] < 0 else ""
    if s2["pnl"] < 0:
        all_qtrs_profitable = False
    print(f"  {ql:10s}  {s2['n']:>4}  {s2['wr']:>5.1f}%  {s2['avgr']:>+5.2f}R  "
          f"${s2['pnl']:>+9,.2f}  ${s2['final_bal']:>8,.2f}  {s2['maxdd']:>6.1f}%{tag}")

n_qtrs = len(quarters)
prof_qtrs = sum(1 for ts_ in quarters.values() if _stats(ts_)["pnl"] > 0)
print("  " + "-" * 70)
print(f"  Profitable quarters: {prof_qtrs}/{n_qtrs}  {'(100%) ALL PROFITABLE' if all_qtrs_profitable else ''}")
print()

# ── Section 6: Monthly breakdown ──────────────────────────────────────────────
print(SEP)
print("  SECTION 6: MONTHLY BREAKDOWN")
print(SEP)
print(f"  {'Month':>7}  {'#':>3}  {'WR%':>5}  {'AvgR':>6}  {'PnL$':>9}  {'Bal$':>8}  "
      f"{'VB':>3}  {'Brk':>3}  {'Ret':>3}  {'L/S':>6}  Notes")
print("  " + "-" * 88)

monthly = defaultdict(list)
for t in trades:
    ym = str(t["open_time"])[:7]
    monthly[ym].append(t)

losing_months    = 0
max_consec_loss  = 0
cur_consec_loss  = 0

for ym in sorted(monthly.keys()):
    ts_  = monthly[ym]
    s2   = _stats(ts_)
    vb_  = len([t for t in ts_ if "Volatility" in t.get("strategy_used", "")])
    brk_ = len([t for t in ts_ if t.get("entry_type") == "break"])
    ret_ = len([t for t in ts_ if t.get("entry_type") == "retest"])
    lo_  = len([t for t in ts_ if t["direction"] == "long"])
    sh_  = len([t for t in ts_ if t["direction"] == "short"])

    if s2["pnl"] < 0:
        losing_months  += 1
        cur_consec_loss += 1
        max_consec_loss  = max(max_consec_loss, cur_consec_loss)
        note = "LOSS"
    else:
        cur_consec_loss = 0
        note = ""

    if s2["wr"] >= 60:  note += " HIGH-WR"
    if s2["pnl"] >= 5000: note += " BIG"

    print(f"  {ym:>7}  {s2['n']:>3}  {s2['wr']:>4.1f}%  {s2['avgr']:>+5.2f}R  "
          f"${s2['pnl']:>+8,.2f}  ${s2['final_bal']:>7,.2f}  "
          f"{vb_:>3}  {brk_:>3}  {ret_:>3}  {lo_}L/{sh_}S  {note}")

n_months    = len(monthly)
prof_months = n_months - losing_months

total = _stats(trades)
print("  " + "-" * 88)
print(f"  {'TOTAL':>7}  {total['n']:>3}  {total['wr']:>4.1f}%  {total['avgr']:>+5.2f}R  "
      f"${total['pnl']:>+8,.2f}  ${total['final_bal']:>7,.2f}")
print()

# ── Section 7: Consistency scorecard ──────────────────────────────────────────
print(SEP)
print("  SECTION 7: CONSISTENCY SCORECARD")
print(SEP)

# Monthly Sharpe (monthly returns)
monthly_rets = []
for ym in sorted(monthly.keys()):
    monthly_rets.append(_stats(monthly[ym])["pnl"])

if len(monthly_rets) > 1:
    arr = np.array(monthly_rets)
    sharpe = float(np.mean(arr) / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0.0
else:
    sharpe = 0.0

# Calmar = total_pnl / max_drawdown
calmar = round(s["pnl"] / (s["maxdd"] / 100 * b2cfg.STARTING_BALANCE), 1) if s["maxdd"] > 0 else float("inf")

print(f"  Profitable halves       : {prof_halves}/{n_halves}  ({prof_halves/n_halves*100:.0f}%)")
print(f"  Profitable quarters     : {prof_qtrs}/{n_qtrs}  ({prof_qtrs/n_qtrs*100:.0f}%)")
print(f"  Profitable months       : {prof_months}/{n_months}  ({prof_months/n_months*100:.0f}%)")
print(f"  Max consec losing months: {max_consec_loss}")
print(f"  Profit Factor           : {s['pf']:.2f}")
print(f"  Monthly Sharpe          : {sharpe:.2f}")
print(f"  Calmar Ratio            : {calmar:.1f}")
print()
print(f"  Overall  : {s['n']} trades | WR {s['wr']}% | AvgR {s['avgr']:+.2f}R | "
      f"PnL ${s['pnl']:+,.0f} | MaxDD {s['maxdd']}%")
print()

# Summary verdict
print(SEP)
print("  FINAL CONFIG VERDICT")
print(SEP)
print(f"  Config    : VB + Swing Level Break v2  [both 2xATR]")
print(f"  Hours     : {b2cfg.KZ_HOURS} UTC  (Asia Night + EU Open)")
print(f"  Filters   : EMA200 + ADX>={b2cfg.ADX_THRESHOLD} + ADX-split risk")
print(f"  Trades    : {s['n']}")
print(f"  Win Rate  : {s['wr']}%")
print(f"  Avg R     : {s['avgr']:+.2f}R")
print(f"  PnL       : ${s['pnl']:+,.0f}  (from ${b2cfg.STARTING_BALANCE:,.0f} start)")
print(f"  Max DD    : {s['maxdd']}%")
print(f"  PF        : {s['pf']:.2f}")
print(f"  Halves    : {prof_halves}/{n_halves} ({prof_halves/n_halves*100:.0f}%)")
print(f"  Quarters  : {prof_qtrs}/{n_qtrs} ({prof_qtrs/n_qtrs*100:.0f}%)")
print(f"  Months    : {prof_months}/{n_months} ({prof_months/n_months*100:.0f}%)")
print(f"  Max consec loss months: {max_consec_loss}")
print()
print("  RUN: python -m btc_research.btc_bot_2.main   to start the live bot")
print(SEP)
