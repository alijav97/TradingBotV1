"""
btc_research/scan_day_btc.py — Single-day BTC backtest using Version D strategy.

Mirrors scripts/scan_day.py for WTI but uses BTC Version D logic:
  - Kill-zone: 21:00-24:00 UTC
  - EMA200 filter (longs above EMA200 only, shorts below only)
  - ADX filter (skip if ADX < 20)
  - Flipped risk: 3% if ADX 20-28, 2% if ADX > 28
  - CombinedStrategy (Volatility Breakout > Swing Level > Morning Range)
  - Trailing SL at 2×ATR after TP1

Usage:
    python btc_research/scan_day_btc.py            # defaults to yesterday
    python btc_research/scan_day_btc.py 2026-05-29
"""
from __future__ import annotations

import sys
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# ── Project root setup ────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import pandas as pd
import numpy as np

import btc_research.settings as cfg
from btc_research.data.fetcher import fetch_all
from btc_research.strategies.combined import CombinedStrategy

# ── Settings ──────────────────────────────────────────────────────────────────
STARTING_BALANCE   = 500.0
RISK_PCT           = 0.02       # 2% — ADX > 28
RISK_PCT_EARLY     = 0.03       # 3% — ADX 20-28 (early trend)
ADX_THRESHOLD      = 20
ADX_EARLY_MAX      = 28
EMA200_PERIOD      = 200
ADX_PERIOD         = 14
TRAIL_ATR_MULT     = 2.0
TP1_RR             = 2.0
TP2_RR             = 5.0
MAX_HOLD_BARS      = 96
KZ_START           = cfg.KZ_START_UTC   # 21
KZ_END             = cfg.KZ_END_UTC     # 24


# ── Date argument ─────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    try:
        target_date = date.fromisoformat(sys.argv[1])
    except ValueError:
        print(f"ERROR: invalid date '{sys.argv[1]}' — use YYYY-MM-DD")
        sys.exit(1)
else:
    target_date = date.today() - timedelta(days=1)

now_utc = datetime.now(timezone.utc)
is_today      = target_date == now_utc.date()
is_future     = target_date > now_utc.date()

if is_future:
    print(f"ERROR: {target_date} is in the future — can't backtest future data")
    sys.exit(1)

print("=" * 70)
print(f"  BTC Bot 1 — Day Scan: {target_date}  (Version D Strategy)")
print("=" * 70)

# ── Fetch data ────────────────────────────────────────────────────────────────
print("\nFetching data...")
data   = fetch_all(use_cache=True, force_refresh=is_today)
df_btc = data.get(cfg.BTC_SYMBOL, pd.DataFrame())

if df_btc.empty:
    print("ERROR: No BTCUSD data. Make sure MT5 is running.")
    sys.exit(1)

df_btc["time"] = pd.to_datetime(df_btc["time"], utc=True)
df_btc = df_btc.sort_values("time").reset_index(drop=True)

# ── Pre-compute indicators ────────────────────────────────────────────────────
close_s = df_btc["close"].astype(float)
high_s  = df_btc["high"].astype(float)
low_s   = df_btc["low"].astype(float)

# EMA200
ema200_series = close_s.ewm(span=EMA200_PERIOD, adjust=False).mean()

# ATR(14)
tr = pd.concat([
    high_s - low_s,
    (high_s - close_s.shift(1)).abs(),
    (low_s  - close_s.shift(1)).abs(),
], axis=1).max(axis=1)
atr_series = tr.rolling(ADX_PERIOD).mean()

# ADX(14)
sp   = 2 * ADX_PERIOD - 1
hd   = high_s.diff()
ld   = low_s.diff()
pdm  = hd.where((hd > 0) & (hd > -ld), 0.0)
mdm  = (-ld).where((-ld > 0) & (-ld > hd), 0.0)
aw   = tr.ewm(span=sp, adjust=False).mean()
pw   = pdm.ewm(span=sp, adjust=False).mean()
mw   = mdm.ewm(span=sp, adjust=False).mean()
pdi  = 100 * pw / aw
mdi  = 100 * mw / aw
dx   = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, float("nan"))
adx_series = dx.ewm(span=sp, adjust=False).mean().fillna(0)

# ── Find kill-zone bars on target date ────────────────────────────────────────
kz_mask = (
    (df_btc["time"].dt.date == target_date) &
    (df_btc["time"].dt.hour >= KZ_START) &
    (df_btc["time"].dt.hour < KZ_END)
)
kz_bars = df_btc[kz_mask]

print(f"\nKill-zone bars ({KZ_START}:00-{KZ_END}:00 UTC) on {target_date}: {len(kz_bars)} bars")

if kz_bars.empty:
    print(f"  No kill-zone bars found for {target_date}")
    print("  (Market may have been closed or data not available)")
    sys.exit(0)

# ── Check pre-KZ consolidation range (17-21 UTC = MR) ────────────────────────
mr_mask = (
    (df_btc["time"].dt.date == target_date) &
    (df_btc["time"].dt.hour >= cfg.MR_START_UTC) &
    (df_btc["time"].dt.hour < cfg.MR_END_UTC)
)
mr_bars  = df_btc[mr_mask]
mr_high  = float(mr_bars["high"].max()) if not mr_bars.empty else 0
mr_low   = float(mr_bars["low"].min())  if not mr_bars.empty else 0

print(f"\nPre-KZ Range ({cfg.MR_START_UTC}:00-{cfg.MR_END_UTC}:00 UTC):")
if mr_bars.empty:
    print("  No pre-KZ range bars found")
else:
    print(f"  High: ${mr_high:,.2f}  |  Low: ${mr_low:,.2f}  |  Range: ${mr_high - mr_low:,.2f}  |  Bars: {len(mr_bars)}")

# ── Show each kill-zone bar ───────────────────────────────────────────────────
strat   = CombinedStrategy(atr_multiplier=1.2, close_zone=0.45, range_bars=6)
balance = STARTING_BALANCE

print(f"\n{'─'*70}")
print(f"  KILL-ZONE BAR-BY-BAR ANALYSIS")
print(f"{'─'*70}")

signals_found = []

for idx in kz_bars.index:
    row       = df_btc.loc[idx]
    bar_time  = pd.Timestamp(row["time"])
    bar_close = float(row["close"])
    bar_high  = float(row["high"])
    bar_low   = float(row["low"])
    utc_hour  = bar_time.hour

    ema200    = float(ema200_series.loc[idx])
    adx       = float(adx_series.loc[idx])
    atr       = float(atr_series.loc[idx]) if not pd.isna(atr_series.loc[idx]) else 0

    above_ema = bar_close > ema200

    print(f"\n  Bar: {bar_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Close: ${bar_close:,.2f}  |  H: ${bar_high:,.2f}  L: ${bar_low:,.2f}")
    print(f"  EMA200: ${ema200:,.2f}  ({'ABOVE ✅' if above_ema else 'BELOW ✅'})")
    print(f"  ADX: {adx:.1f}  |  ATR: ${atr:,.2f}")

    # ADX filter
    if adx < ADX_THRESHOLD:
        print(f"  ❌ ADX {adx:.1f} < {ADX_THRESHOLD} — no clear trend, skip")
        continue

    # Flipped risk
    risk_pct = RISK_PCT_EARLY if adx <= ADX_EARLY_MAX else RISK_PCT
    print(f"  Risk: {risk_pct*100:.0f}%  (ADX {'20-28 early trend' if adx <= ADX_EARLY_MAX else '>28 extended'})")

    # Run strategy for each direction
    df_window = df_btc.iloc[:idx + 1]

    for direction in ("long", "short"):
        # EMA200 filter
        if direction == "long" and not above_ema:
            print(f"  {direction.upper():5s}: ❌ EMA200 filter — price below EMA200")
            continue
        if direction == "short" and above_ema:
            print(f"  {direction.upper():5s}: ❌ EMA200 filter — price above EMA200")
            continue

        sig = strat.generate_signal(df_window, bar_time, direction)

        if not sig.get("signal"):
            print(f"  {direction.upper():5s}: ❌ {sig.get('reason', 'no signal')}")
            continue

        # Signal fired!
        entry   = float(sig["entry"])
        sl      = float(sig["sl"])
        sl_dist = abs(entry - sl)
        tp1     = round(entry + TP1_RR * sl_dist, 2) if direction == "long" else round(entry - TP1_RR * sl_dist, 2)
        tp2     = round(entry + TP2_RR * sl_dist, 2) if direction == "long" else round(entry - TP2_RR * sl_dist, 2)
        risk_usd = round(balance * risk_pct, 2)
        lots     = round(risk_usd / sl_dist, 6) if sl_dist > 0 else 0
        strategy_name = sig.get("strategy_used", "?")
        needs_filter  = sig.get("needs_im_filter", False)

        print(f"\n  {'🟢' if direction == 'long' else '🔴'} SIGNAL: {direction.upper()} via {strategy_name}")
        print(f"  Entry:    ${entry:,.2f}")
        print(f"  SL:       ${sl:,.2f}  (${sl_dist:,.2f} distance)")
        print(f"  TP1 (2R): ${tp1:,.2f}")
        print(f"  TP2 (5R): ${tp2:,.2f}")
        print(f"  Lots:     {lots:.4f} BTC  |  Risk: ${risk_usd:.2f} ({risk_pct*100:.0f}%)")
        print(f"  IM Filter needed: {'Yes' if needs_filter else 'No'}")

        # ── Outcome: check future bars ────────────────────────────────────────
        future_bars = df_btc.iloc[idx + 1:]
        outcome  = "OPEN (no future data)"
        exit_p   = None
        exit_r   = None
        tp1_hit  = False
        pnl      = None

        for _, frow in future_bars.iterrows():
            fh = float(frow["high"])
            fl = float(frow["low"])

            if not tp1_hit:
                # SL check
                if (direction == "long" and fl <= sl) or (direction == "short" and fh >= sl):
                    exit_p  = sl
                    exit_r  = -1.0
                    pnl     = round(-risk_usd, 2)
                    outcome = "SL"
                    break
                # TP1 check
                if (direction == "long" and fh >= tp1) or (direction == "short" and fl <= tp1):
                    tp1_hit = True
                    sl      = entry   # move to breakeven
                    pnl_tp1 = round(risk_usd * TP1_RR, 2)
                    balance += pnl_tp1
                    continue
            else:
                # After TP1: check BE SL
                if (direction == "long" and fl <= sl) or (direction == "short" and fh >= sl):
                    exit_p  = sl
                    exit_r  = round(abs(sl - entry) / (abs(entry - float(sig["sl"]))), 2) if sl != entry else 0
                    pnl     = round(pnl_tp1 - risk_usd + (risk_usd * exit_r * 0.5 if exit_r > 0 else 0), 2)
                    outcome = "SL_AFTER_TP1 (BE)"
                    break
                # TP2 check
                if (direction == "long" and fh >= tp2) or (direction == "short" and fl <= tp2):
                    exit_p  = tp2
                    exit_r  = TP2_RR
                    pnl_tp2 = round(risk_usd * TP2_RR * 0.5, 2)
                    pnl     = round(pnl_tp1 + pnl_tp2, 2)
                    outcome = "TP2"
                    balance += pnl_tp2
                    break
        else:
            if tp1_hit:
                outcome = "TP1 hit — still open / trailing"

        print(f"\n  Outcome:  {outcome}")
        if exit_p:
            print(f"  Exit:     ${exit_p:,.2f}")
        if pnl is not None:
            sign = "+" if pnl >= 0 else ""
            print(f"  P&L:      {sign}${pnl:.2f}  ({sign}{pnl/balance*100:.1f}% of balance)")
        elif tp1_hit:
            print(f"  TP1 hit — partial profit banked, trailing stop active")

        signals_found.append({
            "bar_time":  bar_time,
            "direction": direction,
            "entry":     entry,
            "sl":        sl,
            "tp1":       tp1,
            "tp2":       tp2,
            "outcome":   outcome,
            "pnl":       pnl or 0,
            "strategy":  strategy_name,
            "adx":       adx,
            "risk_pct":  risk_pct,
        })
        break   # one trade per bar

print(f"\n{'='*70}")
print(f"  SUMMARY — {target_date}")
print(f"{'='*70}")

if not signals_found:
    print(f"  No signals fired during kill-zone on {target_date}")
    print(f"  (Filters: EMA200 alignment + ADX >= {ADX_THRESHOLD} + strategy pattern)")
else:
    total_pnl = sum(s["pnl"] for s in signals_found)
    wins  = [s for s in signals_found if s["pnl"] > 0]
    print(f"  Signals: {len(signals_found)}  |  Wins: {len(wins)}  |  Total P&L: ${total_pnl:+.2f}")
    for s in signals_found:
        sign = "+" if s["pnl"] >= 0 else ""
        print(f"  {s['bar_time'].strftime('%H:%M UTC')}  {s['direction'].upper():5s}  "
              f"@ ${s['entry']:,.2f}  ADX={s['adx']:.1f}  risk={s['risk_pct']*100:.0f}%  "
              f"{s['outcome']:20s}  P&L: {sign}${s['pnl']:.2f}")

print()
