"""
v2/backtest/wti_day_analysis.py

Deep-dive WTI backtest for a specific date using the LIVE v2 signal engine
(ConfluenceEngine → NYMomentumWTIStrategy).  No ML training, no journal writes.
Just shows exactly what signals were generated and how each played out.

Run from repo root:
    python -m v2.backtest.wti_day_analysis              # defaults 2026-05-29
    python -m v2.backtest.wti_day_analysis 2026-05-28   # any date
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path
from datetime import timezone

import pandas as pd

# ── Logging: suppress everything except this script ──────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("wti_day_analysis")
logger.setLevel(logging.INFO)

TARGET_DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-05-29"

W = 95   # print width

# ── Connect MT5 + fetch data ──────────────────────────────────────────────────
from v2 import settings
from v2.connectors.unified_data import DataFeed
from v2.signals.confluence_engine import ConfluenceEngine

print(f"\n{'='*W}")
print(f"  WTI Day Analysis — {TARGET_DATE}")
print(f"  Strategy : London Range Breakout (NYMomentumWTI) — live v2 signal engine")
print(f"{'='*W}")

feed = DataFeed()
print(f"\n  Connecting to MT5...")
status = feed.connect(
    mt5_login    = settings.MT5_LOGIN,
    mt5_password = settings.MT5_PASSWORD,
    mt5_server   = settings.MT5_SERVER,
)
if not status.get("mt5"):
    print("  ERROR: MT5 not connected. Make sure MT5 is running on the VPS.")
    sys.exit(1)
print(f"  MT5 connected ✓")

# Fetch enough bars: ~30 days for H1 lookback + H4 + D1
BARS_H1 = 30 * 24   # 720 H1 bars (~1 month)
BARS_H4 = 200
BARS_D1 = 90

print(f"  Fetching WTI H1/H4/D1 data...")
df_h1 = feed.get_ohlcv("WTI", "H1", BARS_H1)
df_h4 = feed.get_ohlcv("WTI", "H4", BARS_H4)
df_d1 = feed.get_ohlcv("WTI", "D1", BARS_D1)

if df_h1 is None or df_h1.empty:
    print("  ERROR: No H1 data returned for WTI.")
    sys.exit(1)

# Normalise time column to UTC-aware timestamps
for df_ in (df_h1, df_h4, df_d1):
    if df_ is not None and not df_.empty and "time" in df_.columns:
        df_["time"] = pd.to_datetime(df_["time"], utc=True)

df_h1 = df_h1.sort_values("time").reset_index(drop=True)
print(f"  H1 data  : {len(df_h1)} bars  ({df_h1['time'].iloc[0].date()} → {df_h1['time'].iloc[-1].date()})")
if df_h4 is not None and not df_h4.empty:
    print(f"  H4 data  : {len(df_h4)} bars")
if df_d1 is not None and not df_d1.empty:
    print(f"  D1 data  : {len(df_d1)} bars")

# ── Filter to target date ─────────────────────────────────────────────────────
target_ts   = pd.Timestamp(TARGET_DATE, tz="UTC")
target_date = target_ts.date()
times_utc   = df_h1["time"].dt.tz_convert("UTC")
day_mask    = times_utc.dt.date == target_date

if not day_mask.any():
    print(f"\n  No H1 bars found for {TARGET_DATE} in the fetched data.")
    print(f"  Available range: {df_h1['time'].iloc[0].date()} → {df_h1['time'].iloc[-1].date()}")
    sys.exit(1)

# ── London range (08-13 UTC) ──────────────────────────────────────────────────
LONDON_START = 8
LONDON_END   = 13
NY_START     = 13
NY_END       = 17
MAX_HOLD     = 96   # bars forward to scan for outcome

london_mask = day_mask & (times_utc.dt.hour >= LONDON_START) & (times_utc.dt.hour < LONDON_END)
london_bars = df_h1[london_mask]

print(f"\n{'─'*W}")
print(f"  LONDON SESSION  08:00–13:00 UTC  ({target_date})")
print(f"{'─'*W}")

if len(london_bars) < 3:
    print(f"  ⚠  Only {len(london_bars)} London bar(s) found — not enough to define a range.")
    london_valid = False
else:
    london_valid = True
    london_high  = float(london_bars["high"].max())
    london_low   = float(london_bars["low"].min())
    london_range = london_high - london_low

    print(f"  {'Time (UTC)':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8}", end="")
    print(f"  {'Vol':>8}" if "volume" in london_bars.columns else "")
    print(f"  {'-'*70}")
    for _, r in london_bars.iterrows():
        t_str = str(r["time"])[11:16]
        vol_s = f"  {int(r.get('volume', 0)):>8,}" if "volume" in london_bars.columns else ""
        print(f"  {t_str:<12} {r['open']:>8.3f} {r['high']:>8.3f} {r['low']:>8.3f} {r['close']:>8.3f}{vol_s}")

    print(f"\n  London High  : {london_high:.3f}")
    print(f"  London Low   : {london_low:.3f}")
    print(f"  London Range : {london_range:.3f}")

# ── NY signal scan (13-17 UTC) ────────────────────────────────────────────────
print(f"\n{'─'*W}")
print(f"  NY / NYMEX WINDOW  13:00–17:00 UTC  (signal window)")
print(f"{'─'*W}")

ny_mask   = day_mask & (times_utc.dt.hour >= NY_START) & (times_utc.dt.hour < NY_END)
ny_idxs   = df_h1.index[ny_mask].tolist()

if not ny_idxs:
    print(f"  No bars in NY window on {target_date}.")
    sys.exit(0)

engine       = ConfluenceEngine()
signals_taken = []

print(f"  Evaluating {len(ny_idxs)} bar(s) with live ConfluenceEngine...\n")
print(f"  {'Time':>5}  {'Close':>7}  {'High':>7}  {'Low':>7}  {'Dir':>6}  {'Score':>5}  Signal")
print(f"  {'-'*80}")

for bar_idx in ny_idxs:
    row       = df_h1.iloc[bar_idx]
    bar_time  = row["time"]

    # Slice history up to this bar (no look-ahead)
    window_h1 = df_h1.iloc[:bar_idx + 1].copy()
    window_h4 = df_h4[df_h4["time"] <= bar_time].copy() if (df_h4 is not None and not df_h4.empty) else None
    window_d1 = df_d1[df_d1["time"] <= bar_time].copy() if (df_d1 is not None and not df_d1.empty) else None

    bt_context = {"bar_time": bar_time}
    t_str      = str(bar_time)[11:16]
    close      = float(row["close"])
    high       = float(row["high"])
    low_       = float(row["low"])

    for direction in ("long", "short"):
        result = engine.score("WTI", direction, window_h1, window_h4, window_d1, bt_context)
        score  = result.get("score", 0)
        signal = result.get("signal", False)

        dir_arrow = "▲ LONG" if direction == "long" else "▼ SHORT"
        sig_str   = ""

        if signal:
            entry = float(result.get("entry_price") or close)
            sl    = float(result.get("stop_loss",  0))
            tp1   = float(result.get("tp1_price",  0))
            tp2   = float(result.get("tp2_price",  0))
            sig_str = (f"  ✓ SIGNAL  entry={entry:.3f}  SL={sl:.3f}  "
                       f"TP1={tp1:.3f}  TP2={tp2:.3f}")
            signals_taken.append({
                "bar_idx": bar_idx, "bar_time": bar_time,
                "direction": direction,
                "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
                "score": score, "result": result,
            })
        elif score >= 5.0:
            sig_str = f"  (score {score:.1f} — below threshold 7.0)"
        else:
            sig_str = f"  —"

        print(f"  {t_str:>5}  {close:>7.3f}  {high:>7.3f}  {low_:>7.3f}  "
              f"{dir_arrow:>6}  {score:>5.1f}{sig_str}")

        if signal:
            # Print reasons
            reasons = result.get("reasons") or result.get("factors", {})
            if isinstance(reasons, list):
                for r in reasons:
                    if r:
                        print(f"              {'':>50} → {r}")
            break  # don't also evaluate short on same bar if long triggered

# ── Outcomes ──────────────────────────────────────────────────────────────────
if not signals_taken:
    print(f"\n  No signals generated on {TARGET_DATE}.")
    print(f"  This means: either the London range was invalid, no breakout occurred,")
    print(f"  or the breakout failed the score threshold (<7.0).")
    sys.exit(0)

print(f"\n{'='*W}")
print(f"  TRADE OUTCOME(S)")
print(f"{'='*W}")

for sig in signals_taken:
    bar_idx   = sig["bar_idx"]
    entry     = sig["entry"]
    sl        = sig["sl"]
    tp1       = sig["tp1"]
    tp2       = sig["tp2"]
    direction = sig["direction"]
    is_long   = direction == "long"
    sl_dist   = abs(entry - sl)

    print(f"\n  Signal  : {direction.upper()}  @  {str(sig['bar_time'])[11:16]} UTC"
          f"  (score {sig['score']:.1f})")
    print(f"  Entry   : {entry:.3f}")
    print(f"  SL      : {sl:.3f}  (dist = {sl_dist:.3f})")
    print(f"  TP1     : {tp1:.3f}  (+{abs(tp1-entry):.3f} = 2×R)")
    print(f"  TP2     : {tp2:.3f}  (+{abs(tp2-entry):.3f} = 5×R)")

    # Walk future bars
    future = df_h1.iloc[bar_idx + 1: bar_idx + 1 + MAX_HOLD]
    if future.empty:
        print(f"  No future bars to evaluate outcome.")
        continue

    print(f"\n  Bar-by-bar after entry (showing first 24 + exit bar):")
    print(f"  {'#':>3}  {'Time':>5}  {'High':>7}  {'Low':>7}  {'Close':>7}  Status")
    print(f"  {'-'*65}")

    tp1_hit    = False
    exit_event = None

    for n, (_, frow) in enumerate(future.iterrows(), 1):
        fh    = float(frow["high"])
        fl    = float(frow["low"])
        fc    = float(frow["close"])
        ft    = str(frow["time"])[11:16]
        status = ""

        if exit_event is None:
            if is_long:
                if fl <= sl:
                    reason = "SL_AFTER_TP1 (breakeven)" if tp1_hit else "SL ❌"
                    exit_p = max(sl, entry) if tp1_hit else sl
                    exit_event = (n, ft, exit_p, reason, tp1_hit)
                    status = f"  ← {reason}  @ {exit_p:.3f}"
                elif not tp1_hit and fh >= tp1:
                    tp1_hit = True
                    sl      = entry
                    status  = f"  ← TP1 ✓  @ {tp1:.3f}  |  SL → BE {entry:.3f}"
                elif tp1_hit and fh >= tp2:
                    exit_event = (n, ft, tp2, "TP2 ✅", True)
                    status = f"  ← TP2 ✅  @ {tp2:.3f}"
            else:
                if fh >= sl:
                    reason = "SL_AFTER_TP1 (breakeven)" if tp1_hit else "SL ❌"
                    exit_p = min(sl, entry) if tp1_hit else sl
                    exit_event = (n, ft, exit_p, reason, tp1_hit)
                    status = f"  ← {reason}  @ {exit_p:.3f}"
                elif not tp1_hit and fl <= tp1:
                    tp1_hit = True
                    sl      = entry
                    status  = f"  ← TP1 ✓  @ {tp1:.3f}  |  SL → BE {entry:.3f}"
                elif tp1_hit and fl <= tp2:
                    exit_event = (n, ft, tp2, "TP2 ✅", True)
                    status = f"  ← TP2 ✅  @ {tp2:.3f}"

        if n <= 24 or status:
            print(f"  {n:>3}  {ft:>5}  {fh:>7.3f}  {fl:>7.3f}  {fc:>7.3f}{status}")

        if exit_event:
            break
    else:
        # MAX_HOLD reached
        lp     = float(future["close"].iloc[-1])
        pnl_d  = (lp - entry) if is_long else (entry - lp)
        pnl_r  = round(pnl_d / sl_dist, 2) if sl_dist else 0
        exit_event = (len(future), str(future["time"].iloc[-1])[11:16],
                      lp, f"MAX_HOLD ({pnl_r:+.2f}R)", tp1_hit)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_exit, t_exit, p_exit, reason_exit, tp1_was = exit_event
    price_diff = (p_exit - entry) if is_long else (entry - p_exit)
    r_achieved = round(price_diff / sl_dist, 2) if sl_dist else 0

    # P&L for $1,500 account
    balance    = 1500.0
    risk_2pct  = balance * 0.02
    risk_3pct  = balance * 0.03
    pnl_2pct   = round(risk_2pct  * r_achieved, 2) if r_achieved > 0 else round(-risk_2pct, 2)
    pnl_3pct   = round(risk_3pct  * r_achieved, 2) if r_achieved > 0 else round(-risk_3pct, 2)

    print(f"\n  {'─'*65}")
    print(f"  RESULT    : {reason_exit}")
    print(f"  Exit      : {p_exit:.3f}  (bar {n_exit}, ~{t_exit} UTC)")
    print(f"  R achieved: {r_achieved:+.2f}R")
    print(f"  P&L @ 2% risk ($1,500) : ${pnl_2pct:+.2f}")
    print(f"  P&L @ 3% risk ($1,500) : ${pnl_3pct:+.2f}")
    print(f"  {'─'*65}")

    if "SL" in reason_exit and "TP1" not in reason_exit:
        print(f"\n  POST-MORTEM:")
        if n_exit <= 2:
            print(f"  → Stopped out in {n_exit} bar(s) — immediate rejection / fake breakout")
        elif n_exit <= 6:
            print(f"  → Stopped in {n_exit} bars — quick reversal, breakout had no follow-through")
        else:
            print(f"  → Ran {n_exit} bars before stopping — slow grind-back, direction was right but timing off")
        print(f"  → 1 SL is normal. Strategy expects ~48–52% losses by design.")

print(f"\n{'='*W}\n")
