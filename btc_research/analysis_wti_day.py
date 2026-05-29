"""
btc_research/analysis_wti_day.py

Deep-dive on WTI for a specific date using the live NYMomentumWTI strategy.
Shows: London range, breakout signal, entry/SL/TP, bar-by-bar outcome.

Run:
    python btc_research/analysis_wti_day.py              # defaults to 2026-05-29
    python btc_research/analysis_wti_day.py 2026-05-28   # any date
"""
import sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(str(_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime, timezone

TARGET_DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-05-29"

# ── Fetch WTI data ────────────────────────────────────────────────────────────
print(f"\nFetching WTI (SpotCrude) H1 data...")
from btc_research.data.fetcher import fetch_symbol

df = fetch_symbol("SpotCrude", "H1", use_cache=True, force_refresh=False)
if df.empty:
    print("ERROR: No SpotCrude data in cache. Make sure MT5 has run at least once.")
    sys.exit(1)

# Ensure UTC datetime index
if "time" in df.columns:
    df = df.set_index(pd.to_datetime(df["time"], utc=True)).drop(columns=["time"])
elif not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index, utc=True)

print(f"  Data range : {df.index[0].date()} → {df.index[-1].date()} ({len(df):,} H1 bars)")

# ── Session constants (from NYMomentumWTI) ────────────────────────────────────
LONDON_START = 8   # UTC
LONDON_END   = 13  # UTC
NY_START     = 13  # UTC
NY_END       = 17  # UTC

MIN_LONDON_BARS   = 3
MIN_RANGE_ATR_PCT = 0.25
RETEST_TOLERANCE  = 0.8
BREAKOUT_CHASE    = 1.5

# ── ATR helper ────────────────────────────────────────────────────────────────
def calc_atr(df_slice, period=14):
    h = df_slice["high"].astype(float)
    l = df_slice["low"].astype(float)
    c = df_slice["close"].astype(float)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

W = 90
target_dt = pd.Timestamp(TARGET_DATE, tz="UTC")
target_date = target_dt.date()

print(f"\n{'='*W}")
print(f"  WTI (SpotCrude) — {TARGET_DATE}  |  Session: London 08-13 UTC + NY 13-17 UTC")
print(f"  Strategy: London Range Breakout (live NYMomentumWTI logic)")
print(f"{'='*W}")

# ── Extract relevant bars ─────────────────────────────────────────────────────
# London bars: today 08-12 UTC
# NY bars (signals): today 13-16 UTC
# Future bars (outcome): after entry up to 96h
date_mask = df.index.date == target_date
day_bars  = df[date_mask].copy()

if day_bars.empty:
    print(f"\n  No WTI data found for {TARGET_DATE}. Check cache.")
    sys.exit(1)

print(f"\n  Total H1 bars on {TARGET_DATE}: {len(day_bars)}")

# ── London session range ──────────────────────────────────────────────────────
london_mask  = (day_bars.index.hour >= LONDON_START) & (day_bars.index.hour < LONDON_END)
london_bars  = day_bars[london_mask]

print(f"\n{'─'*W}")
print(f"  LONDON SESSION (08:00–13:00 UTC) — Range Formation")
print(f"{'─'*W}")

if len(london_bars) < MIN_LONDON_BARS:
    print(f"  ⚠  Only {len(london_bars)} London bars (need ≥{MIN_LONDON_BARS}) — range undefined")
    london_high = london_low = None
else:
    london_high  = float(london_bars["high"].max())
    london_low   = float(london_bars["low"].min())
    london_range = london_high - london_low

    # Print each London bar
    print(f"  {'Time (UTC)':<14}  {'Open':>8}  {'High':>8}  {'Low':>8}  {'Close':>8}  {'Vol':>8}")
    print(f"  {'-'*65}")
    for ts, row in london_bars.iterrows():
        vol_str = f"{int(row.get('volume', 0)):,}" if "volume" in row else "—"
        print(f"  {str(ts)[11:16]:<14}  {row['open']:>8.3f}  {row['high']:>8.3f}"
              f"  {row['low']:>8.3f}  {row['close']:>8.3f}  {vol_str:>8}")

    # ATR from last 50 bars before NY open
    pre_ny_idx = df.index.get_indexer([day_bars[day_bars.index.hour < NY_START].index[-1]], method="nearest")[0]
    atr_slice  = df.iloc[max(0, pre_ny_idx - 50): pre_ny_idx + 1]
    atr        = calc_atr(atr_slice)

    print(f"\n  London High  : {london_high:.3f}")
    print(f"  London Low   : {london_low:.3f}")
    print(f"  London Range : {london_range:.3f}  (ATR={atr:.3f}, min={atr * MIN_RANGE_ATR_PCT:.3f})")

    if london_range < atr * MIN_RANGE_ATR_PCT:
        print(f"  ⚠  Range too tight vs ATR — flat London session, low-confidence setup")
    else:
        print(f"  ✓  Range valid (≥ {MIN_RANGE_ATR_PCT*100:.0f}% ATR)")

# ── NY session — signal scan ──────────────────────────────────────────────────
print(f"\n{'─'*W}")
print(f"  NY SESSION (13:00–17:00 UTC) — Signal Window")
print(f"{'─'*W}")

ny_mask = (day_bars.index.hour >= NY_START) & (day_bars.index.hour < NY_END)
ny_bars = day_bars[ny_mask]

if ny_bars.empty:
    print(f"  No bars in NY window on {TARGET_DATE}")
else:
    print(f"  {'Time':>5}  {'Close':>7}  {'High':>7}  {'Low':>7}  {'Signal':}")
    print(f"  {'-'*70}")

    signal_found = None

    for ts, row in ny_bars.iterrows():
        close = float(row["close"])
        high  = float(row["high"])
        low_  = float(row["low"])

        # Determine signal
        sig_str = "  —"
        if london_high is not None:
            atr_at_bar_idx = df.index.get_indexer([ts], method="nearest")[0]
            atr_slice_bar  = df.iloc[max(0, atr_at_bar_idx - 50): atr_at_bar_idx + 1]
            atr_bar        = calc_atr(atr_slice_bar)

            # LONG: bar high broke london high AND close above it
            if high > london_high and close > london_high:
                dist  = abs(close - london_high)
                mode  = "RETEST" if dist <= atr_bar * RETEST_TOLERANCE else (
                        "BREAKOUT" if dist <= atr_bar * BREAKOUT_CHASE else "CHASE")
                if mode in ("RETEST", "BREAKOUT"):
                    sl_long  = round(max(close - atr_bar * 1.2, london_low), 3)
                    tp1_long = round(close + 2.0 * abs(close - sl_long), 3)
                    tp2_long = round(close + 5.0 * abs(close - sl_long), 3)
                    sig_str  = f"  ▲ LONG [{mode}]  Entry={close:.3f}  SL={sl_long:.3f}  TP1={tp1_long:.3f}  TP2={tp2_long:.3f}"
                    if signal_found is None:
                        signal_found = {"dir": "long", "entry": close, "sl": sl_long,
                                        "tp1": tp1_long, "tp2": tp2_long,
                                        "ts": ts, "mode": mode, "atr": atr_bar,
                                        "london_h": london_high, "london_l": london_low}
                else:
                    sig_str = f"  ✗ LONG chasing (dist={dist:.3f} > {atr_bar * BREAKOUT_CHASE:.3f})"

            # SHORT: bar low broke london low AND close below it
            elif low_ < london_low and close < london_low:
                dist  = abs(close - london_low)
                mode  = "RETEST" if dist <= atr_bar * RETEST_TOLERANCE else (
                        "BREAKOUT" if dist <= atr_bar * BREAKOUT_CHASE else "CHASE")
                if mode in ("RETEST", "BREAKOUT"):
                    sl_short  = round(min(close + atr_bar * 1.2, london_high), 3)
                    tp1_short = round(close - 2.0 * abs(sl_short - close), 3)
                    tp2_short = round(close - 5.0 * abs(sl_short - close), 3)
                    sig_str   = f"  ▼ SHORT [{mode}]  Entry={close:.3f}  SL={sl_short:.3f}  TP1={tp1_short:.3f}  TP2={tp2_short:.3f}"
                    if signal_found is None:
                        signal_found = {"dir": "short", "entry": close, "sl": sl_short,
                                        "tp1": tp1_short, "tp2": tp2_short,
                                        "ts": ts, "mode": mode, "atr": atr_bar,
                                        "london_h": london_high, "london_l": london_low}
                else:
                    sig_str = f"  ✗ SHORT chasing (dist={dist:.3f} > {atr_bar * BREAKOUT_CHASE:.3f})"

        print(f"  {str(ts)[11:16]:>5}  {close:>7.3f}  {high:>7.3f}  {low_:>7.3f}{sig_str}")

# ── Trade outcome ─────────────────────────────────────────────────────────────
if signal_found is None:
    print(f"\n  No valid signal generated on {TARGET_DATE}")
else:
    sig      = signal_found
    entry    = sig["entry"]
    sl       = sig["sl"]
    tp1      = sig["tp1"]
    tp2      = sig["tp2"]
    direction = sig["dir"]
    is_long  = direction == "long"
    sl_dist  = abs(entry - sl)

    print(f"\n{'─'*W}")
    print(f"  SIGNAL TAKEN — {direction.upper()} [{sig['mode']}]  @  {str(sig['ts'])[11:16]} UTC")
    print(f"{'─'*W}")
    print(f"  Entry         : {entry:.3f}")
    print(f"  Stop Loss     : {sl:.3f}  ({sl_dist:.3f} pts = {sl_dist/sig['atr']:.2f}× ATR)")
    print(f"  TP1 (2×R)     : {tp1:.3f}  (+{abs(tp1-entry):.3f} pts)")
    print(f"  TP2 (5×R)     : {tp2:.3f}  (+{abs(tp2-entry):.3f} pts)")
    print(f"  London range  : {sig['london_l']:.3f} – {sig['london_h']:.3f}")
    print(f"  ATR at entry  : {sig['atr']:.3f}")
    print(f"\n  Risk $1500 cap:")
    risk_2pct = round(1500 * 0.02, 2)
    risk_3pct = round(1500 * 0.03, 2)
    print(f"    ADX≤28 (3%): ${risk_3pct} at risk  |  lot ≈ {risk_3pct/sl_dist/1000:.3f} lots (if 1pt=$10/lot)")
    print(f"    ADX>28 (2%): ${risk_2pct} at risk  |  lot ≈ {risk_2pct/sl_dist/1000:.3f} lots")

    # Walk future bars
    sig_idx = df.index.get_indexer([sig["ts"]], method="nearest")[0]
    future  = df.iloc[sig_idx + 1: sig_idx + 97]

    print(f"\n  BAR-BY-BAR AFTER ENTRY (max 96 bars shown, SL/TP marked):")
    print(f"  {'Bar':>3}  {'Time':>5}  {'High':>7}  {'Low':>7}  {'Close':>7}  Status")
    print(f"  {'-'*60}")

    tp1_hit    = False
    exit_event = None

    for bar_n, (ts, row) in enumerate(future.iterrows(), 1):
        bar_h = float(row["high"])
        bar_l = float(row["low"])
        bar_c = float(row["close"])
        status = ""

        if exit_event is None:
            if is_long:
                if bar_l <= sl:
                    exit_p = sl if not tp1_hit else max(sl, entry)
                    exit_r = "SL" if not tp1_hit else "SL after TP1 (BE)"
                    pnl_r  = -1.0 if not tp1_hit else 0.0
                    extra  = f"+{tp1 - entry:.3f}(TP1 partial)" if tp1_hit else ""
                    exit_event = (bar_n, ts, exit_p, exit_r, pnl_r, extra)
                    status = f"  ← ❌ {exit_r}  exit={exit_p:.3f}"
                elif not tp1_hit and bar_h >= tp1:
                    tp1_hit = True
                    sl      = entry  # BE
                    status  = f"  ← ✓ TP1 HIT @ {tp1:.3f}  SL → BE ({entry:.3f})"
                elif tp1_hit and bar_h >= tp2:
                    exit_event = (bar_n, ts, tp2, "TP2", 2.0 + 5.0, "Full TP2")
                    status = f"  ← ✅ TP2 HIT @ {tp2:.3f}"
            else:  # short
                if bar_h >= sl:
                    exit_p = sl if not tp1_hit else min(sl, entry)
                    exit_r = "SL" if not tp1_hit else "SL after TP1 (BE)"
                    pnl_r  = -1.0 if not tp1_hit else 0.0
                    extra  = f"+{entry - tp1:.3f}(TP1 partial)" if tp1_hit else ""
                    exit_event = (bar_n, ts, exit_p, exit_r, pnl_r, extra)
                    status = f"  ← ❌ {exit_r}  exit={exit_p:.3f}"
                elif not tp1_hit and bar_l <= tp1:
                    tp1_hit = True
                    sl      = entry
                    status  = f"  ← ✓ TP1 HIT @ {tp1:.3f}  SL → BE ({entry:.3f})"
                elif tp1_hit and bar_l <= tp2:
                    exit_event = (bar_n, ts, tp2, "TP2", 2.0 + 5.0, "Full TP2")
                    status = f"  ← ✅ TP2 HIT @ {tp2:.3f}"

        # Only print first 24 bars in detail, then only if notable
        if bar_n <= 24 or status:
            time_str = str(ts)[11:16]
            print(f"  {bar_n:>3}  {time_str:>5}  {bar_h:>7.3f}  {bar_l:>7.3f}  {bar_c:>7.3f}{status}")

        if exit_event:
            break

    if exit_event is None:
        last_c  = float(future["close"].iloc[-1]) if not future.empty else entry
        pnl_dir = (last_c - entry) if is_long else (entry - last_c)
        pnl_r   = pnl_dir / sl_dist
        exit_event = (len(future), future.index[-1] if not future.empty else sig["ts"],
                      last_c, "MAX_HOLD", round(pnl_r, 2), "")

    # ── Summary ───────────────────────────────────────────────────────────────
    bar_n_, ts_, exit_p_, exit_r_, pnl_r_, extra_ = exit_event
    pnl_usd = round(1500 * 0.02 * pnl_r_, 2) if pnl_r_ > 0 else round(-1500 * 0.02, 2)
    pnl_usd_3 = round(1500 * 0.03 * pnl_r_, 2) if pnl_r_ > 0 else round(-1500 * 0.03, 2)

    result_icon = "✅" if pnl_r_ > 0 else ("➖" if pnl_r_ == 0 else "❌")
    print(f"\n{'='*W}")
    print(f"  OUTCOME : {result_icon}  {exit_r_}")
    print(f"{'─'*W}")
    print(f"  Exit price  : {exit_p_:.3f}  (bar {bar_n_}, ~{str(ts_)[11:16]} UTC)")
    print(f"  R achieved  : {pnl_r_:+.2f}R  {extra_}")
    if exit_r_ == "SL":
        move = abs(exit_p_ - entry)
        print(f"  SL distance : {sl_dist:.3f} pts  |  Price moved: {move:.3f} pts against entry")
        print(f"  Filled at   : {exit_p_:.3f}  (original SL level — no slippage assumed)")
    print(f"  P&L (2% risk, $1,500): {pnl_usd:+.2f} USD")
    print(f"  P&L (3% risk, $1,500): {pnl_usd_3:+.2f} USD")
    print(f"{'='*W}\n")

    # ── Why it hit SL — contextual analysis ──────────────────────────────────
    if "SL" in exit_r_:
        print(f"  WHY IT HIT SL — POST-MORTEM")
        print(f"{'─'*W}")
        # Check if the move was quick (< 3 bars) or slow (> 10 bars)
        if bar_n_ <= 2:
            print(f"  → Immediate rejection ({bar_n_} bar(s)): market likely had news/spike or was never trending")
        elif bar_n_ <= 6:
            print(f"  → Stopped within {bar_n_} bars: quick reversal after breakout — classic fake-out pattern")
        else:
            print(f"  → Trade ran for {bar_n_} bars before stopping: slow grind back through entry area")

        # Check london range quality
        if london_high is not None:
            range_atr_ratio = london_range / sig['atr']
            if range_atr_ratio < 0.5:
                print(f"  → London range was very tight ({london_range:.3f} = {range_atr_ratio:.1f}×ATR) — weak setup day")
            elif range_atr_ratio < 1.0:
                print(f"  → London range was moderate ({london_range:.3f} = {range_atr_ratio:.1f}×ATR)")
            else:
                print(f"  → London range was healthy ({london_range:.3f} = {range_atr_ratio:.1f}×ATR) — setup quality was good")

        print(f"  → 1 SL on 1 day is NORMAL. Strategy expects losses ~52% of trades.")
        print(f"  → Check the broader monthly WTI trend context before concluding the strategy failed.")
        print(f"{'─'*W}\n")
