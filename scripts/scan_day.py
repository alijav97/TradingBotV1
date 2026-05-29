"""
scripts/scan_day.py — Same as scan_today.py but for ANY date.

Shows the London range, kill-zone signals, and (for past dates) the
full trade outcome bar by bar.

Run from project root:
    python scripts/scan_day.py              # defaults to today
    python scripts/scan_day.py 2026-05-29   # specific past date
"""
import sys
import os
from pathlib import Path

# Resolve project root dynamically — works on both VPS (C:\Temp\TradingBotV1)
# and local dev (C:\Users\...\Downloads\TradingBotV1)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from datetime import datetime, timezone, date as date_type
import pandas as pd

# ── Date argument ─────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    try:
        target_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    except ValueError:
        print(f"ERROR: Invalid date '{sys.argv[1]}' — use YYYY-MM-DD format.")
        sys.exit(1)
else:
    target_date = datetime.now(timezone.utc).date()

now_utc      = datetime.now(timezone.utc)
is_future    = (target_date > now_utc.date())
# Treat as historical if: past date, OR same UTC date but kill-zone already closed
# (17:00 UTC = session end). This prevents running session-quality checks against
# the current wall clock when reviewing earlier bars from today's closed session.
_kz_closed_today = (target_date == now_utc.date()) and (now_utc.hour >= 17)
is_today      = (target_date == now_utc.date()) and not _kz_closed_today
is_historical = not is_today and not is_future

print("=" * 70)
print(f"  WTI KILL-ZONE SCAN  —  {target_date}")
if is_today:
    print(f"  Mode    : LIVE (today)")
elif is_historical:
    print(f"  Mode    : HISTORICAL BACKTEST  (shows outcome after signal)")
else:
    print(f"  Mode    : FUTURE DATE — nothing to show yet")
    sys.exit(0)
print(f"  Session : 13:00–17:00 UTC  (5PM–9PM UAE)  |  London range: 08:00–13:00 UTC")
print("=" * 70)
print()

# ── MT5 connection ────────────────────────────────────────────────────────────
try:
    from v2.settings import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    import v2.connectors.mt5_connector as mt5_conn
    from v2.signals.confluence_engine import ConfluenceEngine
    from v2.signals.entry_checklist import validate_entry

    ok = mt5_conn.connect(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not ok:
        print("ERROR: MT5 connection failed. Make sure MT5 is running.")
        sys.exit(1)

    # Fetch enough bars to cover lookback + full day + outcome window
    # For a past date: need ~30d lookback + 96h outcome = ~820 H1 bars
    BARS_H1 = 1000
    BARS_H4 = 250
    BARS_D1 = 60

    df_h1 = mt5_conn.get_ohlcv("SpotCrude", "H1", BARS_H1)
    df_h4 = mt5_conn.get_ohlcv("SpotCrude", "H4", BARS_H4)
    df_d1 = mt5_conn.get_ohlcv("SpotCrude", "D1", BARS_D1)

    if df_h1.empty:
        print("ERROR: No H1 data returned for SpotCrude.")
        sys.exit(1)

    # Normalise timestamps to UTC
    for _df in (df_h1, df_h4, df_d1):
        if not _df.empty and "time" in _df.columns:
            _df["time"] = pd.to_datetime(_df["time"])
            if _df["time"].dt.tz is None:
                _df["time"] = _df["time"].dt.tz_localize("UTC")
            else:
                _df["time"] = _df["time"].dt.tz_convert("UTC")

    df_h1 = df_h1.sort_values("time").reset_index(drop=True)
    times_utc = df_h1["time"].dt.tz_convert("UTC")

    print(f"  H1 data  : {len(df_h1)} bars  "
          f"({df_h1['time'].iloc[0].date()} → {df_h1['time'].iloc[-1].date()})")
    print()

    # Check target date is in the data
    if not (times_utc.dt.date == target_date).any():
        print(f"  No H1 bars found for {target_date}.")
        print(f"  Available: {df_h1['time'].iloc[0].date()} → {df_h1['time'].iloc[-1].date()}")
        mt5_conn.disconnect()
        sys.exit(0)

    # ── London session (08-13 UTC) ─────────────────────────────────────────────
    london_mask = (
        (times_utc.dt.date == target_date) &
        (times_utc.dt.hour >= 8) &
        (times_utc.dt.hour < 13)
    )
    london_bars = df_h1[london_mask]

    print(f"  LONDON SESSION  08:00–13:00 UTC")
    print(f"  {'─'*60}")

    if len(london_bars) < 3:
        print(f"  ⚠  Only {len(london_bars)} London bar(s) — not enough for a valid range (need ≥3).")
        print()
    else:
        london_high  = float(london_bars["high"].max())
        london_low   = float(london_bars["low"].min())
        london_range = london_high - london_low

        print(f"  {'Time':>5}  {'Open':>8}  {'High':>8}  {'Low':>8}  {'Close':>8}")
        print(f"  {'─'*50}")
        for _, row in london_bars.iterrows():
            print(f"  {str(row['time'])[11:16]:>5}  {row['open']:>8.3f}  "
                  f"{row['high']:>8.3f}  {row['low']:>8.3f}  {row['close']:>8.3f}")

        print(f"\n  London High  : {london_high:.3f}")
        print(f"  London Low   : {london_low:.3f}")
        print(f"  London Range : {london_range:.3f}")
    print()

    # ── Kill-zone bars (13-17 UTC) ─────────────────────────────────────────────
    kz_mask = (
        (times_utc.dt.date == target_date) &
        (times_utc.dt.hour >= 13) &
        (times_utc.dt.hour < 17)
    )
    kz_bars = df_h1[kz_mask]

    if kz_bars.empty:
        if is_today and now_utc.hour < 13:
            print(f"  Kill-zone hasn't opened yet today (UTC {now_utc.hour:02d}:xx).")
            print(f"  Run again after 13:00 UTC (5PM UAE).")
        else:
            print(f"  No bars found in kill-zone window (13:00–17:00 UTC) for {target_date}.")
        mt5_conn.disconnect()
        sys.exit(0)

    print(f"  KILL-ZONE BARS  13:00–17:00 UTC  ({len(kz_bars)} bar(s))")
    print(f"  {'─'*60}")
    for _, row in kz_bars.iterrows():
        print(f"  {str(row['time'])[11:16]}  O={row['open']:.3f}  "
              f"H={row['high']:.3f}  L={row['low']:.3f}  C={row['close']:.3f}")
    print()

    # ── Signal evaluation ──────────────────────────────────────────────────────
    engine          = ConfluenceEngine()
    all_bar_indices = df_h1.index.tolist()
    signals_found   = []

    print(f"  {'Bar UTC':20s}  {'Dir':6s}  {'Score':5s}  {'Engine':8s}  {'Checklist':12s}  Detail")
    print(f"  {'─'*100}")

    for df_idx, bar_row in kz_bars.iterrows():
        pos       = all_bar_indices.index(df_idx)
        window    = df_h1.iloc[:pos + 1].copy()
        wh4       = df_h4[df_h4["time"] <= bar_row["time"]].copy() if not df_h4.empty else None
        wd1       = df_d1[df_d1["time"] <= bar_row["time"]].copy() if not df_d1.empty else None
        bar_time  = bar_row["time"]
        bar_str   = str(bar_time)[:16]

        for direction in ("long", "short"):
            ctx    = {"bar_time": bar_time}
            result = engine.score("WTI", direction, window, wh4, wd1, ctx)

            signal  = result.get("signal", False)
            score   = result.get("score", 0.0)
            blocked = result.get("blocked_by", "")
            entry   = result.get("entry_price", 0)
            sl      = result.get("stop_loss", 0)
            tp1     = result.get("tp1_price", 0)
            tp2     = result.get("tp2_price", 0)

            engine_str   = "SIGNAL" if signal else "no"
            checklist_str = "—"
            detail        = (blocked or "score too low")[:55] if not signal else (
                            f"entry={entry:.3f} sl={sl:.3f} tp1={tp1:.3f} tp2={tp2:.3f}")

            # ── Run entry checklist on signals (mirrors live bot) ─────────────
            if signal:
                sig_dict = {
                    "symbol":      "WTI",
                    "direction":   direction,
                    "entry_price": entry,
                    "stop_loss":   sl,
                    "tp1_price":   tp1,
                    "tp2_price":   tp2,
                    "score":       score,
                    "strategy":    result.get("strategy", "ny_momentum_wti"),
                }
                # For historical dates skip live news check; for today use real news filter
                chk = validate_entry(sig_dict, window, skip_news=is_historical)
                if chk["passed"]:
                    checklist_str = "✓ PASS"
                    signals_found.append({
                        "bar_idx": pos, "bar_time": bar_time,
                        "direction": direction,
                        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
                        "score": score,
                    })
                else:
                    checklist_str = f"✗ {chk['failed_at']}"
                    detail        = f"BLOCKED: {chk['failed_at']} — {list(chk['checks'].values())[list(chk['checks'].keys()).index(chk['failed_at'])]['reason'][:60]}"

            print(f"  {bar_str:20s}  {direction.upper():6s}  {score:5.1f}  {engine_str:8s}  {checklist_str:12s}  {detail}")

            if signal:
                reasons = result.get("reasons") or []
                if isinstance(reasons, list):
                    for r in reasons:
                        if r:
                            print(f"  {'':60s}→ {r}")

    print()

    # ── Outcomes (historical only) ─────────────────────────────────────────────
    if is_historical and signals_found:
        MAX_HOLD = 96

        print(f"  {'='*70}")
        print(f"  TRADE OUTCOME(S)  — historical simulation")
        print(f"  {'='*70}")

        for sig in signals_found:
            entry     = sig["entry"]
            sl        = sig["sl"]
            tp1       = sig["tp1"]
            tp2       = sig["tp2"]
            direction = sig["direction"]
            is_long   = direction == "long"
            sl_dist   = abs(entry - sl)

            print(f"\n  {direction.upper()}  @  {str(sig['bar_time'])[11:16]} UTC  "
                  f"(score {sig['score']:.1f})")
            print(f"  Entry={entry:.3f}  SL={sl:.3f}  TP1={tp1:.3f}  TP2={tp2:.3f}  "
                  f"(SL dist={sl_dist:.3f})")
            print()
            print(f"  {'#':>3}  {'Time':>5}  {'High':>7}  {'Low':>7}  {'Close':>7}  Status")
            print(f"  {'─'*55}")

            future  = df_h1.iloc[sig["bar_idx"] + 1: sig["bar_idx"] + 1 + MAX_HOLD]
            tp1_hit = False
            outcome = None

            for n, (_, frow) in enumerate(future.iterrows(), 1):
                fh, fl, fc = float(frow["high"]), float(frow["low"]), float(frow["close"])
                ft         = str(frow["time"])[11:16]
                status     = ""

                if outcome is None:
                    if is_long:
                        if fl <= sl:
                            exit_p = max(sl, entry) if tp1_hit else sl
                            label  = "SL_after_TP1 (BE)" if tp1_hit else "SL ❌"
                            outcome = (n, ft, exit_p, label, tp1_hit)
                            status  = f"  ← {label}  @ {exit_p:.3f}"
                        elif not tp1_hit and fh >= tp1:
                            tp1_hit = True
                            sl      = entry
                            status  = f"  ← TP1 ✓ @ {tp1:.3f}  SL → BE {entry:.3f}"
                        elif tp1_hit and fh >= tp2:
                            outcome = (n, ft, tp2, "TP2 ✅", True)
                            status  = f"  ← TP2 ✅ @ {tp2:.3f}"
                    else:
                        if fh >= sl:
                            exit_p = min(sl, entry) if tp1_hit else sl
                            label  = "SL_after_TP1 (BE)" if tp1_hit else "SL ❌"
                            outcome = (n, ft, exit_p, label, tp1_hit)
                            status  = f"  ← {label}  @ {exit_p:.3f}"
                        elif not tp1_hit and fl <= tp1:
                            tp1_hit = True
                            sl      = entry
                            status  = f"  ← TP1 ✓ @ {tp1:.3f}  SL → BE {entry:.3f}"
                        elif tp1_hit and fl <= tp2:
                            outcome = (n, ft, tp2, "TP2 ✅", True)
                            status  = f"  ← TP2 ✅ @ {tp2:.3f}"

                if n <= 20 or status:
                    print(f"  {n:>3}  {ft:>5}  {fh:>7.3f}  {fl:>7.3f}  {fc:>7.3f}{status}")

                if outcome:
                    break
            else:
                lp      = float(future["close"].iloc[-1]) if not future.empty else entry
                pnl_dir = (lp - entry) if is_long else (entry - lp)
                pnl_r   = round(pnl_dir / sl_dist, 2) if sl_dist else 0
                outcome = (len(future), str(future["time"].iloc[-1])[11:16],
                           lp, f"MAX_HOLD ({pnl_r:+.2f}R)", tp1_hit)

            n_, t_, p_, label_, tp1_was = outcome
            diff    = (p_ - entry) if is_long else (entry - p_)
            r_val   = round(diff / sl_dist, 2) if sl_dist else 0
            WTI_BALANCE = 500   # WTI paper trading account
            pnl_2   = round(WTI_BALANCE * 0.02 * r_val, 2) if r_val > 0 else round(-WTI_BALANCE * 0.02, 2)
            pnl_3   = round(WTI_BALANCE * 0.03 * r_val, 2) if r_val > 0 else round(-WTI_BALANCE * 0.03, 2)

            print(f"\n  Result   : {label_}")
            print(f"  Exit     : {p_:.3f}  (bar {n_}, ~{t_} UTC)")
            print(f"  R        : {r_val:+.2f}R")
            print(f"  P&L 2%/3%: ${pnl_2:+.2f} / ${pnl_3:+.2f}  (on ${WTI_BALANCE} account)")
            print()

    elif is_historical and not signals_found:
        print(f"  No signals generated on {target_date}.")
        print(f"  The strategy filtered out all kill-zone bars.")
        print(f"  (score 0.0 = strategy rejected early — London range invalid or no breakout)")

    # ── Live summary ───────────────────────────────────────────────────────────
    print("=" * 70)
    if is_today:
        live_price = mt5_conn.get_live_price("SpotCrude")
        if live_price:
            print(f"  Current WTI : {live_price.get('price', '?'):.3f}  "
                  f"(bid={live_price.get('bid', '?')}  ask={live_price.get('ask', '?')})")
        in_kz = 13 <= now_utc.hour < 17
        print(f"  Kill-zone   : {'ACTIVE ▶' if in_kz else 'CLOSED'}")
    else:
        print(f"  Backtest for {target_date} complete.")
        if not signals_found:
            print(f"  0 signals → strategy had no valid setup that day.")
    print("=" * 70)
    print()

    mt5_conn.disconnect()

except Exception as e:
    print(f"ERROR: {e}")
    import traceback; traceback.print_exc()
