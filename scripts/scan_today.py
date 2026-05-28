"""
scripts/scan_today.py — Scan TODAY's kill-zone bars and show what signals
the strategy would have fired, without opening any real trades.

Useful for verifying time alignment and signal detection after a fix.

Run from the project root:
    python scripts/scan_today.py

Output: for each H1 bar inside today's 13:00-17:00 UTC window, shows
whether a LONG or SHORT signal fired and why it was accepted or rejected.
"""
import sys
import os
sys.path.insert(0, r"C:\Temp\TradingBotV1")
os.chdir(r"C:\Temp\TradingBotV1")

from datetime import datetime, timezone, timedelta
import pandas as pd

print("=" * 65)
print("TODAY'S KILL-ZONE SIGNAL SCAN")
print("=" * 65)

now_utc = datetime.now(timezone.utc)
today   = now_utc.date()
print(f"Current UTC : {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Kill-zone   : 13:00 – 17:00 UTC  (5PM – 9PM UAE)")
print()

try:
    from v2.settings import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    import v2.connectors.mt5_connector as mt5_conn
    from v2.signals.confluence_engine import ConfluenceEngine

    ok = mt5_conn.connect(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not ok:
        print("ERROR: MT5 connection failed")
        sys.exit(1)

    # Fetch enough bars: 120 lookback + today's bars
    df_h1 = mt5_conn.get_ohlcv("SpotCrude", "H1", 200)
    df_h4 = mt5_conn.get_ohlcv("SpotCrude", "H4", 100)
    df_d1 = mt5_conn.get_ohlcv("SpotCrude", "D1", 50)

    if df_h1.empty:
        print("ERROR: No H1 bars returned")
        sys.exit(1)

    print(f"Loaded {len(df_h1)} H1 bars. Last bar: "
          f"{df_h1['time'].iloc[-1]}  close={df_h1['close'].iloc[-1]:.3f}")
    print()

    # Find today's bars inside the kill-zone window (13-17 UTC)
    times = pd.to_datetime(df_h1["time"])
    if times.dt.tz is not None:
        times_utc = times.dt.tz_convert("UTC")
    else:
        times_utc = times.dt.tz_localize("UTC")

    kz_mask = (
        (times_utc.dt.date == today) &
        (times_utc.dt.hour >= 13) &
        (times_utc.dt.hour < 17)
    )
    kz_bars = df_h1[kz_mask]

    if kz_bars.empty:
        # Kill-zone may not have started yet today
        if now_utc.hour < 13:
            print(f"Kill-zone hasn't opened yet today (current UTC {now_utc.hour:02d}:xx).")
            print(f"Showing what London range would be for the strategy:")
        else:
            print("No bars found in today's 13:00-17:00 UTC window yet.")
        # Show London range anyway
        london_mask = (
            (times_utc.dt.date == today) &
            (times_utc.dt.hour >= 8) &
            (times_utc.dt.hour < 13)
        )
        london_bars = df_h1[london_mask]
        if not london_bars.empty:
            lh = float(london_bars["high"].max())
            ll = float(london_bars["low"].min())
            print(f"  London High : {lh:.3f}")
            print(f"  London Low  : {ll:.3f}")
            print(f"  London Range: {lh - ll:.3f}  ({len(london_bars)} bars)")
        else:
            print("  London bars: none found yet today (before 08:00 UTC?)")
        print()
        print("Run this script again after 13:00 UTC (5PM UAE) to see signals.")
        mt5_conn.disconnect()
        sys.exit(0)

    print(f"Found {len(kz_bars)} kill-zone bar(s) for today:")
    for _, row in kz_bars.iterrows():
        print(f"  {row['time']}  O={row['open']:.3f} H={row['high']:.3f} "
              f"L={row['low']:.3f} C={row['close']:.3f}")
    print()

    # Run strategy on each kill-zone bar (use history up to that bar)
    engine = ConfluenceEngine()
    print(f"{'Bar UTC':20s}  {'Dir':6s}  {'Signal':8s}  {'Score':6s}  {'Reason'}")
    print("-" * 90)

    all_bar_indices = df_h1.index.tolist()

    for bar_idx, (df_idx, bar_row) in enumerate(kz_bars.iterrows()):
        # Find position of this bar in the full df
        pos = all_bar_indices.index(df_idx)
        window    = df_h1.iloc[:pos + 1].copy()
        window_h4 = df_h4[pd.to_datetime(df_h4["time"]) <= bar_row["time"]].copy() if not df_h4.empty else None
        window_d1 = df_d1[pd.to_datetime(df_d1["time"]) <= bar_row["time"]].copy() if not df_d1.empty else None

        bar_time = bar_row["time"]

        for direction in ("long", "short"):
            ctx = {"bar_time": bar_time}
            result = engine.score("WTI", direction, window, window_h4, window_d1, ctx)

            signal  = result.get("signal", False)
            score   = result.get("score", 0.0)
            blocked = result.get("blocked_by", "")
            entry   = result.get("entry_price", 0)
            sl      = result.get("stop_loss", 0)
            tp1     = result.get("tp1_price", 0)
            tp2     = result.get("tp2_price", 0)

            bar_str = str(bar_row["time"])[:16]
            if signal:
                print(f"{bar_str:20s}  {direction.upper():6s}  {'SIGNAL':8s}  {score:5.1f}  "
                      f"entry={entry:.3f} sl={sl:.3f} tp1={tp1:.3f} tp2={tp2:.3f}")
            else:
                short_reason = (blocked or "score too low")[:55]
                print(f"{bar_str:20s}  {direction.upper():6s}  {'no':8s}  {score:5.1f}  {short_reason}")

    print()
    print("=" * 65)
    print("SUMMARY")
    live_price = mt5_conn.get_live_price("SpotCrude")
    if live_price:
        print(f"Current WTI price: {live_price.get('price', '?'):.3f}  "
              f"(bid={live_price.get('bid','?')} ask={live_price.get('ask','?')})")
    in_kz = 13 <= now_utc.hour < 17
    print(f"Kill-zone now    : {'ACTIVE' if in_kz else 'CLOSED'}")
    print("=" * 65)

    mt5_conn.disconnect()

except Exception as e:
    print(f"ERROR: {e}")
    import traceback; traceback.print_exc()
