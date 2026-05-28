"""
scripts/check_time_alignment.py — Verify time alignment between system clock,
MT5 server bars, and the kill-zone strategy session windows.

Run from the project root at any time:
    python scripts/check_time_alignment.py

Checks:
  1. Windows system clock UTC vs UAE/EST
  2. MT5 last bar timestamp for SpotCrude (WTI)
  3. Detected server TZ offset (what the strategy auto-detects)
  4. Whether the kill-zone window is currently active
  5. Whether London bars would be found for today
  6. Live price freshness (tick age)
"""
import sys
import os
sys.path.insert(0, r"C:\Temp\TradingBotV1")
os.chdir(r"C:\Temp\TradingBotV1")

from datetime import datetime, timezone, timedelta
import pandas as pd

# ── 1. System clock ───────────────────────────────────────────────────────────
now_utc = datetime.now(timezone.utc)
now_uae = now_utc + timedelta(hours=4)   # GST = UTC+4
now_est = now_utc - timedelta(hours=4)   # EST = UTC-4 (no DST adjustment)

print("=" * 60)
print("1. SYSTEM CLOCK")
print(f"   UTC  : {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"   UAE  : {now_uae.strftime('%Y-%m-%d %H:%M:%S')} GST (UTC+4)")
print(f"   EST  : {now_est.strftime('%Y-%m-%d %H:%M:%S')} EST (UTC-4)")
print(f"   UTC hour = {now_utc.hour}  (kill-zone is 13–17 UTC)")

# ── 2. Kill-zone window status ────────────────────────────────────────────────
NY_START = 13
NY_END   = 17
in_kz    = NY_START <= now_utc.hour < NY_END

print()
print("2. KILL-ZONE WINDOW STATUS")
if in_kz:
    mins_left = (NY_END - now_utc.hour) * 60 - now_utc.minute
    print(f"   >> ACTIVE << (UTC {now_utc.hour:02d}:{now_utc.minute:02d} is inside 13:00-17:00)")
    print(f"   Minutes remaining in window: {mins_left}")
else:
    if now_utc.hour < NY_START:
        mins_to_open = (NY_START - now_utc.hour) * 60 - now_utc.minute
        print(f"   NOT active — opens in {mins_to_open} minutes (at 13:00 UTC)")
    else:
        mins_to_next = (24 - now_utc.hour + NY_START) * 60 - now_utc.minute
        print(f"   NOT active — next window in {mins_to_next} minutes (tomorrow 13:00 UTC)")

# ── 3. MT5 connection + bar timestamps ───────────────────────────────────────
print()
print("3. MT5 BAR TIMESTAMPS")
try:
    from v2.settings import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    import v2.connectors.mt5_connector as mt5_conn

    ok = mt5_conn.connect(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not ok:
        print("   ERROR: MT5 connection failed")
    else:
        df = mt5_conn.get_ohlcv("SpotCrude", "H1", 5)
        if df.empty:
            print("   ERROR: No bars returned for SpotCrude")
        else:
            print(f"   Last 5 H1 bars for SpotCrude:")
            for _, row in df.iterrows():
                bar_t = pd.to_datetime(row["time"])
                if bar_t.tzinfo is not None:
                    bar_t_utc = bar_t.tz_convert("UTC")
                    tz_label  = "UTC-aware"
                else:
                    bar_t_utc = bar_t  # naive — treat as-is
                    tz_label  = "NAIVE (assumed server TZ)"
                print(f"     {bar_t}  [{tz_label}]  close={row['close']:.3f}")

            # Detect server TZ offset (same logic as the strategy)
            raw_time = df["time"].iloc[-1]
            bar_time = pd.to_datetime(raw_time)
            if bar_time.tzinfo is not None:
                server_tz_offset = 0
                offset_note = "bars are UTC-aware, no offset needed"
            else:
                raw_hour = bar_time.hour
                diff = (raw_hour - now_utc.hour + 12) % 24 - 12
                server_tz_offset = diff
                offset_note = f"naive bars: detected offset = bar_hour({raw_hour}) - utc_hour({now_utc.hour}) = {diff:+d}h"

            print()
            print(f"   Detected server_tz_offset: {server_tz_offset:+d}h  ({offset_note})")

            # London window in server time
            LONDON_START_UTC = 8
            LONDON_END_UTC   = 13
            server_london_start = (LONDON_START_UTC + server_tz_offset) % 24
            server_london_end   = (LONDON_END_UTC   + server_tz_offset) % 24
            print(f"   London session in server time: {server_london_start:02d}:00 – {server_london_end:02d}:00")
            print(f"   London session in UTC        : {LONDON_START_UTC:02d}:00 – {LONDON_END_UTC:02d}:00")

            # Count London bars for today
            server_today = (now_utc.replace(tzinfo=None) + __import__("datetime").timedelta(hours=server_tz_offset)).date()
            raw_times = pd.to_datetime(df["time"])
            # need more bars for London check — fetch 30
            df30 = mt5_conn.get_ohlcv("SpotCrude", "H1", 30)
            raw_times30 = pd.to_datetime(df30["time"])
            if raw_times30.dt.tz is not None:
                raw_naive = raw_times30.dt.tz_convert("UTC").dt.tz_localize(None)
            else:
                raw_naive = raw_times30

            if server_london_start < server_london_end:
                hour_mask = ((raw_naive.dt.hour >= server_london_start) &
                             (raw_naive.dt.hour <  server_london_end))
            else:
                hour_mask = ((raw_naive.dt.hour >= server_london_start) |
                             (raw_naive.dt.hour <  server_london_end))

            mask = (raw_naive.dt.date == server_today) & hour_mask
            london_bars = df30[mask]
            print()
            print("4. LONDON RANGE FOR TODAY")
            print(f"   Server date used : {server_today}")
            if len(london_bars) == 0:
                print(f"   London bars found: 0  (need >= 3 — strategy will SKIP today)")
            else:
                lh = float(london_bars["high"].max())
                ll = float(london_bars["low"].min())
                print(f"   London bars found: {len(london_bars)}  (need >= 3: {'OK' if len(london_bars) >= 3 else 'NOT ENOUGH'})")
                print(f"   London High : {lh:.3f}")
                print(f"   London Low  : {ll:.3f}")
                print(f"   London Range: {lh - ll:.3f}")

        # ── 5. Live price freshness ───────────────────────────────────────────
        print()
        print("5. LIVE PRICE FRESHNESS")
        price_info = mt5_conn.get_live_price("SpotCrude")
        if not price_info:
            print("   WARNING: get_live_price returned empty — tick is stale or unavailable")
            print("   The bot will SKIP trades until a fresh tick is available")
        else:
            age = price_info.get("age_seconds", "?")
            print(f"   Bid   : {price_info.get('bid', '?')}")
            print(f"   Ask   : {price_info.get('ask', '?')}")
            print(f"   Mid   : {price_info.get('price', '?')}")
            print(f"   Tick age: {age}s  ({'FRESH' if isinstance(age, (int,float)) and age < 60 else 'STALE' if isinstance(age,(int,float)) and age >= 300 else 'OK'})")
            print(f"   Tick time: {price_info.get('time', '?')}")

        mt5_conn.disconnect()

except ImportError as e:
    print(f"   Cannot import v2 modules: {e}")
    print("   Run this script from C:\\Temp\\TradingBotV1")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback; traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print(f"  System UTC clock : {now_utc.strftime('%H:%M:%S')} UTC  (this is what the strategy uses)")
print(f"  Kill-zone active : {'YES - trades can open NOW' if in_kz else 'NO  - outside 13:00-17:00 UTC'}")
print("=" * 60)
