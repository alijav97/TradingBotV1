"""
scripts/check_mt5_usage.py — Estimate daily MT5 API message usage.

Pepperstone allows 30,000 messages/day on MT5.
This script shows exactly how many calls the bot makes and what % of the
limit is used, so we stay well within fair-use limits.

Run any time:
    python scripts/check_mt5_usage.py
"""
import sys, os
sys.path.insert(0, r"C:\Temp\TradingBotV1")
os.chdir(r"C:\Temp\TradingBotV1")

from datetime import datetime, timezone

PEPPERSTONE_DAILY_LIMIT = 30_000

print("=" * 62)
print("MT5 API MESSAGE USAGE ESTIMATE")
print(f"Pepperstone daily limit: {PEPPERSTONE_DAILY_LIMIT:,}")
print("=" * 62)

now_utc   = datetime.now(timezone.utc)
utc_hour  = now_utc.hour
utc_min   = now_utc.minute

# ── Per-job call counts ───────────────────────────────────────────────────────
# Each MT5 Python API call (copy_rates_from_pos, symbol_info_tick,
# symbol_select) counts as 1 message.

# H1 scan (every 2 min, but ONLY 12:30-17:30 UTC = 5 hours with buffer)
# Per scan: get_ohlcv(H1) + get_ohlcv(H4) + get_ohlcv(D1) + get_price
#           + symbol_select × 4 = 8 calls per scan cycle
H1_SCAN_INTERVAL_MIN   = 2
H1_ACTIVE_HOURS        = 5.0        # 12:30–17:30 UTC window
H1_SCANS_PER_ACTIVE_HR = 60 / H1_SCAN_INTERVAL_MIN
H1_CALLS_PER_SCAN      = 8          # 4 data + 4 symbol_select
h1_daily               = int(H1_SCANS_PER_ACTIVE_HR * H1_ACTIVE_HOURS * H1_CALLS_PER_SCAN)

# Monitor (every 60s, all day, only when trade is open)
# Per cycle: get_price + symbol_select + get_ohlcv(H1 for ATR) + symbol_select = 4
MONITOR_INTERVAL_SEC   = 60
MONITOR_CALLS_PER_TICK = 4
monitor_daily_open     = int((3600 / MONITOR_INTERVAL_SEC) * 24 * MONITOR_CALLS_PER_TICK)
monitor_daily_none     = 0   # no open trade = 0 MT5 calls from monitor

# H4 scan (6×/day at 00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC)
H4_SCANS_PER_DAY  = 6
H4_CALLS_PER_SCAN = 8
h4_daily          = H4_SCANS_PER_DAY * H4_CALLS_PER_SCAN

print()
print(f"{'Job':<30} {'Freq':<18} {'Calls/day'}")
print("-" * 62)
print(f"{'H1 scan (kill-zone only)':<30} {'every 2min/12:30-17:30':18} {h1_daily:>6}")
print(f"{'H4 scan':<30} {'6x/day':18} {h4_daily:>6}")
print(f"{'Monitor (trade OPEN)':<30} {'every 60s all day':18} {monitor_daily_open:>6}")
print(f"{'Monitor (no trade)':<30} {'every 60s all day':18} {monitor_daily_none:>6}")

print()
print("SCENARIOS:")
scenario_open = h1_daily + h4_daily + monitor_daily_open
scenario_none = h1_daily + h4_daily + monitor_daily_none

pct_open = scenario_open / PEPPERSTONE_DAILY_LIMIT * 100
pct_none = scenario_none / PEPPERSTONE_DAILY_LIMIT * 100

print(f"  With 1 trade open all day : {scenario_open:>5,} calls  "
      f"({pct_open:.1f}% of {PEPPERSTONE_DAILY_LIMIT:,} limit)")
print(f"  No open trade (scanning)  : {scenario_none:>5,} calls  "
      f"({pct_none:.1f}% of {PEPPERSTONE_DAILY_LIMIT:,} limit)")

headroom_open = PEPPERSTONE_DAILY_LIMIT - scenario_open
headroom_none = PEPPERSTONE_DAILY_LIMIT - scenario_none
print(f"  Headroom (trade open)     : {headroom_open:>5,} calls remaining")
print(f"  Headroom (no trade)       : {headroom_none:>5,} calls remaining")

# ── Kill-zone window check ────────────────────────────────────────────────────
print()
print("KILL-ZONE SCAN SCHEDULE:")
kz_open_min  = 12 * 60 + 30
kz_close_min = 17 * 60 + 30
now_min      = utc_hour * 60 + utc_min
in_active    = kz_open_min <= now_min <= kz_close_min

print(f"  H1 scans active window : 12:30 – 17:30 UTC  (5PM – 9:30PM UAE)")
print(f"  H1 scans outside window: SKIPPED (0 MT5 calls from scanner)")
print(f"  Right now (UTC {utc_hour:02d}:{utc_min:02d})  : "
      f"{'SCANNING every 2 min' if in_active else 'IDLE - no scan calls'}")

# ── Safety rating ─────────────────────────────────────────────────────────────
print()
print("SAFETY RATING:")
if pct_open < 20:
    rating = "EXCELLENT"
    note   = "Far under limit — multiple instruments could be added safely"
elif pct_open < 40:
    rating = "GOOD"
    note   = "Well within limits"
elif pct_open < 70:
    rating = "OK"
    note   = "Monitor if adding more instruments"
else:
    rating = "CAUTION"
    note   = "Review scan frequency before adding instruments"

print(f"  {rating} — worst case {pct_open:.1f}% of daily limit used")
print(f"  {note}")

print()
print("NOTE: 'Messages' = MT5 Python API calls (copy_rates_from_pos,")
print("      symbol_info_tick, symbol_select). No real trade orders are")
print("      sent during paper trading, so order-message limits do not apply.")
print("=" * 62)
