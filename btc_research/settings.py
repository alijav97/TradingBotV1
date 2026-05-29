"""
btc_research/settings.py - Configuration for BTC research & backtest.

Completely isolated from v2/settings.py — no interference with the live WTI bot.
Change anything here freely; it will never affect live trading.
"""
from pathlib import Path

# ── Symbols (Pepperstone MT5) ────────────────────────────────────────────────
BTC_SYMBOL  = "BTCUSD"    # Bitcoin vs USD
GOLD_SYMBOL = "XAUUSD"    # Gold — inverse correlation factor
NAS_SYMBOL  = "NAS100"    # Nasdaq 100 — risk-on/off factor

# ── MT5 server offset (same as WTI bot) ─────────────────────────────────────
# Pepperstone server is UTC+3. The v2 mt5_connector already corrects this,
# so data fetched through it will already be in true UTC.
MT5_SERVER_UTC_OFFSET = 3

# ── Kill-zone window — SET FROM SESSION SCANNER RESULTS ─────────────────────
# Session scanner tested all 24 hours on 2yr BTC data and found:
#
#   02:00 UTC — 57.1% WR, +$9,704 (best single hour)
#   03:00 UTC — 52.6% WR, +$9,962 (second best)
#   00:00 UTC — 31.7% WR  (avoid)
#   01:00 UTC — 26.7% WR  (avoid)
#   13-17 UTC — 38.2% WR  (WTI assumption was wrong for BTC)
#
# Asia Night session (00-04 UTC) overall: 46.9% WR, +$15,634 total PnL — BEST
# Session scanner best single hours: 02:00 UTC (57.1% WR) and 03:00 UTC (52.6% WR)
# Optimizer best window: 01-05 UTC → 159 trades over 2yr @ 43.4% WR, 6.6 trades/month
# This 4-hour block captures late Tokyo close + pre-London positioning + early London open.
#
# UAE time equivalent: 05:00-09:00 AM UAE
KZ_START_UTC = 1
KZ_END_UTC   = 5

# ── Morning range window (range forms BEFORE kill-zone entry) ─────────────────
# For 01-05 UTC kill-zone, range forms during prior US late session / early Asia: 20:00-01:00 UTC
# Using 6-bar lookback in strategy (last 6 bars before signal bar) is cleaner
# than a fixed session window here, so MR_START/END are kept for documentation.
MR_START_UTC = 20   # prior day late US / early Asia
MR_END_UTC   = 1    # just before kill-zone opens

# ── Risk & position sizing ────────────────────────────────────────────────────
STARTING_BALANCE = 10_000   # USD — backtest starting capital
RISK_PCT         = 0.03     # 3% risk per trade (matches WTI bot)
TP1_RR           = 2.0      # Take-profit 1 at 1:2 R:R → partial close + SL to BE
TP2_RR           = 5.0      # Take-profit 2 at 1:5 R:R → full close
MAX_HOLD_BARS    = 96       # Force-close after 96 H1 bars (4 days)

# ── Backtest period ───────────────────────────────────────────────────────────
LOOKBACK_YEARS = 2          # Pull 2 years of H1 history from MT5

# ── Confluence gate ───────────────────────────────────────────────────────────
# Minimum total score required to open a trade.
# This is the key parameter to tune after the first backtest run.
# Start at 3.0 — adjust up (fewer trades, higher WR) or down (more trades) based on results.
MIN_CONFLUENCE_SCORE = 3.0

# ── Data cache ────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "data" / "cache"
