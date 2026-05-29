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

# ── Kill-zone window (same alignment as WTI strategy) ────────────────────────
# 13:00-17:00 UTC = 5 PM – 9 PM UAE = US market open
KZ_START_UTC = 13
KZ_END_UTC   = 17

# ── Morning range window (range forms before kill-zone entry) ─────────────────
# 08:00-13:00 UTC = London session = pre-kill-zone
MR_START_UTC = 8
MR_END_UTC   = 13

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
