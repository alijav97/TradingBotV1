"""
btc_research/btc_bot_2/settings.py — BTC Bot 2 configuration.

Session  : Asia Night  02:00-04:00 UTC  (06:00-08:00 UAE)
Strategy : Volatility Breakout + Swing Level Break  (no Morning Range)

Why no Morning Range?
  Morning Range fires on quiet consolidation breaks. It performs best at
  US Open (13-17 UTC). At 02-03 UTC (Asia Night), BTC is actively trending
  with momentum — VB and Swing Level capture this better.

Why 02-04 UTC?
  Session scanner on 2yr data:
    02:00 UTC → 57.1% WR  (best single hour)
    03:00 UTC → 52.6% WR  (second best)
  Both hours are in Asia Night. Combined = 2 bars per night.

Bot 2 is research-only until backtest validates the filter set.
All filter settings here are STARTING POINTS — run_backtest_btc2.py
will compare filter combinations and update these.
"""
from __future__ import annotations
import os
from pathlib import Path

# ── Kill-zone ─────────────────────────────────────────────────────────────────
KZ_START_UTC = 2    # 02:00 UTC  (06:00 UAE)
KZ_END_UTC   = 4    # 04:00 UTC  (exclusive → bars at 02:00 and 03:00)

# ── Strategy ──────────────────────────────────────────────────────────────────
# VB params (optimised for Bot 1 21-24 UTC — may differ for Asia Night)
VB_ATR_MULTIPLIER = 1.2
VB_CLOSE_ZONE     = 0.45

# ── Risk & sizing (TBD — backtest will determine optimal) ─────────────────────
STARTING_BALANCE     = 500.0
RISK_PCT             = 0.02    # 2% base
RISK_PCT_EARLY_TREND = 0.03    # 3% when ADX 20-28 (flipped risk)
TP1_RR               = 2.0
TP2_RR               = 5.0
TRAIL_ATR_MULT       = 2.0
MAX_HOLD_BARS        = 96

# ── Filters (TBD — backtest will validate each) ───────────────────────────────
EMA200_PERIOD       = 200
ADX_PERIOD          = 14
ADX_THRESHOLD       = 20
ADX_EARLY_TREND_MAX = 28

# ── Paths ─────────────────────────────────────────────────────────────────────
_BOT_DIR = Path(__file__).parent
DATA_DIR  = _BOT_DIR / "data"
DB_PATH   = DATA_DIR / "btc2_trades.db"
LOG_DIR   = DATA_DIR / "logs"
