"""
btc_research/eth_bot/settings.py — ETH Bot runtime configuration.

Fully standalone — reads from .env, no dependency on any other bot's settings.

== KILL-ZONE ==
  KZ_HOURS = [2, 14, 15, 16] UTC — derived from 6-year ETH backtest.
    02 UTC      → RSI 50-Cross (Asia Night, 50.9% WR)
    14-16 UTC   → Swing Break + Keltner Channel (London Close / NY, ~55-65% combined WR)
  Override via ETH_KZ_HOURS in .env (comma-separated, e.g. "2,14,15,16").

== STRATEGY ==
  Swing+Keltner at 14-16 UTC (primary) | RSI 50-Cross at 02 UTC (secondary).
  Determined by 6-year ETH backtest — see eth_combined.py for full rationale.

== .env KEYS ==
  ETH_TELEGRAM_BOT_TOKEN  — Telegram bot token (dedicated ETH bot from BotFather)
  ETH_TELEGRAM_CHAT_ID    — Telegram chat/channel ID for ETH alerts
  ETH_KZ_HOURS            — comma-separated UTC hours e.g. "1,2,3,8"
  ETH_API_PORT            — FastAPI port (default 8003)

  Shared with other bots (same Pepperstone MT5 account):
  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
"""
from __future__ import annotations

import os
from pathlib import Path

# ── .env loader ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(os.environ.get("ENV_FILE",
                     Path(__file__).resolve().parents[2] / ".env"))
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

# ── Telegram — dedicated ETH bot ───────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("ETH_TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("ETH_TELEGRAM_CHAT_ID",   "")

# ── MT5 credentials (shared Pepperstone account) ───────────────────────────────
MT5_LOGIN             = int(os.environ.get("MT5_LOGIN",    "0") or "0")
MT5_PASSWORD          = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER            = os.environ.get("MT5_SERVER",   "")
MT5_SERVER_UTC_OFFSET = 3   # Pepperstone server is UTC+3

# ── Symbols ────────────────────────────────────────────────────────────────────
SYMBOL = "ETHUSD"   # Pepperstone MT5 symbol for Ethereum

# ── Kill-zone hours (UTC) ──────────────────────────────────────────────────────
# Set from 6-year ETH backtest results (23 strategies, 43,955 signals):
#   02 UTC  → RSI 50-Cross:          WR=50.9%, AvgR=+0.754, PF=2.54  (N=55)
#   14 UTC  → Swing+Keltner:         WR≈48% individually, ~55-65% combined
#   15 UTC  → Keltner Channel peak:  WR=53.7%, AvgR=+0.508, PF=2.10  (N=82)
#   16 UTC  → Swing Break volume:    highest N, consistent edge
# Hours 14-16 map to London Close / NY session — ETH's strongest volatility window.
_kz_env = os.environ.get("ETH_KZ_HOURS", "")
if _kz_env:
    KZ_HOURS: list[int] = [int(h.strip()) for h in _kz_env.split(",") if h.strip()]
else:
    KZ_HOURS = [2, 14, 15, 16]

# ── Risk & position sizing ─────────────────────────────────────────────────────
# Same ADX-split logic as BTC Bot 2 — validated on crypto in general.
# Revisit after ETH-specific backtest.
STARTING_BALANCE      = 500.0   # USD paper trading account
RISK_PCT_EARLY_TREND  = 0.03    # 3% — ADX ≤ 25 (early trend, lower conviction)
RISK_PCT_TRANSITION   = 0.02    # 2% — ADX 25-40 (transition / dead zone)
RISK_PCT_STRONG       = 0.04    # 4% — ADX ≥ 40 (strong trend + double confluence = high conviction)
                                 # 4% on $500 = $20 at risk per trade in the strong zone

ADX_SPLIT_EARLY_MAX   = 25      # ADX ≤ 25  → early trend → 3% risk
ADX_SPLIT_STRONG_MIN  = 40      # ADX ≥ 40  → strong trend → 3% risk
                                 # ADX 25-40 → transition  → 2% risk

# ── TP / SL ratios ─────────────────────────────────────────────────────────────
TP1_RR          = 2.0    # TP1 at 2R — partial close (50%), SL to breakeven
TP2_RR          = 4.0    # TP2 at 4R — full close (remaining 50%)
                          # Reduced from 5R: phase-1 backtest showed avg_r ~0.5R
                          # across all strategies → very few trades reached 5R on ETH.
                          # 4R materially increases hit rate while preserving 2:1 TP2/TP1 ratio.
TRAIL_ATR_MULT  = 2.0    # Trailing SL after TP1: peak/trough ± 2×ATR
MAX_HOLD_BARS   = 96     # 96 H1 bars = 4 days — force-close if still open

# ── Signal filters ─────────────────────────────────────────────────────────────
ADX_THRESHOLD   = 20     # Skip trade if ADX < 20 (no clear trend)
ADX_PERIOD      = 14
EMA200_PERIOD   = 200    # EMA200 — only longs above, only shorts below
MIN_CONFLUENCE_SCORE = 0.0   # Placeholder — set after backtest

# ── Strategy parameters ────────────────────────────────────────────────────────
# SwingLevel v2 mode — "both" checks retest first then break (same as BTC Bot 2)
SWING_ENTRY_MODE = "both"
SWING_MAX_SL_ATR = 2.0    # SL cap for break entries (× ATR)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BOT_DIR = Path(__file__).parent
DATA_DIR = _BOT_DIR / "data"
DB_PATH  = DATA_DIR / "eth_trades.db"
LOG_DIR  = DATA_DIR / "logs"

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = os.environ.get("ETH_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("ETH_API_PORT", "8003"))
