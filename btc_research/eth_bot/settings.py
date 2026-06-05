"""
btc_research/eth_bot/settings.py — ETH Bot runtime configuration.

Fully standalone — reads from .env, no dependency on any other bot's settings.

== KILL-ZONE ==
  Placeholder hours set to [1, 2, 3, 8] UTC (same as BTC Bot 2).
  Run backtests to determine optimal ETH kill-zone hours before going live.
  Change ETH_KZ_HOURS in .env or edit KZ_HOURS directly below.

== STRATEGY ==
  Framework uses VB + SwingLevel v2 as a starting point (same as BTC Bot 2).
  Run backtests to validate or swap strategy for ETH's specific behaviour.

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
# Placeholder: [1, 2, 3, 8] UTC — same as BTC Bot 2 (Asia Night + EU Open).
# !! Run ETH backtest to confirm these are optimal for ETH before live trading !!
_kz_env = os.environ.get("ETH_KZ_HOURS", "")
if _kz_env:
    KZ_HOURS: list[int] = [int(h.strip()) for h in _kz_env.split(",") if h.strip()]
else:
    KZ_HOURS = [1, 2, 3, 8]

# ── Risk & position sizing ─────────────────────────────────────────────────────
# Same ADX-split logic as BTC Bot 2 — validated on crypto in general.
# Revisit after ETH-specific backtest.
STARTING_BALANCE      = 500.0   # USD paper trading account
RISK_PCT_EARLY_TREND  = 0.03    # 3% — ADX ≤ ADX_SPLIT_EARLY_MAX (early trend)
RISK_PCT_TRANSITION   = 0.02    # 2% — ADX between early and strong (dead zone)
RISK_PCT_STRONG       = 0.03    # 3% — ADX ≥ ADX_SPLIT_STRONG_MIN (strong trend)

ADX_SPLIT_EARLY_MAX   = 25      # ADX ≤ 25  → early trend → 3% risk
ADX_SPLIT_STRONG_MIN  = 40      # ADX ≥ 40  → strong trend → 3% risk
                                 # ADX 25-40 → transition  → 2% risk

# ── TP / SL ratios ─────────────────────────────────────────────────────────────
TP1_RR          = 2.0    # TP1 at 2R — partial close, SL to breakeven
TP2_RR          = 5.0    # TP2 at 5R — full close
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
