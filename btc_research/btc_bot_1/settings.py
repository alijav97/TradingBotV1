"""
btc_research/btc_bot_1/settings.py — BTC Bot 1 runtime configuration.

Reads from environment variables / .env file.
Completely isolated from v2/settings.py — no cross-contamination.

Add to your .env file (or VPS environment):
    BTC_TELEGRAM_BOT_TOKEN=<token>
    BTC_TELEGRAM_CHAT_ID=<chat_id>
    BTC_API_PORT=8001           (optional, defaults to 8001)
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER are shared with v2 (same Pepperstone account)
"""
from __future__ import annotations

import os
from pathlib import Path

# ── .env loader ───────────────────────────────────────────────────────────────
# Load from the same .env as the WTI bot (repo root or C:\TradingBotV2\.env)
try:
    from dotenv import load_dotenv
    _env_path = Path(os.environ.get("ENV_FILE",
                     Path(__file__).resolve().parents[3] / ".env"))
    if _env_path.exists():
        load_dotenv(_env_path, override=False)   # don't override already-set vars
except ImportError:
    pass

# ── Telegram — dedicated BTCPaperTrader bot ────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("BTC_TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("BTC_TELEGRAM_CHAT_ID",   "")

# ── MT5 credentials (shared Pepperstone account) ──────────────────────────────
MT5_LOGIN    = int(os.environ.get("MT5_LOGIN",    "0") or "0")
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER   = os.environ.get("MT5_SERVER",   "")
MT5_SERVER_UTC_OFFSET = 3  # Pepperstone server is UTC+3

# ── Symbols ───────────────────────────────────────────────────────────────────
SYMBOL      = "BTCUSD"    # Main instrument
GOLD_SYMBOL = "XAUUSD"   # Inter-market correlation factor
NAS_SYMBOL  = "NAS100"   # Risk-on/off factor

# ── Kill-zone (US Late session — best combined-strategy session) ───────────────
KZ_START_UTC = 21
KZ_END_UTC   = 24   # midnight UTC

# ── Risk & position sizing (Version D — EMA200 + Flipped Risk) ───────────────
STARTING_BALANCE     = 500.0   # USD paper trading account
RISK_PCT             = 0.02    # 2% base risk (used when ADX > 28 — extended trend)
RISK_PCT_EARLY_TREND = 0.03    # 3% risk when ADX 20-28 — early trend (flipped risk)
TP1_RR               = 2.0    # Partial close + SL to breakeven
TP2_RR               = 5.0    # Fallback fixed TP2 (used if trailing SL not active)
TRAIL_ATR_MULT       = 2.0    # Trailing SL after TP1: move SL to peak - 2×ATR
MAX_HOLD_HOURS       = 96     # 4 days — force-close if still open

# ── Version D entry filters ───────────────────────────────────────────────────
# EMA200 filter: only longs above EMA200, only shorts below EMA200
# ADX filter:    skip if ADX < 20 (no clear trend)
# Flipped risk:  3% when ADX 20-28 (early trend), 2% when ADX > 28 (extended)
EMA200_PERIOD        = 200    # EMA200 trend filter
ADX_PERIOD           = 14     # ADX period
ADX_THRESHOLD        = 20     # minimum ADX — below this, skip trade
ADX_EARLY_TREND_MAX  = 28     # ADX <= this → early trend → use RISK_PCT_EARLY_TREND

# ── Signal filtering ──────────────────────────────────────────────────────────
MIN_CONFLUENCE_SCORE = 3.0  # same as backtest setting

# ── Paths ─────────────────────────────────────────────────────────────────────
_BOT_DIR = Path(__file__).parent
DATA_DIR  = _BOT_DIR / "data"
DB_PATH   = DATA_DIR / "btc_trades.db"
LOG_DIR   = DATA_DIR / "logs"

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = os.environ.get("BTC_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("BTC_API_PORT", "8001"))
