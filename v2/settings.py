"""
settings.py — TradingBotV2 global configuration
All environment-variable overrides go here. No magic strings elsewhere.
"""
from __future__ import annotations
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
DB_PATH     = DATA_DIR / "trades.db"
LOG_DIR     = DATA_DIR / "logs"
MODEL_DIR   = DATA_DIR / "models"

for _d in (DATA_DIR, LOG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Account ───────────────────────────────────────────────────────────────────
ACCOUNT_BALANCE    = float(os.environ.get("ACCOUNT_BALANCE", "10000"))   # USD
RISK_PER_TRADE_PCT = float(os.environ.get("RISK_PER_TRADE_PCT", "1.0"))  # % per trade
MAX_OPEN_TRADES    = int(os.environ.get("MAX_OPEN_TRADES", "6"))
MAX_PORTFOLIO_HEAT = float(os.environ.get("MAX_PORTFOLIO_HEAT", "30.0")) # % total at risk
DAILY_LOSS_LIMIT   = float(os.environ.get("DAILY_LOSS_LIMIT", "3.0"))    # % of account
WEEKLY_LOSS_LIMIT  = float(os.environ.get("WEEKLY_LOSS_LIMIT", "6.0"))   # % of account

# ── Signal thresholds ─────────────────────────────────────────────────────────
MIN_CONFLUENCE_SCORE = float(os.environ.get("MIN_CONFLUENCE_SCORE", "3.0"))
MIN_RR_RATIO         = float(os.environ.get("MIN_RR_RATIO", "2.0"))
MAX_SPREAD_PIPS      = float(os.environ.get("MAX_SPREAD_PIPS", "3.0"))

# ── Broker credentials (MT5) ──────────────────────────────────────────────────
MT5_LOGIN    = int(os.environ.get("MT5_LOGIN", "0"))
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER   = os.environ.get("MT5_SERVER", "")

# ── Binance credentials ───────────────────────────────────────────────────────
BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
BINANCE_TESTNET    = os.environ.get("BINANCE_TESTNET", "true").lower() == "true"

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── API server ────────────────────────────────────────────────────────────────
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# ── ML ────────────────────────────────────────────────────────────────────────
ML_MIN_TRADES_TO_TRAIN = int(os.environ.get("ML_MIN_TRADES_TO_TRAIN", "50"))
ML_RETRAIN_INTERVAL    = int(os.environ.get("ML_RETRAIN_INTERVAL", "50"))   # every N new trades

# ── Timezone ──────────────────────────────────────────────────────────────────
TIMEZONE_OFFSET_HOURS = 4   # GST (UTC+4)

# ── Scheduler intervals ───────────────────────────────────────────────────────
TRADE_MONITOR_INTERVAL_SEC = 60
H1_SCAN_INTERVAL_MIN       = 60
H4_SCAN_INTERVAL_MIN       = 240
MORNING_BRIEFING_HOUR_GST  = 6
NIGHTLY_RETRAIN_HOUR_GST   = 23
