"""
btc_research/eth_bot/main.py — ETH Bot entry point.

Mirrors btc_bot_2/main.py exactly.

Start-up sequence:
  1. Load .env, configure file logging (UTC timestamps).
  2. Connect MT5 DataFeed (Pepperstone ETHUSD).
  3. Open SQLite Journal (eth_trades.db).
  4. Construct ETHSignalEngine, ETHPaperTrader, ETHAlerter.
  5. Build ETHScheduler and start it.
  6. Register atexit for crash/kill shutdown alert.
  7. Send Telegram startup alert.
  8. Block main thread; SIGINT/SIGTERM → graceful shutdown.

Run:
    python -m btc_research.eth_bot.main

Kill-zone: configured in settings.py (default [1, 2, 3, 8] UTC — pending backtest)
Strategy:  Swing Level v2 + VB placeholder (pending ETH backtest)

== Environment variables ==
  ETH_TELEGRAM_BOT_TOKEN  — Telegram bot token (dedicated ETH bot from BotFather)
  ETH_TELEGRAM_CHAT_ID    — chat/channel ID for ETH alerts
  ETH_API_PORT            — FastAPI port (default 8003)
  ETH_KZ_HOURS            — optional: override kill-zone hours e.g. "1,2,3,8"

  Shared MT5 credentials (same Pepperstone account as BTC bots):
  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
"""
from __future__ import annotations

import atexit
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time as _time
from pathlib import Path
from typing import Any

# ── .env loader (before any eth_bot imports) ───────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(os.environ.get("ENV_FILE",
                     Path(__file__).resolve().parents[2] / ".env"))
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

# ── Logging setup (UTC timestamps) ────────────────────────────────────────────
from btc_research.eth_bot.settings import DATA_DIR, LOG_DIR, API_PORT, KZ_HOURS, STARTING_BALANCE, DB_PATH

LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = LOG_DIR / "eth_bot.log"

_log_fmt     = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_log_datefmt = "%Y-%m-%dT%H:%M:%S"

_file_handler = logging.handlers.RotatingFileHandler(
    filename    = _LOG_FILE,
    maxBytes    = 10 * 1024 * 1024,   # 10 MB
    backupCount = 7,
    encoding    = "utf-8",
)
_fmt_utc = logging.Formatter(_log_fmt, datefmt=_log_datefmt)
_fmt_utc.converter = _time.gmtime
_file_handler.setFormatter(_fmt_utc)

_console_handler = logging.StreamHandler(sys.stdout)
_console_fmt = logging.Formatter(_log_fmt, datefmt=_log_datefmt)
_console_fmt.converter = _time.gmtime
_console_handler.setFormatter(_console_fmt)

logging.basicConfig(
    level    = logging.INFO,
    handlers = [_console_handler, _file_handler],
)
logger = logging.getLogger(__name__)
logger.info("ETH Bot log file: %s", _LOG_FILE)

# ── ETH Bot imports ────────────────────────────────────────────────────────────
from btc_research.eth_bot.signal_engine              import ETHSignalEngine
from btc_research.eth_bot.paper_trader               import ETHPaperTrader
from btc_research.eth_bot.scheduler                  import ETHScheduler
from btc_research.eth_bot.telegram                   import ETHAlerter
from btc_research.eth_bot.connectors.unified_data    import DataFeed
from btc_research.eth_bot.journal.sqlite_journal     import Journal

# ── Shutdown event ─────────────────────────────────────────────────────────────
_shutdown_event = threading.Event()


def _send_shutdown_alert() -> None:
    try:
        alerter = ETHAlerter()
        alerter.send_shutdown("Process terminated")
    except Exception:
        pass


def _handle_signal(signum: int, frame: Any) -> None:
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — initiating graceful shutdown...", sig_name)
    _shutdown_event.set()


def main() -> None:
    """Wire all ETH Bot subsystems and keep the process running."""

    atexit.register(_send_shutdown_alert)

    kz_str = ", ".join(f"{h:02d}:00" for h in KZ_HOURS)

    logger.info("=" * 60)
    logger.info("ETH Bot starting up")
    logger.info("Kill-zone : %s UTC  (placeholder — confirm after backtest)", kz_str)
    logger.info("Strategy  : Swing Level v2 + VB (placeholder — pending ETH backtest)")
    logger.info("Risk      : 3%% ADX<=25 | 2%% ADX 25-40 | 3%% ADX>=40")
    logger.info("Balance   : $%.0f starting", STARTING_BALANCE)
    logger.info("API port  : %d", API_PORT)
    logger.info("DB        : %s", DB_PATH)
    logger.info("=" * 60)

    # ── 1. DataFeed (MT5 — Pepperstone ETHUSD) ────────────────────────────────
    logger.info("Connecting MT5 DataFeed (Pepperstone ETHUSD)...")
    feed = DataFeed()
    try:
        conn = feed.connect()
        logger.info("DataFeed connected: MT5=%s", conn.get("mt5"))
        if not conn.get("mt5"):
            logger.warning(
                "MT5 not connected — check MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in .env"
            )
    except Exception as exc:
        logger.warning("DataFeed connect error (will retry on first scan): %s", exc)

    # ── 2. Journal (eth_trades.db) ────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Opening journal: %s", DB_PATH)
    journal = Journal(db_path=DB_PATH)

    # ── 3. Core components ────────────────────────────────────────────────────
    alerter      = ETHAlerter()
    engine       = ETHSignalEngine(feed=feed, journal=journal)
    paper_trader = ETHPaperTrader(journal=journal, feed=feed)
    logger.info("ETHSignalEngine, ETHPaperTrader, ETHAlerter constructed")

    # ── 4. Scheduler ──────────────────────────────────────────────────────────
    scheduler = ETHScheduler(
        engine       = engine,
        paper_trader = paper_trader,
        journal      = journal,
        alerter      = alerter,
    )
    try:
        scheduler.start()
        logger.info("ETHScheduler started")
    except Exception as exc:
        logger.error("ETHScheduler failed to start: %s", exc, exc_info=True)

    # ── 5. Signal handlers ────────────────────────────────────────────────────
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("ETH Bot running — press Ctrl+C or send SIGTERM to stop")

    # ── 6. Telegram startup alert ─────────────────────────────────────────────
    try:
        balance = journal.get_paper_balance()
        alerter.send_startup(balance)
    except Exception:
        pass

    # ── 7. Block until shutdown ───────────────────────────────────────────────
    _shutdown_event.wait()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    try:
        atexit.unregister(_send_shutdown_alert)
    except Exception:
        pass

    logger.info("ETH Bot shutting down...")

    try:
        scheduler.stop()
        logger.info("Scheduler stopped")
    except Exception as exc:
        logger.error("Scheduler stop error: %s", exc)

    try:
        journal.close()
        logger.info("Journal closed")
    except Exception as exc:
        logger.error("Journal close error: %s", exc)

    try:
        alerter.send_shutdown("SIGINT / SIGTERM (manual stop)")
    except Exception:
        pass

    logger.info("ETH Bot shutdown complete")


if __name__ == "__main__":
    main()
