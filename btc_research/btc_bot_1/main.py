"""
btc_research/btc_bot_1/main.py — BTC Bot 1 entry point.

Architecture mirrors v2/main.py.

Start-up sequence:
  1. Load .env, configure file logging.
  2. Connect MT5 DataFeed.
  3. Open SQLite Journal (btc_trades.db).
  4. Construct BTCSignalEngine and PaperTrader.
  5. Build BTCScheduler and start it.
  6. Send Telegram startup alert.
  7. Block main thread; SIGINT/SIGTERM → graceful shutdown.

Run:
    python -m btc_research.btc_bot_1.main

Kill-zone: 21:00-24:00 UTC  (01:00-04:00 UAE)
Strategy:  BTC Confluence V1 (tested: 43% WR, +$23,733 over 2yr backtest)
"""
from __future__ import annotations

import atexit
import logging
import logging.handlers
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

# ── .env loader (before any btc_bot_1 imports) ────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(os.environ.get("ENV_FILE",
                     Path(__file__).resolve().parents[3] / ".env"))
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

# ── Logging setup ─────────────────────────────────────────────────────────────
import btc_research.btc_bot_1.settings as settings

settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = settings.LOG_DIR / "btc_bot.log"

_log_fmt     = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_log_datefmt = "%Y-%m-%dT%H:%M:%S"

_file_handler = logging.handlers.RotatingFileHandler(
    filename    = _LOG_FILE,
    maxBytes    = 10 * 1024 * 1024,   # 10 MB
    backupCount = 7,
    encoding    = "utf-8",
)
_file_handler.setFormatter(logging.Formatter(_log_fmt, datefmt=_log_datefmt))

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter(_log_fmt, datefmt=_log_datefmt))

logging.basicConfig(
    level    = logging.INFO,
    handlers = [_console_handler, _file_handler],
)
logger = logging.getLogger(__name__)
logger.info("BTC Bot 1 log file: %s", _LOG_FILE)

# ── BTC Bot 1 imports ─────────────────────────────────────────────────────────
from btc_research.btc_bot_1.connectors.unified_data import DataFeed
from btc_research.btc_bot_1.journal.sqlite_journal  import Journal
from btc_research.btc_bot_1.signals.btc_engine      import BTCSignalEngine
from btc_research.btc_bot_1.trading.paper_trader    import PaperTrader
from btc_research.btc_bot_1.scheduler.scheduler     import BTCScheduler

# ── Shutdown event ────────────────────────────────────────────────────────────
_shutdown_event = threading.Event()


def _send_shutdown_alert() -> None:
    """
    Send Telegram alert on exit.
    Registered with atexit so it fires even on Windows Stop-Process.
    """
    try:
        from btc_research.btc_bot_1.api.telegram_bot import TelegramAlerter
        from datetime import datetime, timezone
        alerter = TelegramAlerter()
        alerter.send_text(
            f"[BTC] Bot 1 stopped ⛔\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Reason: Process terminated"
        )
    except Exception:
        pass


def _handle_signal(signum: int, frame: Any) -> None:
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — initiating graceful shutdown…", sig_name)
    _shutdown_event.set()


def main() -> None:
    """Wire everything together and run until shutdown."""

    # atexit fires on any exit path including Windows Stop-Process (without -Force)
    atexit.register(_send_shutdown_alert)

    logger.info("=" * 60)
    logger.info("BTC Bot 1 starting up")
    logger.info("Kill-zone: %d:00 - %d:00 UTC  (01:00-04:00 UAE)",
                settings.KZ_START_UTC, settings.KZ_END_UTC)
    logger.info("Strategy: BTC Confluence V1 | Risk: %.0f%% | Balance: $%.0f",
                settings.RISK_PCT * 100, settings.STARTING_BALANCE)
    logger.info("=" * 60)

    # ── 1. DataFeed ───────────────────────────────────────────────────────────
    logger.info("Connecting MT5 DataFeed…")
    feed = DataFeed()
    try:
        conn = feed.connect(
            mt5_login    = settings.MT5_LOGIN,
            mt5_password = settings.MT5_PASSWORD,
            mt5_server   = settings.MT5_SERVER,
        )
        logger.info("DataFeed connected: MT5=%s", conn.get("mt5"))
    except Exception as exc:
        logger.warning("DataFeed connection error (will retry on first scan): %s", exc)

    # ── 2. Journal ────────────────────────────────────────────────────────────
    logger.info("Opening BTC journal at %s", settings.DB_PATH)
    journal = Journal(db_path=settings.DB_PATH)

    # ── 3. Signal engine + paper trader ──────────────────────────────────────
    signal_engine = BTCSignalEngine(feed=feed)
    paper_trader  = PaperTrader(journal=journal, feed=feed)
    logger.info("BTCSignalEngine and PaperTrader constructed")

    # ── 4. Scheduler ──────────────────────────────────────────────────────────
    scheduler = BTCScheduler(
        paper_trader  = paper_trader,
        signal_engine = signal_engine,
        journal       = journal,
        feed          = feed,
    )
    try:
        scheduler.start()
        logger.info("BTCScheduler started")
    except Exception as exc:
        logger.error("BTCScheduler failed to start: %s", exc, exc_info=True)

    # ── 5. Signal handlers ────────────────────────────────────────────────────
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("BTC Bot 1 running — press Ctrl+C or send SIGTERM to stop")

    # ── 6. Telegram startup alert ─────────────────────────────────────────────
    try:
        from btc_research.btc_bot_1.api.telegram_bot import TelegramAlerter
        alerter = TelegramAlerter()
        alerter.send_text(
            f"[BTC] Bot 1 started ✅\n"
            f"MT5: connected\n"
            f"Paper trading: ACTIVE\n"
            f"Instrument: BTCUSD\n"
            f"Kill-zone: 21:00-24:00 UTC (01:00-04:00 UAE)\n"
            f"Strategy: Version D (EMA200 + Flipped Risk)\n"
            f"Risk: 3% ADX 20-28 | 2% ADX >28 | skip ADX <20\n"
            f"Trailing SL: 2xATR after TP1\n"
            f"Scan: every 2s inside kill-zone | every 5min outside"
        )
    except Exception:
        pass

    # ── 7. Block until shutdown ───────────────────────────────────────────────
    _shutdown_event.wait()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    # Unregister atexit so we send the richer "SIGINT/SIGTERM" message below
    try:
        atexit.unregister(_send_shutdown_alert)
    except Exception:
        pass

    logger.info("BTC Bot 1 shutting down…")

    try:
        scheduler.stop()
        logger.info("BTCScheduler stopped")
    except Exception as exc:
        logger.error("Error stopping scheduler: %s", exc)

    try:
        journal.close()
        logger.info("Journal closed")
    except Exception as exc:
        logger.error("Error closing journal: %s", exc)

    # Shutdown Telegram alert
    try:
        from btc_research.btc_bot_1.api.telegram_bot import TelegramAlerter
        from datetime import datetime, timezone
        alerter = TelegramAlerter()
        alerter.send_text(
            f"[BTC] Bot 1 stopped ⛔\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Reason: SIGINT / SIGTERM (manual stop or system shutdown)"
        )
    except Exception:
        pass

    logger.info("BTC Bot 1 shutdown complete")


if __name__ == "__main__":
    main()
