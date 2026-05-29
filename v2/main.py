"""
main.py — TradingBotV2 entry point.

Wires all subsystems together and keeps the main thread alive with clean
SIGINT/SIGTERM handling.

Start-up sequence
-----------------
1.  Load settings (env vars → v2/settings.py).
2.  Connect DataFeed (MT5 + Binance — partial connectivity is fine).
3.  Open SQLite Journal.
4.  Construct ConfluenceEngine, PaperTrader, and AutoTrader.
5.  Build BotScheduler and start it (registers all APScheduler jobs in a
    background thread).
6.  Launch FastAPI via uvicorn in a separate daemon thread.
7.  Block the main thread; handle SIGINT/SIGTERM → graceful shutdown.

Run
---
    python -m v2.main
    # or from the repo root:
    python v2/main.py

Docker
------
    uvicorn v2.api.api_server:app --host 0.0.0.0 --port 8000
    (main.py is for direct VPS execution; the Dockerfile CMD uses uvicorn directly)
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

# Load .env from C:\TradingBotV2\.env or repo root before anything else
try:
    from dotenv import load_dotenv
    _env_path = Path(os.environ.get("ENV_FILE", Path(__file__).parent.parent / ".env"))
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

# ── Logging must be configured before any v2 imports so all modules pick it up.
# Log directory is created by settings.py on import; define the path here
# before importing settings so we can use it in the handler.
_LOG_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data")) / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "bot.log"

_log_fmt     = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_log_datefmt = "%Y-%m-%dT%H:%M:%S"

# Rotating file handler — 10 MB per file, keep last 7 files (~70 MB max)
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
logger.info("Log file: %s", _LOG_FILE)

# ── V2 imports ────────────────────────────────────────────────────────────────
import v2.settings as settings
from v2.connectors.unified_data import DataFeed
from v2.journal.sqlite_journal import Journal
from v2.signals.confluence_engine import ConfluenceEngine
from v2.trading.paper_trader import PaperTrader
from v2.trading.auto_trader import AutoTrader
from v2.scheduler.scheduler import BotScheduler


# ── Shutdown event shared between signal handlers and the main loop ───────────
_shutdown_event = threading.Event()


def _handle_signal(signum: int, frame: Any) -> None:
    """SIGINT / SIGTERM handler — signals the main loop to exit cleanly."""
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — initiating graceful shutdown…", sig_name)
    _shutdown_event.set()


def _start_api_server(host: str, port: int) -> threading.Thread:
    """
    Launch the FastAPI app (uvicorn) in a daemon thread.

    Using a daemon thread means Python will not wait for uvicorn to finish
    when the main thread exits — the BotScheduler shutdown happens first,
    then the process terminates.
    """
    import uvicorn
    from v2.api.api_server import app

    config = uvicorn.Config(
        app     = app,
        host    = host,
        port    = port,
        log_level = "info",
        access_log = True,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(
        target = server.run,
        name   = "uvicorn",
        daemon = True,
    )
    thread.start()
    logger.info("FastAPI server starting on http://%s:%d", host, port)
    return thread


def main() -> None:
    """Wire everything together and keep the process alive until shutdown."""

    logger.info("=" * 60)
    logger.info("TradingBotV2 starting up")
    logger.info("=" * 60)

    # ── 1. Settings are module-level constants; log key values. ──────────────
    logger.info(
        "Settings loaded: balance=$%.0f risk=%.1f%% daily_limit=%.1f%%",
        settings.ACCOUNT_BALANCE,
        settings.RISK_PER_TRADE_PCT,
        settings.DAILY_LOSS_LIMIT,
    )

    # ── 2. DataFeed ───────────────────────────────────────────────────────────
    logger.info("Connecting DataFeed…")
    feed = DataFeed()
    try:
        conn_status = feed.connect(
            mt5_login         = settings.MT5_LOGIN,
            mt5_password      = settings.MT5_PASSWORD,
            mt5_server        = settings.MT5_SERVER,
            binance_api_key   = settings.BINANCE_API_KEY,
            binance_api_secret= settings.BINANCE_API_SECRET,
            binance_testnet   = settings.BINANCE_TESTNET,
        )
        logger.info(
            "DataFeed connected: MT5=%s  Binance=%s",
            conn_status.get("mt5"),
            conn_status.get("binance"),
        )
    except Exception as exc:
        # Partial connectivity is acceptable — stub connectors will return
        # empty DataFrames, and the scheduler will skip affected symbols.
        logger.warning("DataFeed connection error (partial ok): %s", exc)

    # ── 3. Journal ────────────────────────────────────────────────────────────
    logger.info("Opening journal at %s", settings.DB_PATH)
    journal = Journal(db_path=settings.DB_PATH)

    # ── 4. Core signal / trading objects ─────────────────────────────────────
    confluence_engine = ConfluenceEngine(
        min_score=settings.MIN_CONFLUENCE_SCORE
    )
    paper_trader = PaperTrader(journal=journal, feed=feed)
    auto_trader  = AutoTrader(
        paper_trader      = paper_trader,
        confluence_engine = confluence_engine,
        journal           = journal,
        feed              = feed,
    )
    logger.info("ConfluenceEngine, PaperTrader, AutoTrader constructed")

    # ── 5. Scheduler ─────────────────────────────────────────────────────────
    scheduler = BotScheduler(
        paper_trader = paper_trader,
        confluence   = confluence_engine,
        journal      = journal,
        feed         = feed,
    )
    try:
        scheduler.start()
        logger.info("BotScheduler started successfully")
    except Exception as exc:
        logger.error("BotScheduler failed to start: %s", exc, exc_info=True)
        # Non-fatal — the API server can still serve status endpoints.

    # ── 6. FastAPI (uvicorn in daemon thread) ─────────────────────────────────
    api_thread = _start_api_server(
        host = settings.API_HOST,
        port = settings.API_PORT,
    )

    # ── 7. Signal handlers + main-thread block ────────────────────────────────
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("TradingBotV2 running — press Ctrl+C or send SIGTERM to stop")

    # Send Telegram startup notification
    try:
        from v2.api.telegram_bot import TelegramAlerter
        from v2.instrument_config import ALL_SYMBOLS
        active = settings.ACTIVE_SYMBOLS if settings.ACTIVE_SYMBOLS else ALL_SYMBOLS
        alerter = TelegramAlerter()
        alerter.send_text(
            f"TradingBotV2 started ✅\n"
            f"MT5: connected\n"
            f"Paper trading: ACTIVE\n"
            f"Instruments: {', '.join(active)}\n"
            f"Kill-zone: 13:00-17:00 UTC (5PM-9PM UAE)\n"
            f"Scan: every 2s inside kill-zone | every 5min outside"
        )
    except Exception:
        pass

    # Block here until a shutdown signal is received.
    _shutdown_event.wait()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down...")

    try:
        scheduler.stop()
        logger.info("BotScheduler stopped")
    except Exception as exc:
        logger.error("Error stopping scheduler: %s", exc)

    try:
        journal.close()
        logger.info("Journal closed")
    except Exception as exc:
        logger.error("Error closing journal: %s", exc)

    # Send Telegram shutdown notification
    try:
        from v2.api.telegram_bot import TelegramAlerter
        from datetime import datetime, timezone
        alerter = TelegramAlerter()
        alerter.send_text(
            f"TradingBotV2 stopped ⛔\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Reason: SIGINT / SIGTERM (manual stop or system shutdown)"
        )
    except Exception:
        pass

    logger.info("TradingBotV2 shutdown complete")


if __name__ == "__main__":
    main()
