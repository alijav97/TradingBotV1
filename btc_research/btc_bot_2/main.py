"""
btc_research/btc_bot_2/main.py — BTC Bot 2 entry point.

Wires all BTC Bot 2 subsystems together:
  1. Load .env (tokens, credentials)
  2. Configure logging (console + rotating file)
  3. Connect DataFeed (Binance for BTCUSDT)
  4. Open Journal (btc2_trades.db)
  5. Construct SignalEngine, BTC2PaperTrader, BTC2Alerter
  6. Build BTC2Scheduler and start it
  7. Launch FastAPI (port 8002) in a daemon thread
  8. Block main thread; SIGINT/SIGTERM → graceful shutdown

== Run ==
    python -m btc_research.btc_bot_2.main
    # or from project root:
    python btc_research/btc_bot_2/main.py

== Config ==
  Environment variables (in .env or shell):
    TELEGRAM_BOT_TOKEN       — shared with Bot 1
    BTC2_TELEGRAM_CHAT_ID    — Bot 2 Telegram channel (or use TELEGRAM_CHAT_ID)
    BTC2_API_KEY             — read-only API key for /health /trades /performance
    BTC2_API_KEY_FULL        — full API key for POST /scan
    BTC2_API_PORT            — API port (default 8002)
    BINANCE_API_KEY          — Binance API key (for live price data)
    BINANCE_API_SECRET       — Binance API secret
    BINANCE_TESTNET          — "true" for testnet (default true)
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER — optional (not needed for BTC)
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

# ── .env loading ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(os.environ.get("ENV_FILE", Path(__file__).parent.parent.parent / ".env"))
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

# ── Logging setup (before any imports that log) ───────────────────────────────
from btc_research.btc_bot_2.settings import DATA_DIR, LOG_DIR, API_PORT

LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = LOG_DIR / "btc_bot_2.log"

_fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

_file_handler = logging.handlers.RotatingFileHandler(
    filename    = _LOG_FILE,
    maxBytes    = 10 * 1024 * 1024,
    backupCount = 7,
    encoding    = "utf-8",
)
_file_handler.setFormatter(logging.Formatter(_fmt))

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter(_fmt))

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)
logger.info("Log file: %s", _LOG_FILE)

# ── BTC Bot 2 imports ─────────────────────────────────────────────────────────
from btc_research.btc_bot_2.settings   import STARTING_BALANCE, DB_PATH
from btc_research.btc_bot_2.signal_engine import SignalEngine
from btc_research.btc_bot_2.paper_trader  import BTC2PaperTrader
from btc_research.btc_bot_2.scheduler     import BTC2Scheduler
from btc_research.btc_bot_2.telegram      import BTC2Alerter
from btc_research.btc_bot_2.api           import app as api_app, set_app_state

# ── Shared infrastructure from v2 ────────────────────────────────────────────
from v2.connectors.unified_data import DataFeed
from v2.journal.sqlite_journal  import Journal

# ── Shutdown event ────────────────────────────────────────────────────────────
_shutdown_event = threading.Event()


def _handle_signal(signum: int, frame: Any) -> None:
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — initiating graceful shutdown…", sig_name)
    _shutdown_event.set()


def _start_api_server(port: int) -> threading.Thread:
    """Launch FastAPI in a background daemon thread."""
    import uvicorn

    config = uvicorn.Config(
        app       = api_app,
        host      = "0.0.0.0",
        port      = port,
        log_level = "info",
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, name="btc2-api", daemon=True)
    t.start()
    logger.info("FastAPI started on http://0.0.0.0:%d", port)
    return t


def main() -> None:
    """Wire all Bot 2 subsystems and keep the process running."""

    logger.info("=" * 60)
    logger.info("BTC Bot 2 starting")
    logger.info("Strategy : VB + Swing Level Break v2  [both 2×ATR]")
    logger.info("KZ hours : 01:00, 02:00, 03:00, 08:00 UTC")
    logger.info("API port : %d", API_PORT)
    logger.info("DB       : %s", DB_PATH)
    logger.info("=" * 60)

    # ── 1. DataFeed (Binance for BTCUSDT) ─────────────────────────────────────
    logger.info("Connecting DataFeed…")
    feed = DataFeed()
    try:
        conn = feed.connect(
            mt5_login          = int(os.environ.get("MT5_LOGIN", "0")),
            mt5_password       = os.environ.get("MT5_PASSWORD", ""),
            mt5_server         = os.environ.get("MT5_SERVER", ""),
            binance_api_key    = os.environ.get("BINANCE_API_KEY", ""),
            binance_api_secret = os.environ.get("BINANCE_API_SECRET", ""),
            binance_testnet    = os.environ.get("BINANCE_TESTNET", "true").lower() == "true",
        )
        logger.info(
            "DataFeed: MT5=%s  Binance=%s",
            conn.get("mt5"),
            conn.get("binance"),
        )
    except Exception as exc:
        logger.warning("DataFeed connect error (partial ok): %s", exc)

    # ── 2. Journal (btc2_trades.db) ───────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Opening journal: %s", DB_PATH)
    journal = Journal(db_path=DB_PATH)

    # ── 3. Core components ────────────────────────────────────────────────────
    alerter      = BTC2Alerter()
    engine       = SignalEngine(feed=feed, journal=journal)
    paper_trader = BTC2PaperTrader(journal=journal, feed=feed)
    logger.info("SignalEngine, BTC2PaperTrader, BTC2Alerter constructed")

    # ── 4. Inject into API ────────────────────────────────────────────────────
    set_app_state(journal=journal, engine=engine, paper_trader=paper_trader)

    # ── 5. Scheduler ─────────────────────────────────────────────────────────
    scheduler = BTC2Scheduler(
        engine       = engine,
        paper_trader = paper_trader,
        journal      = journal,
        alerter      = alerter,
    )
    try:
        scheduler.start()
        logger.info("BTC2Scheduler started")
    except Exception as exc:
        logger.error("BTC2Scheduler failed: %s", exc, exc_info=True)

    # ── 6. FastAPI ────────────────────────────────────────────────────────────
    _start_api_server(API_PORT)

    # ── 7. Signal handlers ────────────────────────────────────────────────────
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Startup Telegram alert ────────────────────────────────────────────────
    try:
        balance = STARTING_BALANCE
        try:
            stats   = journal.get_stats(days=9999)
            balance = stats.get("current_balance") or STARTING_BALANCE
        except Exception:
            pass
        alerter.send_startup(balance)
    except Exception:
        pass

    logger.info("BTC Bot 2 running — press Ctrl+C to stop")
    _shutdown_event.wait()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down…")

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
        alerter.send_shutdown("SIGINT / SIGTERM")
    except Exception:
        pass

    logger.info("BTC Bot 2 shutdown complete")


if __name__ == "__main__":
    main()
