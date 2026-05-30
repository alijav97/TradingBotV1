"""
btc_research/btc_bot_2/main.py — BTC Bot 2 entry point.

Mirrors Bot 1's main.py pattern exactly.

Start-up sequence:
  1. Load .env, configure file logging (UTC timestamps).
  2. Connect MT5 DataFeed (Pepperstone BTCUSD — same connection as Bot 1).
  3. Open SQLite Journal (btc2_trades.db).
  4. Construct SignalEngine, BTC2PaperTrader, BTC2Alerter.
  5. Build BTC2Scheduler and start it.
  6. Register atexit for crash/kill shutdown alert.
  7. Send Telegram startup alert.
  8. Block main thread; SIGINT/SIGTERM → graceful shutdown.

Run:
    python -m btc_research.btc_bot_2.main

Kill-zone: 01:00, 02:00, 03:00, 08:00 UTC  (05:00-07:00, 12:00 UAE)
Strategy:  SwingLevelBreak v2 [both 2xATR] + VB fallback (46.7% WR, $119k 2yr)

== Environment variables ==

  Telegram (Bot 2's own dedicated bot — recommended):
    BTC2_TELEGRAM_BOT_TOKEN  — token from BotFather for BTC Bot 2's bot
    BTC2_TELEGRAM_CHAT_ID    — chat/channel ID for Bot 2 alerts
  Telegram (shared fallback — uses Bot 1's bot):
    TELEGRAM_BOT_TOKEN       — shared bot token fallback
    TELEGRAM_CHAT_ID         — shared chat ID fallback

  MT5 (shared Pepperstone account — same as Bot 1):
    MT5_ACCOUNT   — Pepperstone account number
    MT5_PASSWORD  — Pepperstone password
    MT5_SERVER    — Pepperstone server name

  API:
    BTC2_API_KEY   — API key for /health /trades /performance endpoints
    BTC2_API_PORT  — API port (default 8002)
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

# ── .env loader (before any btc_bot_2 imports) ────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(os.environ.get("ENV_FILE",
                     Path(__file__).resolve().parents[2] / ".env"))
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

# ── Logging setup (UTC timestamps — mirrors Bot 1) ────────────────────────────
from btc_research.btc_bot_2.settings import DATA_DIR, LOG_DIR, API_PORT

LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = LOG_DIR / "btc_bot_2.log"

_log_fmt     = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_log_datefmt = "%Y-%m-%dT%H:%M:%S"

_file_handler = logging.handlers.RotatingFileHandler(
    filename    = _LOG_FILE,
    maxBytes    = 10 * 1024 * 1024,   # 10 MB
    backupCount = 7,
    encoding    = "utf-8",
)
_fmt_utc = logging.Formatter(_log_fmt, datefmt=_log_datefmt)
_fmt_utc.converter = _time.gmtime   # force UTC timestamps in log file
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
logger.info("BTC Bot 2 log file: %s", _LOG_FILE)

# ── BTC Bot 2 imports ─────────────────────────────────────────────────────────
from btc_research.btc_bot_2.settings      import STARTING_BALANCE, DB_PATH, KZ_HOURS
from btc_research.btc_bot_2.signal_engine import SignalEngine
from btc_research.btc_bot_2.paper_trader  import BTC2PaperTrader
from btc_research.btc_bot_2.scheduler     import BTC2Scheduler
from btc_research.btc_bot_2.telegram      import BTC2Alerter

# ── Shared infrastructure from v2 ────────────────────────────────────────────
from v2.connectors.unified_data import DataFeed
from v2.journal.sqlite_journal  import Journal

# ── Shutdown event ────────────────────────────────────────────────────────────
_shutdown_event = threading.Event()


def _send_shutdown_alert() -> None:
    """
    Send Telegram alert on exit.
    Registered with atexit — fires even on crash or Windows Stop-Process.
    """
    try:
        from datetime import datetime, timezone
        alerter = BTC2Alerter()
        alerter.send_shutdown("Process terminated")
    except Exception:
        pass


def _handle_signal(signum: int, frame: Any) -> None:
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — initiating graceful shutdown...", sig_name)
    _shutdown_event.set()


def _start_api_server(port: int) -> None:
    """Launch FastAPI in a background daemon thread."""
    try:
        import uvicorn
        from btc_research.btc_bot_2.api import app as api_app, set_app_state
        config = uvicorn.Config(
            app       = api_app,
            host      = "0.0.0.0",
            port      = port,
            log_level = "warning",   # suppress uvicorn INFO spam
        )
        server = uvicorn.Server(config)
        t = threading.Thread(target=server.run, name="btc2-api", daemon=True)
        t.start()
        logger.info("FastAPI started on http://0.0.0.0:%d", port)
    except ImportError:
        logger.warning("uvicorn / fastapi not installed — API disabled (run: pip install fastapi uvicorn)")
    except Exception as exc:
        logger.error("Failed to start API: %s", exc)


def main() -> None:
    """Wire all Bot 2 subsystems and keep the process running."""

    # atexit fires on any exit path, including Windows Stop-Process (without -Force)
    atexit.register(_send_shutdown_alert)

    # Determine Telegram token source for startup log
    _tg_token_src = (
        "BTC2_TELEGRAM_BOT_TOKEN" if os.environ.get("BTC2_TELEGRAM_BOT_TOKEN")
        else ("TELEGRAM_BOT_TOKEN" if os.environ.get("TELEGRAM_BOT_TOKEN") else "(none)")
    )
    _tg_chat_src = (
        "BTC2_TELEGRAM_CHAT_ID" if os.environ.get("BTC2_TELEGRAM_CHAT_ID")
        else ("TELEGRAM_CHAT_ID" if os.environ.get("TELEGRAM_CHAT_ID") else "(none)")
    )
    kz_str = ", ".join(f"{h:02d}:00" for h in KZ_HOURS)

    logger.info("=" * 60)
    logger.info("BTC Bot 2 starting up")
    logger.info("Kill-zone : %s UTC  (Asia Night + EU Open)", kz_str)
    logger.info("Strategy  : SwingLevelBreak v2 [both 2xATR] + VB fallback")
    logger.info("Risk      : 3%% ADX<=25 | 2%% ADX 25-40 | 3%% ADX>=40")
    logger.info("Balance   : $%.0f starting", STARTING_BALANCE)
    logger.info("API port  : %d", API_PORT)
    logger.info("DB        : %s", DB_PATH)
    logger.info("Telegram  : token=%s  chat=%s", _tg_token_src, _tg_chat_src)
    logger.info("=" * 60)

    # ── 1. DataFeed (MT5 — Pepperstone BTCUSD, same as Bot 1) ────────────────
    logger.info("Connecting MT5 DataFeed...")
    feed = DataFeed()
    try:
        conn = feed.connect(
            mt5_login    = int(os.environ.get("MT5_ACCOUNT",
                               os.environ.get("MT5_LOGIN", "0")) or "0"),
            mt5_password = os.environ.get("MT5_PASSWORD", ""),
            mt5_server   = os.environ.get("MT5_SERVER", ""),
        )
        logger.info("DataFeed connected: MT5=%s", conn.get("mt5"))
        if not conn.get("mt5"):
            logger.warning(
                "MT5 not connected — check MT5_ACCOUNT / MT5_PASSWORD / MT5_SERVER in .env"
            )
    except Exception as exc:
        logger.warning("DataFeed connect error (will retry on first scan): %s", exc)

    # ── 2. Journal (btc2_trades.db — separate from Bot 1) ───────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Opening journal: %s", DB_PATH)
    journal = Journal(db_path=DB_PATH)

    # ── 3. Core components ────────────────────────────────────────────────────
    alerter      = BTC2Alerter()
    engine       = SignalEngine(feed=feed, journal=journal)
    paper_trader = BTC2PaperTrader(journal=journal, feed=feed)
    logger.info("SignalEngine, BTC2PaperTrader, BTC2Alerter constructed")

    # ── 4. API (optional — gracefully disabled if uvicorn not installed) ──────
    try:
        from btc_research.btc_bot_2.api import set_app_state
        set_app_state(journal=journal, engine=engine, paper_trader=paper_trader)
    except Exception:
        pass
    _start_api_server(API_PORT)

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
        logger.error("BTC2Scheduler failed to start: %s", exc, exc_info=True)

    # ── 6. Signal handlers ────────────────────────────────────────────────────
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("BTC Bot 2 running — press Ctrl+C or send SIGTERM to stop")

    # ── 7. Telegram startup alert ─────────────────────────────────────────────
    try:
        # Get live balance from journal
        balance = STARTING_BALANCE
        try:
            all_trades = journal.get_trades(status="CLOSED")
            total_pnl  = sum(float(t.get("pnl_usd") or 0) for t in all_trades)
            if total_pnl != 0:
                balance = STARTING_BALANCE + total_pnl
        except Exception:
            pass
        alerter.send_startup(balance)
    except Exception:
        pass

    # ── 8. Block until shutdown ───────────────────────────────────────────────
    _shutdown_event.wait()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    # Unregister atexit so we send the richer SIGINT/SIGTERM message below
    try:
        atexit.unregister(_send_shutdown_alert)
    except Exception:
        pass

    logger.info("BTC Bot 2 shutting down...")

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
        alerter.send_shutdown("SIGINT / SIGTERM (manual stop or system shutdown)")
    except Exception:
        pass

    logger.info("BTC Bot 2 shutdown complete")


if __name__ == "__main__":
    main()
