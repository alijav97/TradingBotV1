"""
btc_bot_1/scheduler/scheduler.py — APScheduler setup for BTC Bot 1.

Same 3-job pattern as v2/scheduler/scheduler.py but with BTC kill-zone:

  Job 1 — Background (5 min, outside 21-24 UTC):
    Tracks the 17-21 UTC consolidation range that the strategy uses as
    the morning range. Keeps data fresh so the signal engine has a valid
    range when the kill-zone opens at 21:00 UTC.

  Job 2 — Kill-zone (2 seconds, inside 21-24 UTC):
    Catches intrabar breakouts every 2 seconds during the active window.

  Job 3 — Post-kill-zone trade watch (5 seconds):
    After 24:00 UTC (00:00), if a BTC trade is still open,
    monitors it every 5 seconds until it closes.

  Job 4 — Morning briefing (daily at 20:00 UTC = before kill-zone):
    Sends stats and reminds the user the kill-zone opens soon.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from btc_research.btc_bot_1.trading.paper_trader import PaperTrader
    from btc_research.btc_bot_1.journal.sqlite_journal import Journal
    from btc_research.btc_bot_1.connectors.unified_data import DataFeed
    from btc_research.btc_bot_1.signals.btc_engine import BTCSignalEngine

from btc_research.btc_bot_1.settings import KZ_START_UTC, KZ_END_UTC
from btc_research.btc_bot_1.api.telegram_bot import TelegramAlerter

logger = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron     import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.executors.pool    import ThreadPoolExecutor as APSThreadPool
    _APS_OK = True
except ImportError:
    _APS_OK = False
    logger.error("APScheduler not installed — BTC scheduler disabled")


class BTCScheduler:
    """Manages all recurring jobs for BTC Bot 1."""

    def __init__(
        self,
        paper_trader: "PaperTrader",
        signal_engine: "BTCSignalEngine",
        journal:       "Journal",
        feed:          "DataFeed",
    ) -> None:
        self._pt      = paper_trader
        self._engine  = signal_engine
        self._journal = journal
        self._feed    = feed
        self._alerter = TelegramAlerter()
        self._scheduler = (
            BackgroundScheduler(
                executors={"default": APSThreadPool(5)},
                timezone="UTC",
            ) if _APS_OK else None
        )

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        if self._scheduler is None:
            logger.error("Cannot start — APScheduler not available")
            return

        import logging as _logging
        _logging.getLogger("apscheduler").setLevel(_logging.WARNING)

        # Job 1: Background scan every 5 min (outside kill-zone)
        self._scheduler.add_job(
            func     = self._job_scan_background,
            trigger  = IntervalTrigger(minutes=5),
            id       = "btc_scan_background",
            name     = "BTC H1 background scan (5min)",
            max_instances      = 1,
            misfire_grace_time = 60,
        )

        # Job 2: Kill-zone scan every 2 seconds (inside 21-24 UTC)
        self._scheduler.add_job(
            func     = self._job_scan_killzone,
            trigger  = IntervalTrigger(seconds=2),
            id       = "btc_scan_killzone",
            name     = "BTC H1 kill-zone scan (2s)",
            max_instances      = 1,
            misfire_grace_time = 5,
        )

        # Job 3: Post-kill-zone trade watch every 5 seconds
        self._scheduler.add_job(
            func     = self._job_post_killzone_watch,
            trigger  = IntervalTrigger(seconds=5),
            id       = "btc_post_kz_watch",
            name     = "BTC post-kill-zone trade monitor (5s)",
            max_instances      = 1,
            misfire_grace_time = 10,
        )

        # Job 4: Pre-kill-zone briefing at 20:00 UTC (1 hour before KZ opens)
        self._scheduler.add_job(
            func    = self._job_morning_briefing,
            trigger = CronTrigger(hour=20, minute=0, timezone="UTC"),
            id      = "btc_daily_briefing",
            name    = "BTC daily briefing (20:00 UTC)",
        )

        self._scheduler.start()
        logger.info(
            "BTCScheduler started — %d jobs | kill-zone %d-%d UTC",
            len(self._scheduler.get_jobs()), KZ_START_UTC, KZ_END_UTC,
        )

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("BTCScheduler stopped")

    # ── Job implementations ───────────────────────────────────────────────────

    def _job_scan_background(self) -> None:
        """5-min scan outside kill-zone — keeps pre-KZ range data fresh."""
        _now = datetime.now(timezone.utc)
        if KZ_START_UTC <= _now.hour < KZ_END_UTC:
            return   # kill-zone job covers this window
        self._run_scan()

    def _job_scan_killzone(self) -> None:
        """2-second scan — ONLY inside kill-zone (21-24 UTC)."""
        _now = datetime.now(timezone.utc)
        # Handle midnight wrap: KZ_END_UTC = 24 means hour 0 is still in range
        in_kz = (KZ_START_UTC <= _now.hour < 24) or (_now.hour == 0 and KZ_END_UTC == 24)
        if not in_kz:
            return
        self._run_scan()

    def _job_post_killzone_watch(self) -> None:
        """5-second trade monitor after kill-zone closes, while BTC trade is open."""
        _now = datetime.now(timezone.utc)
        in_kz = (KZ_START_UTC <= _now.hour < 24) or (_now.hour == 0 and KZ_END_UTC == 24)
        if in_kz:
            return   # kill-zone job handles this

        try:
            open_count = self._pt.get_open_summary().get("count", 0)
        except Exception:
            return
        if open_count == 0:
            return

        logger.debug("BTC post-KZ watch: %d open trade(s)", open_count)
        self._monitor_trades()

    def _run_scan(self) -> None:
        """Run the signal engine for both directions. Open a trade if signal fires."""
        for direction in ("long", "short"):
            try:
                signal = self._engine.scan(direction)
            except Exception as exc:
                logger.error("BTC scan error (%s): %s", direction, exc)
                continue

            if not signal.get("signal"):
                continue

            logger.info(
                "BTC SIGNAL %s @ %.2f  SL=%.2f  score=%.2f",
                direction.upper(),
                signal.get("entry_price", 0),
                signal.get("stop_loss", 0),
                signal.get("score", 0),
            )

            # Log every signal to journal regardless of whether trade opens
            self._journal.log_signal(signal, taken=False)

            trade_id = self._pt.open_trade(signal)
            if trade_id:
                self._journal.log_signal(signal, taken=True)
                self._send_trade_alert(signal, trade_id)
                break   # one trade per scan — don't check the other direction

    def _monitor_trades(self) -> None:
        """Check all open BTC trades for SL/TP/max-hold."""
        try:
            actions = self._pt.check_all_open_trades()
            for action in actions:
                self._send_action_alert(action)
        except Exception as exc:
            logger.error("BTC monitor job error: %s", exc)

    def _job_morning_briefing(self) -> None:
        """Daily stats + kill-zone reminder."""
        try:
            stats = self._journal.get_stats(days=30)
            logger.info(
                "BTC daily briefing: %d trades | WR=%.1f%% | PnL=$%.2f",
                stats.get("trades", 0), stats.get("win_rate", 0), stats.get("total_pnl", 0),
            )
            self._alerter.send_morning_briefing(stats)
        except Exception as exc:
            logger.error("BTC briefing error: %s", exc)

    # ── Alert helpers ─────────────────────────────────────────────────────────

    def _send_trade_alert(self, signal: dict, trade_id: str) -> None:
        trade = self._journal.get_trade(trade_id)
        if trade:
            self._alerter.send_trade_opened(trade)
        else:
            self._alerter.send_text(
                f"[BTC] TRADE OPENED\n"
                f"{signal.get('direction','').upper()} @ ${signal.get('entry_price',0):,.2f}\n"
                f"Score: {signal.get('score',0):.1f}"
            )

    def _send_action_alert(self, action: dict) -> None:
        trade_id    = action.get("trade_id", "")
        action_type = action.get("action", "")
        trade       = self._journal.get_trade(trade_id)

        if action_type == "TP1":
            if trade:
                self._alerter.send_tp1_hit(trade, action.get("price", 0))
        elif action_type in ("SL", "SL_AFTER_TP1", "TP2", "MAX_HOLD"):
            if trade:
                self._alerter.send_trade_closed(trade)
        else:
            self._alerter.send_text(
                f"[BTC] TRADE ACTION\n"
                f"ID: {trade_id[:8]}\n"
                f"Action: {action_type}\n"
                f"Price: ${action.get('price', 0):,.2f}"
            )
