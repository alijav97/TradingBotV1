"""
btc_research/btc_bot_2/scheduler.py — APScheduler setup for BTC Bot 2.

Kill-zone: non-contiguous hours [1, 2, 3, 8] UTC
  - 01:00 UTC → Asia Night (05:00 UAE)
  - 02:00 UTC → Asia Night (06:00 UAE)
  - 03:00 UTC → Asia Night (07:00 UAE)
  - 08:00 UTC → EU Session Open (12:00 UAE)

Registered jobs:
  - CronTrigger at :02 past each KZ hour  → signal scan
  - IntervalTrigger every 60s             → trade monitor (SL/TP/trailing)
  - Post-KZ watch every 30s              → if trade open outside KZ, monitor faster
  - Daily 02:00 UTC (06:00 UAE)          → morning briefing

The scan fires at :02 past the hour (not :00) to ensure the H1 bar has
fully closed and the data feed has the complete candle available.

Usage:
    from btc_research.btc_bot_2.scheduler import BTC2Scheduler
    sched = BTC2Scheduler(engine, paper_trader, journal, alerter)
    sched.start()
    ...
    sched.stop()
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from btc_research.btc_bot_2.signal_engine import SignalEngine
    from btc_research.btc_bot_2.paper_trader  import BTC2PaperTrader
    from v2.journal.sqlite_journal            import Journal
    from btc_research.btc_bot_2.telegram      import BTC2Alerter

from btc_research.btc_bot_2.settings import KZ_HOURS

logger = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron         import CronTrigger
    from apscheduler.triggers.interval     import IntervalTrigger
    from apscheduler.executors.pool        import ThreadPoolExecutor as APSPool
    _APS_OK = True
except ImportError:
    _APS_OK = False
    logger.error("APScheduler not installed — BTC2 scheduler disabled")


class BTC2Scheduler:
    """
    Manages all recurring jobs for BTC Bot 2.

    Jobs:
      1. KZ signal scan   — fires at :02 past each of [1, 2, 3, 8] UTC
      2. Trade monitor    — every 60 seconds (SL/TP/trailing SL check)
      3. Post-KZ watch    — every 30 seconds IF a trade is open outside KZ
      4. Morning briefing — 02:00 UTC daily (06:00 UAE)
    """

    def __init__(
        self,
        engine:       "SignalEngine",
        paper_trader: "BTC2PaperTrader",
        journal:      "Journal",
        alerter:      "BTC2Alerter",
    ) -> None:
        self._engine  = engine
        self._pt      = paper_trader
        self._journal = journal
        self._alerter = alerter

        self._scheduler = (
            BackgroundScheduler(
                executors={"default": APSPool(4)},
                timezone="UTC",
            ) if _APS_OK else None
        )

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        if self._scheduler is None:
            logger.error("Cannot start — APScheduler not available")
            return

        # Suppress noisy per-execution APScheduler INFO logs
        import logging as _log
        _log.getLogger("apscheduler").setLevel(_log.WARNING)

        # ── Job 1: KZ signal scan at :02 past each KZ hour ────────────────────
        # CronTrigger with comma-separated hours fires at each one independently
        kz_hours_str = ",".join(str(h) for h in KZ_HOURS)   # "1,2,3,8"
        self._scheduler.add_job(
            func     = self._job_kz_scan,
            trigger  = CronTrigger(hour=kz_hours_str, minute=2),
            id       = "kz_scan",
            name     = f"BTC2 kill-zone scan [{kz_hours_str}]:02 UTC",
            max_instances       = 1,
            misfire_grace_time  = 120,
        )

        # ── Job 2: Trade monitor every 60 seconds ─────────────────────────────
        self._scheduler.add_job(
            func     = self._job_monitor,
            trigger  = IntervalTrigger(seconds=60),
            id       = "trade_monitor",
            name     = "BTC2 trade monitor (60s)",
            max_instances       = 1,
            misfire_grace_time  = 30,
        )

        # ── Job 3: Post-KZ faster watch when trade is open ────────────────────
        self._scheduler.add_job(
            func     = self._job_post_kz_watch,
            trigger  = IntervalTrigger(seconds=30),
            id       = "post_kz_watch",
            name     = "BTC2 post-KZ open trade watch (30s)",
            max_instances       = 1,
            misfire_grace_time  = 15,
        )

        # ── Job 4: Morning briefing at 02:00 UTC (06:00 UAE) ──────────────────
        self._scheduler.add_job(
            func     = self._job_morning_briefing,
            trigger  = CronTrigger(hour=2, minute=0, timezone="UTC"),
            id       = "morning_briefing",
            name     = "BTC2 morning briefing",
        )

        self._scheduler.start()
        logger.info(
            "BTC2Scheduler started — %d jobs  KZ hours UTC: %s",
            len(self._scheduler.get_jobs()),
            kz_hours_str,
        )

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("BTC2Scheduler stopped")

    # ── Job implementations ───────────────────────────────────────────────────

    def _job_kz_scan(self) -> None:
        """
        Fire at :02 past each KZ hour.
        Scans BTCUSD for a signal and opens a paper trade if one fires.
        """
        now = datetime.now(timezone.utc)
        logger.info("KZ scan starting  hr=%02d:%02d UTC", now.hour, now.minute)

        # Guard: skip if a trade is already open
        summary = self._pt.get_open_summary()
        if summary["count"] > 0:
            logger.info("KZ scan skipped — %d trade(s) already open", summary["count"])
            return

        try:
            signal = self._engine.scan(now=now)
        except Exception as exc:
            logger.error("Signal engine error: %s", exc, exc_info=True)
            return

        if signal is None:
            logger.debug("KZ scan: no signal")
            return

        # Open paper trade
        trade_id = self._pt.open_trade(signal)
        if trade_id:
            # Send Telegram alert
            try:
                trade = self._journal.get_trade(trade_id)
                if trade:
                    self._alerter.send_trade_opened(trade)
                else:
                    self._alerter.send_signal_opened(signal)
            except Exception as exc:
                logger.warning("Alert error: %s", exc)
            logger.info("Trade opened: %s", trade_id[:8])
        else:
            logger.info("Signal fired but trade blocked (risk check failed)")

    def _job_monitor(self) -> None:
        """Monitor all open trades every 60s. Sends alerts on any event."""
        try:
            actions = self._pt.check_all_open_trades()
        except Exception as exc:
            logger.error("Monitor job error: %s", exc)
            return

        for action in actions:
            self._send_action_alert(action)

    def _job_post_kz_watch(self) -> None:
        """
        Runs every 30s but only acts when:
          - We are OUTSIDE the kill-zone hours
          - At least one trade is still open (waiting for SL/TP)
        This provides faster monitoring when a trade is running post-session.
        """
        now = datetime.now(timezone.utc)

        # Skip during KZ hours — the main monitor already runs every 60s there
        if now.hour in KZ_HOURS:
            return

        summary = self._pt.get_open_summary()
        if summary["count"] == 0:
            return   # no open trades — nothing to do

        logger.debug("Post-KZ watch: %d open trade(s) — monitoring", summary["count"])
        try:
            actions = self._pt.check_all_open_trades()
            for action in actions:
                self._send_action_alert(action)
        except Exception as exc:
            logger.error("Post-KZ watch error: %s", exc)

    def _job_morning_briefing(self) -> None:
        """Log morning briefing stats and send Telegram."""
        try:
            stats = self._journal.get_stats(days=7)
            logger.info(
                "BTC2 morning briefing: 7d WR=%.1f%% PnL=$%.2f trades=%d",
                stats.get("win_rate", 0),
                stats.get("total_pnl", 0),
                stats.get("trades", 0),
            )
            self._alerter.send_morning_briefing(stats)
        except Exception as exc:
            logger.error("Morning briefing error: %s", exc)

    # ── Alert dispatcher ──────────────────────────────────────────────────────

    def _send_action_alert(self, action: dict) -> None:
        """Send Telegram alert for a trade event (TP1, TP2, SL, MAX_HOLD)."""
        action_type = action.get("action", "")
        trade_id    = action.get("trade_id", "")

        try:
            trade = self._journal.get_trade(trade_id)
        except Exception:
            trade = None

        if action_type == "TP1":
            if trade:
                self._alerter.send_tp1_hit(trade, action.get("price", 0))
            return

        if action_type in ("TP2", "SL", "SL_AFTER_TP1", "MAX_HOLD", "MANUAL"):
            if trade:
                self._alerter.send_trade_closed(trade)
            else:
                pnl  = action.get("pnl_usd", 0)
                sign = "+" if pnl >= 0 else ""
                self._alerter.send_text(
                    f"[BTC BOT 2] TRADE CLOSED\n"
                    f"ID: {trade_id[:8]}\n"
                    f"Reason: {action_type}\n"
                    f"PnL: {sign}${pnl:.2f}"
                )
