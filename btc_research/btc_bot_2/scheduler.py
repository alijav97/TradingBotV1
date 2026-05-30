"""
btc_research/btc_bot_2/scheduler.py — APScheduler for BTC Bot 2.

Kill-zone: non-contiguous hours [1, 2, 3, 8] UTC
  01:00 UTC = 05:00 UAE (Asia Night)
  02:00 UTC = 06:00 UAE (Asia Night)
  03:00 UTC = 07:00 UAE (Asia Night)
  08:00 UTC = 12:00 UAE (EU Open)

== SCAN SCHEDULE (mirrors Bot 1 v2 approach) ==

  1. Kill-zone scan   — every 2 seconds INSIDE kill-zone hours [1,2,3,8]
     Catches intrabar breakouts — VB / Swing conditions can be met partway
     through the bar, not just at close. Skips instantly outside KZ.

  2. Background scan  — every 5 minutes OUTSIDE kill-zone hours
     Keeps the data window fresh so strategies have valid recent bars
     the moment the kill-zone opens. Skips when KZ is active.

  3. Post-KZ watch    — every 2 seconds AFTER kill-zone closes IF a trade is open
     Once a trade is entered during KZ, it may run for hours/days.
     2-second monitoring ensures trailing SL and TP2 are caught promptly
     even after the session window closes.
     Skips immediately if no open trades OR if still inside KZ.

  4. Trade monitor    — every 60 seconds (always-on safety net)
     Catches any events the faster jobs might miss during heavy load.

  5. Morning briefing — 02:00 UTC daily (06:00 UAE)

== USAGE ==
  from btc_research.btc_bot_2.scheduler import BTC2Scheduler
  sched = BTC2Scheduler(engine, paper_trader, journal, alerter)
  sched.start()
  ...
  sched.stop()
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
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

    Three-tier scan approach:
      - 2s  inside kill-zone       : catch intrabar breakouts fast
      - 5m  outside kill-zone      : keep data fresh, low overhead
      - 2s  post-KZ + trade open   : fast trailing SL / TP monitoring
      - 60s always-on monitor      : safety net
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
                executors={"default": APSPool(6)},
                timezone="UTC",
            ) if _APS_OK else None
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        if self._scheduler is None:
            logger.error("Cannot start — APScheduler not available")
            return

        # Suppress APScheduler's per-execution INFO spam (2s jobs fire ~1800×/hr)
        import logging as _log
        _log.getLogger("apscheduler").setLevel(_log.WARNING)

        # ── Job 1: Kill-zone fast scan — every 2 seconds ──────────────────────
        # Only active INSIDE kill-zone hours. Exits immediately otherwise.
        # max_instances=1 prevents overlapping scans if a scan takes > 2s.
        self._scheduler.add_job(
            func                = self._job_scan_killzone,
            trigger             = IntervalTrigger(seconds=2),
            id                  = "scan_kz_2s",
            name                = "BTC2 kill-zone scan (2s)",
            max_instances       = 1,
            misfire_grace_time  = 5,
        )

        # ── Job 2: Background scan — every 5 minutes ─────────────────────────
        # Only active OUTSIDE kill-zone hours. Keeps data window fresh.
        self._scheduler.add_job(
            func                = self._job_scan_background,
            trigger             = IntervalTrigger(minutes=5),
            id                  = "scan_bg_5m",
            name                = "BTC2 background scan (5m)",
            max_instances       = 1,
            misfire_grace_time  = 60,
        )

        # ── Job 3: Post-KZ open trade watch — every 2 seconds ────────────────
        # Only runs when OUTSIDE kill-zone AND at least one trade is open.
        # Ensures trailing SL and TP2 are checked in near-real-time after KZ.
        self._scheduler.add_job(
            func                = self._job_post_kz_watch,
            trigger             = IntervalTrigger(seconds=2),
            id                  = "post_kz_2s",
            name                = "BTC2 post-KZ trade watch (2s)",
            max_instances       = 1,
            misfire_grace_time  = 5,
        )

        # ── Job 4: Always-on trade monitor — every 60 seconds ────────────────
        # Safety net that catches anything the faster jobs might miss.
        self._scheduler.add_job(
            func                = self._job_monitor,
            trigger             = IntervalTrigger(seconds=60),
            id                  = "trade_monitor_60s",
            name                = "BTC2 trade monitor (60s)",
            max_instances       = 1,
            misfire_grace_time  = 30,
        )

        # ── Job 5: Morning briefing — 02:00 UTC (06:00 UAE) ──────────────────
        self._scheduler.add_job(
            func                = self._job_morning_briefing,
            trigger             = CronTrigger(hour=2, minute=0, timezone="UTC"),
            id                  = "morning_briefing",
            name                = "BTC2 morning briefing",
        )

        self._scheduler.start()

        kz_str = ",".join(f"{h:02d}:00" for h in KZ_HOURS)
        logger.info(
            "BTC2Scheduler started — %d jobs  |  KZ: [%s] UTC  |  "
            "2s inside KZ  |  5m outside KZ  |  2s post-KZ if trade open",
            len(self._scheduler.get_jobs()),
            kz_str,
        )

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("BTC2Scheduler stopped")

    # ── Job implementations ───────────────────────────────────────────────────

    def _job_scan_killzone(self) -> None:
        """
        Fires every 2 seconds.
        Exits immediately if NOT inside a kill-zone hour.
        Runs a signal scan and opens a trade if one fires.
        """
        now = datetime.now(timezone.utc)
        if now.hour not in KZ_HOURS:
            return   # not our session — exit immediately, no API call

        # Skip if trade already open
        if self._pt.get_open_summary()["count"] > 0:
            return

        self._run_scan(now)

    def _job_scan_background(self) -> None:
        """
        Fires every 5 minutes.
        Exits immediately if inside a kill-zone hour (KZ job handles that).
        Keeps the OHLCV data window fresh so the strategy always has
        recent bars ready the moment the kill-zone opens.
        """
        now = datetime.now(timezone.utc)
        if now.hour in KZ_HOURS:
            return   # kill-zone active — let the 2s job handle it

        # Only scan if no trade open (informational scan, not entry scan)
        # Don't open trades outside kill-zone hours
        logger.debug("Background scan hr=%02d UTC (data refresh only)", now.hour)

        # We still call the engine so the data feed cache stays warm,
        # but we intentionally don't open a trade outside KZ.
        # The engine itself will return None outside KZ hours.
        try:
            self._engine.scan(now=now)   # returns None outside KZ — that's fine
        except Exception as exc:
            logger.debug("Background scan error: %s", exc)

    def _job_post_kz_watch(self) -> None:
        """
        Fires every 2 seconds.
        Only acts when:
          - OUTSIDE kill-zone hours (KZ job already covers inside)
          - At least one trade is open (otherwise no point checking)

        This gives near-real-time trailing SL and TP2 monitoring
        for trades that opened during the session and are still running.
        """
        now = datetime.now(timezone.utc)

        # KZ active → the KZ scan job is already running at 2s
        if now.hour in KZ_HOURS:
            return

        # No open trades → nothing to do
        summary = self._pt.get_open_summary()
        if summary["count"] == 0:
            return

        # Trade is open outside KZ → check at full 2s speed
        try:
            actions = self._pt.check_all_open_trades()
            for action in actions:
                self._send_action_alert(action)
        except Exception as exc:
            logger.error("Post-KZ watch error: %s", exc)

    def _job_monitor(self) -> None:
        """
        Fires every 60 seconds (always-on safety net).
        Covers any events the 2s/5m jobs might miss during heavy load.
        """
        try:
            actions = self._pt.check_all_open_trades()
            for action in actions:
                self._send_action_alert(action)
        except Exception as exc:
            logger.error("Trade monitor error: %s", exc)

    def _job_morning_briefing(self) -> None:
        """Send daily morning briefing at 02:00 UTC (06:00 UAE)."""
        try:
            stats = self._journal.get_stats(days=7)
            logger.info(
                "BTC2 morning briefing: 7d WR=%.1f%% PnL=$%.2f trades=%d",
                stats.get("win_rate", 0), stats.get("total_pnl", 0), stats.get("trades", 0),
            )
            self._alerter.send_morning_briefing(stats)
        except Exception as exc:
            logger.error("Morning briefing error: %s", exc)

    # ── Core scan logic ───────────────────────────────────────────────────────

    def _run_scan(self, now: datetime) -> None:
        """
        Run one signal scan cycle. Opens a trade if a signal fires.
        Called by both the KZ 2s job and (if needed) any other scan job.
        """
        try:
            signal = self._engine.scan(now=now)
        except Exception as exc:
            logger.error("Signal engine error: %s", exc, exc_info=True)
            return

        if signal is None:
            return

        # Signal fired — open paper trade
        trade_id = self._pt.open_trade(signal)
        if trade_id:
            try:
                trade = self._journal.get_trade(trade_id)
                if trade:
                    self._alerter.send_trade_opened(trade)
                else:
                    self._alerter.send_signal_opened(signal)
            except Exception as exc:
                logger.warning("Trade alert error: %s", exc)
            logger.info(
                "Trade opened: %s  %s %s @ %.0f  strategy=%s(%s)",
                trade_id[:8],
                signal.get("symbol",""),
                signal.get("direction","").upper(),
                signal.get("entry_price", 0),
                signal.get("strategy",""),
                signal.get("entry_type",""),
            )
        else:
            logger.debug("Signal fired but trade was blocked (one-trade rule)")

    # ── Alert dispatcher ──────────────────────────────────────────────────────

    def _send_action_alert(self, action: dict) -> None:
        """Send Telegram alert for trade events (TP1, TP2, SL, MAX_HOLD)."""
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

        if action_type in ("TP2", "SL", "SL_AFTER_TP1", "TRAIL_SL", "MAX_HOLD", "MANUAL"):
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
