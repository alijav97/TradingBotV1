"""
btc_research/eth_bot/scheduler.py — APScheduler for ETH Bot.

Mirrors btc_bot_2/scheduler.py exactly with ETH naming.

Kill-zone hours: configurable via KZ_HOURS in settings.py
  Default placeholder: [1, 2, 3, 8] UTC (same as BTC Bot 2)
  Update after running ETH backtest to confirm optimal hours.

== SCAN SCHEDULE ==
  1. Kill-zone scan    — every 2 seconds INSIDE kill-zone hours
  2. Background scan   — every 5 minutes OUTSIDE kill-zone hours (data refresh)
  3. Post-KZ watch     — every 2 seconds AFTER kill-zone IF a trade is open
  4. Trade monitor     — every 60 seconds (always-on safety net)
  5. Morning briefing  — 02:00 UTC daily
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from btc_research.eth_bot.signal_engine import ETHSignalEngine
    from btc_research.eth_bot.paper_trader  import ETHPaperTrader
    from btc_research.eth_bot.journal.sqlite_journal import Journal
    from btc_research.eth_bot.telegram      import ETHAlerter

from btc_research.eth_bot.settings import KZ_HOURS

logger = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron         import CronTrigger
    from apscheduler.triggers.interval     import IntervalTrigger
    from apscheduler.executors.pool        import ThreadPoolExecutor as APSPool
    _APS_OK = True
except ImportError:
    _APS_OK = False
    logger.error("APScheduler not installed — ETH scheduler disabled")


class ETHScheduler:
    """Manages all recurring jobs for ETH Bot."""

    def __init__(
        self,
        engine:       "ETHSignalEngine",
        paper_trader: "ETHPaperTrader",
        journal:      "Journal",
        alerter:      "ETHAlerter",
    ) -> None:
        self._engine  = engine
        self._pt      = paper_trader
        self._journal = journal
        self._alerter = alerter
        self._kz_scan_count = 0
        self._bg_scan_count = 0

        self._scheduler = (
            BackgroundScheduler(
                executors={"default": APSPool(6)},
                timezone="UTC",
            ) if _APS_OK else None
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._scheduler is None:
            logger.error("Cannot start — APScheduler not available")
            return

        import logging as _log
        _log.getLogger("apscheduler").setLevel(_log.WARNING)

        # Job 1: Kill-zone fast scan — every 2 seconds
        self._scheduler.add_job(
            func               = self._job_scan_killzone,
            trigger            = IntervalTrigger(seconds=2),
            id                 = "eth_scan_kz_2s",
            name               = "ETH kill-zone scan (2s)",
            max_instances      = 1,
            misfire_grace_time = 5,
        )

        # Job 2: Background scan — every 5 minutes
        self._scheduler.add_job(
            func               = self._job_scan_background,
            trigger            = IntervalTrigger(minutes=5),
            id                 = "eth_scan_bg_5m",
            name               = "ETH background scan (5m)",
            max_instances      = 1,
            misfire_grace_time = 60,
        )

        # Job 3: Post-KZ open trade watch — every 2 seconds
        self._scheduler.add_job(
            func               = self._job_post_kz_watch,
            trigger            = IntervalTrigger(seconds=2),
            id                 = "eth_post_kz_2s",
            name               = "ETH post-KZ trade watch (2s)",
            max_instances      = 1,
            misfire_grace_time = 5,
        )

        # Job 4: Always-on trade monitor — every 60 seconds
        self._scheduler.add_job(
            func               = self._job_monitor,
            trigger            = IntervalTrigger(seconds=60),
            id                 = "eth_trade_monitor_60s",
            name               = "ETH trade monitor (60s)",
            max_instances      = 1,
            misfire_grace_time = 30,
        )

        # Job 5: Morning briefing — 02:00 UTC daily
        self._scheduler.add_job(
            func    = self._job_morning_briefing,
            trigger = CronTrigger(hour=2, minute=0, timezone="UTC"),
            id      = "eth_morning_briefing",
            name    = "ETH morning briefing",
        )

        self._scheduler.start()

        kz_str = ", ".join(f"{h:02d}:00" for h in KZ_HOURS)
        logger.info(
            "ETHScheduler started — %d jobs  |  KZ: [%s] UTC  |  "
            "2s inside KZ  |  5m outside KZ  |  2s post-KZ if trade open",
            len(self._scheduler.get_jobs()), kz_str,
        )

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("ETHScheduler stopped")

    # ── Job implementations ────────────────────────────────────────────────────

    def _job_scan_killzone(self) -> None:
        now = datetime.now(timezone.utc)
        if now.hour not in KZ_HOURS:
            return

        self._kz_scan_count += 1
        logger.info("ETH KZ scan #%d | UTC %02d:%02d:%02d",
                    self._kz_scan_count, now.hour, now.minute, now.second)

        if self._pt.get_open_summary()["count"] > 0:
            logger.info("  → trade already open — skipping signal scan")
            return

        self._run_scan(now)

    def _job_scan_background(self) -> None:
        now = datetime.now(timezone.utc)
        if now.hour in KZ_HOURS:
            return

        self._bg_scan_count += 1
        logger.info("ETH BG scan #%d | UTC %02d:%02d (data refresh — outside KZ)",
                    self._bg_scan_count, now.hour, now.minute)

        try:
            snap = self._engine.get_market_snapshot()
            if snap:
                ema_flag = "ABOVE EMA200 (longs valid)" if snap["above_ema"] else "BELOW EMA200 (shorts valid)"
                adx_str  = (
                    f"{snap['adx']} — trend OK ✓" if snap["adx"] >= 20
                    else f"{snap['adx']} — weak trend (ADX < 20)"
                )
                logger.info(
                    "ETH snapshot | $%.2f | %s | ADX %s | ATR %.2f",
                    snap["price"], ema_flag, adx_str, snap["atr"],
                )
        except Exception as exc:
            logger.info("ETH background scan error: %s", exc)

    def _job_post_kz_watch(self) -> None:
        now = datetime.now(timezone.utc)
        if now.hour in KZ_HOURS:
            return

        summary = self._pt.get_open_summary()
        if summary["count"] == 0:
            return

        try:
            actions = self._pt.check_all_open_trades()
            for action in actions:
                self._send_action_alert(action)
        except Exception as exc:
            logger.error("ETH post-KZ watch error: %s", exc)

    def _job_monitor(self) -> None:
        try:
            actions = self._pt.check_all_open_trades()
            for action in actions:
                self._send_action_alert(action)
        except Exception as exc:
            logger.error("ETH trade monitor error: %s", exc)

    def _job_morning_briefing(self) -> None:
        try:
            stats = self._journal.get_stats(days=7)
            logger.info(
                "ETH morning briefing: 7d WR=%.1f%% PnL=$%.2f trades=%d",
                stats.get("win_rate", 0), stats.get("total_pnl", 0), stats.get("trades", 0),
            )
            self._alerter.send_morning_briefing(stats)
        except Exception as exc:
            logger.error("ETH morning briefing error: %s", exc)

    # ── Core scan logic ────────────────────────────────────────────────────────

    def _run_scan(self, now: datetime) -> None:
        try:
            signal = self._engine.scan(now=now)
        except Exception as exc:
            logger.error("ETH signal engine error: %s", exc, exc_info=True)
            return

        if signal is None:
            return

        trade_id = self._pt.open_trade(signal)
        if trade_id:
            try:
                trade = self._journal.get_trade(trade_id)
                if trade:
                    self._alerter.send_trade_opened(trade)
            except Exception as exc:
                logger.warning("ETH trade alert error: %s", exc)
            logger.info(
                "ETH trade opened: %s  %s %s @ %.4f  strategy=%s(%s)",
                trade_id[:8],
                signal.get("symbol", ""),
                signal.get("direction", "").upper(),
                signal.get("entry_price", 0),
                signal.get("strategy", ""),
                signal.get("entry_type", ""),
            )
        else:
            logger.debug("ETH signal fired but trade was blocked (one-trade rule)")

    # ── Alert dispatcher ───────────────────────────────────────────────────────

    def _send_action_alert(self, action: dict) -> None:
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
                    f"[ETH BOT] TRADE CLOSED\n"
                    f"ID: {trade_id[:8]}\n"
                    f"Reason: {action_type}\n"
                    f"PnL: {sign}${pnl:.2f}"
                )
