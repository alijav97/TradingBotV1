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
from datetime import datetime, timezone, timedelta
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
        self._kz_scan_count  = 0
        self._bg_scan_count  = 0
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
        """
        5-min background job outside kill-zone.

        Does two things:
          1. Fetches fresh OHLCV data for BTC, Gold, NAS100 so that EMA200,
             ADX, ATR and the pre-KZ range are all up to date before 21:00 UTC.
          2. Logs a readable snapshot so the operator can watch conditions
             build throughout the day (EMA200 direction, ADX trend, pre-KZ range).
        """
        _now = datetime.now(timezone.utc)
        if KZ_START_UTC <= _now.hour < KZ_END_UTC:
            return   # kill-zone job covers this window

        self._bg_scan_count += 1
        logger.info("BG scan #%d | UTC %02d:%02d", self._bg_scan_count, _now.hour, _now.minute)

        hours_to_kz = (KZ_START_UTC - _now.hour) % 24
        snap = self._engine.get_market_snapshot()

        if snap:
            ema_flag = "ABOVE EMA200 (longs valid)" if snap["above_ema"] else "BELOW EMA200 (shorts valid)"
            adx_str  = (
                f"{snap['adx']} — trend OK ✅"  if snap["adx"] >= 20
                else f"{snap['adx']} — weak trend ⚠️"
            )
            if snap.get("mr_high"):
                mr_str = (
                    f"Pre-KZ range: ${snap['mr_high']:,.0f}-${snap['mr_low']:,.0f}  "
                    f"(${snap['mr_high'] - snap['mr_low']:,.0f} range, {snap['mr_bars']} bars)"
                )
            else:
                mr_str = "Pre-KZ range: not yet (17:00 UTC onwards)"

            gold_str = f"  Gold: ${snap['gold']:,.2f}" if snap.get("gold") else ""
            nas_str  = f"  NAS: ${snap['nas']:,.0f}"  if snap.get("nas")  else ""

            price_str = f"${snap['price']:,.2f}"
            atr_str   = f"${snap['atr']:,.0f}"
            logger.info(
                "BTC snapshot UTC %02d:%02d | %s | %s | ADX %s | ATR %s | %s |%s%s | KZ in ~%dh",
                _now.hour, _now.minute,
                price_str, ema_flag, adx_str, atr_str,
                mr_str, gold_str, nas_str, hours_to_kz,
            )
        else:
            logger.warning(
                "BTC background snapshot failed — no data (UTC %02d:%02d, KZ in ~%dh)",
                _now.hour, _now.minute, hours_to_kz,
            )

    def _job_scan_killzone(self) -> None:
        """2-second scan — ONLY inside kill-zone (21-24 UTC)."""
        _now = datetime.now(timezone.utc)
        in_kz = KZ_START_UTC <= _now.hour < 24   # 21,22,23 only
        if not in_kz:
            return

        self._kz_scan_count += 1
        logger.info("KZ scan #%d | UTC %02d:%02d:%02d",
                    self._kz_scan_count, _now.hour, _now.minute, _now.second)

        self._run_scan()

    def _job_post_killzone_watch(self) -> None:
        """5-second trade monitor after kill-zone closes, while BTC trade is open."""
        _now = datetime.now(timezone.utc)
        in_kz = KZ_START_UTC <= _now.hour < 24   # 21,22,23 only
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
                blocked = signal.get("blocked_by", "no signal")
                logger.info("  %s → SKIP: %s", direction.upper(), blocked)
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
