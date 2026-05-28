"""
scheduler/scheduler.py — APScheduler setup for TradingBotV2.

Registers all recurring jobs:
  - Every 60s  : monitor open paper trades (SL/TP check)
  - Every 1H   : signal scan all 6 instruments (H1 timeframe)
  - Every 4H   : signal scan all 6 instruments (H4 timeframe)
  - Daily 06:00 GST : morning briefing + load economic calendar
  - Daily 23:00 GST : ML retrain (if enough new trades)

Usage:
    from v2.scheduler.scheduler import BotScheduler
    scheduler = BotScheduler(paper_trader, confluence_engine, journal, feed)
    scheduler.start()
    # ... runs forever, scheduler handles all jobs in background
    scheduler.stop()
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2.trading.paper_trader import PaperTrader
    from v2.signals.confluence_engine import ConfluenceEngine
    from v2.journal.sqlite_journal import Journal
    from v2.connectors.unified_data import DataFeed

from v2.instrument_config import ALL_SYMBOLS
from v2.settings import ACTIVE_SYMBOLS
from v2.signals.entry_checklist import validate_entry
from v2.risk.loss_limits import LossLimits
from v2.api.telegram_bot import TelegramAlerter

logger = logging.getLogger(__name__)

GST = timezone(timedelta(hours=4))

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.executors.pool import ThreadPoolExecutor as APSThreadPool
    _APScheduler_OK = True
except ImportError:
    _APScheduler_OK = False
    logger.error("APScheduler not installed — scheduler disabled")


class BotScheduler:
    """
    Manages all recurring bot jobs.
    Inject dependencies — no global state.
    """

    def __init__(
        self,
        paper_trader: "PaperTrader",
        confluence:   "ConfluenceEngine",
        journal:      "Journal",
        feed:         "DataFeed",
    ) -> None:
        self._pt        = paper_trader
        self._ce        = confluence
        self._journal   = journal
        self._feed      = feed
        self._limits    = LossLimits(journal)
        # 10 threads: monitor + H1 scan + H4 scan can all run simultaneously
        # without blocking each other when MT5 calls are slow
        self._scheduler = (
            BackgroundScheduler(
                executors={"default": APSThreadPool(10)},
                timezone="UTC",
            ) if _APScheduler_OK else None
        )
        self._alerter   = TelegramAlerter()

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        if self._scheduler is None:
            logger.error("Cannot start — APScheduler not available")
            return

        # Trade monitor — every 60 seconds
        self._scheduler.add_job(
            func     = self._job_monitor_trades,
            trigger  = IntervalTrigger(seconds=60),
            id       = "monitor_trades",
            name     = "Monitor open paper trades",
            max_instances = 1,
            misfire_grace_time = 30,
        )

        # H1 signal scan — every 2 minutes
        # Strategy's own time gate (13:00-17:00 UTC) blocks trades outside the
        # kill-zone window, so frequent scanning is safe and catches setups fast
        self._scheduler.add_job(
            func     = lambda: self._job_scan("H1"),
            trigger  = IntervalTrigger(minutes=2),
            id       = "scan_h1",
            name     = "H1 signal scan",
            max_instances = 1,
            misfire_grace_time = 30,
        )

        # H4 signal scan — every 4 hours at :00
        self._scheduler.add_job(
            func     = lambda: self._job_scan("H4"),
            trigger  = CronTrigger(hour="0,4,8,12,16,20", minute=5),
            id       = "scan_h4",
            name     = "H4 signal scan",
            max_instances = 1,
            misfire_grace_time = 120,
        )

        # Morning briefing — 06:00 GST = 02:00 UTC
        self._scheduler.add_job(
            func     = self._job_morning_briefing,
            trigger  = CronTrigger(hour=2, minute=0, timezone="UTC"),
            id       = "morning_briefing",
            name     = "Daily morning briefing",
        )

        # Nightly ML retrain — 23:00 GST = 19:00 UTC
        self._scheduler.add_job(
            func     = self._job_retrain,
            trigger  = CronTrigger(hour=19, minute=0, timezone="UTC"),
            id       = "ml_retrain",
            name     = "Nightly ML retrain",
        )

        self._scheduler.start()
        logger.info("BotScheduler started — %d jobs registered", len(self._scheduler.get_jobs()))

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("BotScheduler stopped")

    # ── Job implementations ───────────────────────────────────────────────────

    def _job_monitor_trades(self) -> None:
        """Check all open paper trades for SL/TP/max-hold."""
        try:
            actions = self._pt.check_all_open_trades()
            if actions:
                for a in actions:
                    logger.info("Trade action: %s %s @ %.5f",
                                a["trade_id"][:8], a["action"], a["price"])
                    self._send_alert(a)
        except Exception as exc:
            logger.error("Monitor job error: %s", exc)

    def _job_scan(self, timeframe: str) -> None:
        """Run confluence scan on active instruments for a given timeframe."""
        symbols = ACTIVE_SYMBOLS if ACTIVE_SYMBOLS else ALL_SYMBOLS
        logger.info("Signal scan starting: %s  instruments=%s", timeframe, symbols)

        # Pre-checks
        allowed, reason = self._limits.can_trade()
        if not allowed:
            logger.info("Scan aborted: %s", reason)
            return

        for symbol in symbols:
            for direction in ("long", "short"):
                try:
                    self._scan_one(symbol, direction, timeframe)
                except Exception as exc:
                    logger.error("Scan error %s %s %s: %s", symbol, direction, timeframe, exc)

    def _scan_one(self, symbol: str, direction: str, timeframe: str) -> None:
        """Scan one instrument/direction/timeframe combination."""
        df = self._feed.get_ohlcv(symbol, timeframe, 300)
        if df.empty or len(df) < 50:
            return

        df_h4 = self._feed.get_ohlcv(symbol, "H4", 200) if timeframe == "H1" else None
        df_d1 = self._feed.get_ohlcv(symbol, "D1", 100)

        result = self._ce.score(symbol, direction, df, df_h4, df_d1)

        if not result.get("signal"):
            return

        logger.info(
            "SIGNAL %s %s score=%.1f strategy=%s",
            symbol, direction, result["score"], result.get("strategy", "")
        )

        # Override entry price with live tick so we never trade on stale H1 close
        live = self._feed.get_price(symbol)
        live_price = live.get("price") if live else None

        # Sanity check: if live price deviates >3% from H1 close, data is stale — skip
        h1_close = result.get("entry_price") or 0
        if live_price and h1_close:
            deviation_pct = abs(live_price - h1_close) / h1_close * 100
            if deviation_pct > 3.0:
                logger.warning(
                    "SKIPPING %s — live price %.5f vs H1 close %.5f (%.1f%% apart, likely stale data)",
                    symbol, live_price, h1_close, deviation_pct,
                )
                return

        entry_price = live_price or result.get("entry_price")

        # Recalculate SL distance relative to live price (keep same pip distance)
        h1_entry = result.get("entry_price") or 0
        h1_sl    = result.get("stop_loss") or 0
        if h1_entry and h1_sl and entry_price and entry_price != h1_entry:
            sl_dist  = abs(h1_entry - h1_sl)
            is_long  = direction.lower() in ("long", "buy")
            stop_loss = round(entry_price - sl_dist, 5) if is_long else round(entry_price + sl_dist, 5)
        else:
            stop_loss = h1_sl

        logger.info(
            "Entry override: H1 close=%.5f -> live=%.5f (SL=%.5f)",
            h1_entry, entry_price, stop_loss,
        )

        # Run entry checklist
        signal = {
            "symbol":           symbol,
            "direction":        direction,
            "entry_price":      entry_price,
            "stop_loss":        stop_loss,
            "tp1_price":        result.get("tp1_price"),
            "tp2_price":        result.get("tp2_price"),
            "score":            result.get("score"),
            "confluence_score": result.get("score"),
            "strategy":         result.get("strategy", ""),
            "timeframe":        timeframe,
            "signal_path":      result.get("signal_path", "unknown"),
            "reasons":          result.get("reasons", []),
            "factors":          result.get("factors", {}),
        }

        checklist = validate_entry(signal, df)
        self._journal.log_signal(signal, taken=False, skip_reason="")

        if not checklist["passed"]:
            logger.info("Signal rejected at checklist: %s", checklist["failed_at"])
            self._journal.log_signal(signal, taken=False, skip_reason=checklist["failed_at"])
            return

        # ── ML confidence gate ─────────────────────────────────────────────────
        ml_confidence = self._get_ml_confidence(signal, df)
        if ml_confidence < 0.40:
            logger.info(
                "Signal rejected by ML: %s %s confidence=%.2f",
                symbol, direction, ml_confidence,
            )
            self._journal.log_signal(signal, taken=False, skip_reason=f"ML confidence {ml_confidence:.2f}")
            return
        signal["ml_confidence"] = round(ml_confidence, 3)

        # Open paper trade
        trade_id = self._pt.open_trade(signal)
        if trade_id:
            self._journal.log_signal(signal, taken=True)
            self._send_trade_alert(signal, trade_id)

    def _get_ml_confidence(self, signal: dict, df) -> float:
        """Return ML win probability for a signal. Returns 0.5 if ML unavailable."""
        try:
            from v2.ml.ml_engine import MLEngine
            if not hasattr(self, "_ml_engine"):
                self._ml_engine = MLEngine(journal=self._journal, feed=self._feed)
            return self._ml_engine.get_signal_confidence(signal, df=df)
        except Exception as exc:
            logger.debug("ML confidence unavailable: %s — defaulting to 0.5", exc)
            return 0.5   # allow trade when ML is not trained yet

    def _job_morning_briefing(self) -> None:
        """Log morning briefing stats and send Telegram alert."""
        try:
            from v2.intelligence.news_filter import get_calendar_summary
            cal   = get_calendar_summary()
            stats = self._journal.get_stats(days=7)
            logger.info(
                "Morning briefing: %d events today | 7d WR=%.1f%% PnL=$%.2f",
                cal["count"], stats.get("win_rate", 0), stats.get("total_pnl", 0)
            )
            if cal["warnings"]:
                for w in cal["warnings"]:
                    logger.warning("Calendar warning: %s", w)
            self._alerter.send_morning_briefing(stats, cal)
        except Exception as exc:
            logger.error("Morning briefing error: %s", exc)

    def _job_retrain(self) -> None:
        """Trigger ML retrain if enough new trades available."""
        try:
            from v2.settings import ML_MIN_TRADES_TO_TRAIN
            data = self._journal.get_ml_training_data()
            if len(data) >= ML_MIN_TRADES_TO_TRAIN:
                logger.info("ML retrain: %d samples available — triggering", len(data))
                # ML trainer will be wired here in Week 2
            else:
                logger.info("ML retrain skipped: %d/%d samples", len(data), ML_MIN_TRADES_TO_TRAIN)
        except Exception as exc:
            logger.error("Retrain job error: %s", exc)

    # ── Alert helpers ─────────────────────────────────────────────────────────

    def _send_alert(self, action: dict) -> None:
        """Send a trade-action alert (SL/TP hit, BE move, etc.) via Telegram."""
        action_type = action.get("action", "")
        trade_id    = action.get("trade_id", "")

        trade = self._journal.get_trade(trade_id)

        if action_type == "TP1":
            # TP1 = partial hit, SL moved to breakeven — trade still open
            if trade:
                self._alerter.send_tp1_hit(trade, action.get("price", 0))
            return

        if action_type in ("SL", "TP2", "MANUAL", "MAX_HOLD", "SL_AFTER_TP1"):
            # These are full closes
            if trade:
                self._alerter.send_trade_closed(trade)
            return

        # Fallback for any other action
        self._alerter.send_text(
            f"TRADE ACTION\n"
            f"ID: {trade_id[:8]}\n"
            f"Action: {action_type}\n"
            f"Price: {action.get('price', '?')}"
        )

    def _send_trade_alert(self, signal: dict, trade_id: str) -> None:
        """Send new-trade-opened alert via Telegram."""
        logger.info(
            "TRADE OPENED [%s]: %s %s @ %.5f score=%.1f",
            trade_id[:8],
            signal["symbol"],
            signal["direction"].upper(),
            signal.get("entry_price", 0),
            signal.get("score", 0),
        )
        # Enrich signal with trade_id so the message can reference it
        trade = self._journal.get_trade(trade_id)
        if trade:
            self._alerter.send_trade_opened(trade)
        else:
            # Fallback if trade row not yet readable
            self._alerter.send_signal(signal)
