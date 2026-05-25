"""
trading/auto_trader.py — Signal-to-paper-trade orchestrator for TradingBotV2.

Drives one complete scan cycle: for every symbol × direction, fetch OHLCV data,
score with the confluence engine, validate the entry checklist, enforce risk
limits, open a paper trade via PaperTrader, and log the signal to the journal.

Design notes (V1 issues addressed):
  - No global state — all dependencies are constructor-injected.
  - No threading inside AutoTrader itself; the BotScheduler owns that layer.
  - Each scan step catches its own exception so one bad symbol cannot abort
    the whole scan.
  - Loss-limit check happens once per scan cycle (not per symbol), then
    portfolio-heat is checked inside PaperTrader.open_trade() per symbol.

Usage:
    from v2.trading.auto_trader import AutoTrader
    from v2.journal.sqlite_journal import Journal
    from v2.connectors.unified_data import DataFeed
    from v2.signals.confluence_engine import ConfluenceEngine
    from v2.trading.paper_trader import PaperTrader

    journal  = Journal()
    feed     = DataFeed()
    engine   = ConfluenceEngine()
    pt       = PaperTrader(journal=journal, feed=feed)
    trader   = AutoTrader(paper_trader=pt, confluence_engine=engine,
                          journal=journal, feed=feed)

    actions = trader.run_scan("H1")
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2.trading.paper_trader import PaperTrader
    from v2.signals.confluence_engine import ConfluenceEngine
    from v2.journal.sqlite_journal import Journal
    from v2.connectors.unified_data import DataFeed
    from v2.intelligence.news_monitor import NewsMonitor
    from v2.intelligence.dxy_correlation import DXYCorrelation

from v2.instrument_config import ALL_SYMBOLS
from v2.signals.entry_checklist import validate_entry
from v2.risk.loss_limits import LossLimits

logger = logging.getLogger(__name__)


class AutoTrader:
    """
    Orchestrates the full signal → paper trade pipeline for one scan cycle.

    Parameters
    ----------
    paper_trader      : PaperTrader instance (handles trade open + heat checks)
    confluence_engine : ConfluenceEngine instance
    journal           : Journal instance (signal logging)
    feed              : DataFeed instance (OHLCV + price data)
    news_monitor      : optional NewsMonitor (used to enrich signal context)
    dxy               : optional DXYCorrelation (used to enrich signal context)
    """

    def __init__(
        self,
        paper_trader:      "PaperTrader",
        confluence_engine: "ConfluenceEngine",
        journal:           "Journal",
        feed:              "DataFeed",
        news_monitor:      "NewsMonitor | None" = None,
        dxy:               "DXYCorrelation | None" = None,
    ) -> None:
        self._pt       = paper_trader
        self._ce       = confluence_engine
        self._journal  = journal
        self._feed     = feed
        self._news     = news_monitor
        self._dxy      = dxy
        self._limits   = LossLimits(journal)

    # ── Public API ────────────────────────────────────────────────────────────

    def run_scan(self, timeframe: str = "H1") -> list[dict]:
        """
        Run one complete scan cycle across all symbols and directions.

        Steps per symbol/direction:
          1. Fetch OHLCV data (H1 + H4 + D1 for context).
          2. Run confluence engine to get score.
          3. If signal fires: run entry checklist.
          4. If checklist passes: open paper trade via PaperTrader
             (PaperTrader enforces portfolio-heat + correlation limits).
          5. Log signal to journal (taken=True/False with skip reason).

        A single pre-scan loss-limit check aborts the whole cycle early if the
        daily or weekly drawdown cap has been hit — avoids redundant DB queries
        for every symbol.

        Returns
        -------
        list[dict]
            One entry per (symbol, direction) pair where a signal fired,
            whether or not a trade was ultimately opened.
            Keys: symbol, direction, timeframe, score, signal_fired,
                  checklist_passed, trade_id (None if not opened), skip_reason.
        """
        logger.info("AutoTrader scan starting: timeframe=%s symbols=%d",
                    timeframe, len(ALL_SYMBOLS))

        # Pre-scan: check daily/weekly loss limits once for the whole cycle.
        limits_ok, limits_reason = self._limits.can_trade()
        if not limits_ok:
            logger.warning("Scan aborted — loss limits breached: %s", limits_reason)
            return [{"scan_aborted": True, "reason": limits_reason}]

        actions: list[dict] = []

        for symbol in ALL_SYMBOLS:
            for direction in ("long", "short"):
                try:
                    result = self._scan_one(symbol, direction, timeframe)
                    if result is not None:
                        actions.append(result)
                except Exception as exc:
                    logger.error(
                        "Scan error for %s %s %s: %s",
                        symbol, direction, timeframe, exc,
                        exc_info=True,
                    )

        open_count = len([a for a in actions if a.get("trade_id")])
        logger.info(
            "Scan complete: %d signals, %d trades opened",
            len(actions), open_count,
        )
        return actions

    def run_forever(
        self,
        h1_interval_sec:      int = 3600,
        monitor_interval_sec: int = 60,
    ) -> None:
        """
        Blocking loop that alternates between scan cycles and trade monitoring.

        Every ``monitor_interval_sec`` seconds it calls
        ``paper_trader.check_all_open_trades()`` to evaluate SL/TP/max-hold.
        Every ``h1_interval_sec`` seconds it calls ``run_scan("H1")``.

        Note: this method is provided for simple single-process deployments.
        For production use the BotScheduler, which runs everything in a
        BackgroundScheduler thread and supports H4 scans, morning briefings,
        and ML retraining as well.

        Handles ``KeyboardInterrupt`` cleanly — logs shutdown and returns.
        """
        logger.info(
            "AutoTrader.run_forever() starting "
            "(scan_interval=%ds, monitor_interval=%ds)",
            h1_interval_sec, monitor_interval_sec,
        )

        last_scan_at    = 0.0  # epoch seconds — run immediately on first tick
        last_monitor_at = 0.0

        try:
            while True:
                now = time.monotonic()

                # Trade monitor tick
                if now - last_monitor_at >= monitor_interval_sec:
                    try:
                        actions = self._pt.check_all_open_trades()
                        if actions:
                            logger.info(
                                "Monitor tick: %d trade action(s)", len(actions)
                            )
                            for a in actions:
                                logger.info(
                                    "  Trade %s: %s @ %.5f",
                                    a.get("trade_id", "?")[:8],
                                    a.get("action", "?"),
                                    a.get("price", 0.0),
                                )
                    except Exception as exc:
                        logger.error("Monitor tick error: %s", exc, exc_info=True)
                    last_monitor_at = now

                # H1 scan tick
                if now - last_scan_at >= h1_interval_sec:
                    try:
                        self.run_scan("H1")
                    except Exception as exc:
                        logger.error("Scan tick error: %s", exc, exc_info=True)
                    last_scan_at = now

                # Sleep until the next monitor tick is due — avoids busy-waiting.
                time.sleep(min(monitor_interval_sec, 10))

        except KeyboardInterrupt:
            logger.info("AutoTrader.run_forever() stopped by KeyboardInterrupt")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _scan_one(
        self,
        symbol:    str,
        direction: str,
        timeframe: str,
    ) -> dict | None:
        """
        Evaluate one symbol/direction/timeframe combination.

        Returns a result dict if a signal was generated (regardless of whether
        a trade was opened), or None if there was no signal.
        """
        # ── 1. Fetch OHLCV data ───────────────────────────────────────────────
        df = self._feed.get_ohlcv(symbol, timeframe, 300)
        if df is None or df.empty or len(df) < 50:
            logger.debug("No OHLCV data for %s %s — skipping", symbol, timeframe)
            return None

        # Fetch higher-timeframe context for confluence scoring
        df_h4: object = None
        df_d1: object = None
        try:
            if timeframe == "H1":
                df_h4 = self._feed.get_ohlcv(symbol, "H4", 200)
            df_d1 = self._feed.get_ohlcv(symbol, "D1", 100)
        except Exception as exc:
            logger.debug("HTF fetch error for %s: %s", symbol, exc)

        # ── 2. Build optional context dict ────────────────────────────────────
        context: dict = {}
        if self._news is not None:
            try:
                context["news_score"] = self._news.get_score(symbol)
            except Exception:
                pass
        if self._dxy is not None:
            try:
                context["dxy_bias"] = self._dxy.get_bias()
            except Exception:
                pass

        # ── 3. Score with confluence engine ───────────────────────────────────
        try:
            result = self._ce.score(symbol, direction, df, df_h4, df_d1, context)
        except Exception as exc:
            logger.error(
                "Confluence engine error for %s %s: %s", symbol, direction, exc
            )
            return None

        if not result.get("signal"):
            return None  # below MIN_SCORE — no action needed

        score    = float(result.get("score", 0))
        strategy = result.get("strategy", "")
        logger.info(
            "SIGNAL %s %s %s score=%.1f strategy=%s",
            symbol, direction, timeframe, score, strategy,
        )

        # ── 4. Build signal dict ──────────────────────────────────────────────
        signal: dict = {
            "symbol":           symbol,
            "direction":        direction,
            "timeframe":        timeframe,
            "entry_price":      result.get("entry_price"),
            "stop_loss":        result.get("stop_loss"),
            "tp1_price":        result.get("tp1_price"),
            "tp2_price":        result.get("tp2_price"),
            "score":            score,
            "confluence_score": score,
            "strategy":         strategy,
            "session":          result.get("session", ""),
            "regime":           result.get("regime", ""),
            "news_score":       context.get("news_score"),
        }

        # ── 5. Entry checklist ────────────────────────────────────────────────
        try:
            checklist = validate_entry(signal, df)
        except Exception as exc:
            logger.error("Checklist error for %s %s: %s", symbol, direction, exc)
            checklist = {"passed": False, "failed_at": f"checklist_exception:{exc}"}

        checklist_passed = checklist.get("passed", False)
        skip_reason      = "" if checklist_passed else checklist.get("failed_at", "unknown")

        # ── 6. Open paper trade if checklist passes ───────────────────────────
        trade_id: str | None = None
        if checklist_passed:
            try:
                # PaperTrader internally checks portfolio heat + correlation limits.
                trade_id = self._pt.open_trade(signal)
                if trade_id is None:
                    # Blocked by portfolio heat / correlation (already logged inside PT)
                    skip_reason = "portfolio_heat_or_correlation"
            except Exception as exc:
                logger.error(
                    "PaperTrader.open_trade error for %s %s: %s",
                    symbol, direction, exc,
                )
                skip_reason = f"open_trade_exception:{exc}"

        # ── 7. Log signal to journal ──────────────────────────────────────────
        taken = trade_id is not None
        try:
            self._journal.log_signal(signal, taken=taken, skip_reason=skip_reason)
        except Exception as exc:
            logger.error("Journal.log_signal error for %s %s: %s", symbol, direction, exc)

        return {
            "symbol":           symbol,
            "direction":        direction,
            "timeframe":        timeframe,
            "score":            score,
            "signal_fired":     True,
            "checklist_passed": checklist_passed,
            "trade_id":         trade_id,
            "skip_reason":      skip_reason,
            "timestamp":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
