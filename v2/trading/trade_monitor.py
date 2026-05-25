"""
trading/trade_monitor.py — Standalone trade monitoring for TradingBotV2.

Extracted from PaperTrader for clarity: wraps
``paper_trader.check_all_open_trades()`` and adds structured summary logging
so the scheduler and API have a single, observable entry point for monitoring.

Design notes:
  - Depends only on Journal and DataFeed, not on PaperTrader directly.
    This allows TradeMonitor to be used independently (e.g. in tests or
    a dedicated monitoring process) without pulling in the full PaperTrader.
  - PaperTrader is still the canonical implementation of SL/TP/max-hold
    logic; TradeMonitor delegates to it rather than reimplementing checks.
  - All portfolio-heat context is logged *before* each check so the log
    stream is useful for post-hoc analysis.

Usage:
    from v2.trading.trade_monitor import TradeMonitor
    from v2.trading.paper_trader import PaperTrader
    from v2.journal.sqlite_journal import Journal
    from v2.connectors.unified_data import DataFeed

    journal  = Journal()
    feed     = DataFeed()
    pt       = PaperTrader(journal=journal, feed=feed)
    monitor  = TradeMonitor(paper_trader=pt, journal=journal, feed=feed)

    summary  = monitor.check_all()
    # {"checked": 3, "actions": [...], "still_open": 2}
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2.trading.paper_trader import PaperTrader
    from v2.journal.sqlite_journal import Journal
    from v2.connectors.unified_data import DataFeed

from v2.risk.portfolio_heat import PortfolioHeat

logger = logging.getLogger(__name__)


class TradeMonitor:
    """
    Thin orchestration layer around PaperTrader's monitoring logic.

    Adds per-cycle logging of open-trade count and portfolio heat before
    delegating to PaperTrader for the actual SL/TP/max-hold evaluation.

    Parameters
    ----------
    paper_trader : PaperTrader  — handles the actual SL/TP/max-hold decisions
    journal      : Journal      — reads open-trade state and logs results
    feed         : DataFeed     — not used directly here, wired into PT
    """

    def __init__(
        self,
        paper_trader: "PaperTrader",
        journal:      "Journal",
        feed:         "DataFeed",
    ) -> None:
        self._pt      = paper_trader
        self._journal = journal
        self._feed    = feed
        self._heat    = PortfolioHeat(journal)

    # ── Public API ────────────────────────────────────────────────────────────

    def check_all(self) -> dict:
        """
        Check every open trade for SL/TP/max-hold conditions.

        Logs open-trade count and portfolio heat percentage before running
        checks, then returns a structured summary.

        Returns
        -------
        dict with keys:
            checked    : int  — number of open trades evaluated
            actions    : list[dict]  — actions taken (SL/TP/MAX_HOLD hits)
            still_open : int  — trades still open after this cycle
            heat_pct   : float  — portfolio heat % before this cycle
            timestamp  : str   — ISO-8601 UTC timestamp of this check
        """
        # Snapshot state before checking
        open_trades = self._journal.get_open_trades()
        open_count  = len(open_trades)
        heat_pct    = self._heat.current_heat_pct()

        logger.info(
            "TradeMonitor cycle start: %d open trade(s), heat=%.1f%%",
            open_count, heat_pct,
        )

        if open_count == 0:
            logger.debug("No open trades — monitor cycle skipped")
            return {
                "checked":    0,
                "actions":    [],
                "still_open": 0,
                "heat_pct":   0.0,
                "timestamp":  _now_iso(),
            }

        # Delegate to PaperTrader for SL/TP/max-hold evaluation
        actions: list[dict] = []
        try:
            actions = self._pt.check_all_open_trades()
        except Exception as exc:
            logger.error("PaperTrader.check_all_open_trades failed: %s", exc, exc_info=True)

        # Log each action taken this cycle
        for action in actions:
            logger.info(
                "Trade action: %s %s @ %.5f",
                action.get("trade_id", "?")[:8],
                action.get("action", "?"),
                action.get("price", 0.0),
            )

        # Re-count open trades after the cycle
        still_open = len(self._journal.get_open_trades())

        closed_this_cycle = open_count - still_open
        if closed_this_cycle > 0:
            logger.info(
                "TradeMonitor cycle end: %d trade(s) closed, %d still open",
                closed_this_cycle, still_open,
            )
        else:
            logger.debug(
                "TradeMonitor cycle end: no changes, %d still open", still_open
            )

        return {
            "checked":    open_count,
            "actions":    actions,
            "still_open": still_open,
            "heat_pct":   heat_pct,
            "timestamp":  _now_iso(),
        }

    def check_one(self, trade_id: str) -> dict | None:
        """
        Check a single open trade by ID.

        Fetches the trade from the journal, logs pre-check context, then
        delegates to ``PaperTrader.monitor_trade()`` for evaluation.

        Returns
        -------
        dict | None
            The action dict if something happened (SL/TP/max-hold), or
            ``{"trade_id": ..., "action": "NO_ACTION"}`` if the trade was
            found but nothing triggered, or ``None`` if the trade was not
            found (already closed, or ID typo).
        """
        trade = self._journal.get_trade(trade_id)
        if trade is None:
            logger.warning("check_one: trade %s not found or already closed", trade_id[:8])
            return None

        if trade.get("status") != "OPEN":
            logger.debug(
                "check_one: trade %s is %s — skipping",
                trade_id[:8], trade.get("status"),
            )
            return None

        symbol    = trade.get("symbol", "?")
        direction = trade.get("direction", "?")
        heat_pct  = self._heat.current_heat_pct()

        logger.info(
            "check_one: %s %s %s heat=%.1f%%",
            trade_id[:8], symbol, direction, heat_pct,
        )

        try:
            action = self._pt.monitor_trade(trade)
        except Exception as exc:
            logger.error(
                "PaperTrader.monitor_trade failed for %s: %s", trade_id[:8], exc,
                exc_info=True,
            )
            return None

        if action is not None:
            logger.info(
                "Trade action: %s %s @ %.5f",
                action.get("trade_id", trade_id)[:8],
                action.get("action", "?"),
                action.get("price", 0.0),
            )
            return action

        # Trade was checked but nothing happened
        return {
            "trade_id":  trade_id,
            "symbol":    symbol,
            "direction": direction,
            "action":    "NO_ACTION",
            "timestamp": _now_iso(),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
