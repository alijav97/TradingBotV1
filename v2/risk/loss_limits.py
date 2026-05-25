"""
risk/loss_limits.py — Daily/weekly drawdown limits for TradingBotV2.

Reads closed trade PnL from the journal and compares against configured
loss limits. The scanner checks this before opening any new trade.

Usage:
    from v2.risk.loss_limits import LossLimits
    limits = LossLimits(journal)
    allowed, reason = limits.can_trade()
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2.journal.sqlite_journal import Journal

logger = logging.getLogger(__name__)


class LossLimits:
    """Check daily and weekly drawdown limits before trading."""

    def __init__(self, journal: "Journal") -> None:
        self._journal = journal

    def can_trade(self) -> tuple[bool, str]:
        """
        Returns (True, "") if within limits, (False, reason) if breached.
        """
        from v2.settings import ACCOUNT_BALANCE, DAILY_LOSS_LIMIT, WEEKLY_LOSS_LIMIT

        daily_pnl  = self._get_pnl_since(hours=24)
        weekly_pnl = self._get_pnl_since(hours=168)

        daily_pct  = abs(daily_pnl)  / ACCOUNT_BALANCE * 100
        weekly_pct = abs(weekly_pnl) / ACCOUNT_BALANCE * 100

        if daily_pnl < 0 and daily_pct >= DAILY_LOSS_LIMIT:
            msg = (
                f"Daily loss limit reached: {daily_pct:.1f}% "
                f"(limit {DAILY_LOSS_LIMIT}%) — no new trades today"
            )
            logger.warning(msg)
            return False, msg

        if weekly_pnl < 0 and weekly_pct >= WEEKLY_LOSS_LIMIT:
            msg = (
                f"Weekly loss limit reached: {weekly_pct:.1f}% "
                f"(limit {WEEKLY_LOSS_LIMIT}%) — no new trades this week"
            )
            logger.warning(msg)
            return False, msg

        return True, ""

    def get_summary(self) -> dict:
        """Return current daily/weekly P&L and limit status."""
        from v2.settings import ACCOUNT_BALANCE, DAILY_LOSS_LIMIT, WEEKLY_LOSS_LIMIT

        daily_pnl  = self._get_pnl_since(hours=24)
        weekly_pnl = self._get_pnl_since(hours=168)

        return {
            "daily_pnl":      round(daily_pnl, 2),
            "daily_pct":      round(daily_pnl / ACCOUNT_BALANCE * 100, 2),
            "daily_limit":    DAILY_LOSS_LIMIT,
            "daily_ok":       daily_pnl >= 0 or abs(daily_pnl) / ACCOUNT_BALANCE * 100 < DAILY_LOSS_LIMIT,
            "weekly_pnl":     round(weekly_pnl, 2),
            "weekly_pct":     round(weekly_pnl / ACCOUNT_BALANCE * 100, 2),
            "weekly_limit":   WEEKLY_LOSS_LIMIT,
            "weekly_ok":      weekly_pnl >= 0 or abs(weekly_pnl) / ACCOUNT_BALANCE * 100 < WEEKLY_LOSS_LIMIT,
        }

    def _get_pnl_since(self, hours: int) -> float:
        """Sum PnL of all trades closed in the last N hours."""
        try:
            since_iso = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            rows = self._journal._conn.execute(
                """SELECT COALESCE(SUM(pnl_usd), 0)
                   FROM trades
                   WHERE status='CLOSED' AND close_time >= ?""",
                (since_iso,)
            ).fetchone()
            return float(rows[0]) if rows else 0.0
        except Exception as exc:
            logger.error("Loss limit check failed: %s", exc)
            return 0.0
