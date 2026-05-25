"""
risk/portfolio_heat.py — Cross-instrument risk monitor for TradingBotV2.

Tracks total open risk across all paper trades and enforces:
  - MAX_PORTFOLIO_HEAT: no more than X% of account at risk at once
  - MAX_OPEN_TRADES: hard cap on concurrent open trades
  - Correlation check: prevents opening 3+ correlated longs simultaneously

Usage:
    from v2.risk.portfolio_heat import PortfolioHeat
    heat = PortfolioHeat(journal)
    if heat.can_open_trade("XAUUSD", "long", risk_usd=100):
        ...
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2.journal.sqlite_journal import Journal

from v2.instrument_config import get_instrument

logger = logging.getLogger(__name__)

# Correlation groups — instruments that move together
# Opening too many in the same direction at once multiplies hidden risk
_CORRELATION_GROUPS: list[list[str]] = [
    ["XAUUSD", "WTI"],           # commodity / anti-dollar
    ["BTCUSDT", "ETHUSDT"],      # crypto
    ["NAS100"],                  # indices (solo for now)
    ["GBPJPY"],                  # FX (solo for now)
]

_MAX_CORRELATED_SAME_DIR = 2   # max instruments from same group in same direction


class PortfolioHeat:
    """
    Checks whether opening a new trade would breach risk limits.
    Reads live open trade state from the Journal.
    """

    def __init__(self, journal: "Journal") -> None:
        self._journal = journal

    def current_heat_pct(self, account_balance: float | None = None) -> float:
        """
        Return current portfolio heat as % of account balance.
        Heat = sum of all open trade risk amounts (SL distance × pip value × lots).
        """
        from v2.settings import ACCOUNT_BALANCE
        balance  = account_balance or ACCOUNT_BALANCE
        open_trades = self._journal.get_open_trades()

        total_risk = 0.0
        for t in open_trades:
            total_risk += self._estimate_risk_usd(t)

        return round(total_risk / balance * 100, 2) if balance else 0.0

    def can_open_trade(
        self,
        symbol: str,
        direction: str,
        risk_usd: float,
        account_balance: float | None = None,
    ) -> tuple[bool, str]:
        """
        Check all portfolio risk limits before opening a new trade.

        Returns (allowed: bool, reason: str)
        reason is empty string if allowed, explanation if blocked.
        """
        from v2.settings import ACCOUNT_BALANCE, MAX_OPEN_TRADES, MAX_PORTFOLIO_HEAT

        balance     = account_balance or ACCOUNT_BALANCE
        open_trades = self._journal.get_open_trades()

        # 1. Max concurrent trades
        if len(open_trades) >= MAX_OPEN_TRADES:
            msg = f"Max open trades reached ({MAX_OPEN_TRADES})"
            logger.info("Trade blocked: %s", msg)
            return False, msg

        # 2. Portfolio heat cap
        current_risk = sum(self._estimate_risk_usd(t) for t in open_trades)
        new_risk_pct = (current_risk + risk_usd) / balance * 100
        if new_risk_pct > MAX_PORTFOLIO_HEAT:
            msg = f"Portfolio heat {new_risk_pct:.1f}% would exceed {MAX_PORTFOLIO_HEAT}%"
            logger.info("Trade blocked: %s", msg)
            return False, msg

        # 3. Correlation check
        corr_check = self._correlation_check(symbol, direction, open_trades)
        if corr_check:
            logger.info("Trade blocked: %s", corr_check)
            return False, corr_check

        return True, ""

    def get_heat_summary(self, account_balance: float | None = None) -> dict:
        """Return a dict with full portfolio risk breakdown."""
        from v2.settings import ACCOUNT_BALANCE, MAX_PORTFOLIO_HEAT, MAX_OPEN_TRADES
        balance     = account_balance or ACCOUNT_BALANCE
        open_trades = self._journal.get_open_trades()
        total_risk  = sum(self._estimate_risk_usd(t) for t in open_trades)
        heat_pct    = round(total_risk / balance * 100, 2) if balance else 0.0

        return {
            "open_trades":    len(open_trades),
            "max_trades":     MAX_OPEN_TRADES,
            "total_risk_usd": round(total_risk, 2),
            "heat_pct":       heat_pct,
            "max_heat_pct":   MAX_PORTFOLIO_HEAT,
            "headroom_pct":   round(MAX_PORTFOLIO_HEAT - heat_pct, 2),
            "at_capacity":    heat_pct >= MAX_PORTFOLIO_HEAT or len(open_trades) >= MAX_OPEN_TRADES,
            "per_trade":      [
                {
                    "symbol":    t["symbol"],
                    "direction": t["direction"],
                    "risk_usd":  self._estimate_risk_usd(t),
                }
                for t in open_trades
            ],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _estimate_risk_usd(self, trade: dict) -> float:
        """Estimate open USD risk from a trade row."""
        try:
            from v2.risk.position_sizer import calculate_risk_usd
            return calculate_risk_usd(
                trade["symbol"],
                float(trade["entry_price"]),
                float(trade["stop_loss"]),
                float(trade["lot_size"]),
            )
        except Exception:
            return 0.0

    def _correlation_check(
        self,
        symbol: str,
        direction: str,
        open_trades: list[dict],
    ) -> str:
        """
        Return a block reason if adding this trade would create too many
        correlated positions in the same direction. Returns "" if OK.
        """
        dir_norm = direction.lower()

        for group in _CORRELATION_GROUPS:
            if symbol not in group:
                continue
            count = sum(
                1 for t in open_trades
                if t["symbol"] in group and t["direction"].lower() == dir_norm
            )
            if count >= _MAX_CORRELATED_SAME_DIR:
                return (
                    f"Correlation limit: already {count} {dir_norm} "
                    f"in group {group}"
                )
        return ""
