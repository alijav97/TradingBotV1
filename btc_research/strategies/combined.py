"""
btc_research/strategies/combined.py — Multi-strategy combiner.

Key insight: each strategy fires under DIFFERENT market conditions.

  Volatility Breakout  → fires on explosive ATR-expansion bars (momentum)
  Morning Range Break  → fires on quiet consolidation breaks (accumulation)
  Swing Level Break    → fires on key structural level breaks (structure)
  EMA Trend Follow     → fires when EMAs are stacked (sustained trend)

When market is explosive  → Volatility fires, others may not
When market is quiet      → Morning Range fires, Volatility won't
When price hits structure → Swing Level fires
When strong trend runs    → EMA fires

By combining all four, we get more signals WITHOUT degrading quality —
each signal has its own edge in its own market condition.

Priority order (based on backtest quality ranking):
  1. Volatility Breakout     (best avg R, lowest DD — highest quality)
  2. Morning Range Breakout  (best total PnL with IM filter)
  3. Swing Level Break       (good pure performance)
  4. EMA Trend Follow        (decent with IM filter)

If multiple strategies fire on the same bar, the HIGHEST PRIORITY one wins.
Only one trade is open at a time.

Usage:
    from btc_research.strategies.combined import CombinedStrategy
    strat = CombinedStrategy()
    result = strat.generate_signal(df_window, bar_time, direction)
    # result["strategy_used"] tells you which one fired
"""
from __future__ import annotations

import pandas as pd
from btc_research.strategies.base               import BTCStrategy
from btc_research.strategies.volatility_breakout import VolatilityBreakout
from btc_research.strategies.morning_range       import MorningRangeBreakout
from btc_research.strategies.swing_level         import SwingLevelBreak
from btc_research.strategies.ema_trend           import EMATrendFollow


class CombinedStrategy(BTCStrategy):
    """
    Runs multiple strategies in priority order.
    First one to fire gets the trade.
    Logs which strategy triggered so we can track per-strategy live stats.
    """

    name        = "Combined (All Strategies)"
    description = "Volatility > Morning Range > Swing Level > EMA Trend"

    def __init__(
        self,
        atr_multiplier: float = 1.2,   # Volatility Breakout param (optimized)
        close_zone:     float = 0.45,  # Volatility Breakout param (optimized)
        range_bars:     int   = 6,     # Morning Range param
    ):
        # Ordered by priority — highest quality first
        self._strategies: list[BTCStrategy] = [
            VolatilityBreakout(atr_multiplier=atr_multiplier, close_zone=close_zone),
            MorningRangeBreakout(range_bars=range_bars),
            SwingLevelBreak(),
            EMATrendFollow(),
        ]

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        """
        Try each strategy in priority order.
        Return the first signal that fires.
        Adds 'strategy_used' key so caller knows which strategy triggered.
        """
        for strat in self._strategies:
            result = strat.generate_signal(df_window, bar_time, direction)
            if result.get("signal"):
                result["strategy_used"] = strat.name
                return result

        # None fired
        return {
            "signal":        False,
            "entry":         0.0,
            "sl":            0.0,
            "reason":        "no strategy fired",
            "strategy_used": None,
        }

    @property
    def strategy_names(self) -> list[str]:
        return [s.name for s in self._strategies]
