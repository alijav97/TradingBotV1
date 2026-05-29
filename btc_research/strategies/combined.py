"""
btc_research/strategies/combined.py — Multi-strategy combiner (3-strategy version).

Key insight: each strategy fires under DIFFERENT market conditions.

  Volatility Breakout  → fires on explosive ATR-expansion bars (momentum)
  Swing Level Break    → fires on key structural level breaks (structure)
  Morning Range Break  → fires on quiet consolidation breaks (accumulation)

When market is explosive  → Volatility fires, others may not
When price hits structure → Swing Level fires (Volatility already handled explosive)
When market is quiet      → Morning Range mops up what the others missed

EMA Trend Follow and RSI Mean Reversion are EXCLUDED from Combined:
  - EMA fired 145 trades at 34.5% WR / +0.04R inside Combined — dragged DD to 60%
  - RSI is counter-trend and conflicts with momentum-based entries
  - Both strategies work better as standalone signals with their own filter logic

Priority order (based on backtest quality ranking):
  1. Volatility Breakout  (best avg R +0.48R, lowest MaxDD 14% — highest quality)
  2. Swing Level Break    (best WR 46%, only 18% MaxDD — safest structure trades)
  3. Morning Range Break  (best total PnL $26K — highest frequency / best volume)

IM filter applies SELECTIVELY:
  - Volatility Breakout  → YES filter (Gold+NAS add +$2,363 edge)
  - Swing Level Break    → NO filter  (filter neutral, cleaner without)
  - Morning Range Break  → NO filter  (filter HURTS by -$3,698 — strategy is self-filtering)

If multiple strategies fire on the same bar, the HIGHEST PRIORITY one wins.
Only one trade is open at a time.

Usage:
    from btc_research.strategies.combined import CombinedStrategy
    strat = CombinedStrategy()
    result = strat.generate_signal(df_window, bar_time, direction)
    # result["strategy_used"] tells you which one fired
    # result["needs_im_filter"] tells caller whether to apply Gold/NAS gate
"""
from __future__ import annotations

import pandas as pd
from btc_research.strategies.base               import BTCStrategy
from btc_research.strategies.volatility_breakout import VolatilityBreakout
from btc_research.strategies.morning_range       import MorningRangeBreakout
from btc_research.strategies.swing_level         import SwingLevelBreak


# Strategies that benefit from the inter-market filter (Gold + NAS100)
_FILTER_STRATEGIES: set[str] = {"Volatility Breakout"}


class CombinedStrategy(BTCStrategy):
    """
    Runs 3 complementary strategies in priority order.
    First one to fire gets the trade.
    Logs which strategy triggered so we can track per-strategy live stats.
    Also flags whether the caller should apply the inter-market filter.
    """

    name        = "Combined (3-Strategy)"
    description = "Volatility > Swing Level > Morning Range  (selective IM filter)"

    def __init__(
        self,
        atr_multiplier: float = 1.2,   # Volatility Breakout param (optimized)
        close_zone:     float = 0.45,  # Volatility Breakout param (optimized)
        range_bars:     int   = 6,     # Morning Range param
    ):
        # Ordered by quality/priority — best avg R first
        self._strategies: list[BTCStrategy] = [
            VolatilityBreakout(atr_multiplier=atr_multiplier, close_zone=close_zone),
            SwingLevelBreak(),
            MorningRangeBreakout(range_bars=range_bars),
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

        Extra keys added to result:
          strategy_used   : name of the sub-strategy that fired
          needs_im_filter : True if caller should apply Gold/NAS gate before taking trade
        """
        for strat in self._strategies:
            result = strat.generate_signal(df_window, bar_time, direction)
            if result.get("signal"):
                result["strategy_used"]   = strat.name
                result["needs_im_filter"] = strat.name in _FILTER_STRATEGIES
                # Overwrite reason with strategy name so sub-strategy breakdown
                # can group trades correctly (otherwise each unique reason string
                # creates its own row in value_counts())
                result["reason"] = strat.name
                return result

        # None fired
        return {
            "signal":          False,
            "entry":           0.0,
            "sl":              0.0,
            "reason":          "no strategy fired",
            "strategy_used":   None,
            "needs_im_filter": False,
        }

    @property
    def strategy_names(self) -> list[str]:
        return [s.name for s in self._strategies]
