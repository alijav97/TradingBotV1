"""
btc_research/strategies/base.py — Abstract base class for all BTC strategies.

Each strategy is responsible for:
  1. Deciding whether a signal exists (signal: bool)
  2. Setting the entry price (always bar close for consistency)
  3. Setting the stop-loss level (strategy-specific logic)
  4. Providing a human-readable reason

The comparison engine then applies the SAME exit logic to all strategies:
  TP1 = entry ± 2×SL-distance  (50% close, SL → breakeven)
  TP2 = entry ± 5×SL-distance  (remaining 50% close)
  MAX_HOLD = 96 bars
  3% risk per trade (compounding)

Inter-market filters (Gold, NAS100) are applied by the comparison engine
AFTER the strategy generates its signal — so we can test with/without filters.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class BTCStrategy(ABC):
    """
    Abstract base for all BTC trading strategies.
    Subclasses implement generate_signal() only.
    """

    # Override in each subclass
    name:        str = "BaseStrategy"
    description: str = ""

    @abstractmethod
    def generate_signal(
        self,
        df_window: pd.DataFrame,   # BTCUSD H1 — bars UP TO AND INCLUDING current bar
        bar_time:  pd.Timestamp,   # UTC-aware timestamp of the current bar
        direction: str,            # "long" or "short"
    ) -> dict:
        """
        Evaluate the current bar for a trade entry.

        Args:
            df_window : BTC H1 history up to and including current bar (UTC-aware times)
            bar_time  : current bar timestamp
            direction : "long" (looking for a buy) or "short" (looking for a sell)

        Returns dict:
            signal : bool   — True if entry conditions are met
            entry  : float  — entry price (typically bar close)
            sl     : float  — stop-loss price
            reason : str    — why signal fired / why it didn't
        """

    def __repr__(self) -> str:
        return self.name
