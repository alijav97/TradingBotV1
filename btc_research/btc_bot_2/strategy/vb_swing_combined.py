"""
btc_bot_2/strategy/vb_swing_combined.py — VB + Swing Level strategy for Asia Night.

Why only 2 strategies (not 3 like Bot 1)?

  Bot 1 (21-24 UTC) uses all 3:
    Volatility Breakout  — explosive momentum moves
    Swing Level Break    — structural level breaks
    Morning Range Break  — quiet consolidation breaks after 17-21 UTC range

  Bot 2 (02-04 UTC) drops Morning Range because:
    - Morning Range uses a PRE-KZ consolidation window (Bot 1 uses 17-21 UTC range)
    - At 02-04 UTC, the "morning range" would be ~22:00-02:00 UTC — which is
      Bot 1's own kill-zone session. That range is NOT a quiet consolidation;
      it's an active trending session. Morning Range logic breaks down here.
    - Per-strategy session analysis confirms: Morning Range's best session is
      US Open (13-17 UTC), NOT Asia Night (00-04 UTC).
    - VB + Swing Level both show their BEST performance in Asia Night 00-04 UTC.

IM filter (Gold + NAS100):
  Applied to Volatility Breakout ONLY (same as Bot 1).
  At 02-04 UTC, US markets are closed but NAS100 futures and Gold are active.
  Whether the correlation holds is tested in run_backtest_btc2.py.

Priority order:
  1. Volatility Breakout  (higher avg R, catches explosive Asia momentum)
  2. Swing Level Break    (structural breaks off Asia session levels)
"""
from __future__ import annotations

import pandas as pd
from btc_research.strategies.base               import BTCStrategy
from btc_research.strategies.volatility_breakout import VolatilityBreakout
from btc_research.strategies.swing_level         import SwingLevelBreak

# VB benefits from inter-market filter; Swing Level is neutral
_IM_FILTER_STRATEGIES: set[str] = {"Volatility Breakout"}


class VBSwingStrategy(BTCStrategy):
    """
    Two-strategy combiner for Asia Night session.
    Volatility Breakout fires first (momentum), then Swing Level (structure).
    """

    name        = "VB + Swing Level (Asia Night)"
    description = "Volatility Breakout > Swing Level  |  02-04 UTC"

    def __init__(
        self,
        atr_multiplier: float = 1.2,
        close_zone:     float = 0.45,
    ):
        self._strategies: list[BTCStrategy] = [
            VolatilityBreakout(atr_multiplier=atr_multiplier, close_zone=close_zone),
            SwingLevelBreak(),
        ]

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        """
        Try VB first, then Swing Level. Return first signal that fires.

        Extra keys:
          strategy_used   : which sub-strategy fired
          needs_im_filter : True only for Volatility Breakout
        """
        for strat in self._strategies:
            result = strat.generate_signal(df_window, bar_time, direction)
            if result.get("signal"):
                result["strategy_used"]   = strat.name
                result["needs_im_filter"] = strat.name in _IM_FILTER_STRATEGIES
                result["reason"]          = strat.name
                return result

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
