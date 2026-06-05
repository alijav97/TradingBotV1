"""
btc_research/eth_bot/strategy/eth_combined.py — Strategy combiner for ETH Bot.

== CURRENT STATE: PLACEHOLDER ==
  Using VB + SwingLevel v2 (same as BTC Bot 2) as a starting point.
  Run ETH-specific backtests to determine the optimal strategy combination,
  entry modes, and parameter values before going live.

== TO CUSTOMISE AFTER BACKTEST ==
  1. Change SWING_ENTRY_MODE in settings.py ("both" / "retest" / "break_capped")
  2. Change SWING_MAX_SL_ATR in settings.py (SL cap multiplier)
  3. Swap out VolatilityBreakout for a different strategy if ETH backtest shows
     a better approach (e.g. SwingLevel-only, or Inside Bar Breakout)
  4. Adjust TP1_RR / TP2_RR in settings.py if ETH's avg RR profile differs

== PRIORITY ORDER ==
  1. SwingLevelBreakV2 — checked first (higher WR on BTC, likely similar on ETH)
  2. VolatilityBreakout — fallback (ETH is highly volatile, VB may fire more often)

== ETH vs BTC DIFFERENCES TO WATCH ==
  - ETH has larger ATR as % of price (~3-5% vs BTC ~2-3%) → SL distances wider
  - ETH reacts more sharply to risk-on/off events
  - ETH often leads or lags BTC by 1-2 bars — worth checking in backtest
  - Asia Night session behaviour may differ from BTC
"""
from __future__ import annotations

import pandas as pd
from btc_research.strategies.base               import BTCStrategy
from btc_research.strategies.volatility_breakout import VolatilityBreakout
from btc_research.strategies.swing_level_v2      import SwingLevelBreakV2
from btc_research.eth_bot.settings import (
    TP1_RR, TP2_RR,
    ADX_SPLIT_EARLY_MAX, ADX_SPLIT_STRONG_MIN,
    RISK_PCT_EARLY_TREND, RISK_PCT_TRANSITION, RISK_PCT_STRONG,
    SWING_ENTRY_MODE, SWING_MAX_SL_ATR,
)


def get_risk_pct(adx: float) -> float:
    """
    ADX-split risk sizing for ETH Bot.

    ADX ≤ ADX_SPLIT_EARLY_MAX  : RISK_PCT_EARLY_TREND  (3%)
    ADX > EARLY and < STRONG   : RISK_PCT_TRANSITION   (2%)
    ADX ≥ ADX_SPLIT_STRONG_MIN : RISK_PCT_STRONG       (3%)
    """
    if adx >= ADX_SPLIT_STRONG_MIN:
        return RISK_PCT_STRONG
    elif adx <= ADX_SPLIT_EARLY_MAX:
        return RISK_PCT_EARLY_TREND
    else:
        return RISK_PCT_TRANSITION


class ETHStrategy(BTCStrategy):
    """
    Strategy combiner for ETH Bot.

    PRIORITY: SwingLevelBreakV2 fires FIRST, VolatilityBreakout fires as fallback.
    TP levels standardised to TP1=2R / TP2=5R (same as BTC Bot 2 — validate with ETH backtest).
    """

    name        = "ETH: Swing Level v2 + VB (placeholder)"
    description = "SwingLevelBreak v2 [both 2xATR] > Volatility Breakout | kill-zone TBD"

    def __init__(
        self,
        swing_entry_mode: str   = SWING_ENTRY_MODE,
        swing_max_sl_atr: float = SWING_MAX_SL_ATR,
    ):
        self._strategies: list[BTCStrategy] = [
            SwingLevelBreakV2(entry_mode=swing_entry_mode, max_sl_atr=swing_max_sl_atr),
            VolatilityBreakout(),
        ]

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        """
        Try SwingLevelBreakV2 first, then Volatility Breakout as fallback.
        Returns first signal that fires, with standardised TP levels.
        """
        for strat in self._strategies:
            result = strat.generate_signal(df_window, bar_time, direction)
            if result.get("signal"):
                result["tp1_rr"]        = TP1_RR   # 2.0R — standardised
                result["tp2_rr"]        = TP2_RR   # 5.0R — standardised
                result["strategy_used"] = strat.name
                result["reason"]        = strat.name
                return result

        return {
            "signal":        False,
            "entry":         0.0,
            "sl":            0.0,
            "tp1_rr":        TP1_RR,
            "tp2_rr":        TP2_RR,
            "reason":        "no strategy fired",
            "strategy_used": None,
        }

    @property
    def strategy_names(self) -> list[str]:
        return [s.name for s in self._strategies]
