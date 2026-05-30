"""
btc_bot_2/strategy/vb_swing_combined.py — VB + Swing Level strategy for Asia Night.

== SESSION ==
  Kill-zone hours: 01:00, 02:00, 03:00 UTC (Asia Night) + 08:00 UTC (EU open)
  UAE times:       05:00, 06:00, 07:00 AM + 12:00 noon

== WHY ONLY 2 STRATEGIES (not 3 like Bot 1) ==
  Bot 1 (21-24 UTC) uses VB + Swing Level + Morning Range.
  Bot 2 drops Morning Range because at 01-03 UTC the "pre-session range"
  would be Bot 1's own kill-zone (21-24 UTC) — an active trending session,
  not a quiet consolidation. Morning Range logic breaks here.
  Per-strategy session analysis: VB and Swing Level both peak at Asia Night.
  Morning Range peaks at US Open (13-17 UTC).

== SL PLACEMENT ==

  Volatility Breakout:
    Entry : bar close  (the large momentum bar itself)
    SL    : low of the breakout bar  (for longs)
            high of the breakout bar (for shorts)
    Logic : The momentum bar's edge IS the entry protection. If price returns
            below the low of a bar that just expanded 1.2×ATR with a strong
            close, the breakout has failed.
    Typical SL distance: 0.5-1.0× ATR  (~$250-600 at Asia Night ATR)

  Swing Level Break:
    Entry : bar close (after price breaks beyond the swing level)
    SL    : most recent swing LOW before the broken swing HIGH (for longs)
            most recent swing HIGH before the broken swing LOW  (for shorts)
    Logic : The swing structure is the key level. If price returns through
            the previous swing structure, the breakout thesis is invalidated.
    Typical SL distance: variable, 1-3× ATR depending on swing structure

== TP LEVELS (STANDARDISED — matching backtest) ==

  Both strategies use:  TP1 = 2.0R  |  TP2 = 5.0R
  Trailing SL: 2×ATR after TP1 (kicks in when trade goes in our favour)

  Why override VB's 9R default?
    The backtest tested: current(VB=9R) vs Bot1-same(TP1=2R/TP2=5R) vs others.
    Bot1-same (TP1=2R, TP2=5R) gave the BEST PnL: $2,257 vs $1,698 for VB=9R.
    The trailing SL after TP1 does a better job of letting winners run than
    a fixed 9R target that rarely gets hit in Asia Night sessions.

  Why override Swing Level's 1.5R TP1?
    Same test showed 2R TP1 outperforms 1.5R. The swing structure moves are
    substantial enough at Asia Night ATR to reach 2R before retracing.

== RISK SIZING (ADX-SPLIT — applied by the signal engine, not here) ==

  ADX 20-25 (early trend):  3%  — market just starting to trend, clean entry
  ADX 25-40 (transition):   2%  — dead zone in Asia Night, conservative sizing
  ADX 40+   (strong trend): 3%  — powerful momentum, ride with more size

  This is implemented in btc_bot_2/settings.py and the signal engine.
  The strategy itself only returns entry/SL. Risk sizing is the caller's job.
"""
from __future__ import annotations

import pandas as pd
from btc_research.strategies.base               import BTCStrategy
from btc_research.strategies.volatility_breakout import VolatilityBreakout
from btc_research.strategies.swing_level         import SwingLevelBreak
from btc_research.btc_bot_2.settings import (
    TP1_RR, TP2_RR,
    ADX_SPLIT_EARLY_MAX, ADX_SPLIT_STRONG_MIN,
    RISK_PCT_EARLY_TREND, RISK_PCT_TRANSITION, RISK_PCT_STRONG,
)

# VB benefits from inter-market filter; Swing Level is neutral
_IM_FILTER_STRATEGIES: set[str] = {"Volatility Breakout"}


def get_risk_pct(adx: float) -> float:
    """
    ADX-split risk sizing for Bot 2.

    ADX 20-25 (early trend):  3%
    ADX 25-40 (transition):   2%
    ADX 40+   (strong trend): 3%

    Rationale from 2yr backtest:
      ADX 40+ zone: 53.6% WR, +0.96R — strong trends at Asia Night are reliable
      ADX 20-25 zone: 48.5% WR, +0.77R — early trend entries also reliable
      ADX 25-40 zone: 37.5% WR, +0.36R — transition / choppy, size down
    """
    if adx >= ADX_SPLIT_STRONG_MIN:
        return RISK_PCT_STRONG        # 3% — strong trend
    elif adx <= ADX_SPLIT_EARLY_MAX:
        return RISK_PCT_EARLY_TREND   # 3% — early trend
    else:
        return RISK_PCT_TRANSITION    # 2% — dead zone


class VBSwingStrategy(BTCStrategy):
    """
    Two-strategy combiner for Asia Night + EU Open session.
    Kill-zone: hours [1, 2, 3, 8] UTC

    Volatility Breakout fires first (momentum), Swing Level second (structure).
    TP levels standardised to TP1=2R / TP2=5R for both sub-strategies —
    matching the backtested configuration that produced $20,354 PnL / 50.6% WR.
    """

    name        = "VB + Swing Level (Asia Night + EU Open)"
    description = "Volatility Breakout > Swing Level  |  01,02,03,08 UTC"

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

        TP levels are STANDARDISED here — sub-strategy defaults are overridden
        to TP1=2R / TP2=5R (validated by 2yr backtest).

        Extra keys returned:
          strategy_used   : name of sub-strategy that fired
          needs_im_filter : True only for Volatility Breakout
          tp1_rr          : always 2.0 (standardised)
          tp2_rr          : always 5.0 (standardised)
        """
        for strat in self._strategies:
            result = strat.generate_signal(df_window, bar_time, direction)
            if result.get("signal"):
                # Standardise TP levels — override sub-strategy defaults
                result["tp1_rr"]          = TP1_RR   # 2.0R
                result["tp2_rr"]          = TP2_RR   # 5.0R
                result["strategy_used"]   = strat.name
                result["needs_im_filter"] = strat.name in _IM_FILTER_STRATEGIES
                result["reason"]          = strat.name
                return result

        return {
            "signal":          False,
            "entry":           0.0,
            "sl":              0.0,
            "tp1_rr":          TP1_RR,
            "tp2_rr":          TP2_RR,
            "reason":          "no strategy fired",
            "strategy_used":   None,
            "needs_im_filter": False,
        }

    @property
    def strategy_names(self) -> list[str]:
        return [s.name for s in self._strategies]
