"""
btc_bot_2/strategy/vb_swing_combined.py — VB + Swing Level Break v2 for Asia Night.

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

== SWING LEVEL v2 — MODE 6 "both" 2×ATR  (FINAL CHOSEN CONFIG) ==

  Why upgrade from v1?
    v1 SL = prior swing structure = avg 4.42×ATR — enormous.
    Wide SL → tiny position size → trades stay open for days blocking new entries.
    PnL: $20,354 over 2yr starting from $500.

  v2 "both" mode:
    First checks for RETEST entry (level broken, price came back, bar rejects).
      SL = current bar extreme + tiny buffer → ~0.6×ATR (7× tighter).
    If no retest, fires on FIRST BREAK with SL CAPPED at 2×ATR.
      Structural SL is kept if it's already tighter.
    Result: trades resolve ~3-4× faster → more compounding → $89,362 (+4.4×).

  Full temporal analysis (2yr, all 7 modes):
    Mode 6: 5/5 halves ✅  9/9 quarters ✅  23/24 months ✅  Highest score 5.42

== SL PLACEMENT ==

  Volatility Breakout:
    Entry : bar close  (the large momentum bar itself)
    SL    : low of the breakout bar  (for longs)
            high of the breakout bar (for shorts)
    Typical SL distance: 0.5-1.0× ATR

  Swing Level Break v2 (Mode 6 "both"):
    Retest entry : SL = bar_low - 0.05×ATR  (long)  →  ~0.6×ATR average
    Break  entry : SL = max(prior_swing_struct, entry - 2×ATR)  →  ≤2×ATR

== TP LEVELS (STANDARDISED — matching backtest) ==

  VB:    TP1 = 2.0R  |  TP2 = 5.0R  (trailing SL 2×ATR after TP1)
  SLv2 retest: TP1 = 2.0R | TP2 = 5.0R  (tighter SL allows 2R TP1)
  SLv2 break:  TP1 = 1.5R | TP2 = 5.0R  (same as v1, overridden to 2R here)

  All overridden to TP1=2.0R / TP2=5.0R in this combiner — validated by backtest.

== RISK SIZING (ADX-SPLIT — applied by the signal engine, not here) ==

  ADX 20-25 (early trend):  3%  — market just starting to trend, clean entry
  ADX 25-40 (transition):   2%  — dead zone in Asia Night, conservative sizing
  ADX 40+   (strong trend): 3%  — powerful momentum, ride with more size

  Implemented in btc_bot_2/settings.py. The strategy returns entry/SL only.
"""
from __future__ import annotations

import pandas as pd
from btc_research.strategies.base                import BTCStrategy
from btc_research.strategies.volatility_breakout  import VolatilityBreakout
from btc_research.strategies.swing_level_v2       import SwingLevelBreakV2
from btc_research.btc_bot_2.settings import (
    TP1_RR, TP2_RR,
    ADX_SPLIT_EARLY_MAX, ADX_SPLIT_STRONG_MIN,
    RISK_PCT_EARLY_TREND, RISK_PCT_TRANSITION, RISK_PCT_STRONG,
    SWING_ENTRY_MODE, SWING_MAX_SL_ATR,
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

    PRIORITY: SwingLevelBreakV2 fires FIRST, VB fires as fallback.

    Why Swing first (validated by 2yr backtest):
      SwingV2 → 46.7% WR  +0.90R avg  $119,329 PnL  MaxDD 17.5%  PF 2.50
      VB first → 42.6% WR  +0.82R avg  $89,362  PnL  MaxDD 18.3%  PF 2.25
      Swing-first wins on ALL metrics: +33.5% more PnL, higher WR, lower MaxDD.

    Why the gap?
      On bars where BOTH strategies fire, VB-first was discarding the higher-quality
      Swing entry (47.3% WR, +1.14R on break entries) in favour of VB (41.8% WR).
      VB is now the backup — it fires only when no Swing setup is present on that bar.
      VB also becomes more selective as a result (56.2% WR when Swing-first).

    Swing Level uses Mode "both" 2×ATR — the FINAL chosen config:
      - Retest entry preferred (tight SL ~0.6×ATR): level broken, price comes back, bar rejects
      - Break entry fallback (SL capped at 2×ATR): first break of swing level
    TP levels standardised to TP1=2R / TP2=5R for both sub-strategies.
    """

    name        = "Swing Level v2 + VB (Asia Night + EU Open)"
    description = "SwingLevelBreak v2 [both 2xATR] > Volatility Breakout  |  01,02,03,08 UTC"

    def __init__(
        self,
        atr_multiplier: float = 1.2,
        close_zone:     float = 0.45,
        swing_entry_mode: str   = SWING_ENTRY_MODE,   # "both"
        swing_max_sl_atr: float = SWING_MAX_SL_ATR,   # 2.0
    ):
        # SwingLevelBreakV2 FIRST — higher WR and AvgR
        # VolatilityBreakout SECOND — fallback when no swing setup present
        self._strategies: list[BTCStrategy] = [
            SwingLevelBreakV2(entry_mode=swing_entry_mode, max_sl_atr=swing_max_sl_atr),
            VolatilityBreakout(atr_multiplier=atr_multiplier, close_zone=close_zone),
        ]

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        """
        Try SwingLevelBreakV2 first, then Volatility Breakout as fallback.
        Return first signal that fires.

        TP levels are STANDARDISED here — sub-strategy defaults are overridden
        to TP1=2R / TP2=5R (validated by 2yr backtest).

        Extra keys returned:
          strategy_used   : name of sub-strategy that fired
          needs_im_filter : True only for Volatility Breakout
          tp1_rr          : always 2.0 (standardised)
          tp2_rr          : always 5.0 (standardised)
          entry_type      : "break" | "retest" | None  (from SwingLevelBreakV2)
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
