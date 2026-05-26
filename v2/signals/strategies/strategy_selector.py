"""
signals/strategies/strategy_selector.py — Picks the best strategy per instrument.

For each instrument, runs all applicable strategies in priority order.
Returns the highest-scoring signal or None if no strategy fires.

Instrument strategy priority:
  XAUUSD  → ICT Gold, Liq Sweep, SMC OB, London Breakout, FVG Fill, EMA Trend, Squeeze
  GBPJPY  → London Breakout, SMC OB, Liq Sweep, EMA Trend, FVG Fill, NY Momentum
  WTI     → NY Momentum, FVG Fill, EMA Trend, Squeeze
  NAS100  → NY Momentum, Squeeze, EMA Trend, FVG Fill
  BTCUSDT → Crypto Cipher, Liq Sweep, FVG Fill, EMA Trend
  ETHUSDT → Crypto Cipher, Liq Sweep, FVG Fill, EMA Trend
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from v2.signals.strategies.base import StrategyBase, StrategyResult
from v2.signals.strategies.ict_gold        import ICTGoldStrategy
from v2.signals.strategies.smc_order_block import SMCOrderBlockStrategy
from v2.signals.strategies.london_breakout import LondonBreakoutStrategy
from v2.signals.strategies.liquidity_sweep import LiquiditySweepStrategy
from v2.signals.strategies.fvg_fill        import FVGFillStrategy
from v2.signals.strategies.ema_trend       import EMATrendStrategy
from v2.signals.strategies.ny_momentum     import NYMomentumStrategy
from v2.signals.strategies.squeeze_breakout import SqueezeBreakoutStrategy
from v2.signals.strategies.crypto_cipher   import CryptoCipherStrategy

logger = logging.getLogger(__name__)

# Minimum score for a strategy result to be considered tradeable
MIN_STRATEGY_SCORE = 5.0

# Per-instrument strategy priority lists (highest priority first)
_INSTRUMENT_STRATEGIES: dict[str, list[StrategyBase]] = {
    "XAUUSD": [
        ICTGoldStrategy(),
        LiquiditySweepStrategy(),
        SMCOrderBlockStrategy(),
        LondonBreakoutStrategy(),
        FVGFillStrategy(),
        EMATrendStrategy(),
        SqueezeBreakoutStrategy(),
    ],
    "GBPJPY": [
        LondonBreakoutStrategy(),
        SMCOrderBlockStrategy(),
        LiquiditySweepStrategy(),
        EMATrendStrategy(),
        FVGFillStrategy(),
        NYMomentumStrategy(),
    ],
    "WTI": [
        NYMomentumStrategy(),
        FVGFillStrategy(),
        EMATrendStrategy(),
        SqueezeBreakoutStrategy(),
    ],
    "NAS100": [
        NYMomentumStrategy(),
        SqueezeBreakoutStrategy(),
        EMATrendStrategy(),
        FVGFillStrategy(),
    ],
    "BTCUSDT": [
        CryptoCipherStrategy(),
        LiquiditySweepStrategy(),
        FVGFillStrategy(),
        EMATrendStrategy(),
    ],
    "ETHUSDT": [
        CryptoCipherStrategy(),
        LiquiditySweepStrategy(),
        FVGFillStrategy(),
        EMATrendStrategy(),
    ],
}


class StrategySelector:
    """
    Runs all applicable strategies for a symbol and returns the best signal.

    Usage:
        selector = StrategySelector()
        result = selector.select(symbol, direction, df_h1, df_h4, df_d1)
        if result and result.signal:
            open_trade(result.to_signal_dict())
    """

    def select(
        self,
        symbol:    str,
        direction: str,
        df_h1:     pd.DataFrame,
        df_h4:     pd.DataFrame | None = None,
        df_d1:     pd.DataFrame | None = None,
        context:   dict | None = None,
    ) -> StrategyResult | None:
        """
        Run all applicable strategies for the symbol and return the best one.

        Returns the StrategyResult with the highest score that has signal=True
        and score >= MIN_STRATEGY_SCORE. Returns None if nothing qualifies.
        """
        strategies = _INSTRUMENT_STRATEGIES.get(symbol.upper(), [])
        if not strategies:
            logger.debug("No strategies configured for %s", symbol)
            return None

        results: list[StrategyResult] = []
        for strat in strategies:
            try:
                result = strat.evaluate(symbol, direction, df_h1, df_h4, df_d1, context)
                if result.signal and result.score >= MIN_STRATEGY_SCORE:
                    results.append(result)
                    logger.debug(
                        "%s %s — %s: signal=True score=%.1f",
                        symbol, direction, strat.name, result.score,
                    )
                else:
                    logger.debug(
                        "%s %s — %s: blocked_by=%s score=%.1f",
                        symbol, direction, strat.name,
                        result.blocked_by or "score too low", result.score,
                    )
            except Exception as exc:
                logger.error("Strategy %s error for %s %s: %s",
                             strat.name, symbol, direction, exc, exc_info=True)

        if not results:
            return None

        # Return highest-scoring result
        best = max(results, key=lambda r: r.score)
        logger.info(
            "BEST STRATEGY %s %s: %s score=%.1f entry=%.5f sl=%.5f",
            symbol, direction, best.strategy_name, best.score,
            best.entry_price, best.stop_loss,
        )
        return best

    def select_all_directions(
        self,
        symbol:    str,
        df_h1:     pd.DataFrame,
        df_h4:     pd.DataFrame | None = None,
        df_d1:     pd.DataFrame | None = None,
        context:   dict | None = None,
    ) -> tuple[StrategyResult | None, StrategyResult | None]:
        """
        Run both directions and return (long_result, short_result).
        Only returns results that qualify (signal=True, score >= MIN).
        """
        long_result  = self.select(symbol, "long",  df_h1, df_h4, df_d1, context)
        short_result = self.select(symbol, "short", df_h1, df_h4, df_d1, context)
        return long_result, short_result

    def get_all_results(
        self,
        symbol:    str,
        direction: str,
        df_h1:     pd.DataFrame,
        df_h4:     pd.DataFrame | None = None,
        df_d1:     pd.DataFrame | None = None,
        context:   dict | None = None,
    ) -> list[StrategyResult]:
        """Return ALL strategy results (including blocked ones) for diagnostics."""
        strategies = _INSTRUMENT_STRATEGIES.get(symbol.upper(), [])
        results = []
        for strat in strategies:
            try:
                results.append(strat.evaluate(symbol, direction, df_h1, df_h4, df_d1, context))
            except Exception as exc:
                logger.error("Strategy %s error: %s", strat.name, exc)
        return results
