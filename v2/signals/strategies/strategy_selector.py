"""
signals/strategies/strategy_selector.py — Picks the best strategy per instrument.

For each instrument, runs all applicable strategies in priority order.
Returns the highest-scoring signal that meets the per-instrument score threshold.

Instrument strategy priority:
  XAUUSD  → London Breakout, ICT Gold, Liq Sweep, SMC OB, FVG Fill
  GBPJPY  → London Breakout, SMC OB, Liq Sweep, EMA Trend, FVG Fill, NY Momentum
  WTI     → NY Momentum WTI (kill-zone), NY Momentum, FVG Fill, EMA Trend, Squeeze
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
from v2.signals.strategies.ny_momentum_wti import NYMomentumWTIStrategy

logger = logging.getLogger(__name__)

# Global fallback minimum score (strategy must score >= this to be tradeable)
MIN_STRATEGY_SCORE = 5.0

# Per-instrument overrides — tighter thresholds for volatile/harder instruments
_INSTRUMENT_MIN_SCORE: dict[str, float] = {
    "XAUUSD":  7.5,   # Gold: high volatility, news-driven — only high-conviction setups
    "GBPJPY":  7.0,   # JPY pairs: wide spreads, requires clear structure
    "WTI":     6.0,   # Oil: NY session strategy works well, slightly relaxed
    "NAS100":  7.0,   # Indices: NY only, require strong momentum confirmation
    "BTCUSDT": 6.0,   # Crypto: 24/7, cipher strategy filters well
    "ETHUSDT": 6.0,
}

# Per-instrument strategy priority lists (highest priority first)
# XAUUSD: London Breakout moved to #1 (clean, session-bounded, proven)
#         Removed EMATrend + SqueezeBreakout (too generic for gold volatility)
_INSTRUMENT_STRATEGIES: dict[str, list[StrategyBase]] = {
    "XAUUSD": [
        LondonBreakoutStrategy(),    # Primary — Asian range breakout, clean rules
        ICTGoldStrategy(),            # Secondary — full ICT model (fixed)
        LiquiditySweepStrategy(),     # Tertiary — sweep + CHoCH
        SMCOrderBlockStrategy(),      # Quaternary — OB retest
        FVGFillStrategy(),            # Fallback — FVG fill
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
        NYMomentumWTIStrategy(),   # ONLY strategy — London kill-zone breakout + retest
        # FVG Fill removed: it outscores the kill-zone strategy and then fails R:R.
        # WTI is purely a kill-zone instrument — no off-session fallback.
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
        and score >= per-instrument minimum. Returns None if nothing qualifies.
        """
        strategies = _INSTRUMENT_STRATEGIES.get(symbol.upper(), [])
        if not strategies:
            logger.debug("No strategies configured for %s", symbol)
            return None

        min_score = _INSTRUMENT_MIN_SCORE.get(symbol.upper(), MIN_STRATEGY_SCORE)

        results: list[StrategyResult] = []
        for strat in strategies:
            try:
                result = strat.evaluate(symbol, direction, df_h1, df_h4, df_d1, context)
                if result.signal and result.score >= min_score:
                    results.append(result)
                    logger.debug(
                        "%s %s — %s: signal=True score=%.1f (threshold=%.1f)",
                        symbol, direction, strat.name, result.score, min_score,
                    )
                else:
                    logger.debug(
                        "%s %s — %s: blocked_by=%s score=%.1f (threshold=%.1f)",
                        symbol, direction, strat.name,
                        result.blocked_by or "score too low", result.score, min_score,
                    )
            except Exception as exc:
                logger.error("Strategy %s error for %s %s: %s",
                             strat.name, symbol, direction, exc, exc_info=True)

        if not results:
            return None

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
        """Run both directions and return (long_result, short_result)."""
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
