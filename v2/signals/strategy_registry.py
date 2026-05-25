"""
signals/strategy_registry.py — Unified strategy list for TradingBotV2.

Each strategy is a dataclass describing what conditions it needs,
what instruments it applies to, and its historical performance metadata.
The ConfluenceEngine uses this registry to match strategies to live signals.

Top 15 strategies to start — expand when bot is stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Direction = Literal["long", "short", "both"]
TimeFrame = Literal["M15", "H1", "H4", "D1"]


@dataclass(frozen=True)
class Strategy:
    name:         str
    direction:    Direction
    timeframes:   list[TimeFrame]
    instruments:  list[str]     # empty list = all instruments
    description:  str

    # Historical performance targets (updated by ML layer over time)
    target_wr:    float = 0.0   # target win rate %
    target_rr:    float = 2.0   # minimum R:R
    min_score:    float = 3.0   # minimum confluence score to fire

    # Which confluence factors this strategy requires
    requires:     list[str] = field(default_factory=list)


STRATEGIES: dict[str, Strategy] = {

    # ── Trend strategies ──────────────────────────────────────────────────────

    "ema_trend": Strategy(
        name        = "EMA Trend Continuation",
        direction   = "both",
        timeframes  = ["H1", "H4"],
        instruments = [],
        description = "Price pulling back to EMA in trending market, bounce continuation",
        target_wr   = 55.0,
        target_rr   = 2.0,
        requires    = ["adx_trending", "supertrend_aligned", "pullback_to_ema"],
    ),

    "ichimoku_breakout": Strategy(
        name        = "Ichimoku Cloud Breakout",
        direction   = "both",
        timeframes  = ["H4", "D1"],
        instruments = [],
        description = "Price breaks above/below Ichimoku cloud with TK cross confirmation",
        target_wr   = 52.0,
        target_rr   = 2.5,
        requires    = ["above_cloud", "tk_cross", "chikou_clear"],
    ),

    "macd_crossover": Strategy(
        name        = "MACD Signal Cross",
        direction   = "both",
        timeframes  = ["H1", "H4"],
        instruments = [],
        description = "MACD crosses signal line with histogram expanding",
        target_wr   = 50.0,
        target_rr   = 2.0,
        requires    = ["macd_cross", "histogram_expanding"],
    ),

    # ── Reversal strategies ───────────────────────────────────────────────────

    "rsi_oversold_bounce": Strategy(
        name        = "RSI Oversold Bounce",
        direction   = "long",
        timeframes  = ["H1", "H4"],
        instruments = ["XAUUSD", "BTCUSDT", "ETHUSDT", "NAS100"],
        description = "StochRSI oversold + bullish candle pattern at key support",
        target_wr   = 58.0,
        target_rr   = 2.0,
        requires    = ["stoch_rsi_oversold", "support_level", "bullish_candle"],
    ),

    "rsi_overbought_reversal": Strategy(
        name        = "RSI Overbought Reversal",
        direction   = "short",
        timeframes  = ["H1", "H4"],
        instruments = ["XAUUSD", "BTCUSDT", "ETHUSDT", "NAS100"],
        description = "StochRSI overbought + bearish candle pattern at key resistance",
        target_wr   = 56.0,
        target_rr   = 2.0,
        requires    = ["stoch_rsi_overbought", "resistance_level", "bearish_candle"],
    ),

    "market_cipher_cross": Strategy(
        name        = "Market Cipher Cross",
        direction   = "both",
        timeframes  = ["H1", "H4"],
        instruments = ["BTCUSDT", "ETHUSDT"],
        description = "Market Cipher B WT cross from OB/OS zone with green/red dot",
        target_wr   = 54.0,
        target_rr   = 2.0,
        requires    = ["mc_cross", "mc_extreme_zone"],
    ),

    # ── Smart Money strategies ────────────────────────────────────────────────

    "smc_order_block": Strategy(
        name        = "SMC Order Block Retest",
        direction   = "both",
        timeframes  = ["H1", "H4"],
        instruments = ["XAUUSD", "GBPJPY"],
        description = "Price returns to institutional order block after BOS",
        target_wr   = 60.0,
        target_rr   = 2.5,
        requires    = ["order_block_active", "bos_confirmed", "near_order_block"],
    ),

    "fvg_fill": Strategy(
        name        = "Fair Value Gap Fill",
        direction   = "both",
        timeframes  = ["H1", "H4"],
        instruments = [],
        description = "Price returns to fill open FVG imbalance",
        target_wr   = 57.0,
        target_rr   = 2.0,
        requires    = ["fvg_open", "price_at_fvg"],
    ),

    "liquidity_sweep": Strategy(
        name        = "Liquidity Sweep + Reversal",
        direction   = "both",
        timeframes  = ["H1", "H4"],
        instruments = ["XAUUSD", "BTCUSDT", "GBPJPY"],
        description = "Stop hunt sweeps equal highs/lows then reverses (CHoCH)",
        target_wr   = 62.0,
        target_rr   = 3.0,
        requires    = ["liquidity_sweep_detected", "choch_formed"],
    ),

    # ── Session / volatility strategies ──────────────────────────────────────

    "london_breakout": Strategy(
        name        = "London Open Breakout",
        direction   = "both",
        timeframes  = ["M15", "H1"],
        instruments = ["XAUUSD", "GBPJPY"],
        description = "Asian range breakout at London open with volume expansion",
        target_wr   = 55.0,
        target_rr   = 2.0,
        requires    = ["london_session", "asian_range_defined", "breakout_candle"],
    ),

    "news_spike_fade": Strategy(
        name        = "News Spike Fade",
        direction   = "both",
        timeframes  = ["M15", "H1"],
        instruments = ["XAUUSD", "GBPJPY"],
        description = "Fade overextended spike after high-impact news event",
        target_wr   = 53.0,
        target_rr   = 2.0,
        min_score   = 4.0,
        requires    = ["post_news_spike", "spike_overextended", "reversal_candle"],
    ),

    "squeeze_breakout": Strategy(
        name        = "Squeeze Momentum Breakout",
        direction   = "both",
        timeframes  = ["H1", "H4"],
        instruments = [],
        description = "BB inside KC squeeze fires — enter on squeeze off with momentum",
        target_wr   = 52.0,
        target_rr   = 2.5,
        requires    = ["squeeze_fired", "momentum_positive"],
    ),

    # ── Commodity-specific ────────────────────────────────────────────────────

    "gold_dxy_divergence": Strategy(
        name        = "Gold / DXY Divergence",
        direction   = "long",
        timeframes  = ["H4", "D1"],
        instruments = ["XAUUSD"],
        description = "DXY weakening while gold shows bullish structure",
        target_wr   = 58.0,
        target_rr   = 2.5,
        requires    = ["dxy_falling", "gold_bullish_structure"],
    ),

    "crypto_funding_fade": Strategy(
        name        = "Crypto Funding Rate Fade",
        direction   = "both",
        timeframes  = ["H1", "H4"],
        instruments = ["BTCUSDT", "ETHUSDT"],
        description = "Extreme positive/negative funding rate fade — crowded trade reversal",
        target_wr   = 54.0,
        target_rr   = 2.0,
        requires    = ["extreme_funding_rate", "price_at_resistance_or_support"],
    ),

    "wyckoff_spring": Strategy(
        name        = "Wyckoff Spring / UTAD",
        direction   = "both",
        timeframes  = ["H4", "D1"],
        instruments = ["XAUUSD", "BTCUSDT", "NAS100"],
        description = "Wyckoff spring (accumulation) or UTAD (distribution) test",
        target_wr   = 60.0,
        target_rr   = 3.0,
        requires    = ["wyckoff_accumulation_or_distribution", "spring_test"],
    ),
}


def get_strategy(name: str) -> Strategy | None:
    """Return strategy by key name (case-insensitive). None if not found."""
    return STRATEGIES.get(name) or STRATEGIES.get(name.lower().replace(" ", "_"))


def get_strategies_for(
    symbol: str,
    direction: str = "both",
    timeframe: str = "H1",
) -> list[Strategy]:
    """Return all strategies applicable to this instrument/direction/timeframe."""
    results = []
    for strat in STRATEGIES.values():
        if strat.instruments and symbol not in strat.instruments:
            continue
        if strat.direction != "both" and direction != "both" and strat.direction != direction:
            continue
        if timeframe not in strat.timeframes:
            continue
        results.append(strat)
    return results


ALL_STRATEGY_NAMES = list(STRATEGIES.keys())
