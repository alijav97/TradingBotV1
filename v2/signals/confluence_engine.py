"""
signals/confluence_engine.py — Signal engine for TradingBotV2.

Primary path: StrategySelector — each instrument has dedicated strategy modules
  that implement real trading logic (ICT Gold, London Breakout, SMC OB, etc.)

Fallback path: the original 12-factor generic scorer fires when no strategy
  produces a qualifying signal (score >= min_score).

Usage:
    from v2.signals.confluence_engine import ConfluenceEngine
    engine = ConfluenceEngine()
    result = engine.score(symbol, direction, df_h1, df_h4, df_d1, context)
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from v2.analysis.indicators import get_all_indicators
from v2.analysis.candle_patterns import detect_patterns
from v2.signals.strategies.strategy_selector import StrategySelector

logger = logging.getLogger(__name__)

# Minimum score to generate a tradeable signal
# Matches entry_checklist.py MIN_CONFLUENCE = 7.0 so fallback and checklist are consistent
MIN_SCORE = 7.0

_selector = StrategySelector()


class ConfluenceEngine:
    """
    Signal engine with two paths:
      1. Strategy selector (primary) — specific logic per instrument
      2. 12-factor generic scorer (fallback)

    score() always returns the same dict shape regardless of which path fired.
    """

    def __init__(self, min_score: float = MIN_SCORE) -> None:
        self.min_score = min_score

    def score(
        self,
        symbol: str,
        direction: str,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame | None = None,
        df_d1: pd.DataFrame | None = None,
        context: dict | None = None,
    ) -> dict:
        """
        Run all 12 confluence factors and return a full scoring report.

        Parameters
        ----------
        symbol    : instrument symbol
        direction : "long" | "short"
        df_h1     : H1 OHLCV DataFrame (required)
        df_h4     : H4 OHLCV DataFrame (optional — boosts score if aligned)
        df_d1     : D1 OHLCV DataFrame (optional — boosts score if aligned)
        context   : extra info dict — can include:
                      news_blocked, dxy, vix, session, regime,
                      geo_score, cot_bias, funding_rate

        Returns
        -------
        {
            "symbol":     str,
            "direction":  str,
            "score":      float,
            "max_score":  12,
            "signal":     bool,   # True if score >= min_score
            "factors":    dict,   # per-factor breakdown
            "reasons":    list,   # human-readable reasons
            "entry":      float,  # suggested entry (last close)
            "stop_loss":  float,  # ATR-based SL
            "strategy":   str,    # best matching strategy name
        }
        """
        if context is None:
            context = {}

        # ── PRIMARY PATH: Strategy Selector ──────────────────────────────────
        try:
            strat_result = _selector.select(symbol, direction, df_h1, df_h4, df_d1, context)
            if strat_result and strat_result.signal and strat_result.score >= self.min_score:
                logger.info(
                    "Strategy signal: %s %s %s score=%.1f entry=%.5f",
                    symbol, direction, strat_result.strategy_name,
                    strat_result.score, strat_result.entry_price,
                )
                return {
                    "symbol":      symbol,
                    "direction":   direction,
                    "score":       strat_result.score,
                    "max_score":   10,
                    "signal":      True,
                    "factors":     strat_result.factors,
                    "reasons":     strat_result.reasons,
                    "entry_price": strat_result.entry_price,
                    "stop_loss":   strat_result.stop_loss,
                    "tp1_price":   strat_result.tp1_price,
                    "tp2_price":   strat_result.tp2_price,
                    "strategy":    strat_result.strategy_name,
                    "timeframe":   "H1",
                    "signal_path": "strategy",
                }
        except Exception as exc:
            logger.error("StrategySelector error for %s %s: %s", symbol, direction, exc, exc_info=True)

        # ── FALLBACK PATH: 12-factor generic scorer ───────────────────────────
        # For instruments with dedicated strategies, only use the strategy path.
        # The generic scorer does not understand instrument-specific nuances and
        # produces lower-quality signals that drag down overall WR.
        from v2.signals.strategies.strategy_selector import _INSTRUMENT_MIN_SCORE
        _STRATEGY_ONLY = {"XAUUSD", "BTCUSDT", "ETHUSDT", "WTI"}
        if symbol.upper() in _STRATEGY_ONLY:
            return {"symbol": symbol, "direction": direction, "score": 0.0,
                    "max_score": 10, "signal": False, "factors": {}, "reasons": [],
                    "entry_price": 0.0, "stop_loss": 0.0, "tp1_price": 0.0,
                    "tp2_price": 0.0, "strategy": "", "timeframe": "H1",
                    "signal_path": "strategy_only_blocked"}
        is_long = direction.lower() in ("long", "buy")
        factors: dict[str, dict] = {}
        total_score = 0.0
        reasons: list[str] = []

        # ── Run all indicators on H1 ──────────────────────────────────────────
        inds_h1 = {}
        try:
            inds_h1 = get_all_indicators(df_h1)
        except Exception as exc:
            logger.warning("Indicators failed for %s: %s", symbol, exc)

        # ── Factor 1: Trend direction (ADX + Supertrend) ──────────────────────
        f1, r1 = self._factor_trend(inds_h1, direction)
        factors["trend"]      = {"score": f1, "reason": r1}
        total_score += f1
        if r1: reasons.append(r1)

        # ── Factor 2: Momentum (MACD + Market Cipher) ─────────────────────────
        f2, r2 = self._factor_momentum(inds_h1, direction)
        factors["momentum"]   = {"score": f2, "reason": r2}
        total_score += f2
        if r2: reasons.append(r2)

        # ── Factor 3: Oscillator (StochRSI) ──────────────────────────────────
        f3, r3 = self._factor_oscillator(inds_h1, direction)
        factors["oscillator"] = {"score": f3, "reason": r3}
        total_score += f3
        if r3: reasons.append(r3)

        # ── Factor 4: Ichimoku cloud ──────────────────────────────────────────
        f4, r4 = self._factor_ichimoku(inds_h1, direction)
        factors["ichimoku"]   = {"score": f4, "reason": r4}
        total_score += f4
        if r4: reasons.append(r4)

        # ── Factor 5: Volatility / squeeze ───────────────────────────────────
        f5, r5 = self._factor_squeeze(inds_h1)
        factors["squeeze"]    = {"score": f5, "reason": r5}
        total_score += f5
        if r5: reasons.append(r5)

        # ── Factor 6: Kill zone / session ─────────────────────────────────────
        f6, r6 = self._factor_killzone(inds_h1, context)
        factors["killzone"]   = {"score": f6, "reason": r6}
        total_score += f6
        if r6: reasons.append(r6)

        # ── Factor 7: Candle pattern ──────────────────────────────────────────
        f7, r7 = self._factor_candle_pattern(df_h1, direction)
        factors["candle"]     = {"score": f7, "reason": r7}
        total_score += f7
        if r7: reasons.append(r7)

        # ── Factor 8: HTF alignment (H4 + D1) ────────────────────────────────
        f8, r8 = self._factor_htf_alignment(direction, df_h4, df_d1)
        factors["htf"]        = {"score": f8, "reason": r8}
        total_score += f8
        if r8: reasons.append(r8)

        # ── Factor 9: Volume (OBV) ────────────────────────────────────────────
        f9, r9 = self._factor_volume(inds_h1, direction)
        factors["volume"]     = {"score": f9, "reason": r9}
        total_score += f9
        if r9: reasons.append(r9)

        # ── Factor 10: News / event block ────────────────────────────────────
        f10, r10 = self._factor_news(context)
        factors["news"]       = {"score": f10, "reason": r10}
        total_score += f10
        if r10: reasons.append(r10)

        # ── Factor 11: DXY / macro context ───────────────────────────────────
        f11, r11 = self._factor_macro(inds_h1, direction, context, symbol)
        factors["macro"]      = {"score": f11, "reason": r11}
        total_score += f11
        if r11: reasons.append(r11)

        # ── Factor 12: Wyckoff / market regime ───────────────────────────────
        f12, r12 = self._factor_regime(inds_h1, direction, context)
        factors["regime"]     = {"score": f12, "reason": r12}
        total_score += f12
        if r12: reasons.append(r12)

        # ── Entry / SL suggestion ─────────────────────────────────────────────
        entry_price, sl_price = self._suggest_entry_sl(df_h1, direction)

        # ── Best matching strategy ────────────────────────────────────────────
        strategy = self._match_strategy(symbol, direction, factors)

        # ── Hard gate: at least one HTF timeframe (H4 or D1) must not
        #    contradict direction — avoids trading against the higher TF trend
        htf_score = factors["htf"]["score"]
        htf_available = (df_h4 is not None and not df_h4.empty) or \
                        (df_d1 is not None and not df_d1.empty)
        htf_blocked = htf_available and htf_score == 0

        signal_fires = (total_score >= self.min_score) and (not htf_blocked)

        logger.debug(
            "%s %s score=%.1f/12 htf=%.1f signal=%s",
            symbol, direction, total_score, htf_score, signal_fires
        )

        # Fallback TP calculation (2:1 and 4:1 RR)
        sl_dist = abs(entry_price - sl_price) if entry_price and sl_price else 0
        is_long_fb = direction.lower() in ("long", "buy")
        tp1_fb = round(entry_price + sl_dist * 2, 5) if is_long_fb else round(entry_price - sl_dist * 2, 5)
        tp2_fb = round(entry_price + sl_dist * 4, 5) if is_long_fb else round(entry_price - sl_dist * 4, 5)

        return {
            "symbol":      symbol,
            "direction":   direction,
            "score":       round(total_score, 1),
            "max_score":   12,
            "signal":      signal_fires,
            "factors":     factors,
            "reasons":     reasons,
            "entry_price": entry_price,
            "stop_loss":   sl_price,
            "tp1_price":   tp1_fb,
            "tp2_price":   tp2_fb,
            "strategy":    strategy,
            "timeframe":   "H1",
            "signal_path": "fallback_12factor",
        }

    # ── Scoring factors ───────────────────────────────────────────────────────

    def _factor_trend(self, inds: dict, direction: str) -> tuple[float, str]:
        score = 0.0
        is_long = direction.lower() in ("long", "buy")
        adx = inds.get("adx", {})
        st  = inds.get("supertrend", {})
        al  = inds.get("alligator", {})

        if adx.get("trending") and adx.get("bias") == ("bullish" if is_long else "bearish"):
            score += 0.5
        if st.get("bias") == ("bullish" if is_long else "bearish"):
            score += 0.5
        if al.get("bias") == ("bullish" if is_long else "bearish") and not al.get("sleeping"):
            score += 0.0  # bonus only if others align

        if score > 0:
            return min(score, 1.0), f"Trend aligned: ADX={adx.get('adx', 0):.0f}, ST={st.get('trend','')}"
        return 0.0, ""

    def _factor_momentum(self, inds: dict, direction: str) -> tuple[float, str]:
        is_long = direction.lower() in ("long", "buy")
        macd = inds.get("macd", {})
        mc   = inds.get("market_cipher", {})

        score = 0.0
        macd_ok = macd.get("bias", "") in ("bullish", "strongly_bullish") if is_long else macd.get("bias", "") in ("bearish", "strongly_bearish")
        mc_ok   = mc.get("bias", "") in ("bullish", "strongly_bullish") if is_long else mc.get("bias", "") in ("bearish", "strongly_bearish")

        if macd_ok: score += 0.5
        if mc_ok:   score += 0.5
        if (macd.get("bullish_cross") and is_long) or (macd.get("bearish_cross") and not is_long):
            score = min(score + 0.5, 1.0)

        if score > 0:
            return min(score, 1.0), f"Momentum {'bullish' if is_long else 'bearish'}: MACD={macd.get('bias','')}"
        return 0.0, ""

    def _factor_oscillator(self, inds: dict, direction: str) -> tuple[float, str]:
        is_long = direction.lower() in ("long", "buy")
        srsi = inds.get("stoch_rsi", {})
        k    = srsi.get("k", 50)

        if is_long and srsi.get("oversold") and srsi.get("bullish_cross"):
            return 1.0, f"StochRSI oversold ({k:.0f}) with bullish cross"
        if is_long and srsi.get("oversold"):
            return 0.5, f"StochRSI oversold ({k:.0f})"
        if not is_long and srsi.get("overbought") and srsi.get("bearish_cross"):
            return 1.0, f"StochRSI overbought ({k:.0f}) with bearish cross"
        if not is_long and srsi.get("overbought"):
            return 0.5, f"StochRSI overbought ({k:.0f})"
        return 0.0, ""

    def _factor_ichimoku(self, inds: dict, direction: str) -> tuple[float, str]:
        is_long = direction.lower() in ("long", "buy")
        ichi = inds.get("ichimoku", {})
        bias = ichi.get("bias", "neutral")

        if is_long and ichi.get("above_cloud") and bias in ("bullish", "strongly_bullish"):
            return 1.0, "Price above Ichimoku cloud — bullish structure"
        if is_long and bias == "bullish":
            return 0.5, "Ichimoku bullish bias"
        if not is_long and ichi.get("below_cloud") and bias in ("bearish", "strongly_bearish"):
            return 1.0, "Price below Ichimoku cloud — bearish structure"
        if not is_long and bias == "bearish":
            return 0.5, "Ichimoku bearish bias"
        return 0.0, ""

    def _factor_squeeze(self, inds: dict) -> tuple[float, str]:
        sq = inds.get("squeeze", {})
        if sq.get("squeeze_off"):
            return 1.0, "Bollinger/Keltner squeeze just fired — momentum release"
        if sq.get("squeeze_on"):
            return 0.5, "Squeeze building — breakout pending"
        return 0.0, ""

    def _factor_killzone(self, inds: dict, context: dict) -> tuple[float, str]:
        kz      = inds.get("killzones", {})
        session = context.get("session", "")

        if kz.get("high_quality"):
            zones = ", ".join(kz.get("active_zones", []))
            return 1.0, f"ICT Kill Zone active: {zones}"
        if kz.get("in_killzone"):
            return 0.5, "In trading session window"
        if session in ("London", "NewYork", "LondonNY"):
            return 0.5, f"Active session: {session}"
        return 0.0, ""

    def _factor_candle_pattern(self, df: pd.DataFrame, direction: str) -> tuple[float, str]:
        try:
            patterns = detect_patterns(df)
            is_long  = direction.lower() in ("long", "buy")
            score    = patterns.get("score", 0)
            if is_long and score > 0:
                return min(score * 0.5, 1.0), f"Candle pattern: {patterns.get('strongest', '')}"
            if not is_long and score < 0:
                return min(abs(score) * 0.5, 1.0), f"Candle pattern: {patterns.get('strongest', '')}"
        except Exception:
            pass
        return 0.0, ""

    def _factor_htf_alignment(
        self,
        direction: str,
        df_h4: pd.DataFrame | None,
        df_d1: pd.DataFrame | None,
    ) -> tuple[float, str]:
        score = 0.0
        is_long = direction.lower() in ("long", "buy")
        aligned_tfs: list[str] = []

        for tf, df in [("H4", df_h4), ("D1", df_d1)]:
            if df is None or df.empty:
                continue
            try:
                inds = get_all_indicators(df)
                st   = inds.get("supertrend", {})
                macd = inds.get("macd", {})
                tf_bullish = st.get("bias") == "bullish" and macd.get("bias") in ("bullish", "strongly_bullish")
                tf_bearish = st.get("bias") == "bearish" and macd.get("bias") in ("bearish", "strongly_bearish")
                if (is_long and tf_bullish) or (not is_long and tf_bearish):
                    score += 0.5
                    aligned_tfs.append(tf)
            except Exception:
                pass

        if score > 0:
            return min(score, 1.0), f"HTF aligned ({', '.join(aligned_tfs)}) with {direction}"
        return 0.0, ""

    def _factor_volume(self, inds: dict, direction: str) -> tuple[float, str]:
        is_long = direction.lower() in ("long", "buy")
        obv = inds.get("obv", {})
        if is_long and obv.get("bias") == "bullish":
            div = obv.get("divergence")
            if div == "bullish_divergence":
                return 1.0, "Bullish OBV divergence — accumulation"
            return 0.5, "OBV trending bullish"
        if not is_long and obv.get("bias") == "bearish":
            div = obv.get("divergence")
            if div == "bearish_divergence":
                return 1.0, "Bearish OBV divergence — distribution"
            return 0.5, "OBV trending bearish"
        return 0.0, ""

    def _factor_news(self, context: dict) -> tuple[float, str]:
        if context.get("news_blocked"):
            return 0.0, ""  # signal still scored — entry checklist will block it
        if context.get("news_favorable"):
            return 0.5, "News sentiment supportive"
        return 0.0, ""

    def _factor_macro(self, inds: dict, direction: str, context: dict, symbol: str) -> tuple[float, str]:
        is_long = direction.lower() in ("long", "buy")

        # Real rate model (relevant for gold/crypto)
        if symbol in ("XAUUSD", "BTCUSDT", "ETHUSDT"):
            rr = inds.get("real_rate", {})
            rr_bias = rr.get("bias", "neutral")
            if is_long and rr_bias in ("bullish", "strongly_bullish"):
                return 1.0, f"Real rate model bullish: {rr.get('note', '')}"
            if not is_long and rr_bias in ("bearish", "strongly_bearish"):
                return 1.0, f"Real rate model bearish: {rr.get('note', '')}"
            if rr.get("available"):
                return 0.0, ""

        # DXY correlation
        dxy_falling = context.get("dxy_falling", False)
        dxy_rising  = context.get("dxy_rising", False)
        if symbol == "XAUUSD" and is_long and dxy_falling:
            return 1.0, "DXY falling → gold bullish bias"
        if symbol == "XAUUSD" and not is_long and dxy_rising:
            return 1.0, "DXY rising → gold bearish bias"

        return 0.0, ""

    def _factor_regime(self, inds: dict, direction: str, context: dict) -> tuple[float, str]:
        is_long = direction.lower() in ("long", "buy")
        wyck    = inds.get("wyckoff", {})
        regime  = context.get("regime", "")

        if is_long and wyck.get("bias") == "bullish":
            return 1.0, f"Wyckoff: {wyck.get('phase', '')} phase"
        if not is_long and wyck.get("bias") == "bearish":
            return 1.0, f"Wyckoff: {wyck.get('phase', '')} phase"
        if regime and regime.upper() in ("TRENDING_STRONG", "TRENDING"):
            return 0.5, f"Regime: {regime}"
        return 0.0, ""

    # ── Entry / SL suggestion ─────────────────────────────────────────────────

    def _suggest_entry_sl(self, df: pd.DataFrame, direction: str) -> tuple[float, float]:
        """Suggest entry (last close) and ATR-based SL."""
        try:
            entry = float(df["close"].iloc[-1])
            hl    = df["high"] - df["low"]
            hc    = (df["high"] - df["close"].shift()).abs()
            lc    = (df["low"]  - df["close"].shift()).abs()
            atr   = float(pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean().iloc[-1])

            is_long = direction.lower() in ("long", "buy")
            sl = round(entry - atr * 1.5, 5) if is_long else round(entry + atr * 1.5, 5)
            return round(entry, 5), sl
        except Exception:
            return 0.0, 0.0

    def _match_strategy(self, symbol: str, direction: str, factors: dict) -> str:
        """Return the best matching strategy name based on active factors."""
        from v2.signals.strategy_registry import get_strategies_for
        candidates = get_strategies_for(symbol, direction, "H1")
        if not candidates:
            return "general"
        # Simple heuristic — pick first candidate (ML layer will improve this)
        return candidates[0].name
