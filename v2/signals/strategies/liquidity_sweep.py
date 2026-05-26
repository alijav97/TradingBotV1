"""
strategies/liquidity_sweep.py — Liquidity Sweep + Reversal (CHoCH) strategy.

Fires when institutions hunt stops then reverse:
  1. Equal highs / lows recently swept (stop hunt)
  2. Price reversed back inside the range (trap candle)
  3. CHoCH formed on H1 confirming the reversal
  4. Enter in the direction of the reversal

Works on all instruments. Strongest signal on XAUUSD, GBPJPY, BTCUSDT.
"""
from __future__ import annotations

import logging

import pandas as pd

from v2.signals.strategies.base import StrategyBase, StrategyResult
from v2.analysis.smart_money import SmartMoneyAnalyzer

logger = logging.getLogger(__name__)
_SMC = SmartMoneyAnalyzer()


class LiquiditySweepStrategy(StrategyBase):
    name        = "liquidity_sweep"
    instruments = []   # all instruments
    timeframes  = ["H1"]
    min_df_bars = 50

    def evaluate(
        self,
        symbol:    str,
        direction: str,
        df_h1:     pd.DataFrame,
        df_h4:     pd.DataFrame | None = None,
        df_d1:     pd.DataFrame | None = None,
        context:   dict | None = None,
    ) -> StrategyResult:

        if len(df_h1) < self.min_df_bars:
            return self._no_signal(symbol, direction, "Insufficient H1 bars")

        is_long = direction.lower() in ("long", "buy")

        # ── HTF check ─────────────────────────────────────────────────────────
        htf_ok, htf_reason = self._htf_bias(df_h4, df_d1, direction)
        if not htf_ok:
            return self._no_signal(symbol, direction, f"HTF opposing: {htf_reason}")

        # ── Liquidity sweep detection ─────────────────────────────────────────
        liq    = _SMC.find_liquidity_levels(df_h1)
        struct = _SMC.detect_market_structure(df_h1)

        if not liq["swept_recently"]:
            return self._no_signal(symbol, direction, "No recent liquidity sweep detected")

        # ── CHoCH required ────────────────────────────────────────────────────
        choch = struct.get("last_choch")
        bias  = struct.get("bias", "neutral")

        # Sweep direction must match trade direction
        if is_long and bias not in ("bullish", "neutral"):
            return self._no_signal(symbol, direction,
                                   f"Sweep reversed bearish — no long entry (bias={bias})")
        if not is_long and bias not in ("bearish", "neutral"):
            return self._no_signal(symbol, direction,
                                   f"Sweep reversed bullish — no short entry (bias={bias})")

        # ── Confirm reversal candle ───────────────────────────────────────────
        # Last candle should be in the direction of the trade
        last_open  = float(df_h1["open"].iloc[-1])
        last_close = float(df_h1["close"].iloc[-1])
        candle_ok  = (is_long and last_close > last_open) or \
                     (not is_long and last_close < last_open)

        if not candle_ok:
            return self._no_signal(symbol, direction,
                                   "Last candle direction doesn't confirm reversal")

        # ── Entry at post-sweep price + OB/FVG if available ───────────────────
        atr   = self._atr(df_h1)
        price = float(df_h1["close"].iloc[-1])

        # Look for FVG created during the sweep reversal
        fvgs = _SMC.find_fair_value_gaps(df_h1)
        target_fvg = "bullish" if is_long else "bearish"
        active_fvgs = [f for f in fvgs if f["active"] and not f["filled"] and f["fvg_type"] == target_fvg]

        entry = active_fvgs[0]["fvg_midpoint"] if active_fvgs else price

        # SL: below/above the swept level
        if is_long:
            ssl = liq["sell_side_liquidity"]
            swept_level = ssl[0] if ssl else (price - atr * 2)
            stop_loss = round(swept_level - atr * 0.3, 5)
        else:
            bsl = liq["buy_side_liquidity"]
            swept_level = bsl[0] if bsl else (price + atr * 2)
            stop_loss = round(swept_level + atr * 0.3, 5)

        tp1, tp2 = self._calc_tps(entry, stop_loss, direction, rr1=2.0, rr2=4.0)

        # ── Score ─────────────────────────────────────────────────────────────
        score = 0.0
        score += 3.0                              # sweep confirmed
        score += 2.0 if htf_ok else 0.5
        score += 2.0 if choch else 1.0            # CHoCH is ideal
        score += 1.5 if active_fvgs else 0.5      # FVG entry is cleaner
        score += 1.5 if candle_ok else 0.0

        reasons = [
            f"Liquidity swept — stop hunt detected",
            f"Structure: {struct.get('detail', '')[:80]}",
            htf_reason,
            f"Entry: {'FVG' if active_fvgs else 'market'} at {entry:.5f}",
        ]

        return StrategyResult(
            signal=True,
            strategy_name=self.name,
            symbol=symbol,
            direction=direction,
            score=round(min(score, 10.0), 1),
            entry_price=round(entry, 5),
            stop_loss=stop_loss,
            tp1_price=tp1,
            tp2_price=tp2,
            reasons=[r for r in reasons if r],
            factors={
                "swept":       True,
                "choch":       choch,
                "bias":        bias,
                "has_fvg":     bool(active_fvgs),
                "htf_ok":      htf_ok,
            },
        )
