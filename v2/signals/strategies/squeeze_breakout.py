"""
strategies/squeeze_breakout.py — Bollinger/Keltner Squeeze Momentum strategy.

Fires when volatility compression releases:
  1. Bollinger Bands were inside Keltner Channels (squeeze ON) for 3+ bars
  2. Squeeze just fired (BB expanded outside KC)
  3. Momentum histogram positive (long) or negative (short)
  4. Candle body confirms direction

Effective across all instruments in any session.
"""
from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from v2.signals.strategies.base import StrategyBase, StrategyResult

logger = logging.getLogger(__name__)

BB_STD    = 2.0
KC_MULT   = 1.5
ATR_PER   = 14
SQUEEZE_LOOKBACK = 5    # bars of squeeze required before fire


class SqueezeBreakoutStrategy(StrategyBase):
    name        = "squeeze_breakout"
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
        close = df_h1["close"]
        high  = df_h1["high"]
        low   = df_h1["low"]

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = bb_mid + BB_STD * bb_std
        bb_lower = bb_mid - BB_STD * bb_std

        # ── Keltner Channels ─────────────────────────────────────────────────
        kc_mid = close.ewm(span=20, adjust=False).mean()
        tr     = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr    = tr.rolling(ATR_PER).mean()
        kc_upper = kc_mid + KC_MULT * atr
        kc_lower = kc_mid - KC_MULT * atr

        # ── Squeeze detection ─────────────────────────────────────────────────
        # squeeze_on = BB inside KC (both upper and lower)
        squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        n = len(df_h1)
        if n < SQUEEZE_LOOKBACK + 2:
            return self._no_signal(symbol, direction, "Not enough bars to detect squeeze history")

        # Were last N-1 bars in squeeze?
        squeeze_was_on = all(squeeze_on.iloc[-(SQUEEZE_LOOKBACK + 1): -1])
        # Is last bar out of squeeze?
        squeeze_just_fired = not bool(squeeze_on.iloc[-1]) and squeeze_was_on

        if not squeeze_just_fired:
            return self._no_signal(symbol, direction, "No squeeze firing — either still squeezed or never was")

        # ── Momentum direction ────────────────────────────────────────────────
        # Use linear regression of close - mid as momentum proxy
        lookback = 10
        vals = (close - kc_mid).iloc[-lookback:].values
        x    = np.arange(len(vals), dtype=float)
        if len(x) >= 2:
            slope = float(np.polyfit(x, vals, 1)[0])
        else:
            slope = 0.0

        momentum_bullish = slope > 0
        momentum_bearish = slope < 0

        if is_long and not momentum_bullish:
            return self._no_signal(symbol, direction, f"Squeeze fired bearish (slope={slope:.4f}) — no long")
        if not is_long and not momentum_bearish:
            return self._no_signal(symbol, direction, f"Squeeze fired bullish (slope={slope:.4f}) — no short")

        # ── Confirming candle ─────────────────────────────────────────────────
        last_open  = float(df_h1["open"].iloc[-1])
        last_close = float(df_h1["close"].iloc[-1])
        candle_ok  = (is_long and last_close > last_open) or \
                     (not is_long and last_close < last_open)

        if not candle_ok:
            return self._no_signal(symbol, direction, "Candle direction doesn't match momentum")

        # ── HTF ──────────────────────────────────────────────────────────────
        htf_ok, htf_reason = self._htf_bias(df_h4, df_d1, direction)

        # ── Entry, SL, TP ─────────────────────────────────────────────────────
        price    = float(close.iloc[-1])
        atr_val  = float(atr.iloc[-1])
        entry    = price
        stop_loss = round(price - atr_val * 1.5, 5) if is_long else round(price + atr_val * 1.5, 5)
        tp1, tp2 = self._calc_tps(entry, stop_loss, direction, rr1=2.0, rr2=4.0)

        # How many bars was squeeze on (score factor)
        squeeze_bars = 0
        for i in range(2, min(20, n)):
            if bool(squeeze_on.iloc[-i]):
                squeeze_bars += 1
            else:
                break

        # ── Score ─────────────────────────────────────────────────────────────
        score = 0.0
        score += 3.0                                          # squeeze fired
        score += min(squeeze_bars * 0.3, 2.0)                # longer squeeze = bigger move
        score += 1.5 if abs(slope) > atr_val * 0.005 else 0.5  # strong momentum
        score += 1.5 if htf_ok else 0.5
        score += 1.5 if candle_ok else 0.0

        reasons = [
            f"Squeeze fired after {squeeze_bars}+ bars of compression",
            f"Momentum slope={slope:.5f} ({'bullish' if momentum_bullish else 'bearish'})",
            htf_reason,
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
                "squeeze_bars":     squeeze_bars,
                "slope":            round(slope, 6),
                "momentum_bullish": momentum_bullish,
                "htf_ok":           htf_ok,
                "candle_ok":        candle_ok,
            },
        )
