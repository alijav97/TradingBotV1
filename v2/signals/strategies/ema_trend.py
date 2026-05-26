"""
strategies/ema_trend.py — EMA Trend + Pullback to EMA21 strategy.

Entry model:
  1. EMA21 > EMA50 > EMA200 aligned (long) or reverse (short)
  2. ADX > 20 — confirmed trending market
  3. Price pulled back to EMA21 (within 0.6%)
  4. Rejection candle: long lower wick (long) or long upper wick (short)
  5. MACD agrees with direction (not required but adds score)

Universal strategy — works on all instruments in trending regimes.
"""
from __future__ import annotations

import logging

import pandas as pd

from v2.signals.strategies.base import StrategyBase, StrategyResult

logger = logging.getLogger(__name__)

EMA21_PROX_PCT = 0.006   # price within 0.6% of EMA21 = "at the pullback"
ADX_MIN        = 20


class EMATrendStrategy(StrategyBase):
    name        = "ema_trend"
    instruments = []   # all instruments
    timeframes  = ["H1"]
    min_df_bars = 60

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
        close   = df_h1["close"]
        price   = float(close.iloc[-1])

        # ── EMA alignment check ───────────────────────────────────────────────
        ema21  = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        ema50  = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

        if is_long:
            ema_aligned = ema21 > ema50 > ema200
        else:
            ema_aligned = ema21 < ema50 < ema200

        if not ema_aligned:
            return self._no_signal(symbol, direction,
                                   f"EMAs not aligned for {direction}: EMA21={ema21:.5f} EMA50={ema50:.5f} EMA200={ema200:.5f}")

        # ── ADX trending check ────────────────────────────────────────────────
        adx = self._adx(df_h1)
        if adx.get("adx", 0) < ADX_MIN:
            return self._no_signal(symbol, direction,
                                   f"ADX {adx.get('adx', 0):.1f} < {ADX_MIN} — market not trending")

        # ── Price at EMA21 pullback ───────────────────────────────────────────
        prox = abs(price - ema21) / max(ema21, 1e-9)
        if prox > EMA21_PROX_PCT:
            return self._no_signal(symbol, direction,
                                   f"Price {price:.5f} not near EMA21 {ema21:.5f} ({prox*100:.2f}% away, need <{EMA21_PROX_PCT*100:.1f}%)")

        # ── Rejection candle check ────────────────────────────────────────────
        body  = abs(float(df_h1["close"].iloc[-1]) - float(df_h1["open"].iloc[-1]))
        hi    = float(df_h1["high"].iloc[-1])
        lo    = float(df_h1["low"].iloc[-1])
        total = hi - lo

        if total > 0:
            if is_long:
                # Bullish rejection: long lower wick
                lower_wick = (min(float(df_h1["open"].iloc[-1]), float(df_h1["close"].iloc[-1])) - lo)
                rejection_ok = lower_wick > body * 1.5
            else:
                # Bearish rejection: long upper wick
                upper_wick = (hi - max(float(df_h1["open"].iloc[-1]), float(df_h1["close"].iloc[-1])))
                rejection_ok = upper_wick > body * 1.5
        else:
            rejection_ok = False

        # ── HTF alignment (bonus, not required) ──────────────────────────────
        htf_ok, htf_reason = self._htf_bias(df_h4, df_d1, direction)

        # ── MACD ─────────────────────────────────────────────────────────────
        macd = self._macd(df_h1)
        macd_ok = (is_long  and macd.get("bias") in ("bullish", "strongly_bullish")) or \
                  (not is_long and macd.get("bias") in ("bearish", "strongly_bearish"))

        # ── Entry, SL, TP ─────────────────────────────────────────────────────
        atr       = self._atr(df_h1)
        entry     = price
        stop_loss = round(ema21 - atr * 1.2, 5) if is_long else round(ema21 + atr * 1.2, 5)
        tp1, tp2  = self._calc_tps(entry, stop_loss, direction, rr1=2.0, rr2=4.0)

        # ── Score ─────────────────────────────────────────────────────────────
        score = 0.0
        score += 3.0                                    # EMA triple alignment
        score += min(adx.get("adx", 0) / 10, 2.0)     # ADX strength
        score += 1.5 if rejection_ok else 0.5          # rejection candle
        score += 1.5 if htf_ok else 0.5
        score += 1.0 if macd_ok else 0.0
        score += 1.0 if adx.get("bias") == ("bullish" if is_long else "bearish") else 0.0

        reasons = [
            f"EMA21={ema21:.2f} > EMA50={ema50:.2f} > EMA200={ema200:.2f}" if is_long
            else f"EMA21={ema21:.2f} < EMA50={ema50:.2f} < EMA200={ema200:.2f}",
            f"ADX={adx.get('adx', 0):.1f} ({adx.get('strength', 'n/a')})",
            f"Price at EMA21 pullback ({prox*100:.2f}% proximity)",
            "Rejection candle confirmed" if rejection_ok else "Candle structure borderline",
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
                "ema21":      ema21, "ema50": ema50, "ema200": ema200,
                "adx":        adx.get("adx", 0),
                "rejection":  rejection_ok,
                "htf_ok":     htf_ok,
                "macd_ok":    macd_ok,
            },
        )
