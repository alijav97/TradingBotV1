"""
btc_research/strategies/volatility_breakout.py — Volatility/Momentum Breakout strategy.

Logic (TREND-FOLLOWING — catches explosive moves):
  A signal fires when a bar is significantly larger than recent bars,
  indicating a surge of momentum. Enter in the direction of the surge.

  Conditions:
  1. Current bar range (high - low) > ATR_multiplier × ATR(14)
     — bar is unusually large = momentum is accelerating
  2. Bar is DIRECTIONAL: close in the top 30% of bar for longs
                          close in the bottom 30% of bar for shorts
     — strong conviction candle, not a doji/reversal
  3. D1-equivalent trend (EMA96) must align with direction
     — never catch a falling knife against the macro trend

  SL placement:
  Long  : below the low of the current bar  (momentum bar's low as support)
  Short : above the high of the current bar

  This strategy captures BTC's signature explosive breakout moves —
  common when major news hits or when large buyers/sellers step in.
"""
from __future__ import annotations
import pandas as pd
from btc_research.strategies.base import BTCStrategy


class VolatilityBreakout(BTCStrategy):
    name        = "Volatility Breakout"
    description = "Enter on ATR-expansion bar in the direction of the surge"

    def __init__(
        self,
        atr_period:     int   = 14,
        atr_multiplier: float = 1.5,   # bar must be > 1.5× ATR to qualify
        close_zone:     float = 0.30,  # close must be in top/bottom 30% of bar
        macro_ema:      int   = 96,
    ):
        self.atr_period     = atr_period
        self.atr_multiplier = atr_multiplier
        self.close_zone     = close_zone
        self.macro_ema      = macro_ema

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        min_bars = self.atr_period + self.macro_ema + 2
        if len(df_window) < min_bars:
            return {"signal": False, "entry": 0.0, "sl": 0.0,
                    "reason": "insufficient bars"}

        current   = df_window.iloc[-1]
        bar_open  = float(current["open"])
        bar_high  = float(current["high"])
        bar_low   = float(current["low"])
        bar_close = float(current["close"])
        bar_range = bar_high - bar_low
        is_long   = direction.lower() in ("long", "buy")

        close_s   = df_window["close"].astype(float)
        high_s    = df_window["high"].astype(float)
        low_s     = df_window["low"].astype(float)

        # ── ATR(14) ────────────────────────────────────────────────────────────
        tr = pd.concat([
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s  - close_s.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(self.atr_period).mean().iloc[-2])  # use prev bar's ATR

        if atr <= 0:
            return {"signal": False, "entry": bar_close, "sl": 0.0,
                    "reason": "ATR is zero"}

        # ── Condition 1: bar is large (ATR expansion) ─────────────────────────
        if bar_range < self.atr_multiplier * atr:
            return {"signal": False, "entry": bar_close, "sl": 0.0,
                    "reason": f"bar range {bar_range:.2f} < {self.atr_multiplier}×ATR {atr:.2f}"}

        # ── Condition 2: close in direction zone ──────────────────────────────
        if bar_range > 0:
            close_position = (bar_close - bar_low) / bar_range  # 0=at low, 1=at high
        else:
            close_position = 0.5

        if is_long and close_position < (1 - self.close_zone):
            return {"signal": False, "entry": bar_close, "sl": 0.0,
                    "reason": f"close not in top {self.close_zone*100:.0f}% of bar "
                              f"(pos={close_position:.2f})"}
        if not is_long and close_position > self.close_zone:
            return {"signal": False, "entry": bar_close, "sl": 0.0,
                    "reason": f"close not in bottom {self.close_zone*100:.0f}% of bar "
                              f"(pos={close_position:.2f})"}

        # ── Condition 3: macro trend alignment ────────────────────────────────
        ema_macro = float(close_s.ewm(span=self.macro_ema, adjust=False).mean().iloc[-1])
        if is_long and bar_close < ema_macro:
            return {"signal": False, "entry": bar_close, "sl": 0.0,
                    "reason": f"price below D1 EMA (macro bearish)"}
        if not is_long and bar_close > ema_macro:
            return {"signal": False, "entry": bar_close, "sl": 0.0,
                    "reason": f"price above D1 EMA (macro bullish)"}

        # ── SL: low/high of the breakout bar ──────────────────────────────────
        sl_val  = bar_low if is_long else bar_high
        sl_dist = abs(bar_close - sl_val)
        if sl_dist <= 0:
            return {"signal": False, "entry": bar_close, "sl": 0.0,
                    "reason": "zero SL distance"}

        return {
            "signal":  True,
            "entry":   round(bar_close, 2),
            "sl":      round(sl_val, 2),
            "reason":  (f"ATR breakout {bar_range:.2f} = {bar_range/atr:.1f}×ATR "
                        f"| close@{close_position:.0%} of bar"),
            # Per-strategy TP levels — explosive bars kick off large directional moves
            # TP2 at 9R lets the momentum run instead of capping at the global 5R
            "tp1_rr":  2.0,
            "tp2_rr":  9.0,
        }
