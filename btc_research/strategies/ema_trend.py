"""
btc_research/strategies/ema_trend.py — EMA Trend-Following strategy.

Logic:
  Three-layer EMA alignment: all three must agree before entering.

  Long  : EMA8 > EMA21 > EMA50  AND  price above EMA8  (full bull stack)
  Short : EMA8 < EMA21 < EMA50  AND  price below EMA8  (full bear stack)

  Additionally:
  - D1-equivalent trend (EMA96) must be on the same side as the trade
    (never trade against the macro trend)
  - RSI must NOT be at an extreme in the trade direction
    (don't buy overbought, don't short oversold)

  SL placement:
  Long  : below the lowest low of the last 8 bars  (recent structure)
  Short : above the highest high of the last 8 bars

  This strategy captures strong trending moves and avoids counter-trend entries.
  It tends to have fewer signals but higher quality setups.
"""
from __future__ import annotations
import pandas as pd
from btc_research.strategies.base import BTCStrategy


class EMATrendFollow(BTCStrategy):
    name        = "EMA Trend Follow"
    description = "Full EMA stack alignment (EMA8 > EMA21 > EMA50) + macro trend"

    def __init__(
        self,
        fast:   int = 8,
        mid:    int = 21,
        slow:   int = 50,
        macro:  int = 96,    # ~D1 trend on H1 data
        sl_lookback: int = 8,
    ):
        self.fast        = fast
        self.mid         = mid
        self.slow        = slow
        self.macro       = macro
        self.sl_lookback = sl_lookback

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        min_bars = self.macro + 5
        if len(df_window) < min_bars:
            return {"signal": False, "entry": 0.0, "sl": 0.0,
                    "reason": "insufficient bars for EMA calc"}

        close   = df_window["close"].astype(float)
        current = float(close.iloc[-1])
        is_long = direction.lower() in ("long", "buy")

        ema_fast  = float(close.ewm(span=self.fast,  adjust=False).mean().iloc[-1])
        ema_mid   = float(close.ewm(span=self.mid,   adjust=False).mean().iloc[-1])
        ema_slow  = float(close.ewm(span=self.slow,  adjust=False).mean().iloc[-1])
        ema_macro = float(close.ewm(span=self.macro, adjust=False).mean().iloc[-1])

        # RSI(14)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta).clip(lower=0).rolling(14).mean()
        rsi   = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-10)))

        # ── Long conditions ────────────────────────────────────────────────────
        if is_long:
            if not (ema_fast > ema_mid > ema_slow):
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"EMA stack not bull: {ema_fast:.0f}<{ema_mid:.0f}<{ema_slow:.0f}"}
            if current < ema_fast:
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"price {current:.2f} below EMA{self.fast} {ema_fast:.2f}"}
            if current < ema_macro:
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"price below D1 EMA (macro bearish)"}
            if rsi > 78:
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"RSI {rsi:.1f} overbought — skip long"}
            sl_val = float(df_window["low"].iloc[-self.sl_lookback:].min())

        # ── Short conditions ───────────────────────────────────────────────────
        else:
            if not (ema_fast < ema_mid < ema_slow):
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"EMA stack not bear: {ema_fast:.0f}>{ema_mid:.0f}>{ema_slow:.0f}"}
            if current > ema_fast:
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"price {current:.2f} above EMA{self.fast} {ema_fast:.2f}"}
            if current > ema_macro:
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"price above D1 EMA (macro bullish)"}
            if rsi < 22:
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"RSI {rsi:.1f} oversold — skip short"}
            sl_val = float(df_window["high"].iloc[-self.sl_lookback:].max())

        sl_dist = abs(current - sl_val)
        if sl_dist <= 0:
            return {"signal": False, "entry": current, "sl": 0.0,
                    "reason": "zero SL distance"}

        return {
            "signal":  True,
            "entry":   round(current, 2),
            "sl":      round(sl_val, 2),
            "reason":  (f"EMA stack {'bull' if is_long else 'bear'} "
                        f"| RSI={rsi:.1f} "
                        f"| macro={'above' if current > ema_macro else 'below'} D1 EMA"),
            # Per-strategy TP levels — trend entries can ride multi-bar moves.
            # TP2=6R gives room for the trend to extend without cutting too early.
            "tp1_rr":  2.0,
            "tp2_rr":  6.0,
        }
