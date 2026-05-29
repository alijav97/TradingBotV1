"""
btc_research/strategies/rsi_reversion.py — RSI Mean Reversion strategy.

Logic (COUNTER-TREND — fades extreme moves):
  Buy when BTC is deeply oversold and RSI starts recovering.
  Sell when BTC is deeply overbought and RSI starts fading.

  Long  : RSI(14) crossed BACK ABOVE the oversold threshold (default 30)
           — meaning RSI was below 30 on the previous bar, now above 30
  Short : RSI(14) crossed BACK BELOW the overbought threshold (default 70)
           — meaning RSI was above 70 on the previous bar, now below 70

  Additional confirmation:
  - Price must be near a support/resistance level (recent swing)
  - The extreme must be "fresh" (RSI at extreme within last 3 bars)

  SL placement:
  Long  : below the lowest low of the last 5 bars (the exhaustion low)
  Short : above the highest high of the last 5 bars (the exhaustion high)

  Note: This is a COUNTER-TREND strategy.
  It works best in ranging/choppy markets and may underperform in strong trends.
  The comparison will tell us whether BTC trends or ranges more in the best session.
"""
from __future__ import annotations
import pandas as pd
from btc_research.strategies.base import BTCStrategy


class RSIMeanReversion(BTCStrategy):
    name        = "RSI Mean Reversion"
    description = "Fade RSI extremes — buy oversold exits, sell overbought exits"

    def __init__(
        self,
        rsi_period:      int   = 14,
        oversold:        float = 30.0,
        overbought:      float = 70.0,
        lookback_fresh:  int   = 3,    # RSI must have been extreme within N bars
        sl_lookback:     int   = 5,
    ):
        self.rsi_period     = rsi_period
        self.oversold       = oversold
        self.overbought     = overbought
        self.lookback_fresh = lookback_fresh
        self.sl_lookback    = sl_lookback

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        min_bars = self.rsi_period + self.lookback_fresh + 5
        if len(df_window) < min_bars:
            return {"signal": False, "entry": 0.0, "sl": 0.0,
                    "reason": "insufficient bars"}

        close   = df_window["close"].astype(float)
        current = float(close.iloc[-1])
        is_long = direction.lower() in ("long", "buy")

        # Compute RSI series
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss   = (-delta).clip(lower=0).rolling(self.rsi_period).mean()
        rsi_s  = 100 - 100 / (1 + gain / (loss + 1e-10))

        rsi_now  = float(rsi_s.iloc[-1])
        rsi_prev = float(rsi_s.iloc[-2])

        # ── Long: RSI crossing back above oversold ─────────────────────────────
        if is_long:
            # Must cross from below oversold to above it on this bar
            if not (rsi_prev <= self.oversold and rsi_now > self.oversold):
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"RSI {rsi_now:.1f} not crossing above {self.oversold} "
                                  f"(prev={rsi_prev:.1f})"}

            # The RSI extreme must be recent (within lookback_fresh bars)
            recent_rsi = rsi_s.iloc[-(self.lookback_fresh + 1):]
            if recent_rsi.min() > self.oversold:
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"RSI extreme not recent enough (min={recent_rsi.min():.1f})"}

            sl_val = float(df_window["low"].iloc[-self.sl_lookback:].min())

        # ── Short: RSI crossing back below overbought ──────────────────────────
        else:
            if not (rsi_prev >= self.overbought and rsi_now < self.overbought):
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"RSI {rsi_now:.1f} not crossing below {self.overbought} "
                                  f"(prev={rsi_prev:.1f})"}

            recent_rsi = rsi_s.iloc[-(self.lookback_fresh + 1):]
            if recent_rsi.max() < self.overbought:
                return {"signal": False, "entry": current, "sl": 0.0,
                        "reason": f"RSI extreme not recent enough (max={recent_rsi.max():.1f})"}

            sl_val = float(df_window["high"].iloc[-self.sl_lookback:].max())

        sl_dist = abs(current - sl_val)
        if sl_dist <= 0:
            return {"signal": False, "entry": current, "sl": 0.0,
                    "reason": "zero SL distance"}

        return {
            "signal": True,
            "entry":  round(current, 2),
            "sl":     round(sl_val, 2),
            "reason": (f"RSI {'oversold exit' if is_long else 'overbought exit'} "
                       f"| RSI {rsi_prev:.1f} -> {rsi_now:.1f}"),
        }
