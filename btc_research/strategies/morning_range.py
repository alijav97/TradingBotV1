"""
btc_research/strategies/morning_range.py — Morning Range Breakout strategy.

Logic:
  Define the "range" as the N bars immediately before the current bar.
  A signal fires when price CLOSES outside that range.

  Long  : bar close > range high  → buy breakout
  Short : bar close < range low   → sell breakout

  SL placement:
  Long  : stop below range LOW  (full range as risk)
  Short : stop above range HIGH

  This is the same concept as the WTI London-range breakout but applied to
  any time window without a fixed session restriction.

  Works best when there is a clear consolidation followed by a breakout —
  common in BTC during low-volume windows (e.g. early Asia, 02-04 UTC).
"""
from __future__ import annotations
import pandas as pd
from btc_research.strategies.base import BTCStrategy


class MorningRangeBreakout(BTCStrategy):
    name        = "Morning Range Breakout"
    description = "Close breaks above/below recent N-bar consolidation range"

    def __init__(self, range_bars: int = 6):
        """
        Args:
            range_bars : number of bars to use for the reference range
                         (6 = last 6 closed bars before current bar)
        """
        self.range_bars = range_bars

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        min_bars = self.range_bars + 2
        if len(df_window) < min_bars:
            return {"signal": False, "entry": 0.0, "sl": 0.0,
                    "reason": "insufficient bars"}

        current   = df_window.iloc[-1]
        range_win = df_window.iloc[-(self.range_bars + 1):-1]

        bar_close  = float(current["close"])
        range_high = float(range_win["high"].max())
        range_low  = float(range_win["low"].min())
        is_long    = direction.lower() in ("long", "buy")

        # Breakout condition
        if is_long and bar_close <= range_high:
            return {"signal": False, "entry": bar_close, "sl": 0.0,
                    "reason": f"no breakout: close {bar_close:.2f} <= range_H {range_high:.2f}"}
        if not is_long and bar_close >= range_low:
            return {"signal": False, "entry": bar_close, "sl": 0.0,
                    "reason": f"no breakout: close {bar_close:.2f} >= range_L {range_low:.2f}"}

        sl      = range_low if is_long else range_high
        sl_dist = abs(bar_close - sl)
        if sl_dist <= 0:
            return {"signal": False, "entry": bar_close, "sl": sl,
                    "reason": "zero SL distance"}

        side = "above" if is_long else "below"
        ref  = range_high if is_long else range_low
        return {
            "signal": True,
            "entry":  round(bar_close, 2),
            "sl":     round(sl, 2),
            "reason": f"range breakout {side} {ref:.2f} (range={range_high-range_low:.2f})",
        }
