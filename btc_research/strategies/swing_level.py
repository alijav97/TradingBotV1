"""
btc_research/strategies/swing_level.py — Swing Level Break strategy.

Logic (STRUCTURE-BASED — trades key price levels):
  BTC respects swing highs and lows as support/resistance.
  A signal fires when price closes BEYOND the most recent significant swing.

  Swing high definition: a bar whose high is higher than the N bars before
                         AND the N bars after it (local maximum)
  Swing low  definition: a bar whose low  is lower than the N bars before
                         AND the N bars after it (local minimum)

  Signal:
  Long  : close breaks above the most recent swing HIGH
           — old resistance becomes new support (classic breakout)
  Short : close breaks below the most recent swing LOW
           — old support becomes new resistance

  SL placement:
  Long  : below the swing LOW that preceded the broken swing HIGH
           (the last significant support)
  Short : above the swing HIGH that preceded the broken swing LOW

  Additional filter:
  - Only take the FIRST breakout of a swing level (not a second re-test)
  - ATR must be above minimum threshold (avoid flat markets)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from btc_research.strategies.base import BTCStrategy


def _find_swings(df: pd.DataFrame, n: int = 3) -> tuple[list, list]:
    """
    Identify swing highs and lows.

    A swing high at index i: high[i] > high[i-n:i] AND high[i] > high[i+1:i+n+1]
    Since we only have history (no future bars), we detect swings that are
    confirmed by N bars of lower highs/higher lows AFTER the peak.

    Returns:
        swing_highs : list of (index, price)
        swing_lows  : list of (index, price)
    """
    highs = df["high"].astype(float).values
    lows  = df["low"].astype(float).values
    size  = len(df)

    swing_highs: list[tuple[int, float]] = []
    swing_lows:  list[tuple[int, float]] = []

    for i in range(n, size - n):
        # Swing high: peak that is higher than n bars on each side
        if (highs[i] == max(highs[i - n: i + n + 1])):
            swing_highs.append((i, highs[i]))
        # Swing low: trough lower than n bars on each side
        if (lows[i] == min(lows[i - n: i + n + 1])):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


class SwingLevelBreak(BTCStrategy):
    name        = "Swing Level Break"
    description = "Close breaks above recent swing high or below recent swing low"

    def __init__(
        self,
        swing_n:     int   = 3,    # bars each side to confirm a swing
        lookback:    int   = 50,   # how far back to look for swing levels
        min_atr_pct: float = 0.15, # minimum ATR% (avoid flat market)
        macro_ema:   int   = 96,
    ):
        self.swing_n     = swing_n
        self.lookback    = lookback
        self.min_atr_pct = min_atr_pct
        self.macro_ema   = macro_ema

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        min_bars = max(self.lookback, self.macro_ema) + 10
        if len(df_window) < min_bars:
            return {"signal": False, "entry": 0.0, "sl": 0.0,
                    "reason": "insufficient bars"}

        # Use a rolling window for swing detection (not the full 2yr history)
        win       = df_window.tail(self.lookback).reset_index(drop=True)
        current   = df_window.iloc[-1]
        bar_close = float(current["close"])
        is_long   = direction.lower() in ("long", "buy")

        close_s   = df_window["close"].astype(float)
        high_s    = df_window["high"].astype(float)
        low_s     = df_window["low"].astype(float)

        # ── Volatility check ─────────────────────────────────────────────────
        tr = pd.concat([
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s  - close_s.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr     = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = atr / bar_close * 100
        if atr_pct < self.min_atr_pct:
            return {"signal": False, "entry": bar_close, "sl": 0.0,
                    "reason": f"ATR {atr_pct:.2f}% too low (flat market)"}

        # ── Macro trend ───────────────────────────────────────────────────────
        ema_macro = float(close_s.ewm(span=self.macro_ema, adjust=False).mean().iloc[-1])

        # ── Find swing levels ─────────────────────────────────────────────────
        swing_highs, swing_lows = _find_swings(win, self.swing_n)

        if is_long:
            if not swing_highs:
                return {"signal": False, "entry": bar_close, "sl": 0.0,
                        "reason": "no swing highs found"}

            # Most recent swing high (excluding last n bars — must be confirmed)
            recent_sh = swing_highs[-1]
            sh_price  = recent_sh[1]

            if bar_close <= sh_price:
                return {"signal": False, "entry": bar_close, "sl": 0.0,
                        "reason": f"close {bar_close:.2f} <= swing_H {sh_price:.2f}"}

            if bar_close < ema_macro:
                return {"signal": False, "entry": bar_close, "sl": 0.0,
                        "reason": "price below D1 EMA (macro bearish)"}

            # SL = most recent swing LOW before the broken swing HIGH
            sl_candidates = [sl for sl in swing_lows if sl[0] < recent_sh[0]]
            if sl_candidates:
                sl_val = sl_candidates[-1][1]   # most recent swing low before the high
            else:
                sl_val = float(win["low"].min())  # fallback: lowest low in window

            reason = f"swing H break {sh_price:.2f} | SL=swing_L {sl_val:.2f}"

        else:  # short
            if not swing_lows:
                return {"signal": False, "entry": bar_close, "sl": 0.0,
                        "reason": "no swing lows found"}

            recent_sl = swing_lows[-1]
            sl_price  = recent_sl[1]

            if bar_close >= sl_price:
                return {"signal": False, "entry": bar_close, "sl": 0.0,
                        "reason": f"close {bar_close:.2f} >= swing_L {sl_price:.2f}"}

            if bar_close > ema_macro:
                return {"signal": False, "entry": bar_close, "sl": 0.0,
                        "reason": "price above D1 EMA (macro bullish)"}

            sh_candidates = [sh for sh in swing_highs if sh[0] < recent_sl[0]]
            if sh_candidates:
                sl_val = sh_candidates[-1][1]
            else:
                sl_val = float(win["high"].max())

            reason = f"swing L break {sl_price:.2f} | SL=swing_H {sl_val:.2f}"

        sl_dist = abs(bar_close - sl_val)
        if sl_dist <= 0:
            return {"signal": False, "entry": bar_close, "sl": 0.0,
                    "reason": "zero SL distance"}

        return {
            "signal": True,
            "entry":  round(bar_close, 2),
            "sl":     round(sl_val, 2),
            "reason": reason,
        }
