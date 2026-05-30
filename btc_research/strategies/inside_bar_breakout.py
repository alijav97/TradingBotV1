"""
btc_research/strategies/inside_bar_breakout.py — Inside Bar Breakout (IBB) strategy.

== CONCEPT ==
  "Inside bars = compression → explosive moves."
  "The highest-probability entry is the first candle that closes outside
  a volatility compression zone, confirmed by ATR or range expansion."

  Asia Night (01-03 UTC) is a natural compression session — BTC consolidates
  after the NY close before institutional Asian players step in. Inside bars
  form repeatedly at this time. When the compression breaks, the move is clean.

  An inside bar is a bar whose HIGH and LOW are BOTH within the range of the
  preceding bar (the "mother bar"). It represents a pause, a coil.

  Signal fires when:
    1. There are ≥1 consecutive inside bars before the current bar
    2. Current bar BREAKS OUT beyond the mother bar's range
    3. For longs : current close > mother bar's HIGH
    4. For shorts: current close < mother bar's LOW

== SL PLACEMENT ==
  Long  → SL = mother bar's LOW  (if compression fails, the whole range failed)
  Short → SL = mother bar's HIGH

  The mother bar anchors both the breakout trigger AND the stop — clean and
  well-defined. SL width = mother bar's range = 0.5-1.5× ATR typically.

== CHAIN COMPRESSION ==
  If multiple consecutive inside bars exist (NR3, NR4... patterns), we use
  the ORIGINAL mother bar (first bar of the chain) — its range is the broadest
  and the definitive boundary for the compression zone.

== ADDITIONAL FILTERS ==
  1. Mother bar range must be at least min_range_atr × ATR
     (avoid trivially small compressions)
  2. SL distance (mother bar range) must not exceed max_sl_atr × ATR
     (avoid entries with oversized SL from large mother bars)
  3. Breakout bar must show conviction: close in top/bottom close_zone of
     its own range (not just a marginal poke above/below)
  4. ATR filter: current ATR must be above min_atr_pct of price
     (avoid flat market entries)

== PARAMETERS ==
  min_inside_bars  : minimum consecutive inside bars required   (default 1)
  max_inside_chain : max chain length to look back              (default 8)
  close_zone       : close must be in top/bottom X% of bar      (default 0.40)
  min_range_atr    : mother bar range ≥ this × ATR              (default 0.3)
  max_sl_atr       : reject if mother bar range > this × ATR    (default 2.0)
  min_atr_pct      : minimum ATR as % of price                  (default 0.15)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from btc_research.strategies.base import BTCStrategy


class InsideBarBreakout(BTCStrategy):
    name        = "Inside Bar Breakout"
    description = ("Compression inside bar(s) → breakout beyond mother bar  |  "
                   "SL = mother bar extreme  |  best at Asia Night consolidation")

    def __init__(
        self,
        min_inside_bars:  int   = 1,    # minimum consecutive inside bars
        max_inside_chain: int   = 8,    # max bars to walk back looking for chain
        close_zone:       float = 0.40, # close must be in top/bottom 40% of bar
        min_range_atr:    float = 0.30, # mother bar range must be ≥ this × ATR
        max_sl_atr:       float = 2.0,  # reject if SL > this × ATR
        min_atr_pct:      float = 0.15, # minimum ATR% of price (flat market filter)
    ):
        self.min_inside_bars  = min_inside_bars
        self.max_inside_chain = max_inside_chain
        self.close_zone       = close_zone
        self.min_range_atr    = min_range_atr
        self.max_sl_atr       = max_sl_atr
        self.min_atr_pct      = min_atr_pct

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        no_sig = {"signal": False, "entry": 0.0, "sl": 0.0, "reason": ""}

        if len(df_window) < 20:
            no_sig["reason"] = "insufficient bars"
            return no_sig

        highs  = df_window["high"].astype(float).values
        lows   = df_window["low"].astype(float).values
        closes = df_window["close"].astype(float).values
        n      = len(highs)

        curr_close = closes[-1]
        curr_high  = highs[-1]
        curr_low   = lows[-1]
        curr_range = curr_high - curr_low

        # ── ATR(14) ───────────────────────────────────────────────────────────
        high_s  = df_window["high"].astype(float)
        low_s   = df_window["low"].astype(float)
        close_s = df_window["close"].astype(float)
        tr = pd.concat([
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s  - close_s.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-2])
        if atr <= 0:
            no_sig["reason"] = "ATR zero"
            return no_sig

        # Flat market filter
        atr_pct = atr / curr_close * 100
        if atr_pct < self.min_atr_pct:
            no_sig["reason"] = f"ATR {atr_pct:.2f}% too low (flat market)"
            return no_sig

        # ── Walk back to find inside bar chain ────────────────────────────────
        # Start from bar BEFORE current (index n-2) and walk backwards
        # An inside bar at index j: high[j] < high[j-1] AND low[j] > low[j-1]
        inside_count = 0
        j = n - 2   # start from bar before current

        while j >= 1 and inside_count < self.max_inside_chain:
            if highs[j] < highs[j - 1] and lows[j] > lows[j - 1]:
                inside_count += 1
                j -= 1
            else:
                break
        # j is now the mother bar index (the bar that IS NOT an inside bar)

        if inside_count < self.min_inside_bars:
            no_sig["reason"] = f"only {inside_count} inside bars (need {self.min_inside_bars})"
            return no_sig

        mother_idx   = j
        mother_high  = highs[mother_idx]
        mother_low   = lows[mother_idx]
        mother_range = mother_high - mother_low

        # ── Filters on mother bar ─────────────────────────────────────────────
        if mother_range < self.min_range_atr * atr:
            no_sig["reason"] = (f"mother bar range {mother_range:.0f} too small "
                                f"(< {self.min_range_atr}×ATR {atr:.0f})")
            return no_sig

        if mother_range > self.max_sl_atr * atr:
            no_sig["reason"] = (f"mother bar range {mother_range:.0f} too wide "
                                f"(> {self.max_sl_atr}×ATR {atr:.0f})")
            return no_sig

        is_long = direction.lower() in ("long", "buy")

        if is_long:
            # Current bar must close ABOVE the mother bar's high
            if curr_close <= mother_high:
                no_sig["reason"] = (f"close {curr_close:.0f} not above mother_H "
                                    f"{mother_high:.0f}")
                return no_sig

            # Conviction: close must be in the top close_zone of the breakout bar
            if curr_range > 0:
                close_pos = (curr_close - curr_low) / curr_range
            else:
                close_pos = 0.5
            if close_pos < (1 - self.close_zone):
                no_sig["reason"] = (f"close not in top {self.close_zone*100:.0f}% "
                                    f"of bar (pos={close_pos:.2f})")
                return no_sig

            entry  = curr_close
            sl_val = mother_low  # SL below mother bar's low

        else:
            # Current bar must close BELOW the mother bar's low
            if curr_close >= mother_low:
                no_sig["reason"] = (f"close {curr_close:.0f} not below mother_L "
                                    f"{mother_low:.0f}")
                return no_sig

            if curr_range > 0:
                close_pos = (curr_close - curr_low) / curr_range
            else:
                close_pos = 0.5
            if close_pos > self.close_zone:
                no_sig["reason"] = (f"close not in bottom {self.close_zone*100:.0f}% "
                                    f"of bar (pos={close_pos:.2f})")
                return no_sig

            entry  = curr_close
            sl_val = mother_high  # SL above mother bar's high

        sl_dist = abs(entry - sl_val)
        if sl_dist <= 0:
            no_sig["reason"] = "zero SL distance"
            return no_sig

        return {
            "signal": True,
            "entry":  round(entry, 2),
            "sl":     round(sl_val, 2),
            "reason": (f"IBB {'long' if is_long else 'short'}: "
                       f"{inside_count} inside bar(s) | "
                       f"mother_H={mother_high:.0f} mother_L={mother_low:.0f} "
                       f"range={mother_range:.0f} ({mother_range/atr:.1f}×ATR) "
                       f"| bars ago={n-1-mother_idx}"),
            "tp1_rr": 2.0,
            "tp2_rr": 5.0,
        }
