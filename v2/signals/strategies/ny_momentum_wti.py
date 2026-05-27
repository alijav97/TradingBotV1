"""
strategies/ny_momentum_wti.py — WTI Kill-Zone London Breakout strategy.

Setup (the classic "kill-zone" model for crude oil):
  1. Detect the London session range (08:00–13:00 UTC = 12PM–5PM UAE)
  2. At / after London close / NYMEX ramp-up (13:00 UTC = 5PM UAE), check for a
     breakout of the London high (LONG) or London low (SHORT)
  3. Wait for price to pull back and retest the broken level
  4. Entry on retest; SL = opposite side of the London range
  5. TP1 = 2× SL distance (50% partial close, SL shifts to breakeven)
  6. TP2 = 5× SL distance (remaining 50% — 1:5 RR)
  7. Only trade 13:00–17:00 UTC (5PM–9PM UAE / 9AM–1PM EST)

Why this works on WTI:
  - NYMEX electronic market opens at 14:00 UTC, injecting directional liquidity
  - London session creates a clear price range to trade against
  - NY traders fade or extend the London direction — the retest entry catches both
  - The 13:00–17:00 UTC window captures the most volatile + liquid WTI hours

Note on the screenshot timezone claim ("9AM EST / 1PM UAE"):
  9AM EST = 14:00 UTC = 18:00 UAE.  The actual overlap window used here is
  13:00–17:00 UTC = 17:00–21:00 UAE, which is correct for NYMEX.
"""
from __future__ import annotations

import logging

import pandas as pd

from v2.signals.strategies.base import StrategyBase, StrategyResult

logger = logging.getLogger(__name__)

# Session boundaries (UTC)
LONDON_START_UTC = 8    # 8AM UTC = 12PM UAE
LONDON_END_UTC   = 13   # 1PM UTC = 5PM UAE  (defines London range window)
NY_START_UTC     = 13   # 1PM UTC = 5PM UAE  (start looking for breakouts)
NY_END_UTC       = 17   # 5PM UTC = 9PM UAE  (stop taking new entries)

# Quality filters
MIN_LONDON_BARS  = 3    # need ≥ 3 London H1 bars to define a valid range
MIN_RANGE_ATR    = 0.3  # London range must be ≥ 30% of ATR (skip flat/choppy days)
RETEST_TOLERANCE = 0.5  # retest entry must be within 50% of ATR from the broken level


class NYMomentumWTIStrategy(StrategyBase):
    """
    WTI-specific kill-zone strategy: London range breakout + retest at NYMEX open.
    """
    name        = "ny_momentum_wti"
    instruments = ["WTI"]
    timeframes  = ["H1"]
    min_df_bars = 30

    def evaluate(
        self,
        symbol:    str,
        direction: str,
        df_h1:     pd.DataFrame,
        df_h4:     pd.DataFrame | None = None,
        df_d1:     pd.DataFrame | None = None,
        context:   dict | None = None,
    ) -> StrategyResult:

        if symbol.upper() not in ("WTI", "XTIUSD", "SPOTCRUDE", "USOIL"):
            return self._no_signal(symbol, direction, "WTI kill-zone: wrong instrument")
        if len(df_h1) < self.min_df_bars:
            return self._no_signal(symbol, direction, "Insufficient H1 bars")

        is_long = direction.lower() in ("long", "buy")

        # ── Session timing ────────────────────────────────────────────────────
        current_hour = None
        current_date = None
        try:
            last_time    = pd.to_datetime(df_h1["time"].iloc[-1], utc=True)
            current_hour = last_time.hour
            current_date = last_time.date()
        except Exception:
            pass

        if current_hour is not None and not (NY_START_UTC <= current_hour < NY_END_UTC):
            return self._no_signal(
                symbol, direction,
                f"Not NY/NYMEX window (UTC {current_hour:02d}:xx, need {NY_START_UTC:02d}–{NY_END_UTC:02d})",
            )

        # ── Build London session range from today's bars ──────────────────────
        london_bars = pd.DataFrame()
        if current_date is not None and "time" in df_h1.columns:
            try:
                times = pd.to_datetime(df_h1["time"], utc=True)
                mask  = (
                    (times.dt.date == current_date) &
                    (times.dt.hour >= LONDON_START_UTC) &
                    (times.dt.hour <  LONDON_END_UTC)
                )
                london_bars = df_h1[mask]
            except Exception:
                pass

        if len(london_bars) < MIN_LONDON_BARS:
            return self._no_signal(
                symbol, direction,
                f"Too few London bars today ({len(london_bars)} < {MIN_LONDON_BARS}) — range undefined",
            )

        london_high  = float(london_bars["high"].max())
        london_low   = float(london_bars["low"].min())
        london_range = london_high - london_low

        # ── Range sanity check ────────────────────────────────────────────────
        atr = self._atr(df_h1)
        if atr <= 0:
            return self._no_signal(symbol, direction, "ATR calculation failed")

        if london_range < atr * MIN_RANGE_ATR:
            return self._no_signal(
                symbol, direction,
                f"London range too tight ({london_range:.3f} < {atr * MIN_RANGE_ATR:.3f}) — choppy session",
            )

        # ── Breakout check ────────────────────────────────────────────────────
        price    = float(df_h1["close"].iloc[-1])
        bar_high = float(df_h1["high"].iloc[-1])
        bar_low  = float(df_h1["low"].iloc[-1])

        if is_long:
            if bar_high <= london_high:
                return self._no_signal(
                    symbol, direction,
                    f"No London high breakout — bar high {bar_high:.3f} ≤ london high {london_high:.3f}",
                )
            breakout_level = london_high
            sl             = london_low
        else:
            if bar_low >= london_low:
                return self._no_signal(
                    symbol, direction,
                    f"No London low breakout — bar low {bar_low:.3f} ≥ london low {london_low:.3f}",
                )
            breakout_level = london_low
            sl             = london_high

        # ── Retest quality ────────────────────────────────────────────────────
        dist_to_level = abs(price - breakout_level)
        retest_window = atr * RETEST_TOLERANCE

        if dist_to_level > retest_window:
            return self._no_signal(
                symbol, direction,
                f"Waiting for retest — price {price:.3f} is {dist_to_level:.3f} from "
                f"breakout level {breakout_level:.3f} (window={retest_window:.3f})",
            )

        sl_dist = abs(price - sl)
        if sl_dist <= 0:
            return self._no_signal(symbol, direction, "Zero SL distance — invalid levels")

        # ── Entry, SL, TP ─────────────────────────────────────────────────────
        entry    = price
        tp1, tp2 = self._calc_tps(entry, sl, direction, rr1=2.0, rr2=5.0)

        # ── HTF bias ──────────────────────────────────────────────────────────
        htf_ok, htf_reason = self._htf_bias(df_h4, df_d1, direction)

        # ── Volume spike confirmation ─────────────────────────────────────────
        vol_ok = True
        vol_ratio = 1.0
        if "volume" in df_h1.columns:
            try:
                avg_vol   = float(df_h1["volume"].tail(20).mean())
                cur_vol   = float(df_h1["volume"].iloc[-1])
                vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
                vol_ok    = vol_ratio >= 1.1
            except Exception:
                vol_ok = True

        # ── Closed-beyond-level (price closed on the breakout side) ──────────
        closed_beyond = (is_long and price > london_high) or (not is_long and price < london_low)

        # ── Score ─────────────────────────────────────────────────────────────
        score = 0.0

        # 1. Session timing: earlier in the NY window = fresher breakout
        hours_into_session = (current_hour - NY_START_UTC) if current_hour is not None else 2
        score += 2.5 if hours_into_session <= 1 else (1.5 if hours_into_session <= 2 else 0.5)

        # 2. London range quality vs ATR
        range_pct = london_range / atr
        score += min(range_pct * 2.0, 2.0)

        # 3. Retest quality (0–2): tighter retest = better entry
        retest_quality = max(1.0 - (dist_to_level / retest_window), 0.0)
        score += retest_quality * 2.0

        # 4. HTF alignment
        score += 1.5 if htf_ok else 0.0

        # 5. Volume spike
        score += 1.0 if vol_ok else 0.0

        # 6. Price closed beyond level (breakout conviction)
        score += 1.0 if closed_beyond else 0.0

        reasons = [
            f"London range {london_low:.3f}–{london_high:.3f}  "
            f"(range={london_range:.3f}  ATR={atr:.3f}  bars={len(london_bars)})",
            f"{'Bullish' if is_long else 'Bearish'} breakout of London "
            f"{'high' if is_long else 'low'} at {breakout_level:.3f}",
            f"Retest quality {retest_quality*100:.0f}%  (dist={dist_to_level:.3f})",
            f"Volume {'spike ×{:.1f}'.format(vol_ratio) if vol_ok else 'weak (×{:.1f})'.format(vol_ratio)}",
            htf_reason,
        ]

        return StrategyResult(
            signal=True,
            strategy_name=self.name,
            symbol=symbol,
            direction=direction,
            score=round(min(score, 10.0), 1),
            entry_price=round(entry, 5),
            stop_loss=round(sl, 5),
            tp1_price=tp1,
            tp2_price=tp2,
            reasons=[r for r in reasons if r],
            factors={
                "london_high":       london_high,
                "london_low":        london_low,
                "london_range":      round(london_range, 3),
                "breakout_level":    round(breakout_level, 3),
                "dist_to_level":     round(dist_to_level, 3),
                "retest_quality":    round(retest_quality, 2),
                "closed_beyond":     closed_beyond,
                "vol_ratio":         round(vol_ratio, 2),
                "vol_ok":            vol_ok,
                "htf_ok":            htf_ok,
                "hour_utc":          current_hour,
                "london_bars_count": len(london_bars),
            },
        )
