"""
strategies/ny_momentum_wti.py — WTI Kill-Zone London Breakout strategy.

Setup (the classic "kill-zone" model for crude oil):
  1. Detect the London session range (08:00–13:00 UTC = 12PM–5PM UAE)
  2. At / after NYMEX ramp-up (13:00 UTC = 5PM UAE), check for a breakout
     of the London high (LONG) or London low (SHORT)
  3. Two valid entry modes:
       a) Retest entry  — price broke the level, pulled back close to it
       b) Breakout entry — bar just broke the level (within same bar, no pull-back yet)
  4. SL = opposite side of the London range
  5. TP1 = 2× SL distance (50% partial close, SL shifts to breakeven)
  6. TP2 = 5× SL distance (remaining 50% — 1:5 RR)
  7. Only trade 13:00–17:00 UTC (5PM–9PM UAE / 9AM–1PM EST)

Score components (max 10.0):
  Session freshness      0.5–2.5
  London range quality   0.0–2.0
  Entry quality          0.0–2.0
  HTF alignment          0.0–1.5
  Volume spike           0.0–1.0
  Closed beyond level    0.0–1.0
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
MIN_LONDON_BARS   = 3    # need ≥ 3 London H1 bars to define a valid range
MIN_RANGE_ATR_PCT = 0.25 # London range must be ≥ 25% of ATR (skip flat days)
RETEST_TOLERANCE  = 0.8  # retest entry: price within 80% of ATR from broken level
BREAKOUT_CHASE    = 1.5  # breakout entry: price no more than 1.5× ATR beyond level


class NYMomentumWTIStrategy(StrategyBase):
    """
    WTI kill-zone: London range breakout + retest (or fresh breakout) at NYMEX open.
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

        def _reject(reason: str) -> StrategyResult:
            logger.info("NYMomentumWTI [%s %s] SKIP: %s", symbol, direction.upper(), reason)
            return self._no_signal(symbol, direction, reason)

        if symbol.upper() not in ("WTI", "XTIUSD", "SPOTCRUDE", "USOIL"):
            return _reject("WTI kill-zone: wrong instrument")
        if len(df_h1) < self.min_df_bars:
            return _reject(f"Insufficient H1 bars ({len(df_h1)} < {self.min_df_bars})")

        is_long = direction.lower() in ("long", "buy")

        # ── Session timing ────────────────────────────────────────────────────
        current_hour = None
        current_date = None
        try:
            raw_time  = df_h1["time"].iloc[-1]
            last_time = pd.to_datetime(raw_time)
            # Handle both timezone-aware and naive timestamps
            if last_time.tzinfo is None:
                last_time = last_time.tz_localize("UTC")
            else:
                last_time = last_time.tz_convert("UTC")
            current_hour = last_time.hour
            current_date = last_time.date()
            logger.debug(
                "NYMomentumWTI [%s %s] last bar timestamp: %s → UTC hour=%d, date=%s",
                symbol, direction.upper(), last_time.isoformat(), current_hour, current_date,
            )
        except Exception as exc:
            logger.warning("NYMomentumWTI [%s %s] timestamp parse error: %s", symbol, direction.upper(), exc)

        if current_hour is not None and not (NY_START_UTC <= current_hour < NY_END_UTC):
            return _reject(
                f"Not NY/NYMEX window (UTC {current_hour:02d}:xx, need {NY_START_UTC:02d}–{NY_END_UTC:02d})",
            )

        # ── Build London session range from today's bars ──────────────────────
        london_bars = pd.DataFrame()
        if current_date is not None and "time" in df_h1.columns:
            try:
                raw_times = pd.to_datetime(df_h1["time"])
                if raw_times.dt.tz is None:
                    raw_times = raw_times.dt.tz_localize("UTC")
                else:
                    raw_times = raw_times.dt.tz_convert("UTC")
                mask = (
                    (raw_times.dt.date == current_date) &
                    (raw_times.dt.hour >= LONDON_START_UTC) &
                    (raw_times.dt.hour <  LONDON_END_UTC)
                )
                london_bars = df_h1[mask]
                # Log the first few timestamps from df_h1 so we can see if MT5 timezone offset is an issue
                sample_times = raw_times.tail(10).dt.strftime("%Y-%m-%dT%H:%M").tolist()
                logger.info(
                    "NYMomentumWTI [%s %s] last 10 bar UTC times: %s",
                    symbol, direction.upper(), sample_times,
                )
                logger.info(
                    "NYMomentumWTI [%s %s] London bars found: %d (date=%s, hours %d–%d UTC)",
                    symbol, direction.upper(), len(london_bars), current_date,
                    LONDON_START_UTC, LONDON_END_UTC,
                )
            except Exception as exc:
                logger.warning("NYMomentumWTI [%s %s] London bar filter error: %s", symbol, direction.upper(), exc)

        if len(london_bars) < MIN_LONDON_BARS:
            return _reject(
                f"Too few London bars today ({len(london_bars)} < {MIN_LONDON_BARS}) — range undefined",
            )

        london_high  = float(london_bars["high"].max())
        london_low   = float(london_bars["low"].min())
        london_range = london_high - london_low

        # ── Range sanity check ────────────────────────────────────────────────
        atr = self._atr(df_h1)
        if atr <= 0:
            return _reject("ATR calculation failed")

        logger.info(
            "NYMomentumWTI [%s %s] London range: %.3f–%.3f (range=%.3f, ATR=%.3f, min=%.3f)",
            symbol, direction.upper(),
            london_low, london_high, london_range, atr, atr * MIN_RANGE_ATR_PCT,
        )

        if london_range < atr * MIN_RANGE_ATR_PCT:
            return _reject(
                f"London range too tight ({london_range:.3f} < {atr * MIN_RANGE_ATR_PCT:.3f}) — flat session",
            )

        # ── Breakout check ────────────────────────────────────────────────────
        price    = float(df_h1["close"].iloc[-1])
        bar_high = float(df_h1["high"].iloc[-1])
        bar_low  = float(df_h1["low"].iloc[-1])

        logger.info(
            "NYMomentumWTI [%s %s] current bar: close=%.3f high=%.3f low=%.3f",
            symbol, direction.upper(), price, bar_high, bar_low,
        )

        if is_long:
            if bar_high <= london_high:
                return _reject(
                    f"No London high breakout — bar high {bar_high:.3f} ≤ london high {london_high:.3f}",
                )
            breakout_level = london_high
            sl             = london_low
        else:
            if bar_low >= london_low:
                return _reject(
                    f"No London low breakout — bar low {bar_low:.3f} ≥ london low {london_low:.3f}",
                )
            breakout_level = london_low
            sl             = london_high

        sl_dist = abs(price - sl)
        if sl_dist <= 0:
            return _reject("Zero SL distance — invalid levels")

        # ── Entry mode: retest OR fresh breakout ──────────────────────────────
        dist_to_level = abs(price - breakout_level)

        is_retest   = dist_to_level <= atr * RETEST_TOLERANCE
        is_breakout = dist_to_level <= atr * BREAKOUT_CHASE

        logger.info(
            "NYMomentumWTI [%s %s] breakout_level=%.3f dist=%.3f ATR=%.3f "
            "retest_thresh=%.3f chase_thresh=%.3f is_retest=%s is_breakout=%s",
            symbol, direction.upper(),
            breakout_level, dist_to_level, atr,
            atr * RETEST_TOLERANCE, atr * BREAKOUT_CHASE,
            is_retest, is_breakout,
        )

        if not is_breakout:
            return _reject(
                f"Price {price:.3f} has run too far from {breakout_level:.3f} "
                f"(dist={dist_to_level:.3f} > {atr * BREAKOUT_CHASE:.3f}) — chasing",
            )

        # ── Entry, SL, TP ─────────────────────────────────────────────────────
        entry    = price
        tp1, tp2 = self._calc_tps(entry, sl, direction, rr1=2.0, rr2=5.0)

        # ── HTF bias ──────────────────────────────────────────────────────────
        htf_ok, htf_reason = self._htf_bias(df_h4, df_d1, direction)

        # ── Volume spike confirmation ─────────────────────────────────────────
        vol_ok    = True
        vol_ratio = 1.0
        if "volume" in df_h1.columns:
            try:
                avg_vol   = float(df_h1["volume"].tail(20).mean())
                cur_vol   = float(df_h1["volume"].iloc[-1])
                vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
                vol_ok    = vol_ratio >= 1.1
            except Exception:
                vol_ok = True

        # ── Closed beyond level (bar closed on the breakout side) ─────────────
        closed_beyond = (is_long and price > london_high) or (not is_long and price < london_low)

        # ── Score ─────────────────────────────────────────────────────────────
        score = 0.0

        # 1. Session freshness (first 2 hours of NY window = best)
        hours_into = (current_hour - NY_START_UTC) if current_hour is not None else 2
        score += 2.5 if hours_into == 0 else (1.5 if hours_into == 1 else 0.5)

        # 2. London range quality (well-defined range = higher confidence)
        range_pct = london_range / atr
        score += min(range_pct * 2.0, 2.0)

        # 3. Entry quality
        if is_retest:
            retest_q = max(1.0 - (dist_to_level / (atr * RETEST_TOLERANCE)), 0.0)
            score += 1.0 + retest_q * 1.0   # 1.0–2.0 for a retest
        else:
            score += 0.5                      # fresh breakout, no pull-back yet

        # 4. HTF alignment
        score += 1.5 if htf_ok else 0.0

        # 5. Volume spike
        score += 1.0 if vol_ok else 0.0

        # 6. Bar closed beyond level (conviction)
        score += 1.0 if closed_beyond else 0.0

        mode = "retest" if is_retest else "fresh-breakout"
        reasons = [
            f"London range {london_low:.3f}–{london_high:.3f}  "
            f"(range={london_range:.3f}, ATR={atr:.3f}, {len(london_bars)} bars)",
            f"{'Bullish' if is_long else 'Bearish'} breakout of London "
            f"{'high' if is_long else 'low'} at {breakout_level:.3f}  [{mode}]",
            f"Volume {'spike ×{:.1f}'.format(vol_ratio) if vol_ok else 'weak ×{:.1f}'.format(vol_ratio)}",
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
                "entry_mode":        mode,
                "closed_beyond":     closed_beyond,
                "vol_ratio":         round(vol_ratio, 2),
                "vol_ok":            vol_ok,
                "htf_ok":            htf_ok,
                "hour_utc":          current_hour,
                "london_bars_count": len(london_bars),
            },
        )
