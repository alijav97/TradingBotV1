"""
strategies/london_breakout.py — London Open Breakout strategy.

Logic:
  1. Define Asian Range = H1 bars from 00:00–07:00 UTC
  2. Entry when H1 closes ABOVE Asian High (long) or BELOW Asian Low (short)
  3. Must be during London open window: 07:00–11:00 UTC (11:00–15:00 GST)
  4. Minimum range size to avoid false breakouts (≥ 0.5× ATR)
  5. SL: inside Asian range (just below Asian high for long, above Asian low for short)
  6. TP1: Asian range × 1.5; TP2: Asian range × 3.0

Best instruments: XAUUSD, GBPJPY
"""
from __future__ import annotations

import logging
from datetime import timezone

import pandas as pd

from v2.signals.strategies.base import StrategyBase, StrategyResult

logger = logging.getLogger(__name__)

LONDON_OPEN_UTC  = 7    # 07:00 UTC
LONDON_CLOSE_UTC = 11   # Stop looking after 11:00 UTC (trades become chasing)
MIN_RANGE_ATR_MULT = 0.5  # Asian range must be at least 0.5× ATR


class LondonBreakoutStrategy(StrategyBase):
    name        = "london_breakout"
    instruments = ["XAUUSD", "GBPJPY"]
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

        if symbol not in self.instruments:
            return self._no_signal(symbol, direction, "London breakout: wrong instrument")
        if len(df_h1) < self.min_df_bars:
            return self._no_signal(symbol, direction, "Insufficient H1 bars")

        is_long = direction.lower() in ("long", "buy")

        # ── Session timing check ──────────────────────────────────────────────
        last_time = df_h1["time"].iloc[-1] if "time" in df_h1.columns else None
        current_hour = None
        if last_time is not None:
            try:
                if hasattr(last_time, "tzinfo") and last_time.tzinfo:
                    current_hour = last_time.astimezone(timezone.utc).hour
                else:
                    current_hour = pd.Timestamp(last_time, tz="UTC").hour
            except Exception:
                pass

        if current_hour is not None:
            if not (LONDON_OPEN_UTC <= current_hour < LONDON_CLOSE_UTC):
                return self._no_signal(symbol, direction,
                                       f"Not London open window (UTC {current_hour:02d}:xx)")

        # ── Extract Asian session bars (00:00–07:00 UTC) ──────────────────────
        asian_bars = self._get_asian_bars(df_h1)
        if len(asian_bars) < 3:
            return self._no_signal(symbol, direction, "Not enough Asian session bars (need ≥3)")

        asian_high = float(asian_bars["high"].max())
        asian_low  = float(asian_bars["low"].min())
        asian_range = asian_high - asian_low

        atr = self._atr(df_h1)
        if asian_range < atr * MIN_RANGE_ATR_MULT:
            return self._no_signal(symbol, direction,
                                   f"Asian range {asian_range:.2f} < {atr * MIN_RANGE_ATR_MULT:.2f} (too tight)")

        # ── Breakout confirmation ─────────────────────────────────────────────
        current_close = float(df_h1["close"].iloc[-1])
        current_open  = float(df_h1["open"].iloc[-1])

        broke_long  = current_close > asian_high and current_close > current_open
        broke_short = current_close < asian_low  and current_close < current_open

        if is_long and not broke_long:
            return self._no_signal(symbol, direction,
                                   f"No bullish breakout above Asian high {asian_high:.2f} (close={current_close:.2f})")
        if not is_long and not broke_short:
            return self._no_signal(symbol, direction,
                                   f"No bearish breakout below Asian low {asian_low:.2f} (close={current_close:.2f})")

        # ── HTF check ─────────────────────────────────────────────────────────
        htf_ok, htf_reason = self._htf_bias(df_h4, df_d1, direction)
        # HTF is advisory for London breakout (session momentum can override)
        htf_score = 2.0 if htf_ok else 0.5

        # ── Entry, SL, TP ─────────────────────────────────────────────────────
        entry = current_close
        if is_long:
            stop_loss = round(asian_high - atr * 0.2, 5)   # just inside Asian high
        else:
            stop_loss = round(asian_low  + atr * 0.2, 5)   # just inside Asian low

        tp1 = round(entry + asian_range * 1.5, 5) if is_long else round(entry - asian_range * 1.5, 5)
        tp2 = round(entry + asian_range * 3.0, 5) if is_long else round(entry - asian_range * 3.0, 5)

        # ── Score ─────────────────────────────────────────────────────────────
        rr = abs(entry - tp1) / max(abs(entry - stop_loss), 1e-9)
        range_mult = asian_range / max(atr, 1e-9)

        score = 0.0
        score += 3.0                                          # breakout confirmed
        score += htf_score                                    # HTF alignment
        score += min(range_mult * 1.5, 2.5)                  # larger range = better breakout
        score += 1.5 if current_hour == LONDON_OPEN_UTC else 1.0  # first hour premium
        score += 1.0 if rr >= 2.0 else 0.5

        reasons = [
            f"Asian range {asian_high:.2f}–{asian_low:.2f} ({asian_range:.2f} pts, {asian_range/atr:.1f}× ATR)",
            f"Breakout {'above' if is_long else 'below'} at {current_close:.2f}",
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
                "asian_high":  asian_high,
                "asian_low":   asian_low,
                "asian_range": round(asian_range, 5),
                "htf_ok":      htf_ok,
                "hour_utc":    current_hour,
            },
        )

    # ── Helper ────────────────────────────────────────────────────────────────

    def _get_asian_bars(self, df_h1: pd.DataFrame) -> pd.DataFrame:
        """Return H1 bars belonging to the Asian session (00:00–07:00 UTC)."""
        if "time" not in df_h1.columns:
            return df_h1.iloc[:7]  # fallback: first 7 bars

        try:
            times = pd.to_datetime(df_h1["time"], utc=True)
            hours = times.dt.hour
            # Get today's Asian bars (last 24h window to avoid yesterday's)
            last_ts = times.iloc[-1]
            cutoff  = last_ts.normalize()   # midnight UTC of current day
            mask    = (times >= cutoff) & (hours < 7)
            asian   = df_h1[mask]
            if len(asian) < 2:
                # Fallback: any bar today with hour < 7
                mask = hours < 7
                asian = df_h1[mask].tail(7)
            return asian
        except Exception:
            return df_h1.iloc[:7]
