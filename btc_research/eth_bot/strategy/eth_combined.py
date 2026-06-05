"""
btc_research/eth_bot/strategy/eth_combined.py — ETH Bot multi-confluence strategy.

Derived from 6-year ETH backtest results (23 strategies, 43,955 signals).

== STRATEGY DESIGN ==

  PRIMARY PATH — hours 14, 15, 16 UTC (London Close / NY session):
    Swing Break[20] AND Keltner Channel[EMA20 ± 2×ATR14] — BOTH must agree.

    Rationale from backtest:
      - Keltner Channel at 15:00 UTC alone: WR=53.7%, AvgR=+0.508, N=82
      - Keltner Channel at 14:00 UTC alone: WR=48.3%, AvgR=+0.498, N=87
      - Swing Break at 14-16 UTC: highest volume (N=400-455), consistent edge
      - Requiring BOTH filters removes most false breakouts →
        expected combined WR 55-65% (vs. 38-53% individually)

  SECONDARY PATH — hour 02 UTC (Asia Night):
    RSI(14) 50-cross in the EMA200 direction.

    Rationale from backtest:
      - RSI 50-Cross at 02:00 UTC alone: WR=50.9%, AvgR=+0.754, PF=2.54, N=55
      - Already above the 50% target on its own

== PARAMETERS ==
  _SWING_LOOKBACK  = 20 bars   — recent structure high / low
  _KELT_EMA_PERIOD = 20 bars   — Keltner middle band (EMA)
  _KELT_ATR_MULT   = 2.0       — Keltner band width (× ATR14)
  _ATR_PERIOD      = 14 bars   — ATR for Keltner + SL padding
  _RSI_PERIOD      = 14 bars   — RSI for 50-cross path
  _SL_PADDING_ATR  = 0.25      — buffer beyond swing level for SL

== NOTE ON BTC ALIGNMENT ==
  Backtest showed BTC direction filter adds only +0.5% WR globally.
  Not implemented in live bot (would require a second DataFeed call inside
  generate_signal — complexity not justified by the marginal gain).

== INTERFACE ==
  ETHStrategy.generate_signal(df_window, bar_time, direction) → dict
  Signal engine already handles: EMA200 direction filter, ADX ≥ 20 gate,
  kill-zone gating, risk sizing, TP price calculation.
"""
from __future__ import annotations

import logging

import pandas as pd
from btc_research.strategies.base import BTCStrategy
from btc_research.eth_bot.settings import (
    TP1_RR, TP2_RR,
    ADX_SPLIT_EARLY_MAX, ADX_SPLIT_STRONG_MIN,
    RISK_PCT_EARLY_TREND, RISK_PCT_TRANSITION, RISK_PCT_STRONG,
)

logger = logging.getLogger(__name__)

# ── Strategy constants ─────────────────────────────────────────────────────────
_SWING_LOOKBACK   = 20      # bars used for swing high / low
_KELT_EMA_PERIOD  = 20      # Keltner middle band EMA period
_KELT_ATR_MULT    = 2.0     # Keltner band width multiplier (× ATR)
_ATR_PERIOD       = 14      # ATR period (Keltner bands + SL padding)
_RSI_PERIOD       = 14      # RSI period for 50-cross path
_SL_PADDING_ATR   = 0.25    # SL buffer = swing_level ± _SL_PADDING_ATR × ATR

# Session routing — must stay in sync with settings.KZ_HOURS = [2, 14, 15, 16]
_PRIMARY_HOURS   = frozenset({14, 15, 16})   # Swing + Keltner confluence
_SECONDARY_HOURS = frozenset({2})            # RSI 50-cross


# ── Public helper ──────────────────────────────────────────────────────────────

def get_risk_pct(adx: float) -> float:
    """
    ADX-split risk sizing (shared with signal_engine).
      ADX ≤ 25  → 3%  (early trend)
      ADX 25-40 → 2%  (transition / dead zone)
      ADX ≥ 40  → 3%  (strong trend)
    """
    if adx >= ADX_SPLIT_STRONG_MIN:
        return RISK_PCT_STRONG
    elif adx <= ADX_SPLIT_EARLY_MAX:
        return RISK_PCT_EARLY_TREND
    else:
        return RISK_PCT_TRANSITION


# ── Indicator helpers (self-contained, no external dependencies) ───────────────

def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range → ATR(period) simple rolling mean."""
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _calc_keltner(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 14,
    mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (upper, mid, lower) Keltner Channel Series."""
    mid   = df["close"].astype(float).ewm(span=ema_period, adjust=False).mean()
    atr_s = _calc_atr(df, atr_period)
    upper = mid + mult * atr_s
    lower = mid - mult * atr_s
    return upper, mid, lower


def _calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Standard Wilder RSI on close."""
    delta = df["close"].astype(float).diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


# ── Strategy class ─────────────────────────────────────────────────────────────

class ETHStrategy(BTCStrategy):
    """
    Multi-confluence ETH strategy derived from 6-year backtest evidence.

    Hour routing:
      14, 15, 16 UTC → _swing_keltner()   (Swing Break + Keltner Channel)
      02 UTC         → _rsi50_cross()     (RSI 14 crosses above/below 50)

    The signal_engine already:
      • Gates by kill-zone hours (KZ_HOURS)
      • Applies EMA200 direction filter (direction arg reflects this)
      • Applies ADX ≥ 20 gate
      • Handles risk sizing, TP price computation, and position sizing
    """

    name        = "ETH: Swing+Keltner [14-16 UTC] | RSI50-Cross [02 UTC]"
    description = (
        "14-16 UTC: Swing Break[20] + Keltner[EMA20 ± 2×ATR14] (both required) | "
        "02 UTC: RSI14 50-cross"
    )

    def __init__(self) -> None:
        pass   # no sub-strategies to construct

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        """
        Evaluate entry conditions for the current bar.

        Args:
            df_window : Last ~300 ETHUSD H1 bars with columns open/high/low/close/volume
            bar_time  : UTC timestamp of the last bar
            direction : "long" or "short" (EMA200-filtered by signal_engine)

        Returns dict with keys:
            signal, entry, sl, tp1_rr, tp2_rr, strategy_used, reason, entry_type
        """
        base = {
            "signal":        False,
            "entry":         0.0,
            "sl":            0.0,
            "tp1_rr":        TP1_RR,
            "tp2_rr":        TP2_RR,
            "strategy_used": None,
            "reason":        "no pattern",
            "entry_type":    "",
        }

        min_bars = _SWING_LOOKBACK + _ATR_PERIOD + 10
        if len(df_window) < min_bars:
            base["reason"] = f"insufficient bars ({len(df_window)} < {min_bars})"
            return base

        hour    = bar_time.hour
        is_long = direction == "long"

        if hour in _PRIMARY_HOURS:
            return self._swing_keltner(df_window, is_long, base)
        elif hour in _SECONDARY_HOURS:
            return self._rsi50_cross(df_window, is_long, base)
        else:
            base["reason"] = (
                f"UTC {hour:02d} not in strategy hours "
                f"(primary={sorted(_PRIMARY_HOURS)}, secondary={sorted(_SECONDARY_HOURS)})"
            )
            return base

    # ── Sub-strategies ─────────────────────────────────────────────────────────

    def _swing_keltner(
        self,
        df: pd.DataFrame,
        is_long: bool,
        base: dict,
    ) -> dict:
        """
        PRIMARY PATH (14-16 UTC) — Swing Break + Keltner Channel.

        LONG  entry: close > 20-bar swing_high  AND  close > kelt_upper
        SHORT entry: close < 20-bar swing_low   AND  close < kelt_lower

        SL placement:
          Long:  swing_low  - 0.25 × ATR14   (below the swing structure)
          Short: swing_high + 0.25 × ATR14   (above the swing structure)
        """
        try:
            close = df["close"].astype(float)
            high  = df["high"].astype(float)
            low   = df["low"].astype(float)

            # Indicators computed on the full window
            atr_s                       = _calc_atr(df, _ATR_PERIOD)
            kelt_upper, _, kelt_lower   = _calc_keltner(
                df, _KELT_EMA_PERIOD, _ATR_PERIOD, _KELT_ATR_MULT
            )

            curr_close  = float(close.iloc[-1])
            curr_kelt_u = float(kelt_upper.iloc[-1])
            curr_kelt_l = float(kelt_lower.iloc[-1])
            curr_atr    = float(atr_s.iloc[-1])

            if curr_atr <= 0 or pd.isna(curr_atr):
                base["reason"] = "ATR unavailable"
                return base

            # Swing levels from the _SWING_LOOKBACK bars BEFORE the current bar
            # (exclude current bar to avoid lookahead)
            prior_hi = float(high.iloc[-_SWING_LOOKBACK - 1 : -1].max())
            prior_lo = float(low.iloc[-_SWING_LOOKBACK - 1 : -1].min())

            if is_long:
                swing_ok   = curr_close > prior_hi
                keltner_ok = curr_close > curr_kelt_u

                if not swing_ok or not keltner_ok:
                    fails = []
                    if not swing_ok:
                        fails.append(f"close {curr_close:.2f} ≤ swing_hi {prior_hi:.2f}")
                    if not keltner_ok:
                        fails.append(f"close {curr_close:.2f} ≤ kelt_upper {curr_kelt_u:.2f}")
                    base["reason"] = "swing+keltner LONG miss: " + " | ".join(fails)
                    return base

                entry   = curr_close
                sl      = round(prior_lo - _SL_PADDING_ATR * curr_atr, 2)
                sl_dist = entry - sl

                reason = (
                    f"LONG: close {curr_close:.2f} > swing_hi {prior_hi:.2f}"
                    f" + kelt_upper {curr_kelt_u:.2f}"
                )

            else:  # short
                swing_ok   = curr_close < prior_lo
                keltner_ok = curr_close < curr_kelt_l

                if not swing_ok or not keltner_ok:
                    fails = []
                    if not swing_ok:
                        fails.append(f"close {curr_close:.2f} ≥ swing_lo {prior_lo:.2f}")
                    if not keltner_ok:
                        fails.append(f"close {curr_close:.2f} ≥ kelt_lower {curr_kelt_l:.2f}")
                    base["reason"] = "swing+keltner SHORT miss: " + " | ".join(fails)
                    return base

                entry   = curr_close
                sl      = round(prior_hi + _SL_PADDING_ATR * curr_atr, 2)
                sl_dist = sl - entry

                reason = (
                    f"SHORT: close {curr_close:.2f} < swing_lo {prior_lo:.2f}"
                    f" + kelt_lower {curr_kelt_l:.2f}"
                )

            if sl_dist <= 0:
                base["reason"] = f"zero/negative SL distance ({sl_dist:.4f})"
                return base

            return {
                "signal":        True,
                "entry":         round(entry, 2),
                "sl":            sl,
                "tp1_rr":        TP1_RR,
                "tp2_rr":        TP2_RR,
                "strategy_used": "Swing+Keltner",
                "reason":        reason,
                "entry_type":    "swing_keltner_break",
            }

        except Exception as exc:
            logger.warning("_swing_keltner error: %s", exc, exc_info=True)
            base["reason"] = f"swing_keltner error: {exc}"
            return base

    def _rsi50_cross(
        self,
        df: pd.DataFrame,
        is_long: bool,
        base: dict,
    ) -> dict:
        """
        SECONDARY PATH (02 UTC) — RSI(14) 50-cross.

        LONG  entry: RSI was below 50 on prior bar, now ≥ 50 (bullish momentum cross)
        SHORT entry: RSI was above 50 on prior bar, now ≤ 50 (bearish momentum cross)

        SL placement:
          Long:  10-bar swing_low  - 0.25 × ATR14
          Short: 10-bar swing_high + 0.25 × ATR14
        """
        try:
            close = df["close"].astype(float)
            high  = df["high"].astype(float)
            low   = df["low"].astype(float)
            atr_s = _calc_atr(df, _ATR_PERIOD)
            rsi_s = _calc_rsi(df, _RSI_PERIOD)

            curr_rsi = float(rsi_s.iloc[-1])
            prev_rsi = float(rsi_s.iloc[-2])
            curr_atr = float(atr_s.iloc[-1])

            if pd.isna(curr_rsi) or pd.isna(prev_rsi):
                base["reason"] = "RSI unavailable (insufficient warm-up bars)"
                return base
            if curr_atr <= 0 or pd.isna(curr_atr):
                base["reason"] = "ATR unavailable"
                return base

            curr_close = float(close.iloc[-1])

            if is_long:
                crossed = prev_rsi < 50.0 <= curr_rsi
                if not crossed:
                    base["reason"] = (
                        f"RSI no cross above 50 (prev={prev_rsi:.1f}, curr={curr_rsi:.1f})"
                    )
                    return base

                sl_ref  = float(low.iloc[-10:].min())
                entry   = curr_close
                sl      = round(sl_ref - _SL_PADDING_ATR * curr_atr, 2)
                sl_dist = entry - sl
                reason  = (
                    f"LONG: RSI crossed above 50 ({prev_rsi:.1f} → {curr_rsi:.1f})"
                )

            else:  # short
                crossed = prev_rsi > 50.0 >= curr_rsi
                if not crossed:
                    base["reason"] = (
                        f"RSI no cross below 50 (prev={prev_rsi:.1f}, curr={curr_rsi:.1f})"
                    )
                    return base

                sl_ref  = float(high.iloc[-10:].max())
                entry   = curr_close
                sl      = round(sl_ref + _SL_PADDING_ATR * curr_atr, 2)
                sl_dist = sl - entry
                reason  = (
                    f"SHORT: RSI crossed below 50 ({prev_rsi:.1f} → {curr_rsi:.1f})"
                )

            if sl_dist <= 0:
                base["reason"] = f"zero/negative SL distance ({sl_dist:.4f})"
                return base

            return {
                "signal":        True,
                "entry":         round(entry, 2),
                "sl":            sl,
                "tp1_rr":        TP1_RR,
                "tp2_rr":        TP2_RR,
                "strategy_used": "RSI50-Cross",
                "reason":        reason,
                "entry_type":    "rsi50_cross",
            }

        except Exception as exc:
            logger.warning("_rsi50_cross error: %s", exc, exc_info=True)
            base["reason"] = f"rsi50_cross error: {exc}"
            return base

    @property
    def strategy_names(self) -> list[str]:
        return [
            "Swing+Keltner (14-16 UTC)",
            "RSI50-Cross   (02 UTC)",
        ]
