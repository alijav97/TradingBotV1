"""
strategies/crypto_cipher.py — Market Cipher B (WaveTrend) for BTC/ETH.

WaveTrend oscillator fires when:
  1. WT1 is in extreme zone (< -60 long, > 60 short)
  2. WT1 crosses above WT2 (long) or below WT2 (short)
  3. RSI not overbought/oversold against direction
  4. HTF agrees (H4 or D1)

This is the primary strategy for BTCUSDT and ETHUSDT.
"""
from __future__ import annotations

import logging

import pandas as pd

from v2.signals.strategies.base import StrategyBase, StrategyResult

logger = logging.getLogger(__name__)

WT_OB_LEVEL = 60.0     # WaveTrend overbought threshold
WT_OS_LEVEL = -60.0    # WaveTrend oversold threshold
WT_N1 = 10
WT_N2 = 21


def _wave_trend(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute WaveTrend WT1 and WT2 oscillators."""
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    esa  = hlc3.ewm(span=WT_N1, adjust=False).mean()
    d    = (hlc3 - esa).abs().ewm(span=WT_N1, adjust=False).mean()
    d_safe = d.replace(0, 1e-10)
    ci   = (hlc3 - esa) / (0.015 * d_safe)
    wt1  = ci.ewm(span=WT_N2, adjust=False).mean()
    wt2  = wt1.rolling(4).mean()
    return wt1, wt2


class CryptoCipherStrategy(StrategyBase):
    name        = "crypto_cipher"
    instruments = ["BTCUSDT", "ETHUSDT"]
    timeframes  = ["H1"]
    min_df_bars = 60

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
            return self._no_signal(symbol, direction, "Crypto Cipher only for BTCUSDT/ETHUSDT")
        if len(df_h1) < self.min_df_bars:
            return self._no_signal(symbol, direction, "Insufficient H1 bars")

        is_long = direction.lower() in ("long", "buy")

        # ── WaveTrend computation ─────────────────────────────────────────────
        try:
            wt1, wt2 = _wave_trend(df_h1)
        except Exception as exc:
            return self._no_signal(symbol, direction, f"WaveTrend computation failed: {exc}")

        wt1_now  = float(wt1.iloc[-1])
        wt2_now  = float(wt2.iloc[-1])
        wt1_prev = float(wt1.iloc[-2])
        wt2_prev = float(wt2.iloc[-2])

        # ── Zone check ────────────────────────────────────────────────────────
        if is_long and wt1_now > WT_OS_LEVEL:
            return self._no_signal(symbol, direction,
                                   f"WT1={wt1_now:.1f} not in oversold zone (need < {WT_OS_LEVEL})")
        if not is_long and wt1_now < WT_OB_LEVEL:
            return self._no_signal(symbol, direction,
                                   f"WT1={wt1_now:.1f} not in overbought zone (need > {WT_OB_LEVEL})")

        # ── Cross check ───────────────────────────────────────────────────────
        if is_long:
            crossed = wt1_prev < wt2_prev and wt1_now > wt2_now
        else:
            crossed = wt1_prev > wt2_prev and wt1_now < wt2_now

        if not crossed:
            return self._no_signal(symbol, direction,
                                   f"WT cross not confirmed (WT1={wt1_now:.1f}, WT2={wt2_now:.1f})")

        # ── RSI sanity check ─────────────────────────────────────────────────
        rsi_ok = True
        rsi_val = 50.0
        try:
            delta = df_h1["close"].diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, 1e-10)
            rsi   = 100 - (100 / (1 + rs))
            rsi_val = float(rsi.iloc[-1])
            # Don't long if RSI already overbought; don't short if oversold
            if is_long and rsi_val > 75:
                rsi_ok = False
            if not is_long and rsi_val < 25:
                rsi_ok = False
        except Exception:
            pass

        if not rsi_ok:
            return self._no_signal(symbol, direction,
                                   f"RSI={rsi_val:.1f} contradicts {direction} entry")

        # ── HTF check ─────────────────────────────────────────────────────────
        htf_ok, htf_reason = self._htf_bias(df_h4, df_d1, direction)

        # ── Funding rate bonus (from context if available) ────────────────────
        funding = (context or {}).get("funding_rate", 0.0)
        funding_aligned = (is_long and funding < -0.0005) or (not is_long and funding > 0.0005)

        # ── Entry, SL, TP ─────────────────────────────────────────────────────
        atr   = self._atr(df_h1)
        price = float(df_h1["close"].iloc[-1])

        # SL at swing low/high
        lookback = df_h1.iloc[-10:]
        if is_long:
            swing_level = float(lookback["low"].min())
            stop_loss   = round(swing_level - atr * 0.3, 2)
        else:
            swing_level = float(lookback["high"].max())
            stop_loss   = round(swing_level + atr * 0.3, 2)

        tp1, tp2 = self._calc_tps(price, stop_loss, direction, rr1=2.0, rr2=4.0)

        # ── Score ─────────────────────────────────────────────────────────────
        extremeness = abs(wt1_now) / 60.0   # how deep in OB/OS
        score = 0.0
        score += 3.0                                 # cross in extreme zone
        score += min(extremeness * 2.0, 2.5)         # deeper = stronger
        score += 2.0 if htf_ok else 0.5
        score += 1.0 if rsi_ok else 0.0
        score += 1.0 if funding_aligned else 0.0

        reasons = [
            f"WT1={wt1_now:.1f} cross in {'oversold' if is_long else 'overbought'} zone "
            f"(threshold ±{WT_OS_LEVEL:.0f})",
            f"RSI={rsi_val:.1f} — not contradicting",
            htf_reason,
        ]
        if funding_aligned:
            reasons.append(f"Funding rate {funding:.4f} supports {direction}")

        return StrategyResult(
            signal=True,
            strategy_name=self.name,
            symbol=symbol,
            direction=direction,
            score=round(min(score, 10.0), 1),
            entry_price=round(price, 2),
            stop_loss=stop_loss,
            tp1_price=tp1,
            tp2_price=tp2,
            reasons=[r for r in reasons if r],
            factors={
                "wt1":             round(wt1_now, 2),
                "wt2":             round(wt2_now, 2),
                "crossed":         crossed,
                "rsi":             round(rsi_val, 1),
                "htf_ok":          htf_ok,
                "funding_aligned": funding_aligned,
            },
        )
