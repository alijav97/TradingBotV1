"""
btc_research/eth_bot/strategy/eth_combined.py — ETH Bot multi-confluence strategy.

Confirmed by 6-year ETH backtest (25 strategies × all hours, MaxDD-gated expansion):

  PATH A — 02 UTC  (Asia Night):       RSI(14) 50-cross
    WR=50.0%  AvgR=+0.786  PF=2.57  N=46  (BTC-aligned baseline)

  PATH B — 06 UTC  (EU Pre-Open):      MACD cross + ADX ≥ 25
    WR=45.0%  AvgR=+0.660  PF=2.65  N=20  (BTC-aligned baseline)
           15 UTC  (NY Open):           MACD+ADX fallback after Keltner check
    WR=45.2%  AvgR=+0.774  PF=2.50  N=31

  PATH C — 10 UTC  (EU Mid-Session):   RSI(14) 50-cross + EMA9/21 stack + ADX≥25
    WR=48.1%  AvgR=+0.706  PF=2.36  N=27  (ADX 20-25 near-random → gated)

  PATH D — 14 UTC  (NY Pre-Open):      Keltner Channel fresh breakout [primary]
    WR=48.3%  AvgR=+0.450  PF=1.89  N=87
           15 UTC  (NY Open):           Keltner Channel fresh breakout [primary]
    WR=53.7%  AvgR=+0.492  PF=2.06  N=82  ← Best single expansion slot

  PATH E — 05 UTC  (Asia Morning):     EMA 9/21 golden/death cross [standalone]
    WR=52.4%  AvgR=+0.787  PF=3.07  N=21
           14 UTC  (NY Pre-Open):       EMA 9/21 cross [fallback after Keltner check]
    WR=44.7%  AvgR=+0.430  PF=1.78  N=38

== HOUR ROUTING ==
  H02 → Path A (RSI 50-cross)
  H05 → Path E (EMA 9/21 cross — Asia Morning, standalone)
  H06 → Path B (MACD+ADX)
  H10 → Path C (RSI+EMA Stack, ADX≥25 gate)
  H14 → Path D first (Keltner), fallback Path E (EMA cross) if Keltner doesn't fire
  H15 → Path D first (Keltner), fallback Path B (MACD+ADX) if Keltner doesn't fire

  BTC alignment: Signal engine checks BTC EMA200 direction before calling generate_signal().
  KZ_HOURS = [2, 5, 6, 10, 14, 15] — set in settings.py.

== RISK SIZING (Config D — confirmed optimal by 6yr ETH ADX sweep) ==
  ADX ≤ 25  → 2%  (early/weak trend — WR 45%, AvgR +0.34 → bet light)
  ADX 25-40 → 3%  (sweet spot — WR 47-60%, AvgR +0.87 to +1.64)
  ADX ≥ 40  → 5%  (strong trend — WR 60%, high conviction)

== TP / SL ==
  TP1 = 2R  (close 50%, SL to breakeven)
  TP2 = 4R  (close remainder)
  Trailing SL = price ± 2×ATR after TP1

== COMPOUND PROJECTION ==
  8-slot portfolio (3 base + 5 expansion):
    CAGR ≈ +168%  |  5yr from $500 ≈ $69,255  |  MaxDD ≈ −33.5%
    Trades ≈ 4.9/month (up from 1.3 baseline)
"""
from __future__ import annotations

import logging

import pandas as pd
from btc_research.strategies.base import BTCStrategy
from btc_research.eth_bot.settings import (
    TP1_RR, TP2_RR,
    ADX_SPLIT_EARLY_MAX, ADX_SPLIT_STRONG_MIN,
    RISK_PCT_EARLY_TREND, RISK_PCT_TRANSITION, RISK_PCT_STRONG,
    KZ_HOURS, RSI_EMA_ADX_MIN,
)

logger = logging.getLogger(__name__)

# ── Strategy constants ─────────────────────────────────────────────────────────
_RSI_PERIOD   = 14
_MACD_FAST    = 12
_MACD_SLOW    = 26
_MACD_SIG     = 9
_EMA9_PERIOD  = 9
_EMA21_PERIOD = 21
_ADX_MIN_B    = 25      # Path B / Path D MACD+ADX: minimum ADX to trade
_ATR_PERIOD   = 14
_SL_ATR_MULT  = 1.5     # default SL = entry ± _SL_ATR_MULT × ATR14

# Keltner Channel parameters (matches backtest run_backtest.py _compute_keltner)
_KELT_EMA_PERIOD = 20   # Keltner midline = EMA20
_KELT_ATR_PERIOD = 10   # Keltner width   = 2 × ATR10
_KELT_ATR_MULT   = 2.0
# Keltner SL sanity bounds (as multiples of ATR14)
_KELT_SL_MIN_ATR = 0.3
_KELT_SL_MAX_ATR = 3.5

# Session routing (stays in sync with settings.KZ_HOURS = [2, 5, 6, 10, 14, 15])
_PATH_A_HOURS = frozenset({2})       # RSI 50-Cross            (Asia Night)
_PATH_B_HOURS = frozenset({6})       # MACD+ADX primary        (EU Pre-Open)
_PATH_C_HOURS = frozenset({10})      # RSI+EMA Stack           (EU Mid-Session)
_PATH_D_HOURS = frozenset({14, 15})  # Keltner fresh breakout  (NY Pre/Open)
_PATH_E_HOURS = frozenset({5})       # EMA 9/21 cross standalone (Asia Morning)
# H14: Keltner primary → EMA cross fallback
# H15: Keltner primary → MACD+ADX fallback


# ── Public helper ──────────────────────────────────────────────────────────────

def get_risk_pct(adx: float) -> float:
    """
    ADX-split risk sizing — Config D, confirmed best by 6yr ETH ADX sweep.
      ADX ≤ 25  → 2%  (early/weak trend zone)
      ADX 25-40 → 3%  (sweet spot — best WR/AvgR bucket)
      ADX ≥ 40  → 5%  (strong trend, high conviction)
    """
    if adx >= ADX_SPLIT_STRONG_MIN:
        return RISK_PCT_STRONG
    elif adx <= ADX_SPLIT_EARLY_MAX:
        return RISK_PCT_EARLY_TREND
    else:
        return RISK_PCT_TRANSITION


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].astype(float).diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _calc_macd(df: pd.DataFrame,
               fast: int = 12, slow: int = 26,
               signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """Return (macd_line, signal_line)."""
    c     = df["close"].astype(float)
    line  = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sig   = line.ewm(span=signal, adjust=False).mean()
    return line, sig


def _calc_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].astype(float).ewm(span=period, adjust=False).mean()


def _calc_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Return ADX value for the last bar."""
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    sp  = 2 * period - 1
    hd  = h.diff();  ld = l.diff()
    tr  = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    pdm = hd.where((hd > 0) & (hd > -ld), 0.0)
    mdm = (-ld).where((-ld > 0) & (-ld > hd), 0.0)
    aw  = tr.ewm(span=sp, adjust=False).mean()
    pw  = pdm.ewm(span=sp, adjust=False).mean()
    mw  = mdm.ewm(span=sp, adjust=False).mean()
    pdi = 100 * pw / (aw + 1e-12)
    ndi = 100 * mw / (aw + 1e-12)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-12)
    return float(dx.ewm(span=sp, adjust=False).mean().fillna(0).iloc[-1])


def _calc_keltner(df: pd.DataFrame,
                  ema_period: int = 20,
                  atr_period: int = 10,
                  atr_mult:   float = 2.0,
                  ) -> tuple[float, float, float]:
    """Return (upper, mid, lower) Keltner values for the last bar.

    Matches backtest: mid = EMA(close, ema_period), width = atr_mult × ATR(atr_period).
    """
    c   = df["close"].astype(float)
    h   = df["high"].astype(float)
    lo  = df["low"].astype(float)

    mid = c.ewm(span=ema_period, adjust=False).mean()

    tr  = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr

    return (
        float(upper.iloc[-1]),
        float(mid.iloc[-1]),
        float(lower.iloc[-1]),
    )


# ── Strategy class ─────────────────────────────────────────────────────────────

class ETHStrategy(BTCStrategy):
    """
    Five-path ETH strategy confirmed by 6-year backtest + MaxDD-gated expansion.

      Path A [02 UTC]: RSI(14) crosses above/below 50                    (Asia Night)
      Path B [06 UTC]: MACD line/signal cross + ADX ≥ 25                (EU Pre-Open)
      Path C [10 UTC]: RSI(14) 50-cross + EMA9/21 stack + ADX ≥ 25     (EU Mid-Session)
      Path D [14/15 UTC]: Keltner Channel fresh breakout (EMA20±2×ATR10)(NY Pre-Open / Open)
      Path E [05 UTC]: EMA 9/21 golden/death cross                       (Asia Morning)

    H14: Path D (Keltner) → fallback Path E (EMA cross) if Keltner doesn't fire
    H15: Path D (Keltner) → fallback Path B (MACD+ADX) if Keltner doesn't fire

    Signal engine gates by: kill-zone hour, EMA200 direction, ADX ≥ 20.
    This class handles entry-pattern logic only.
    """

    name = (
        "ETH: RSI50[02] | EMAxEMA[05] | MACD+ADX[06] | RSI+EMA[10] | "
        "Keltner[14/15] | MACD+ADX[15]"
    )
    description = (
        "02 UTC: RSI14 50-cross | "
        "05 UTC: EMA9/21 cross | "
        "06 UTC: MACD+ADX≥25 | "
        "10 UTC: RSI50+EMA stack (ADX≥25) | "
        "14 UTC: Keltner breakout → EMA cross fallback | "
        "15 UTC: Keltner breakout → MACD+ADX fallback"
    )

    def __init__(self) -> None:
        pass

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        """
        Route to the correct sub-strategy based on UTC hour.

        Args:
            df_window : Last ~300 ETHUSD H1 bars
            bar_time  : UTC timestamp of the latest bar
            direction : "long" or "short" (EMA200-filtered by signal_engine)

        Returns dict: signal, entry, sl, tp1_rr, tp2_rr, strategy_used, reason, entry_type
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

        if len(df_window) < 50:
            base["reason"] = f"insufficient bars ({len(df_window)})"
            return base

        hour    = bar_time.hour
        is_long = direction == "long"

        if hour in _PATH_A_HOURS:
            # H02 — RSI 50-cross
            return self._path_a_rsi50(df_window, is_long, base)

        elif hour in _PATH_E_HOURS:
            # H05 — EMA 9/21 cross (standalone, Asia Morning)
            return self._path_e_ema_cross(df_window, is_long, base)

        elif hour in _PATH_B_HOURS:
            # H06 — MACD+ADX (EU Pre-Open)
            return self._path_b_macd_adx(df_window, is_long, base)

        elif hour in _PATH_C_HOURS:
            # H10 — RSI+EMA Stack (ADX≥25 gate)
            return self._path_c_rsi_ema(df_window, is_long, base)

        elif hour == 14:
            # H14 — Keltner primary → EMA cross fallback
            sig = self._path_d_keltner(df_window, is_long, dict(base))
            if sig["signal"]:
                return sig
            return self._path_e_ema_cross(df_window, is_long, base)

        elif hour == 15:
            # H15 — Keltner primary → MACD+ADX fallback
            sig = self._path_d_keltner(df_window, is_long, dict(base))
            if sig["signal"]:
                return sig
            return self._path_b_macd_adx(df_window, is_long, base)

        else:
            base["reason"] = (
                f"UTC {hour:02d} not in active strategy hours "
                f"(A={sorted(_PATH_A_HOURS)}, B={sorted(_PATH_B_HOURS)}, "
                f"C={sorted(_PATH_C_HOURS)}, D={sorted(_PATH_D_HOURS)}, "
                f"E={sorted(_PATH_E_HOURS)})"
            )
            return base

    # ── Path A — RSI(14) 50-cross [02 UTC] ────────────────────────────────────

    def _path_a_rsi50(self, df: pd.DataFrame, is_long: bool, base: dict) -> dict:
        """
        Asia Night — RSI(14) 50-cross in EMA200 direction.
        LONG : RSI was < 50, now ≥ 50
        SHORT: RSI was > 50, now ≤ 50
        SL   : 10-bar low/high ± 1.5×ATR14
        """
        try:
            rsi_s = _calc_rsi(df, _RSI_PERIOD)
            atr_s = _calc_atr(df, _ATR_PERIOD)

            if len(rsi_s) < 3:
                base["reason"] = "insufficient bars for RSI"
                return base

            curr_rsi = float(rsi_s.iloc[-1])
            prev_rsi = float(rsi_s.iloc[-2])
            curr_atr = float(atr_s.iloc[-1])

            if pd.isna(curr_rsi) or pd.isna(prev_rsi) or curr_atr <= 0 or pd.isna(curr_atr):
                base["reason"] = "RSI/ATR unavailable"
                return base

            curr_close = float(df["close"].astype(float).iloc[-1])

            if is_long:
                if not (prev_rsi < 50.0 <= curr_rsi):
                    base["reason"] = f"RSI no cross above 50 (prev={prev_rsi:.1f} curr={curr_rsi:.1f})"
                    return base
                sl_ref  = float(df["low"].astype(float).iloc[-10:].min())
                sl      = round(sl_ref - _SL_ATR_MULT * curr_atr, 2)
                sl_dist = curr_close - sl
                reason  = f"LONG: RSI crossed above 50 ({prev_rsi:.1f}→{curr_rsi:.1f})"
            else:
                if not (prev_rsi > 50.0 >= curr_rsi):
                    base["reason"] = f"RSI no cross below 50 (prev={prev_rsi:.1f} curr={curr_rsi:.1f})"
                    return base
                sl_ref  = float(df["high"].astype(float).iloc[-10:].max())
                sl      = round(sl_ref + _SL_ATR_MULT * curr_atr, 2)
                sl_dist = sl - curr_close
                reason  = f"SHORT: RSI crossed below 50 ({prev_rsi:.1f}→{curr_rsi:.1f})"

            if sl_dist <= 0:
                base["reason"] = "zero SL distance (path A)"
                return base

            return {
                "signal":        True,
                "entry":         round(curr_close, 2),
                "sl":            sl,
                "tp1_rr":        TP1_RR,
                "tp2_rr":        TP2_RR,
                "strategy_used": "RSI50-Cross",
                "reason":        reason,
                "entry_type":    "rsi50_cross",
            }

        except Exception as exc:
            logger.warning("_path_a_rsi50 error: %s", exc, exc_info=True)
            base["reason"] = f"path_a error: {exc}"
            return base

    # ── Path B — MACD + ADX [06 UTC primary | 15 UTC fallback] ───────────────

    def _path_b_macd_adx(self, df: pd.DataFrame, is_long: bool, base: dict) -> dict:
        """
        MACD line/signal cross + ADX ≥ 25 + MACD line on correct zero side.
        Used at H06 (EU Pre-Open, primary) and H15 (NY Open, Keltner fallback).
        LONG : MACD line crosses above signal AND MACD line > 0 AND ADX ≥ 25
        SHORT: MACD line crosses below signal AND MACD line < 0 AND ADX ≥ 25
        SL   : entry ± 1.5×ATR14
        """
        try:
            if len(df) < 2:
                base["reason"] = "insufficient bars for MACD"
                return base

            macd_l, macd_s = _calc_macd(df, _MACD_FAST, _MACD_SLOW, _MACD_SIG)
            atr_s          = _calc_atr(df, _ATR_PERIOD)

            curr_ml = float(macd_l.iloc[-1]);  prev_ml = float(macd_l.iloc[-2])
            curr_ms = float(macd_s.iloc[-1]);  prev_ms = float(macd_s.iloc[-2])
            curr_atr = float(atr_s.iloc[-1])

            if pd.isna(curr_ml) or pd.isna(prev_ml) or curr_atr <= 0:
                base["reason"] = "MACD/ATR unavailable"
                return base

            adx_val = _calc_adx(df)
            if adx_val < _ADX_MIN_B:
                base["reason"] = f"ADX {adx_val:.1f} < {_ADX_MIN_B} (MACD+ADX threshold)"
                return base

            curr_close = float(df["close"].astype(float).iloc[-1])

            if is_long:
                cross = prev_ml <= prev_ms and curr_ml > curr_ms
                if not cross:
                    base["reason"] = "MACD no bullish cross"
                    return base
                if curr_ml < 0:
                    base["reason"] = f"MACD line {curr_ml:.4f} < 0 (not above zero for long)"
                    return base
                sl      = round(curr_close - _SL_ATR_MULT * curr_atr, 2)
                sl_dist = curr_close - sl
                reason  = (f"LONG: MACD crossed above signal ({prev_ml:.4f}→{curr_ml:.4f})"
                           f" + ADX={adx_val:.1f}")
            else:
                cross = prev_ml >= prev_ms and curr_ml < curr_ms
                if not cross:
                    base["reason"] = "MACD no bearish cross"
                    return base
                if curr_ml > 0:
                    base["reason"] = f"MACD line {curr_ml:.4f} > 0 (not below zero for short)"
                    return base
                sl      = round(curr_close + _SL_ATR_MULT * curr_atr, 2)
                sl_dist = sl - curr_close
                reason  = (f"SHORT: MACD crossed below signal ({prev_ml:.4f}→{curr_ml:.4f})"
                           f" + ADX={adx_val:.1f}")

            if sl_dist <= 0:
                base["reason"] = "zero SL distance (path B)"
                return base

            return {
                "signal":        True,
                "entry":         round(curr_close, 2),
                "sl":            sl,
                "tp1_rr":        TP1_RR,
                "tp2_rr":        TP2_RR,
                "strategy_used": "MACD+ADX",
                "reason":        reason,
                "entry_type":    "macd_adx_cross",
            }

        except Exception as exc:
            logger.warning("_path_b_macd_adx error: %s", exc, exc_info=True)
            base["reason"] = f"path_b error: {exc}"
            return base

    # ── Path C — RSI + EMA Stack [10 UTC] ─────────────────────────────────────

    def _path_c_rsi_ema(self, df: pd.DataFrame, is_long: bool, base: dict) -> dict:
        """
        EU Mid-Session — RSI(14) 50-cross + EMA9/21 stack aligned.
        LONG : RSI crosses above 50 + EMA9 > EMA21
        SHORT: RSI crosses below 50 + EMA9 < EMA21
        SL   : entry ± 1.5×ATR14

        ADX gate: ADX ≥ RSI_EMA_ADX_MIN (default 25).
        ADX 20-25 bucket for rsi_ema is near-worthless: WR=41.7%, AvgR=+0.070.
        Raising the bar to 25 lifts quality to WR≥57%, AvgR≥1.40.
        """
        try:
            adx_now = _calc_adx(df)
            if adx_now < RSI_EMA_ADX_MIN:
                base["reason"] = (
                    f"rsi_ema ADX {adx_now:.1f} < {RSI_EMA_ADX_MIN} "
                    f"(path C requires ADX≥{RSI_EMA_ADX_MIN})"
                )
                return base

            if len(df) < 3:
                base["reason"] = "insufficient bars for RSI+EMA"
                return base

            rsi_s  = _calc_rsi(df, _RSI_PERIOD)
            ema9   = _calc_ema(df, _EMA9_PERIOD)
            ema21  = _calc_ema(df, _EMA21_PERIOD)
            atr_s  = _calc_atr(df, _ATR_PERIOD)

            curr_rsi = float(rsi_s.iloc[-1]);  prev_rsi = float(rsi_s.iloc[-2])
            curr_e9  = float(ema9.iloc[-1]);   curr_e21 = float(ema21.iloc[-1])
            curr_atr = float(atr_s.iloc[-1])
            curr_close = float(df["close"].astype(float).iloc[-1])

            if pd.isna(curr_rsi) or pd.isna(prev_rsi) or curr_atr <= 0:
                base["reason"] = "RSI/EMA/ATR unavailable"
                return base

            if is_long:
                rsi_cross = prev_rsi < 50.0 <= curr_rsi
                ema_stack = curr_e9 > curr_e21
                if not rsi_cross:
                    base["reason"] = f"RSI no cross above 50 ({prev_rsi:.1f}→{curr_rsi:.1f})"
                    return base
                if not ema_stack:
                    base["reason"] = f"EMA stack bearish (EMA9={curr_e9:.2f} < EMA21={curr_e21:.2f})"
                    return base
                sl      = round(curr_close - _SL_ATR_MULT * curr_atr, 2)
                sl_dist = curr_close - sl
                reason  = (f"LONG: RSI crossed above 50 ({prev_rsi:.1f}→{curr_rsi:.1f})"
                           f" + EMA9({curr_e9:.2f})>EMA21({curr_e21:.2f})")
            else:
                rsi_cross = prev_rsi > 50.0 >= curr_rsi
                ema_stack = curr_e9 < curr_e21
                if not rsi_cross:
                    base["reason"] = f"RSI no cross below 50 ({prev_rsi:.1f}→{curr_rsi:.1f})"
                    return base
                if not ema_stack:
                    base["reason"] = f"EMA stack bullish (EMA9={curr_e9:.2f} > EMA21={curr_e21:.2f})"
                    return base
                sl      = round(curr_close + _SL_ATR_MULT * curr_atr, 2)
                sl_dist = sl - curr_close
                reason  = (f"SHORT: RSI crossed below 50 ({prev_rsi:.1f}→{curr_rsi:.1f})"
                           f" + EMA9({curr_e9:.2f})<EMA21({curr_e21:.2f})")

            if sl_dist <= 0:
                base["reason"] = "zero SL distance (path C)"
                return base

            return {
                "signal":        True,
                "entry":         round(curr_close, 2),
                "sl":            sl,
                "tp1_rr":        TP1_RR,
                "tp2_rr":        TP2_RR,
                "strategy_used": "RSI+EMA Stack",
                "reason":        reason,
                "entry_type":    "rsi_ema_stack",
            }

        except Exception as exc:
            logger.warning("_path_c_rsi_ema error: %s", exc, exc_info=True)
            base["reason"] = f"path_c error: {exc}"
            return base

    # ── Path D — Keltner Channel fresh breakout [14 UTC primary | 15 UTC primary]

    def _path_d_keltner(self, df: pd.DataFrame, is_long: bool, base: dict) -> dict:
        """
        NY Pre-Open / NY Open — Keltner Channel fresh breakout.
        Matches backtest _check_keltner exactly:
          Keltner: EMA20 ± 2×ATR10
          LONG : curr_close > kelt_upper  AND  prev_close ≤ kelt_upper  (fresh break above)
          SHORT: curr_close < kelt_lower  AND  prev_close ≥ kelt_lower  (fresh break below)
          SL   : kelt_mid (EMA20)
          SL sanity: 0.3×ATR14 ≤ sl_dist ≤ 3.5×ATR14
        """
        try:
            if len(df) < max(_KELT_EMA_PERIOD, _KELT_ATR_PERIOD) + 2:
                base["reason"] = "insufficient bars for Keltner"
                return base

            # Current bar Keltner levels
            kelt_upper, kelt_mid, kelt_lower = _calc_keltner(
                df, _KELT_EMA_PERIOD, _KELT_ATR_PERIOD, _KELT_ATR_MULT
            )
            # Previous bar Keltner levels (exclude last row, recalculate)
            prev_kelt_upper, _, prev_kelt_lower = _calc_keltner(
                df.iloc[:-1], _KELT_EMA_PERIOD, _KELT_ATR_PERIOD, _KELT_ATR_MULT
            )

            atr14_s    = _calc_atr(df, _ATR_PERIOD)
            curr_atr14 = float(atr14_s.iloc[-1])
            curr_close = float(df["close"].astype(float).iloc[-1])
            prev_close = float(df["close"].astype(float).iloc[-2])

            if pd.isna(kelt_upper) or pd.isna(kelt_mid) or curr_atr14 <= 0:
                base["reason"] = "Keltner/ATR unavailable"
                return base

            if is_long:
                # Fresh breakout above upper band
                if curr_close <= kelt_upper:
                    base["reason"] = f"close {curr_close:.2f} ≤ kelt_upper {kelt_upper:.2f}"
                    return base
                if prev_close > prev_kelt_upper:
                    base["reason"] = f"prev_close {prev_close:.2f} already above kelt_upper (not fresh)"
                    return base
                sl      = round(kelt_mid, 2)
                sl_dist = curr_close - sl
                reason  = (f"LONG: Keltner fresh breakout above "
                           f"{kelt_upper:.2f} (mid={kelt_mid:.2f})")
            else:
                # Fresh breakout below lower band
                if curr_close >= kelt_lower:
                    base["reason"] = f"close {curr_close:.2f} ≥ kelt_lower {kelt_lower:.2f}"
                    return base
                if prev_close < prev_kelt_lower:
                    base["reason"] = f"prev_close {prev_close:.2f} already below kelt_lower (not fresh)"
                    return base
                sl      = round(kelt_mid, 2)
                sl_dist = sl - curr_close
                reason  = (f"SHORT: Keltner fresh breakout below "
                           f"{kelt_lower:.2f} (mid={kelt_mid:.2f})")

            # SL sanity check (matches backtest filter)
            if sl_dist < _KELT_SL_MIN_ATR * curr_atr14:
                base["reason"] = f"SL dist {sl_dist:.2f} < {_KELT_SL_MIN_ATR}×ATR14 (too tight)"
                return base
            if sl_dist > _KELT_SL_MAX_ATR * curr_atr14:
                base["reason"] = f"SL dist {sl_dist:.2f} > {_KELT_SL_MAX_ATR}×ATR14 (too wide)"
                return base

            return {
                "signal":        True,
                "entry":         round(curr_close, 2),
                "sl":            sl,
                "tp1_rr":        TP1_RR,
                "tp2_rr":        TP2_RR,
                "strategy_used": "Keltner-Breakout",
                "reason":        reason,
                "entry_type":    "keltner_breakout",
            }

        except Exception as exc:
            logger.warning("_path_d_keltner error: %s", exc, exc_info=True)
            base["reason"] = f"path_d error: {exc}"
            return base

    # ── Path E — EMA 9/21 golden/death cross [05 UTC | 14 UTC fallback] ───────

    def _path_e_ema_cross(self, df: pd.DataFrame, is_long: bool, base: dict) -> dict:
        """
        Asia Morning (H05) or NY Pre-Open fallback (H14) — EMA 9/21 cross.
        Matches backtest _check_ema_cross exactly:
          LONG : curr EMA9 > EMA21  AND  prev EMA9 ≤ EMA21  (fresh golden cross)
          SHORT: curr EMA9 < EMA21  AND  prev EMA9 ≥ EMA21  (fresh death cross)
          SL   : entry ± 1.5×ATR14
          EMA200 direction already enforced by signal engine.
        """
        try:
            if len(df) < 2:
                base["reason"] = "insufficient bars for EMA cross"
                return base

            ema9  = _calc_ema(df, _EMA9_PERIOD)
            ema21 = _calc_ema(df, _EMA21_PERIOD)
            atr_s = _calc_atr(df, _ATR_PERIOD)

            curr_e9  = float(ema9.iloc[-1]);  prev_e9  = float(ema9.iloc[-2])
            curr_e21 = float(ema21.iloc[-1]); prev_e21 = float(ema21.iloc[-2])
            curr_atr = float(atr_s.iloc[-1])
            curr_close = float(df["close"].astype(float).iloc[-1])

            if pd.isna(curr_e9) or pd.isna(curr_e21) or curr_atr <= 0:
                base["reason"] = "EMA/ATR unavailable"
                return base

            if is_long:
                # Fresh golden cross: EMA9 just crossed above EMA21
                cross = curr_e9 > curr_e21 and prev_e9 <= prev_e21
                if not cross:
                    base["reason"] = (f"EMA9 no golden cross above EMA21 "
                                      f"(curr EMA9={curr_e9:.2f} EMA21={curr_e21:.2f})")
                    return base
                sl      = round(curr_close - _SL_ATR_MULT * curr_atr, 2)
                sl_dist = curr_close - sl
                reason  = (f"LONG: EMA9({curr_e9:.2f}) golden cross above "
                           f"EMA21({curr_e21:.2f})")
            else:
                # Fresh death cross: EMA9 just crossed below EMA21
                cross = curr_e9 < curr_e21 and prev_e9 >= prev_e21
                if not cross:
                    base["reason"] = (f"EMA9 no death cross below EMA21 "
                                      f"(curr EMA9={curr_e9:.2f} EMA21={curr_e21:.2f})")
                    return base
                sl      = round(curr_close + _SL_ATR_MULT * curr_atr, 2)
                sl_dist = sl - curr_close
                reason  = (f"SHORT: EMA9({curr_e9:.2f}) death cross below "
                           f"EMA21({curr_e21:.2f})")

            if sl_dist <= 0:
                base["reason"] = "zero SL distance (path E)"
                return base

            return {
                "signal":        True,
                "entry":         round(curr_close, 2),
                "sl":            sl,
                "tp1_rr":        TP1_RR,
                "tp2_rr":        TP2_RR,
                "strategy_used": "EMA-Cross-9/21",
                "reason":        reason,
                "entry_type":    "ema_cross_9_21",
            }

        except Exception as exc:
            logger.warning("_path_e_ema_cross error: %s", exc, exc_info=True)
            base["reason"] = f"path_e error: {exc}"
            return base

    @property
    def strategy_names(self) -> list[str]:
        return [
            "RSI50-Cross      (02 UTC — Asia Night)",
            "EMA-Cross-9/21   (05 UTC — Asia Morning)",
            "MACD+ADX         (06 UTC — EU Pre-Open)",
            "RSI+EMA Stack    (10 UTC — EU Mid-Session, ADX≥25)",
            "Keltner-Breakout (14 UTC — NY Pre-Open, primary)",
            "EMA-Cross-9/21   (14 UTC — NY Pre-Open, Keltner fallback)",
            "Keltner-Breakout (15 UTC — NY Open, primary)",
            "MACD+ADX         (15 UTC — NY Open, Keltner fallback)",
        ]
