"""
btc_research/eth_bot/strategy/eth_combined.py — ETH Bot multi-confluence strategy.

Confirmed by 6-year ETH backtest phase-2 (25 strategies, BTC-aligned filter):

  PATH A — 02 UTC  (Asia Night):    RSI(14) 50-cross
    WR=50.0%  AvgR=+0.786  PF=2.57  N=46

  PATH B — 06 UTC  (EU Pre-Open):   MACD cross + ADX ≥ 25 + MACD line side confirms
    WR=45.0%  AvgR=+0.660  PF=2.65  N=20

  PATH C — 10 UTC  (EU Mid-Session): RSI(14) 50-cross + EMA9/21 stack + EMA200
    WR=48.1%  AvgR=+0.706  PF=2.36  N=27

  All three paths require BTC to be on the same side of its EMA200 (BTC alignment).
  The signal engine enforces BTC alignment before calling generate_signal().

  KZ_HOURS = [2, 6, 10] — set in settings.py.

== RISK SIZING ==
  ADX ≤ 25  → 3%  ($15 on $500)
  ADX 25-40 → 2%  ($10 on $500)
  ADX ≥ 40  → 4%  ($20 on $500)

== TP / SL ==
  TP1 = 2R  (close 50%, SL to breakeven)
  TP2 = 4R  (close remainder)
  Trailing SL = price ± 2×ATR after TP1

== PARAMETERS ==
  _RSI_PERIOD      = 14
  _MACD_FAST/SLOW  = 12/26,  signal = 9
  _ADX_MIN_B       = 25   (MACD+ADX path: min ADX to trade)
  _EMA9 / _EMA21   = 9 / 21 (RSI+EMA stack)
  _ATR_PERIOD      = 14   (SL calculation)
  _SL_ATR_MULT     = 1.5  (SL = entry ± 1.5×ATR when no clear swing level)
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
_ADX_MIN_B    = 25      # Path B: minimum ADX for MACD+ADX confluence
_ATR_PERIOD   = 14
_SL_ATR_MULT  = 1.5    # SL = entry ± _SL_ATR_MULT × ATR

# Session routing (must stay in sync with settings.KZ_HOURS = [2, 6, 10])
_PATH_A_HOURS = frozenset({2})     # RSI 50-Cross
_PATH_B_HOURS = frozenset({6})     # MACD+ADX
_PATH_C_HOURS = frozenset({10})    # RSI+EMA Stack


# ── Public helper ──────────────────────────────────────────────────────────────

def get_risk_pct(adx: float) -> float:
    """
    ADX-split risk sizing — Config D, confirmed best by 6yr ETH ADX sweep.
      ADX ≤ 25  → 2%  (early/weak trend zone — WR 45%, AvgR +0.34 → bet light)
      ADX 25-40 → 3%  (sweet spot — WR 47-60%, AvgR +0.87 to +1.64 → normal)
      ADX ≥ 40  → 5%  (strong trend — WR 60%, high conviction → bet heavy)
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


# ── Strategy class ─────────────────────────────────────────────────────────────

class ETHStrategy(BTCStrategy):
    """
    Three-path ETH strategy confirmed by 6-year backtest (BTC-aligned).

      Path A [02 UTC]: RSI(14) crosses above/below 50
      Path B [06 UTC]: MACD line/signal cross + ADX ≥ 25 + MACD line on correct side of zero
      Path C [10 UTC]: RSI(14) 50-cross + EMA9/21 stack aligned + EMA200 direction

    Signal engine already gates by: kill-zone, EMA200 direction, ADX ≥ 20.
    This class handles entry-pattern logic only.
    """

    name        = "ETH: RSI50-Cross[02] | MACD+ADX[06] | RSI+EMA[10]"
    description = (
        "02 UTC: RSI14 50-cross | "
        "06 UTC: MACD cross + ADX≥25 | "
        "10 UTC: RSI50-cross + EMA9/21 stack"
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
            return self._path_a_rsi50(df_window, is_long, base)
        elif hour in _PATH_B_HOURS:
            return self._path_b_macd_adx(df_window, is_long, base)
        elif hour in _PATH_C_HOURS:
            return self._path_c_rsi_ema(df_window, is_long, base)
        else:
            base["reason"] = (
                f"UTC {hour:02d} not in strategy hours "
                f"(A={sorted(_PATH_A_HOURS)}, B={sorted(_PATH_B_HOURS)}, C={sorted(_PATH_C_HOURS)})"
            )
            return base

    # ── Path A — RSI(14) 50-cross [02 UTC] ────────────────────────────────────

    def _path_a_rsi50(self, df: pd.DataFrame, is_long: bool, base: dict) -> dict:
        """
        Asia Night — RSI(14) 50-cross in EMA200 direction.
        LONG : RSI was < 50, now ≥ 50
        SHORT: RSI was > 50, now ≤ 50
        SL   : 10-bar low/high ± 1.5×ATR
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

    # ── Path B — MACD + ADX [06 UTC] ──────────────────────────────────────────

    def _path_b_macd_adx(self, df: pd.DataFrame, is_long: bool, base: dict) -> dict:
        """
        EU Pre-Open — MACD line/signal cross + ADX ≥ 25 + MACD line on correct zero side.
        LONG : MACD line crosses above signal AND MACD line > 0 AND ADX ≥ 25
        SHORT: MACD line crosses below signal AND MACD line < 0 AND ADX ≥ 25
        SL   : entry ± 1.5×ATR
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

            # Compute ADX inline (signal engine already confirms ADX≥20 globally;
            # MACD+ADX path raises the bar to ADX≥25)
            adx_val = _calc_adx(df)
            if adx_val < _ADX_MIN_B:
                base["reason"] = f"ADX {adx_val:.1f} < {_ADX_MIN_B} (path B threshold)"
                return base

            curr_close = float(df["close"].astype(float).iloc[-1])

            if is_long:
                cross = prev_ml <= prev_ms and curr_ml > curr_ms   # bullish cross
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
                cross = prev_ml >= prev_ms and curr_ml < curr_ms   # bearish cross
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
        EU Mid-Session — RSI(14) 50-cross + EMA9/21 stack aligned + EMA200.
        LONG : RSI crosses above 50 + EMA9 > EMA21 + close > EMA200
        SHORT: RSI crosses below 50 + EMA9 < EMA21 + close < EMA200
        SL   : entry ± 1.5×ATR

        ADX gate: ADX ≥ RSI_EMA_ADX_MIN (default 25).
        ADX 20-25 bucket for rsi_ema is near-worthless: WR=41.7%, AvgR=+0.070.
        Raising the bar to 25 lifts quality to WR≥57%, AvgR≥1.40.
        """
        try:
            # Per-strategy ADX gate (stricter than the global ADX_THRESHOLD=20)
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
                rsi_cross  = prev_rsi < 50.0 <= curr_rsi
                ema_stack  = curr_e9 > curr_e21
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
                rsi_cross  = prev_rsi > 50.0 >= curr_rsi
                ema_stack  = curr_e9 < curr_e21
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
                "strategy_used": "RSI+EMA",
                "reason":        reason,
                "entry_type":    "rsi_ema_stack",
            }

        except Exception as exc:
            logger.warning("_path_c_rsi_ema error: %s", exc, exc_info=True)
            base["reason"] = f"path_c error: {exc}"
            return base

    @property
    def strategy_names(self) -> list[str]:
        return [
            "RSI50-Cross  (02 UTC — Asia Night)",
            "MACD+ADX     (06 UTC — EU Pre-Open)",
            "RSI+EMA Stack (10 UTC — EU Mid-Session)",
        ]
