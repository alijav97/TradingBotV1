"""
btc_bot_1/signals/btc_engine.py — Live signal engine for BTC Bot 1.

Implements Version D strategy logic:
  1. Kill-zone gate       : 21-24 UTC only
  2. EMA200 filter        : longs only above EMA200, shorts only below
  3. ADX filter           : ADX >= 20 required (trend must be present)
  4. Confluence check     : score_bar() from btc_research/strategy/confluence.py
  5. Flipped risk flag    : ADX 20-28 → 3%, ADX > 28 → 2% (set in signal dict)

Trailing SL after TP1 (2×ATR) is handled by paper_trader.py.

Backtest reference: 2yr, US Late 21-24 UTC, 43% WR, +$23,733
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from btc_research.btc_bot_1.connectors.unified_data import DataFeed

from btc_research.strategy.confluence import score_bar
from btc_research.btc_bot_1.settings import (
    SYMBOL, GOLD_SYMBOL, NAS_SYMBOL,
    KZ_START_UTC, KZ_END_UTC,
    TP1_RR, TP2_RR,
    MIN_CONFLUENCE_SCORE,
    EMA200_PERIOD, ADX_PERIOD, ADX_THRESHOLD, ADX_EARLY_TREND_MAX,
    RISK_PCT, RISK_PCT_EARLY_TREND,
)

logger = logging.getLogger(__name__)

_HISTORY_BARS = 500


class BTCSignalEngine:
    """
    Evaluates the current H1 bar for BTC entry signals using Version D logic.
    """

    def __init__(self, feed: "DataFeed") -> None:
        self._feed = feed

    def scan(self, direction: str) -> dict:
        """
        Evaluate current bar for one direction.

        Returns standardised signal dict with extra keys:
          adx_at_entry  : float — ADX value at signal bar
          risk_pct      : float — 0.03 if ADX 20-28, else 0.02
          above_ema200  : bool
        """
        _empty = self._empty_signal(direction)

        # ── Kill-zone gate ────────────────────────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        in_kz = (KZ_START_UTC <= now_utc.hour < 24) or (now_utc.hour == 0 and KZ_END_UTC == 24)
        if not in_kz:
            _empty["blocked_by"] = f"outside kill-zone (UTC {now_utc.hour:02d}:xx)"
            return _empty

        # ── Fetch data ────────────────────────────────────────────────────────
        df_btc  = self._feed.get_ohlcv(SYMBOL,       "H1", _HISTORY_BARS)
        df_gold = self._feed.get_ohlcv(GOLD_SYMBOL,  "H1", _HISTORY_BARS)
        df_nas  = self._feed.get_ohlcv(NAS_SYMBOL,   "H1", _HISTORY_BARS)

        if df_btc.empty or len(df_btc) < EMA200_PERIOD + 20:
            _empty["blocked_by"] = "insufficient BTC H1 data for EMA200"
            return _empty

        # ── Current bar ───────────────────────────────────────────────────────
        last_bar  = df_btc.iloc[-1]
        bar_time  = pd.Timestamp(last_bar["time"])
        bar_close = float(last_bar["close"])
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        close_s = df_btc["close"].astype(float)
        high_s  = df_btc["high"].astype(float)
        low_s   = df_btc["low"].astype(float)

        # ── EMA200 ────────────────────────────────────────────────────────────
        ema200      = float(close_s.ewm(span=EMA200_PERIOD, adjust=False).mean().iloc[-1])
        above_ema200 = bar_close > ema200
        is_long      = direction.lower() in ("long", "buy")

        if is_long and not above_ema200:
            _empty["blocked_by"] = (
                f"EMA200 filter: LONG blocked — price {bar_close:.2f} < EMA200 {ema200:.2f}"
            )
            return _empty
        if not is_long and above_ema200:
            _empty["blocked_by"] = (
                f"EMA200 filter: SHORT blocked — price {bar_close:.2f} > EMA200 {ema200:.2f}"
            )
            return _empty

        # ── ADX ───────────────────────────────────────────────────────────────
        adx = _calc_adx(high_s, low_s, close_s, ADX_PERIOD)
        if adx < ADX_THRESHOLD:
            _empty["blocked_by"] = (
                f"ADX filter: ADX {adx:.1f} < threshold {ADX_THRESHOLD} (no clear trend)"
            )
            return _empty

        # ── Flipped risk sizing ───────────────────────────────────────────────
        # Early trend (ADX 20-28) → size up at 3%
        # Extended trend (ADX > 28) → normal 2%
        risk_pct = RISK_PCT_EARLY_TREND if adx <= ADX_EARLY_TREND_MAX else RISK_PCT

        # ── ATR for trailing SL ───────────────────────────────────────────────
        tr = pd.concat([
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s  - close_s.shift(1)).abs(),
        ], axis=1).max(axis=1)
        current_atr = float(tr.rolling(ADX_PERIOD).mean().iloc[-1])

        # ── Confluence score_bar ──────────────────────────────────────────────
        result = score_bar(
            bar_time  = bar_time,
            bar_close = bar_close,
            direction = direction,
            df_btc    = df_btc,
            df_gold   = df_gold,
            df_nas    = df_nas,
        )

        if not result["signal"]:
            logger.debug(
                "BTC %s blocked: %s (score=%.2f, ADX=%.1f)",
                direction.upper(), result.get("blocked_by", ""), result.get("score", 0), adx,
            )
            return {**_empty, **result}

        # ── Live price ────────────────────────────────────────────────────────
        live = self._feed.get_price(SYMBOL)
        live_price = live.get("price") if live else None

        if not live_price or live_price <= 0:
            _empty["blocked_by"] = "no live price available"
            return _empty

        deviation_pct = abs(live_price - bar_close) / bar_close * 100
        if deviation_pct > 3.0:
            _empty["blocked_by"] = f"live price deviation {deviation_pct:.1f}% > 3%"
            return _empty

        # Recalculate SL/TP relative to live price
        sl_dist  = abs(result["entry"] - result["sl"])
        live_sl  = round(live_price - sl_dist, 2) if is_long else round(live_price + sl_dist, 2)
        live_tp1 = round(live_price + TP1_RR * sl_dist, 2) if is_long else round(live_price - TP1_RR * sl_dist, 2)
        live_tp2 = round(live_price + TP2_RR * sl_dist, 2) if is_long else round(live_price - TP2_RR * sl_dist, 2)

        logger.info(
            "BTC SIGNAL %s @ %.2f  SL=%.2f  TP1=%.2f  score=%.2f  "
            "ADX=%.1f  EMA200=%.2f  risk_pct=%.0f%%",
            direction.upper(), live_price, live_sl, live_tp1,
            result["score"], adx, ema200, risk_pct * 100,
        )

        return {
            "signal":        True,
            "symbol":        SYMBOL,
            "direction":     direction,
            "entry_price":   live_price,
            "stop_loss":     live_sl,
            "tp1_price":     live_tp1,
            "tp2_price":     live_tp2,
            "score":         result["score"],
            "confluence_score": result["score"],
            "strategy":      "BTC_Version_D",
            "timeframe":     "H1",
            "session":       "US Late 21-24 UTC",
            "factors":       result.get("factors", {}),
            "blocked_by":    "",
            # Version D extras — used by paper_trader for risk sizing + trailing SL
            "adx_at_entry":  round(adx, 1),
            "risk_pct":      risk_pct,
            "above_ema200":  above_ema200,
            "ema200":        round(ema200, 2),
            "current_atr":   round(current_atr, 2),
        }

    @staticmethod
    def _empty_signal(direction: str) -> dict:
        return {
            "signal":        False,
            "symbol":        SYMBOL,
            "direction":     direction,
            "entry_price":   0.0,
            "stop_loss":     0.0,
            "tp1_price":     0.0,
            "tp2_price":     0.0,
            "score":         0.0,
            "confluence_score": 0.0,
            "strategy":      "BTC_Version_D",
            "timeframe":     "H1",
            "session":       "US Late 21-24 UTC",
            "factors":       {},
            "blocked_by":    "",
            "adx_at_entry":  0.0,
            "risk_pct":      RISK_PCT,
            "above_ema200":  False,
            "ema200":        0.0,
            "current_atr":   0.0,
        }


def _calc_adx(
    high_s:  pd.Series,
    low_s:   pd.Series,
    close_s: pd.Series,
    period:  int = 14,
) -> float:
    """Compute ADX(period) for the last bar. Returns 0 on failure."""
    try:
        sp   = 2 * period - 1
        hd   = high_s.diff()
        ld   = low_s.diff()
        tr   = pd.concat([
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s  - close_s.shift(1)).abs(),
        ], axis=1).max(axis=1)
        pdm  = hd.where((hd > 0) & (hd > -ld), 0.0)
        mdm  = (-ld).where((-ld > 0) & (-ld > hd), 0.0)
        aw   = tr.ewm(span=sp,  adjust=False).mean()
        pw   = pdm.ewm(span=sp, adjust=False).mean()
        mw   = mdm.ewm(span=sp, adjust=False).mean()
        pdi  = 100 * pw / aw
        mdi  = 100 * mw / aw
        dx   = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, float("nan"))
        adx  = dx.ewm(span=sp, adjust=False).mean().fillna(0)
        return float(adx.iloc[-1])
    except Exception:
        return 0.0
