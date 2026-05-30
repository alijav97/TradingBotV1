"""
btc_bot_1/signals/btc_engine.py — Live signal engine for BTC Bot 1.

Wraps btc_research/strategy/confluence.py (the tested backtest logic).
Takes current live data from MT5, evaluates the last completed H1 bar,
and returns a standardised signal dict.

Tested results (from session_scanner + backtest engine):
  Kill-zone: 21-24 UTC | 223 trades | WR=43% | AvgR=+0.47R | PnL=+$23,733

Usage:
    from btc_research.btc_bot_1.signals.btc_engine import BTCSignalEngine
    engine = BTCSignalEngine(feed)
    signal = engine.scan(direction="long")
    if signal["signal"]:
        # open trade
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from btc_research.btc_bot_1.connectors.unified_data import DataFeed

from btc_research.strategy.confluence import score_bar
from btc_research.btc_bot_1.settings import (
    SYMBOL, GOLD_SYMBOL, NAS_SYMBOL,
    KZ_START_UTC, KZ_END_UTC,
    TP1_RR, TP2_RR,
    MIN_CONFLUENCE_SCORE,
)

logger = logging.getLogger(__name__)

_HISTORY_BARS = 500   # how many H1 bars to fetch per scan


class BTCSignalEngine:
    """
    Evaluates the current H1 bar for BTC entry signals.

    Calls score_bar() from the tested btc_research/strategy/confluence.py
    using live MT5 data. Returns a standardised signal dict.
    """

    def __init__(self, feed: "DataFeed") -> None:
        self._feed = feed

    def scan(self, direction: str) -> dict:
        """
        Evaluate the current bar for one direction ("long" or "short").

        Returns:
            signal       : bool   — True if entry conditions are met
            entry_price  : float  — entry price (live tick or bar close)
            stop_loss    : float  — SL price
            tp1_price    : float  — TP1 price
            tp2_price    : float  — TP2 price
            score        : float  — confluence score
            strategy     : str    — always "BTC_Confluence_V1"
            direction    : str
            symbol       : str
            factors      : dict   — per-factor breakdown
            blocked_by   : str    — rejection reason (empty when signal=True)
        """
        _empty = self._empty_signal(direction)

        # ── Guard: only scan inside kill-zone ─────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        if not (KZ_START_UTC <= now_utc.hour < KZ_END_UTC):
            _empty["blocked_by"] = f"outside kill-zone (UTC {now_utc.hour:02d}:xx)"
            return _empty

        # ── Fetch H1 data for BTC, Gold, Nasdaq ───────────────────────────────
        df_btc  = self._feed.get_ohlcv(SYMBOL,      "H1", _HISTORY_BARS)
        df_gold = self._feed.get_ohlcv(GOLD_SYMBOL,  "H1", _HISTORY_BARS)
        df_nas  = self._feed.get_ohlcv(NAS_SYMBOL,   "H1", _HISTORY_BARS)

        if df_btc.empty or len(df_btc) < 50:
            _empty["blocked_by"] = "insufficient BTC H1 data"
            return _empty

        # ── Use the last completed bar ─────────────────────────────────────────
        last_bar  = df_btc.iloc[-1]
        bar_time  = pd.Timestamp(last_bar["time"])
        bar_close = float(last_bar["close"])

        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        # ── Run confluence engine ──────────────────────────────────────────────
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
                "BTC %s blocked: %s (score=%.2f)",
                direction.upper(), result.get("blocked_by", ""), result.get("score", 0)
            )
            return {**_empty, **result}

        # ── Get live price for accurate entry ──────────────────────────────────
        live = self._feed.get_price(SYMBOL)
        live_price = live.get("price") if live else None

        if not live_price or live_price <= 0:
            logger.warning("BTC: no live price — skipping signal")
            _empty["blocked_by"] = "no live price available"
            return _empty

        # Sanity: live price should be within 3% of bar close
        deviation_pct = abs(live_price - bar_close) / bar_close * 100
        if deviation_pct > 3.0:
            logger.warning(
                "BTC: live price %.2f vs bar close %.2f (%.1f%% apart — stale data?)",
                live_price, bar_close, deviation_pct,
            )
            _empty["blocked_by"] = f"live price deviation {deviation_pct:.1f}% > 3%"
            return _empty

        # Recalculate SL relative to live price (keep same USD distance)
        h1_entry  = result["entry"]
        h1_sl     = result["sl"]
        sl_dist   = abs(h1_entry - h1_sl)
        is_long   = direction.lower() in ("long", "buy")
        live_sl   = round(live_price - sl_dist, 2) if is_long else round(live_price + sl_dist, 2)
        live_tp1  = round(live_price + TP1_RR * sl_dist, 2) if is_long else round(live_price - TP1_RR * sl_dist, 2)
        live_tp2  = round(live_price + TP2_RR * sl_dist, 2) if is_long else round(live_price - TP2_RR * sl_dist, 2)

        logger.info(
            "BTC SIGNAL %s @ %.2f  SL=%.2f  TP1=%.2f  TP2=%.2f  score=%.2f",
            direction.upper(), live_price, live_sl, live_tp1, live_tp2, result["score"],
        )

        return {
            "signal":      True,
            "symbol":      SYMBOL,
            "direction":   direction,
            "entry_price": live_price,
            "stop_loss":   live_sl,
            "tp1_price":   live_tp1,
            "tp2_price":   live_tp2,
            "score":       result["score"],
            "confluence_score": result["score"],
            "strategy":    "BTC_Confluence_V1",
            "timeframe":   "H1",
            "session":     "US Late 21-24 UTC",
            "factors":     result.get("factors", {}),
            "blocked_by":  "",
        }

    @staticmethod
    def _empty_signal(direction: str) -> dict:
        return {
            "signal":      False,
            "symbol":      SYMBOL,
            "direction":   direction,
            "entry_price": 0.0,
            "stop_loss":   0.0,
            "tp1_price":   0.0,
            "tp2_price":   0.0,
            "score":       0.0,
            "confluence_score": 0.0,
            "strategy":    "BTC_Confluence_V1",
            "timeframe":   "H1",
            "session":     "US Late 21-24 UTC",
            "factors":     {},
            "blocked_by":  "",
        }
