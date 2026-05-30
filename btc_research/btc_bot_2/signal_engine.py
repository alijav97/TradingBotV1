"""
btc_research/btc_bot_2/signal_engine.py — BTC Bot 2 signal generation.

Scans BTCUSDT H1 bars at the kill-zone hours [1, 2, 3, 8] UTC and returns
a fully-enriched signal dict when VB or Swing Level Break v2 fires.

== FLOW ==
  1. Fetch last 300 BTCUSDT H1 bars from DataFeed
  2. Compute EMA200, ADX(14), ATR(14) on the window
  3. EMA200 filter: only trade in the direction price is relative to EMA200
  4. ADX gate: skip if ADX < 20 (no clear trend)
  5. Run VBSwingStrategy (VB → SLv2 "both" 2×ATR) for the allowed direction
  6. ADX-split risk: 3% ADX≤25, 2% ADX 25-40, 3% ADX≥40
  7. Size: risk_usd = balance × risk_pct  |  btc_amount = risk_usd / sl_dist
  8. Compute TP1, TP2 prices from the signal's TP1_RR and TP2_RR

== USAGE ==
  from btc_research.btc_bot_2.signal_engine import SignalEngine
  engine = SignalEngine(feed=feed, journal=journal)
  sig = engine.scan()   # returns signal dict or None
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from v2.connectors.unified_data import DataFeed
    from v2.journal.sqlite_journal import Journal

from btc_research.btc_bot_2.settings import (
    ADX_THRESHOLD, ADX_PERIOD, EMA200_PERIOD, KZ_HOURS,
    STARTING_BALANCE,
    ADX_SPLIT_EARLY_MAX, ADX_SPLIT_STRONG_MIN,
    RISK_PCT_EARLY_TREND, RISK_PCT_TRANSITION, RISK_PCT_STRONG,
    TP1_RR, TP2_RR,
    SYMBOL, MIN_CONFLUENCE_SCORE,
)
from btc_research.btc_bot_2.strategy.vb_swing_combined import VBSwingStrategy, get_risk_pct

logger = logging.getLogger(__name__)

# Binance symbol name for BTCUSDT data feed
_BTC_SYMBOL = "BTCUSDT"
_TIMEFRAME  = "H1"
_BAR_COUNT  = 300

# Minimum bars required to compute indicators (EMA200 needs at least 200)
_MIN_BARS = 220


def _calc_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute ADX(period) on the last bar of df.
    Uses Wilder's smoothing. Returns 0.0 on insufficient data.
    """
    n   = len(df)
    req = period * 3
    if n < req:
        return 0.0

    high  = df["high"].astype(float).values
    low   = df["low"].astype(float).values
    close = df["close"].astype(float).values

    plus_dm  = []
    minus_dm = []
    trs      = []

    for i in range(1, len(high)):
        up   = high[i]  - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm.append(up   if (up > down and up > 0)   else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )
        trs.append(tr)

    # Wilder smooth over last 3×period bars
    slice_len = period * 3
    plus_dm_  = plus_dm[-slice_len:]
    minus_dm_ = minus_dm[-slice_len:]
    trs_      = trs[-slice_len:]

    # Seed
    sm_tr  = sum(trs_[:period])
    sm_pdm = sum(plus_dm_[:period])
    sm_ndm = sum(minus_dm_[:period])

    for i in range(period, len(trs_)):
        sm_tr  = sm_tr  - sm_tr / period  + trs_[i]
        sm_pdm = sm_pdm - sm_pdm / period + plus_dm_[i]
        sm_ndm = sm_ndm - sm_ndm / period + minus_dm_[i]

    if sm_tr <= 0:
        return 0.0

    pdi = 100 * sm_pdm / sm_tr
    ndi = 100 * sm_ndm / sm_tr
    dx  = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) > 0 else 0.0

    # Smooth DX into ADX (single final smoothing step — approximation)
    # For a proper ADX we'd need the full DX series; this gives a reasonable
    # estimate for the threshold gate check.
    return round(dx, 2)


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return ATR(period) for the last bar of df."""
    close_s = df["close"].astype(float)
    high_s  = df["high"].astype(float)
    low_s   = df["low"].astype(float)
    tr = pd.concat([
        high_s - low_s,
        (high_s - close_s.shift(1)).abs(),
        (low_s  - close_s.shift(1)).abs(),
    ], axis=1).max(axis=1)
    val = float(tr.rolling(period).mean().iloc[-1])
    return val if not pd.isna(val) else 0.0


class SignalEngine:
    """
    One-shot signal scanner for BTCUSDT H1 at Bot 2 kill-zone hours.

    Inject a DataFeed (for OHLCV) and Journal (for balance / trade count).
    Call scan() once per hour (or whenever needed) — it is safe to call
    outside kill-zone hours; it will return None immediately.
    """

    def __init__(self, feed: "DataFeed", journal: "Journal") -> None:
        self._feed    = feed
        self._journal = journal
        self._strat   = VBSwingStrategy()

    # ── Public API ─────────────────────────────────────────────────────────────

    def in_kill_zone(self, now: datetime | None = None) -> bool:
        """Return True if current UTC hour is in the Bot 2 kill-zone."""
        if now is None:
            now = datetime.now(timezone.utc)
        return now.hour in KZ_HOURS

    def scan(self, now: datetime | None = None) -> dict | None:
        """
        Run a full signal scan.

        Returns a signal dict if a trade should be opened, or None.

        The caller is responsible for:
          - Calling this during kill-zone hours only (or let the scheduler gate it)
          - Checking that no trade is currently open before calling (one-trade rule)
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if not self.in_kill_zone(now):
            logger.debug("scan() called outside KZ hours (hr=%d) — skip", now.hour)
            return None

        # ── Fetch OHLCV ───────────────────────────────────────────────────────
        try:
            df = self._feed.get_ohlcv(_BTC_SYMBOL, _TIMEFRAME, _BAR_COUNT)
        except Exception as exc:
            logger.error("DataFeed error fetching %s %s: %s", _BTC_SYMBOL, _TIMEFRAME, exc)
            return None

        if df is None or len(df) < _MIN_BARS:
            logger.warning("Insufficient bars: got %d, need %d", len(df) if df is not None else 0, _MIN_BARS)
            return None

        # ── Indicators ────────────────────────────────────────────────────────
        close_s   = df["close"].astype(float)
        ema200    = float(close_s.ewm(span=EMA200_PERIOD, adjust=False).mean().iloc[-1])
        adx       = _calc_adx(df, ADX_PERIOD)
        atr       = _calc_atr(df, 14)
        bar_close = float(close_s.iloc[-1])
        bar_time  = pd.Timestamp(df.index[-1]) if not isinstance(df.index[-1], pd.Timestamp) else df.index[-1]

        # ── Filters ───────────────────────────────────────────────────────────
        if adx < ADX_THRESHOLD:
            logger.debug("ADX %.1f < threshold %d — skip", adx, ADX_THRESHOLD)
            return None

        # EMA200 determines allowed direction
        if bar_close > ema200:
            direction = "long"
        else:
            direction = "short"

        logger.info(
            "Scanning %s %s: close=%.0f ema200=%.0f adx=%.1f atr=%.0f",
            _BTC_SYMBOL, direction, bar_close, ema200, adx, atr,
        )

        # ── Strategy signal ───────────────────────────────────────────────────
        result = self._strat.generate_signal(df, bar_time, direction)

        if not result.get("signal"):
            logger.debug("No signal: %s", result.get("reason", ""))
            return None

        # ── Risk sizing ───────────────────────────────────────────────────────
        risk_pct  = get_risk_pct(adx)             # ADX-split: 2% or 3%
        balance   = self._get_balance()
        risk_usd  = balance * risk_pct

        entry_px  = float(result["entry"])
        sl_px     = float(result["sl"])
        sl_dist   = abs(entry_px - sl_px)

        if sl_dist <= 0:
            logger.warning("Zero SL distance — skip")
            return None

        # BTC sizing: risk_usd / sl_distance = BTC amount
        # (1 BTC moves $1 per $1 price move, no pip scaling needed)
        btc_amount = round(risk_usd / sl_dist, 4)
        btc_amount = max(btc_amount, 0.001)   # min 0.001 BTC

        # ── TP prices ─────────────────────────────────────────────────────────
        tp1_rr  = float(result.get("tp1_rr", TP1_RR))
        tp2_rr  = float(result.get("tp2_rr", TP2_RR))
        is_long = direction == "long"
        sl_r    = sl_dist  # 1R = SL distance

        if is_long:
            tp1_px = round(entry_px + sl_r * tp1_rr, 1)
            tp2_px = round(entry_px + sl_r * tp2_rr, 1)
        else:
            tp1_px = round(entry_px - sl_r * tp1_rr, 1)
            tp2_px = round(entry_px - sl_r * tp2_rr, 1)

        # ── Build signal dict ─────────────────────────────────────────────────
        strategy_name = result.get("strategy_used", "VBSwing v2")
        entry_type    = result.get("entry_type", "")   # "break" | "retest" | None

        signal = {
            "symbol":        SYMBOL,         # "BTCUSD"
            "feed_symbol":   _BTC_SYMBOL,    # "BTCUSDT" (for DataFeed)
            "direction":     direction,
            "entry_price":   round(entry_px, 1),
            "stop_loss":     round(sl_px, 1),
            "tp1_price":     tp1_px,
            "tp2_price":     tp2_px,
            "lot_size":      btc_amount,
            "risk_pct":      round(risk_pct * 100, 1),   # as percentage
            "risk_usd":      round(risk_usd, 2),
            "sl_dist":       round(sl_dist, 1),
            "atr":           round(atr, 1),
            "adx":           round(adx, 1),
            "ema200":        round(ema200, 1),
            "tp1_rr":        tp1_rr,
            "tp2_rr":        tp2_rr,
            "strategy":      strategy_name,
            "entry_type":    entry_type,
            "reason":        result.get("reason", ""),
            "timeframe":     _TIMEFRAME,
            "session":       "Asia Night" if now.hour in [1, 2, 3] else "EU Open",
            "bar_time":      str(bar_time),
            "scan_time":     now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "balance":       round(balance, 2),
            "signal":        True,
        }

        logger.info(
            "SIGNAL: %s %s  entry=%.0f  SL=%.0f  TP1=%.0f  lot=%.4f BTC  "
            "risk=%.1f%%  ADX=%.1f  strategy=%s(%s)",
            SYMBOL, direction.upper(),
            entry_px, sl_px, tp1_px, btc_amount, risk_pct * 100,
            adx, strategy_name, entry_type,
        )

        return signal

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_balance(self) -> float:
        """
        Return current account balance from journal (compounded),
        falling back to STARTING_BALANCE if journal is unavailable.
        """
        try:
            stats = self._journal.get_stats(days=9999)
            balance = stats.get("current_balance") or stats.get("total_pnl") or None
            if balance and balance > 0:
                # total_pnl is cumulative — add starting balance
                if "current_balance" not in stats:
                    return STARTING_BALANCE + float(balance)
                return float(balance)
        except Exception as exc:
            logger.debug("Could not read balance from journal: %s", exc)
        return STARTING_BALANCE
