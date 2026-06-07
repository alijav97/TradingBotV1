"""
btc_research/eth_bot/signal_engine.py — ETH Bot signal generation.

Scans ETHUSD H1 bars at kill-zone hours and returns a fully-enriched
signal dict when a strategy fires.

== FLOW ==
  1. Check kill-zone gate (VBSwing hours [1,2,3,8]) — skip outside KZ hours
  2. Fetch last 300 ETHUSD H1 bars from DataFeed (MT5)
  3. Compute EMA200, ADX(14), ATR(14) on the window
  4. EMA200 filter: only trade in the direction price is relative to EMA200
  5. ADX gate: skip if ADX < 20 (no clear trend)
  6. Run VBSwingStrategy (SwingLevelV2 > VolatilityBreakout) for the allowed direction
  7. Flat 8% base risk; THROTTLE-2: halve risk after 2 consecutive losses until a win
  8. Size: risk_usd = balance × risk_pct  |  eth_amount = risk_usd / sl_dist
  9. Compute TP1 (2R), TP2 (5R) prices from the signal's TP1_RR / TP2_RR

== USAGE ==
  from btc_research.eth_bot.signal_engine import ETHSignalEngine
  engine = ETHSignalEngine(feed=feed, journal=journal)
  sig = engine.scan()   # returns signal dict or None
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from btc_research.eth_bot.connectors.unified_data import DataFeed
    from btc_research.eth_bot.journal.sqlite_journal  import Journal

from btc_research.eth_bot.settings import (
    ADX_THRESHOLD, ADX_PERIOD, EMA200_PERIOD, KZ_HOURS,
    STARTING_BALANCE,
    ADX_SPLIT_EARLY_MAX, ADX_SPLIT_STRONG_MIN,
    RISK_PCT_EARLY_TREND, RISK_PCT_TRANSITION, RISK_PCT_STRONG,
    TP1_RR, TP2_RR,
    OCT_RISK_FACTOR, MONTHLY_CB_ENABLED, CB_MONTHLY_DD_LIMIT,
    THROTTLE_DD_TRIGGER, THROTTLE_FACTOR,
    THROTTLE2_LOSSES, THROTTLE2_FACTOR,
    SYMBOL,
)
# VBSwing is the live strategy (ported BTC SwingLevelV2 > VolatilityBreakout).
# get_risk_pct still comes from eth_combined; all ADX tiers are now flat 8%, so it
# returns 0.08 regardless of ADX.
from btc_research.btc_bot_2.strategy.vb_swing_combined import VBSwingStrategy
from btc_research.eth_bot.strategy.eth_combined import get_risk_pct

logger = logging.getLogger(__name__)

_ETH_SYMBOL = "ETHUSD"
_BTC_SYMBOL = "BTCUSD"
_TIMEFRAME  = "H1"
_BAR_COUNT  = 300
_MIN_BARS   = 220   # EMA200 needs at least 200 bars

# BTC same-bar alignment gate — DISABLED for the VBSwing config (empty set = never
# gates). The validated VBSwing backtest (eth_vbswing.py best config, BTC hours
# [1,2,3,8]) does NOT use a live BTC-trend alignment filter; it trades the ETH signal
# on its own EMA200/ADX gates. Leaving this empty keeps every VBSwing entry.
_BTC_ALIGN_HOURS = frozenset()


def _calc_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ADX(period) on the last bar of df. Returns 0.0 on insufficient data."""
    try:
        close_s = df["close"].astype(float)
        high_s  = df["high"].astype(float)
        low_s   = df["low"].astype(float)
        sp      = 2 * period - 1
        hd      = high_s.diff()
        ld      = low_s.diff()
        tr      = pd.concat([
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s  - close_s.shift(1)).abs(),
        ], axis=1).max(axis=1)
        pdm     = hd.where((hd > 0) & (hd > -ld), 0.0)
        mdm     = (-ld).where((-ld > 0) & (-ld > hd), 0.0)
        aw      = tr.ewm(span=sp,  adjust=False).mean()
        pw      = pdm.ewm(span=sp, adjust=False).mean()
        mw      = mdm.ewm(span=sp, adjust=False).mean()
        pdi     = 100 * pw / aw
        ndi     = 100 * mw / aw
        dx      = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, float("nan"))
        adx     = dx.ewm(span=sp, adjust=False).mean().fillna(0)
        return round(float(adx.iloc[-1]), 2)
    except Exception:
        return 0.0


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return ATR(period) for the last bar of df."""
    try:
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
    except Exception:
        return 0.0


class ETHSignalEngine:
    """
    One-shot signal scanner for ETHUSD H1 at ETH Bot kill-zone hours.

    Inject a DataFeed (for OHLCV) and Journal (for balance / trade count).
    Call scan() once per scheduled tick — safe to call outside KZ hours,
    returns None immediately.
    """

    def __init__(self, feed: "DataFeed", journal: "Journal") -> None:
        self._feed    = feed
        self._journal = journal
        self._strat   = VBSwingStrategy()

    # ── Public API ─────────────────────────────────────────────────────────────

    def in_kill_zone(self, now: datetime | None = None) -> bool:
        """Return True if current UTC hour is in the ETH Bot kill-zone."""
        if now is None:
            now = datetime.now(timezone.utc)
        return now.hour in KZ_HOURS

    def scan(self, now: datetime | None = None) -> dict | None:
        """
        Run a full signal scan.

        Returns a signal dict if a trade should be opened, or None.

        The caller is responsible for:
          - Calling this during kill-zone hours only (or letting scheduler gate it)
          - Checking that no trade is currently open (one-trade rule)
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if not self.in_kill_zone(now):
            logger.debug("scan() called outside KZ hours (hr=%d) — skip", now.hour)
            return None

        # ── Monthly circuit breaker (DISABLED for VBSwing) ─────────────────────
        # A month-level hard halt forfeits the post-streak recovery winners that are
        # VBSwing's edge (see eth_vbswing_cb.py). Gated off via MONTHLY_CB_ENABLED;
        # damage control is handled by THROTTLE-2 in the sizing block instead.
        if MONTHLY_CB_ENABLED and self._month_circuit_broken():
            logger.info("  → SKIP: monthly circuit breaker active (month DD <= %.0f%%)",
                        CB_MONTHLY_DD_LIMIT * 100)
            return None

        # ── Fetch OHLCV ───────────────────────────────────────────────────────
        try:
            df = self._feed.get_ohlcv(_ETH_SYMBOL, _TIMEFRAME, _BAR_COUNT)
        except Exception as exc:
            logger.error("DataFeed error fetching %s %s: %s", _ETH_SYMBOL, _TIMEFRAME, exc)
            return None

        if df is None or len(df) < _MIN_BARS:
            logger.warning("Insufficient bars: got %d, need %d",
                           len(df) if df is not None else 0, _MIN_BARS)
            return None

        # ── Indicators ────────────────────────────────────────────────────────
        close_s   = df["close"].astype(float)
        ema200    = float(close_s.ewm(span=EMA200_PERIOD, adjust=False).mean().iloc[-1])
        adx       = _calc_adx(df, ADX_PERIOD)
        atr       = _calc_atr(df, 14)
        bar_close = float(close_s.iloc[-1])
        bar_time  = (df.index[-1] if isinstance(df.index[-1], pd.Timestamp)
                     else pd.Timestamp(df.index[-1]))

        # ── Filters ───────────────────────────────────────────────────────────
        if adx < ADX_THRESHOLD:
            logger.info(
                "  → SKIP: ADX %.1f < threshold %d (weak trend) | ETH $%.2f | EMA200 $%.2f",
                adx, ADX_THRESHOLD, bar_close, ema200,
            )
            return None

        # EMA200 determines allowed direction
        direction = "long" if bar_close > ema200 else "short"

        logger.info(
            "Scanning %s %s: close=%.2f ema200=%.2f adx=%.1f atr=%.2f",
            _ETH_SYMBOL, direction, bar_close, ema200, adx, atr,
        )

        # ── Strategy signal ───────────────────────────────────────────────────
        result = self._strat.generate_signal(df, bar_time, direction)

        if not result.get("signal"):
            logger.info("  → SKIP: %s", result.get("reason", "no signal"))
            return None

        # ── BTC same-bar alignment filter ───────────────────────────────────────
        # The $70k backtest only counts H02/H06/H10 signals when BTC is trending
        # the same way as the ETH trade (S4_SLOTS btc_req=True). Match it live so
        # the bot trades the exact config the backtest models.
        if now.hour in _BTC_ALIGN_HOURS and not self._btc_aligned(direction):
            logger.info(
                "  → SKIP: BTC not aligned with %s (H%02d requires BTC same-bar trend)",
                direction.upper(), now.hour,
            )
            return None

        # ── Risk sizing ───────────────────────────────────────────────────────
        # Flat 8% base (all ADX tiers equal in settings) — VBSwing FINAL DECISION.
        risk_pct  = get_risk_pct(adx)
        # October cut (disabled for VBSwing: OCT_RISK_FACTOR = 1.0)
        if now.month == 10 and OCT_RISK_FACTOR != 1.0:
            risk_pct *= OCT_RISK_FACTOR
            logger.info("  October risk cut applied: risk_pct -> %.1f%%", risk_pct * 100)
        # THROTTLE-2 — after N consecutive losses, halve risk until the next win.
        # The validated damage-control rule (eth_vbswing_final.py): softens the worst
        # drawdowns while keeping every trade, incl. the post-streak recovery winner.
        if THROTTLE2_FACTOR < 1.0:
            consec = self._consecutive_losses()
            if consec >= THROTTLE2_LOSSES:
                risk_pct *= THROTTLE2_FACTOR
                logger.warning(
                    "  THROTTLE-2 engaged (%d consecutive losses >= %d): risk_pct -> %.1f%%",
                    consec, THROTTLE2_LOSSES, risk_pct * 100,
                )
        balance   = self._get_balance()
        # Equity HWM throttle (disabled for VBSwing: THROTTLE_FACTOR = 1.0)
        if self._equity_throttled(balance):
            risk_pct *= THROTTLE_FACTOR
            logger.warning(
                "  Equity throttle engaged (>= %.0f%% below peak): risk_pct -> %.1f%%",
                abs(THROTTLE_DD_TRIGGER) * 100, risk_pct * 100,
            )
        risk_usd  = balance * risk_pct

        entry_px  = float(result["entry"])
        sl_px     = float(result["sl"])
        sl_dist   = abs(entry_px - sl_px)

        if sl_dist <= 0:
            logger.warning("Zero SL distance — skip")
            return None

        # ETH sizing: risk_usd / sl_distance = ETH amount
        # (1 ETH moves $1 per $1 price move — same mechanics as BTC)
        eth_amount = round(risk_usd / sl_dist, 4)
        eth_amount = max(eth_amount, 0.01)   # min 0.01 ETH

        # ── TP prices ─────────────────────────────────────────────────────────
        tp1_rr  = float(result.get("tp1_rr", TP1_RR))
        tp2_rr  = float(result.get("tp2_rr", TP2_RR))
        is_long = direction == "long"

        if is_long:
            tp1_px = round(entry_px + sl_dist * tp1_rr, 2)
            tp2_px = round(entry_px + sl_dist * tp2_rr, 2)
        else:
            tp1_px = round(entry_px - sl_dist * tp1_rr, 2)
            tp2_px = round(entry_px - sl_dist * tp2_rr, 2)

        # ── Build signal dict ─────────────────────────────────────────────────
        strategy_name = result.get("strategy_used", "ETH Strategy")
        entry_type    = result.get("entry_type", "")

        # Determine session label (VBSwing KZ_HOURS = [1, 2, 3, 8])
        _SESSION_LABELS = {
            1:  "Asia Night",
            2:  "Asia Night",
            3:  "Asia Night",
            8:  "EU Open",
        }
        session = _SESSION_LABELS.get(now.hour, f"UTC {now.hour:02d}:xx")

        signal = {
            "symbol":       SYMBOL,          # "ETHUSD"
            "feed_symbol":  _ETH_SYMBOL,     # "ETHUSD" (for DataFeed)
            "direction":    direction,
            "entry_price":  round(entry_px, 2),
            "stop_loss":    round(sl_px, 2),
            "tp1_price":    tp1_px,
            "tp2_price":    tp2_px,
            "lot_size":     eth_amount,
            "risk_pct":     round(risk_pct * 100, 1),
            "risk_usd":     round(risk_usd, 2),
            "sl_dist":      round(sl_dist, 2),
            "atr":          round(atr, 2),
            "adx":          round(adx, 1),
            "ema200":       round(ema200, 2),
            "tp1_rr":       tp1_rr,
            "tp2_rr":       tp2_rr,
            "strategy":     strategy_name,
            "entry_type":   entry_type,
            "reason":       result.get("reason", ""),
            "timeframe":    _TIMEFRAME,
            "session":      session,
            "bar_time":     str(bar_time),
            "scan_time":    now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "balance":      round(balance, 2),
            "signal":       True,
        }

        logger.info(
            "SIGNAL: %s %s  entry=%.2f  SL=%.2f  TP1=%.2f  lot=%.4f ETH  "
            "risk=%.1f%%  ADX=%.1f  strategy=%s(%s)",
            SYMBOL, direction.upper(),
            entry_px, sl_px, tp1_px, eth_amount, risk_pct * 100,
            adx, strategy_name, entry_type,
        )

        return signal

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _month_circuit_broken(self) -> bool:
        """
        S4 monthly circuit breaker.

        Returns True if the current calendar month's REALISED return has drawn
        down to CB_MONTHLY_DD_LIMIT or worse, in which case no new entries are
        taken for the rest of the month.

        month_dd = month_realised_pnl / month_start_balance
        month_start_balance = current_balance - month_realised_pnl
        """
        try:
            month_pnl = self._journal.get_current_month_pnl()
            if month_pnl >= 0:
                return False  # up or flat on the month — never halt
            cur_bal   = self._get_balance()
            month_start = cur_bal - month_pnl
            if month_start <= 0:
                return False
            month_dd = month_pnl / month_start
            if month_dd <= CB_MONTHLY_DD_LIMIT:
                logger.warning(
                    "Circuit breaker: month DD %.1f%% (pnl=%.2f, start=%.2f) <= limit %.0f%%",
                    month_dd * 100, month_pnl, month_start, CB_MONTHLY_DD_LIMIT * 100,
                )
                return True
            return False
        except Exception as exc:
            logger.debug("circuit breaker check failed (allowing trade): %s", exc)
            return False

    def _consecutive_losses(self) -> int:
        """
        Number of consecutive losing closed trades (most recent backwards), used by
        the THROTTLE-2 brake. Reads the journal; on any error returns 0 (full size).
        """
        try:
            return int(self._journal.get_consecutive_losses())
        except Exception as exc:
            logger.debug("consecutive-loss check failed (full size): %s", exc)
            return 0

    def _equity_throttled(self, balance: float) -> bool:
        """
        Equity throttle (high-water-mark drawdown brake).

        Returns True when the compounded balance is THROTTLE_DD_TRIGGER or more
        below its all-time peak, in which case risk-per-trade is multiplied by
        THROTTLE_FACTOR. Mirrors the peak/throttle logic in realistic_risk_sweep.py.
        Disabled when THROTTLE_FACTOR >= 1.0.
        """
        if THROTTLE_FACTOR >= 1.0:
            return False
        try:
            peak = self._journal.get_peak_balance()
            if peak <= 0:
                return False
            return (balance - peak) / peak <= THROTTLE_DD_TRIGGER
        except Exception as exc:
            logger.debug("equity throttle check failed (full size): %s", exc)
            return False

    def _btc_aligned(self, direction: str) -> bool:
        """
        BTC same-bar trend alignment, matching run_backtest._btc_aligned.

        Fetches the latest BTCUSD H1 window, computes BTC EMA200 (same formula as
        the ETH EMA200 filter and the backtest's _ema), and returns True when BTC
        is on the same side of its EMA200 as the ETH trade direction:
          LONG  ETH  → BTC close > BTC EMA200
          SHORT ETH  → BTC close < BTC EMA200

        On any data error this returns False (fail-safe: skip the trade rather than
        take a signal the backtest would have excluded).
        """
        try:
            btc = self._feed.get_ohlcv(_BTC_SYMBOL, _TIMEFRAME, _BAR_COUNT)
            if btc is None or len(btc) < _MIN_BARS:
                logger.warning(
                    "BTC alignment: insufficient BTC bars (got %d) — skipping signal",
                    len(btc) if btc is not None else 0,
                )
                return False
            close_s   = btc["close"].astype(float)
            btc_ema   = float(close_s.ewm(span=EMA200_PERIOD, adjust=False).mean().iloc[-1])
            btc_close = float(close_s.iloc[-1])
            above     = btc_close > btc_ema
            aligned   = (above and direction == "long") or (not above and direction == "short")
            logger.info(
                "  BTC alignment: close=%.2f ema200=%.2f (%s) vs ETH %s → %s",
                btc_close, btc_ema, "above" if above else "below",
                direction.upper(), "ALIGNED" if aligned else "NOT aligned",
            )
            return aligned
        except Exception as exc:
            logger.warning("BTC alignment check failed (skipping signal): %s", exc)
            return False

    def _get_balance(self) -> float:
        """Return current account balance from journal, falling back to STARTING_BALANCE."""
        try:
            balance = self._journal.get_paper_balance()
            if balance and balance > 0:
                return float(balance)
        except Exception as exc:
            logger.debug("Could not read balance from journal: %s", exc)
        return STARTING_BALANCE

    def get_market_snapshot(self) -> dict:
        """
        Fetch current ETH indicators WITHOUT the kill-zone gate.
        Used by background scan to keep data fresh and log pre-KZ conditions.
        """
        try:
            df = self._feed.get_ohlcv(_ETH_SYMBOL, _TIMEFRAME, _BAR_COUNT)
            if df is None or len(df) < _MIN_BARS:
                return {}

            close_s = df["close"].astype(float)
            price   = float(close_s.iloc[-1])
            ema200  = float(close_s.ewm(span=EMA200_PERIOD, adjust=False).mean().iloc[-1])
            adx     = _calc_adx(df, ADX_PERIOD)
            atr     = _calc_atr(df, 14)

            return {
                "price":     round(price,  2),
                "ema200":    round(ema200, 2),
                "above_ema": price > ema200,
                "adx":       round(adx,   1),
                "atr":       round(atr,   2),
            }
        except Exception as exc:
            logger.warning("get_market_snapshot error: %s", exc)
            return {}
