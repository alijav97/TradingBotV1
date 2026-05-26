"""
backtest/backtester.py — Historical warm-start backtester for TradingBotV2.

Replays historical H1 OHLCV data through the LIVE signal engine (same
confluence scoring, same entry checklist, same risk rules) and simulates
trade outcomes using subsequent bars.  Writes all labeled trades to the
SQLite journal so the ML trainer has a real dataset before the bot goes live.

No look-ahead bias:
  - Signal evaluated at bar N uses only bars 0..N
  - Outcome determined by scanning bars N+1..N+MAX_HOLD_BARS forward
  - High/low of each bar used conservatively (SL on low for long, TP on high)

Usage:
    python -m v2.backtest.run_backtest
    python -m v2.backtest.run_backtest --days 180 --instruments XAUUSD BTCUSDT
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from v2.instrument_config import ALL_SYMBOLS, get_instrument
from v2.signals.confluence_engine import ConfluenceEngine
from v2.signals.entry_checklist import validate_entry
from v2.risk.position_sizer import calculate_lot_size, calculate_tp_prices, calculate_risk_usd
from v2.ml.feature_engineer import FeatureEngineer

if TYPE_CHECKING:
    from v2.journal.sqlite_journal import Journal
    from v2.connectors.unified_data import DataFeed

logger = logging.getLogger(__name__)

# How many H1 bars forward to scan for SL/TP outcome
MAX_HOLD_BARS = 48          # 48 hours (matches live MAX_HOLD_HOURS)
# Minimum lookback bars needed to compute indicators reliably
MIN_LOOKBACK  = 120
# Step size: evaluate a signal every N bars (avoids overlapping signals)
SCAN_STEP     = 4           # evaluate every 4 hours


class Backtester:
    """
    Walk-forward historical backtester.

    Parameters
    ----------
    journal : Journal
        SQLite journal — trades and ML features written here.
    feed : DataFeed
        Connected data feed — used to load historical OHLCV bars.
    days : int
        How many calendar days of history to replay (default 180 = ~6 months).
    instruments : list[str] | None
        Subset of instruments to backtest.  None = all 6.
    """

    def __init__(
        self,
        journal:     "Journal",
        feed:        "DataFeed",
        days:        int = 180,
        instruments: list[str] | None = None,
    ) -> None:
        self._journal     = journal
        self._feed        = feed
        self._days        = days
        self._instruments = instruments or list(ALL_SYMBOLS)
        self._engine      = ConfluenceEngine()
        self._fe          = FeatureEngineer(journal=journal)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run the full backtest across all configured instruments.

        Returns a summary dict:
        {
            "instruments_processed": int,
            "signals_evaluated":     int,
            "trades_simulated":      int,
            "wins": int, "losses": int,
            "win_rate": float,
            "by_instrument": {symbol: {...}}
        }
        """
        logger.info(
            "Backtest starting: %d instruments, %d days of history",
            len(self._instruments), self._days
        )

        total_signals   = 0
        total_trades    = 0
        total_wins      = 0
        total_losses    = 0
        by_instrument:  dict = {}

        for symbol in self._instruments:
            logger.info("Backtesting %s ...", symbol)
            try:
                result = self._backtest_instrument(symbol)
                by_instrument[symbol] = result
                total_signals  += result["signals_evaluated"]
                total_trades   += result["trades_simulated"]
                total_wins     += result["wins"]
                total_losses   += result["losses"]
                logger.info(
                    "  %s done: %d trades | WR=%.1f%%",
                    symbol,
                    result["trades_simulated"],
                    result["win_rate"] * 100,
                )
            except Exception as exc:
                logger.error("Backtest failed for %s: %s", symbol, exc, exc_info=True)
                by_instrument[symbol] = {"error": str(exc)}

        win_rate = (total_wins / total_trades) if total_trades > 0 else 0.0

        summary = {
            "instruments_processed": len(self._instruments),
            "signals_evaluated":     total_signals,
            "trades_simulated":      total_trades,
            "wins":                  total_wins,
            "losses":                total_losses,
            "win_rate":              round(win_rate, 4),
            "by_instrument":         by_instrument,
        }

        logger.info(
            "Backtest complete: %d trades | WR=%.1f%% | wins=%d losses=%d",
            total_trades, win_rate * 100, total_wins, total_losses,
        )
        return summary

    # ── Per-instrument logic ──────────────────────────────────────────────────

    def _backtest_instrument(self, symbol: str) -> dict:
        """Run walk-forward backtest for one instrument."""
        # Load maximum available history
        # We request more bars than needed to ensure MIN_LOOKBACK is always available
        bars_needed = int(self._days * 24) + MIN_LOOKBACK + MAX_HOLD_BARS + 100
        bars_needed = min(bars_needed, 5000)   # most connectors cap at 5000

        df_full = self._feed.get_ohlcv(symbol, "H1", bars_needed)
        if df_full is None or df_full.empty:
            logger.warning("%s: no historical data available — skipping", symbol)
            return {"signals_evaluated": 0, "trades_simulated": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

        # Trim to requested days
        if "time" in df_full.columns:
            df_full = df_full.sort_values("time").reset_index(drop=True)
        elif df_full.index.dtype != "object":
            df_full = df_full.sort_index()

        logger.debug("%s: loaded %d H1 bars", symbol, len(df_full))

        # Also load H4 and D1 for HTF alignment checks
        df_h4 = self._feed.get_ohlcv(symbol, "H4", min(bars_needed // 4, 1500))
        df_d1 = self._feed.get_ohlcv(symbol, "D1", min(bars_needed // 24, 500))

        signals_evaluated = 0
        trades_simulated  = 0
        wins = 0
        losses = 0

        # Walk forward: start after MIN_LOOKBACK bars, step every SCAN_STEP bars
        # Leave MAX_HOLD_BARS at the end so every trade has room to resolve
        end_bar = len(df_full) - MAX_HOLD_BARS - 1

        for bar_idx in range(MIN_LOOKBACK, end_bar, SCAN_STEP):
            # Slice history up to current bar (no look-ahead)
            window     = df_full.iloc[:bar_idx + 1].copy()
            window_h4  = self._slice_htf(df_h4,  window, "H4")
            window_d1  = self._slice_htf(df_d1,  window, "D1")

            for direction in ("long", "short"):
                signals_evaluated += 1

                result = self._engine.score(symbol, direction, window, window_h4, window_d1)

                if not result.get("signal"):
                    continue

                # Build signal dict matching the live format
                entry = float(result.get("entry_price") or window["close"].iloc[-1])
                sl    = float(result.get("stop_loss", 0))
                if sl <= 0 or entry <= 0:
                    continue

                tp1, tp2 = calculate_tp_prices(entry, sl, direction)
                signal = {
                    "symbol":           symbol,
                    "direction":        direction,
                    "entry_price":      entry,
                    "stop_loss":        sl,
                    "tp1_price":        tp1,
                    "tp2_price":        tp2,
                    "score":            result.get("score"),
                    "confluence_score": result.get("score"),
                    "strategy":         result.get("strategy", ""),
                    "timeframe":        "H1",
                    "factors":          result.get("factors", {}),
                    "regime":           result.get("regime", ""),
                }

                # Entry checklist (same gates as live)
                checklist = validate_entry(signal, window)
                if not checklist.get("passed"):
                    continue

                # Simulate trade outcome using future bars
                future_bars = df_full.iloc[bar_idx + 1: bar_idx + 1 + MAX_HOLD_BARS]
                outcome = self._simulate_outcome(entry, sl, tp1, tp2, direction, future_bars)

                if outcome is None:
                    continue  # no clear outcome in window

                # Write to journal as a historical trade
                trade_id = self._write_backtest_trade(signal, outcome, window)
                trades_simulated += 1

                if outcome["exit_reason"] in ("TP1", "TP2", "SL_AFTER_TP1"):
                    wins += 1
                else:
                    losses += 1

                # Break direction loop — don't take both long AND short at same bar
                break

        win_rate = (wins / trades_simulated) if trades_simulated > 0 else 0.0
        return {
            "signals_evaluated": signals_evaluated,
            "trades_simulated":  trades_simulated,
            "wins":              wins,
            "losses":            losses,
            "win_rate":          round(win_rate, 4),
        }

    # ── Outcome simulation ────────────────────────────────────────────────────

    def _simulate_outcome(
        self,
        entry:     float,
        sl:        float,
        tp1:       float,
        tp2:       float,
        direction: str,
        future:    pd.DataFrame,
    ) -> dict | None:
        """
        Walk future bars and determine what happened to the trade.

        Returns dict with: exit_price, exit_reason, pnl_pips, bars_held
        or None if outcome was ambiguous (e.g., no data).
        """
        if future.empty:
            return None

        is_long   = direction.lower() in ("long", "buy")
        tp1_hit   = False

        for bar_num, (_, bar) in enumerate(future.iterrows()):
            bar_high = float(bar["high"])
            bar_low  = float(bar["low"])

            if is_long:
                # Check SL first (conservative — assume worst fill within bar)
                if bar_low <= sl:
                    if tp1_hit:
                        # SL hit after TP1 — exit at BE (entry) or SL whichever is better
                        exit_p = max(sl, entry)
                        return self._outcome(exit_p, "SL_AFTER_TP1", entry, bar_num + 1)
                    return self._outcome(sl, "SL", entry, bar_num + 1)

                if not tp1_hit and bar_high >= tp1:
                    tp1_hit = True
                    # After TP1, SL moves to BE — update sl to entry
                    sl = entry

                if tp1_hit and bar_high >= tp2:
                    return self._outcome(tp2, "TP2", entry, bar_num + 1)

            else:  # short
                if bar_high >= sl:
                    if tp1_hit:
                        exit_p = min(sl, entry)
                        return self._outcome(exit_p, "SL_AFTER_TP1", entry, bar_num + 1)
                    return self._outcome(sl, "SL", entry, bar_num + 1)

                if not tp1_hit and bar_low <= tp1:
                    tp1_hit = True
                    sl = entry

                if tp1_hit and bar_low <= tp2:
                    return self._outcome(tp2, "TP2", entry, bar_num + 1)

        # Time exit — MAX_HOLD reached
        last_price = float(future["close"].iloc[-1])
        return self._outcome(last_price, "MAX_HOLD", entry, len(future))

    @staticmethod
    def _outcome(exit_price: float, reason: str, entry: float, bars: int) -> dict:
        return {
            "exit_price":  exit_price,
            "exit_reason": reason,
            "bars_held":   bars,
        }

    # ── Journal write ─────────────────────────────────────────────────────────

    def _write_backtest_trade(
        self,
        signal:   dict,
        outcome:  dict,
        window:   pd.DataFrame,
    ) -> str:
        """
        Write a completed backtest trade to the SQLite journal.
        Opens it as OPEN, then immediately closes it with outcome data.
        Saves ML features at entry time.
        """
        from v2.risk.position_sizer import calculate_lot_size, calculate_risk_usd
        from v2.instrument_config import price_to_pips

        symbol    = signal["symbol"]
        entry     = float(signal["entry_price"])
        sl        = float(signal["stop_loss"])
        direction = signal["direction"]

        lot_size = calculate_lot_size(symbol, entry, sl)
        risk_usd = calculate_risk_usd(symbol, entry, sl, lot_size)

        # Compute PnL
        exit_price  = float(outcome["exit_price"])
        is_long     = direction.lower() in ("long", "buy")
        price_diff  = (exit_price - entry) if is_long else (entry - exit_price)
        try:
            cfg      = get_instrument(symbol)
            pips     = price_diff / cfg.pip_size
            pnl_usd  = pips * cfg.pip_value_usd * lot_size
        except Exception:
            pips    = price_diff * 100
            pnl_usd = price_diff * lot_size * 1000

        pnl_usd = round(pnl_usd, 2)
        pips    = round(pips, 1)

        # RR achieved
        sl_dist = abs(entry - sl)
        rr_achieved = round(abs(price_diff) / sl_dist, 2) if sl_dist > 0 else 0.0
        if pnl_usd < 0:
            rr_achieved = -rr_achieved

        trade = {
            "symbol":           symbol,
            "direction":        direction,
            "entry_price":      entry,
            "stop_loss":        sl,
            "tp1_price":        signal.get("tp1_price"),
            "tp2_price":        signal.get("tp2_price"),
            "lot_size":         lot_size,
            "strategy":         signal.get("strategy", ""),
            "confluence_score": signal.get("confluence_score"),
            "timeframe":        signal.get("timeframe", "H1"),
            "session":          "",
            "regime":           signal.get("regime", ""),
            "news_score":       0.0,
            "factors":          signal.get("factors", {}),
            "raw_signal":       signal,
            "notes":            "backtest",
        }

        trade_id = self._journal.open_trade(trade)

        hold_minutes = float(outcome["bars_held"]) * 60.0

        # Save ML features BEFORE close_trade so close_trade can set the label
        trade_row = self._journal.get_trade(trade_id) or {}
        trade_row["factors_json"] = trade.get("factors", {})
        trade_row["hold_time_minutes"] = hold_minutes
        try:
            features = self._fe.extract(trade_row, df=window)
            self._journal.save_ml_features(trade_id, features)
        except Exception as exc:
            logger.debug("Feature extraction failed for backtest trade %s: %s", trade_id[:8], exc)

        # Now close — close_trade will UPDATE the label on the ml_features row
        exit_context = {
            "hold_time_minutes": hold_minutes,
            "exit_atr":          0.0,
            "exit_regime":       "",
        }

        self._journal.close_trade(
            trade_id,
            exit_price  = exit_price,
            exit_reason = outcome["exit_reason"],
            pnl_usd     = pnl_usd,
            pips        = pips,
            rr_achieved = rr_achieved,
            notes       = "backtest",
            exit_context= exit_context,
        )

        return trade_id

    # ── HTF slicing helper ────────────────────────────────────────────────────

    @staticmethod
    def _slice_htf(
        df_htf:    pd.DataFrame | None,
        window_h1: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame | None:
        """Return HTF bars that end before the last H1 bar (no look-ahead)."""
        if df_htf is None or df_htf.empty:
            return None
        try:
            if "time" in window_h1.columns and "time" in df_htf.columns:
                cutoff = window_h1["time"].iloc[-1]
                return df_htf[df_htf["time"] <= cutoff].copy()
        except Exception:
            pass
        return df_htf.copy()
