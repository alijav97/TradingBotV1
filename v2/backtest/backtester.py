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
MAX_HOLD_BARS = 96          # 96 hours / 4 days (matches live MAX_HOLD_HOURS)
# Minimum lookback bars needed to compute indicators reliably
MIN_LOOKBACK  = 120
# Step size: evaluate a signal every N bars (avoids overlapping signals)
SCAN_STEP     = 1           # evaluate every bar (kill-zone window is only 4h wide)


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
        start_date:  "datetime | None" = None,
        end_date:    "datetime | None" = None,
    ) -> None:
        self._journal     = journal
        self._feed        = feed
        self._instruments = instruments or list(ALL_SYMBOLS)
        self._engine      = ConfluenceEngine()
        self._fe          = FeatureEngineer(journal=journal)
        self._start_date  = start_date
        self._end_date    = end_date

        # Auto-expand days when a date range is provided so enough bars are fetched
        if start_date is not None:
            ref = end_date if end_date is not None else datetime.now(timezone.utc)
            self._days = max(days, (ref - start_date).days + 30)
        else:
            self._days = days

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run the full backtest across all configured instruments.

        Returns a summary dict including compounded P&L stats.
        """
        from v2.settings import ACCOUNT_BALANCE
        logger.info(
            "Backtest starting: %d instruments, %d days of history | Starting balance: $%.2f",
            len(self._instruments), self._days, ACCOUNT_BALANCE,
        )

        total_signals    = 0
        total_trades     = 0
        total_wins       = 0
        total_losses     = 0
        total_breakevens = 0
        by_instrument:   dict = {}

        # Each instrument gets its own fresh $500 — independent compounding.
        # The overall summary shows combined P&L across all instruments.
        total_pnl_usd    = 0.0
        max_drawdown_pct = 0.0

        for symbol in self._instruments:
            logger.info("Backtesting %s ... (starting balance: $%.2f)", symbol, ACCOUNT_BALANCE)
            try:
                # Always start each instrument fresh from ACCOUNT_BALANCE
                result = self._backtest_instrument(symbol, starting_balance=ACCOUNT_BALANCE)
                by_instrument[symbol] = result
                total_signals    += result["signals_evaluated"]
                total_trades     += result["trades_simulated"]
                total_wins       += result["wins"]
                total_losses     += result["losses"]
                total_breakevens += result.get("breakevens", 0)

                # Track per-instrument drawdown
                inst_peak = ACCOUNT_BALANCE
                inst_bal  = ACCOUNT_BALANCE
                for pnl in result.get("pnl_series", []):
                    inst_bal = max(inst_bal + pnl, 0.01)
                    if inst_bal > inst_peak:
                        inst_peak = inst_bal
                    dd = (inst_peak - inst_bal) / inst_peak * 100
                    if dd > max_drawdown_pct:
                        max_drawdown_pct = dd

                inst_ending = result.get("ending_balance", ACCOUNT_BALANCE)
                inst_pnl    = round(inst_ending - ACCOUNT_BALANCE, 2)
                total_pnl_usd += inst_pnl

                logger.info(
                    "  %s done: %d trades | WR=%.1f%% | BE=%d | $%.2f -> $%.2f (%+.1f%%)",
                    symbol,
                    result["trades_simulated"],
                    result["win_rate"] * 100,
                    result.get("breakevens", 0),
                    ACCOUNT_BALANCE,
                    inst_ending,
                    (inst_pnl / ACCOUNT_BALANCE * 100),
                )
            except Exception as exc:
                logger.error("Backtest failed for %s: %s", symbol, exc, exc_info=True)
                by_instrument[symbol] = {"error": str(exc)}

        decisive       = total_wins + total_losses
        win_rate       = (total_wins / decisive) if decisive > 0 else 0.0
        total_invested = ACCOUNT_BALANCE * len([s for s in by_instrument if "error" not in by_instrument[s]])
        return_pct     = round(total_pnl_usd / total_invested * 100, 1) if total_invested > 0 else 0.0

        summary = {
            "instruments_processed": len(self._instruments),
            "signals_evaluated":     total_signals,
            "trades_simulated":      total_trades,
            "wins":                  total_wins,
            "losses":                total_losses,
            "breakevens":            total_breakevens,
            "win_rate":              round(win_rate, 4),
            "by_instrument":         by_instrument,
            # Compounding stats — each instrument independent, $500 each
            "starting_balance":  ACCOUNT_BALANCE,
            "ending_balance":    round(ACCOUNT_BALANCE + total_pnl_usd, 2),
            "total_pnl_usd":     round(total_pnl_usd, 2),
            "total_return_pct":  return_pct,
            "peak_balance":      round(ACCOUNT_BALANCE + total_pnl_usd, 2),
            "max_drawdown_pct":  round(max_drawdown_pct, 1),
        }

        logger.info(
            "Backtest complete: %d trades | WR=%.1f%% | wins=%d losses=%d BE=%d",
            total_trades, win_rate * 100, total_wins, total_losses, total_breakevens,
        )

        # ── Per-instrument hold-time summary ──────────────────────────────────
        logger.info("=" * 60)
        logger.info("HOLD-TIME SUMMARY  (H1 bars = hours)")
        logger.info("  %-10s  %6s  %6s  %6s  %6s  %6s  %6s  %6s",
                    "Symbol", "Trades", "WR%", "AvgAll", "AvgWin", "AvgLoss", "MaxWin", "MaxAll")
        logger.info("  " + "-" * 62)
        for sym, res in by_instrument.items():
            if "error" in res:
                continue
            hs  = res.get("hold_stats", {})
            wr  = round(res.get("win_rate", 0) * 100, 1)
            logger.info(
                "  %-10s  %6d  %5.1f%%  %5.1fh  %5.1fh  %6.1fh  %5.1fh  %5.1fh",
                sym,
                res["trades_simulated"],
                wr,
                hs.get("avg_all_h", 0),
                hs.get("avg_wins_h", 0),
                hs.get("avg_losses_h", 0),
                hs.get("max_wins_h", 0),
                hs.get("max_all_h", 0),
            )
        logger.info("=" * 60)
        logger.info("COMPOUNDING RESULTS  (starting balance: $%.2f)", ACCOUNT_BALANCE)
        logger.info("  Ending balance : $%.2f", ACCOUNT_BALANCE + total_pnl_usd)
        logger.info("  Total P&L      : $%+.2f  (%+.1f%%)", total_pnl_usd, return_pct)
        logger.info("  Peak balance   : $%.2f", ACCOUNT_BALANCE + total_pnl_usd)
        logger.info("  Max drawdown   : %.1f%%", max_drawdown_pct)
        logger.info("=" * 60)
        return summary

    # ── Per-instrument logic ──────────────────────────────────────────────────

    def _backtest_instrument(self, symbol: str, starting_balance: float | None = None) -> dict:
        """Run walk-forward backtest for one instrument."""
        from v2.settings import ACCOUNT_BALANCE
        current_balance = starting_balance if starting_balance is not None else ACCOUNT_BALANCE
        pnl_series: list[float] = []

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
        wins       = 0
        losses     = 0
        breakevens = 0
        hold_hours_wins:   list[float] = []
        hold_hours_losses: list[float] = []
        hold_hours_be:     list[float] = []

        # Walk forward: start after MIN_LOOKBACK bars, step every SCAN_STEP bars
        # Leave MAX_HOLD_BARS at the end so every trade has room to resolve
        end_bar = len(df_full) - MAX_HOLD_BARS - 1

        # ── Date-range window restriction ─────────────────────────────────────
        range_start_idx = MIN_LOOKBACK
        range_end_idx   = end_bar

        if (self._start_date is not None or self._end_date is not None) and "time" in df_full.columns:
            try:
                times = pd.to_datetime(df_full["time"], utc=True)
                data_start = times.iloc[0]
                data_end   = times.iloc[-1]
                logger.info("%s: available data %s -> %s",
                            symbol, data_start.date(), data_end.date())

                if self._start_date is not None:
                    sd = pd.Timestamp(self._start_date)
                    if sd.tzinfo is None:
                        sd = sd.tz_localize("UTC")
                    if sd > data_end:
                        logger.warning("%s: requested start %s is AFTER available data end %s — 0 trades",
                                       symbol, sd.date(), data_end.date())
                        return {"signals_evaluated": 0, "trades_simulated": 0,
                                "wins": 0, "losses": 0, "breakevens": 0,
                                "win_rate": 0.0, "pnl_usd": 0.0,
                                "pnl_series": [], "ending_balance": current_balance,
                                "note": f"No data for {sd.date()} — broker only has from {data_start.date()}"}
                    idxs = df_full.index[times >= sd].tolist()
                    if idxs:
                        range_start_idx = max(int(idxs[0]), MIN_LOOKBACK)
                        logger.info("%s: date range from %s -> bar %d", symbol, sd.date(), range_start_idx)

                if self._end_date is not None:
                    ed = pd.Timestamp(self._end_date)
                    if ed.tzinfo is None:
                        ed = ed.tz_localize("UTC")
                    if ed < data_start:
                        logger.warning("%s: requested end %s is BEFORE available data start %s — 0 trades",
                                       symbol, ed.date(), data_start.date())
                        return {"signals_evaluated": 0, "trades_simulated": 0,
                                "wins": 0, "losses": 0, "breakevens": 0,
                                "win_rate": 0.0, "pnl_usd": 0.0,
                                "pnl_series": [], "ending_balance": current_balance,
                                "note": f"No data for {ed.date()} — broker only has from {data_start.date()}"}
                    idxs = df_full.index[times <= ed].tolist()
                    if idxs:
                        range_end_idx = min(int(idxs[-1]) - MAX_HOLD_BARS, end_bar)
                        logger.info("%s: date range to   %s -> bar %d", symbol, ed.date(), range_end_idx)
                    else:
                        logger.warning("%s: no bars before %s — 0 trades", symbol, ed.date())
                        range_end_idx = range_start_idx  # empty range

            except Exception as exc:
                logger.warning("%s: date-range filter failed (%s) — using full range", symbol, exc)

        total_bars = range_end_idx - range_start_idx
        # Skip-ahead cursor: after a trade is taken, jump forward by however
        # many bars the trade was held — prevents re-entering the same setup
        # on consecutive bars (simulates real "one trade at a time" behaviour)
        skip_until_bar = range_start_idx

        for bar_idx in range(range_start_idx, range_end_idx, SCAN_STEP):
            if bar_idx < skip_until_bar:
                continue

            # Progress log every 200 bars
            bars_done = bar_idx - MIN_LOOKBACK
            if bars_done > 0 and bars_done % 200 == 0:
                pct = int(bars_done / total_bars * 100)
                logger.info(
                    "  %s: %d%% done — %d trades so far (W:%d L:%d BE:%d)",
                    symbol, pct, trades_simulated, wins, losses, breakevens,
                )

            # Slice history up to current bar (no look-ahead)
            window     = df_full.iloc[:bar_idx + 1].copy()
            window_h4  = self._slice_htf(df_h4,  window, "H4")
            window_d1  = self._slice_htf(df_d1,  window, "D1")

            for direction in ("long", "short"):
                signals_evaluated += 1

                # Pass bar_time through context so kill-zone strategy uses
                # the historical bar's timestamp instead of datetime.now(UTC)
                bar_time = window["time"].iloc[-1] if "time" in window.columns else None
                bt_context = {"bar_time": bar_time} if bar_time is not None else {}
                result = self._engine.score(symbol, direction, window, window_h4, window_d1, bt_context)

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

                # Entry checklist — skip live news check during backtest
                checklist = validate_entry(signal, window, skip_news=True)
                if not checklist.get("passed"):
                    continue

                # Simulate trade outcome using future bars
                future_bars = df_full.iloc[bar_idx + 1: bar_idx + 1 + MAX_HOLD_BARS]
                outcome = self._simulate_outcome(entry, sl, tp1, tp2, direction, future_bars)

                if outcome is None:
                    continue  # no clear outcome in window

                # Write to journal as a historical trade (use current compounded balance)
                trade_id, pnl_usd = self._write_backtest_trade(
                    signal, outcome, window, account_balance=current_balance
                )
                trades_simulated += 1

                # Update compounded balance
                current_balance = max(current_balance + pnl_usd, 0.01)
                pnl_series.append(pnl_usd)

                hold_h = outcome["bars_held"]   # H1 bars = hours
                if outcome["exit_reason"] in ("TP1", "TP2"):
                    wins += 1
                    hold_hours_wins.append(hold_h)
                elif outcome["exit_reason"] == "SL_AFTER_TP1":
                    breakevens += 1  # TP1 hit then stopped at entry
                    hold_hours_be.append(hold_h)
                else:
                    losses += 1
                    hold_hours_losses.append(hold_h)

                # Skip ahead: don't evaluate another signal until this trade
                # has closed — simulates real "one trade at a time" behaviour
                skip_until_bar = bar_idx + outcome["bars_held"] + 1

                # Break direction loop — don't take both long AND short at same bar
                break

        decisive = wins + losses
        win_rate = (wins / decisive) if decisive > 0 else 0.0
        instrument_pnl = sum(pnl_series)

        def _avg(lst):  return round(sum(lst) / len(lst), 1) if lst else 0.0
        def _max(lst):  return round(max(lst), 1)           if lst else 0.0
        def _min(lst):  return round(min(lst), 1)           if lst else 0.0

        all_hold = hold_hours_wins + hold_hours_losses + hold_hours_be
        hold_stats = {
            "avg_all_h":    _avg(all_hold),
            "avg_wins_h":   _avg(hold_hours_wins),
            "avg_losses_h": _avg(hold_hours_losses),
            "avg_be_h":     _avg(hold_hours_be),
            "max_wins_h":   _max(hold_hours_wins),
            "max_losses_h": _max(hold_hours_losses),
            "max_all_h":    _max(all_hold),
        }

        return {
            "signals_evaluated": signals_evaluated,
            "trades_simulated":  trades_simulated,
            "wins":              wins,
            "losses":            losses,
            "breakevens":        breakevens,
            "win_rate":          round(win_rate, 4),
            "pnl_usd":           round(instrument_pnl, 2),
            "pnl_series":        pnl_series,
            "ending_balance":    round(current_balance, 2),
            "hold_stats":        hold_stats,
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

        is_long      = direction.lower() in ("long", "buy")
        tp1_hit      = False
        tp1_actual   = 0.0   # actual TP1 exit price (for partial close P&L)

        for bar_num, (_, bar) in enumerate(future.iterrows()):
            bar_high = float(bar["high"])
            bar_low  = float(bar["low"])

            if is_long:
                if bar_low <= sl:
                    if tp1_hit:
                        exit_p = max(sl, entry)
                        return self._outcome(exit_p, "SL_AFTER_TP1", entry, bar_num + 1,
                                             tp1_hit=True, tp1_price=tp1_actual)
                    return self._outcome(sl, "SL", entry, bar_num + 1)

                if not tp1_hit and bar_high >= tp1:
                    tp1_hit    = True
                    tp1_actual = tp1      # record TP1 fill price
                    sl         = entry   # SL moves to BE

                if tp1_hit and bar_high >= tp2:
                    return self._outcome(tp2, "TP2", entry, bar_num + 1,
                                         tp1_hit=True, tp1_price=tp1_actual)

            else:  # short
                if bar_high >= sl:
                    if tp1_hit:
                        exit_p = min(sl, entry)
                        return self._outcome(exit_p, "SL_AFTER_TP1", entry, bar_num + 1,
                                             tp1_hit=True, tp1_price=tp1_actual)
                    return self._outcome(sl, "SL", entry, bar_num + 1)

                if not tp1_hit and bar_low <= tp1:
                    tp1_hit    = True
                    tp1_actual = tp1
                    sl         = entry

                if tp1_hit and bar_low <= tp2:
                    return self._outcome(tp2, "TP2", entry, bar_num + 1,
                                         tp1_hit=True, tp1_price=tp1_actual)

        last_price = float(future["close"].iloc[-1])
        return self._outcome(last_price, "MAX_HOLD", entry, len(future),
                             tp1_hit=tp1_hit, tp1_price=tp1_actual)

    @staticmethod
    def _outcome(
        exit_price: float,
        reason: str,
        entry: float,
        bars: int,
        tp1_hit: bool = False,
        tp1_price: float = 0.0,
    ) -> dict:
        return {
            "exit_price":  exit_price,
            "exit_reason": reason,
            "bars_held":   bars,
            "tp1_hit":     tp1_hit,
            "tp1_price":   tp1_price,
        }

    # ── Journal write ─────────────────────────────────────────────────────────

    def _write_backtest_trade(
        self,
        signal:          dict,
        outcome:         dict,
        window:          pd.DataFrame,
        account_balance: float | None = None,
    ) -> tuple[str, float]:
        """
        Write a completed backtest trade to the SQLite journal.
        Opens it as OPEN, then immediately closes it with outcome data.
        Saves ML features at entry time.

        Returns (trade_id, pnl_usd) so caller can update compounded balance.
        """
        from v2.risk.position_sizer import calculate_lot_size, calculate_risk_usd
        from v2.instrument_config import price_to_pips

        symbol    = signal["symbol"]
        entry     = float(signal["entry_price"])
        sl        = float(signal["stop_loss"])
        direction = signal["direction"]

        from v2.settings import ACCOUNT_BALANCE

        # Cap effective balance at 20× starting balance to prevent compounding overflow.
        # In real trading, margin constraints limit position growth beyond this.
        starting_balance = ACCOUNT_BALANCE
        effective_balance = min(account_balance or starting_balance,
                                starting_balance * 20)

        lot_size = calculate_lot_size(symbol, entry, sl, account_balance=effective_balance)
        risk_usd = calculate_risk_usd(symbol, entry, sl, lot_size)

        # Compute PnL — with 50% partial close at TP1
        exit_price  = float(outcome["exit_price"])
        is_long     = direction.lower() in ("long", "buy")
        tp1_was_hit = outcome.get("tp1_hit", False)
        tp1_price   = float(outcome.get("tp1_price") or 0)

        price_diff = (exit_price - entry) if is_long else (entry - exit_price)

        try:
            cfg = get_instrument(symbol)

            if tp1_was_hit and tp1_price > 0:
                # 50% closed at TP1, 50% closed at final exit
                tp1_diff   = (tp1_price - entry) if is_long else (entry - tp1_price)
                final_diff = price_diff
                pnl_tp1    = (tp1_diff   / cfg.pip_size) * cfg.pip_value_usd * lot_size * 0.5
                pnl_final  = (final_diff / cfg.pip_size) * cfg.pip_value_usd * lot_size * 0.5
                pnl_usd    = pnl_tp1 + pnl_final
                pips       = ((tp1_diff + final_diff) / 2) / cfg.pip_size
            else:
                pips       = price_diff / cfg.pip_size
                pnl_usd    = pips * cfg.pip_value_usd * lot_size

        except Exception:
            pips    = price_diff / 0.01 if price_diff != 0 else 0
            pnl_usd = risk_usd * (price_diff / abs(entry - sl)) if abs(entry - sl) > 0 else 0

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

        return trade_id, pnl_usd

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
