"""
btc_research/btc_bot_2/paper_trader.py — BTC Bot 2 paper trading engine.

Manages paper trades for BTCUSDT:
  - Open trade: write to journal, track SL/TP/trailing state
  - Monitor: check current price against SL, TP1, TP2, max-hold
  - Trailing SL: 2×ATR below current price after TP1 hit (for longs)
  - Max hold: 96 bars (H1) = 96 hours

== KEY DIFFERENCES FROM v2/trading/paper_trader.py ==
  - BTC-only (no instrument config lookup needed)
  - ADX-split risk (stored at open time in the signal dict)
  - Trailing SL uses ATR stored at open time (atr_at_open)
  - One active trade at a time (BTC single-instrument rule)
  - Uses btc2_trades.db (separate from Bot 1)

== USAGE ==
  from btc_research.btc_bot_2.paper_trader import BTC2PaperTrader
  pt = BTC2PaperTrader(journal=journal, feed=feed)
  trade_id = pt.open_trade(signal)
  actions  = pt.check_all_open_trades()   # call every 60s
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2.connectors.unified_data import DataFeed
    from v2.journal.sqlite_journal import Journal

from btc_research.btc_bot_2.settings import (
    TP1_RR, TP2_RR, TRAIL_ATR_MULT, MAX_HOLD_BARS,
)

logger = logging.getLogger(__name__)

# One trade at a time on BTC — lock prevents race on the "is there an open trade?" check
_TRADE_LOCK = threading.Lock()

# H1 = 1 hour per bar → max hold in seconds
_MAX_HOLD_SECONDS = MAX_HOLD_BARS * 3600   # 96h


class BTC2PaperTrader:
    """
    Paper trading engine for BTC Bot 2.

    open_trade(signal)       → open a new paper position (returns trade_id or None)
    check_all_open_trades()  → scan all open trades, return list of action dicts
    get_open_summary()       → {count, trades} for scheduler polling
    """

    def __init__(self, journal: "Journal", feed: "DataFeed") -> None:
        self._journal = journal
        self._feed    = feed

    # ── Open ──────────────────────────────────────────────────────────────────

    def open_trade(self, signal: dict) -> str | None:
        """
        Convert a signal dict into a live paper trade.

        Returns trade_id string on success, None if blocked.
        """
        with _TRADE_LOCK:
            # One open trade at a time for BTC
            try:
                open_count = self._journal.get_open_trades_count()
            except Exception:
                try:
                    open_count = len(self._journal.get_open_trades())
                except Exception:
                    open_count = 0

            if open_count > 0:
                logger.info("Trade blocked: already have %d open trade(s)", open_count)
                return None

            symbol    = signal.get("symbol", "BTCUSD")
            direction = signal.get("direction", "long")
            entry     = float(signal.get("entry_price", 0))
            sl        = float(signal.get("stop_loss", 0))
            tp1       = float(signal.get("tp1_price", 0))
            tp2       = float(signal.get("tp2_price", 0))
            lot       = float(signal.get("lot_size", 0.001))
            atr       = float(signal.get("atr", 0))

            if entry <= 0 or sl <= 0 or lot <= 0:
                logger.error("Invalid signal values: entry=%.1f sl=%.1f lot=%.4f", entry, sl, lot)
                return None

            trade_record = {
                "symbol":          symbol,
                "direction":       direction,
                "entry_price":     entry,
                "stop_loss":       sl,
                "tp1_price":       tp1,
                "tp2_price":       tp2,
                "lot_size":        lot,
                "strategy":        signal.get("strategy", "VBSwing v2"),
                "confluence_score": signal.get("adx", 0),     # use ADX as proxy score
                "timeframe":       signal.get("timeframe", "H1"),
                "session":         signal.get("session", ""),
                "notes":           (
                    f"entry_type={signal.get('entry_type', '')} "
                    f"adx={signal.get('adx', 0):.1f} "
                    f"risk={signal.get('risk_pct', 0):.1f}% "
                    f"atr={atr:.0f}"
                ),
                # Store original SL and ATR for trailing calculations
                "original_sl": sl,
                "exit_atr":    atr,   # atr_at_open — reused field to store this
                "raw_signal":  str(signal),
            }

            try:
                trade_id = self._journal.open_trade(trade_record)
                logger.info(
                    "TRADE OPENED [%s]: %s %s @ %.0f  SL=%.0f  TP1=%.0f  TP2=%.0f  lot=%.4f BTC",
                    trade_id[:8] if trade_id else "?",
                    symbol, direction.upper(), entry, sl, tp1, tp2, lot,
                )
                return trade_id
            except Exception as exc:
                logger.error("Failed to open trade: %s", exc)
                return None

    # ── Monitor ───────────────────────────────────────────────────────────────

    def check_all_open_trades(self) -> list[dict]:
        """
        Check all open trades against current price.

        Returns list of action dicts for each trade that had an event.
        Each action dict has: trade_id, action, price, pnl_usd
        """
        actions = []
        try:
            open_trades = self._journal.get_open_trades()
        except Exception as exc:
            logger.error("Failed to get open trades: %s", exc)
            return actions

        for trade in open_trades:
            try:
                action = self._check_trade(trade)
                if action:
                    actions.append(action)
            except Exception as exc:
                logger.error("Error monitoring trade %s: %s", trade.get("id", "?")[:8], exc)

        return actions

    def get_open_summary(self) -> dict:
        """Return {count, trades} for scheduler polling."""
        try:
            trades = self._journal.get_open_trades()
            return {"count": len(trades), "trades": trades}
        except Exception:
            return {"count": 0, "trades": []}

    # ── Internal: single trade check ──────────────────────────────────────────

    def _check_trade(self, trade: dict) -> dict | None:
        """
        Check a single open trade. Returns action dict or None.

        Action types:
          TP1      — first target hit, SL moved to breakeven, trade continues
          TP2      — second target hit, full close
          SL       — stop loss hit (original or after TP1)
          MAX_HOLD — forced close after 96 bars
          TRAIL_SL — trailing SL updated (log only, no journal event)
        """
        trade_id  = trade.get("id", "")
        direction = trade.get("direction", "long")
        entry     = float(trade.get("entry_price", 0))
        sl        = float(trade.get("stop_loss", 0))
        tp1       = float(trade.get("tp1_price", 0))
        tp2       = float(trade.get("tp2_price", 0))
        tp1_hit   = bool(trade.get("tp1_hit", 0))
        original_sl = float(trade.get("original_sl") or sl)
        atr_at_open = float(trade.get("exit_atr") or 0)   # stored ATR (reused field)
        lot       = float(trade.get("lot_size", 0.001))
        is_long   = direction.lower() == "long"
        symbol    = trade.get("feed_symbol", "BTCUSDT")

        # Max hold check
        open_time_str = trade.get("open_time", "")
        if open_time_str:
            try:
                open_time = datetime.fromisoformat(open_time_str.replace("Z", "+00:00"))
                held_secs = (datetime.now(timezone.utc) - open_time).total_seconds()
                if held_secs > _MAX_HOLD_SECONDS:
                    price = self._get_current_price(symbol) or entry
                    pnl   = self._calc_pnl(is_long, entry, price, lot)
                    logger.info("MAX_HOLD: %s closed after %.0fh  PnL=%.2f", trade_id[:8], held_secs / 3600, pnl)
                    self._close_trade(trade_id, price, "MAX_HOLD", pnl, lot, entry)
                    return {"trade_id": trade_id, "action": "MAX_HOLD", "price": price, "pnl_usd": pnl}
            except Exception:
                pass

        # Fetch current price
        price = self._get_current_price(symbol)
        if price is None or price <= 0:
            logger.debug("No current price for %s — skipping", symbol)
            return None

        # ── TP2 check (full close) ─────────────────────────────────────────────
        if tp2 > 0:
            tp2_hit = price >= tp2 if is_long else price <= tp2
            if tp2_hit:
                pnl = self._calc_pnl(is_long, entry, tp2, lot)
                logger.info("TP2 HIT: %s %s  entry=%.0f tp2=%.0f  PnL=+$%.2f",
                            trade_id[:8], direction.upper(), entry, tp2, pnl)
                self._close_trade(trade_id, tp2, "TP2", pnl, lot, entry)
                return {"trade_id": trade_id, "action": "TP2", "price": tp2, "pnl_usd": pnl}

        # ── TP1 check (partial, SL to BE) ─────────────────────────────────────
        if not tp1_hit and tp1 > 0:
            tp1_reached = price >= tp1 if is_long else price <= tp1
            if tp1_reached:
                logger.info("TP1 HIT: %s %s  entry=%.0f tp1=%.0f  SL→BE",
                            trade_id[:8], direction.upper(), entry, tp1)
                # Move SL to breakeven
                new_sl = entry
                self._journal.update_trade(trade_id, {
                    "tp1_hit":   1,
                    "be_moved":  1,
                    "stop_loss": new_sl,
                })
                return {"trade_id": trade_id, "action": "TP1", "price": tp1, "pnl_usd": 0.0}

        # ── SL check ──────────────────────────────────────────────────────────
        sl_hit = price <= sl if is_long else price >= sl
        if sl_hit:
            pnl = self._calc_pnl(is_long, entry, sl, lot)
            reason = "SL_AFTER_TP1" if tp1_hit else "SL"
            logger.info("SL HIT: %s %s  entry=%.0f sl=%.0f  PnL=$%.2f  reason=%s",
                        trade_id[:8], direction.upper(), entry, sl, pnl, reason)
            self._close_trade(trade_id, sl, reason, pnl, lot, entry)
            return {"trade_id": trade_id, "action": reason, "price": sl, "pnl_usd": pnl}

        # ── Trailing SL update (after TP1) ────────────────────────────────────
        if tp1_hit and atr_at_open > 0:
            trail_dist = TRAIL_ATR_MULT * atr_at_open
            new_trail  = (round(price - trail_dist, 1) if is_long
                          else round(price + trail_dist, 1))

            # Only tighten (ratchet up for longs, down for shorts)
            if is_long and new_trail > sl:
                self._journal.update_trade(trade_id, {"stop_loss": new_trail})
                logger.debug("TRAIL SL: %s  old=%.0f → new=%.0f  price=%.0f",
                             trade_id[:8], sl, new_trail, price)
            elif not is_long and new_trail < sl:
                self._journal.update_trade(trade_id, {"stop_loss": new_trail})
                logger.debug("TRAIL SL: %s  old=%.0f → new=%.0f  price=%.0f",
                             trade_id[:8], sl, new_trail, price)

        return None   # nothing notable

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_current_price(self, symbol: str) -> float | None:
        """Return current mid price for BTCUSDT from DataFeed."""
        try:
            tick = self._feed.get_price(symbol)
            if tick and tick.get("price"):
                return float(tick["price"])
        except Exception as exc:
            logger.debug("get_price(%s) error: %s", symbol, exc)
        return None

    @staticmethod
    def _calc_pnl(is_long: bool, entry: float, exit_price: float, lot_size: float) -> float:
        """
        Calculate PnL for a BTC paper trade.

        BTC: 1 lot = 1 BTC.  $1 move = $1 per BTC.
        PnL = (exit - entry) * lot_size  (positive for winning longs)
        """
        dist = exit_price - entry if is_long else entry - exit_price
        return round(dist * lot_size, 2)

    def _close_trade(
        self,
        trade_id: str,
        exit_price: float,
        reason: str,
        pnl_usd: float,
        lot_size: float,
        entry_price: float,
    ) -> None:
        """Write close event to journal."""
        try:
            sl_dist  = abs(exit_price - entry_price)
            rr_val   = 0.0
            if sl_dist > 0:
                # Approximate R achieved from the TP ratios
                # (proper R calculation needs original SL dist)
                rr_val = round(pnl_usd / (sl_dist * lot_size), 2) if lot_size > 0 else 0.0

            self._journal.close_trade(
                trade_id      = trade_id,
                exit_price    = exit_price,
                exit_reason   = reason,
                pnl_usd       = pnl_usd,
                rr_achieved   = rr_val,
            )
        except Exception as exc:
            logger.error("Failed to close trade %s in journal: %s", trade_id[:8], exc)
