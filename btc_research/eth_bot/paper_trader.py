"""
btc_research/eth_bot/paper_trader.py — ETH Bot paper trading engine.

Mirrors btc_bot_2/paper_trader.py exactly, adapted for ETH:
  - ETH-only (no instrument config lookup)
  - ADX-split risk (stored at open time in signal dict)
  - Trailing SL uses ATR stored at open time
  - One active trade at a time (one-trade rule)
  - Uses eth_trades.db (separate from BTC bots)

PnL formula:
  ETH: 1 lot = 1 ETH. $1 move = $1 per ETH (same mechanics as BTC).
  PnL = (exit - entry) × lot_size  (positive for winning longs)
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from btc_research.eth_bot.connectors.unified_data import DataFeed
    from btc_research.eth_bot.journal.sqlite_journal  import Journal

from btc_research.eth_bot.settings import TP1_RR, TP2_RR, TRAIL_ATR_MULT, MAX_HOLD_BARS

logger = logging.getLogger(__name__)

_TRADE_LOCK       = threading.Lock()
_MAX_HOLD_SECONDS = MAX_HOLD_BARS * 3600   # 96h


class ETHPaperTrader:
    """
    Paper trading engine for ETH Bot.

    open_trade(signal)       → open a new paper position (returns trade_id or None)
    check_all_open_trades()  → scan all open trades, return list of action dicts
    get_open_summary()       → {count, trades} for scheduler polling
    """

    def __init__(self, journal: "Journal", feed: "DataFeed") -> None:
        self._journal = journal
        self._feed    = feed

    # ── Open ───────────────────────────────────────────────────────────────────

    def open_trade(self, signal: dict) -> str | None:
        """
        Convert a signal dict into a live paper trade.
        Returns trade_id string on success, None if blocked.
        """
        with _TRADE_LOCK:
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

            symbol    = signal.get("symbol", "ETHUSD")
            direction = signal.get("direction", "long")
            entry     = float(signal.get("entry_price", 0))
            sl        = float(signal.get("stop_loss", 0))
            tp1       = float(signal.get("tp1_price", 0))
            tp2       = float(signal.get("tp2_price", 0))
            lot       = float(signal.get("lot_size", 0.01))
            atr       = float(signal.get("atr", 0))

            if entry <= 0 or sl <= 0 or lot <= 0:
                logger.error("Invalid signal values: entry=%.4f sl=%.4f lot=%.4f",
                             entry, sl, lot)
                return None

            trade_record = {
                "symbol":           symbol,
                "direction":        direction,
                "entry_price":      entry,
                "stop_loss":        sl,
                "tp1_price":        tp1,
                "tp2_price":        tp2,
                "lot_size":         lot,
                "strategy":         signal.get("strategy", "ETH Strategy"),
                "confluence_score": signal.get("adx", 0),
                "timeframe":        signal.get("timeframe", "H1"),
                "session":          signal.get("session", ""),
                "notes": (
                    f"entry_type={signal.get('entry_type', '')} "
                    f"adx={signal.get('adx', 0):.1f} "
                    f"risk={signal.get('risk_pct', 0):.1f}% "
                    f"atr={atr:.2f}"
                ),
                "original_sl": sl,
                "exit_atr":    atr,
                "raw_signal":  str(signal),
            }

            try:
                trade_id = self._journal.open_trade(trade_record)
                logger.info(
                    "TRADE OPENED [%s]: %s %s @ %.4f  SL=%.4f  TP1=%.4f  TP2=%.4f  lot=%.4f ETH",
                    trade_id[:8] if trade_id else "?",
                    symbol, direction.upper(), entry, sl, tp1, tp2, lot,
                )
                return trade_id
            except Exception as exc:
                logger.error("Failed to open trade: %s", exc)
                return None

    # ── Monitor ────────────────────────────────────────────────────────────────

    def check_all_open_trades(self) -> list[dict]:
        """
        Check all open trades against current price.
        Returns list of action dicts for each trade that had an event.
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
                logger.error("Error monitoring trade %s: %s",
                             trade.get("id", "?")[:8], exc)

        return actions

    def get_open_summary(self) -> dict:
        """Return {count, trades} for scheduler polling."""
        try:
            trades = self._journal.get_open_trades()
            return {"count": len(trades), "trades": trades}
        except Exception:
            return {"count": 0, "trades": []}

    # ── Internal: single trade check ───────────────────────────────────────────

    def _check_trade(self, trade: dict) -> dict | None:
        """
        Check a single open trade. Returns action dict or None.

        Action types:
          TP1      — first target hit, SL moved to breakeven, trade continues
          TP2      — second target hit, full close
          SL       — stop loss hit (original or after TP1)
          MAX_HOLD — forced close after 96 bars
          TRAIL_SL — trailing SL updated (log only, no close)
        """
        trade_id    = trade.get("id", "")
        direction   = trade.get("direction", "long")
        entry       = float(trade.get("entry_price", 0))
        sl          = float(trade.get("stop_loss", 0))
        tp1         = float(trade.get("tp1_price", 0))
        tp2         = float(trade.get("tp2_price", 0))
        tp1_hit     = bool(trade.get("tp1_hit", 0))
        atr_at_open = float(trade.get("exit_atr") or 0)
        lot         = float(trade.get("lot_size", 0.01))
        is_long     = direction.lower() == "long"
        symbol      = trade.get("feed_symbol", "ETHUSD")

        # Max hold check
        open_time_str = trade.get("open_time", "")
        if open_time_str:
            try:
                open_time = datetime.fromisoformat(open_time_str.replace("Z", "+00:00"))
                held_secs = (datetime.now(timezone.utc) - open_time).total_seconds()
                if held_secs > _MAX_HOLD_SECONDS:
                    price = self._get_current_price(symbol) or entry
                    pnl   = self._calc_pnl(is_long, entry, price, lot)
                    logger.info("MAX_HOLD: %s closed after %.0fh  PnL=%.2f",
                                trade_id[:8], held_secs / 3600, pnl)
                    self._close_trade(trade_id, price, "MAX_HOLD", pnl, lot, entry)
                    return {"trade_id": trade_id, "action": "MAX_HOLD",
                            "price": price, "pnl_usd": pnl}
            except Exception:
                pass

        # Fetch current price
        price = self._get_current_price(symbol)
        if price is None or price <= 0:
            logger.debug("No current price for %s — skipping", symbol)
            return None

        # ── TP2 check ─────────────────────────────────────────────────────────
        if tp2 > 0:
            tp2_hit = price >= tp2 if is_long else price <= tp2
            if tp2_hit:
                pnl = self._calc_pnl(is_long, entry, tp2, lot)
                logger.info("TP2 HIT: %s %s  entry=%.4f tp2=%.4f  PnL=+$%.2f",
                            trade_id[:8], direction.upper(), entry, tp2, pnl)
                self._close_trade(trade_id, tp2, "TP2", pnl, lot, entry)
                return {"trade_id": trade_id, "action": "TP2",
                        "price": tp2, "pnl_usd": pnl}

        # ── TP1 check ─────────────────────────────────────────────────────────
        if not tp1_hit and tp1 > 0:
            tp1_reached = price >= tp1 if is_long else price <= tp1
            if tp1_reached:
                logger.info("TP1 HIT: %s %s  entry=%.4f tp1=%.4f  SL→BE",
                            trade_id[:8], direction.upper(), entry, tp1)
                self._journal.update_trade(trade_id, {
                    "tp1_hit":   1,
                    "be_moved":  1,
                    "stop_loss": entry,   # SL to breakeven
                })
                return {"trade_id": trade_id, "action": "TP1",
                        "price": tp1, "pnl_usd": 0.0}

        # ── SL check ──────────────────────────────────────────────────────────
        sl_hit = price <= sl if is_long else price >= sl
        if sl_hit:
            pnl    = self._calc_pnl(is_long, entry, sl, lot)
            reason = "SL_AFTER_TP1" if tp1_hit else "SL"
            logger.info("SL HIT: %s %s  entry=%.4f sl=%.4f  PnL=$%.2f  reason=%s",
                        trade_id[:8], direction.upper(), entry, sl, pnl, reason)
            self._close_trade(trade_id, sl, reason, pnl, lot, entry)
            return {"trade_id": trade_id, "action": reason,
                    "price": sl, "pnl_usd": pnl}

        # ── Trailing SL (after TP1) ────────────────────────────────────────────
        if tp1_hit and atr_at_open > 0:
            trail_dist = TRAIL_ATR_MULT * atr_at_open
            new_trail  = (round(price - trail_dist, 2) if is_long
                          else round(price + trail_dist, 2))

            if is_long and new_trail > sl:
                self._journal.update_trade(trade_id, {"stop_loss": new_trail})
                logger.debug("TRAIL SL: %s  old=%.4f → new=%.4f  price=%.4f",
                             trade_id[:8], sl, new_trail, price)
            elif not is_long and new_trail < sl:
                self._journal.update_trade(trade_id, {"stop_loss": new_trail})
                logger.debug("TRAIL SL: %s  old=%.4f → new=%.4f  price=%.4f",
                             trade_id[:8], sl, new_trail, price)

        return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_current_price(self, symbol: str) -> float | None:
        """Return current mid price for ETHUSD from DataFeed."""
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
        Calculate PnL for an ETH paper trade.
        ETH: 1 lot = 1 ETH. $1 move = $1 per ETH.
        """
        dist = exit_price - entry if is_long else entry - exit_price
        return round(dist * lot_size, 2)

    def _close_trade(
        self,
        trade_id:    str,
        exit_price:  float,
        reason:      str,
        pnl_usd:     float,
        lot_size:    float,
        entry_price: float,
    ) -> None:
        """Write close event to journal with true R calculation."""
        try:
            # True R = pnl / risk_usd — needs original_sl
            # Approximation here: use exit dist / original sl dist
            original_sl = 0.0
            try:
                trade = self._journal.get_trade(trade_id)
                if trade:
                    original_sl = float(trade.get("original_sl") or 0)
            except Exception:
                pass

            if original_sl > 0 and lot_size > 0:
                orig_sl_dist = abs(entry_price - original_sl)
                risk_usd     = orig_sl_dist * lot_size
                rr_val       = round(pnl_usd / risk_usd, 2) if risk_usd > 0 else 0.0
            else:
                rr_val = 0.0

            self._journal.close_trade(
                trade_id     = trade_id,
                exit_price   = exit_price,
                exit_reason  = reason,
                pnl_usd      = pnl_usd,
                rr_achieved  = rr_val,
            )
        except Exception as exc:
            logger.error("Failed to close trade %s in journal: %s", trade_id[:8], exc)
