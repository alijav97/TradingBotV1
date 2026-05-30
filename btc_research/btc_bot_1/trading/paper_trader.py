"""
btc_bot_1/trading/paper_trader.py — BTC Paper Trading Engine.

Adapted from v2/trading/paper_trader.py.
Key differences for BTC:
  - PnL = price_diff * lots  (BTC is USD-denominated — no pip conversion)
  - Lot sizing: risk_usd / sl_distance  (BTC units, not forex lots)
  - One trade at a time (same principle as WTI)
  - No portfolio heat check (single instrument)
  - MAX_HOLD_HOURS from btc_bot_1.settings

Usage:
    pt = PaperTrader(journal=Journal(), feed=DataFeed())
    trade_id = pt.open_trade(signal)
    pt.check_all_open_trades()   # call every 5s from scheduler during KZ
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from btc_research.btc_bot_1.journal.sqlite_journal import Journal
    from btc_research.btc_bot_1.connectors.unified_data import DataFeed

from btc_research.btc_bot_1.settings import (
    SYMBOL, RISK_PCT, RISK_PCT_EARLY_TREND, TP1_RR, TP2_RR,
    MAX_HOLD_HOURS, TRAIL_ATR_MULT, ADX_PERIOD,
)

logger = logging.getLogger(__name__)

# Per-symbol lock (BTC only has one symbol but keep the pattern for safety)
_SYMBOL_LOCK = threading.Lock()


class PaperTrader:
    """BTC paper trading engine."""

    def __init__(self, journal: "Journal", feed: "DataFeed") -> None:
        self._journal = journal
        self._feed    = feed

    # ── Open ──────────────────────────────────────────────────────────────────

    def open_trade(self, signal: dict) -> str | None:
        """
        Convert a signal into a paper trade.

        Required: entry_price, stop_loss, direction
        Returns trade_id or None if blocked.
        """
        entry = float(signal.get("entry_price") or signal.get("entry", 0))
        sl    = float(signal.get("stop_loss") or signal.get("sl", 0))

        if entry <= 0 or sl <= 0:
            logger.warning("BTC: invalid signal — missing entry/SL")
            return None

        if not _SYMBOL_LOCK.acquire(blocking=False):
            logger.info("BTC: trade blocked — concurrent open_trade in progress")
            return None

        try:
            # One trade at a time
            open_trades = self._journal.get_open_trades(symbol=SYMBOL)
            if open_trades:
                logger.info(
                    "BTC: trade blocked — already have open trade %s",
                    open_trades[0]["id"][:8]
                )
                return None

            # Position sizing: BTC units = risk_usd / sl_distance
            # Version D flipped risk: signal carries the correct risk_pct
            # (3% if ADX 20-28, 2% if ADX > 28)
            current_balance = self._journal.get_paper_balance()
            risk_pct        = signal.get("risk_pct", RISK_PCT)   # from btc_engine
            risk_usd        = current_balance * risk_pct
            sl_dist         = abs(entry - sl)
            if sl_dist <= 0:
                logger.warning("BTC: zero SL distance — blocking trade")
                return None
            lot_size = round(risk_usd / sl_dist, 6)
            lot_size = max(lot_size, 0.001)   # minimum 0.001 BTC

            # TP prices from signal or recalculate
            direction = signal.get("direction", "long")
            is_long   = direction.lower() in ("long", "buy")
            tp1 = signal.get("tp1_price") or signal.get("tp1")
            tp2 = signal.get("tp2_price") or signal.get("tp2")
            if not tp1 or not tp2:
                tp1 = round(entry + TP1_RR * sl_dist, 2) if is_long else round(entry - TP1_RR * sl_dist, 2)
                tp2 = round(entry + TP2_RR * sl_dist, 2) if is_long else round(entry - TP2_RR * sl_dist, 2)

            trade = {
                "symbol":           SYMBOL,
                "direction":        direction,
                "entry_price":      entry,
                "stop_loss":        sl,
                "tp1_price":        tp1,
                "tp2_price":        tp2,
                "lot_size":         lot_size,
                "strategy":         signal.get("strategy", "BTC_Confluence_V1"),
                "confluence_score": signal.get("confluence_score") or signal.get("score"),
                "timeframe":        "H1",
                "session":          "US Late 21-24 UTC",
                "factors":          signal.get("factors", {}),
                "raw_signal":       signal,
            }

            trade_id = self._journal.open_trade(trade)
            logger.info(
                "BTC paper trade opened: %s %s @ %.2f  SL=%.2f  lots=%.4f  risk=$%.2f",
                trade_id[:8], direction.upper(), entry, sl, lot_size, risk_usd,
            )
            return trade_id

        finally:
            _SYMBOL_LOCK.release()

    # ── Monitor ───────────────────────────────────────────────────────────────

    def check_all_open_trades(self) -> list[dict]:
        """Check all open trades for SL/TP/max-hold. Returns list of actions taken."""
        open_trades = self._journal.get_open_trades(symbol=SYMBOL)
        actions: list[dict] = []
        for trade in open_trades:
            action = self.monitor_trade(trade)
            if action:
                actions.append(action)
        return actions

    def monitor_trade(self, trade: dict) -> dict | None:
        """
        Check a single open trade against current BTC price.
        Returns action dict if SL/TP/max-hold triggered, else None.
        """
        trade_id    = trade["id"]
        direction   = trade["direction"]
        entry       = float(trade["entry_price"])
        sl          = float(trade["stop_loss"])
        tp1         = float(trade["tp1_price"] or 0)
        tp2         = float(trade["tp2_price"] or 0)
        lot_size    = float(trade["lot_size"])
        tp1_hit     = bool(trade.get("tp1_hit", 0))
        open_time   = trade.get("open_time", "")
        original_sl = float(trade.get("original_sl") or trade["stop_loss"])

        # Get live BTC price (5s timeout)
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                price_info = ex.submit(self._feed.get_price, SYMBOL).result(timeout=5)
        except (FuturesTimeoutError, Exception) as exc:
            logger.debug("BTC price fetch timeout: %s — skipping monitor tick", exc)
            return None

        if not price_info or not price_info.get("price"):
            logger.debug("BTC: no live price — skipping monitor tick")
            return None

        current_price = float(price_info["price"])
        is_long = direction.lower() in ("long", "buy")

        # Hold time
        hold_minutes = 0.0
        if open_time:
            try:
                opened_at    = datetime.fromisoformat(open_time.replace("Z", "+00:00"))
                hold_minutes = (datetime.now(timezone.utc) - opened_at).total_seconds() / 60
            except Exception:
                pass

        logger.debug(
            "Monitor BTC %s: price=%.2f | entry=%.2f sl=%.2f tp1=%.2f tp2=%.2f hold=%.0fm",
            direction.upper(), current_price, entry, sl, tp1, tp2, hold_minutes,
        )

        # ── Max hold ──────────────────────────────────────────────────────────
        if hold_minutes >= MAX_HOLD_HOURS * 60:
            pnl = self._calc_pnl(entry, current_price, direction, lot_size, sl)
            self._journal.close_trade(
                trade_id, current_price, "MAX_HOLD",
                pnl_usd=pnl[0], rr_achieved=pnl[1],
            )
            logger.info("BTC trade %s closed: MAX_HOLD (%.0fh)", trade_id[:8], hold_minutes / 60)
            return {"trade_id": trade_id, "action": "MAX_HOLD", "price": current_price}

        # ── SL ────────────────────────────────────────────────────────────────
        if (is_long and current_price <= sl) or (not is_long and current_price >= sl):
            sl_reason = "SL_AFTER_TP1" if tp1_hit else "SL"
            pnl = self._calc_pnl(entry, sl, direction, lot_size, original_sl)
            self._journal.close_trade(
                trade_id, sl, sl_reason,
                pnl_usd=pnl[0], rr_achieved=pnl[1],
            )
            logger.info("BTC trade %s closed: %s @ %.2f", trade_id[:8], sl_reason, sl)
            return {"trade_id": trade_id, "action": sl_reason, "price": sl}

        # ── TP1 ───────────────────────────────────────────────────────────────
        if tp1 > 0 and not tp1_hit:
            if (is_long and current_price >= tp1) or (not is_long and current_price <= tp1):
                self._journal.mark_tp1_hit(trade_id)
                self._journal.update_stop_loss(trade_id, entry)   # move SL to BE
                self._journal.mark_breakeven(trade_id)
                logger.info("BTC trade %s: TP1 hit @ %.2f — SL moved to BE %.2f",
                            trade_id[:8], tp1, entry)
                return {"trade_id": trade_id, "action": "TP1", "price": tp1}

        # ── Trailing SL after TP1 (Version D: 2×ATR trail) ───────────────────
        # After TP1 hit, we trail the SL at 2×ATR behind the current price peak.
        # This replaces the fixed TP2 — lets winners run as BTC can move 5-10R+.
        if tp1_hit:
            current_atr = self._get_current_atr()
            if current_atr > 0:
                trail_dist = TRAIL_ATR_MULT * current_atr
                if is_long:
                    trail_sl = round(current_price - trail_dist, 2)
                    if trail_sl > sl:   # only move SL up, never down
                        self._journal.update_stop_loss(trade_id, trail_sl)
                        sl = trail_sl
                        logger.debug("BTC %s: trail SL → %.2f (price=%.2f ATR=%.2f)",
                                     trade_id[:8], trail_sl, current_price, current_atr)
                else:
                    trail_sl = round(current_price + trail_dist, 2)
                    if trail_sl < sl:   # only move SL down, never up
                        self._journal.update_stop_loss(trade_id, trail_sl)
                        sl = trail_sl
                        logger.debug("BTC %s: trail SL → %.2f (price=%.2f ATR=%.2f)",
                                     trade_id[:8], trail_sl, current_price, current_atr)

        # ── TP2 (fixed fallback if trail SL isn't active) ─────────────────────
        if tp2 > 0 and tp1_hit:
            if (is_long and current_price >= tp2) or (not is_long and current_price <= tp2):
                pnl = self._calc_pnl(entry, tp2, direction, lot_size, original_sl)
                self._journal.close_trade(
                    trade_id, tp2, "TP2",
                    pnl_usd=pnl[0], rr_achieved=pnl[1],
                )
                logger.info("BTC trade %s closed: TP2 @ %.2f  pnl=$%.2f  R=%.2f",
                            trade_id[:8], tp2, pnl[0], pnl[1])
                return {"trade_id": trade_id, "action": "TP2", "price": tp2}

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_current_atr(self) -> float:
        """Fetch current H1 ATR(14) for BTC. Returns 0 on failure."""
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                df = ex.submit(self._feed.get_ohlcv, SYMBOL, "H1", 30).result(timeout=5)
            if df.empty or len(df) < 5:
                return 0.0
            h = df["high"].astype(float)
            l = df["low"].astype(float)
            c = df["close"].astype(float)
            tr = pd.concat([
                h - l,
                (h - c.shift(1)).abs(),
                (l - c.shift(1)).abs(),
            ], axis=1).max(axis=1)
            return float(tr.rolling(ADX_PERIOD).mean().iloc[-1])
        except Exception:
            return 0.0

    def _calc_pnl(
        self,
        entry:     float,
        exit_p:    float,
        direction: str,
        lot_size:  float,
        original_sl: float,
    ) -> tuple[float, float]:
        """Returns (pnl_usd, r_multiple)."""
        is_long    = direction.lower() in ("long", "buy")
        price_diff = (exit_p - entry) if is_long else (entry - exit_p)
        pnl_usd    = round(price_diff * lot_size, 2)
        sl_dist    = abs(entry - original_sl)
        r_multiple = round(price_diff / sl_dist, 2) if sl_dist > 0 else 0.0
        return pnl_usd, r_multiple

    def get_open_summary(self) -> dict:
        """Return summary of open trades."""
        trades = self._journal.get_open_trades(symbol=SYMBOL)
        return {
            "count":  len(trades),
            "trades": [
                {
                    "id":        t["id"][:8],
                    "symbol":    t["symbol"],
                    "direction": t["direction"],
                    "entry":     t["entry_price"],
                    "sl":        t["stop_loss"],
                    "tp1":       t.get("tp1_price"),
                    "tp2":       t.get("tp2_price"),
                    "strategy":  t.get("strategy", ""),
                }
                for t in trades
            ],
        }
