"""
trading/paper_trader.py — Autonomous paper trading engine for TradingBotV2.

Converts signals into paper trades, monitors them, and closes them
automatically when SL/TP/max-hold conditions are met.

All state lives in SQLite (via Journal) — no JSON files.

Usage:
    from v2.trading.paper_trader import PaperTrader
    from v2.journal.sqlite_journal import Journal
    from v2.connectors.unified_data import DataFeed

    pt = PaperTrader(journal=Journal(), feed=DataFeed())
    trade_id = pt.open_trade(signal)   # from confluence engine
    pt.check_all_open_trades()         # call every 60s from scheduler
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2.journal.sqlite_journal import Journal
    from v2.connectors.unified_data import DataFeed

from v2.instrument_config import get_instrument, price_to_pips
from v2.risk.position_sizer import calculate_lot_size, calculate_tp_prices, calculate_risk_usd
from v2.risk.portfolio_heat import PortfolioHeat

logger = logging.getLogger(__name__)

MAX_HOLD_HOURS = 48   # force-close any trade open longer than this


class PaperTrader:
    """
    Autonomous paper trading engine.

    open_trade()           → open a new paper position
    check_all_open_trades() → scan all open trades for SL/TP/max-hold
    monitor_trade()        → check a single trade
    """

    def __init__(self, journal: "Journal", feed: "DataFeed") -> None:
        self._journal = journal
        self._feed    = feed
        self._heat    = PortfolioHeat(journal)

    # ── Open ──────────────────────────────────────────────────────────────────

    def open_trade(self, signal: dict) -> str | None:
        """
        Convert a signal dict into a paper trade.

        Required signal keys: symbol, direction, entry_price (or entry),
                              stop_loss, strategy
        Optional: tp1_price, tp2_price, confluence_score, timeframe,
                  session, regime, news_score

        Returns trade_id on success, None if blocked by risk checks.
        """
        symbol    = signal.get("symbol", "")
        direction = signal.get("direction", "long")
        entry     = float(signal.get("entry_price") or signal.get("entry", 0))
        sl        = float(signal.get("stop_loss", 0))

        if not symbol or entry <= 0 or sl <= 0:
            logger.warning("Invalid signal — missing symbol/entry/SL")
            return None

        # Position sizing
        lot_size = calculate_lot_size(symbol, entry, sl)
        risk_usd = calculate_risk_usd(symbol, entry, sl, lot_size)

        # Portfolio heat check
        allowed, reason = self._heat.can_open_trade(symbol, direction, risk_usd)
        if not allowed:
            logger.info("Trade blocked for %s: %s", symbol, reason)
            return None

        # TP prices (use signal values if provided, else calculate)
        tp1 = signal.get("tp1_price") or signal.get("tp1")
        tp2 = signal.get("tp2_price") or signal.get("tp2")
        if not tp1 or not tp2:
            tp1, tp2 = calculate_tp_prices(entry, sl, direction)

        trade = {
            "symbol":           symbol,
            "direction":        direction,
            "entry_price":      entry,
            "stop_loss":        sl,
            "tp1_price":        tp1,
            "tp2_price":        tp2,
            "lot_size":         lot_size,
            "strategy":         signal.get("strategy", ""),
            "confluence_score": signal.get("confluence_score") or signal.get("score"),
            "timeframe":        signal.get("timeframe", "H1"),
            "session":          signal.get("session", ""),
            "regime":           signal.get("regime", ""),
            "news_score":       signal.get("news_score"),
            "factors":          signal.get("factors", {}),   # per-factor breakdown for ML
            "raw_signal":       signal,
        }

        trade_id = self._journal.open_trade(trade)
        logger.info("Paper trade opened: %s %s %s @ %.5f (lots=%.3f risk=$%.2f)",
                    trade_id[:8], symbol, direction, entry, lot_size, risk_usd)
        return trade_id

    # ── Monitor ───────────────────────────────────────────────────────────────

    def check_all_open_trades(self) -> list[dict]:
        """
        Check all open trades against current prices.
        Called every 60 seconds by the scheduler.
        Returns list of dicts describing any actions taken.
        """
        open_trades = self._journal.get_open_trades()
        actions: list[dict] = []

        for trade in open_trades:
            action = self.monitor_trade(trade)
            if action:
                actions.append(action)

        return actions

    def monitor_trade(self, trade: dict) -> dict | None:
        """
        Check a single open trade.
        Returns action dict if something happened (SL/TP/max-hold), else None.
        """
        trade_id  = trade["id"]
        symbol    = trade["symbol"]
        direction = trade["direction"]
        entry     = float(trade["entry_price"])
        sl        = float(trade["stop_loss"])
        tp1       = float(trade["tp1_price"] or 0)
        tp2       = float(trade["tp2_price"] or 0)
        lot_size  = float(trade["lot_size"])
        tp1_hit   = bool(trade.get("tp1_hit", 0))
        open_time = trade.get("open_time", "")

        # Get current price — 5s timeout so a slow MT5 never blocks the scheduler
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                price_info = ex.submit(self._feed.get_price, symbol).result(timeout=5)
        except (FuturesTimeoutError, Exception) as exc:
            logger.debug("Price fetch timeout/error for %s: %s — skipping monitor tick", symbol, exc)
            return None
        if not price_info or not price_info.get("price"):
            return None
        current_price = float(price_info["price"])

        is_long = direction.lower() in ("long", "buy")

        # ── Hold time ─────────────────────────────────────────────────────────
        hold_minutes = 0.0
        if open_time:
            try:
                opened_at = datetime.fromisoformat(open_time.replace("Z", "+00:00"))
                hold_minutes = (datetime.now(timezone.utc) - opened_at).total_seconds() / 60
            except Exception:
                pass

        # ── Fetch ATR for trailing SL ─────────────────────────────────────────
        current_atr = self._get_current_atr(symbol)

        # ── Trailing SL after TP1 ─────────────────────────────────────────────
        if tp1_hit and current_atr and current_atr > 0:
            trail_dist = 1.5 * current_atr
            if is_long:
                trail_sl = current_price - trail_dist
                if trail_sl > sl:  # only move up, never down
                    self._journal.update_stop_loss(trade_id, trail_sl)
                    sl = trail_sl
            else:
                trail_sl = current_price + trail_dist
                if trail_sl < sl:  # only move down, never up
                    self._journal.update_stop_loss(trade_id, trail_sl)
                    sl = trail_sl

        # ── Max hold check ────────────────────────────────────────────────────
        if hold_minutes >= MAX_HOLD_HOURS * 60:
            pnl = self._calc_pnl(symbol, entry, current_price, direction, lot_size)
            ctx = self._build_exit_context(symbol, hold_minutes, current_atr)
            self._journal.close_trade(trade_id, current_price, "MAX_HOLD",
                                      pnl_usd=pnl[0], pips=pnl[1], exit_context=ctx)
            logger.info("Trade %s closed: MAX_HOLD (%.0fh)", trade_id[:8], hold_minutes / 60)
            return {"trade_id": trade_id, "action": "MAX_HOLD", "price": current_price}

        # ── SL check ──────────────────────────────────────────────────────────
        if (is_long and current_price <= sl) or (not is_long and current_price >= sl):
            pnl = self._calc_pnl(symbol, entry, sl, direction, lot_size)
            ctx = self._build_exit_context(symbol, hold_minutes, current_atr)
            self._journal.close_trade(trade_id, sl, "SL",
                                      pnl_usd=pnl[0], pips=pnl[1], exit_context=ctx)
            logger.info("Trade %s closed: SL hit at %.5f", trade_id[:8], sl)
            return {"trade_id": trade_id, "action": "SL", "price": sl}

        # ── TP1 check ─────────────────────────────────────────────────────────
        if tp1 > 0 and not tp1_hit:
            if (is_long and current_price >= tp1) or (not is_long and current_price <= tp1):
                self._journal.mark_tp1_hit(trade_id)
                self._journal.update_stop_loss(trade_id, entry)  # move SL to BE
                self._journal.mark_breakeven(trade_id)
                logger.info("Trade %s: TP1 hit at %.5f — SL moved to BE", trade_id[:8], tp1)
                return {"trade_id": trade_id, "action": "TP1", "price": tp1}

        # ── TP2 check ─────────────────────────────────────────────────────────
        if tp2 > 0 and tp1_hit:
            if (is_long and current_price >= tp2) or (not is_long and current_price <= tp2):
                pnl = self._calc_pnl(symbol, entry, tp2, direction, lot_size)
                ctx = self._build_exit_context(symbol, hold_minutes, current_atr)
                self._journal.close_trade(trade_id, tp2, "TP2",
                                          pnl_usd=pnl[0], pips=pnl[1], rr_achieved=pnl[2],
                                          exit_context=ctx)
                logger.info("Trade %s closed: TP2 at %.5f", trade_id[:8], tp2)
                return {"trade_id": trade_id, "action": "TP2", "price": tp2}

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _calc_pnl(
        self,
        symbol: str,
        entry: float,
        exit_price: float,
        direction: str,
        lot_size: float,
    ) -> tuple[float, float, float]:
        """Returns (pnl_usd, pips, rr_ratio)."""
        try:
            cfg      = get_instrument(symbol)
            is_long  = direction.lower() in ("long", "buy")
            price_diff = (exit_price - entry) if is_long else (entry - exit_price)
            pips       = price_to_pips(symbol, abs(price_diff)) * (1 if price_diff >= 0 else -1)
            pnl_usd    = float(pips) * cfg.pip_value_usd * lot_size
            sl_dist    = cfg.pip_size  # fallback — real SL dist not available here
            rr         = abs(pips) / (abs(price_diff) / cfg.pip_size) if sl_dist > 0 else 0.0
            return round(pnl_usd, 2), round(pips, 1), round(rr, 2)
        except Exception:
            return 0.0, 0.0, 0.0

    def _get_current_atr(self, symbol: str) -> float:
        """Fetch the current ATR for a symbol (used for trailing SL). Returns 0 on failure."""
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                df = ex.submit(self._feed.get_ohlcv, symbol, "H1", 20).result(timeout=5)
            if df.empty or len(df) < 5:
                return 0.0
            high, low, close = df["high"], df["low"], df["close"]
            tr = (high - low).combine(
                (high - close.shift(1)).abs(), max
            ).combine(
                (low - close.shift(1)).abs(), max
            )
            return float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
        except Exception:
            return 0.0

    def _build_exit_context(self, symbol: str, hold_minutes: float, atr: float) -> dict:
        """Build exit-time context dict for ML enrichment."""
        ctx: dict = {"hold_time_minutes": round(hold_minutes, 1), "exit_atr": round(atr, 5)}
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                df = ex.submit(self._feed.get_ohlcv, symbol, "H1", 50).result(timeout=5)
            if not df.empty:
                from v2.analysis.indicators import get_adx
                adx_result = get_adx(df)
                ctx["exit_regime"] = adx_result.get("bias", "unknown")
        except Exception:
            pass
        return ctx

    def get_open_summary(self) -> dict:
        """Return summary of all open paper trades."""
        trades = self._journal.get_open_trades()
        heat   = self._heat.get_heat_summary()
        return {
            "count":       len(trades),
            "heat":        heat,
            "trades":      [
                {
                    "id":        t["id"][:8],
                    "symbol":    t["symbol"],
                    "direction": t["direction"],
                    "entry":     t["entry_price"],
                    "sl":        t["stop_loss"],
                    "tp1":       t.get("tp1_price"),
                    "strategy":  t.get("strategy", ""),
                }
                for t in trades
            ],
        }
