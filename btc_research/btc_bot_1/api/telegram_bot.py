"""
btc_bot_1/api/telegram_bot.py — Telegram alert sender for BTC Bot 1.

Same pattern as v2/api/telegram_bot.py but uses BTC-specific
TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from btc_bot_1.settings.

All messages are prefixed with [BTC] for easy identification in
Telegram when both bots are active.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT_SEC  = 10


class TelegramAlerter:
    """Sends formatted Telegram alerts for BTC trading events."""

    def __init__(
        self,
        token:   str | None = None,
        chat_id: str | None = None,
    ) -> None:
        from btc_research.btc_bot_1.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        self._token   = token   or TELEGRAM_BOT_TOKEN
        self._chat_id = chat_id or TELEGRAM_CHAT_ID

        if not self._token:
            logger.info("BTC TelegramAlerter: token not set — alerts disabled")
        else:
            logger.info("BTC TelegramAlerter ready (chat_id=%s)", self._chat_id or "(not set)")

    # ── Alert methods ─────────────────────────────────────────────────────────

    def send_trade_opened(self, trade: dict) -> bool:
        direction = (trade.get("direction") or "").upper()
        entry     = trade.get("entry_price", 0)
        sl        = trade.get("stop_loss", 0)
        tp1       = trade.get("tp1_price", 0)
        tp2       = trade.get("tp2_price", 0)
        lot       = trade.get("lot_size", 0)
        score     = trade.get("confluence_score", 0)
        strategy  = trade.get("strategy", "")
        now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        sl_dist   = abs(entry - sl) if entry and sl else 0

        msg = (
            f"[BTC] TRADE OPENED\n"
            f"Direction:  {direction}\n"
            f"Entry:      ${entry:,.2f}\n"
            f"Stop Loss:  ${sl:,.2f}  (${sl_dist:,.2f} away)\n"
            f"TP1:        ${tp1:,.2f}  (2R)\n"
            f"TP2:        ${tp2:,.2f}  (5R)\n"
            f"BTC units:  {lot:.4f}\n"
            f"Score:      {score:.1f}\n"
            f"Strategy:   {strategy}\n"
            f"Session:    US Late (21-24 UTC)\n"
            f"Time:       {now}"
        )
        return self.send_text(msg)

    def send_tp1_hit(self, trade: dict, tp1_price: float) -> bool:
        direction = (trade.get("direction") or "").upper()
        entry     = trade.get("entry_price", 0)
        now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        msg = (
            f"[BTC] TP1 HIT — SL moved to breakeven\n"
            f"Direction:  {direction}\n"
            f"Entry:      ${entry:,.2f}\n"
            f"TP1 price:  ${tp1_price:,.2f}\n"
            f"SL now at:  ${entry:,.2f} (breakeven)\n"
            f"Time:       {now}\n"
            f"Still open — targeting TP2"
        )
        return self.send_text(msg)

    def send_trade_closed(self, trade: dict) -> bool:
        direction  = (trade.get("direction") or "").upper()
        reason     = trade.get("exit_reason", "")
        entry      = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        pnl        = float(trade.get("pnl_usd") or 0)
        rr         = trade.get("rr_achieved", 0)
        now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        icon     = "WIN" if pnl > 0 else ("BREAKEVEN" if pnl == 0 else "LOSS")
        pnl_sign = "+" if pnl > 0 else ""

        msg = (
            f"[BTC] TRADE CLOSED — {icon}\n"
            f"Direction:  {direction}\n"
            f"Reason:     {reason}\n"
            f"Entry:      ${entry:,.2f}\n"
            f"Exit:       ${exit_price:,.2f}\n"
            f"PnL:        {pnl_sign}${pnl:.2f}\n"
            f"R multiple: {rr}R\n"
            f"Time:       {now}"
        )
        return self.send_text(msg)

    def send_morning_briefing(self, stats: dict) -> bool:
        """Send daily stats summary."""
        wr    = stats.get("win_rate", 0.0)
        pnl   = stats.get("total_pnl", 0.0)
        trades = stats.get("trades", 0)
        pf    = stats.get("profit_factor", 0.0)
        pnl_sign = "+" if pnl >= 0 else ""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d UTC")

        msg = (
            f"[BTC] DAILY BRIEFING — {now}\n"
            f"30d Stats:\n"
            f"  Trades: {trades}\n"
            f"  Win rate: {wr:.1f}%\n"
            f"  PnL: {pnl_sign}${pnl:.2f}\n"
            f"  Profit factor: {pf:.2f}\n"
            f"Kill-zone opens at 21:00 UTC (01:00 UAE)"
        )
        return self.send_text(msg)

    def send_text(self, message: str) -> bool:
        """Send a raw text message. Returns False (never raises) on failure."""
        if not self._token:
            logger.debug("BTC Telegram send skipped — no token")
            return False
        if not self._chat_id:
            logger.warning("BTC Telegram send skipped — no chat_id")
            return False

        url     = _TELEGRAM_API.format(token=self._token)
        payload = {
            "chat_id":    self._chat_id,
            "text":       message,
            "parse_mode": "HTML",
        }
        try:
            resp = requests.post(url, json=payload, timeout=_TIMEOUT_SEC)
            if resp.status_code == 200:
                return True
            logger.warning("BTC Telegram API %d: %s", resp.status_code, resp.text[:200])
            return False
        except requests.exceptions.Timeout:
            logger.warning("BTC Telegram timeout after %ds", _TIMEOUT_SEC)
            return False
        except requests.exceptions.RequestException as exc:
            logger.warning("BTC Telegram error: %s", exc)
            return False
