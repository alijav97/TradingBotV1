"""
btc_research/eth_bot/telegram.py — Telegram alert wrapper for ETH Bot.

Uses DEDICATED ETH bot credentials ONLY (no cross-bot fallback):
  ETH_TELEGRAM_BOT_TOKEN  — bot token from BotFather for the ETH bot
  ETH_TELEGRAM_CHAT_ID    — chat/channel ID for ETH alerts

There is intentionally NO fallback to the shared TELEGRAM_BOT_TOKEN /
TELEGRAM_CHAT_ID: falling back would post ETH alerts into another bot's channel
(this happened once — ETH alerts landed in the WTI/SpotCrude channel). If the
ETH-specific vars are not set, alerts are simply DISABLED rather than sent to the
wrong place. To share one Telegram bot across instruments, set
ETH_TELEGRAM_BOT_TOKEN explicitly (it may equal the other bot's token) and point
ETH_TELEGRAM_CHAT_ID at a separate ETH channel.

All messages prefixed with [ETH BOT] so alerts are distinct in Telegram.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT_SEC  = 10
_BOT_LABEL    = "ETH BOT"


class ETHAlerter:
    """Telegram alerter for ETH Bot."""

    def __init__(
        self,
        token:   str | None = None,
        chat_id: str | None = None,
    ) -> None:
        # DEDICATED ETH credentials only — no shared/cross-bot fallback (see module
        # docstring). This guarantees ETH alerts never post into another bot's channel.
        self._token   = token   or os.environ.get("ETH_TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.environ.get("ETH_TELEGRAM_CHAT_ID", "")

        if not self._token or not self._chat_id:
            logger.warning(
                "ETHAlerter: alerts DISABLED — set ETH_TELEGRAM_BOT_TOKEN and "
                "ETH_TELEGRAM_CHAT_ID in .env to enable (token=%s, chat_id=%s). "
                "No fallback to the shared/WTI channel is used.",
                "set" if self._token else "missing",
                "set" if self._chat_id else "missing",
            )
        else:
            src = "ETH_TELEGRAM_BOT_TOKEN" if not token else "__init__ arg"
            logger.info(
                "ETHAlerter ready — token src: %s  |  chat_id: %s",
                src, self._chat_id,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def send_trade_opened(self, trade: dict) -> bool:
        symbol    = trade.get("symbol", "ETHUSD")
        direction = trade.get("direction", "?").upper()
        entry     = trade.get("entry_price", 0)
        sl        = trade.get("stop_loss", 0)
        tp1       = trade.get("tp1_price", 0)
        tp2       = trade.get("tp2_price", 0)
        lot       = trade.get("lot_size", 0)
        strategy  = trade.get("strategy", "")
        trade_id  = str(trade.get("id", ""))[:8]
        notes     = trade.get("notes", "")
        now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        msg = (
            f"[{_BOT_LABEL}] TRADE OPENED\n"
            f"ID:    {trade_id}\n"
            f"{symbol} {direction}\n"
            f"Entry: {entry:,.4f}\n"
            f"SL:    {sl:,.4f}\n"
            f"TP1:   {tp1:,.4f}\n"
            f"TP2:   {tp2:,.4f}\n"
            f"Size:  {lot:.4f} ETH\n"
            f"Strat: {strategy}\n"
            f"Info:  {notes}\n"
            f"Time:  {now}"
        )
        return self.send_text(msg)

    def send_tp1_hit(self, trade: dict, price: float) -> bool:
        symbol    = trade.get("symbol", "ETHUSD")
        direction = trade.get("direction", "?").upper()
        entry     = trade.get("entry_price", 0)
        tp2       = trade.get("tp2_price", 0)
        trade_id  = str(trade.get("id", ""))[:8]
        now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        msg = (
            f"[{_BOT_LABEL}] TP1 HIT — SL to breakeven\n"
            f"ID:    {trade_id}\n"
            f"{symbol} {direction}\n"
            f"Entry: {entry:,.4f}\n"
            f"TP1:   {price:,.4f} (HIT)\n"
            f"SL:    {entry:,.4f} (moved to BE)\n"
            f"TP2:   {tp2:,.4f} (still open)\n"
            f"Time:  {now}"
        )
        return self.send_text(msg)

    def send_trade_closed(self, trade: dict) -> bool:
        symbol     = trade.get("symbol", "ETHUSD")
        direction  = trade.get("direction", "?").upper()
        entry      = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        reason     = trade.get("exit_reason", "?")
        pnl        = float(trade.get("pnl_usd") or 0)
        rr         = trade.get("rr_achieved", 0)
        trade_id   = str(trade.get("id", ""))[:8]
        now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        icon     = "WIN ✅" if pnl > 0 else ("BREAKEVEN" if pnl == 0 else "LOSS ❌")
        pnl_sign = "+" if pnl >= 0 else ""

        msg = (
            f"[{_BOT_LABEL}] TRADE CLOSED — {icon}\n"
            f"ID:     {trade_id}\n"
            f"{symbol} {direction}\n"
            f"Entry:  {entry:,.4f}\n"
            f"Exit:   {exit_price:,.4f}\n"
            f"Reason: {reason}\n"
            f"PnL:    {pnl_sign}${pnl:.2f}\n"
            f"R:R:    {rr}\n"
            f"Time:   {now}"
        )
        return self.send_text(msg)

    def send_startup(self, balance: float) -> bool:
        kz_hours = ", ".join(
            f"{h:02d}:00"
            for h in __import__("btc_research.eth_bot.settings",
                                fromlist=["KZ_HOURS"]).KZ_HOURS
        )
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        msg = (
            f"[{_BOT_LABEL}] STARTED\n"
            f"Strategy: VBSwing (SwingLevelV2 > VB) | flat 8% + throttle-2\n"
            f"Kill-zone: {kz_hours} UTC\n"
            f"Balance:  ${balance:,.2f}\n"
            f"Time:     {now}"
        )
        return self.send_text(msg)

    def send_shutdown(self, reason: str = "manual stop") -> bool:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        msg = (
            f"[{_BOT_LABEL}] STOPPED\n"
            f"Reason: {reason}\n"
            f"Time:   {now}"
        )
        return self.send_text(msg)

    def send_morning_briefing(self, stats: dict) -> bool:
        win_rate  = stats.get("win_rate", 0.0)
        total_pnl = stats.get("total_pnl", 0.0)
        trades    = stats.get("trades", 0)
        pf        = stats.get("profit_factor", 0.0)
        balance   = stats.get("current_balance", 0.0)
        pnl_sign  = "+" if total_pnl >= 0 else ""
        now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        msg = (
            f"[{_BOT_LABEL}] MORNING BRIEFING\n"
            f"7d: {trades} trades | WR: {win_rate:.1f}% | "
            f"PnL: {pnl_sign}${total_pnl:.2f} | PF: {pf:.2f}\n"
            f"Balance: ${balance:,.2f}\n"
            f"Time: {now}"
        )
        return self.send_text(msg)

    def send_text(self, message: str) -> bool:
        """Send raw text to Telegram. Returns False (never raises) on any error."""
        if not self._token:
            logger.debug("Telegram send skipped — no token")
            return False
        if not self._chat_id:
            logger.warning("Telegram send skipped — no chat_id")
            return False

        url     = _TELEGRAM_API.format(token=self._token)
        payload = {"chat_id": self._chat_id, "text": message, "parse_mode": "HTML"}

        try:
            resp = requests.post(url, json=payload, timeout=_TIMEOUT_SEC)
            if resp.status_code == 200:
                return True
            logger.warning("Telegram API %d: %s", resp.status_code, resp.text[:200])
            return False
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False
