"""
api/telegram_bot.py — Outbound Telegram alert sender for TradingBotV2.

This module is purely for SENDING alerts to a Telegram chat.
It is NOT a bidirectional bot and handles no incoming commands.

All send_* methods return False (and log a warning) if:
  - TELEGRAM_BOT_TOKEN is not configured
  - The Telegram API call fails for any reason

The bot will never raise an exception — callers don't need try/except.

Usage:
    from v2.api.telegram_bot import TelegramAlerter
    alerter = TelegramAlerter()
    alerter.send_signal(signal_dict)
    alerter.send_trade_opened(trade_dict)
"""
from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT_SEC = 10


class TelegramAlerter:
    """
    Sends formatted Telegram messages for trading events.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from v2.settings at
    construction time (or they can be passed directly for testing).
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        from v2.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        self._token   = token   or TELEGRAM_BOT_TOKEN
        self._chat_id = chat_id or TELEGRAM_CHAT_ID

        if not self._token:
            logger.info("TelegramAlerter: TELEGRAM_BOT_TOKEN not set — alerts disabled")
        else:
            logger.info("TelegramAlerter ready (chat_id=%s)", self._chat_id or "(not set)")

    # ── Public API ────────────────────────────────────────────────────────────

    def send_signal(self, signal: dict) -> bool:
        """
        Send a new trading signal alert.

        Expected signal keys:
            symbol, direction, entry_price (or entry), stop_loss (or sl),
            tp1_price (or tp1), score (or confluence_score), strategy
        """
        symbol    = signal.get("symbol", "?")
        direction = (signal.get("direction") or "").upper()
        price     = signal.get("entry_price") or signal.get("entry") or 0
        sl        = signal.get("stop_loss") or signal.get("sl") or 0
        score     = signal.get("score") or signal.get("confluence_score") or 0
        strategy  = signal.get("strategy", "")

        msg = (
            f"🚨 SIGNAL\n"
            f"{symbol} {direction}\n"
            f"Entry: {price}\n"
            f"SL: {sl}\n"
            f"Score: {score}/12\n"
            f"Strategy: {strategy}"
        )
        return self.send_text(msg)

    def send_trade_opened(self, trade: dict) -> bool:
        """
        Send an alert when a paper trade is opened.

        Expected trade keys:
            symbol, direction, entry_price, stop_loss, tp1_price, lot_size
        """
        symbol    = trade.get("symbol", "?")
        direction = (trade.get("direction") or "").upper()
        entry     = trade.get("entry_price", 0)
        sl        = trade.get("stop_loss", 0)
        tp1       = trade.get("tp1_price", 0)
        lot       = trade.get("lot_size", 0)

        msg = (
            f"✅ TRADE OPENED\n"
            f"{symbol} {direction}\n"
            f"@ {entry}\n"
            f"SL: {sl} | TP1: {tp1}\n"
            f"Lot: {lot}"
        )
        return self.send_text(msg)

    def send_trade_closed(self, trade: dict) -> bool:
        """
        Send an alert when a paper trade is closed.

        Expected trade keys:
            symbol, direction, exit_reason, pnl_usd (or pnl), rr_achieved
        """
        symbol     = trade.get("symbol", "?")
        direction  = (trade.get("direction") or "").upper()
        reason     = trade.get("exit_reason", "")
        pnl        = float(trade.get("pnl_usd") or trade.get("pnl") or 0)
        rr         = trade.get("rr_achieved", 0)

        icon     = "🟢" if pnl > 0 else "🔴"
        pnl_sign = "+" if pnl > 0 else ""

        msg = (
            f"{icon} TRADE CLOSED\n"
            f"{symbol} {direction} {reason}\n"
            f"PnL: {pnl_sign}{pnl:.2f}\n"
            f"R:R: {rr}"
        )
        return self.send_text(msg)

    def send_morning_briefing(self, stats: dict, calendar: dict) -> bool:
        """
        Send the daily morning briefing.

        stats    — output of Journal.get_stats()
        calendar — output of intelligence.news_filter.get_calendar_summary()
        """
        win_rate  = stats.get("win_rate", 0.0)
        total_pnl = stats.get("total_pnl", 0.0)
        trades    = stats.get("trades", 0)
        pf        = stats.get("profit_factor", 0.0)

        event_count = calendar.get("count", 0)
        warnings    = calendar.get("warnings", [])
        high_impact = calendar.get("high_impact", [])

        pnl_sign = "+" if total_pnl >= 0 else ""

        lines = [
            "☀️ MORNING BRIEFING",
            f"7d Stats: {trades} trades | WR: {win_rate:.1f}% | PnL: {pnl_sign}{total_pnl:.2f} | PF: {pf:.2f}",
            f"Calendar: {event_count} event(s) today",
        ]

        if high_impact:
            lines.append("⚠️ High-impact: " + ", ".join(str(e) for e in high_impact[:5]))

        if warnings:
            for w in warnings[:3]:
                lines.append(f"🚫 {w}")

        return self.send_text("\n".join(lines))

    def send_text(self, message: str) -> bool:
        """
        Send a raw text message to the configured Telegram chat.

        All other send_* methods call this.  Returns False (never raises)
        if the token is not configured or the API call fails.
        """
        if not self._token:
            logger.debug("Telegram send skipped — no token configured")
            return False

        if not self._chat_id:
            logger.warning("Telegram send skipped — TELEGRAM_CHAT_ID not set")
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
            logger.warning(
                "Telegram API returned %d: %s",
                resp.status_code,
                resp.text[:200],
            )
            return False
        except requests.exceptions.Timeout:
            logger.warning("Telegram API timeout after %ds", _TIMEOUT_SEC)
            return False
        except requests.exceptions.ConnectionError as exc:
            logger.warning("Telegram API connection error: %s", exc)
            return False
        except requests.exceptions.RequestException as exc:
            logger.warning("Telegram API request error: %s", exc)
            return False
