"""
btc_research/btc_bot_2/telegram.py — Telegram alert wrapper for BTC Bot 2.

Thin wrapper around v2's TelegramAlerter that:
  - Prefixes all messages with "[BTC BOT 2]" so alerts are distinct from Bot 1
  - Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment (same as Bot 1)
    but uses BTC2_TELEGRAM_CHAT_ID if set (allows separate channel for Bot 2)
  - Adds BTC2-specific message formats (ADX, entry_type, risk %)

Usage:
    from btc_research.btc_bot_2.telegram import BTC2Alerter
    alerter = BTC2Alerter()
    alerter.send_signal_opened(signal_dict)
    alerter.send_trade_opened(trade_dict)
    alerter.send_trade_closed(trade_dict)
    alerter.send_text("custom message")
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT_SEC  = 10
_BOT_LABEL    = "BTC BOT 2"


class BTC2Alerter:
    """
    Telegram alerter for BTC Bot 2.

    Reads tokens from environment:
      TELEGRAM_BOT_TOKEN     — shared with Bot 1 (or set separately)
      BTC2_TELEGRAM_CHAT_ID  — Bot 2 Telegram channel (falls back to TELEGRAM_CHAT_ID)
    """

    def __init__(
        self,
        token:   str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self._token   = token   or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = (
            chat_id
            or os.environ.get("BTC2_TELEGRAM_CHAT_ID", "")
            or os.environ.get("TELEGRAM_CHAT_ID", "")
        )

        if not self._token:
            logger.info("BTC2Alerter: TELEGRAM_BOT_TOKEN not set — alerts disabled")
        else:
            logger.info("BTC2Alerter ready (chat_id=%s)", self._chat_id or "(not set)")

    # ── Public API ─────────────────────────────────────────────────────────────

    def send_signal_opened(self, signal: dict) -> bool:
        """Alert for a new signal that passed all filters."""
        direction   = signal.get("direction", "?").upper()
        entry       = signal.get("entry_price", 0)
        sl          = signal.get("stop_loss", 0)
        tp1         = signal.get("tp1_price", 0)
        tp2         = signal.get("tp2_price", 0)
        adx         = signal.get("adx", 0)
        risk_pct    = signal.get("risk_pct", 0)
        strategy    = signal.get("strategy", "")
        entry_type  = signal.get("entry_type", "")
        lot         = signal.get("lot_size", 0)
        session     = signal.get("session", "")
        sl_dist     = signal.get("sl_dist", 0)
        now         = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        sl_r        = abs(entry - sl)
        risk_r_str  = f"SL: {sl_dist:.0f} pts"

        entry_label = f"{'Retest' if entry_type == 'retest' else 'Break'}" if entry_type else strategy

        msg = (
            f"[{_BOT_LABEL}] SIGNAL\n"
            f"BTCUSD {direction}\n"
            f"Entry:    {entry:,.0f}\n"
            f"SL:       {sl:,.0f}  ({sl_dist:.0f} pts)\n"
            f"TP1:      {tp1:,.0f}  ({signal.get('tp1_rr', 2.0):.1f}R)\n"
            f"TP2:      {tp2:,.0f}  ({signal.get('tp2_rr', 5.0):.1f}R)\n"
            f"Size:     {lot:.4f} BTC\n"
            f"Risk:     {risk_pct:.1f}% | ADX: {adx:.1f}\n"
            f"Entry:    {entry_label}\n"
            f"Session:  {session}\n"
            f"Time:     {now}"
        )
        return self.send_text(msg)

    def send_trade_opened(self, trade: dict) -> bool:
        """Alert for a trade written to journal."""
        symbol    = trade.get("symbol", "BTCUSD")
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
            f"Entry: {entry:,.0f}\n"
            f"SL:    {sl:,.0f}\n"
            f"TP1:   {tp1:,.0f}\n"
            f"TP2:   {tp2:,.0f}\n"
            f"Size:  {lot:.4f} BTC\n"
            f"Strat: {strategy}\n"
            f"Info:  {notes}\n"
            f"Time:  {now}"
        )
        return self.send_text(msg)

    def send_tp1_hit(self, trade: dict, price: float) -> bool:
        """Alert for TP1 hit — SL moved to breakeven, trade continues."""
        symbol    = trade.get("symbol", "BTCUSD")
        direction = trade.get("direction", "?").upper()
        entry     = trade.get("entry_price", 0)
        tp2       = trade.get("tp2_price", 0)
        trade_id  = str(trade.get("id", ""))[:8]
        now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        msg = (
            f"[{_BOT_LABEL}] TP1 HIT - SL to breakeven\n"
            f"ID:    {trade_id}\n"
            f"{symbol} {direction}\n"
            f"Entry: {entry:,.0f}\n"
            f"TP1:   {price:,.0f} (HIT)\n"
            f"SL:    {entry:,.0f} (moved to BE)\n"
            f"TP2:   {tp2:,.0f} (still open)\n"
            f"Time:  {now}"
        )
        return self.send_text(msg)

    def send_trade_closed(self, trade: dict) -> bool:
        """Alert for trade closure (SL, TP2, MAX_HOLD, MANUAL)."""
        symbol     = trade.get("symbol", "BTCUSD")
        direction  = trade.get("direction", "?").upper()
        entry      = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        reason     = trade.get("exit_reason", "?")
        pnl        = float(trade.get("pnl_usd") or 0)
        rr         = trade.get("rr_achieved", 0)
        trade_id   = str(trade.get("id", ""))[:8]
        now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        icon     = "WIN" if pnl > 0 else ("BREAKEVEN" if pnl == 0 else "LOSS")
        pnl_sign = "+" if pnl >= 0 else ""

        msg = (
            f"[{_BOT_LABEL}] TRADE CLOSED - {icon}\n"
            f"ID:     {trade_id}\n"
            f"{symbol} {direction}\n"
            f"Entry:  {entry:,.0f}\n"
            f"Exit:   {exit_price:,.0f}\n"
            f"Reason: {reason}\n"
            f"PnL:    {pnl_sign}${pnl:.2f}\n"
            f"R:R:    {rr}\n"
            f"Time:   {now}"
        )
        return self.send_text(msg)

    def send_startup(self, balance: float) -> bool:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        msg = (
            f"[{_BOT_LABEL}] STARTED\n"
            f"Strategy: VB + Swing Level v2 [both 2xATR]\n"
            f"Kill-zone: 01:00, 02:00, 03:00, 08:00 UTC\n"
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

        pnl_sign = "+" if total_pnl >= 0 else ""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        msg = (
            f"[{_BOT_LABEL}] MORNING BRIEFING\n"
            f"7d: {trades} trades | WR: {win_rate:.1f}% | PnL: {pnl_sign}${total_pnl:.2f} | PF: {pf:.2f}\n"
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
