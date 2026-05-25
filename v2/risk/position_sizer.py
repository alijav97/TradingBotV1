"""
risk/position_sizer.py — Account-based position sizing for TradingBotV2.

Calculates lot size from:
  - account balance
  - risk % per trade
  - stop loss distance in price terms
  - instrument pip value

Usage:
    from v2.risk.position_sizer import calculate_lot_size, calculate_risk_usd
    lot = calculate_lot_size("XAUUSD", entry=3310.0, stop_loss=3295.0, risk_pct=1.0)
"""
from __future__ import annotations

import logging
import math

from v2.instrument_config import get_instrument, price_to_pips

logger = logging.getLogger(__name__)


def calculate_lot_size(
    symbol: str,
    entry: float,
    stop_loss: float,
    risk_pct: float | None = None,
    account_balance: float | None = None,
) -> float:
    """
    Calculate lot size so that a SL hit costs exactly risk_pct % of account.

    Returns lot size rounded to instrument's lot_step.
    Returns min_lot on any error.
    """
    from v2.settings import RISK_PER_TRADE_PCT, ACCOUNT_BALANCE
    risk_pct  = risk_pct        if risk_pct        is not None else RISK_PER_TRADE_PCT
    balance   = account_balance if account_balance is not None else ACCOUNT_BALANCE

    try:
        cfg        = get_instrument(symbol)
        risk_usd   = balance * (risk_pct / 100.0)
        sl_dist    = abs(entry - stop_loss)
        sl_pips    = price_to_pips(symbol, sl_dist)

        if sl_pips <= 0:
            logger.warning("Zero SL pips for %s — using min_lot", symbol)
            return cfg.min_lot

        # pip_value_usd is per pip per 1 standard lot
        # risk_usd = lots × sl_pips × pip_value_usd
        raw_lot = risk_usd / (sl_pips * cfg.pip_value_usd)

        # Round down to lot_step
        lot = _round_down(raw_lot, cfg.lot_step)
        lot = max(cfg.min_lot, lot)

        logger.debug("Size %s: balance=%.2f risk=%.2f%% SL=%.1f pips → %.3f lots",
                     symbol, balance, risk_pct, sl_pips, lot)
        return lot

    except Exception as exc:
        logger.error("Position sizing error for %s: %s", symbol, exc)
        try:
            return get_instrument(symbol).min_lot
        except Exception:
            return 0.01


def calculate_risk_usd(
    symbol: str,
    entry: float,
    stop_loss: float,
    lot_size: float,
) -> float:
    """Return the USD risk amount for a given lot size and SL distance."""
    try:
        cfg     = get_instrument(symbol)
        sl_pips = price_to_pips(symbol, abs(entry - stop_loss))
        return round(sl_pips * cfg.pip_value_usd * lot_size, 2)
    except Exception:
        return 0.0


def calculate_tp_prices(
    entry: float,
    stop_loss: float,
    direction: str,
    rr1: float = 2.0,
    rr2: float = 3.0,
) -> tuple[float, float]:
    """
    Return (tp1_price, tp2_price) based on RR ratios.

    rr1=2.0 means TP1 is 2× the SL distance from entry.
    """
    sl_dist = abs(entry - stop_loss)
    if direction.lower() in ("long", "buy"):
        tp1 = round(entry + sl_dist * rr1, 5)
        tp2 = round(entry + sl_dist * rr2, 5)
    else:
        tp1 = round(entry - sl_dist * rr1, 5)
        tp2 = round(entry - sl_dist * rr2, 5)
    return tp1, tp2


def _round_down(value: float, step: float) -> float:
    """Round value DOWN to the nearest step multiple."""
    if step <= 0:
        return value
    return math.floor(value / step) * step
