"""
btc_bot_1/connectors/mt5_connector.py — MT5 connector for BTC Bot 1.

Duplicated from v2/connectors/mt5_connector.py.
Only change: imports MT5_SERVER_UTC_OFFSET from btc_bot_1.settings
instead of v2.settings.

Full isolation — no shared state with the WTI bot.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore
    _MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not available — MT5 connector in stub mode")


TIMEFRAMES: dict[str, int] = {}
if _MT5_AVAILABLE:
    TIMEFRAMES = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
    }

_connected = False


def connect(login: int = 0, password: str = "", server: str = "") -> bool:
    """Initialise MT5 connection. Returns True on success."""
    global _connected
    if not _MT5_AVAILABLE:
        logger.warning("MT5 not available — running in stub mode")
        return False

    from btc_research.btc_bot_1.settings import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    _login    = login    or MT5_LOGIN
    _password = password or MT5_PASSWORD
    _server   = server   or MT5_SERVER

    if not mt5.initialize():
        logger.error("MT5 initialize() failed: %s", mt5.last_error())
        return False

    if _login:
        authorized = mt5.login(_login, password=_password, server=_server)
        if not authorized:
            logger.error("MT5 login failed: %s", mt5.last_error())
            mt5.shutdown()
            return False

    _connected = True
    info = mt5.account_info()
    if info:
        logger.info("MT5 connected — account %s, balance %.2f", info.login, info.balance)
    return True


def disconnect() -> None:
    global _connected
    if _MT5_AVAILABLE and _connected:
        mt5.shutdown()
    _connected = False


def is_connected() -> bool:
    if not _MT5_AVAILABLE:
        return False
    try:
        return mt5.account_info() is not None
    except Exception:
        return False


def get_ohlcv(
    symbol: str,
    timeframe: str = "H1",
    count: int = 500,
) -> pd.DataFrame:
    """Fetch OHLCV candles from MT5. Returns DataFrame with UTC timestamps."""
    if not _MT5_AVAILABLE or not _connected:
        return _empty_ohlcv()

    tf = TIMEFRAMES.get(timeframe.upper())
    if tf is None:
        logger.error("Unknown timeframe: %s", timeframe)
        return _empty_ohlcv()

    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        logger.warning("No rates returned for %s %s: %s", symbol, timeframe, mt5.last_error())
        return _empty_ohlcv()

    df = pd.DataFrame(rates)
    from btc_research.btc_bot_1.settings import MT5_SERVER_UTC_OFFSET
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    if MT5_SERVER_UTC_OFFSET != 0:
        df["time"] = df["time"] - pd.Timedelta(hours=MT5_SERVER_UTC_OFFSET)
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_live_price(symbol: str, max_age_seconds: int = 300) -> dict:
    """Return latest bid/ask for a symbol. Returns empty dict on failure or stale data."""
    if not _MT5_AVAILABLE or not _connected:
        return {}

    mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.warning("get_live_price: no tick for %s", symbol)
        return {}

    tick_utc  = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    now_utc   = datetime.now(timezone.utc)
    age_secs  = (now_utc - tick_utc).total_seconds()

    if age_secs > max_age_seconds:
        logger.warning(
            "get_live_price: %s tick is STALE (age=%.0fs) — skipping",
            symbol, age_secs,
        )
        return {}

    mid = round((tick.bid + tick.ask) / 2, 2)
    display_age = max(round(age_secs, 1), 0.0)

    return {
        "symbol":      symbol,
        "bid":         round(tick.bid, 2),
        "ask":         round(tick.ask, 2),
        "price":       mid,
        "spread":      round(tick.ask - tick.bid, 2),
        "time":        tick_utc.isoformat(),
        "age_seconds": display_age,
    }


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
