"""
btc_bot_1/connectors/unified_data.py — Data feed for BTC Bot 1.

Simplified vs v2: BTC bot only needs MT5 (BTCUSD, XAUUSD, NAS100 are all
on Pepperstone MT5). No Binance connector needed.

Usage:
    from btc_research.btc_bot_1.connectors.unified_data import DataFeed
    feed = DataFeed()
    feed.connect()
    df_btc  = feed.get_ohlcv("BTCUSD", "H1", count=500)
    df_gold = feed.get_ohlcv("XAUUSD", "H1", count=500)
    df_nas  = feed.get_ohlcv("NAS100", "H1", count=500)
"""
from __future__ import annotations

import logging
import pandas as pd

import btc_research.btc_bot_1.connectors.mt5_connector as mt5_conn

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


class DataFeed:
    """MT5-backed data feed for BTC Bot 1."""

    def __init__(self) -> None:
        self._mt5_ok = False

    def connect(
        self,
        mt5_login:    int = 0,
        mt5_password: str = "",
        mt5_server:   str = "",
    ) -> dict[str, bool]:
        """Connect to MT5. Returns {"mt5": bool}."""
        self._mt5_ok = mt5_conn.connect(
            login    = mt5_login,
            password = mt5_password,
            server   = mt5_server,
        )
        if not self._mt5_ok:
            logger.warning("MT5 not connected — data unavailable")
        return {"mt5": self._mt5_ok}

    def disconnect(self) -> None:
        mt5_conn.disconnect()

    def get_ohlcv(
        self,
        symbol:    str,
        timeframe: str = "H1",
        count:     int = 500,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame. Columns: time (UTC), open, high, low, close, volume."""
        if not self._mt5_ok:
            return _empty_df()
        df = mt5_conn.get_ohlcv(symbol, timeframe, count)
        if df.empty:
            logger.warning("Empty data returned for %s %s", symbol, timeframe)
        return df

    def get_price(self, symbol: str) -> dict:
        """Return current live price. Keys: symbol, price, bid, ask, age_seconds."""
        if not self._mt5_ok:
            return {"symbol": symbol, "price": 0.0}
        raw = mt5_conn.get_live_price(symbol)
        if not raw:
            return {"symbol": symbol, "price": 0.0}
        return raw

    def status(self) -> dict:
        return {"mt5": mt5_conn.is_connected()}


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLUMNS)
