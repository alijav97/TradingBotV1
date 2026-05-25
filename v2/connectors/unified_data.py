"""
connectors/unified_data.py — Single OHLCV interface for TradingBotV2.

Regardless of whether an instrument comes from MT5 or Binance, all callers
receive the same DataFrame format and the same dict shape from get_price().

Usage:
    from v2.connectors.unified_data import DataFeed
    feed = DataFeed()
    df   = feed.get_ohlcv("XAUUSD", "H1", count=300)
    df   = feed.get_ohlcv("BTCUSDT", "H4", count=300)
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from v2.instrument_config import get_instrument, InstrumentConfig
import v2.connectors.mt5_connector     as mt5_conn
import v2.connectors.binance_connector as bnb_conn

logger = logging.getLogger(__name__)

# Canonical OHLCV column order returned by every method
OHLCV_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


class DataFeed:
    """
    Unified data access layer.

    Initialise once per session:
        feed = DataFeed()
        feed.connect()

    Then call get_ohlcv / get_price from anywhere.
    """

    def __init__(self) -> None:
        self._mt5_ok     = False
        self._binance_ok = False

    # ── Connection management ─────────────────────────────────────────────────

    def connect(
        self,
        mt5_login: int = 0,
        mt5_password: str = "",
        mt5_server: str = "",
        binance_api_key: str = "",
        binance_api_secret: str = "",
        binance_testnet: bool | None = None,
    ) -> dict[str, bool]:
        """
        Connect to both brokers.
        Returns {"mt5": bool, "binance": bool}.
        Partial connectivity is fine — instruments from each source
        will work independently.
        """
        self._mt5_ok = mt5_conn.connect(
            login=mt5_login,
            password=mt5_password,
            server=mt5_server,
        )
        self._binance_ok = bnb_conn.connect(
            api_key=binance_api_key,
            api_secret=binance_api_secret,
            testnet=binance_testnet,
        )

        if not self._mt5_ok:
            logger.warning("MT5 not connected — MT5 instruments unavailable")
        if not self._binance_ok:
            logger.warning("Binance not connected — crypto instruments unavailable")

        return {"mt5": self._mt5_ok, "binance": self._binance_ok}

    def disconnect(self) -> None:
        mt5_conn.disconnect()

    # ── Core interface ────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "H1",
        count: int = 500,
    ) -> pd.DataFrame:
        """
        Return OHLCV DataFrame for any supported instrument.

        Columns: time (UTC datetime), open, high, low, close, volume
        Returns empty DataFrame if connector unavailable or symbol unknown.
        """
        try:
            cfg = get_instrument(symbol)
        except KeyError:
            logger.error("Unknown symbol: %s", symbol)
            return _empty_df()

        if cfg.source == "mt5":
            return mt5_conn.get_ohlcv(cfg.mt5_symbol, timeframe, count)
        else:
            return bnb_conn.get_ohlcv(cfg.binance_symbol, timeframe, count)

    def get_price(self, symbol: str) -> dict:
        """
        Return current live price info for any supported instrument.

        Keys always present: symbol, price
        Additional keys depend on source (bid/ask for MT5, funding for Binance).
        """
        try:
            cfg = get_instrument(symbol)
        except KeyError:
            return {"symbol": symbol, "price": 0.0, "error": "unknown symbol"}

        if cfg.source == "mt5":
            raw = mt5_conn.get_live_price(cfg.mt5_symbol)
            if not raw:
                return {"symbol": symbol, "price": 0.0}
            # Normalise: use mid price as canonical "price"
            raw["price"] = round((raw.get("bid", 0) + raw.get("ask", 0)) / 2, 5)
            raw["symbol"] = symbol
            return raw
        else:
            raw = bnb_conn.get_live_price(cfg.binance_symbol)
            if not raw:
                return {"symbol": symbol, "price": 0.0}
            raw["symbol"] = symbol
            return raw

    def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[str] | None = None,
        count: int = 300,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch multiple timeframes for one symbol in one call.

        Returns {"H1": df, "H4": df, "D1": df} (only non-empty frames).
        """
        if timeframes is None:
            timeframes = ["H1", "H4", "D1"]

        result: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            df = self.get_ohlcv(symbol, tf, count)
            if not df.empty:
                result[tf] = df
        return result

    def status(self) -> dict:
        """Return connectivity status for both brokers."""
        return {
            "mt5":     mt5_conn.is_connected(),
            "binance": bnb_conn.is_connected(),
        }


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLUMNS)


# Module-level singleton — import and use directly for convenience
_default_feed: DataFeed | None = None


def get_default_feed() -> DataFeed:
    """Return (creating if needed) the module-level DataFeed singleton."""
    global _default_feed
    if _default_feed is None:
        _default_feed = DataFeed()
    return _default_feed
