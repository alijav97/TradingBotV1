"""
connectors/binance_connector.py — Binance Futures connector for TradingBotV2.

Uses python-binance SDK for BTC/USDT and ETH/USDT USDT-M perpetual futures.

Provides:
  - get_ohlcv(symbol, interval, limit) → pd.DataFrame
  - get_live_price(symbol) → dict
  - get_order_book(symbol, depth) → dict
  - get_funding_rate(symbol) → float
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    _BINANCE_OK = True
except ImportError:
    Client = None  # type: ignore
    BinanceAPIException = Exception  # type: ignore
    _BINANCE_OK = False
    logger.warning("python-binance not installed — Binance connector in stub mode")

# ── Interval mapping ──────────────────────────────────────────────────────────
INTERVALS: dict[str, str] = {
    "M1":  "1m",
    "M5":  "5m",
    "M15": "15m",
    "M30": "30m",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1d",
    "W1":  "1w",
}

_client: "Client | None" = None


def connect(api_key: str = "", api_secret: str = "", testnet: bool | None = None) -> bool:
    """
    Initialise Binance Futures client.
    Credentials read from settings.py if not passed directly.
    Returns True on success.
    """
    global _client
    if not _BINANCE_OK:
        logger.warning("python-binance not available — stub mode")
        return False

    from v2.settings import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET
    _key     = api_key    or BINANCE_API_KEY
    _secret  = api_secret or BINANCE_API_SECRET
    _testnet = testnet if testnet is not None else BINANCE_TESTNET

    try:
        _client = Client(_key, _secret, testnet=_testnet)
        _client.futures_ping()  # verify connection
        logger.info("Binance Futures connected (testnet=%s)", _testnet)
        return True
    except BinanceAPIException as exc:
        logger.error("Binance connection failed: %s", exc)
        _client = None
        return False
    except Exception as exc:
        logger.error("Binance connect error: %s", exc)
        _client = None
        return False


def is_connected() -> bool:
    if _client is None:
        return False
    try:
        _client.futures_ping()
        return True
    except Exception:
        return False


def get_ohlcv(
    symbol: str,
    interval: str = "H1",
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch OHLCV klines from Binance Futures.

    Returns DataFrame with columns: time, open, high, low, close, volume
    Returns empty DataFrame on failure.
    """
    if _client is None:
        return _empty_ohlcv()

    binance_interval = INTERVALS.get(interval.upper(), "1h")

    try:
        klines = _client.futures_klines(
            symbol=symbol.upper(),
            interval=binance_interval,
            limit=limit,
        )
    except BinanceAPIException as exc:
        logger.error("Binance klines error %s %s: %s", symbol, interval, exc)
        return _empty_ohlcv()

    if not klines:
        return _empty_ohlcv()

    rows = []
    for k in klines:
        rows.append({
            "time":   datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            "open":   float(k[1]),
            "high":   float(k[2]),
            "low":    float(k[3]),
            "close":  float(k[4]),
            "volume": float(k[5]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_live_price(symbol: str) -> dict:
    """Return latest mark price for a Binance Futures symbol."""
    if _client is None:
        return {}
    try:
        ticker = _client.futures_mark_price(symbol=symbol.upper())
        return {
            "symbol":     symbol,
            "price":      float(ticker["markPrice"]),
            "index":      float(ticker.get("indexPrice", 0)),
            "funding":    float(ticker.get("lastFundingRate", 0)),
            "time":       datetime.fromtimestamp(
                            ticker.get("time", 0) / 1000, tz=timezone.utc
                          ).isoformat(),
        }
    except BinanceAPIException as exc:
        logger.error("Binance price error %s: %s", symbol, exc)
        return {}


def get_order_book(symbol: str, depth: int = 10) -> dict:
    """Return top N bids and asks from Binance Futures order book."""
    if _client is None:
        return {"bids": [], "asks": []}
    try:
        ob = _client.futures_order_book(symbol=symbol.upper(), limit=depth)
        return {
            "bids": [[float(p), float(q)] for p, q in ob.get("bids", [])],
            "asks": [[float(p), float(q)] for p, q in ob.get("asks", [])],
        }
    except BinanceAPIException as exc:
        logger.error("Binance order book error %s: %s", symbol, exc)
        return {"bids": [], "asks": []}


def get_funding_rate(symbol: str) -> float:
    """Return current funding rate (positive = longs pay shorts)."""
    if _client is None:
        return 0.0
    try:
        data = _client.futures_mark_price(symbol=symbol.upper())
        return float(data.get("lastFundingRate", 0.0))
    except Exception:
        return 0.0


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
