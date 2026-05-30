"""
instrument_config.py — Per-instrument constants for TradingBotV2.

Covers all 6 instruments: XAUUSD, GBP/JPY, WTI, NAS100, BTC/USDT, ETH/USDT.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

DataSource = Literal["mt5", "binance"]


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str               # canonical name used throughout V2
    source: DataSource        # which connector to use
    mt5_symbol: str           # MT5 symbol name (empty if binance)
    binance_symbol: str       # Binance symbol (empty if mt5)
    pip_value_usd: float      # USD value of 1 pip on 1 standard lot
    pip_size: float           # size of 1 pip in price terms
    min_lot: float            # minimum lot size
    lot_step: float           # lot size increment
    max_leverage: int         # max leverage allowed
    sessions: list[str]       # best trading sessions
    correlated_with: list[str] = field(default_factory=list)  # correlated instruments
    description: str = ""


INSTRUMENTS: dict[str, InstrumentConfig] = {

    "XAUUSD": InstrumentConfig(
        symbol          = "XAUUSD",
        source          = "mt5",
        mt5_symbol      = "XAUUSD",
        binance_symbol  = "",
        pip_value_usd   = 1.0,       # $1 per pip per 0.01 lot (gold: $10/pip per lot)
        pip_size        = 0.1,       # 1 pip = $0.10 on XAUUSD
        min_lot         = 0.01,
        lot_step        = 0.01,
        max_leverage    = 10,
        sessions        = ["London", "NewYork", "LondonNY"],
        correlated_with = ["DXY", "WTI"],
        description     = "Gold / USD — most traded commodity pair",
    ),

    "GBPJPY": InstrumentConfig(
        symbol          = "GBPJPY",
        source          = "mt5",
        mt5_symbol      = "GBPJPY",
        binance_symbol  = "",
        pip_value_usd   = 0.65,      # approx $6.5 per pip per lot (varies with USDJPY)
        pip_size        = 0.01,
        min_lot         = 0.01,
        lot_step        = 0.01,
        max_leverage    = 10,
        sessions        = ["London", "LondonNY"],
        correlated_with = ["GBPUSD", "USDJPY"],
        description     = "GBP/JPY — volatile cross pair, London specialist",
    ),

    "WTI": InstrumentConfig(
        symbol          = "WTI",
        source          = "mt5",
        mt5_symbol      = "SpotCrude",
        binance_symbol  = "",
        pip_value_usd   = 1.0,       # $10 per pip per lot ($0.01 = 1 pip for oil)
        pip_size        = 0.01,
        min_lot         = 0.01,
        lot_step        = 0.01,
        max_leverage    = 10,
        sessions        = ["NewYork", "LondonNY"],
        correlated_with = ["XAUUSD", "CAD"],
        description     = "WTI Crude Oil — NY open specialist",
    ),

    "NAS100": InstrumentConfig(
        symbol          = "NAS100",
        source          = "mt5",
        mt5_symbol      = "NAS100",  # may be "US100" or "NASDAQ" on some brokers
        binance_symbol  = "",
        pip_value_usd   = 0.25,      # $2.50 per pip per lot (index: $0.25/pip)
        pip_size        = 1.0,       # 1 pip = 1 index point on NAS100
        min_lot         = 0.01,
        lot_step        = 0.01,
        max_leverage    = 5,
        sessions        = ["NewYork"],
        correlated_with = ["SPX500", "VIX"],
        description     = "Nasdaq 100 — NY session only",
    ),

    "BTCUSD": InstrumentConfig(
        symbol          = "BTCUSD",
        source          = "mt5",
        mt5_symbol      = "BTCUSD",   # Pepperstone MT5 symbol
        binance_symbol  = "",
        pip_value_usd   = 1.0,        # $1 per $1 move on 1 BTC
        pip_size        = 1.0,        # 1 pip = $1 on BTC
        min_lot         = 0.001,      # 0.001 BTC minimum
        lot_step        = 0.001,
        max_leverage    = 3,
        sessions        = ["24/7"],
        correlated_with = ["ETHUSD", "TOTAL_CRYPTO"],
        description     = "Bitcoin / USD via MT5 (Pepperstone)",
    ),

    "BTCUSDT": InstrumentConfig(
        symbol          = "BTCUSDT",
        source          = "binance",
        mt5_symbol      = "",
        binance_symbol  = "BTCUSDT",
        pip_value_usd   = 1.0,       # $1 per $1 move on 1 BTC (leveraged futures)
        pip_size        = 1.0,       # 1 pip = $1 on BTC
        min_lot         = 0.001,     # 0.001 BTC minimum on Binance Futures
        lot_step        = 0.001,
        max_leverage    = 3,
        sessions        = ["24/7"],
        correlated_with = ["ETHUSD", "TOTAL_CRYPTO"],
        description     = "Bitcoin / USDT Perpetual Futures",
    ),

    "ETHUSDT": InstrumentConfig(
        symbol          = "ETHUSDT",
        source          = "binance",
        mt5_symbol      = "",
        binance_symbol  = "ETHUSDT",
        pip_value_usd   = 0.10,      # $0.10 per $0.10 move on 1 ETH
        pip_size        = 0.1,       # 1 pip = $0.10 on ETH
        min_lot         = 0.01,      # 0.01 ETH minimum on Binance Futures
        lot_step        = 0.01,
        max_leverage    = 3,
        sessions        = ["24/7"],
        correlated_with = ["BTCUSDT", "TOTAL_CRYPTO"],
        description     = "Ethereum / USDT Perpetual Futures",
    ),
}


def get_instrument(symbol: str) -> InstrumentConfig:
    """Return config for a symbol. Raises KeyError if unknown."""
    return INSTRUMENTS[symbol.upper()]


def get_mt5_instruments() -> list[InstrumentConfig]:
    return [i for i in INSTRUMENTS.values() if i.source == "mt5"]


def get_binance_instruments() -> list[InstrumentConfig]:
    return [i for i in INSTRUMENTS.values() if i.source == "binance"]


def pip_value_for_lot(symbol: str, lot_size: float) -> float:
    """Return USD pip value for a given lot size."""
    cfg = get_instrument(symbol)
    return cfg.pip_value_usd * lot_size


def pips_to_price(symbol: str, pips: float) -> float:
    """Convert pip count to price distance."""
    cfg = get_instrument(symbol)
    return pips * cfg.pip_size


def price_to_pips(symbol: str, price_distance: float) -> float:
    """Convert price distance to pip count."""
    cfg = get_instrument(symbol)
    return price_distance / cfg.pip_size if cfg.pip_size else 0.0


ALL_SYMBOLS = list(INSTRUMENTS.keys())
