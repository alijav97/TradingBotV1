"""
intelligence/dxy_correlation.py — DXY Correlation Filter for TradingBotV2
==========================================================================
Fetches the US Dollar Index (DXY) via yfinance and derives a directional
bias for each supported instrument.

DXY correlations (inverse = price moves opposite to DXY):
  XAUUSD  — strong inverse  (rising DXY → gold falls)
  GBPJPY  — moderate inverse (USD strength pressures GBP, supports JPY)
  WTI     — moderate inverse (oil priced in USD; DXY up → oil down)
  NAS100  — weak inverse     (risk-off DXY surge hurts equities)
  BTCUSDT — weak/mixed       (loosely inverse under risk-off)
  ETHUSDT — weak/mixed       (follows BTC)

Usage
-----
    from v2.intelligence.dxy_correlation import get_dxy_bias

    result = get_dxy_bias("XAUUSD")
    # {
    #   "dxy_value": 104.32,
    #   "trend": "rising",          # "rising" | "falling" | "neutral"
    #   "instrument_bias": "bearish",  # "bullish" | "bearish" | "neutral"
    #   "note": "DXY rising — bearish for XAUUSD (inverse correlation)",
    # }
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# yfinance — optional but strongly recommended
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logger.warning("yfinance not installed — DXY data will be unavailable")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DXY_TICKER = "DX-Y.NYB"  # ICE/CME DXY futures on Yahoo Finance

# DXY basket components used for fallback reconstruction
_DXY_BASKET: dict[str, dict[str, Any]] = {
    "EURUSD=X": {"weight": 0.576, "inverse": True},
    "USDJPY=X": {"weight": 0.136, "inverse": False},
    "GBPUSD=X": {"weight": 0.119, "inverse": True},
    "USDCAD=X": {"weight": 0.091, "inverse": False},
    "USDSEK=X": {"weight": 0.042, "inverse": False},
    "USDCHF=X": {"weight": 0.036, "inverse": False},
}

# EMA slope thresholds — calibrated for daily DXY scale
_EMA_SLOPE_UP   =  0.05
_EMA_SLOPE_DOWN = -0.05

# Per-instrument correlation rules
# Keys: instrument symbol (uppercase)
# Values: ("direction", "strength")
#   direction "inverse"  → DXY rising  = bearish for instrument
#   direction "positive" → DXY rising  = bullish for instrument
#   direction "mixed"    → weak/no reliable correlation
_INSTRUMENT_CORRELATIONS: dict[str, dict[str, str]] = {
    "XAUUSD":  {"direction": "inverse",  "strength": "strong"},
    "GBPJPY":  {"direction": "inverse",  "strength": "moderate"},
    "WTI":     {"direction": "inverse",  "strength": "moderate"},
    "NAS100":  {"direction": "inverse",  "strength": "weak"},
    "BTCUSDT": {"direction": "inverse",  "strength": "weak"},
    "ETHUSDT": {"direction": "inverse",  "strength": "weak"},
}


# ===========================================================================
# Internal data-fetching helpers
# ===========================================================================

def _fetch_ohlcv(ticker: str, period: str = "60d", interval: str = "1d") -> pd.DataFrame | None:
    """
    Download OHLCV data for *ticker* via yfinance.

    Returns a normalised DataFrame with lowercase columns
    [open, high, low, close, volume], or None on failure.
    """
    if not _YF_AVAILABLE:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
        if raw is None or raw.empty:
            return None
        # Flatten MultiIndex columns produced by newer yfinance versions
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [str(c[0]).lower() for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]
        raw = raw.rename(columns={"adj close": "close"})
        required = {"open", "high", "low", "close"}
        if not required.issubset(raw.columns):
            return None
        if "volume" not in raw.columns:
            raw["volume"] = 0
        return raw[["open", "high", "low", "close", "volume"]].copy()
    except (ValueError, KeyError, TypeError) as exc:
        logger.debug("_fetch_ohlcv(%s) failed: %s", ticker, exc)
        return None


def _reconstruct_dxy(period: str = "60d", interval: str = "1d") -> pd.DataFrame | None:
    """
    Build a DXY proxy by computing a weighted geometric mean of
    daily returns across the ICE basket components.

    Returns a DataFrame with the same schema as _fetch_ohlcv, or None.
    """
    closes: dict[str, pd.Series] = {}
    for ticker, cfg in _DXY_BASKET.items():
        df = _fetch_ohlcv(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            s = df["close"].copy()
            if cfg["inverse"]:
                s = 1.0 / s
            closes[ticker] = s

    if not closes:
        return None

    combined = pd.DataFrame(closes).dropna()
    if len(combined) < 20:
        return None

    weighted_ret = pd.Series(0.0, index=combined.index)
    total_weight = 0.0
    for ticker, cfg in _DXY_BASKET.items():
        if ticker not in combined.columns:
            continue
        pct = combined[ticker].pct_change()
        weighted_ret += pct * cfg["weight"]
        total_weight += cfg["weight"]

    if total_weight < 0.5:
        return None

    index_series = (1.0 + weighted_ret.fillna(0.0)).cumprod() * 100.0
    index_series.iloc[0] = 100.0

    return pd.DataFrame({
        "open":   index_series.shift(1).bfill(),
        "high":   index_series * 1.001,
        "low":    index_series * 0.999,
        "close":  index_series,
        "volume": 0,
    })


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add ema20 and rsi14 columns to a DXY DataFrame in-place (copy returned)."""
    df = df.copy()
    close = df["close"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100.0 - 100.0 / (1.0 + rs)

    return df


def _determine_trend(df: pd.DataFrame) -> str:
    """
    Derive DXY trend from the last row of an enriched DataFrame.

    Returns "rising", "falling", or "neutral".
    """
    if df is None or df.empty or "ema20" not in df.columns:
        return "neutral"

    close   = df["close"].dropna()
    ema20   = df["ema20"].dropna()
    if len(ema20) < 3 or len(close) < 1:
        return "neutral"

    slope       = float(ema20.iloc[-1] - ema20.iloc[-3])
    last_close  = float(close.iloc[-1])
    last_ema    = float(ema20.iloc[-1])

    if slope > _EMA_SLOPE_UP and last_close > last_ema:
        return "rising"
    if slope < _EMA_SLOPE_DOWN and last_close < last_ema:
        return "falling"
    return "neutral"


# ===========================================================================
# Public API
# ===========================================================================

def get_dxy_bias(symbol: str) -> dict[str, Any]:
    """
    Fetch DXY data and return a correlation bias for *symbol*.

    Fetch order:
      1. Direct DX-Y.NYB ticker (ICE futures via yfinance)
      2. Basket reconstruction from 6 currency pairs
      3. EURUSD inverse approximation (direction proxy only)

    Parameters
    ----------
    symbol : str
        One of: XAUUSD, GBPJPY, WTI, NAS100, BTCUSDT, ETHUSDT.
        Unknown symbols receive a "neutral" bias with a note.

    Returns
    -------
    dict with keys:
        dxy_value        float | None  — latest DXY close
        trend            str           — "rising" | "falling" | "neutral"
        instrument_bias  str           — "bullish" | "bearish" | "neutral"
        note             str           — human-readable explanation
    """
    symbol_upper = symbol.upper()
    corr = _INSTRUMENT_CORRELATIONS.get(symbol_upper)

    df: pd.DataFrame | None = None
    source = "unavailable"

    # --- Attempt 1: direct DXY ticker ---
    raw = _fetch_ohlcv(_DXY_TICKER)
    if raw is not None and len(raw) >= 20:
        df = _enrich(raw)
        source = "direct"
        logger.debug("DXY loaded from direct ticker (%d rows)", len(df))

    # --- Attempt 2: basket reconstruction ---
    if df is None:
        reconstructed = _reconstruct_dxy()
        if reconstructed is not None and len(reconstructed) >= 20:
            df = _enrich(reconstructed)
            source = "basket_reconstruction"
            logger.debug("DXY loaded from basket reconstruction")

    # --- Attempt 3: EURUSD inverse proxy ---
    if df is None:
        eur_raw = _fetch_ohlcv("EURUSD=X")
        if eur_raw is not None and len(eur_raw) >= 20:
            proxy = pd.DataFrame({
                "open":   100.0 / eur_raw["open"],
                "high":   100.0 / eur_raw["low"],
                "low":    100.0 / eur_raw["high"],
                "close":  100.0 / eur_raw["close"],
                "volume": 0,
            }, index=eur_raw.index)
            df = _enrich(proxy)
            source = "eurusd_proxy"
            logger.debug("DXY loaded from EURUSD inverse proxy")

    # --- All sources failed ---
    if df is None or df.empty:
        logger.warning("DXY: all data sources failed for symbol=%s", symbol_upper)
        return {
            "dxy_value":       None,
            "trend":           "neutral",
            "instrument_bias": "neutral",
            "note":            "DXY data unavailable — defaulting to neutral bias",
        }

    trend     = _determine_trend(df)
    dxy_value = round(float(df["close"].iloc[-1]), 3)

    # --- Map trend + correlation to instrument bias ---
    if corr is None:
        bias = "neutral"
        note = (
            f"DXY {trend} ({dxy_value}, source={source}) — "
            f"{symbol_upper} not in correlation map; defaulting to neutral"
        )
    else:
        direction = corr["direction"]
        strength  = corr["strength"]

        if trend == "neutral":
            bias = "neutral"
            note = (
                f"DXY {trend} ({dxy_value}, source={source}) — "
                f"no directional pressure on {symbol_upper}"
            )
        elif direction == "inverse":
            if trend == "rising":
                bias = "bearish"
                note = (
                    f"DXY rising ({dxy_value}, source={source}) — "
                    f"{strength} bearish pressure on {symbol_upper} (inverse correlation)"
                )
            else:  # falling
                bias = "bullish"
                note = (
                    f"DXY falling ({dxy_value}, source={source}) — "
                    f"{strength} bullish pressure on {symbol_upper} (inverse correlation)"
                )
        elif direction == "positive":
            if trend == "rising":
                bias = "bullish"
                note = (
                    f"DXY rising ({dxy_value}, source={source}) — "
                    f"{strength} bullish pressure on {symbol_upper} (positive correlation)"
                )
            else:  # falling
                bias = "bearish"
                note = (
                    f"DXY falling ({dxy_value}, source={source}) — "
                    f"{strength} bearish pressure on {symbol_upper} (positive correlation)"
                )
        else:  # "mixed"
            bias = "neutral"
            note = (
                f"DXY {trend} ({dxy_value}, source={source}) — "
                f"mixed/weak correlation for {symbol_upper}; treating as neutral"
            )

    return {
        "dxy_value":       dxy_value,
        "trend":           trend,
        "instrument_bias": bias,
        "note":            note,
    }
