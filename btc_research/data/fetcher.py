"""
btc_research/data/fetcher.py — Pull BTCUSD, XAUUSD, NAS100 H1 data from MT5.

- Uses v2/connectors/mt5_connector.py which already applies the UTC+3 server
  offset fix, so all returned timestamps are true UTC.
- Caches each symbol to CSV so the backtest can re-run without re-fetching.
  Cache refreshes automatically after 24 hours.
- MT5 must be running on the VPS with an active Pepperstone connection.

Usage:
    from btc_research.data.fetcher import fetch_all
    data = fetch_all()  # {"BTCUSD": df, "XAUUSD": df, "NAS100": df}
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Allow importing v2/ modules from project root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from btc_research.settings import (
    BTC_SYMBOL, GOLD_SYMBOL, NAS_SYMBOL,
    CACHE_DIR, LOOKBACK_YEARS,
)

# H1 bars for LOOKBACK_YEARS: years * 365 days * 24 hours + 30% buffer for weekends/gaps
_H1_BARS = int(LOOKBACK_YEARS * 365 * 24 * 1.35)

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(symbol: str, timeframe: str) -> Path:
    return CACHE_DIR / f"{symbol}_{timeframe}.csv"


def _cache_is_fresh(path: Path, max_age_hours: float = 24.0) -> bool:
    if not path.exists():
        return False
    age_h = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    return age_h < max_age_hours


def fetch_symbol(
    symbol:        str,
    timeframe:     str   = "H1",
    bars:          int   = _H1_BARS,
    use_cache:     bool  = True,
    force_refresh: bool  = False,
) -> pd.DataFrame:
    """
    Return an OHLCV DataFrame for `symbol` at `timeframe`.

    Timestamps are UTC-normalised (MT5 server offset already removed by the
    v2 connector).  DataFrame columns: time, open, high, low, close, volume.
    """
    cache_file = _cache_path(symbol, timeframe)

    # ── Serve from cache if fresh ────────────────────────────────────────────
    if use_cache and not force_refresh and _cache_is_fresh(cache_file):
        df = pd.read_csv(cache_file)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        print(f"  [cache] {symbol:8s} {timeframe}: {len(df):,} bars  "
              f"({df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()})")
        return df

    # ── Live fetch from MT5 ───────────────────────────────────────────────────
    try:
        from v2.connectors.mt5_connector import connect, get_ohlcv
        from v2.settings import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    except ImportError as exc:
        raise RuntimeError(f"Cannot import v2 modules: {exc}") from exc

    connected = connect(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not connected:
        # Fall back to stale cache rather than crashing
        if cache_file.exists():
            print(f"  [stale cache] {symbol}: MT5 unavailable, using old cache")
            df = pd.read_csv(cache_file)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df
        raise RuntimeError(f"MT5 connection failed and no cache exists for {symbol}")

    df = get_ohlcv(symbol, timeframe, bars)
    if df.empty:
        raise RuntimeError(f"MT5 returned no data for {symbol} {timeframe}")

    # Ensure UTC-aware timestamps
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)

    # Trim to lookback window
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=LOOKBACK_YEARS)
    df = df[df["time"] >= cutoff].reset_index(drop=True)

    # Save to cache
    df.to_csv(cache_file, index=False)
    print(f"  [MT5]   {symbol:8s} {timeframe}: {len(df):,} bars  "
          f"({df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()})")

    return df


def fetch_all(
    use_cache:     bool = True,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch H1 data for all three research assets.

    Returns:
        {
          "BTCUSD": DataFrame,
          "XAUUSD": DataFrame,
          "NAS100": DataFrame,
        }
    Symbols that fail are returned as empty DataFrames (non-fatal).
    """
    print("Fetching market data...")
    result: dict[str, pd.DataFrame] = {}
    for sym in [BTC_SYMBOL, GOLD_SYMBOL, NAS_SYMBOL]:
        try:
            result[sym] = fetch_symbol(
                sym, "H1",
                use_cache=use_cache,
                force_refresh=force_refresh,
            )
        except Exception as exc:
            print(f"  WARNING: Could not fetch {sym}: {exc}")
            result[sym] = pd.DataFrame()
    return result
