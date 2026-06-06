"""
btc_research/eth_bot/backtest/collect_data.py — Fetch ETH + BTC H1 history from MT5.

Run this ON THE VPS (where MT5 is running) to collect historical data.
Saves H1 (existing) plus M15 (for finer BTC->ETH entry-timing research):
  btc_research/eth_bot/backtest/data/ETHUSD_H1.csv
  btc_research/eth_bot/backtest/data/BTCUSD_H1.csv
  btc_research/eth_bot/backtest/data/ETHUSD_M15.csv
  btc_research/eth_bot/backtest/data/BTCUSD_M15.csv
(Existing H1 files are loaded from cache; only the missing M15 files fetch.)

BTC data is collected alongside ETH because BTC price action directly
drives ETH movement — we'll use BTC trend as a confluence filter in the
backtest to see if it improves ETH signal quality.

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.collect_data

== OUTPUT ==
  Prints a summary: date range, bar count, any gaps found.
  CSV columns: time (UTC), open, high, low, close, volume
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    stream = sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BACKTEST_DIR = Path(__file__).parent
DATA_DIR      = _BACKTEST_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ETH_CSV = DATA_DIR / "ETHUSD_H1.csv"
BTC_CSV = DATA_DIR / "BTCUSD_H1.csv"

# M15 files for the finer BTC->ETH lead-lag / entry-timing research
ETH_M15_CSV = DATA_DIR / "ETHUSD_M15.csv"
BTC_M15_CSV = DATA_DIR / "BTCUSD_M15.csv"
M15_BAR_COUNT = 200_000   # ~5.7y of M15 (96 bars/day); MT5 trims to what it has

# ── How many bars to fetch ─────────────────────────────────────────────────────
# H1 bars:  1 year  ≈  8,760 bars
#           2 years ≈ 17,520 bars
#           3 years ≈ 26,280 bars
# Pepperstone MT5 typically has 3-5 years of H1 history available.
BAR_COUNT = 50_000   # fetch as much as possible, trim what MT5 returns


def _fetch_from_mt5(symbol: str, count: int, timeframe: str = "H1") -> pd.DataFrame:
    """Connect to MT5, fetch OHLCV, return DataFrame with true UTC timestamps."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("MetaTrader5 package not installed — run on VPS venv")
        sys.exit(1)

    # Load credentials from .env
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parents[4] / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Loaded .env from %s", env_path)
    except ImportError:
        pass

    login    = int(os.environ.get("MT5_LOGIN",    "0") or "0")
    password = os.environ.get("MT5_PASSWORD", "")
    server   = os.environ.get("MT5_SERVER",   "")

    logger.info("Initialising MT5...")
    if not mt5.initialize():
        logger.error("MT5 initialize() failed: %s", mt5.last_error())
        sys.exit(1)

    if login:
        if not mt5.login(login, password=password, server=server):
            logger.error("MT5 login failed: %s", mt5.last_error())
            mt5.shutdown()
            sys.exit(1)
        info = mt5.account_info()
        logger.info("MT5 connected — account %s", info.login if info else "?")

    _TF_MAP = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
               "H1": mt5.TIMEFRAME_H1}
    tf = _TF_MAP.get(timeframe, mt5.TIMEFRAME_H1)
    mt5.symbol_select(symbol, True)
    logger.info("Fetching %s %s — requesting %d bars...", symbol, timeframe, count)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

    if rates is None or len(rates) == 0:
        logger.error("No data returned for %s: %s", symbol, mt5.last_error())
        mt5.shutdown()
        return pd.DataFrame()

    df = pd.DataFrame(rates)

    # UTC correction: Pepperstone server = UTC+3
    SERVER_UTC_OFFSET = 3
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time"] = df["time"] - pd.Timedelta(hours=SERVER_UTC_OFFSET)

    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").reset_index(drop=True)

    mt5.shutdown()
    return df


def _check_gaps(df: pd.DataFrame, symbol: str) -> None:
    """Log any gaps > 2 hours in the data (weekends excluded)."""
    df["time"] = pd.to_datetime(df["time"], utc=True)
    diffs = df["time"].diff().dropna()
    # Gaps > 2h that aren't weekend (Fri close → Mon open = ~60h)
    big_gaps = diffs[(diffs > pd.Timedelta(hours=2)) & (diffs < pd.Timedelta(hours=58))]
    if big_gaps.empty:
        logger.info("%s — no unexpected gaps found ✓", symbol)
    else:
        logger.warning("%s — %d gaps > 2h found:", symbol, len(big_gaps))
        for idx, gap in big_gaps.items():
            logger.warning("  Gap %.1fh at %s", gap.total_seconds() / 3600,
                           df["time"].iloc[idx])


def collect(symbol: str, out_path: Path, timeframe: str = "H1",
            count: int = BAR_COUNT) -> pd.DataFrame:
    """Fetch, validate, and save data for one symbol/timeframe."""
    if out_path.exists():
        logger.info("%s already exists — loading cached file", out_path.name)
        df = pd.read_csv(out_path, parse_dates=["time"])
        logger.info("  Loaded %d bars: %s → %s",
                    len(df), df["time"].iloc[0], df["time"].iloc[-1])
        return df

    df = _fetch_from_mt5(symbol, count, timeframe)
    if df.empty:
        logger.error("Failed to fetch %s", symbol)
        return df

    logger.info(
        "%s: fetched %d bars  |  %s → %s",
        symbol, len(df),
        df["time"].iloc[0].strftime("%Y-%m-%d"),
        df["time"].iloc[-1].strftime("%Y-%m-%d"),
    )

    _check_gaps(df, symbol)

    df.to_csv(out_path, index=False)
    logger.info("Saved → %s", out_path)
    return df


def main() -> None:
    logger.info("=" * 55)
    logger.info("ETH Backtest Data Collection")
    logger.info("=" * 55)

    eth_df = collect("ETHUSD", ETH_CSV)
    btc_df = collect("BTCUSD", BTC_CSV)

    # M15 candles for finer entry-timing / lead-lag research
    logger.info("")
    logger.info("Fetching M15 candles (BTC->ETH entry-timing research)...")
    collect("ETHUSD", ETH_M15_CSV, "M15", M15_BAR_COUNT)
    collect("BTCUSD", BTC_M15_CSV, "M15", M15_BAR_COUNT)

    if eth_df.empty or btc_df.empty:
        logger.error("Data collection failed — check MT5 connection")
        sys.exit(1)

    # Alignment check
    eth_df["time"] = pd.to_datetime(eth_df["time"], utc=True)
    btc_df["time"] = pd.to_datetime(btc_df["time"], utc=True)

    eth_start = eth_df["time"].iloc[0]
    btc_start = btc_df["time"].iloc[0]
    eth_end   = eth_df["time"].iloc[-1]
    btc_end   = btc_df["time"].iloc[-1]

    logger.info("")
    logger.info("Summary:")
    logger.info("  ETHUSD: %d bars  |  %s → %s", len(eth_df),
                eth_start.strftime("%Y-%m-%d"), eth_end.strftime("%Y-%m-%d"))
    logger.info("  BTCUSD: %d bars  |  %s → %s", len(btc_df),
                btc_start.strftime("%Y-%m-%d"), btc_end.strftime("%Y-%m-%d"))

    overlap_start = max(eth_start, btc_start)
    overlap_end   = min(eth_end, btc_end)
    overlap_days  = (overlap_end - overlap_start).days
    logger.info("  Overlap: %d days  |  %s → %s",
                overlap_days,
                overlap_start.strftime("%Y-%m-%d"),
                overlap_end.strftime("%Y-%m-%d"))
    logger.info("")
    logger.info("Ready for backtest — run:")
    logger.info("  python -m btc_research.eth_bot.backtest.run_backtest")


if __name__ == "__main__":
    main()
