"""
backtest/run_backtest.py — CLI entry point for the warm-start backtester.

Run this ONCE before going live to pre-train the ML on historical data.
After this completes, the bot starts with a trained model instead of a blank one.

Usage (from repo root on your VPS):
    cd C:\TradingBotV2
    venv\Scripts\activate
    python -m v2.backtest.run_backtest

Options:
    --days N            How many days of history to replay (default: 180)
    --instruments X Y   Specific instruments only (default: all 6)
    --no-train          Write trades to journal but skip ML training
    --clear             Delete existing backtest trades before running
                        (safe to use if you want a clean re-run)

Examples:
    python -m v2.backtest.run_backtest --days 90
    python -m v2.backtest.run_backtest --instruments XAUUSD BTCUSDT --days 180
    python -m v2.backtest.run_backtest --clear --days 180
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)-7s %(name)s - %(message)s",
    datefmt = "%H:%M:%S",
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="TradingBotV2 warm-start backtester")
    parser.add_argument("--days",        type=int,   default=180,
                        help="Days of H1 history to replay (default 180)")
    parser.add_argument("--instruments", nargs="+",  default=None,
                        help="Instruments to backtest (default: all 6)")
    parser.add_argument("--no-train",   action="store_true",
                        help="Skip ML training after backtest")
    parser.add_argument("--clear",      action="store_true",
                        help="Delete existing backtest trades before running")
    parser.add_argument("--from", dest="start_date", default=None,
                        help="Backtest start date YYYY-MM-DD  (e.g. 2024-10-01). "
                             "Overrides --days lower bound.")
    parser.add_argument("--to",   dest="end_date",   default=None,
                        help="Backtest end date YYYY-MM-DD  (default: today). "
                             "Use with --from to test a specific quarter.")
    args = parser.parse_args()

    # ── Parse date range ──────────────────────────────────────────────────────
    start_date: datetime | None = None
    end_date:   datetime | None = None
    try:
        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end_date:
            end_date   = datetime.strptime(args.end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        elif args.start_date:
            end_date   = datetime.now(timezone.utc)   # default end = today
    except ValueError as exc:
        logger.error("Invalid date format: %s — use YYYY-MM-DD", exc)
        sys.exit(1)

    # ── Load settings + dependencies ──────────────────────────────────────────

    logger.info("Loading settings and connecting...")

    try:
        from v2 import settings
    except ImportError as exc:
        logger.error("Cannot import v2.settings — are you running from C:\\TradingBotV2? %s", exc)
        sys.exit(1)

    from v2.journal.sqlite_journal  import Journal
    from v2.connectors.unified_data import DataFeed
    from v2.backtest.backtester     import Backtester

    journal = Journal()
    feed    = DataFeed()

    # ── Connect data feeds ────────────────────────────────────────────────────

    logger.info("Connecting to MT5 and Binance...")
    status = feed.connect(
        mt5_login       = settings.MT5_LOGIN,
        mt5_password    = settings.MT5_PASSWORD,
        mt5_server      = settings.MT5_SERVER,
        binance_api_key = settings.BINANCE_API_KEY,
        binance_api_secret = settings.BINANCE_API_SECRET,
        binance_testnet = settings.BINANCE_TESTNET,
    )

    mt5_ok     = status.get("mt5", False)
    binance_ok = status.get("binance", False)

    if not mt5_ok and not binance_ok:
        logger.error("Both MT5 and Binance failed to connect. Check .env credentials.")
        sys.exit(1)

    if not mt5_ok:
        logger.warning("MT5 not connected — skipping XAUUSD, GBPJPY, WTI, NAS100")
    if not binance_ok:
        logger.warning("Binance not connected — skipping BTCUSDT, ETHUSDT")

    # Filter instruments to what's actually available
    from v2.instrument_config import ALL_SYMBOLS, get_instrument
    available = []
    requested = args.instruments or list(ALL_SYMBOLS)
    for sym in requested:
        try:
            cfg = get_instrument(sym)
            if cfg.source == "mt5" and mt5_ok:
                available.append(sym)
            elif cfg.source == "binance" and binance_ok:
                available.append(sym)
            else:
                logger.warning("Skipping %s — connector not available", sym)
        except Exception:
            logger.warning("Unknown instrument: %s", sym)

    if not available:
        logger.error("No instruments available to backtest.")
        sys.exit(1)

    logger.info("Instruments to backtest: %s", ", ".join(available))

    # ── Optional: clear existing backtest trades ──────────────────────────────

    if args.clear:
        logger.info("Clearing existing backtest trades from journal...")
        try:
            # Delete ml_features first (child), then trades (parent)
            journal._conn.execute("PRAGMA foreign_keys = OFF")
            journal._conn.execute("DELETE FROM ml_features WHERE trade_id IN (SELECT id FROM trades WHERE notes='backtest')")
            journal._conn.execute("DELETE FROM trades WHERE notes='backtest'")
            journal._conn.execute("PRAGMA foreign_keys = ON")
            journal._conn.commit()
            logger.info("Existing backtest trades removed.")
        except Exception as exc:
            logger.warning("Could not clear backtest trades: %s", exc)

    # ── Run backtest ──────────────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 60)
    logger.info(" BACKTEST STARTING")
    logger.info(" Days of history : %d", args.days)
    logger.info(" Instruments     : %s", ", ".join(available))
    logger.info("=" * 60)
    logger.info("")

    t0 = time.time()

    if start_date:
        logger.info(" Date range      : %s  ->  %s",
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d") if end_date else "today")

    backtester = Backtester(
        journal     = journal,
        feed        = feed,
        days        = args.days,
        instruments = available,
        start_date  = start_date,
        end_date    = end_date,
    )

    summary = backtester.run()

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info(" BACKTEST COMPLETE  (%.0fs)", elapsed)
    logger.info("=" * 60)
    logger.info(" Signals evaluated : %d", summary["signals_evaluated"])
    logger.info(" Trades simulated  : %d", summary["trades_simulated"])
    logger.info(" Win rate          : %.1f%%  (excl. breakevens)", summary["win_rate"] * 100)
    logger.info(" Wins / Losses / BE: %d / %d / %d", summary["wins"], summary["losses"], summary.get("breakevens", 0))
    logger.info("")

    for sym, r in summary["by_instrument"].items():
        if "error" in r:
            logger.info("  %-10s  ERROR: %s", sym, r["error"])
        elif r.get("note"):
            logger.info("  %-10s  NO DATA: %s", sym, r["note"])
        else:
            logger.info(
                "  %-10s  trades=%-4d  WR=%-5.1f%%  wins=%-3d  losses=%-3d  P&L=$%+.2f",
                sym, r["trades_simulated"], r["win_rate"] * 100,
                r["wins"], r["losses"], r.get("pnl_usd", 0),
            )

    logger.info("")
    logger.info("=" * 60)
    logger.info(" COMPOUNDING RESULTS  (starting balance: $%.2f)", summary.get("starting_balance", 0))
    logger.info("=" * 60)
    logger.info("  Ending balance   : $%.2f", summary.get("ending_balance", 0))
    logger.info("  Total P&L        : $%+.2f  (%+.1f%%)", summary.get("total_pnl_usd", 0), summary.get("total_return_pct", 0))
    logger.info("  Peak balance     : $%.2f", summary.get("peak_balance", 0))
    logger.info("  Max drawdown     : %.1f%%", summary.get("max_drawdown_pct", 0))
    logger.info("=" * 60)
    logger.info("")

    n_trades = summary["trades_simulated"]
    if n_trades < 30:
        logger.warning(
            "Only %d trades simulated — ML needs at least 50 for a reliable model. "
            "Try --days 180 or check that MT5/Binance historical data is loaded in terminal.",
            n_trades
        )

    # ── ML training ───────────────────────────────────────────────────────────

    if args.no_train:
        logger.info("--no-train flag set — skipping ML training.")
        logger.info("Run the bot to trigger the nightly retrain, or re-run without --no-train.")
        return

    if n_trades < 30:
        logger.warning("Skipping ML training — not enough trades (%d < 30).", n_trades)
        return

    logger.info("Training ML model on %d backtest trades...", n_trades)
    try:
        from v2.ml.ml_engine import MLEngine
        ml = MLEngine(journal=journal, feed=feed)
        retrain_result = ml.retrain()
        logger.info("")
        logger.info("ML training complete:")
        logger.info("  Samples trained  : %d", retrain_result.get("n_samples", 0))
        logger.info("  Val accuracy     : %.3f", retrain_result.get("val_accuracy", 0))
        logger.info("  Model saved to   : %s", retrain_result.get("model_path", "?"))
        logger.info("")
        logger.info("The bot will now use this pre-trained model from the first trade.")
    except Exception as exc:
        logger.error("ML training failed: %s", exc, exc_info=True)
        logger.info("The bot will fall back to confluence-only scoring until the nightly retrain.")

    # ── Summary recommendation ────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 60)
    logger.info(" READY FOR LIVE TRADING")
    logger.info("=" * 60)
    if n_trades >= 100:
        logger.info(" ML model is pre-trained on %d historical trades.", n_trades)
        logger.info(" Bot confidence level: GOOD — model has meaningful training data.")
    elif n_trades >= 50:
        logger.info(" ML model is pre-trained on %d historical trades.", n_trades)
        logger.info(" Bot confidence level: FAIR — model will improve as live trades accumulate.")
    logger.info(" Start the bot: python -m v2.main")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
