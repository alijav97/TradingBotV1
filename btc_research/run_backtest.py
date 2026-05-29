"""
btc_research/run_backtest.py — Entry point for the BTC backtest.

Run from project root (C:\\Temp\\TradingBotV1):
    python btc_research/run_backtest.py

Options:
    --refresh          Force re-fetch all data from MT5 (skip cache)
    --verbose          Print every trade open/close/TP1 event
    --score FLOAT      Override MIN_CONFLUENCE_SCORE (default: 3.0)
                       Try --score 2.0 for more trades, --score 4.0 for fewer

Example:
    python btc_research/run_backtest.py --score 2.5 --verbose
"""
from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(str(_ROOT))

import btc_research.settings as _settings
from btc_research.data.fetcher    import fetch_all
from btc_research.backtest.engine import run
from btc_research.backtest.report import print_report
from btc_research.settings        import BTC_SYMBOL, GOLD_SYMBOL, NAS_SYMBOL


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC inter-market confluence backtest")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch from MT5 (ignore cache)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every trade open/close")
    parser.add_argument("--score", type=float, default=None,
                        help="Override MIN_CONFLUENCE_SCORE (default 3.0)")
    args = parser.parse_args()

    # Apply score override before any imports of confluence module
    if args.score is not None:
        _settings.MIN_CONFLUENCE_SCORE = args.score

    print("=" * 65)
    print("BTC INTER-MARKET CONFLUENCE BACKTEST")
    print(f"Symbols   : {BTC_SYMBOL}  |  {GOLD_SYMBOL} (gold)  |  {NAS_SYMBOL} (NAS)")
    print(f"Lookback  : {_settings.LOOKBACK_YEARS} years")
    print(f"Kill-zone : {_settings.KZ_START_UTC}:00-{_settings.KZ_END_UTC}:00 UTC")
    print(f"Min score : {_settings.MIN_CONFLUENCE_SCORE}")
    print(f"Risk/trade: {_settings.RISK_PCT*100:.0f}%  |  "
          f"TP1={_settings.TP1_RR:.0f}R  TP2={_settings.TP2_RR:.0f}R  "
          f"MaxHold={_settings.MAX_HOLD_BARS}h")
    print("=" * 65)
    print()

    # ── Fetch data ─────────────────────────────────────────────────────────────
    data = fetch_all(use_cache=True, force_refresh=args.refresh)

    df_btc  = data.get(BTC_SYMBOL,  __import__("pandas").DataFrame())
    df_gold = data.get(GOLD_SYMBOL, __import__("pandas").DataFrame())
    df_nas  = data.get(NAS_SYMBOL,  __import__("pandas").DataFrame())

    if df_btc.empty:
        print("ERROR: No BTCUSD data. Ensure MT5 is running and connected.")
        sys.exit(1)

    print(f"\nBTC  : {len(df_btc):,} H1 bars")
    print(f"Gold : {len(df_gold):,} H1 bars" if not df_gold.empty else "Gold : unavailable (inter-market factor disabled)")
    print(f"NAS  : {len(df_nas):,} H1 bars"  if not df_nas.empty  else "NAS  : unavailable (inter-market factor disabled)")
    print()

    # ── Run backtest ──────────────────────────────────────────────────────────
    print("Running backtest (this may take ~30-60s for 2 years of data)...")
    results = run(df_btc, df_gold, df_nas, verbose=args.verbose)

    # ── Print report ──────────────────────────────────────────────────────────
    print_report(results)

    # ── Hint for tuning ───────────────────────────────────────────────────────
    n_trades = len(results.get("trades", []))
    if n_trades < 20:
        print(f"\nTIP: Only {n_trades} trades. Try --score {_settings.MIN_CONFLUENCE_SCORE - 0.5:.1f} for more trades.")
    elif n_trades > 300:
        print(f"\nTIP: {n_trades} trades is high. Try --score {_settings.MIN_CONFLUENCE_SCORE + 0.5:.1f} for higher-quality signals.")


if __name__ == "__main__":
    main()
