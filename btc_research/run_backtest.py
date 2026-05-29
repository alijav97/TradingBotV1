"""
btc_research/run_backtest.py — Entry point for the BTC backtest.

Run from project root (C:\\Temp\\TradingBotV1):
    python btc_research/run_backtest.py

Options:
    --refresh          Force re-fetch all data from MT5 (skip cache)
    --verbose          Print every trade open/close/TP1 event
    --score FLOAT      Override MIN_CONFLUENCE_SCORE (default: 3.0)
                       Try --score 2.0 for more trades, --score 4.0 for fewer
    --scan-sessions    Run the session scanner FIRST to find the best trading
                       window for BTC (tests all 24 hours + session blocks).
                       Do this before running the full backtest.

Examples:
    # Step 1 — find best session window
    python btc_research/run_backtest.py --scan-sessions

    # Step 2 — run full backtest with optimal session (edit settings.py first)
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
    parser.add_argument("--scan-sessions", action="store_true",
                        help="Run session analysis to find the best trading window")
    parser.add_argument("--scan-combined", action="store_true",
                        help="Run session scan using the REAL Combined strategy classes "
                             "(Volatility+Swing+MorningRange with selective IM filter). "
                             "More accurate than --scan-sessions which uses the old confluence engine.")
    parser.add_argument("--scan-all", action="store_true",
                        help="Run session scanner for EVERY strategy independently. "
                             "Shows which session is optimal for each strategy so we can "
                             "identify incompatible strategies (e.g. Volatility needs US Open "
                             "while Morning Range needs Asia). Takes ~10-15 minutes.")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all 5 strategies head-to-head (with & without IM filter)")
    parser.add_argument("--optimize", choices=["volatility", "morning_range", "both"],
                        default=None,
                        help="Optimize parameters for top strategies to find more trades")
    parser.add_argument("--trades", action="store_true",
                        help="Print every trade for the Combined strategy with balance after "
                             "(trade-by-trade log to verify compounding). Also saves trades.csv.")
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

    # ── Combined strategy session scan ───────────────────────────────────────
    if args.scan_combined:
        print("MODE: Combined Strategy Session Scanner")
        print("Using real strategy classes (Volatility + Swing Level + Morning Range)")
        print("Selective IM filter: only Volatility sub-strategy gets Gold/NAS gate")
        print()
        data = fetch_all(use_cache=True, force_refresh=args.refresh)
        df_btc  = data.get(BTC_SYMBOL,  __import__("pandas").DataFrame())
        df_gold = data.get(GOLD_SYMBOL, __import__("pandas").DataFrame())
        df_nas  = data.get(NAS_SYMBOL,  __import__("pandas").DataFrame())
        if df_btc.empty:
            print("ERROR: No BTCUSD data. Check MT5 connection.")
            sys.exit(1)
        from btc_research.backtest.session_scanner import run_combined_session_scan
        run_combined_session_scan(df_btc, df_gold, df_nas)
        print("\nOnce you see the best session, update KZ_START_UTC / KZ_END_UTC")
        print("in btc_research/settings.py, then re-run: --compare")
        sys.exit(0)

    # ── Per-strategy session scan ─────────────────────────────────────────────
    if args.scan_all:
        print("MODE: Per-Strategy Session Scanner")
        print("Testing EVERY strategy across EVERY session block independently.")
        print("This identifies which strategies are session-compatible and which")
        print("belong in completely different time windows.")
        print("(May take 10-15 minutes — testing 6 strategies × 7 sessions)")
        print()
        data = fetch_all(use_cache=True, force_refresh=args.refresh)
        df_btc  = data.get(BTC_SYMBOL,  __import__("pandas").DataFrame())
        df_gold = data.get(GOLD_SYMBOL, __import__("pandas").DataFrame())
        df_nas  = data.get(NAS_SYMBOL,  __import__("pandas").DataFrame())
        if df_btc.empty:
            print("ERROR: No BTCUSD data. Check MT5 connection.")
            sys.exit(1)
        from btc_research.backtest.session_scanner import run_per_strategy_scan
        from btc_research.backtest.strategy_comparison import ALL_STRATEGIES
        run_per_strategy_scan(df_btc, df_gold, df_nas, ALL_STRATEGIES)
        sys.exit(0)

    # ── Session scanner mode ──────────────────────────────────────────────────
    if args.scan_sessions:
        print("MODE: Session Scanner — testing all 24 hours to find best window")
        print()
        data = fetch_all(use_cache=True, force_refresh=args.refresh)
        df_btc  = data.get(BTC_SYMBOL,  __import__("pandas").DataFrame())
        df_gold = data.get(GOLD_SYMBOL, __import__("pandas").DataFrame())
        df_nas  = data.get(NAS_SYMBOL,  __import__("pandas").DataFrame())
        if df_btc.empty:
            print("ERROR: No BTCUSD data. Check MT5 connection.")
            sys.exit(1)
        from btc_research.backtest.session_scanner import run_session_scan
        run_session_scan(df_btc, df_gold, df_nas,
                         min_score=_settings.MIN_CONFLUENCE_SCORE)
        print("\nOnce you see the best session, update KZ_START_UTC / KZ_END_UTC")
        print("in btc_research/settings.py, then run the full backtest.")
        sys.exit(0)

    # ── Optimizer mode ────────────────────────────────────────────────────────
    if args.optimize:
        print(f"MODE: Parameter Optimizer — {args.optimize}")
        print("Testing ATR multipliers × close zones × session windows...")
        print("This may take 3-5 minutes.\n")
        data = fetch_all(use_cache=True, force_refresh=args.refresh)
        df_btc  = data.get(BTC_SYMBOL,  __import__("pandas").DataFrame())
        df_gold = data.get(GOLD_SYMBOL, __import__("pandas").DataFrame())
        df_nas  = data.get(NAS_SYMBOL,  __import__("pandas").DataFrame())
        if df_btc.empty:
            print("ERROR: No BTCUSD data.")
            sys.exit(1)
        from btc_research.backtest.optimizer import run_optimizer
        run_optimizer(df_btc, df_gold, df_nas, strategy=args.optimize)
        sys.exit(0)

    # ── Trade-by-trade log ────────────────────────────────────────────────────
    if args.trades:
        print("MODE: Trade-by-trade log — Combined Strategy")
        print(f"Starting balance : ${_settings.STARTING_BALANCE:,.2f}")
        print(f"Risk per trade   : {_settings.RISK_PCT*100:.1f}% (compounding)")
        print(f"Session          : {_settings.KZ_START_UTC}:00-{_settings.KZ_END_UTC}:00 UTC")
        print()
        data = fetch_all(use_cache=True, force_refresh=args.refresh)
        df_btc  = data.get(BTC_SYMBOL,  __import__("pandas").DataFrame())
        df_gold = data.get(GOLD_SYMBOL, __import__("pandas").DataFrame())
        df_nas  = data.get(NAS_SYMBOL,  __import__("pandas").DataFrame())
        if df_btc.empty:
            print("ERROR: No BTCUSD data.")
            sys.exit(1)
        from btc_research.backtest.strategy_comparison import _simulate, _stats
        from btc_research.strategies.combined import CombinedStrategy
        strat  = CombinedStrategy(atr_multiplier=1.2, close_zone=0.45, range_bars=6)
        trades = _simulate(strat, df_btc, df_gold, df_nas, use_intermarket=False)
        stats  = _stats(trades)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = (f"{'#':>4}  {'Date':>12}  {'Dir':>5}  {'Entry':>9}  "
               f"{'SL':>9}  {'TP1':>9}  {'TP2':>9}  "
               f"{'Lots':>8}  {'Risk$':>7}  {'Sub-Strategy':>22}  "
               f"{'Result':>12}  {'PnL $':>9}  {'R':>6}  {'Balance':>10}")
        print(hdr)
        print("-" * len(hdr))

        running_bal = _settings.STARTING_BALANCE
        for idx, t in enumerate(trades, 1):
            sl_dist    = abs(t["entry"] - t["sl"])
            risk_usd   = round(running_bal * _settings.RISK_PCT, 2)
            tp1_price  = (t["entry"] + t.get("tp1_rr", 2.0) * sl_dist
                          if t["direction"] == "long"
                          else t["entry"] - t.get("tp1_rr", 2.0) * sl_dist)
            tp2_price  = (t["entry"] + t.get("tp2_rr", 5.0) * sl_dist
                          if t["direction"] == "long"
                          else t["entry"] - t.get("tp2_rr", 5.0) * sl_dist)
            sub        = t.get("signal_reason", "")[:22]
            date_str   = str(t["open_time"])[:10]
            running_bal = t["balance_after"]
            print(
                f"{idx:>4}  {date_str:>12}  {t['direction'].upper():>5}  "
                f"{t['entry']:>9,.2f}  {t['sl']:>9,.2f}  "
                f"{tp1_price:>9,.2f}  {tp2_price:>9,.2f}  "
                f"{t['lots']:>8.5f}  ${risk_usd:>6,.2f}  "
                f"{sub:>22}  "
                f"{t['exit_reason']:>12}  "
                f"${t['pnl_usd']:>+8,.2f}  {t['r_multiple']:>+5.2f}R  "
                f"${t['balance_after']:>9,.2f}"
            )

        print("-" * len(hdr))
        print(f"\nSUMMARY: {stats['trades']} trades | "
              f"WR={stats['wr']}% | Avg R={stats['avg_r']:+.2f} | "
              f"Total PnL=${stats['total_pnl']:+,.2f} | MaxDD={stats['max_dd']}%")
        print(f"Final balance: ${trades[-1]['balance_after']:,.2f}  "
              f"(started ${_settings.STARTING_BALANCE:,.2f})")

        # ── Save to CSV ───────────────────────────────────────────────────────
        import pandas as _pd
        csv_path = __import__("pathlib").Path("btc_research/data/trades_combined.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _pd.DataFrame(trades).to_csv(csv_path, index=False)
        print(f"\nFull trade log saved → {csv_path}")
        sys.exit(0)

    # ── Strategy comparison mode ──────────────────────────────────────────────
    if args.compare:
        print("MODE: Strategy Comparison — testing all 5 strategies head-to-head")
        print()
        data = fetch_all(use_cache=True, force_refresh=args.refresh)
        df_btc  = data.get(BTC_SYMBOL,  __import__("pandas").DataFrame())
        df_gold = data.get(GOLD_SYMBOL, __import__("pandas").DataFrame())
        df_nas  = data.get(NAS_SYMBOL,  __import__("pandas").DataFrame())
        if df_btc.empty:
            print("ERROR: No BTCUSD data.")
            sys.exit(1)
        from btc_research.backtest.strategy_comparison import run_comparison
        run_comparison(df_btc, df_gold, df_nas)
        sys.exit(0)

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
