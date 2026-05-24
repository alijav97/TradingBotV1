"""
setup.py - One-Time Setup & Refresh for TradingBotV1

Run modes:
  python setup.py              → full first-time setup (all 4 steps)
  python setup.py --refresh    → weekly data refresh (steps 2-4)
  python setup.py --ingest-only → only read new resources (step 1 only)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

load_dotenv()

BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
DATA_DIR              = os.path.join(BASE_DIR, "data")
RULES_FILE            = os.path.join(DATA_DIR, "rules.json")
BACKTEST_RESULTS_FILE = os.path.join(DATA_DIR, "backtest_results.json")
HIST_CSV              = os.path.join(DATA_DIR, "historical_xauusd.csv")
PATTERNS_FILE         = os.path.join(DATA_DIR, "historical_patterns.json")

from _progress import Spinner, ProgressBar, _bar, _fmt_time, OK, FAIL, SKIP, WARN

os.makedirs(DATA_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Ingest resources
# ═══════════════════════════════════════════════════════════════════════════════

def step_ingest() -> int:
    """Run full ingestion pipeline. Returns total rules in rules.json."""
    print("\n  STEP 1/4 — INGESTING RESOURCES")
    print("  " + "─" * 50)
    t0 = time.time()

    try:
        from ingest import _main_ingest
        total = _main_ingest()
        elapsed = time.time() - t0
        rule_count = 0
        if os.path.exists(RULES_FILE):
            try:
                with open(RULES_FILE, encoding="utf-8") as f:
                    rule_count = len(json.load(f))
            except Exception:
                rule_count = total
        print(f"\n  {OK} Step 1 complete  ({_fmt_time(elapsed)})")
        print(f"     Rules in database: {rule_count}")
        return rule_count
    except Exception as exc:
        print(f"\n  {FAIL} Ingestion failed: {exc}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Download historical data and save to CSV
# ═══════════════════════════════════════════════════════════════════════════════

def step_download_history() -> "pd.DataFrame":
    """Download 2yr XAUUSD H1 data, save to CSV, return DataFrame."""
    print("\n  STEP 2/4 — DOWNLOADING HISTORICAL DATA")
    print("  " + "─" * 50)
    t0 = time.time()

    import pandas as pd

    # Check cache
    if os.path.exists(HIST_CSV):
        try:
            existing = pd.read_csv(HIST_CSV, index_col=0, parse_dates=True)
            last_date = pd.to_datetime(existing.index[-1])
            age_hours = (datetime.now(timezone.utc) - last_date.replace(tzinfo=timezone.utc)).total_seconds() / 3600
            if age_hours < 4:
                print(f"  {OK} Cache fresh ({_fmt_time(age_hours * 3600)} old) — loading from CSV")
                from backtest import get_historical_data
                df = get_historical_data()   # still builds indicators
                return df
        except Exception:
            pass

    sp = Spinner("Downloading 2yr XAUUSD H1 from yfinance...", indent=2).start()
    try:
        from backtest import get_historical_data
        df = get_historical_data()
        sp.stop(success=not df.empty, suffix=f"{len(df):,} candles" if not df.empty else "empty")

        if df.empty:
            print(f"  {FAIL} Download returned empty DataFrame")
            return df

        # Save raw OHLCV to CSV (without indicator columns to keep file light)
        csv_cols = [c for c in ["open", "high", "low", "close", "volume", "datetime"] if c in df.columns]
        df[csv_cols].to_csv(HIST_CSV, index=False)

        elapsed = time.time() - t0
        print(f"  {OK} Saved to {HIST_CSV}  ({_fmt_time(elapsed)})")
        print(f"     Candles: {len(df):,}  |  Date range: "
              f"{df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")
        return df

    except Exception as exc:
        sp.stop(success=False, suffix=str(exc))
        return __import__("pandas").DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Run backtest on all rules
# ═══════════════════════════════════════════════════════════════════════════════

def step_backtest() -> list[dict]:
    """Run backtest engine on all rules. Returns enriched rules list."""
    print("\n  STEP 3/4 — BACKTESTING ALL RULES")
    print("  " + "─" * 50)
    t0 = time.time()

    if not os.path.exists(RULES_FILE):
        print(f"  {SKIP} No rules.json found — skipping backtest")
        return []

    try:
        from backtest import backtest_all_rules, print_backtest_report
        rules = backtest_all_rules()
        elapsed = time.time() - t0

        tested = [r for r in rules if r.get("backtest")]
        tier_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for r in tested:
            t = r.get("tier", "D")
            tier_counts[t] = tier_counts.get(t, 0) + 1

        print(f"\n  {OK} Step 3 complete  ({_fmt_time(elapsed)})")
        print(f"     Rules tested: {len(tested)}")
        print(f"     Tier A (Strong)  : {tier_counts['A']}  (WR ≥ 55% AND PF ≥ 1.2)")
        print(f"     Tier B (Moderate): {tier_counts['B']}  (WR ≥ 45% AND PF ≥ 0.9)")
        print(f"     Tier C (Weak)    : {tier_counts['C']}  (WR ≥ 35% AND PF ≥ 0.7)")
        print(f"     Tier D (Skip)    : {tier_counts['D']}  (below Tier C)")
        return rules
    except Exception as exc:
        print(f"\n  {FAIL} Backtest failed: {exc}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Historical pattern analysis
# ═══════════════════════════════════════════════════════════════════════════════

def step_pattern_analysis(df=None) -> dict:
    """
    Run pattern_matcher on historical data and save results to
    data/historical_patterns.json so morning_briefing.py can load instantly.
    """
    print("\n  STEP 4/4 — ANALYSING HISTORICAL PATTERNS")
    print("  " + "─" * 50)
    t0 = time.time()

    try:
        import pandas as pd
        from pattern_matcher import (
            get_current_snapshot,
            find_similar_historical,
            analyse_outcomes,
        )
        from backtest import get_historical_data

        if df is None or (hasattr(df, "empty") and df.empty):
            sp = Spinner("Loading historical data...", indent=2).start()
            df = get_historical_data()
            sp.stop(success=not df.empty)

        if df.empty:
            print(f"  {FAIL} No historical data for pattern analysis")
            return {}

        sp = Spinner("Capturing current market snapshot...", indent=2).start()
        snapshot = get_current_snapshot(df)
        sp.stop(success=True, suffix=f"price={snapshot['price']}  RSI={snapshot['rsi']}")

        bar = ProgressBar(total=len(df) - 224, label="Scanning candles", indent=2)
        sp2 = Spinner("Finding similar historical setups...", indent=2).start()
        similar = find_similar_historical(snapshot, df, top_n=10)
        sp2.stop(success=True, suffix=f"{len(similar)} matches found")

        sp3 = Spinner("Analysing outcomes...", indent=2).start()
        outcomes = analyse_outcomes(similar, df)
        sp3.stop(success=True)

        # Bundle everything into a single JSON file
        patterns_data = {
            "generated":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "snapshot":     snapshot,
            "similar_count": len(similar),
            "outcomes":     outcomes,
            "top_similar":  [
                {
                    "datetime":   str(m["datetime"]),
                    "similarity": m["similarity"],
                    "rsi":        m["rsi"],
                    "atr_pips":   m["atr_pips"],
                    "close":      m["close"],
                    "session":    m["session"],
                }
                for m in similar
            ],
        }

        with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
            json.dump(patterns_data, f, indent=2, default=str)

        elapsed = time.time() - t0
        verdict = outcomes.get("verdict", "N/A")
        matches = outcomes.get("total_matches", 0)
        print(f"\n  {OK} Step 4 complete  ({_fmt_time(elapsed)})")
        print(f"     Similar patterns found: {matches}")
        print(f"     Historical verdict: {verdict}")
        print(f"     Saved to: {PATTERNS_FILE}")
        return patterns_data

    except ImportError as exc:
        print(f"  {SKIP} pattern_matcher not available: {exc}")
        return {}
    except Exception as exc:
        print(f"\n  {FAIL} Pattern analysis failed: {exc}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Session volatility profiling
# ═══════════════════════════════════════════════════════════════════════════════

def step_session_profiles(df=None) -> dict:
    """Build session volatility profiles from historical data."""
    print("\n  STEP 5/5 — BUILDING SESSION PROFILES")
    print("  " + "─" * 50)
    t0 = time.time()

    try:
        import pandas as pd
        from session_profiler import build_session_profiles
        from backtest import get_historical_data

        if df is None or (hasattr(df, "empty") and df.empty):
            sp = Spinner("Loading historical data...", indent=2).start()
            df = get_historical_data()
            sp.stop(success=not df.empty)

        if df.empty:
            print(f"  {FAIL} No historical data for session profiling")
            return {}

        sp = Spinner("Analysing session volatility...", indent=2).start()
        profiles = build_session_profiles(df)
        sp.stop(success=bool(profiles), suffix=f"{len(profiles)} sessions profiled")

        elapsed = time.time() - t0
        grade_map = {"A": "✦", "B": "●", "C": "○"}
        print(f"\n  {OK} Step 5 complete  ({_fmt_time(elapsed)})")
        print(f"     Session profiles built:")
        for sess in ("London", "Overlap", "NewYork", "Asian", "OffHours"):
            p = profiles.get(sess, {})
            if not p:
                continue
            g     = p.get("session_grade", "B")
            lot_m = p.get("recommended_lot_multiplier", 1.0)
            sl_m  = p.get("recommended_sl_multiplier",  1.0)
            icon  = grade_map.get(g, "○")
            print(f"     {icon} {sess:<10} grade {g} | lot×{lot_m:.1f} | SL×{sl_m:.1f}")
        return profiles

    except ImportError as exc:
        print(f"  {SKIP} session_profiler not available: {exc}")
        return {}
    except Exception as exc:
        print(f"\n  {FAIL} Session profiling failed: {exc}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Setup modes
# ═══════════════════════════════════════════════════════════════════════════════

def _header(title: str) -> None:
    print()
    print("  ╔" + "═" * 42 + "╗")
    print(f"  ║{title.center(42)}║")
    print("  ╚" + "═" * 42 + "╝")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


def run_full_setup() -> None:
    """Full first-time setup: all 5 steps."""
    _header("  TRADING BOT SETUP - FIRST RUN  ")
    total_start = time.time()

    rule_count = step_ingest()
    df         = step_download_history()
    rules      = step_backtest()
    patterns   = step_pattern_analysis(df)
    profiles   = step_session_profiles(df)

    total_elapsed = time.time() - total_start
    tested   = [r for r in rules if r.get("backtest")]
    patterns_n = patterns.get("similar_count", 0) if patterns else 0

    print("\n  " + "═" * 52)
    print("  SETUP COMPLETE")
    print(f"  Total time              : {_fmt_time(total_elapsed)}")
    print(f"  Rules ready             : {rule_count}")
    print(f"  Backtest complete       : {len(tested)} rules tested")
    print(f"  Historical patterns     : {patterns_n} similar setups saved")
    print(f"  Session profiles        : {len(profiles)} sessions profiled")
    print("  " + "═" * 52)
    print(f"  {OK} Bot is ready. Run morning_briefing.py")


def run_refresh() -> None:
    """Weekly data refresh: steps 2–5 only (skip ingest)."""
    _header("  TRADING BOT - WEEKLY REFRESH  ")
    total_start = time.time()

    df       = step_download_history()
    rules    = step_backtest()
    patterns = step_pattern_analysis(df)
    profiles = step_session_profiles(df)

    total_elapsed = time.time() - total_start
    print("\n  " + "═" * 52)
    print("  REFRESH COMPLETE")
    print(f"  Total time: {_fmt_time(total_elapsed)}")
    print(f"  {OK} Data refreshed. Run morning_briefing.py")


def run_ingest_only() -> None:
    """Ingest-only mode: step 1 only."""
    _header("  TRADING BOT - INGEST NEW RESOURCES  ")
    rule_count = step_ingest()
    print(f"\n  {OK} Ingest complete. {rule_count} rules in database.")
    print(f"  Run 'python setup.py --refresh' to backtest new rules.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradingBotV1 Setup")
    parser.add_argument("--refresh",      action="store_true", help="Weekly data refresh (steps 2-4)")
    parser.add_argument("--ingest-only",  action="store_true", help="Only ingest new resources (step 1)")
    args = parser.parse_args()

    if args.refresh:
        run_refresh()
    elif args.ingest_only:
        run_ingest_only()
    else:
        run_full_setup()
