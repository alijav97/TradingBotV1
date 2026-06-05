"""
btc_research/eth_bot/backtest/report.py — Full analysis of ETH backtest results.

Reads the CSV files produced by run_backtest.py and prints:
  1. Overall strategy ranking (base + BTC-aligned)
  2. Per-strategy hour-by-hour breakdown
  3. BTC alignment filter effect (lift in win rate, avg R, profit factor)
  4. Kill-zone recommendations with confidence tiers
  5. Final settings.py update suggestion

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.report

== INPUT ==
  btc_research/eth_bot/backtest/data/backtest_trades.csv
  btc_research/eth_bot/backtest/data/backtest_summary.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    stream = sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BACKTEST_DIR = Path(__file__).parent
DATA_DIR      = _BACKTEST_DIR / "data"
TRADES_CSV    = DATA_DIR / "backtest_trades.csv"
SUMMARY_CSV   = DATA_DIR / "backtest_summary.csv"

# ── Thresholds for "good" kill-zone hours ─────────────────────────────────────
MIN_TRADES          = 15      # minimum trades for statistical validity
MIN_WIN_RATE        = 0.45    # 45% minimum win rate
MIN_AVG_R           = 0.40    # minimum average R per trade
MIN_PROFIT_FACTOR   = 1.20    # minimum profit factor
STRONG_WIN_RATE     = 0.55    # "strong" tier
STRONG_AVG_R        = 0.80
STRONG_PROFIT_FACTOR= 1.60

# Strategies are discovered dynamically from the summary CSV.
# Labels come from the "strategy_label" column written by run_backtest.py.
_FALLBACK_LABELS: dict[str, str] = {}   # populated at load time


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sep(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _hdr(title: str) -> None:
    _sep("═")
    print(f"  {title}")
    _sep("═")


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRADES_CSV.exists() or not SUMMARY_CSV.exists():
        logger.error(
            "Missing data files. Run first:\n"
            "  python -m btc_research.eth_bot.backtest.collect_data\n"
            "  python -m btc_research.eth_bot.backtest.run_backtest"
        )
        sys.exit(1)

    trades  = pd.read_csv(TRADES_CSV, parse_dates=["entry_time", "exit_time"])
    summary = pd.read_csv(SUMMARY_CSV)
    logger.info("Loaded %d trades, %d summary rows", len(trades), len(summary))

    # Build global label map from summary CSV
    global _FALLBACK_LABELS
    if "strategy_label" in summary.columns:
        _FALLBACK_LABELS = dict(zip(summary["strategy"], summary["strategy_label"]))

    return trades, summary


def _label(strat: str) -> str:
    """Return human-readable label for a strategy key."""
    return _FALLBACK_LABELS.get(strat, strat)


def _tier(wr: float, avg_r: float, pf: float, n: int) -> str:
    if n < MIN_TRADES:
        return "LOW_SAMPLE"
    if wr >= STRONG_WIN_RATE and avg_r >= STRONG_AVG_R and pf >= STRONG_PROFIT_FACTOR:
        return "STRONG ✓✓"
    if wr >= MIN_WIN_RATE and avg_r >= MIN_AVG_R and pf >= MIN_PROFIT_FACTOR:
        return "OK ✓"
    return "WEAK ✗"


# ── Section 1: Data overview ───────────────────────────────────────────────────

def _print_overview(trades: pd.DataFrame, summary: pd.DataFrame) -> None:
    _hdr("1. DATA OVERVIEW")

    if trades.empty:
        print("  No trades found in backtest_trades.csv")
        return

    date_min = trades["entry_time"].min()
    date_max = trades["entry_time"].max()
    days = (date_max - date_min).days

    print(f"  Backtest period : {date_min.date()} → {date_max.date()}  ({days} days)")
    print(f"  Total trades    : {len(trades):,}")
    print(f"  With BTC align  : {trades['btc_aligned'].sum():,}  "
          f"({100*trades['btc_aligned'].mean():.1f}% of signals)")
    print(f"  Strategies      : {', '.join(trades['strategy'].unique())}")
    print(f"  Hours covered   : {sorted(trades['hour_utc'].unique())}")
    print()

    wins = trades[trades["outcome"] == "win"]
    losses = trades[trades["outcome"] == "loss"]
    print(f"  Overall win rate: {100*len(wins)/len(trades):.1f}%  "
          f"({len(wins)}W / {len(losses)}L)")
    avg_r = trades["r_achieved"].mean()
    print(f"  Overall avg R   : {avg_r:+.3f}")
    gross_win  = trades.loc[trades["r_achieved"] > 0, "r_achieved"].sum()
    gross_loss = abs(trades.loc[trades["r_achieved"] < 0, "r_achieved"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    print(f"  Profit factor   : {pf:.2f}")
    print()


# ── Section 2: Strategy ranking ────────────────────────────────────────────────

def _print_strategy_ranking(summary: pd.DataFrame) -> None:
    _hdr("2. STRATEGY RANKING  (all hours combined)")

    rows = []
    for strat in sorted(summary["strategy"].unique()):
        sub = summary[summary["strategy"] == strat]
        if sub.empty:
            continue

        # Base (all hours combined)
        n     = sub["n_trades"].sum()
        wins  = (sub["win_rate"] * sub["n_trades"]).sum()
        wr    = wins / n if n > 0 else 0
        avg_r = (sub["avg_r"] * sub["n_trades"]).sum() / n if n > 0 else 0
        gw    = sub.apply(lambda r: r["avg_r"] * r["n_trades"] if r["avg_r"] > 0 else 0, axis=1).sum()
        gl    = sub.apply(lambda r: abs(r["avg_r"]) * r["n_trades"] if r["avg_r"] < 0 else 0, axis=1).sum()
        pf    = gw / gl if gl > 0 else float("inf")

        # BTC-aligned
        nb    = sub["n_btc"].sum()
        winsb = (sub["win_rate_btc"] * sub["n_btc"]).sum()
        wrb   = winsb / nb if nb > 0 else 0
        avg_rb = (sub["avg_r_btc"] * sub["n_btc"]).sum() / nb if nb > 0 else 0
        gwb   = sub.apply(lambda r: r["avg_r_btc"] * r["n_btc"] if r["avg_r_btc"] > 0 else 0, axis=1).sum()
        glb   = sub.apply(lambda r: abs(r["avg_r_btc"]) * r["n_btc"] if r["avg_r_btc"] < 0 else 0, axis=1).sum()
        pfb   = gwb / glb if glb > 0 else float("inf")

        rows.append({
            "strategy": strat,
            "n": n, "wr": wr, "avg_r": avg_r, "pf": pf,
            "nb": nb, "wrb": wrb, "avg_rb": avg_rb, "pfb": pfb,
        })

    rows.sort(key=lambda r: r["avg_rb"] if r["nb"] >= MIN_TRADES else r["avg_r"], reverse=True)

    print(f"  {'Strategy':<22} {'N':>5} {'WR%':>6} {'AvgR':>7} {'PF':>6}  │  "
          f"{'BTC-N':>5} {'BTC-WR%':>8} {'BTC-AvgR':>9} {'BTC-PF':>7}")
    _sep()
    for r in rows:
        label = _label(r["strategy"])
        print(
            f"  {label:<22} {r['n']:>5}  {100*r['wr']:>5.1f}%  {r['avg_r']:>+6.3f}  {r['pf']:>5.2f}  │  "
            f"{r['nb']:>5}  {100*r['wrb']:>6.1f}%  {r['avg_rb']:>+7.3f}  {r['pfb']:>6.2f}"
        )
    print()


# ── Section 3: Per-strategy hour breakdown ─────────────────────────────────────

def _print_hourly(summary: pd.DataFrame, strategy: str) -> None:
    sub = summary[summary["strategy"] == strategy].sort_values("hour_utc")
    if sub.empty:
        print(f"  No data for {strategy}")
        return

    label = _label(strategy)
    print(f"\n  ── {label} ──")
    print(
        f"  {'UTC':>4}  {'N':>5}  {'WR%':>6}  {'AvgR':>7}  {'PF':>6}  {'MaxDD':>7}  {'Tier':<12}  │  "
        f"{'BTC-N':>5}  {'BTC-WR%':>8}  {'BTC-AvgR':>9}  {'BTC-PF':>7}  {'BTC-Tier':<12}"
    )
    _sep()

    good_hours = []
    good_btc_hours = []

    for _, row in sub.iterrows():
        h   = int(row["hour_utc"])
        n   = int(row["n_trades"])
        wr  = row["win_rate"]
        avg = row["avg_r"]
        pf  = row["profit_factor"]
        dd  = row["max_dd_r"]
        t   = _tier(wr, avg, pf, n)

        nb  = int(row["n_btc"])
        wrb = row["win_rate_btc"]
        avgb= row["avg_r_btc"]
        pfb = row["profit_factor_btc"]
        tb  = _tier(wrb, avgb, pfb, nb)

        if "✓" in t:
            good_hours.append(h)
        if "✓" in tb:
            good_btc_hours.append(h)

        print(
            f"  {h:>4}  {n:>5}  {100*wr:>5.1f}%  {avg:>+6.3f}  {pf:>5.2f}  {dd:>+6.2f}R  {t:<12}  │  "
            f"{nb:>5}  {100*wrb:>6.1f}%  {avgb:>+7.3f}  {pfb:>6.2f}  {tb:<12}"
        )

    print()
    print(f"    Base good hours    : {good_hours}")
    print(f"    BTC-aligned good   : {good_btc_hours}")
    return good_hours, good_btc_hours


# ── Section 4: BTC alignment lift analysis ─────────────────────────────────────

def _print_btc_lift(summary: pd.DataFrame) -> None:
    _hdr("4. BTC ALIGNMENT FILTER  —  lift vs base")

    # Only rows with enough BTC-aligned trades
    sub = summary[summary["n_btc"] >= MIN_TRADES].copy()
    if sub.empty:
        print("  Not enough BTC-aligned trades (need ≥ 15 per cell) for lift analysis.")
        print()
        return

    sub["wr_lift"]   = sub["win_rate_btc"]    - sub["win_rate"]
    sub["avgr_lift"] = sub["avg_r_btc"]       - sub["avg_r"]
    sub["pf_lift"]   = sub["profit_factor_btc"] - sub["profit_factor"]

    # Top 10 cells by avg_r lift
    top = sub.nlargest(10, "avgr_lift")[
        ["strategy", "hour_utc", "n_trades", "n_btc",
         "win_rate", "win_rate_btc", "wr_lift",
         "avg_r", "avg_r_btc", "avgr_lift"]
    ]

    print(f"  {'Strategy':<16} {'H':>3}  {'N':>5}  {'BTC-N':>5}  "
          f"{'WR%':>6}  {'BTC-WR%':>8}  {'ΔWR':>6}  "
          f"{'AvgR':>7}  {'BTC-AvgR':>9}  {'ΔAvgR':>7}")
    _sep()
    for _, r in top.iterrows():
        strat = r["strategy"][:14]
        print(
            f"  {strat:<16}  {int(r['hour_utc']):>2}  {int(r['n_trades']):>5}  {int(r['n_btc']):>5}  "
            f"{100*r['win_rate']:>5.1f}%  {100*r['win_rate_btc']:>7.1f}%  {100*r['wr_lift']:>+5.1f}%  "
            f"{r['avg_r']:>+6.3f}  {r['avg_r_btc']:>+7.3f}  {r['avgr_lift']:>+6.3f}"
        )

    # Summary: overall lift
    all_base = summary[(summary["n_trades"] >= MIN_TRADES)]
    all_btc  = summary[(summary["n_btc"]    >= MIN_TRADES)]
    if not all_base.empty and not all_btc.empty:
        avg_wr_base = (all_base["win_rate"] * all_base["n_trades"]).sum() / all_base["n_trades"].sum()
        avg_wr_btc  = (all_btc["win_rate_btc"] * all_btc["n_btc"]).sum() / all_btc["n_btc"].sum()
        avg_r_base  = (all_base["avg_r"] * all_base["n_trades"]).sum() / all_base["n_trades"].sum()
        avg_r_btc   = (all_btc["avg_r_btc"] * all_btc["n_btc"]).sum() / all_btc["n_btc"].sum()
        print()
        print(f"  Global base    : WR {100*avg_wr_base:.1f}%  AvgR {avg_r_base:+.3f}")
        print(f"  Global BTC-flt : WR {100*avg_wr_btc:.1f}%  AvgR {avg_r_btc:+.3f}")
        print(f"  Lift           : WR {100*(avg_wr_btc-avg_wr_base):+.1f}%  "
              f"AvgR {avg_r_btc-avg_r_base:+.3f}")
    print()


# ── Section 5: Kill-zone recommendations ──────────────────────────────────────

def _recommend_kz(summary: pd.DataFrame) -> dict[str, list[int]]:
    _hdr("5. KILL-ZONE RECOMMENDATIONS")

    # Aggregate across strategies: use best strategy per hour
    hour_scores: dict[int, dict] = {}
    for _, row in summary.iterrows():
        h  = int(row["hour_utc"])
        n  = int(row["n_trades"])
        nb = int(row["n_btc"])

        # Score = avg_r × sqrt(n) (reward weighted by sample size)
        score_base = row["avg_r"] * (n ** 0.5) if n >= MIN_TRADES else -99
        score_btc  = row["avg_r_btc"] * (nb ** 0.5) if nb >= MIN_TRADES else -99

        if h not in hour_scores or score_btc > hour_scores[h]["score_btc"]:
            hour_scores[h] = {
                "strategy": row["strategy"],
                "n": n, "wr": row["win_rate"], "avg_r": row["avg_r"], "pf": row["profit_factor"],
                "nb": nb, "wrb": row["win_rate_btc"], "avg_rb": row["avg_r_btc"],
                "pfb": row["profit_factor_btc"],
                "score_btc": score_btc, "score_base": score_base,
            }

    # Sort hours by score
    ranked = sorted(hour_scores.items(), key=lambda kv: kv[1]["score_btc"], reverse=True)

    strong_hours = []
    ok_hours     = []
    weak_hours   = []

    print(f"  {'UTC':>4}  {'Best Strategy':<22}  {'N':>5}  {'WR%':>6}  {'AvgR':>7}  "
          f"{'PF':>5}  │  {'BTC-N':>5}  {'BTC-WR%':>8}  {'BTC-AvgR':>8}  {'Tier'}")
    _sep()

    for h, r in ranked:
        label = _label(r["strategy"])
        t = _tier(r["wrb"], r["avg_rb"], r["pfb"], r["nb"])
        if "STRONG" in t:
            strong_hours.append(h)
        elif "OK" in t:
            ok_hours.append(h)
        else:
            weak_hours.append(h)

        marker = "◀" if "✓" in t else " "
        print(
            f"  {h:>4}  {label:<22}  {r['n']:>5}  {100*r['wr']:>5.1f}%  {r['avg_r']:>+6.3f}  "
            f"{r['pf']:>5.2f}  │  {r['nb']:>5}  {100*r['wrb']:>7.1f}%  {r['avg_rb']:>+7.3f}  "
            f"  {t} {marker}"
        )

    print()
    print(f"  STRONG hours (BTC-aligned): {sorted(strong_hours)}")
    print(f"  OK     hours (BTC-aligned): {sorted(ok_hours)}")
    recommended = sorted(strong_hours + ok_hours)
    if not recommended:
        # Fall back to base stats
        ok_base = [h for h, r in hour_scores.items()
                   if _tier(r["wr"], r["avg_r"], r["pf"], r["n"]) in ("STRONG ✓✓", "OK ✓")]
        recommended = sorted(ok_base)
        print(f"  (No BTC-aligned hours qualify — using base stats)")
    print()

    return {
        "strong": sorted(strong_hours),
        "ok":     sorted(ok_hours),
        "recommended": recommended,
    }


# ── Section 6: Best strategy per recommended hour ──────────────────────────────

def _print_best_strategy(summary: pd.DataFrame, kz_hours: list[int]) -> None:
    _hdr("6. BEST STRATEGY PER KILL-ZONE HOUR")

    if not kz_hours:
        print("  No recommended hours — cannot determine best strategy.")
        print()
        return

    for h in sorted(kz_hours):
        sub = summary[summary["hour_utc"] == h].copy()
        if sub.empty:
            continue

        sub_btc = sub[sub["n_btc"] >= MIN_TRADES].copy()
        if not sub_btc.empty:
            best = sub_btc.loc[sub_btc["avg_r_btc"].idxmax()]
            mode = "BTC-aligned"
            wr   = best["win_rate_btc"]
            avg  = best["avg_r_btc"]
            pf   = best["profit_factor_btc"]
            n    = int(best["n_btc"])
        else:
            best = sub.loc[sub["avg_r"].idxmax()]
            mode = "base"
            wr   = best["win_rate"]
            avg  = best["avg_r"]
            pf   = best["profit_factor"]
            n    = int(best["n_trades"])

        label = _label(best["strategy"])
        print(f"  UTC {h:02d}:xx  →  {label}  [{mode}]  "
              f"N={n}  WR={100*wr:.1f}%  AvgR={avg:+.3f}  PF={pf:.2f}")

    print()


# ── Section 7: Hour distribution (direction bias) ─────────────────────────────

def _print_direction_bias(trades: pd.DataFrame, kz_hours: list[int]) -> None:
    _hdr("7. LONG / SHORT BIAS PER KZ HOUR")

    if trades.empty or not kz_hours:
        print("  No data.")
        print()
        return

    for h in sorted(kz_hours):
        sub = trades[trades["hour_utc"] == h]
        if sub.empty:
            continue
        longs  = sub[sub["direction"] == "long"]
        shorts = sub[sub["direction"] == "short"]
        lw = len(longs[longs["outcome"] == "win"])
        sw = len(shorts[shorts["outcome"] == "win"])
        lr = longs["r_achieved"].mean() if len(longs) else 0
        sr = shorts["r_achieved"].mean() if len(shorts) else 0
        print(f"  UTC {h:02d}:xx  │  "
              f"LONG  {len(longs):>4}  WR {100*lw/len(longs):.0f}%  AvgR {lr:+.2f}  │  "
              f"SHORT {len(shorts):>4}  WR {100*sw/len(shorts):.0f}%  AvgR {sr:+.2f}"
              if len(longs) and len(shorts) else
              f"  UTC {h:02d}:xx  │  LONG {len(longs):>4}  │  SHORT {len(shorts):>4}")
    print()


# ── Section 8: settings.py update suggestion ──────────────────────────────────

def _print_settings_update(rec: dict[str, list[int]]) -> None:
    _hdr("8. SETTINGS.PY UPDATE")

    kz = rec["recommended"]
    if not kz:
        kz = [1, 2, 3, 8]   # keep placeholder
        note = "  ⚠  No hours met quality thresholds — keeping current placeholder."
    else:
        note = (
            f"  Recommended hours based on BTC-aligned quality filters:\n"
            f"    WR ≥ {100*MIN_WIN_RATE:.0f}%  |  AvgR ≥ {MIN_AVG_R:.2f}R  |  "
            f"PF ≥ {MIN_PROFIT_FACTOR:.2f}  |  N ≥ {MIN_TRADES} trades"
        )

    print(note)
    print()
    print("  Update btc_research/eth_bot/settings.py:")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  KZ_HOURS = {kz!r:<40}│")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  Or set environment variable to override without code change:")
    print(f"  ETH_KZ_HOURS={','.join(str(h) for h in kz)}")
    print()
    if rec["strong"]:
        print(f"  STRONG hours only (highest confidence): {rec['strong']}")
    if rec["ok"]:
        print(f"  OK     hours (acceptable, worth including): {rec['ok']}")
    print()


# ── Section 9: Monthly / yearly breakdown ─────────────────────────────────────

def _print_temporal_breakdown(trades: pd.DataFrame) -> None:
    _hdr("9. YEARLY PERFORMANCE BREAKDOWN")

    if trades.empty:
        print("  No trades.")
        print()
        return

    trades = trades.copy()
    trades["year"] = trades["entry_time"].dt.year

    print(f"  {'Year':>5}  {'N':>5}  {'WR%':>6}  {'AvgR':>7}  {'Total R':>8}  {'PF':>6}")
    _sep()

    for yr, grp in trades.groupby("year"):
        n    = len(grp)
        wr   = (grp["outcome"] == "win").mean()
        avg  = grp["r_achieved"].mean()
        tot  = grp["r_achieved"].sum()
        gw   = grp.loc[grp["r_achieved"] > 0, "r_achieved"].sum()
        gl   = abs(grp.loc[grp["r_achieved"] < 0, "r_achieved"].sum())
        pf   = gw / gl if gl > 0 else float("inf")
        print(f"  {yr:>5}  {n:>5}  {100*wr:>5.1f}%  {avg:>+6.3f}  {tot:>+7.1f}R  {pf:>5.2f}")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    _sep("═")
    print("  ETH BACKTEST ANALYSIS REPORT")
    print(f"  Generated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    _sep("═")
    print()

    trades, summary = _load_data()

    # 1. Overview
    _print_overview(trades, summary)

    # 2. Strategy ranking
    _print_strategy_ranking(summary)

    # 3. Per-strategy hourly breakdown (all strategies found in summary)
    _hdr("3. PER-STRATEGY HOURLY BREAKDOWN")
    all_good_hours = []
    for strat in sorted(summary["strategy"].unique()):
        result = _print_hourly(summary, strat)
        if result:
            good, good_btc = result
            all_good_hours.extend(good_btc)

    # 4. BTC alignment lift
    _print_btc_lift(summary)

    # 5. Kill-zone recommendations
    rec = _recommend_kz(summary)

    # 6. Best strategy per KZ hour
    _print_best_strategy(summary, rec["recommended"])

    # 7. Direction bias
    _print_direction_bias(trades, rec["recommended"])

    # 8. Settings suggestion
    _print_settings_update(rec)

    # 9. Yearly breakdown
    _print_temporal_breakdown(trades)

    _sep("═")
    print("  REPORT COMPLETE")
    print()
    print("  Next steps:")
    print("  1. Update KZ_HOURS in btc_research/eth_bot/settings.py (see Section 8)")
    print("  2. Review best strategy per hour (Section 6)")
    print("     → update btc_research/eth_bot/strategy/eth_combined.py accordingly")
    print("  3. Commit settings, start ETH Bot:")
    print("     python -m btc_research.eth_bot.main")
    _sep("═")
    print()


if __name__ == "__main__":
    main()
