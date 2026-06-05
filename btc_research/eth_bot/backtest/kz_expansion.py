"""
btc_research/eth_bot/backtest/kz_expansion.py

Kill-zone expansion analysis — find additional tradeable hours to 4x frequency.

PROBLEM: Current 3 OK hours (2, 6, 10) only generate ~1.3 trades/month.
GOAL   : Find 3-4 more hours with acceptable strategy quality to reach ~5 trades/month.
         At 5/month with Config D risk split, projected 5yr = $60k+ (from $500).

APPROACH:
  1. Read backtest_summary.csv (all 25 strategies × all hours)
  2. Rank by BASE stats (no BTC filter) — BTC alignment only adds 0.5% WR globally
     so we don't want to eliminate whole hours just because of BTC
  3. Find all candidates with: WR≥40%, N≥12, PF≥1.1, AvgR≥0.25
     (relaxed from OK threshold of WR≥45% — we're looking for expansion)
  4. Exclude current 3 OK slots
  5. Show top candidates ranked by score = avg_r × sqrt(n)
  6. Simulate compound growth for best expansion scenario using backtest_trades.csv

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.kz_expansion
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_BACKTEST_DIR = Path(__file__).parent
DATA_DIR      = _BACKTEST_DIR / "data"
TRADES_CSV    = DATA_DIR / "backtest_trades.csv"
SUMMARY_CSV   = DATA_DIR / "backtest_summary.csv"

# ── Parameters ─────────────────────────────────────────────────────────────────
STARTING_BALANCE = 500.0

# Current OK slots (already in the live bot — exclude from expansion candidates)
CURRENT_SLOTS = {(2, "rsi_50"), (6, "macd_adx"), (10, "rsi_ema")}

# Expansion quality thresholds
MIN_N      = 12
MIN_WR     = 0.43    # 43% — slightly relaxed from 45% OK threshold
MIN_AVG_R  = 0.25    # minimum AvgR
MIN_PF     = 1.10    # profit factor ≥ 1.10

# ── MaxDD safety cap ───────────────────────────────────────────────────────────
# A candidate slot is REJECTED if adding it pushes MaxDD below this level.
# -35% means: from the peak balance, the account never loses more than 35%.
# This automatically filters out swing_break (WR=41.5%) and similar low-WR
# high-frequency strategies that cause large losing streaks.
MAX_DD_LIMIT = -35.0

# Config D risk function (best from ADX sweep)
def _risk_fn(adx: float) -> float:
    if adx >= 40:   return 0.05
    elif adx <= 25: return 0.02
    else:           return 0.03

# Session labels for readability
_SESSION = {
    0:  "Asia Early",   1:  "Asia Night",   2:  "Asia Night",
    3:  "Asia Night",   4:  "Asia Morning",  5:  "Asia Morning",
    6:  "EU Pre-Open",  7:  "EU Open",       8:  "EU Open",
    9:  "EU Open",      10: "EU Mid",        11: "EU Mid",
    12: "EU Mid",       13: "NY Pre-Open",   14: "NY Open",
    15: "NY Open",      16: "NY Open",       17: "NY Afternoon",
    18: "NY Afternoon", 19: "NY Afternoon",  20: "NY Late",
    21: "NY Late",      22: "Asia Pre",      23: "Asia Pre",
}


def _bar(char: str = "─", w: int = 76) -> str:
    return char * w


def _compound(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"final_bal": STARTING_BALANCE, "total_ret": 0.0, "cagr": 0.0,
                "max_dd": 0.0, "n": 0}
    balance  = STARTING_BALANCE
    balances = []
    for _, t in trades.iterrows():
        rp      = _risk_fn(float(t["adx"]))
        balance += balance * rp * float(t["r_achieved"])
        balances.append(balance)
    n_years = max(
        (trades["entry_time"].max() - trades["entry_time"].min()).days / 365.25, 0.1
    )
    cagr    = ((balance / STARTING_BALANCE) ** (1 / n_years) - 1) * 100
    peak    = pd.Series(balances).cummax()
    max_dd  = ((pd.Series(balances) - peak) / peak * 100).min()
    return {
        "final_bal": round(balance, 2),
        "total_ret": round((balance - STARTING_BALANCE) / STARTING_BALANCE * 100, 1),
        "cagr":      round(cagr, 1),
        "max_dd":    round(max_dd, 1),
        "n":         len(trades),
    }


def main() -> None:
    for path, label in [(SUMMARY_CSV, "backtest_summary.csv"),
                        (TRADES_CSV,  "backtest_trades.csv")]:
        if not path.exists():
            print(f"ERROR: {label} not found — run run_backtest.py first")
            sys.exit(1)

    summary = pd.read_csv(SUMMARY_CSV)
    trades  = pd.read_csv(TRADES_CSV, parse_dates=["entry_time"])

    print()
    print(_bar("═"))
    print("  ETH BOT — KILL-ZONE EXPANSION ANALYSIS")
    print("  Goal: find 3-4 more hours to reach ~5 trades/month → 4x frequency")
    print("  Using BASE stats (no BTC filter) | Config D risk split")
    print(_bar("═"))

    # ── Current baseline ────────────────────────────────────────────────────────
    base_mask = pd.Series(False, index=trades.index)
    for hour, strat in CURRENT_SLOTS:
        base_mask |= (
            (trades["strategy"] == strat) &
            (trades["hour_utc"] == hour) &
            (trades["btc_aligned"] == True)
        )
    base_trades  = trades[base_mask].sort_values("entry_time").reset_index(drop=True)
    base_result  = _compound(base_trades)
    n_years_base = (base_trades["entry_time"].max() - base_trades["entry_time"].min()).days / 365.25
    base_mo      = len(base_trades) / (n_years_base * 12)

    print()
    print("  CURRENT BASELINE (3 slots, BTC-aligned, Config D risk)")
    print(_bar())
    print(f"  Trades      : {len(base_trades)} over {n_years_base:.1f} yrs "
          f"({base_mo:.1f}/mo)")
    print(f"  CAGR        : {base_result['cagr']:+.1f}%")
    print(f"  5yr proj    : ${STARTING_BALANCE * ((1 + base_result['cagr']/100)**5):,.0f}")
    print(f"  Max DD      : {base_result['max_dd']:+.1f}%")

    # ── Find expansion candidates ───────────────────────────────────────────────
    # Use BASE (non-BTC-aligned) stats from summary
    cands = summary[
        (summary["n_trades"]    >= MIN_N) &
        (summary["win_rate"]    >= MIN_WR) &
        (summary["avg_r"]       >= MIN_AVG_R) &
        (summary["profit_factor"] >= MIN_PF)
    ].copy()

    # Exclude current slots
    cands["_slot"] = list(zip(cands["hour_utc"], cands["strategy"]))
    cands = cands[~cands["_slot"].isin(CURRENT_SLOTS)]

    # Score = avg_r × sqrt(n_trades) — balances quality and frequency
    cands["score"] = cands["avg_r"] * np.sqrt(cands["n_trades"])
    cands = cands.sort_values("score", ascending=False).reset_index(drop=True)

    print()
    print(f"  EXPANSION CANDIDATES (WR≥40%, N≥12, PF≥1.1, AvgR≥0.25)")
    print(f"  Sorted by score = AvgR × √N  (balances quality vs frequency)")
    print(_bar())
    print(f"  {'#':>2}  {'UTC':>4}  {'Session':<14}  {'Strategy':<22}  "
          f"{'N':>4}  {'WR%':>6}  {'AvgR':>6}  {'PF':>5}  {'Score':>6}")
    print(_bar())

    for rank, (_, row) in enumerate(cands.head(20).iterrows(), 1):
        hour    = int(row["hour_utc"])
        strat   = row["strategy"]
        label   = row.get("strategy_label", strat)
        session = _SESSION.get(hour, "")
        wr      = row["win_rate"] * 100
        avg_r   = row["avg_r"]
        pf      = row["profit_factor"]
        n       = int(row["n_trades"])
        score   = row["score"]
        # flag if it also passes the stricter OK threshold
        flag = " ★" if wr >= 45 and avg_r >= 0.40 and pf >= 1.20 else ""
        print(f"  {rank:>2}  {hour:>4}  {session:<14}  {label:<22}  "
              f"{n:>4}  {wr:>5.1f}%  {avg_r:>+6.3f}  {pf:>5.2f}  {score:>6.2f}{flag}")

    print(f"  (★ = also passes strict OK threshold: WR≥45%, AvgR≥0.40, PF≥1.20)")

    # ── Progressive expansion simulation ────────────────────────────────────────
    print()
    print("  PROGRESSIVE EXPANSION — compound simulation")
    print("  Adding top candidates one by one, no BTC alignment required")
    print(_bar())
    print(f"  {'Slots':>5}  {'New slot added':<30}  {'N':>4}  {'Mo/yr':>6}  "
          f"{'CAGR':>7}  {'5yr proj':>10}  {'MaxDD':>7}")
    print(_bar())

    # Baseline
    proj_base = STARTING_BALANCE * ((1 + base_result["cagr"] / 100) ** 5)
    print(f"  {'3 (base)':>8}  {'(current: rsi_50/H2, macd_adx/H6, rsi_ema/H10)':<30}  "
          f"{len(base_trades):>4}  {base_mo:>5.1f}  "
          f"{base_result['cagr']:>+6.1f}%  ${proj_base:>9,.0f}  "
          f"{base_result['max_dd']:>+6.1f}%")

    # Add top candidates progressively
    added_trades  = base_trades.copy()
    n_slots_added = 3
    accepted_slots: list[tuple[int, str]] = []   # (hour, strat) pairs that passed DD check

    for _, row in cands.head(20).iterrows():
        hour  = int(row["hour_utc"])
        strat = row["strategy"]
        label = row.get("strategy_label", strat)

        # Add this slot's trades (NO btc_aligned filter)
        new_t = trades[
            (trades["strategy"]  == strat) &
            (trades["hour_utc"]  == hour)
        ]
        if new_t.empty:
            continue

        # ── MaxDD safety check — test before committing ──────────────────────
        test_trades = pd.concat([added_trades, new_t]).sort_values("entry_time").reset_index(drop=True)
        test_c      = _compound(test_trades)
        slot_str    = f"+{strat}[{hour:02d}]"
        if test_c["max_dd"] < MAX_DD_LIMIT:
            print(f"  SKIP  {slot_str:<30}  MaxDD would be {test_c['max_dd']:+.1f}% "
                  f"(limit {MAX_DD_LIMIT:+.0f}%)")
            continue

        added_trades  = test_trades
        n_slots_added += 1
        accepted_slots.append((hour, strat))

        c      = test_c
        n_yrs  = (added_trades["entry_time"].max() - added_trades["entry_time"].min()).days / 365.25
        mo_rate = len(added_trades) / (n_yrs * 12)
        proj_5 = STARTING_BALANCE * ((1 + c["cagr"] / 100) ** 5)
        flag   = "  ← $60k ✓" if proj_5 >= 60_000 else ""

        print(f"  {n_slots_added:>5}  {slot_str:<30}  {len(added_trades):>4}  "
              f"{mo_rate:>5.1f}  {c['cagr']:>+6.1f}%  ${proj_5:>9,.0f}  "
              f"{c['max_dd']:>+6.1f}%{flag}")

    # ── Best expansion scenario detailed ────────────────────────────────────────
    # Find the scenario that first crosses $60k
    print()
    print("  RECOMMENDED EXPANSION — best slots to add (in priority order)")
    print(_bar())

    added_t2    = base_trades.copy()
    added_slots = [("H02", "rsi_50"), ("H06", "macd_adx"), ("H10", "rsi_ema")]
    target_hit  = False

    for _, row in cands.head(20).iterrows():
        hour    = int(row["hour_utc"])
        strat   = row["strategy"]
        label   = row.get("strategy_label", strat)
        session = _SESSION.get(hour, "")

        new_t = trades[(trades["strategy"] == strat) & (trades["hour_utc"] == hour)]
        if new_t.empty:
            continue

        # ── MaxDD safety check ────────────────────────────────────────────────
        test_t2 = pd.concat([added_t2, new_t]).sort_values("entry_time").reset_index(drop=True)
        c       = _compound(test_t2)
        proj_5  = STARTING_BALANCE * ((1 + c["cagr"] / 100) ** 5)

        if c["max_dd"] < MAX_DD_LIMIT:
            print(f"  SKIP: {strat:<20} at UTC {hour:02d}:xx ({session})")
            print(f"       MaxDD would be {c['max_dd']:+.1f}% — exceeds limit "
                  f"({MAX_DD_LIMIT:+.0f}%) → rejected")
            print()
            continue

        added_t2 = test_t2
        added_slots.append((f"H{hour:02d}", strat))

        print(f"  ADD:  {strat:<20} at UTC {hour:02d}:xx ({session})")
        print(f"       N_new={len(new_t):<4}  WR={row['win_rate']*100:.1f}%  "
              f"AvgR={row['avg_r']:+.3f}  PF={row['profit_factor']:.2f}")
        print(f"       After adding → CAGR={c['cagr']:+.1f}%  "
              f"5yr=${proj_5:,.0f}  MaxDD={c['max_dd']:+.1f}%")
        print()

        if proj_5 >= 60_000 and not target_hit:
            print(f"  ══ $60,000 TARGET REACHED with {len(added_slots)} slots ══")
            target_hit = True
            break

    if not target_hit:
        print("  Note: $60k target not reached from frequency alone.")
        print("  Either combine with slightly higher risk OR accept longer timeline.")

    # ── Updated KZ_HOURS recommendation ─────────────────────────────────────────
    print()
    print("  RECOMMENDED SETTINGS UPDATE")
    print(_bar())
    current_hours = [h for h, _ in CURRENT_SLOTS]
    expansion_hours = [h for h, _ in accepted_slots]
    top_hours = sorted(set(current_hours + expansion_hours))
    print(f"  KZ_HOURS = {top_hours}")
    print(f"  (Only slots that kept MaxDD ≥ {MAX_DD_LIMIT:+.0f}% were included)")
    print(f"  Risk    : ADX≤25→2%  |  ADX 25-40→3%  |  ADX≥40→5%  (Config D)")
    print(f"  rsi_ema : ADX≥25 filter (drop the weak ADX 20-25 trades)")
    print(_bar("═"))
    print()


if __name__ == "__main__":
    main()
