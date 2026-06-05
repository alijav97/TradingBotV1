"""
btc_research/eth_bot/backtest/adx_sweep.py

ETH-specific ADX analysis — 4 questions answered from the existing backtest data.

Current settings (inherited from BTC Bot 2 — NOT validated for ETH):
  Global min ADX  : 20
  Risk tiers      : ADX ≤ 25 → 3%  |  ADX 25-40 → 2%  |  ADX ≥ 40 → 4%

QUESTION 1 — Is ADX ≥ 20 the right minimum for ETH?
  Sweep min_adx from 20 → 45 in steps of 5.
  Show N, WR, AvgR, PF, total_R for the OK strategies at each cutoff.
  Higher cutoff = fewer but potentially better-quality trades.

QUESTION 2 — Which ADX bucket gives the best WR / AvgR for ETH?
  Split OK trades into ADX buckets: 20-25, 25-30, 30-35, 35-40, 40+.
  If WR rises sharply above ADX 30, the risk-split point should move.

QUESTION 3 — What is the optimal risk-split for ETH?
  Test 4 risk configurations and run the compound simulation for each:
    A) Flat 3% (no split — simplest)
    B) Current BTC:  ADX ≤25→3%  |  25-40→2%  |  ≥40→4%
    C) ETH-tuned A:  ADX ≤30→3%  |  30-45→2%  |  ≥45→4%
    D) ETH-tuned B:  ADX ≤25→2%  |  25-40→3%  |  ≥40→5%  (bet on strength)

QUESTION 4 — Target path: what do we need to reach $60k from $500 in 5 years?
  Show the required CAGR and what combination of trades/month + risk achieves it.
  Run compound simulations at varying risk levels using the best ADX config.

All analyses use OK BTC-aligned trades only:
  rsi_50 [02 UTC] | macd_adx [06 UTC] | rsi_ema [10 UTC]

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.adx_sweep
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

# ── Base parameters ────────────────────────────────────────────────────────────
STARTING_BALANCE = 500.0

OK_SLOTS = [
    ("rsi_50",   2,  True),
    ("macd_adx", 6,  True),
    ("rsi_ema",  10, True),
]

# Risk configurations to test (Question 3)
RISK_CONFIGS = {
    "A  Flat 3% (no split)":        lambda adx: 0.03,
    "B  BTC-style  ≤25→3% 25-40→2% ≥40→4%": lambda adx: (
        0.03 if adx <= 25 else (0.02 if adx < 40 else 0.04)
    ),
    "C  ETH-tune A ≤30→3% 30-45→2% ≥45→4%": lambda adx: (
        0.03 if adx <= 30 else (0.02 if adx < 45 else 0.04)
    ),
    "D  ETH-tune B ≤25→2% 25-40→3% ≥40→5%": lambda adx: (
        0.02 if adx <= 25 else (0.03 if adx < 40 else 0.05)
    ),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bar(char: str = "─", w: int = 72) -> str:
    return char * w


def _load_ok_trades(df: pd.DataFrame, min_adx: float = 20.0) -> pd.DataFrame:
    """Filter backtest CSV to OK BTC-aligned trades at the given minimum ADX."""
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in OK_SLOTS:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    ok = df[mask & (df["adx"] >= min_adx)].sort_values("entry_time").reset_index(drop=True)
    return ok


def _stats(grp: pd.DataFrame) -> dict:
    n   = len(grp)
    if n == 0:
        return {"n": 0, "wr": 0.0, "avg_r": 0.0, "total_r": 0.0, "pf": 0.0}
    wins = (grp["outcome"] == "win").sum()
    rs   = grp["r_achieved"]
    gw   = rs[rs > 0].sum()
    gl   = rs[rs < 0].abs().sum()
    pf   = round(gw / gl, 2) if gl > 0 else float("inf")
    return {
        "n":       n,
        "wr":      round(wins / n * 100, 1),
        "avg_r":   round(rs.mean(), 3),
        "total_r": round(rs.sum(), 3),
        "pf":      pf,
    }


def _compound(trades: pd.DataFrame, risk_fn) -> dict:
    """Run compound simulation with a given risk function ADX→risk_pct."""
    balance = STARTING_BALANCE
    for _, t in trades.iterrows():
        rp  = risk_fn(float(t["adx"]))
        pnl = balance * rp * float(t["r_achieved"])
        balance += pnl
    total_pnl = balance - STARTING_BALANCE
    n_years   = max(
        (trades["entry_time"].max() - trades["entry_time"].min()).days / 365.25, 0.1
    )
    cagr = ((balance / STARTING_BALANCE) ** (1 / n_years) - 1) * 100
    balances  = []
    bal       = STARTING_BALANCE
    for _, t in trades.iterrows():
        rp  = risk_fn(float(t["adx"]))
        bal += bal * rp * float(t["r_achieved"])
        balances.append(bal)
    peak    = pd.Series(balances).cummax()
    max_dd  = ((pd.Series(balances) - peak) / peak * 100).min()
    return {
        "final_bal": round(balance, 2),
        "total_pnl": round(total_pnl, 2),
        "total_ret": round(total_pnl / STARTING_BALANCE * 100, 1),
        "cagr":      round(cagr, 1),
        "max_dd":    round(max_dd, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not TRADES_CSV.exists():
        print(f"ERROR: {TRADES_CSV} not found — run run_backtest.py first")
        sys.exit(1)

    df = pd.read_csv(TRADES_CSV, parse_dates=["entry_time"])
    ok_base = _load_ok_trades(df, min_adx=20.0)

    print()
    print(_bar("═"))
    print("  ETH BOT — ADX SWEEP & OPTIMISATION")
    print(f"  Base OK trades (ADX≥20, BTC-aligned): {len(ok_base)} trades")
    print(f"  Strategies: rsi_50[02] | macd_adx[06] | rsi_ema[10]")
    print(_bar("═"))

    # ══════════════════════════════════════════════════════════════════════════
    #  QUESTION 1 — ADX threshold sweep
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("  Q1. ADX MINIMUM THRESHOLD SWEEP")
    print("      Does filtering out lower-ADX trades improve quality for ETH?")
    print(_bar())
    print(f"  {'MinADX':>7}  {'N':>4}  {'Dropped':>8}  {'WR%':>6}  {'AvgR':>6}  "
          f"{'TotalR':>7}  {'PF':>5}  {'EndBal':>9}  {'CAGR':>7}")
    print(_bar())

    ref_cfg = RISK_CONFIGS["B  BTC-style  ≤25→3% 25-40→2% ≥40→4%"]

    for thresh in [20, 22, 25, 28, 30, 33, 35, 38, 40, 45]:
        ok = _load_ok_trades(df, min_adx=thresh)
        if ok.empty:
            print(f"  {thresh:>7}     0  — no trades remain above this threshold")
            continue
        s   = _stats(ok)
        c   = _compound(ok, ref_cfg)
        dropped = len(ok_base) - s["n"]
        print(f"  {thresh:>7}  {s['n']:>4}  {dropped:>+8}  {s['wr']:>5.1f}%  "
              f"{s['avg_r']:>+6.3f}  {s['total_r']:>+7.3f}  {s['pf']:>5.2f}  "
              f"{c['final_bal']:>9,.0f}  {c['cagr']:>+6.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    #  QUESTION 2 — ADX bucket analysis
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("  Q2. ADX BUCKET ANALYSIS")
    print("      WR and AvgR by ADX range — where does ETH perform best?")
    print(_bar())
    print(f"  {'Bucket':>12}  {'N':>4}  {'WR%':>6}  {'AvgR':>6}  "
          f"{'TotalR':>7}  {'PF':>5}")
    print(_bar())

    buckets = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 50), (50, 200)]
    for lo, hi in buckets:
        grp = ok_base[(ok_base["adx"] >= lo) & (ok_base["adx"] < hi)]
        if grp.empty:
            continue
        s = _stats(grp)
        label = f"ADX {lo}-{hi}" if hi < 200 else f"ADX {lo}+"
        print(f"  {label:>12}  {s['n']:>4}  {s['wr']:>5.1f}%  {s['avg_r']:>+6.3f}  "
              f"{s['total_r']:>+7.3f}  {s['pf']:>5.2f}")

    # ADX percentiles for reference
    print()
    adx_vals = ok_base["adx"]
    print(f"  ADX distribution: "
          f"p25={adx_vals.quantile(0.25):.1f}  "
          f"median={adx_vals.median():.1f}  "
          f"p75={adx_vals.quantile(0.75):.1f}  "
          f"p90={adx_vals.quantile(0.90):.1f}  "
          f"max={adx_vals.max():.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Q2b — Per-strategy ADX bucket
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("  Q2b. ADX BUCKET BY STRATEGY (deeper detail)")
    print(_bar())
    for strat, hour, _ in OK_SLOTS:
        sub = ok_base[(ok_base["strategy"] == strat) & (ok_base["hour_utc"] == hour)]
        if sub.empty:
            continue
        print(f"  {strat} [{hour:02d}:xx]")
        for lo, hi in buckets:
            grp = sub[(sub["adx"] >= lo) & (sub["adx"] < hi)]
            if grp.empty:
                continue
            s = _stats(grp)
            label = f"ADX {lo}-{hi}" if hi < 200 else f"ADX {lo}+"
            print(f"    {label:>12}  N={s['n']:>3}  WR={s['wr']:>5.1f}%  "
                  f"AvgR={s['avg_r']:>+6.3f}  PF={s['pf']:>5.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  QUESTION 3 — Risk split optimisation
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("  Q3. RISK SPLIT OPTIMISATION — compound simulation")
    print("      Which risk configuration grows $500 the most?")
    print(_bar())
    print(f"  {'Config':<46}  {'EndBal':>9}  {'TotalRtn':>9}  {'CAGR':>7}  {'MaxDD':>7}")
    print(_bar())

    best_cfg_name  = ""
    best_cfg_fn    = None
    best_final_bal = 0.0

    for name, fn in RISK_CONFIGS.items():
        c = _compound(ok_base, fn)
        print(f"  {name:<46}  {c['final_bal']:>9,.0f}  {c['total_ret']:>+8.1f}%  "
              f"{c['cagr']:>+6.1f}%  {c['max_dd']:>+6.1f}%")
        if c["final_bal"] > best_final_bal:
            best_final_bal = c["final_bal"]
            best_cfg_name  = name
            best_cfg_fn    = fn

    print(_bar())
    print(f"  Best config: {best_cfg_name.strip()}")

    # ══════════════════════════════════════════════════════════════════════════
    #  QUESTION 4 — Path to $60k in 5 years
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("  Q4. PATH TO $60,000 IN 5 YEARS FROM $500")
    print(_bar())

    target = 60_000.0
    years  = 5.0
    req_mult = target / STARTING_BALANCE
    req_cagr = (req_mult ** (1 / years) - 1) * 100
    print(f"  Target         : ${target:,.0f}")
    print(f"  Starting       : ${STARTING_BALANCE:,.0f}")
    print(f"  Required mult  : {req_mult:.0f}x")
    print(f"  Required CAGR  : {req_cagr:.1f}%")
    print()

    # What the current strategy achieves
    n_years_actual = (ok_base["entry_time"].max() - ok_base["entry_time"].min()).days / 365.25
    c_current = _compound(ok_base, ref_cfg)
    proj_5yr  = STARTING_BALANCE * ((1 + c_current["cagr"]/100) ** years)
    print(f"  CURRENT (3 hours, BTC-aligned, BTC risk split):")
    print(f"    CAGR={c_current['cagr']:+.1f}%  →  5-yr projection: ${proj_5yr:,.0f}")
    print()

    # Simulate "what if we had N× more trades per month (same quality)"
    print(f"  IF TRADE FREQUENCY INCREASES (same WR={_stats(ok_base)['wr']:.1f}%, "
          f"AvgR={_stats(ok_base)['avg_r']:+.3f}R):")
    print(f"  {'Freq mult':>10}  {'Trades/mo':>10}  {'CAGR est':>10}  {'5yr proj':>12}")
    print(f"  {'':>10}  {'':>10}  {'':>10}  {'':>12}")

    trades_per_mo  = len(ok_base) / (n_years_actual * 12)
    base_log_cagr  = np.log(1 + c_current["cagr"] / 100)

    for mult in [1, 2, 3, 4, 5, 6, 8, 10]:
        est_log_cagr = base_log_cagr * mult
        est_cagr     = (np.exp(est_log_cagr) - 1) * 100
        est_5yr      = STARTING_BALANCE * ((1 + est_cagr / 100) ** years)
        flag = "  ← TARGET ✓" if est_5yr >= target else ""
        print(f"  {mult:>10}x  {trades_per_mo * mult:>9.1f}  "
              f"{est_cagr:>+9.1f}%  ${est_5yr:>11,.0f}{flag}")

    print()
    print("  HOW TO INCREASE FREQUENCY WITHOUT HURTING QUALITY:")
    print("    1. Expand KZ hours — check which hours have WR≥42% in backtest summary")
    print("    2. Remove BTC alignment filter — only adds 0.5% WR but cuts ~20% of trades")
    print("    3. Test lower ADX threshold — if ADX 15-20 trades are acceptable quality")
    print("    4. Add second strategy per hour — stack MACD+ADX WITH RSI+EMA at H10 etc")
    print()
    print("  RISK LEVER (higher risk = more $ per trade, same frequency):")
    print(f"  {'Risk/trade':>10}  {'EndBal':>12}  {'CAGR':>8}  {'MaxDD':>8}")
    print()

    for base_risk in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
        flat_fn = lambda adx, r=base_risk: r
        c = _compound(ok_base, flat_fn)
        proj = STARTING_BALANCE * ((1 + c["cagr"] / 100) ** years)
        flag = "  ← TARGET ✓" if proj >= target else ""
        print(f"  {base_risk*100:>9.0f}%  ${c['final_bal']:>11,.0f}  "
              f"{c['cagr']:>+7.1f}%  {c['max_dd']:>+7.1f}%{flag}")

    print()
    print("  NOTE: Higher risk magnifies both gains AND losses.")
    print("  Max drawdown stays manageable if strategies maintain WR≥45%.")
    print("  Recommended approach: increase FREQUENCY first, raise risk second.")
    print(_bar("═"))
    print()


if __name__ == "__main__":
    main()
