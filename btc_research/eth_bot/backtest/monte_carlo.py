"""
btc_research/eth_bot/backtest/monte_carlo.py — risk-vs-ruin Monte Carlo.

Answers the real question behind "$500 -> $10,000 in 6 months": what risk-per-trade
would it take, and what is the probability of blowing the account up trying?

Method (honest, data-driven):
  * Pool of outcomes = the realised per-signal R of every pure-S4 trade in history
    (r_achieved, which already bakes in the 50/50 TP1 scale-out). This is the
    true win-rate / R-magnitude distribution the strategy produces.
  * Trades per 6 months = derived from the one-position sim (how many the bot can
    actually take, ~5/month).
  * For each fixed risk-per-trade level, run many bootstrap paths from $500:
        balance *= (1 + risk * R)     R drawn with replacement from the pool
  * Report: median final, P(final >= $10k), P(peak >= $10k), P(ruin <= $50 = -90%),
    P(peak >= $20k). 4% ~= the current live Tier B average.

Caveat: bootstrap draws are independent (it still produces loss streaks naturally,
but ignores any serial correlation). It is meant to show the SHAPE of the
risk/ruin tradeoff, not a promise of any single number.

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.monte_carlo
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DIR     = Path(__file__).parent
DATA_DIR = _DIR / "data"
CSV      = DATA_DIR / "backtest_trades.csv"

START        = 500.0
TARGET       = 10_000.0
TARGET2      = 20_000.0
RUIN_LEVEL   = 50.0          # -90% = effectively dead
HORIZON_MONTHS = 6
N_PATHS      = 50_000
RISK_GRID    = [0.02, 0.04, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30]
SEED         = 7

# -- Horizon sweep: "how long does $10k actually take at a SURVIVABLE risk?" ----
# Same bootstrap method, but hold risk safe and let TIME be the lever.
HORIZON_GRID = [6, 12, 18, 24, 36]      # months
SAFE_RISKS   = [0.04, 0.06]             # the near-zero-ruin live settings

S4_SLOTS = [
    ("rsi_50",    2,  True), ("macd_adx",  6,  True), ("rsi_ema", 10, True),
    ("ema_cross", 5,  False), ("rsi_50",    7,  False), ("keltner", 14, False),
    ("ema_cross",14,  False), ("keltner",  15,  False),
]

# one-position sizing (only used to derive trades/month, not the MC risk sweep)
RISK_EARLY, RISK_TRANS, RISK_STRONG = 0.030, 0.040, 0.060


def _bar(c: str = "-", w: int = 90) -> str:
    return c * w


def _select(df: pd.DataFrame, slots) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in slots:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return df[mask].sort_values("entry_time").reset_index(drop=True)


def _taken_per_month(cand: pd.DataFrame) -> float:
    """Count one-position taken trades / months spanned (no risk overlays needed)."""
    last_exit = pd.Timestamp.min
    n = 0
    for _, t in cand.iterrows():
        if t["entry_time"] < last_exit:
            continue
        last_exit = t["exit_time"]
        n += 1
    span_months = max(
        (cand["entry_time"].max() - cand["entry_time"].min()).days / 30.44, 1.0)
    return n / span_months


def main() -> None:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first"); sys.exit(1)
    df = pd.read_csv(CSV, parse_dates=["entry_time", "exit_time"])
    for c in ("entry_time", "exit_time"):
        if df[c].dt.tz is not None:
            df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)

    cand = _select(df, S4_SLOTS)
    pool = cand["r_achieved"].astype(float).to_numpy()
    if pool.size == 0:
        print("No trades in pool."); return

    tpm = _taken_per_month(cand)
    n_trades = max(int(round(tpm * HORIZON_MONTHS)), 1)
    wr = (pool > 0).mean() * 100
    avg_r = pool.mean()

    rng = np.random.default_rng(SEED)

    print()
    print(_bar("="))
    print("  ETH BOT — RISK vs RUIN MONTE CARLO")
    print(f"  $500 start | {HORIZON_MONTHS}-month horizon | {n_trades} trades "
          f"(~{tpm:.1f}/month, one-position) | {N_PATHS:,} paths")
    print(f"  Outcome pool: {pool.size} real pure-S4 signals | "
          f"WR {wr:.1f}% | avg {avg_r:+.3f}R | best {pool.max():+.1f}R | worst {pool.min():+.1f}R")
    print(_bar("="))
    print(f"  {'risk/trade':>10} {'medianFinal':>12} {'p10':>9} {'p90':>11} "
          f"{'P(>=$10k)':>10} {'P(peak10k)':>11} {'P(>=$20k pk)':>13} {'P(RUIN)':>9}")
    print(_bar())

    for risk in RISK_GRID:
        draws = rng.choice(pool, size=(N_PATHS, n_trades), replace=True)
        mult  = 1.0 + risk * draws
        mult  = np.clip(mult, 1e-6, None)           # guard (R>=-1, risk<1 => always >0)
        paths = START * np.cumprod(mult, axis=1)
        final = paths[:, -1]
        peak  = paths.max(axis=1)
        low   = paths.min(axis=1)

        p_final10 = (final >= TARGET).mean() * 100
        p_peak10  = (peak  >= TARGET).mean() * 100
        p_peak20  = (peak  >= TARGET2).mean() * 100
        p_ruin    = (low   <= RUIN_LEVEL).mean() * 100
        med   = np.median(final)
        p10   = np.percentile(final, 10)
        p90   = np.percentile(final, 90)

        tag = "  <- current live (~Tier B avg)" if abs(risk - 0.04) < 1e-9 else ""
        print(f"  {risk*100:>9.0f}% {med:>12,.0f} {p10:>9,.0f} {p90:>11,.0f} "
              f"{p_final10:>9.1f}% {p_peak10:>10.1f}% {p_peak20:>12.1f}% {p_ruin:>8.1f}%{tag}")

    print(_bar())
    print("  P(>=$10k)   = chance you FINISH the 6 months at/above $10k")
    print("  P(peak10k)  = chance you ever TOUCH $10k (may give it back)")
    print("  P(RUIN)     = chance the account ever falls to <= $50 (-90%, effectively dead)")
    print()
    print("  Read the tradeoff: pushing risk up raises P(touch $10k) but raises P(RUIN)")
    print("  faster. The 'finish >=$10k' column is the honest one — peaks you give back")
    print("  don't pay you. There is no risk level that makes 20x in 6mo likely AND safe.")
    print(_bar("="))

    _horizon_sweep(pool, tpm, rng)
    print()


def _horizon_sweep(pool: np.ndarray, tpm: float, rng) -> None:
    """At SURVIVABLE risk, let TIME be the lever. For each (risk, horizon) report
    median balance and P(>=$10k) — shows when $10k actually becomes likely if you
    stop forcing it into 6 months."""
    print()
    print(_bar("="))
    print("  TIME IS THE LEVER — same edge, survivable risk, longer horizons")
    print(f"  $500 start | {N_PATHS:,} paths | ~{tpm:.1f} trades/month (one-position)")
    print(_bar("="))
    print(f"  {'risk':>5} {'horizon':>9} {'trades':>7} {'medianFinal':>12} "
          f"{'p10':>9} {'p90':>11} {'P(>=$10k)':>10} {'P(RUIN)':>9}")
    print(_bar())
    for risk in SAFE_RISKS:
        for months in HORIZON_GRID:
            n_trades = max(int(round(tpm * months)), 1)
            draws = rng.choice(pool, size=(N_PATHS, n_trades), replace=True)
            mult  = np.clip(1.0 + risk * draws, 1e-6, None)
            paths = START * np.cumprod(mult, axis=1)
            final = paths[:, -1]
            low   = paths.min(axis=1)
            p_final10 = (final >= TARGET).mean() * 100
            p_ruin    = (low   <= RUIN_LEVEL).mean() * 100
            med = np.median(final)
            p10 = np.percentile(final, 10)
            p90 = np.percentile(final, 90)
            print(f"  {risk*100:>4.0f}% {months:>7}mo {n_trades:>7} {med:>12,.0f} "
                  f"{p10:>9,.0f} {p90:>11,.0f} {p_final10:>9.1f}% {p_ruin:>8.1f}%")
        print(_bar())
    print("  Read: at a risk where RUIN stays ~0%, $10k is a question of HOW LONG, not")
    print("  how reckless. Find the row where P(>=$10k) crosses ~50% — that's your honest")
    print("  ETA to the goal with an account that survives the whole way.")
    print(_bar("="))


if __name__ == "__main__":
    main()
