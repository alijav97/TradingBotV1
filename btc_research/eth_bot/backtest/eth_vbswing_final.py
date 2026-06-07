"""
btc_research/eth_bot/backtest/eth_vbswing_final.py — final risk decision, throttle-2.

The chosen damage-control rule is THROTTLE-2: after 2 consecutive losses, halve risk
until the next win (keeps every trade, so the post-streak recovery winners are never
skipped — the circuit-breaker test showed this beats a hard pause).

This script answers the last open question: at what BASE risk do you run it? It
bootstraps the REAL VBSwing-on-ETH R-distribution (best config: BTC hours [1,2,3,8])
and applies the throttle-2 logic inside each path, for base risk 6% / 8% / 10%, over
6- and 12-month horizons. For each it reports BOTH:

   goal odds       : median final, p10, P(>=$8k), P(>=$10k), P(ruin<=$50)
   pain you endure : median and 95th-percentile WORST DRAWDOWN within the window

so the 6-vs-8-vs-10 choice is made on the full tradeoff, not just the upside. Plain
(no-throttle) rows are shown beside each so you can see exactly what throttle buys.

NOTE: bootstrap draws are independent (it produces loss streaks naturally but ignores
serial correlation), and the throttle keys off the running consecutive-loss count of
the SHUFFLED sequence — valid because throttle depends only on order of wins/losses,
not calendar time. It shows the SHAPE of the tradeoff, not a promised single number.

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.eth_vbswing_final
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btc_research.eth_bot.backtest.eth_vbswing import (
    _load_eth, _indicators, simulate, VBSwingStrategy, STARTING_BALANCE,
)

HOURS       = [1, 2, 3, 8]
CB_LOSSES   = 2            # throttle after 2 consecutive losses
THROTTLE_F  = 0.5         # halve risk
RISK_LEVELS = [0.06, 0.08, 0.10]
HORIZONS    = [6, 12]      # months
TARGET_8K   = 8_000.0
TARGET_10K  = 10_000.0
RUIN        = 50.0         # -90% from $500
N_PATHS     = 50_000
SEED        = 7


def _bar(c: str = "-", w: int = 100) -> str:
    return c * w


def _mc(pool, base_risk, throttle, n_tr, rng):
    """Vectorised across paths, loop over trades (state-dependent throttle)."""
    n = N_PATHS
    bal    = np.full(n, STARTING_BALANCE)
    consec = np.zeros(n, dtype=int)
    peak   = np.full(n, STARTING_BALANCE)
    low    = np.full(n, STARTING_BALANCE)
    maxdd  = np.zeros(n)
    for _ in range(n_tr):
        R = rng.choice(pool, size=n)
        if throttle:
            eff = base_risk * np.where(consec >= CB_LOSSES, THROTTLE_F, 1.0)
        else:
            eff = base_risk
        bal = bal * np.clip(1.0 + eff * R, 1e-6, None)
        peak = np.maximum(peak, bal)
        low  = np.minimum(low, bal)
        maxdd = np.maximum(maxdd, (peak - bal) / peak)
        consec = np.where(R <= 0, consec + 1, 0)
    return bal, low, maxdd


def main() -> None:
    df = _load_eth()
    atr, ema200, adx = _indicators(df)
    strat = VBSwingStrategy()
    # R-pool is risk-independent (sizing never changes R), so any risk_tuple works.
    trades = simulate(df, atr, ema200, adx, strat, HOURS, (0.03, 0.02, 0.03))
    pool = np.array([float(t["r_multiple"]) for t in trades], dtype=float)
    n = len(trades)
    tpm = n / max((pd.Timestamp(trades[-1]["open_time"])
                   - pd.Timestamp(trades[0]["open_time"])).days / 30.44, 1.0)
    wr = (pool > 0).mean() * 100
    rng = np.random.default_rng(SEED)

    print()
    print(_bar("="))
    print("  ETH VBSwing — FINAL RISK DECISION (THROTTLE-2: -2 losses -> half risk until a win)")
    print(f"  pool={n} real trades | WR {wr:.1f}% | avg {pool.mean():+.2f}R | "
          f"~{tpm:.1f} trades/mo | {N_PATHS:,} paths")
    print(_bar("="))

    for months in HORIZONS:
        n_tr = max(int(round(tpm * months)), 1)
        print()
        print(_bar("="))
        print(f"  {months}-MONTH HORIZON  ({n_tr} trades/path)")
        print(_bar("="))
        print(f"  {'risk':>5} {'mode':<11} {'medFinal':>10} {'p10':>8} "
              f"{'P(>=8k)':>8} {'P(>=10k)':>9} {'P(ruin)':>8} "
              f"{'medMaxDD':>9} {'p95MaxDD':>9}")
        print(_bar())
        for risk in RISK_LEVELS:
            for mode, throttle in (("plain", False), ("throttle-2", True)):
                bal, low, maxdd = _mc(pool, risk, throttle, n_tr, rng)
                med = np.median(bal); p10 = np.percentile(bal, 10)
                p8  = (bal >= TARGET_8K).mean() * 100
                p10k = (bal >= TARGET_10K).mean() * 100
                pruin = (low <= RUIN).mean() * 100
                meddd = np.median(maxdd) * 100
                p95dd = np.percentile(maxdd, 95) * 100
                star = "  <-- THROTTLE-2" if (throttle and abs(risk-0.08) < 1e-9) else ""
                print(f"  {risk*100:>4.0f}% {mode:<11} {med:>10,.0f} {p10:>8,.0f} "
                      f"{p8:>7.1f}% {p10k:>8.1f}% {pruin:>7.1f}% "
                      f"{meddd:>8.1f}% {p95dd:>8.1f}%{star}")
            print(_bar("-", 100))

    print()
    print("  HOW TO DECIDE:")
    print("   * P(>=$10k) is the goal odds; medMaxDD / p95MaxDD is the pain you must hold")
    print("     through. medMaxDD = your TYPICAL worst dip; p95MaxDD = a bad-but-realistic one.")
    print("   * 'throttle-2' rows vs 'plain' show exactly what the breaker buys at each risk.")
    print("   * Pick the highest risk whose p95MaxDD you can stomach WITHOUT quitting — because")
    print("     quitting in the dip locks the loss and forfeits the recovery winners.")
    print("   * P(ruin) must stay ~0. Any row with meaningful ruin is off the table.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
