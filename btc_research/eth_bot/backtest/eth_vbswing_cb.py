"""
btc_research/eth_bot/backtest/eth_vbswing_cb.py — does a circuit breaker help?

User question: "If I hit 3 losses in a row and then STOP trading for 4 days, does it
control the damage?" Testable on the real ETH VBSwing history. The catch with any
'stop after losses' rule: a blind time-pause also skips the rare big WINNERS that
arrive right after a streak — and this strategy's whole edge is those 5R runners.
So a breaker can cut drawdown AND cut return. We measure both.

Three damage-control modes (all on the best config: BTC hours [1,2,3,8]):
  * BASELINE      — no breaker.
  * HARD PAUSE    — after N consecutive losses, skip ALL entries for D days.
                    (the user's exact idea = N=3, D=4)
  * THROTTLE      — after N consecutive losses, HALVE risk until the next win
                    (stays in the market for the recovery winner, at smaller size).

We run each at flat 10% (the aggressive setting where damage control matters most)
and flat 6% (the safe setting), over FULL / TRAIN / TEST. What matters:
  * MaxDD%   — did damage actually shrink?   (lower = better)
  * retX     — final equity vs the no-breaker baseline (return KEPT; <1 = sacrificed)
  * MaxCL    — max consecutive losses still endured.

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.eth_vbswing_cb
"""
from __future__ import annotations

import pandas as pd

from btc_research.eth_bot.backtest.eth_vbswing import (
    _load_eth, _indicators, simulate, _stats, VBSwingStrategy,
)

HOURS = [1, 2, 3, 8]

# (label, cb_losses, cb_pause_days, cb_throttle)
CB_CONFIGS = [
    ("BASELINE (no breaker)",      None, 0.0,  False),
    ("PAUSE 2 losses -> 4d",       2,    4.0,  False),
    ("PAUSE 3 losses -> 4d  *YOU*",3,    4.0,  False),
    ("PAUSE 3 losses -> 7d",       3,    7.0,  False),
    ("PAUSE 3 losses -> 2d",       3,    2.0,  False),
    ("PAUSE 4 losses -> 4d",       4,    4.0,  False),
    ("THROTTLE 2 losses x0.5",     2,    0.0,  True),
    ("THROTTLE 3 losses x0.5",     3,    0.0,  True),
]

RISK_LEVELS = [("FLAT 10% (aggressive)", 0.10), ("FLAT 6% (safe)", 0.06)]


def _bar(c: str = "-", w: int = 100) -> str:
    return c * w


def _run(df, atr, ema200, adx, strat, risk, cb, ds=None, de=None) -> dict:
    _, n_loss, pause_d, throttle = cb
    rt = (risk, risk, risk)
    trades = simulate(df, atr, ema200, adx, strat, HOURS, rt,
                      date_start=ds, date_end=de,
                      cb_losses=n_loss, cb_pause_days=pause_d, cb_throttle=throttle)
    return _stats(trades)


def _print_table(df, atr, ema200, adx, strat, risk_label, risk, train_end):
    print()
    print(_bar("="))
    print(f"  DAMAGE CONTROL @ {risk_label}")
    print(_bar("="))
    print(f"  {'config':<28} {'N':>4} {'WR%':>6} {'AvgR':>6} {'MaxDD%':>7} "
          f"{'MaxCL':>5} {'PF':>5} {'+mo/mo':>8} {'retX':>7}")
    print(_bar())
    base_final = None
    for cb in CB_CONFIGS:
        s = _run(df, atr, ema200, adx, strat, risk, cb)
        if base_final is None:
            base_final = s["final"]
        retx = s["final"] / base_final if base_final else 0.0
        print(f"  {cb[0]:<28} {s['n']:>4} {s['wr']:>5.1f}% {s['avgr']:>+6.2f} "
              f"{s['maxdd']:>6.1f}% {s['maxcl']:>5} {s['pf']:>5.2f} "
              f"{s['pos_mo']:>3}/{s['tot_mo']:<3} {retx:>6.2f}x")
    print(_bar())
    # robustness of the user's rule + best throttle, on TRAIN / TEST
    print(f"  TRAIN/TEST check (your rule vs throttle, @ {risk_label}):")
    for cb in (CB_CONFIGS[2], CB_CONFIGS[7]):   # PAUSE 3->4d, THROTTLE 3
        for plabel, ds, de in (("TRAIN<=2024", None, train_end),
                               ("TEST >=2025", train_end, None)):
            s = _run(df, atr, ema200, adx, strat, risk, cb, ds=ds, de=de)
            print(f"    {cb[0]:<28} {plabel:<12} N={s['n']:>4} AvgR {s['avgr']:>+5.2f} "
                  f"MaxDD {s['maxdd']:>5.1f}% PF {s['pf']:>5.2f}")
    print(_bar("="))


def main() -> None:
    df = _load_eth()
    atr, ema200, adx = _indicators(df)
    strat = VBSwingStrategy()
    train_end = pd.Timestamp("2025-01-01")

    print()
    print(_bar("="))
    print("  ETH VBSwing — CIRCUIT-BREAKER TEST  (does stopping after losses help?)")
    print("  config: BTC hours [1,2,3,8] | TP1 2R / TP2 5R | one-position | fresh $500")
    print("  retX = final equity vs the no-breaker baseline (1.00 = same, <1 = gave up return)")
    print(_bar("="))

    for risk_label, risk in RISK_LEVELS:
        _print_table(df, atr, ema200, adx, strat, risk_label, risk, train_end)

    print()
    print("  HOW TO READ:")
    print("   * A breaker is WORTH IT only if MaxDD drops MORE than retX falls.")
    print("     (e.g. MaxDD -70%->-50% while keeping 0.85x of the return = good trade.)")
    print("   * HARD PAUSE skips winners too — watch retX collapse if the pause is greedy.")
    print("   * THROTTLE stays in for the recovery winner at half size — usually softer DD")
    print("     for less return given up. Compare the two honestly.")
    print("   * If a breaker barely moves MaxDD, the streaks are too spread out to 'dodge'")
    print("     by pausing — the real fix is then lower BASE risk, not a breaker.")
    print()


if __name__ == "__main__":
    main()
