"""
btc_research/eth_bot/backtest/complement_validate.py — OOS test of clubable legs.

regime_complement.py found (pooled 2023+) that almost nothing is anti-correlated
with S4. The only candidates that are (a) ~uncorrelated and (b) positive in S4's
losing months are:
    * ema_cross  (at NON-S4 hours)        corrS4 ~0.00, badAvgR +0.10
    * pin_bar @ hour 2                     strong bad-month patch (+12.8 badTotR)

A pooled number is not enough to wire anything live — it can be curve-fit. This
script re-tests each candidate with the SAME train/test discipline used across
the project:
    TRAIN = entry year <= 2024     TEST = entry year > 2024
and reports, per period: N, WR, AvgR, plus correlation with S4's monthly R and
performance inside S4's losing months.

VERDICT RULE (club it ONLY if all hold):
    * AvgR >= 0 in BOTH train AND test          (standalone not a money-loser)
    * corrS4 <= ~0.2 in BOTH                     (genuinely diversifying)
    * badAvgR >= 0 in BOTH                       (helps when S4 is bleeding)
Otherwise it's noise / curve-fit and we leave the bot as-is. The point of a
complement is DRAWDOWN smoothing, not reaching any 6-month target.

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.complement_validate
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_DIR = Path(__file__).parent
DATA = _DIR / "data"
CSV  = DATA / "backtest_trades.csv"

START_YEAR     = 2023
TRAIN_END_YEAR = 2024

S4_SLOTS = [
    ("rsi_50",    2,  True), ("macd_adx",  6,  True), ("rsi_ema", 10, True),
    ("ema_cross", 5,  False), ("rsi_50",    7,  False), ("keltner", 14, False),
    ("ema_cross",14,  False), ("keltner",  15,  False),
]

# Candidate complement legs to validate.  Each: (label, strategy, hours_or_None)
#   hours None  -> all NON-S4 hours for that strategy
#   hours set   -> only those hours
CANDIDATES = [
    ("ema_cross (non-S4 hrs)", "ema_cross", None),
    ("pin_bar @ H2",           "pin_bar",   {2}),
    ("pin_bar (non-S4 hrs)",   "pin_bar",   None),
]


def _bar(c: str = "-", w: int = 92) -> str:
    return c * w


def _s4_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in S4_SLOTS:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return mask


def _cand_trades(df: pd.DataFrame, strat: str, hours, s4_strat_hours) -> pd.DataFrame:
    g = df[df["strategy"] == strat].copy()
    # never reuse the exact S4 (strategy,hour) slots
    g = g[~g.apply(lambda r: (r["strategy"], r["hour_utc"]) in s4_strat_hours, axis=1)]
    if hours is not None:
        g = g[g["hour_utc"].isin(hours)]
    return g


def _period_stats(g: pd.DataFrame, s4_monthly: pd.Series, all_months, bad_months):
    """Return dict of N, WR, AvgR, corrS4, badN, badAvgR, badTotR for one slice."""
    n = len(g)
    if n == 0:
        return dict(n=0, wr=0.0, avgr=0.0, corr=float("nan"),
                    bad_n=0, bad_avg=0.0, bad_tot=0.0)
    monthly = (g.groupby("month")["r_achieved"].sum()
                 .reindex(all_months, fill_value=0.0))
    corr = monthly.corr(s4_monthly) if monthly.std() > 0 else float("nan")
    bm = g[g["month"].isin(bad_months)]
    return dict(
        n=n,
        wr=(g["outcome"] == "win").mean() * 100,
        avgr=g["r_achieved"].mean(),
        corr=corr,
        bad_n=len(bm),
        bad_avg=bm["r_achieved"].mean() if len(bm) else 0.0,
        bad_tot=bm["r_achieved"].sum() if len(bm) else 0.0,
    )


def main() -> None:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first"); sys.exit(1)

    df = pd.read_csv(CSV, parse_dates=["entry_time"])
    df = df[df["entry_time"].dt.year >= START_YEAR].copy()
    df["month"]    = df["entry_time"].dt.to_period("M").astype(str)
    df["is_train"] = df["entry_time"].dt.year <= TRAIN_END_YEAR

    tr_months = sorted(df[df["is_train"]]["month"].unique())
    te_months = sorted(df[~df["is_train"]]["month"].unique())

    # S4 monthly R and bad months, computed SEPARATELY per period (causal)
    s4 = df[_s4_mask(df)]
    s4_tr = (s4[s4["is_train"]].groupby("month")["r_achieved"].sum()
               .reindex(tr_months, fill_value=0.0))
    s4_te = (s4[~s4["is_train"]].groupby("month")["r_achieved"].sum()
               .reindex(te_months, fill_value=0.0))
    bad_tr = set(s4_tr[s4_tr < 0].index)
    bad_te = set(s4_te[s4_te < 0].index)

    print()
    print(_bar("="))
    print("  ETH BOT — COMPLEMENT OOS VALIDATION  (club a leg only if robust BOTH periods)")
    print(f"  TRAIN <= {TRAIN_END_YEAR}: S4 TotR {s4_tr.sum():+.1f}, "
          f"{len(bad_tr)} bad months  |  TEST > {TRAIN_END_YEAR}: S4 TotR {s4_te.sum():+.1f}, "
          f"{len(bad_te)} bad months")
    print(_bar("="))
    print(f"  {'candidate':<24} {'period':<6} {'N':>4} {'WR%':>6} {'AvgR':>7} "
          f"{'corrS4':>7} {'badN':>5} {'badAvgR':>8} {'badTotR':>8}  verdict")
    print(_bar())

    s4_strat_hours = {(s, h) for s, h, _ in S4_SLOTS}

    for label, strat, hours in CANDIDATES:
        g = _cand_trades(df, strat, hours, s4_strat_hours)
        tr = _period_stats(g[g["is_train"]],  s4_tr, tr_months, bad_tr)
        te = _period_stats(g[~g["is_train"]], s4_te, te_months, bad_te)

        # robustness verdict
        def _ok(s):
            return (s["n"] >= 15 and s["avgr"] >= 0
                    and (pd.isna(s["corr"]) or s["corr"] <= 0.20)
                    and s["bad_avg"] >= 0)
        robust = _ok(tr) and _ok(te)
        verdict = "CLUB (robust)" if robust else "discard"

        for tag, s in (("TRAIN", tr), ("TEST", te)):
            corr_str = f"{s['corr']:>+7.2f}" if pd.notna(s["corr"]) else "    n/a"
            v = ("  <== " + verdict) if tag == "TEST" else ""
            print(f"  {label if tag=='TRAIN' else '':<24} {tag:<6} {s['n']:>4} "
                  f"{s['wr']:>5.1f}% {s['avgr']:>+7.2f} {corr_str} {s['bad_n']:>5} "
                  f"{s['bad_avg']:>+8.2f} {s['bad_tot']:>+8.1f}{v}")
        print(_bar("."))

    print(_bar("="))
    print("  CLUB rule: AvgR>=0 AND corrS4<=0.20 AND badAvgR>=0 in BOTH train AND test.")
    print("  A clubbed leg is for DRAWDOWN smoothing + a little extra frequency — it does")
    print("  NOT make any 6-month target reachable (see monte_carlo frequency sweep).")
    print("  If a candidate is 'CLUB', next step is to wire it as extra live slots and")
    print("  re-run forward_test to confirm the blended equity curve improves.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
