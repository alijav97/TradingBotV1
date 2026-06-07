"""
btc_research/eth_bot/backtest/regime_analysis.py — CAUSAL regime expectancy study.

Goal: build a "rule book" the HONEST way. Instead of fitting to a handful of
losing months (overfitting that fails live), this tags EVERY pure-S4 signal
across all history with market-regime features measured ONLY from data available
AT ENTRY (no look-ahead), then measures expectancy (avg R, win rate) per regime
bucket — and validates it TRAIN (<=2024) vs TEST (>=2025).

A regime rule is only trustworthy if a bucket is bad in BOTH train and test.
If it's only bad in one, it's noise/curve-fit and we discard it.

Regime features (all causal):
  * daily_trend_aligned : does the trade direction agree with the DAILY trend
                          (daily close vs daily EMA, as-of the prior day)?
                          -> tests the higher-timeframe filter the live bot LACKS.
  * vol_bucket          : trailing volatility = ATR14 / entry price, terciled.
  * adx_bucket          : <=25 / 25-40 / >=40  (the live risk-sizing buckets).
  * ext_bucket          : |entry - H1 EMA200| / EMA200, terciled (how extended).

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.regime_analysis
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR     = Path(__file__).parent
DATA_DIR = _DIR / "data"
CSV      = DATA_DIR / "backtest_trades.csv"
ETH_CSV  = DATA_DIR / "ETHUSD_H1.csv"

# Pure S4 = the live config (same as forward_test / realistic_backtest)
S4_SLOTS = [
    ("rsi_50",    2,  True), ("macd_adx",  6,  True), ("rsi_ema", 10, True),
    ("ema_cross", 5,  False), ("rsi_50",    7,  False), ("keltner", 14, False),
    ("ema_cross",14,  False), ("keltner",  15,  False),
]

TRAIN_END_YEAR = 2024   # train = entry year <= this; test = entry year > this
DAILY_EMA_SPAN = 20     # ~20-day trend for the higher-timeframe filter


def _bar(c: str = "-", w: int = 78) -> str:
    return c * w


def _select(df: pd.DataFrame, slots) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in slots:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return df[mask].sort_values("entry_time").reset_index(drop=True)


def _daily_trend(eth: pd.DataFrame) -> pd.DataFrame:
    """Resample ETH H1 -> daily, compute daily EMA. Returns df[date, dclose, dema]
    SHIFTED by one day so a trade only ever sees the PRIOR completed day (causal)."""
    e = eth.set_index("time").sort_index()
    daily = e["close"].astype(float).resample("1D").last().dropna()
    dema  = daily.ewm(span=DAILY_EMA_SPAN, adjust=False).mean()
    out = pd.DataFrame({"dclose": daily, "dema": dema})
    out = out.shift(1)                      # only prior-day info is knowable
    out = out.dropna().reset_index()
    out.columns = ["asof", "dclose", "dema"]
    return out


def _tercile_labels(s: pd.Series, names=("low", "mid", "high")) -> pd.Series:
    try:
        return pd.qcut(s, 3, labels=list(names))
    except ValueError:
        # not enough distinct values — fall back to a single bucket
        return pd.Series([names[1]] * len(s), index=s.index)


def _stats(g: pd.DataFrame) -> tuple[int, float, float, float]:
    n = len(g)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    r = g["r_achieved"].astype(float)
    wr = (r > 0).mean() * 100
    return n, wr, r.mean(), r.sum()


def _print_bucket(title: str, df: pd.DataFrame, col: str, order=None) -> None:
    print()
    print(f"  {title}")
    print(f"  {'bucket':<14} | {'--- TRAIN <=' + str(TRAIN_END_YEAR) + ' ---':^30} | "
          f"{'--- TEST >' + str(TRAIN_END_YEAR) + ' ---':^30}")
    print(f"  {'':<14} | {'N':>5} {'WR%':>6} {'avgR':>7} {'totR':>8} | "
          f"{'N':>5} {'WR%':>6} {'avgR':>7} {'totR':>8}")
    print(f"  {_bar('-', 60)}")
    cats = order if order is not None else sorted(df[col].dropna().unique(), key=str)
    for b in cats:
        tr = df[(df[col] == b) & (df["is_train"])]
        te = df[(df[col] == b) & (~df["is_train"])]
        n1, w1, a1, t1 = _stats(tr)
        n2, w2, a2, t2 = _stats(te)
        flag = ""
        if n1 >= 20 and n2 >= 10 and a1 < 0 and a2 < 0:
            flag = "  <== negative in BOTH (robust avoid)"
        elif n1 >= 20 and n2 >= 10 and a1 > 0 and a2 > 0:
            flag = "  <== positive in BOTH (robust keep)"
        print(f"  {str(b):<14} | {n1:>5} {w1:>5.1f}% {a1:>+7.2f} {t1:>+8.1f} | "
              f"{n2:>5} {w2:>5.1f}% {a2:>+7.2f} {t2:>+8.1f}{flag}")


def main() -> None:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first"); sys.exit(1)
    if not ETH_CSV.exists():
        print(f"ERROR: {ETH_CSV} not found -- run collect_data.py first"); sys.exit(1)

    df = pd.read_csv(CSV, parse_dates=["entry_time", "exit_time"])
    for c in ("entry_time", "exit_time"):
        if df[c].dt.tz is not None:
            df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
    eth = pd.read_csv(ETH_CSV, parse_dates=["time"])
    if eth["time"].dt.tz is not None:
        eth["time"] = eth["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    # Pure S4 candidates (signal quality — use all candidates, not one-position,
    # to maximise sample for expectancy). r_achieved is per-signal R.
    s = _select(df, S4_SLOTS).copy()
    s = s.sort_values("entry_time").reset_index(drop=True)

    # ---- causal regime features ------------------------------------------------
    s["is_train"] = s["entry_time"].dt.year <= TRAIN_END_YEAR

    # daily trend alignment (higher-timeframe filter the bot currently lacks)
    dt = _daily_trend(eth)
    s = pd.merge_asof(s, dt, left_on="entry_time", right_on="asof", direction="backward")
    daily_up = s["dclose"] > s["dema"]
    s["daily_trend_aligned"] = np.where(
        ((s["direction"] == "long") & daily_up) |
        ((s["direction"] == "short") & ~daily_up),
        "aligned", "counter")

    # trailing volatility, ADX, extension (all from per-trade columns = causal)
    s["atr_pct"] = s["atr"].astype(float) / s["entry"].astype(float)
    s["ext_pct"] = (s["entry"].astype(float) - s["ema200"].astype(float)).abs() \
                   / s["ema200"].astype(float)
    s["vol_bucket"] = _tercile_labels(s["atr_pct"])
    s["ext_bucket"] = _tercile_labels(s["ext_pct"])
    adx = s["adx"].astype(float)
    s["adx_bucket"] = np.where(adx >= 40, "strong>=40",
                       np.where(adx <= 25, "weak<=25", "mid25-40"))

    print()
    print(_bar("="))
    print("  ETH BOT — CAUSAL REGIME EXPECTANCY (pure S4, all candidates)")
    print(f"  TRAIN = entry year <= {TRAIN_END_YEAR}   TEST = entry year > {TRAIN_END_YEAR}")
    print(f"  A rule is only trustworthy if a bucket is bad in BOTH train AND test.")
    print(_bar("="))
    n_tr = int(s["is_train"].sum()); n_te = int((~s["is_train"]).sum())
    base_tr = s[s["is_train"]]["r_achieved"].astype(float)
    base_te = s[~s["is_train"]]["r_achieved"].astype(float)
    print(f"  baseline       | TRAIN N={n_tr:>4} avgR={base_tr.mean():+.3f} "
          f"WR={ (base_tr>0).mean()*100:.1f}%  | "
          f"TEST N={n_te:>4} avgR={base_te.mean():+.3f} WR={(base_te>0).mean()*100:.1f}%")

    _print_bucket("BY DAILY-TREND ALIGNMENT (higher-TF filter the bot lacks):",
                  s, "daily_trend_aligned", order=["aligned", "counter"])
    _print_bucket("BY VOLATILITY (ATR/price tercile):",
                  s, "vol_bucket", order=["low", "mid", "high"])
    _print_bucket("BY ADX (the live risk-sizing buckets):",
                  s, "adx_bucket", order=["weak<=25", "mid25-40", "strong>=40"])
    _print_bucket("BY EXTENSION from H1 EMA200 (tercile):",
                  s, "ext_bucket", order=["low", "mid", "high"])

    # per-strategy x daily alignment, train/test (which patterns fail counter-trend)
    print()
    print(_bar("="))
    print("  PER-STRATEGY avgR by daily-trend alignment (TRAIN | TEST)")
    print(_bar("="))
    print(f"  {'strategy':<12} {'aligned(tr)':>12} {'counter(tr)':>12} | "
          f"{'aligned(te)':>12} {'counter(te)':>12}")
    print(f"  {_bar('-', 70)}")
    for strat in sorted(s["strategy"].unique()):
        row = s[s["strategy"] == strat]
        def _a(sub, al):
            g = sub[sub["daily_trend_aligned"] == al]["r_achieved"].astype(float)
            return f"{g.mean():+.2f}({len(g)})" if len(g) else "    -"
        tr = row[row["is_train"]]; te = row[~row["is_train"]]
        print(f"  {strat:<12} {_a(tr,'aligned'):>12} {_a(tr,'counter'):>12} | "
              f"{_a(te,'aligned'):>12} {_a(te,'counter'):>12}")
    print(_bar("="))
    print("  Read: a 'robust avoid' bucket (negative avgR in BOTH train & test, decent N)")
    print("  is a candidate live filter. A bucket good in train but bad in test = curve-fit.")
    print()


if __name__ == "__main__":
    main()
