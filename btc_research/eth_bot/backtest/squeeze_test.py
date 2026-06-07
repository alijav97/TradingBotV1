"""
btc_research/eth_bot/backtest/squeeze_test.py — volatility-compression add-on test.

Tests the ONE implementable kernel from the "ETH volatility compression breakout"
idea: do our pure-S4 signals have BETTER expectancy when volatility was COMPRESSED
just before entry (a squeeze), vs when it was already expanded?

Squeeze defined the standard (TTM) way, computed on ETH H1:
  * Bollinger Bands  : SMA20 +/- 2.0 * stdev20
  * Keltner Channel  : EMA20  +/- 1.5 * ATR20
  * squeeze_on       : BB sits INSIDE KC (upper_BB < upper_KC and lower_BB > lower_KC)
                       = volatility compressed.
Causal: each signal is tagged with the squeeze state of the bar BEFORE its entry
bar (prior completed bar — no look-ahead).

Also a continuous "compression ratio" = BB bandwidth / its own 50-bar average
(low = compressed), terciled.

Everything is split TRAIN (entry year <= 2024) vs TEST (> 2024). A filter is only
worth wiring live if "compressed" beats "expanded" in BOTH periods.

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.squeeze_test
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DIR     = Path(__file__).parent
DATA_DIR = _DIR / "data"
CSV      = DATA_DIR / "backtest_trades.csv"
ETH_CSV  = DATA_DIR / "ETHUSD_H1.csv"

S4_SLOTS = [
    ("rsi_50",    2,  True), ("macd_adx",  6,  True), ("rsi_ema", 10, True),
    ("ema_cross", 5,  False), ("rsi_50",    7,  False), ("keltner", 14, False),
    ("ema_cross",14,  False), ("keltner",  15,  False),
]

TRAIN_END_YEAR = 2024
BB_PERIOD, BB_STD = 20, 2.0
KC_PERIOD, KC_ATR = 20, 1.5
BANDWIDTH_AVG     = 50


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


def _squeeze_series(eth: pd.DataFrame) -> pd.DataFrame:
    """Return df[time, squeeze_prev, comp_ratio_prev, released] — all causal
    (state of the bar BEFORE each timestamp)."""
    e = eth.sort_values("time").reset_index(drop=True)
    c = e["close"].astype(float)
    h = e["high"].astype(float)
    l = e["low"].astype(float)

    # Bollinger
    bb_mid = c.rolling(BB_PERIOD).mean()
    bb_sd  = c.rolling(BB_PERIOD).std()
    bb_up  = bb_mid + BB_STD * bb_sd
    bb_lo  = bb_mid - BB_STD * bb_sd

    # Keltner (EMA20 +/- 1.5*ATR20)
    kc_mid = c.ewm(span=KC_PERIOD, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(KC_PERIOD).mean()
    kc_up = kc_mid + KC_ATR * atr
    kc_lo = kc_mid - KC_ATR * atr

    squeeze_on = ((bb_up < kc_up) & (bb_lo > kc_lo)).fillna(False).astype(bool)
    bandwidth  = (bb_up - bb_lo) / bb_mid.replace(0, np.nan)
    comp_ratio = bandwidth / bandwidth.rolling(BANDWIDTH_AVG).mean()

    # shift introduces NaN -> recast to bool so unary ~ works
    sq1 = squeeze_on.shift(1).fillna(False).astype(bool)   # prior bar compressed?
    sq2 = squeeze_on.shift(2).fillna(False).astype(bool)   # two bars back

    out = pd.DataFrame({
        "time":            e["time"],
        "squeeze_prev":    sq1,
        "comp_ratio_prev": comp_ratio.shift(1),             # prior bar bandwidth ratio
        "released":        sq2 & ~sq1,                       # squeeze just turned off
    })
    return out


def _stats(g: pd.DataFrame):
    n = len(g)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    r = g["r_achieved"].astype(float)
    return n, (r > 0).mean() * 100, r.mean(), r.sum()


def _print_bucket(title: str, df: pd.DataFrame, col: str, order) -> None:
    print()
    print(f"  {title}")
    print(f"  {'bucket':<16} | {'N':>5} {'WR%':>6} {'avgR':>7} {'totR':>8}  (TRAIN)  | "
          f"{'N':>5} {'WR%':>6} {'avgR':>7} {'totR':>8}  (TEST)")
    print(f"  {_bar('-', 88)}")
    for b in order:
        tr = df[(df[col] == b) & df["is_train"]]
        te = df[(df[col] == b) & ~df["is_train"]]
        n1, w1, a1, t1 = _stats(tr)
        n2, w2, a2, t2 = _stats(te)
        print(f"  {str(b):<16} | {n1:>5} {w1:>5.1f}% {a1:>+7.2f} {t1:>+8.1f}           | "
              f"{n2:>5} {w2:>5.1f}% {a2:>+7.2f} {t2:>+8.1f}")


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

    s = _select(df, S4_SLOTS).sort_values("entry_time").reset_index(drop=True)
    sq = _squeeze_series(eth)
    s = pd.merge_asof(s, sq, left_on="entry_time", right_on="time", direction="backward")

    s["is_train"]  = s["entry_time"].dt.year <= TRAIN_END_YEAR
    s["squeeze_b"] = np.where(s["squeeze_prev"] == True, "compressed", "expanded")
    try:
        s["comp_bucket"] = pd.qcut(s["comp_ratio_prev"], 3,
                                   labels=["tight", "normal", "wide"])
    except ValueError:
        s["comp_bucket"] = "normal"
    s["rel_b"] = np.where(s["released"] == True, "just_released", "not_released")

    print()
    print(_bar("="))
    print("  ETH BOT — VOLATILITY-SQUEEZE ADD-ON TEST (pure S4, all candidates)")
    print(f"  TRAIN entry year <= {TRAIN_END_YEAR}  |  TEST > {TRAIN_END_YEAR}")
    print("  Question: do signals after COMPRESSED vol beat signals after EXPANDED vol?")
    print(_bar("="))
    btr = s[s["is_train"]]["r_achieved"].astype(float)
    bte = s[~s["is_train"]]["r_achieved"].astype(float)
    print(f"  baseline (all)   | TRAIN N={len(btr):>4} avgR={btr.mean():+.3f} "
          f"WR={(btr>0).mean()*100:.1f}%  | TEST N={len(bte):>4} "
          f"avgR={bte.mean():+.3f} WR={(bte>0).mean()*100:.1f}%")

    _print_bucket("BY SQUEEZE STATE (prior bar BB-inside-KC?):",
                  s, "squeeze_b", ["compressed", "expanded"])
    _print_bucket("BY COMPRESSION RATIO (BB bandwidth / 50-bar avg):",
                  s, "comp_bucket", ["tight", "normal", "wide"])
    _print_bucket("BY FRESH SQUEEZE RELEASE (squeeze just turned off):",
                  s, "rel_b", ["just_released", "not_released"])

    # Net effect of a "compressed only" filter on the whole candidate set
    print()
    print(_bar("="))
    print("  NET EFFECT — keep only signals with prior-bar squeeze (compressed):")
    print(_bar("="))
    for lbl, sub in (("TRAIN", s[s["is_train"]]), ("TEST", s[~s["is_train"]])):
        all_r = sub["r_achieved"].astype(float)
        comp  = sub[sub["squeeze_b"] == "compressed"]["r_achieved"].astype(float)
        keep_pct = len(comp) / len(all_r) * 100 if len(all_r) else 0
        print(f"  {lbl:<6}: baseline avgR {all_r.mean():+.3f} (N={len(all_r)})  ->  "
              f"compressed-only avgR {comp.mean():+.3f} (N={len(comp)}, "
              f"keeps {keep_pct:.0f}% of signals)")
    print(_bar("="))
    print("  Verdict rule: wire the squeeze filter live ONLY if compressed > expanded")
    print("  in BOTH train AND test with a usable sample. Otherwise it's noise.")
    print()


if __name__ == "__main__":
    main()
