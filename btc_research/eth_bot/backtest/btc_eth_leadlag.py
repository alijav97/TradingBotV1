"""
btc_research/eth_bot/backtest/btc_eth_leadlag.py

DEEP RESEARCH: does BTC LEAD ETH, and is the lead tradeable?

Hypothesis (user): BTC moves first, then -- after some delay -- triggers an ETH
move. We want to (1) confirm the lead-lag and its delay, (2) scale it (how many
% of ETH move per % of BTC move = beta), and (3) find the CONDITIONS under which
the BTC->ETH follow-through is strongest, so we can build a NEW kill-zone trigger
("BTC just impulsed -> go with ETH") that is orthogonal to the existing
RSI/MACD/engulfing slots and can push realistic PnL past $13k toward $20k.

Reads the same H1 OHLCV the main backtest uses:
    data/BTCUSD_H1.csv , data/ETHUSD_H1.csv   (cols: time,open,high,low,close,volume)

Sections
  1. LEAD-LAG PROFILE   -- corr(r_btc[t], r_eth[t+k]) for k = -3..+12h.
                           Peak at k>0 => BTC leads ETH by k hours.
  2. BETA / SCALING      -- at the best lag, slope of r_eth on r_btc + R^2,
                           split by BTC volatility regime (calm vs wild).
  3. IMPULSE FOLLOW-THRU -- after a BTC hourly move >= threshold, what does ETH
                           do over the next 1/2/3/6h? sign-adjusted mean,
                           hit-rate, and a crude expectancy. THE tradeability test.
  4. CONDITIONS          -- best UTC hours and impulse buckets for the trigger,
                           and the "already moved?" filter (skip if ETH already
                           caught up to BTC in the same bar).

NO lookahead: every forward return is strictly t+1.. relative to the trigger bar.

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.btc_eth_leadlag
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR     = Path(__file__).parent
DATA     = _DIR / "data"
BTC_CSV  = DATA / "BTCUSD_H1.csv"
ETH_CSV  = DATA / "ETHUSD_H1.csv"

START_YEAR = 2023

# -- Research parameters -------------------------------------------------------
LAGS        = list(range(-3, 13))          # hours; +k => ETH future vs BTC now
HORIZONS    = [1, 2, 3, 6]                  # forward-return horizons (hours)
IMPULSES    = [0.003, 0.005, 0.008, 0.012, 0.018]  # BTC |hourly return| triggers
KZ_HOURS    = [2, 5, 6, 7, 10, 14, 15]     # current ETH kill zones (for overlap)


def _bar(c: str = "-", w: int = 96) -> str:
    return c * w


def _load(path: Path, tag: str) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {path} not found -- run collect_data.py first")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").reset_index(drop=True)
    df = df.rename(columns={c: f"{tag}_{c}" for c in
                            ["open", "high", "low", "close", "volume"]})
    return df


def _merged() -> pd.DataFrame:
    btc = _load(BTC_CSV, "btc")
    eth = _load(ETH_CSV, "eth")
    m = pd.merge(btc, eth, on="time", how="inner")
    m = m[m["time"].dt.year >= START_YEAR].reset_index(drop=True)
    # hourly close-to-close returns
    m["r_btc"] = m["btc_close"].pct_change()
    m["r_eth"] = m["eth_close"].pct_change()
    m["hour"]  = m["time"].dt.hour
    # only keep contiguous H1 steps (drop weekend-gap rows from return calc)
    dt = m["time"].diff()
    bad = dt != pd.Timedelta(hours=1)
    m.loc[bad, ["r_btc", "r_eth"]] = np.nan
    return m


def _section_leadlag(m: pd.DataFrame) -> int:
    print()
    print(_bar("="))
    print(f"  1. LEAD-LAG PROFILE  corr(r_btc[t], r_eth[t+k])   ({START_YEAR}+, "
          f"{m['r_btc'].notna().sum():,} hourly returns)")
    print(_bar())
    print(f"  {'lag k (h)':>9} {'corr':>8}   {'(k>0 = BTC leads ETH by k hours)'}")
    print(_bar())
    best_k, best_c = 0, -2.0
    rows = []
    for k in LAGS:
        if k >= 0:
            c = m["r_btc"].corr(m["r_eth"].shift(-k))
        else:
            c = m["r_btc"].corr(m["r_eth"].shift(-k))  # negative shift = ETH past
        rows.append((k, c))
        if k >= 0 and c > best_c:
            best_c, best_k = c, k
    for k, c in rows:
        bar = "#" * int(max(0, c) * 60)
        star = "  <== peak" if k == best_k else ("   (same-bar)" if k == 0 else "")
        print(f"  {k:>9} {c:>+8.3f}   {bar}{star}")
    print(_bar())
    if best_k == 0:
        print("  READ: peak at k=0 -> BTC & ETH move TOGETHER same hour (little tradeable")
        print("        lag at H1). A faster timeframe (M5/M15) may show the real lead.")
    else:
        print(f"  READ: peak at k={best_k}h -> BTC leads ETH by ~{best_k}h. Tradeable lag.")
    return best_k


def _section_beta(m: pd.DataFrame, k: int) -> None:
    print()
    print(_bar("="))
    print(f"  2. BETA / SCALING  r_eth[t+{k}] = alpha + beta * r_btc[t]")
    print(_bar())
    d = pd.DataFrame({"x": m["r_btc"], "y": m["r_eth"].shift(-k)}).dropna()

    def _fit(sub: pd.DataFrame, label: str) -> None:
        if len(sub) < 30:
            print(f"  {label:<22} (too few rows)")
            return
        beta, alpha = np.polyfit(sub["x"], sub["y"], 1)
        r = sub["x"].corr(sub["y"])
        print(f"  {label:<22} beta={beta:>+6.2f}  R^2={r*r:>5.3f}  "
              f"alpha={alpha*100:>+6.3f}%/h  N={len(sub):>6,}")

    _fit(d, "ALL hours")
    # BTC volatility regime: rolling 24h std of r_btc
    vol = m["r_btc"].rolling(24).std()
    med = vol.median()
    calm = d[vol.reindex(d.index) <= med]
    wild = d[vol.reindex(d.index) >  med]
    _fit(calm, "BTC calm (<=med vol)")
    _fit(wild, "BTC wild (> med vol)")
    print(_bar())
    print("  READ: beta ~1 means ETH matches BTC; beta>1 means ETH amplifies the")
    print("  move (more reward, more risk). High R^2 in 'wild' = trigger is reliable")
    print("  exactly when moves are big enough to trade.")


def _section_impulse(m: pd.DataFrame) -> None:
    print()
    print(_bar("="))
    print("  3. IMPULSE FOLLOW-THROUGH  (after BTC |r| >= thr, ETH forward move)")
    print("     follow = eth_fwd_return * sign(btc_move)  -- >0 means ETH followed BTC")
    print(_bar())
    print(f"  {'BTC trig':>8} {'N':>6} {'hz':>3} {'ETHfollow%':>10} "
          f"{'hit%':>6} {'avg|eth|%':>9} {'eth/btc':>8}")
    print(_bar())
    for thr in IMPULSES:
        sig = m[m["r_btc"].abs() >= thr]
        n = len(sig)
        if n < 20:
            print(f"  {thr*100:>7.1f}% {n:>6}  (too few)")
            continue
        for h in HORIZONS:
            fwd = (m["eth_close"].shift(-h) / m["eth_close"] - 1)
            f = (fwd.reindex(sig.index) * np.sign(sig["r_btc"]))
            f = f.dropna()
            if f.empty:
                continue
            avg_follow = f.mean() * 100
            hit = (f > 0).mean() * 100
            avg_abs = fwd.reindex(sig.index).abs().dropna().mean() * 100
            # eth move per unit btc impulse, in the trigger hour
            ratio = (m["r_eth"].reindex(sig.index).abs().mean()
                     / m["r_btc"].reindex(sig.index).abs().mean())
            tag = "  <== best hz" if h == HORIZONS[0] else ""
            print(f"  {thr*100:>7.1f}% {n:>6} {h:>3} {avg_follow:>+9.3f}% "
                  f"{hit:>5.1f}% {avg_abs:>8.3f}% {ratio:>8.2f}{tag if False else ''}")
        print(_bar("."))
    print(_bar())
    print("  READ: a tradeable trigger wants hit% > ~55 AND ETHfollow% clearly")
    print("  positive at a horizon we can hold. eth/btc>1 = ETH amplifies the move.")


def _section_conditions(m: pd.DataFrame, thr: float, h: int) -> None:
    print()
    print(_bar("="))
    print(f"  4. CONDITIONS  (BTC |r|>= {thr*100:.1f}%, ETH follow over {h}h, by UTC hour)")
    print(_bar())
    print(f"  {'hourUTC':>7} {'KZ?':>4} {'N':>5} {'ETHfollow%':>10} {'hit%':>6} {'avg|eth|%':>9}")
    print(_bar())
    fwd = (m["eth_close"].shift(-h) / m["eth_close"] - 1)
    sig_all = m[m["r_btc"].abs() >= thr].copy()
    sig_all["follow"] = (fwd.reindex(sig_all.index) * np.sign(sig_all["r_btc"]))
    rows = []
    for hour, g in sig_all.groupby("hour"):
        gg = g.dropna(subset=["follow"])
        if len(gg) < 8:
            continue
        rows.append((hour, len(gg), gg["follow"].mean() * 100,
                     (gg["follow"] > 0).mean() * 100,
                     gg["follow"].abs().mean() * 100))
    rows.sort(key=lambda r: r[2], reverse=True)
    for hour, n, follow, hit, avgabs in rows:
        kz = "yes" if hour in KZ_HOURS else ""
        flag = "  <== strong" if (follow > 0 and hit >= 55) else ""
        print(f"  {hour:>7} {kz:>4} {n:>5} {follow:>+9.3f}% {hit:>5.1f}% "
              f"{avgabs:>8.3f}%{flag}")
    print(_bar())
    print("  READ: hours with follow%>0 AND hit%>=55 are candidate NEW kill zones")
    print("  for a BTC-trigger ETH strategy. 'KZ?=yes' overlaps an existing slot.")


def main() -> None:
    m = _merged()
    print()
    print(_bar("="))
    print(f"  ETH BOT -- BTC->ETH LEAD-LAG RESEARCH  ({START_YEAR}+)")
    print(f"  H1 close-to-close returns | aim: a NEW orthogonal trigger toward $20k")
    print(_bar("="))

    best_k = _section_leadlag(m)
    _section_beta(m, best_k)
    _section_impulse(m)
    # pick a sensible trigger/horizon for the conditions drill-down
    _section_conditions(m, thr=0.008, h=3)

    print()
    print(_bar("="))
    print("  NEXT: if section 3/4 show a hit%>=55 trigger, I'll build it as a new")
    print("  kill-zone slot (BTC impulse -> ETH continuation) and run it through")
    print("  realistic_backtest.py to measure the PnL lift under the one-trade rule.")
    print("  If the H1 lead is ~0, we re-pull M5/M15 data to catch a faster lead.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
