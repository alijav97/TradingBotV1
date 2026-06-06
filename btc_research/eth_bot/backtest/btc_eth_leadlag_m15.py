"""
btc_research/eth_bot/backtest/btc_eth_leadlag_m15.py

DEEP RESEARCH (finer grain): does BTC lead ETH INTRA-HOUR, at M15?

The H1 study (btc_eth_leadlag.py) showed BTC and ETH move TOGETHER within the
hour (corr peak at lag 0, ~0 elsewhere) -> no tradeable lead at H1. The lead, if
any, must be faster than 1 hour. This script repeats the analysis on M15 candles
to find a 15/30/45-min lead we could use to TIME ETH ENTRIES better:
  "BTC just pushed up on the last M15 -> enter ETH now, ahead of its catch-up."

Reads M15 OHLCV produced by collect_data.py:
    data/BTCUSD_M15.csv , data/ETHUSD_M15.csv   (cols: time,open,high,low,close,volume)

Sections (same logic as the H1 study, M15 units):
  1. LEAD-LAG PROFILE   corr(r_btc[t], r_eth[t+k]) for k = -4..+12 quarters
                        (-1h .. +3h). Peak at k>0 => BTC leads ETH by k*15min.
  2. BETA / SCALING     ETH-on-BTC slope + R^2 at the best lag, calm vs wild.
  3. IMPULSE FOLLOW-THRU after a BTC M15 move >= thr, ETH move over next
                        1/2/4/8 quarters (15m/30m/1h/2h). hit-rate = tradeability.
  4. ENTRY-TIMING EDGE  for each kill-zone HOUR, compare entering ETH at the
                        hour's open vs. waiting for the first M15 where BTC
                        confirms direction -- does the BTC-confirmed entry get a
                        better forward ETH return over the rest of the hour?

NO lookahead: forward returns are strictly t+1.. quarters from the trigger.

== USAGE ==
  # first regenerate data so the M15 files exist:
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.collect_data
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.btc_eth_leadlag_m15
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR    = Path(__file__).parent
DATA    = _DIR / "data"
BTC_CSV = DATA / "BTCUSD_M15.csv"
ETH_CSV = DATA / "ETHUSD_M15.csv"

START_YEAR = 2023
BARS_PER_HR = 4                                  # M15

LAGS      = list(range(-4, 13))                  # quarters; +k => ETH future
HORIZONS  = [1, 2, 4, 8]                          # quarters = 15m/30m/1h/2h
IMPULSES  = [0.002, 0.003, 0.005, 0.008, 0.012]  # BTC |M15 return| triggers
KZ_HOURS  = [2, 5, 6, 7, 10, 14, 15]


def _bar(c: str = "-", w: int = 96) -> str:
    return c * w


def _load(path: Path, tag: str) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("  Run collect_data.py first to fetch the M15 files.")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df.rename(columns={c: f"{tag}_{c}" for c in
                              ["open", "high", "low", "close", "volume"]})


def _merged() -> pd.DataFrame:
    btc = _load(BTC_CSV, "btc")
    eth = _load(ETH_CSV, "eth")
    m = pd.merge(btc, eth, on="time", how="inner")
    m = m[m["time"].dt.year >= START_YEAR].reset_index(drop=True)
    m["r_btc"] = m["btc_close"].pct_change()
    m["r_eth"] = m["eth_close"].pct_change()
    m["hour"]  = m["time"].dt.hour
    m["minute"] = m["time"].dt.minute
    # null out returns across non-contiguous M15 steps (gaps/weekends)
    bad = m["time"].diff() != pd.Timedelta(minutes=15)
    m.loc[bad, ["r_btc", "r_eth"]] = np.nan
    return m


def _section_leadlag(m: pd.DataFrame) -> int:
    print()
    print(_bar("="))
    print(f"  1. LEAD-LAG PROFILE (M15)  corr(r_btc[t], r_eth[t+k])   "
          f"({m['r_btc'].notna().sum():,} returns)")
    print(_bar())
    print(f"  {'lag (q)':>7} {'=min':>5} {'corr':>8}   (k>0 = BTC leads ETH)")
    print(_bar())
    best_k, best_c = 0, -2.0
    rows = []
    for k in LAGS:
        c = m["r_btc"].corr(m["r_eth"].shift(-k))
        rows.append((k, c))
        if k >= 0 and c > best_c:
            best_c, best_k = c, k
    for k, c in rows:
        b = "#" * int(max(0, c) * 60)
        star = "  <== peak" if k == best_k else (" (same-bar)" if k == 0 else "")
        print(f"  {k:>7} {k*15:>5} {c:>+8.3f}   {b}{star}")
    print(_bar())
    if best_k == 0:
        print("  READ: peak still at k=0 -> lead is faster than 15min (need M5/tick).")
    else:
        print(f"  READ: peak at k={best_k}q -> BTC leads ETH by ~{best_k*15}min. "
              f"Tradeable for entry timing.")
    return best_k


def _section_beta(m: pd.DataFrame, k: int) -> None:
    print()
    print(_bar("="))
    print(f"  2. BETA / SCALING (M15)  r_eth[t+{k}] = alpha + beta * r_btc[t]")
    print(_bar())
    d = pd.DataFrame({"x": m["r_btc"], "y": m["r_eth"].shift(-k)}).dropna()

    def _fit(sub: pd.DataFrame, label: str) -> None:
        if len(sub) < 50:
            print(f"  {label:<22} (too few rows)"); return
        beta, alpha = np.polyfit(sub["x"], sub["y"], 1)
        r = sub["x"].corr(sub["y"])
        print(f"  {label:<22} beta={beta:>+6.2f}  R^2={r*r:>5.3f}  N={len(sub):>7,}")

    _fit(d, "ALL")
    vol = m["r_btc"].rolling(96).std()             # 1-day rolling vol on M15
    med = vol.median()
    _fit(d[vol.reindex(d.index) <= med], "BTC calm")
    _fit(d[vol.reindex(d.index) >  med], "BTC wild")
    print(_bar())


def _section_impulse(m: pd.DataFrame) -> None:
    print()
    print(_bar("="))
    print("  3. IMPULSE FOLLOW-THROUGH (M15)  follow = eth_fwd * sign(btc_move)")
    print(_bar())
    print(f"  {'BTC trig':>8} {'N':>6} {'hz(min)':>7} {'ETHfollow%':>10} "
          f"{'hit%':>6} {'eth/btc':>8}")
    print(_bar())
    for thr in IMPULSES:
        sig = m[m["r_btc"].abs() >= thr]
        if len(sig) < 30:
            print(f"  {thr*100:>7.1f}% {len(sig):>6}  (too few)"); continue
        ratio = (m["r_eth"].reindex(sig.index).abs().mean()
                 / m["r_btc"].reindex(sig.index).abs().mean())
        for h in HORIZONS:
            fwd = (m["eth_close"].shift(-h) / m["eth_close"] - 1)
            f = (fwd.reindex(sig.index) * np.sign(sig["r_btc"])).dropna()
            if f.empty:
                continue
            print(f"  {thr*100:>7.1f}% {len(sig):>6} {h*15:>7} "
                  f"{f.mean()*100:>+9.3f}% {(f>0).mean()*100:>5.1f}% {ratio:>8.2f}")
        print(_bar("."))
    print(_bar())
    print("  READ: hit% > ~55 at a horizon we can act on = a real entry-timing edge.")


def _section_entry_timing(m: pd.DataFrame, thr: float = 0.003) -> None:
    print()
    print(_bar("="))
    print(f"  4. ENTRY-TIMING EDGE  (kill-zone hours; BTC-confirmed M15 entry vs")
    print(f"     hour-open entry; forward ETH return to end of hour)")
    print(_bar())
    print(f"  {'hourUTC':>7} {'N':>5} {'openEntry%':>10} {'btcConf%':>9} "
          f"{'edge(bp)':>9} {'conf-hit%':>9}")
    print(_bar())
    m = m.copy()
    m["date"] = m["time"].dt.date
    for hour in KZ_HOURS:
        hh = m[m["hour"] == hour]
        open_rets, conf_rets, conf_hits = [], [], []
        for _, g in hh.groupby("date"):
            g = g.sort_values("minute")
            if len(g) < 2:
                continue
            o_close = g["eth_close"].iloc[-1]
            o_open  = g["eth_open"].iloc[0]
            # baseline: enter at hour open, hold to hour end
            base_dir = 1  # measure raw long bias removed below via sign of move
            open_ret = (o_close / o_open - 1)
            open_rets.append(open_ret)
            # BTC-confirmed: first M15 in the hour where |r_btc|>=thr; enter ETH
            # in that direction at that M15 close, hold to hour end
            trig = g[g["r_btc"].abs() >= thr]
            if trig.empty:
                continue
            row = trig.iloc[0]
            entry_px = row["eth_close"]
            d = np.sign(row["r_btc"])
            conf_ret = (o_close / entry_px - 1) * d
            conf_rets.append(conf_ret)
            conf_hits.append(1.0 if conf_ret > 0 else 0.0)
        if len(conf_rets) < 10:
            continue
        open_bp = np.mean(np.abs(open_rets)) * 1e4
        conf_bp = np.mean(conf_rets) * 1e4
        edge = conf_bp - open_bp
        print(f"  {hour:>7} {len(conf_rets):>5} {open_bp:>+9.1f} "
              f"{conf_bp:>+8.1f} {edge:>+9.1f} {np.mean(conf_hits)*100:>8.1f}%")
    print(_bar())
    print("  READ: btcConf% > openEntry% AND conf-hit% > 55 => waiting for a BTC")
    print("  M15 confirmation gives a better ETH entry than firing at the hour open.")


def main() -> None:
    m = _merged()
    print()
    print(_bar("="))
    print(f"  ETH BOT -- BTC->ETH LEAD-LAG @ M15  ({START_YEAR}+)")
    print(f"  Goal: a 15-45min BTC lead to TIME better ETH entries within kill zones")
    print(_bar("="))

    best_k = _section_leadlag(m)
    _section_beta(m, best_k)
    _section_impulse(m)
    _section_entry_timing(m, thr=0.003)

    print()
    print(_bar("="))
    print("  NEXT: if a 15-45min lead + hit%>=55 shows up, I'll add a BTC-confirm")
    print("  entry filter to the live signal (wait for BTC M15 push before ETH")
    print("  entry) and measure the win-rate / PnL lift. If still flat -> the lead")
    print("  is sub-15min and not reachable by an hourly-scan bot.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
