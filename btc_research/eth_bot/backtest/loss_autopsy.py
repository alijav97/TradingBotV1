"""
btc_research/eth_bot/backtest/loss_autopsy.py

Forensic analysis of LOSING trades for the S4 slot set (2023+).

Goal: instead of dropping a weak strategy, understand WHAT CONDITION at entry
makes its trades lose -- so we can tighten the entry rule. Any condition that
separates winners from losers in one slot usually helps the others too, because
every slot shares the same trend/breakout failure mode (chop).

Only entry-time-known fields are used as candidate FILTERS:
    adx, atr, direction, distance-from-EMA200, SL/ATR ratio, btc_aligned.
Exit-only fields (bars_held, exit_reason) are shown for DIAGNOSIS only -- they
cannot be used as live filters because they aren't known when entering.

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.loss_autopsy
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR  = Path(__file__).parent
DATA  = _DIR / "data"
CSV   = DATA / "backtest_trades.csv"

# -- Parameters ----------------------------------------------------------------
START_YEAR = 2023

# S4 slot set (strategy, hour, require_btc_aligned)
S4_SLOTS = [
    ("rsi_50",    2,  True),
    ("macd_adx",  6,  True),
    ("rsi_ema",  10,  True),
    ("ema_cross", 5,  False),
    ("rsi_50",    7,  False),
    ("keltner",  14,  False),
    ("ema_cross",14,  False),
    ("keltner",  15,  False),
]

# Candidate entry filters: name -> predicate(dataframe) -> boolean Series
CANDIDATE_FILTERS = {
    "ADX>=25":          lambda d: d["adx"] >= 25,
    "ADX>=30":          lambda d: d["adx"] >= 30,
    "ADX 25-40":        lambda d: (d["adx"] >= 25) & (d["adx"] <= 40),
    "long only":        lambda d: d["direction"] == "long",
    "short only":       lambda d: d["direction"] == "short",
    "ext<=2%":          lambda d: d["ext_pct"] <= 2.0,
    "ext<=3%":          lambda d: d["ext_pct"] <= 3.0,
    "ext>=1%":          lambda d: d["ext_pct"] >= 1.0,
    "sl_atr<=1.5":      lambda d: d["sl_atr"] <= 1.5,
    "sl_atr 0.8-2.5":   lambda d: (d["sl_atr"] >= 0.8) & (d["sl_atr"] <= 2.5),
    "btc_aligned":      lambda d: d["btc_aligned"] == True,
}

# Slots considered "weak" -> get the full filter scan
WEAK_SLOTS = {"ema_cross[05]", "ema_cross[14]", "keltner[15]", "macd_adx[06]", "rsi_50[07]"}


def _bar(c: str = "-", w: int = 96) -> str:
    return c * w


def _filter_slots(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in S4_SLOTS:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return df[mask].copy()


def _fingerprint(sub: pd.DataFrame) -> dict:
    """Return mean entry-conditions for a trade subset."""
    if sub.empty:
        return {}
    return {
        "n":        len(sub),
        "wr":       (sub["outcome"] == "win").mean() * 100,
        "avg_r":    sub["r_achieved"].mean(),
        "adx":      sub["adx"].mean(),
        "ext":      sub["ext_pct"].mean(),
        "sl_atr":   sub["sl_atr"].mean(),
        "long_pct": (sub["direction"] == "long").mean() * 100,
        "btc_pct":  (sub["btc_aligned"] == True).mean() * 100,
        "bars":     sub["bars_held"].mean(),
    }


def _print_winloss(label: str, sub: pd.DataFrame) -> None:
    win  = sub[sub["outcome"] == "win"]
    loss = sub[sub["outcome"] == "loss"]
    fw, fl = _fingerprint(win), _fingerprint(loss)
    if not fw or not fl:
        print(f"  {label:<16}  (need both wins and losses; "
              f"W={len(win)} L={len(loss)})")
        return
    print(f"  {label:<16}  {'N':>4}  {'AvgR':>6}  {'ADX':>5}  {'ext%':>6}  "
          f"{'sl/atr':>6}  {'long%':>6}  {'bars':>5}")
    print(f"    {'WIN':<14}  {fw['n']:>4}  {fw['avg_r']:>+6.2f}  {fw['adx']:>5.1f}  "
          f"{fw['ext']:>+6.2f}  {fw['sl_atr']:>6.2f}  {fw['long_pct']:>5.0f}%  {fw['bars']:>5.0f}")
    print(f"    {'LOSS':<14}  {fl['n']:>4}  {fl['avg_r']:>+6.2f}  {fl['adx']:>5.1f}  "
          f"{fl['ext']:>+6.2f}  {fl['sl_atr']:>6.2f}  {fl['long_pct']:>5.0f}%  {fl['bars']:>5.0f}")


def _exit_mix(sub: pd.DataFrame) -> str:
    vc = sub["exit_reason"].value_counts()
    return "  ".join(f"{k}={v}" for k, v in vc.items())


def _filter_scan(slot: str, sub: pd.DataFrame) -> None:
    """For one slot, rank candidate filters by the AvgR of what they DISCARD."""
    base_n   = len(sub)
    base_wr  = (sub["outcome"] == "win").mean() * 100
    base_avg = sub["r_achieved"].mean()
    base_tot = sub["r_achieved"].sum()
    print(f"  {slot}   baseline:  N={base_n}  WR={base_wr:.1f}%  "
          f"AvgR={base_avg:+.3f}  TotR={base_tot:+.2f}")
    print(f"    exits: {_exit_mix(sub)}")
    print(f"    {'filter':<16}  {'cut':>4}  {'cutAvgR':>8}  {'keptN':>6}  "
          f"{'keptWR':>7}  {'keptAvgR':>9}  {'keptTotR':>9}")

    rows = []
    for fname, pred in CANDIDATE_FILTERS.items():
        # skip btc_aligned filter for baseline slots (already all aligned)
        keep = pred(sub)
        passed  = sub[keep]
        removed = sub[~keep]
        if len(removed) < 2 or len(passed) < 2:
            continue
        cut_avg  = removed["r_achieved"].mean()
        kept_avg = passed["r_achieved"].mean()
        # only interesting if filter improves the kept edge
        if kept_avg <= base_avg:
            continue
        rows.append((fname, len(removed), cut_avg, len(passed),
                     (passed["outcome"] == "win").mean() * 100,
                     kept_avg, passed["r_achieved"].sum()))

    # rank by most-negative discarded AvgR (best = cutting the worst trades)
    rows.sort(key=lambda r: r[2])
    if not rows:
        print("    (no single filter improves kept AvgR)")
    for fname, cut_n, cut_avg, kept_n, kept_wr, kept_avg, kept_tot in rows[:5]:
        print(f"    {fname:<16}  {cut_n:>4}  {cut_avg:>+8.3f}  {kept_n:>6}  "
              f"{kept_wr:>6.1f}%  {kept_avg:>+9.3f}  {kept_tot:>+9.2f}")


# ==============================================================================
def main() -> None:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first")
        sys.exit(1)

    df = pd.read_csv(CSV, parse_dates=["entry_time"])
    df = df[df["entry_time"].dt.year >= START_YEAR]
    df = _filter_slots(df).reset_index(drop=True)
    if df.empty:
        print(f"ERROR: no S4 trades from {START_YEAR}+")
        sys.exit(1)

    # Derived entry features
    df["ext_pct"]  = (df["entry"] - df["ema200"]).abs() / df["ema200"] * 100
    df["sl_atr"]   = df["sl_dist"] / df["atr"]
    df["slot"]     = df["strategy"] + "[" + df["hour_utc"].map(lambda h: f"{int(h):02d}") + "]"
    df["month"]    = df["entry_time"].dt.to_period("M").astype(str)

    print()
    print(_bar("="))
    print(f"  ETH BOT -- LOSING-TRADE AUTOPSY  (S4 slots, {START_YEAR}+)")
    print(f"  {len(df)} trades  |  WR={ (df['outcome']=='win').mean()*100:.1f}%  "
          f"|  AvgR={df['r_achieved'].mean():+.3f}")
    print("  ext% = |entry-EMA200|/EMA200 (extension)  |  sl/atr = SL distance in ATRs")
    print(_bar("="))

    # ---- PART 1: pooled winner vs loser fingerprint --------------------------
    print()
    print(_bar("="))
    print("  1. WINNER vs LOSER FINGERPRINT  (all S4 trades pooled)")
    print(_bar())
    _print_winloss("ALL SLOTS", df)
    print()
    print(f"  Loser exit mix : {_exit_mix(df[df['outcome']=='loss'])}")
    print(f"  Winner exit mix: {_exit_mix(df[df['outcome']=='win'])}")

    # ---- PART 2: per-slot winner vs loser ------------------------------------
    print()
    print(_bar("="))
    print("  2. PER-SLOT WINNER vs LOSER FINGERPRINT")
    print(_bar())
    for slot in sorted(df["slot"].unique()):
        _print_winloss(slot, df[df["slot"] == slot])
        print()

    # ---- PART 3: ema_cross[05] full trade dump -------------------------------
    print(_bar("="))
    print("  3. ema_cross[05] -- EVERY TRADE  (the only negative-edge slot)")
    print(_bar())
    ec5 = df[df["slot"] == "ema_cross[05]"].sort_values("entry_time")
    if ec5.empty:
        print("  (no ema_cross[05] trades)")
    else:
        print(f"  {'date':<11}  {'dir':<5}  {'adx':>5}  {'ext%':>6}  {'sl/atr':>6}  "
              f"{'btc':>4}  {'exit':<12}  {'R':>6}  {'bars':>4}")
        for _, t in ec5.iterrows():
            print(f"  {t['entry_time'].strftime('%Y-%m-%d'):<11}  {t['direction']:<5}  "
                  f"{t['adx']:>5.1f}  {t['ext_pct']:>6.2f}  {t['sl_atr']:>6.2f}  "
                  f"{str(bool(t['btc_aligned']))[:1]:>4}  {t['exit_reason']:<12}  "
                  f"{t['r_achieved']:>+6.2f}  {int(t['bars_held']):>4}")

    # ---- PART 4: candidate-filter scan for weak slots ------------------------
    print()
    print(_bar("="))
    print("  4. CANDIDATE-FILTER IMPACT  (weak slots -- which condition to add?)")
    print("     'cut' = trades the filter removes | 'cutAvgR' = their avg R (want very negative)")
    print(_bar())
    for slot in sorted(df["slot"].unique()):
        if slot not in WEAK_SLOTS:
            continue
        _filter_scan(slot, df[df["slot"] == slot])
        print()

    # ---- PART 5: worst-month trade dump --------------------------------------
    print(_bar("="))
    print("  5. WORST MONTHS (WR < 35%) -- what did the losers look like?")
    print(_bar())
    mwr = df.groupby("month").agg(
        n=("outcome", "size"),
        wins=("outcome", lambda s: (s == "win").sum()),
    )
    mwr["wr"] = mwr["wins"] / mwr["n"] * 100
    bad_months = mwr[(mwr["wr"] < 35) & (mwr["n"] >= 3)].sort_values("wr")
    for month in bad_months.index:
        grp = df[df["month"] == month].sort_values("entry_time")
        n, wr = len(grp), (grp["outcome"] == "win").mean() * 100
        print(f"  {month}  N={n}  WR={wr:.0f}%  "
              f"avgADX={grp['adx'].mean():.1f}  avgExt={grp['ext_pct'].mean():.2f}%  "
              f"long={(grp['direction']=='long').mean()*100:.0f}%  "
              f"exits[{_exit_mix(grp)}]")
        for _, t in grp.iterrows():
            print(f"      {t['slot']:<14} {t['direction']:<5} adx={t['adx']:>5.1f} "
                  f"ext={t['ext_pct']:>5.2f}% sl/atr={t['sl_atr']:>4.2f} "
                  f"-> {t['exit_reason']:<12} R={t['r_achieved']:>+5.2f}")
        print()

    print(_bar("="))
    print("  Done. Use PART 4 to pick entry filters, then re-validate via backtest_optimised.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
