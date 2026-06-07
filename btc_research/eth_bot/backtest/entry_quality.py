"""
btc_research/eth_bot/backtest/entry_quality.py

WALK-FORWARD ENTRY-QUALITY / MAE-MFE STUDY  (perfect the entry)

Replays the trades the LIVE bot would actually take (S4+9, one-position-at-a-
time rule) in series, and for each one walks the H1 bars AFTER entry to measure
how price behaved once we were in:

  * MAE  (Maximum Adverse Excursion)   -- worst move AGAINST us, in R ("heat")
  * MFE  (Maximum Favorable Excursion)  -- best move OUR WAY, in R
  * heat-before-TP1                      -- how much it dipped before reaching +2R
  * adverse-first?                       -- did it dip toward SL before pulling up?

The question this answers: do our winners go straight to TP, or do they first
dive toward the stop before reversing? If winners routinely take, say, -0.6R of
heat first, we are entering too early and a small PULLBACK ENTRY would get a
better fill -- so Section 5 simulates resting a limit entry 0.1-0.5R better and
measures the trade-off (better fills vs. the runners a limit order MISSES).

NO lookahead: every excursion is measured strictly on bars at/after the entry
bar; the pullback test only fills when price trades to the limit within a small
window AFTER the signal.

== USAGE ==
  # run AFTER run_backtest.py has regenerated backtest_trades.csv (TP1 fix)
  # and collect_data.py has the ETHUSD_H1.csv path file.
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.entry_quality
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
_DIR    = Path(__file__).parent
CSV     = _DIR / "data" / "backtest_trades.csv"
ETH_CSV = _DIR / "data" / "ETHUSD_H1.csv"

# -- Fixed params (match live / run_backtest) ----------------------------------
START_YEAR     = 2023
TP1_RR         = 2.0
TP2_RR         = 4.0
TRAIL_ATR_MULT = 2.0
MAX_HOLD_BARS  = 96

# -- Pullback-entry test knobs -------------------------------------------------
PULLBACKS      = [0.1, 0.2, 0.3, 0.5]   # R better than market entry
PULLBACK_WINDOW = 3                       # bars after signal a limit may fill in

# -- Slot set: S4 + 9 complement (the live winner) -----------------------------
S4_SLOTS = [
    ("rsi_50",    2,  True), ("macd_adx",  6,  True), ("rsi_ema", 10, True),
    ("ema_cross", 5,  False), ("rsi_50",    7,  False), ("keltner", 14, False),
    ("ema_cross",14,  False), ("keltner",  15,  False),
]
COMPLEMENT_9 = [
    ("engulfing",16,False), ("engulfing", 1,False), ("pin_bar",  2,False),
    ("engulfing", 8,False), ("engulfing", 2,False), ("engulfing",5,False),
    ("pin_bar",  19,False), ("pin_bar",  13,False), ("pin_bar", 11,False),
]


def _bar(c: str = "-", w: int = 96) -> str:
    return c * w


# ── Data loading ─────────────────────────────────────────────────────────────
def _load_trades() -> pd.DataFrame:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found -- run run_backtest.py first")
        sys.exit(1)
    df = pd.read_csv(CSV, parse_dates=["entry_time", "exit_time"])
    for c in ("entry_time", "exit_time"):
        if df[c].dt.tz is not None:
            df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[df["entry_time"].dt.year >= START_YEAR].reset_index(drop=True)
    return df


def _load_eth() -> pd.DataFrame:
    if not ETH_CSV.exists():
        print(f"ERROR: {ETH_CSV} not found -- run collect_data.py first")
        sys.exit(1)
    df = pd.read_csv(ETH_CSV, parse_dates=["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("time").reset_index(drop=True)


def _select(df: pd.DataFrame, slots) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in slots:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return df[mask].sort_values("entry_time").reset_index(drop=True)


def _one_position(cand: pd.DataFrame) -> pd.DataFrame:
    """Keep only trades the live one-position rule would actually take."""
    last_exit = pd.Timestamp.min
    keep = []
    for idx, t in cand.iterrows():
        if t["entry_time"] < last_exit:
            continue
        keep.append(idx)
        last_exit = t["exit_time"]
    return cand.loc[keep].reset_index(drop=True)


# ── Trade-path simulation (mirrors run_backtest._simulate_trade) ──────────────
def _simulate(highs, lows, closes, fill_i, entry, sl_dist, is_long, atr):
    """Return realized R from a fill at bar fill_i (entry price = entry)."""
    sgn = 1.0 if is_long else -1.0
    tp1 = entry + sl_dist * TP1_RR * sgn
    tp2 = entry + sl_dist * TP2_RR * sgn
    sl  = entry - sl_dist * sgn
    cur_sl = sl
    tp1_hit = False
    realized = 0.0
    n = len(closes)
    for j in range(fill_i + 1, min(fill_i + MAX_HOLD_BARS + 1, n)):
        bh, bl, bc = highs[j], lows[j], closes[j]
        if tp1_hit and atr > 0:
            trail = TRAIL_ATR_MULT * atr
            if is_long:
                cur_sl = max(cur_sl, bc - trail)
            else:
                cur_sl = min(cur_sl, bc + trail)
        hit_tp2 = (bh >= tp2) if is_long else (bl <= tp2)
        if hit_tp2:
            if not tp1_hit:
                realized += 0.5 * TP1_RR
            realized += 0.5 * TP2_RR
            return realized
        hit_sl = (bl <= cur_sl) if is_long else (bh >= cur_sl)
        if hit_sl:
            sl_r = (cur_sl - entry) / sl_dist * sgn
            realized += (0.5 if tp1_hit else 1.0) * sl_r
            return realized
        if not tp1_hit:
            hit_tp1 = (bh >= tp1) if is_long else (bl <= tp1)
            if hit_tp1:
                tp1_hit = True
                realized += 0.5 * TP1_RR
                cur_sl = entry
    last_j = min(fill_i + MAX_HOLD_BARS, n - 1)
    mark_r = (closes[last_j] - entry) / sl_dist * sgn
    realized += (0.5 if tp1_hit else 1.0) * mark_r
    return realized


# ── Analysis ─────────────────────────────────────────────────────────────────
def main() -> None:
    trades = _load_trades()
    eth = _load_eth()
    taken = _one_position(_select(trades, S4_SLOTS + COMPLEMENT_9))

    # index ETH bars by timestamp for O(1) entry lookup
    t2i = {t: i for i, t in enumerate(eth["time"])}
    highs = eth["high"].to_numpy()
    lows  = eth["low"].to_numpy()
    closes = eth["close"].to_numpy()
    n = len(eth)

    rows = []
    for _, t in taken.iterrows():
        ei = t2i.get(t["entry_time"])
        if ei is None:
            continue
        entry   = float(t["entry"])
        sl_dist = float(t["sl_dist"])
        if sl_dist <= 0:
            continue
        is_long = (t["direction"] == "long")
        sgn = 1.0 if is_long else -1.0
        held = int(t["bars_held"]) if not pd.isna(t["bars_held"]) else MAX_HOLD_BARS
        last = min(ei + max(held, 1), n - 1)

        mae = mfe = 0.0
        heat_before_tp1 = 0.0
        tp1_seen = False
        adv_first = None
        for j in range(ei + 1, last + 1):
            adv = (entry - lows[j]) / sl_dist if is_long else (highs[j] - entry) / sl_dist
            fav = (highs[j] - entry) / sl_dist if is_long else (entry - lows[j]) / sl_dist
            adv = max(adv, 0.0); fav = max(fav, 0.0)
            mae = max(mae, adv); mfe = max(mfe, fav)
            if not tp1_seen:
                heat_before_tp1 = max(heat_before_tp1, adv)
                if fav >= TP1_RR:
                    tp1_seen = True
            if adv_first is None:
                if adv >= 0.5 and fav < 0.5:
                    adv_first = True
                elif fav >= 0.5 and adv < 0.5:
                    adv_first = False
        rows.append({
            "r": float(t["r_achieved"]), "win": float(t["r_achieved"]) > 0,
            "mae": mae, "mfe": mfe, "heat_tp1": heat_before_tp1,
            "adv_first": adv_first,
        })

    R = pd.DataFrame(rows)
    if R.empty:
        print("No trades mapped to ETH bars -- check the H1 file covers the trade dates.")
        sys.exit(1)

    win = R[R["win"]]
    los = R[~R["win"]]

    print()
    print(_bar("="))
    print(f"  ETH BOT -- ENTRY QUALITY / MAE-MFE  (S4+9, one-position, {START_YEAR}+)")
    print(f"  {len(R)} trades replayed bar-by-bar  |  winners {len(win)}  losers {len(los)}")
    print(_bar("="))

    # 1. how price moves after entry -------------------------------------------
    print("  1. EXCURSION AFTER ENTRY (in R; MAE=heat against us, MFE=best in favour)")
    print(_bar())
    print(f"  {'group':<10} {'N':>5} {'medMAE':>8} {'avgMAE':>8} {'medMFE':>8} {'avgMFE':>8}")
    for label, g in (("ALL", R), ("winners", win), ("losers", los)):
        if g.empty:
            continue
        print(f"  {label:<10} {len(g):>5} {g['mae'].median():>8.2f} {g['mae'].mean():>8.2f} "
              f"{g['mfe'].median():>8.2f} {g['mfe'].mean():>8.2f}")
    print(_bar())

    # 2. adverse-first? --------------------------------------------------------
    known = R[R["adv_first"].notna()]
    if not known.empty:
        af = known["adv_first"].mean() * 100
        print(f"  2. DIRECTION FIRST: {af:.1f}% of trades hit 0.5R AGAINST us before "
              f"0.5R in favour")
        print(f"     -> {'price usually dips toward SL first' if af > 55 else 'no strong dip-first bias'}")
        print(_bar())

    # 3. heat before TP1 (did winners nearly get stopped first?) ----------------
    print("  3. HEAT BEFORE TP1 (winners only -- how deep they dipped before +2R)")
    if not win.empty:
        for thr in (0.3, 0.5, 0.8, 1.0):
            pct = (win["heat_tp1"] >= thr).mean() * 100
            print(f"     winners that first dipped >= {thr:>3.1f}R : {pct:>5.1f}%")
        print(f"     median heat-before-TP1 among winners      : {win['heat_tp1'].median():.2f}R")
    print(_bar())

    # 4. loser rescue potential (did losers show profit first?) -----------------
    print("  4. LOSER MFE (did losing trades show profit we could have banked?)")
    if not los.empty:
        for thr in (1.0, 1.5, 2.0):
            pct = (los["mfe"] >= thr).mean() * 100
            print(f"     losers that reached >= {thr:>3.1f}R in favour first : {pct:>5.1f}%")
        print(f"     median MFE among losers                       : {los['mfe'].median():.2f}R")
    print(_bar())

    # 5. pullback-entry simulation (perfect the fill) ---------------------------
    print("  5. PULLBACK ENTRY TEST  (rest a limit p*R better; fill within "
          f"{PULLBACK_WINDOW} bars)")
    print(_bar())
    base_avg = R["r"].mean()
    print(f"  baseline market entry: N={len(R)}  avgR={base_avg:>+6.3f}  "
          f"totR={R['r'].sum():>+8.1f}")
    print(_bar("."))
    print(f"  {'pullback':>9} {'fill%':>6} {'filledAvgR':>11} {'allAvgR(miss=0)':>16} "
          f"{'totR':>8}")
    for p in PULLBACKS:
        fills = 0
        filled_r = []
        for _, t in taken.iterrows():
            ei = t2i.get(t["entry_time"])
            if ei is None:
                continue
            entry = float(t["entry"]); sl_dist = float(t["sl_dist"])
            if sl_dist <= 0:
                continue
            is_long = (t["direction"] == "long"); sgn = 1.0 if is_long else -1.0
            atr = float(t["atr"]) if not pd.isna(t["atr"]) else 0.0
            limit_px = entry - sl_dist * p * sgn
            # scan window for a fill, but stop if trade already ran to TP1 in our favour
            fill_i = None
            for j in range(ei + 1, min(ei + 1 + PULLBACK_WINDOW, n)):
                fav = (highs[j] - entry) / sl_dist if is_long else (entry - lows[j]) / sl_dist
                if fav >= TP1_RR:
                    break  # ran away before pulling back -> missed
                touched = (lows[j] <= limit_px) if is_long else (highs[j] >= limit_px)
                if touched:
                    fill_i = j
                    break
            if fill_i is None:
                continue
            fills += 1
            r = _simulate(highs, lows, closes, fill_i, limit_px, sl_dist, is_long, atr)
            filled_r.append(r)
        if fills == 0:
            print(f"  {p:>8.1f}R {0.0:>5.1f}% {'--':>11} {'--':>16} {0.0:>8.1f}")
            continue
        fr = np.array(filled_r)
        fill_rate = fills / len(taken) * 100
        all_avg = fr.sum() / len(taken)   # misses contribute 0R
        print(f"  {p:>8.1f}R {fill_rate:>5.1f}% {fr.mean():>+10.3f} "
              f"{all_avg:>+15.3f} {fr.sum():>+8.1f}")
    print(_bar())
    print("  READ: 'filledAvgR' > baseline means a pullback gets a better price on the")
    print("  trades it catches. 'allAvgR(miss=0)' counts the runners a limit MISSES -- if")
    print("  that drops below baseline, waiting for a pullback costs more than it saves.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
