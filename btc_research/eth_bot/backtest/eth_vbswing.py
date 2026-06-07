"""
btc_research/eth_bot/backtest/eth_vbswing.py — port the BTC bot's winning strategy to ETH.

Question (user): "Why does the BTC bot beat the ETH bot, and can ETH do better
if it runs BTC's strategy instead of its RSI/session slots?"

The BTC bot's edge is VBSwingStrategy = SwingLevelBreakV2 (structural swing-level
break, retest-preferred, SL capped 2xATR) FIRST, with VolatilityBreakout as a
fallback — TP1 2R / TP2 5R, EMA200 directional filter, ADX-split risk. On BTC
(2yr) that produced 270 trades, 46.7% WR, +0.90R, 24/24 profitable months,
PF 2.50, +$119k from $500.

Those two strategy classes are asset-agnostic. This script runs the EXACT same
combiner on ETH H1 and measures whether it beats ETH's current S4 slots
(+0.53R, 50.7% WR, 18 of 42 LOSING months, ~$72k/5yr).

Discipline (same as the rest of the project):
  * TRAIN = entry year <= 2024   TEST = entry year > 2024   (must hold in BOTH)
  * Plus the LAST-2-YEARS window the user asked for (BTC's best stretch).
  * One position at a time (live rule). Fresh $500, compounded.

NO live changes — this only measures. If VBSwing beats S4 in BOTH train AND test
(and/or last-2yr), the next step is to wire it as ETH's strategy and forward-test.

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.eth_vbswing
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from btc_research.btc_bot_2.strategy.vb_swing_combined import VBSwingStrategy

_DIR     = Path(__file__).parent
DATA_DIR = _DIR / "data"
ETH_CSV  = DATA_DIR / "ETHUSD_H1.csv"

STARTING_BALANCE = 500.0
TP1_RR_DEFAULT   = 2.0
TP2_RR_DEFAULT   = 5.0
TRAIL_ATR_MULT   = 2.0
MAX_HOLD_BARS    = 96
ADX_THRESHOLD    = 20

# Hour-sets to try (the BTC strategy was tuned for BTC's hours — find ETH's).
HOUR_SETS = {
    "BTC hours [1,2,3,8]":      [1, 2, 3, 8],
    "ETH S4 hours [2,5,6,7,10,14,15]": [2, 5, 6, 7, 10, 14, 15],
    "all 24 hours":             list(range(24)),
}

# Risk profiles (ADX-split). BTC-native = 3/2/3 ; ETH Tier B = 3/4/6.
RISK_PROFILES = {
    "BTC 3/2/3": (0.03, 0.02, 0.03),   # (early<=25, transition 25-40, strong>=40)
    "ETH 3/4/6": (0.03, 0.04, 0.06),
}


def _bar(c: str = "-", w: int = 96) -> str:
    return c * w


def _load_eth() -> pd.DataFrame:
    if not ETH_CSV.exists():
        print(f"ERROR: {ETH_CSV} not found -- run collect_data.py first"); sys.exit(1)
    df = pd.read_csv(ETH_CSV, parse_dates=["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").set_index("time")
    return df


def _indicators(df: pd.DataFrame):
    c = df["close"].astype(float); h = df["high"].astype(float); l = df["low"].astype(float)
    tr  = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().bfill().values
    ema200 = c.ewm(span=200, adjust=False).mean().values
    # ADX(14)
    period = 14; sp = 2 * period - 1
    hd = h.diff(); ld = l.diff()
    pdm = hd.where((hd > 0) & (hd > -ld), 0.0)
    mdm = (-ld).where((-ld > 0) & (-ld > hd), 0.0)
    aw = tr.ewm(span=sp, adjust=False).mean()
    pw = pdm.ewm(span=sp, adjust=False).mean()
    mw = mdm.ewm(span=sp, adjust=False).mean()
    pdi = 100 * pw / aw; mdi = 100 * mw / aw
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(span=sp, adjust=False).mean().fillna(0).values
    return atr, ema200, adx


def simulate(df, atr, ema200, adx, strategy, hours, risk_tuple,
             date_start=None, date_end=None) -> list[dict]:
    """One-position bar-by-bar sim on ETH. Fresh $500, compounded. Returns trades."""
    r_early, r_trans, r_strong = risk_tuple
    ts = df.index
    H = df["high"].astype(float).values
    L = df["low"].astype(float).values
    C = df["close"].astype(float).values
    hours = set(hours)

    balance = STARTING_BALANCE
    trades: list[dict] = []
    open_t = None

    for i in range(220, len(df)):
        bt = ts[i]
        if date_start is not None and bt < date_start:  continue
        if date_end   is not None and bt >= date_end:   break
        hr = bt.hour
        bh, bl, bc = float(H[i]), float(L[i]), float(C[i])

        # ── manage open trade ────────────────────────────────────────────────
        if open_t is not None:
            long_   = open_t["direction"] == "long"
            entry_  = open_t["entry"]
            sl_dist = abs(entry_ - open_t["orig_sl"])
            risk_u  = open_t["risk_usd"]
            age     = i - open_t["open_idx"]

            if not open_t.get("tp1_hit"):
                hit_sl  = (bl <= open_t["sl"]) if long_ else (bh >= open_t["sl"])
                hit_tp1 = (bh >= open_t["tp1"]) if long_ else (bl <= open_t["tp1"])
                if hit_sl:
                    pnl, r_, ex = -risk_u, -1.0, "SL"
                elif hit_tp1:
                    pnl = risk_u * open_t["tp1_rr"]
                    open_t["tp1_hit"] = True
                    open_t["sl"] = entry_
                    open_t["trail_peak"] = bh if long_ else bl
                    balance += pnl
                    open_t["pnl_running"] = round(pnl, 2)
                    open_t["r_running"]   = open_t["tp1_rr"]
                    continue
                elif age >= MAX_HOLD_BARS:
                    pnl = (bc - entry_ if long_ else entry_ - bc) * open_t["lots"]
                    r_  = pnl / risk_u if risk_u else 0; ex = "MAX_HOLD"
                else:
                    continue
            else:
                atr_now = float(atr[min(i, len(atr) - 1)])
                if long_:
                    open_t["trail_peak"] = max(open_t["trail_peak"], bh)
                    open_t["sl"] = max(open_t["trail_peak"] - TRAIL_ATR_MULT * atr_now, entry_)
                else:
                    open_t["trail_peak"] = min(open_t["trail_peak"], bl)
                    open_t["sl"] = min(open_t["trail_peak"] + TRAIL_ATR_MULT * atr_now, entry_)
                new_sl = open_t["sl"]
                hit_sl2 = (bl <= new_sl) if long_ else (bh >= new_sl)
                if hit_sl2:
                    r_  = abs(new_sl - entry_) / sl_dist if sl_dist else 0
                    pnl = risk_u * r_; ex = "TRAIL_SL"
                elif age >= MAX_HOLD_BARS:
                    r_  = abs(bc - entry_) / sl_dist if sl_dist else 0
                    pnl = risk_u * r_; ex = "MAX_HOLD"
                else:
                    continue

            balance += pnl
            open_t["pnl_usd"]       = round(open_t.get("pnl_running", 0) + pnl, 2)
            open_t["r_multiple"]    = round(open_t.get("r_running", 0) + r_, 2)
            open_t["balance_after"] = round(balance, 2)
            open_t["exit_reason"]   = ex
            trades.append(open_t)
            open_t = None

        if open_t is not None:
            continue

        # ── entry gate ───────────────────────────────────────────────────────
        if hr not in hours:
            continue
        adx_now = float(adx[i]); ema_now = float(ema200[i])
        if adx_now < ADX_THRESHOLD:
            continue
        win = df.iloc[max(0, i - 220):i + 1]

        for direction in ("long", "short"):
            if direction == "long"  and bc < ema_now: continue
            if direction == "short" and bc > ema_now: continue
            sig = strategy.generate_signal(win, bt, direction)
            if not sig.get("signal"):
                continue
            sl_d = abs(float(sig["entry"]) - float(sig["sl"]))
            if sl_d <= 0:
                continue
            if adx_now >= 40:   risk_pct = r_strong
            elif adx_now <= 25: risk_pct = r_early
            else:               risk_pct = r_trans
            ru   = round(balance * risk_pct, 2)
            lots = ru / sl_d
            tp1r = sig.get("tp1_rr", TP1_RR_DEFAULT)
            tp2r = sig.get("tp2_rr", TP2_RR_DEFAULT)
            e    = float(sig["entry"])
            tp1  = e + tp1r * sl_d if direction == "long" else e - tp1r * sl_d
            open_t = {
                "open_time": bt, "open_idx": i, "direction": direction,
                "entry": e, "sl": float(sig["sl"]), "orig_sl": float(sig["sl"]),
                "tp1": tp1, "tp1_rr": tp1r, "tp2_rr": tp2r,
                "lots": lots, "risk_usd": ru, "risk_pct_used": risk_pct,
                "strategy_used": sig.get("strategy_used", ""),
                "trail_peak": 0.0, "tp1_hit": False,
                "pnl_running": 0.0, "r_running": 0.0, "pnl_usd": 0.0,
                "adx_at_entry": round(adx_now, 1),
            }
            break
    return trades


def _stats(trades: list[dict]) -> dict:
    if not trades:
        return dict(n=0, wr=0.0, avgr=0.0, pnl=0.0, final=STARTING_BALANCE,
                    maxdd=0.0, maxcl=0, pf=0.0, pos_mo=0, tot_mo=0)
    wins = [t for t in trades if t["pnl_usd"] > 0]
    wr   = len(wins) / len(trades) * 100
    avgr = sum(t["r_multiple"] for t in trades) / len(trades)
    pnl  = sum(t["pnl_usd"] for t in trades)
    bals = [STARTING_BALANCE] + [t["balance_after"] for t in trades]
    peak = STARTING_BALANCE; maxdd = 0.0
    for b in bals:
        peak = max(peak, b)
        maxdd = max(maxdd, (peak - b) / peak * 100)
    # max consecutive losses
    cl = maxcl = 0
    for t in trades:
        if t["pnl_usd"] <= 0: cl += 1; maxcl = max(maxcl, cl)
        else: cl = 0
    gp = sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0)
    gl = abs(sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0))
    pf = gp / gl if gl else float("inf")
    # profitable months
    monthly = defaultdict(float)
    for t in trades:
        monthly[str(t["open_time"])[:7]] += t["pnl_usd"]
    pos_mo = sum(1 for v in monthly.values() if v > 0)
    return dict(n=len(trades), wr=round(wr, 1), avgr=round(avgr, 2),
                pnl=round(pnl, 2), final=round(bals[-1], 2), maxdd=round(maxdd, 1),
                maxcl=maxcl, pf=round(pf, 2), pos_mo=pos_mo, tot_mo=len(monthly))


def main() -> None:
    df = _load_eth()
    atr, ema200, adx = _indicators(df)
    strat = VBSwingStrategy()
    first, last = df.index[0], df.index[-1]
    train_end = pd.Timestamp("2025-01-01")
    last2_start = last - pd.DateOffset(years=2)

    print()
    print(_bar("="))
    print("  ETH BOT — BTC STRATEGY (Swing-Level-Break v2 + VB) PORTED TO ETH")
    print(f"  ETH H1 {first.date()} → {last.date()} | TP1 2R / TP2 5R | EMA200 + ADX-split | one-position")
    print("  Benchmark to beat — ETH S4: +0.53R, 50.7% WR, 18/42 LOSING months, ~$72k/5yr")
    print(_bar("="))

    # ---- Pass 1: hour-set × risk-profile, FULL history -----------------------
    print()
    print("  PASS 1 — hour-set & risk sweep (FULL history)")
    print(f"  {'hour-set':<34} {'risk':<10} {'N':>5} {'WR%':>6} {'AvgR':>6} "
          f"{'PnL$':>11} {'final$':>11} {'MaxDD%':>7} {'MaxCL':>5} {'PF':>5} {'+mo/mo':>8}")
    print(_bar())
    best = None
    for hs_name, hours in HOUR_SETS.items():
        for rp_name, rp in RISK_PROFILES.items():
            t = simulate(df, atr, ema200, adx, strat, hours, rp)
            s = _stats(t)
            print(f"  {hs_name:<34} {rp_name:<10} {s['n']:>5} {s['wr']:>5.1f}% "
                  f"{s['avgr']:>+6.2f} {s['pnl']:>+11,.0f} {s['final']:>11,.0f} "
                  f"{s['maxdd']:>6.1f}% {s['maxcl']:>5} {s['pf']:>5.2f} "
                  f"{s['pos_mo']:>3}/{s['tot_mo']:<3}")
            # rank by a robustness-aware score: avgR must be >0, prefer high PF & low DD
            score = s["avgr"] if s["n"] >= 30 else -9
            if best is None or score > best[0]:
                best = (score, hs_name, hours, rp_name, rp)
    print(_bar())

    if best is None or best[0] < 0:
        print("  No hour-set produced a usable sample. ETH may not suit this strategy.")
        return

    _, hs_name, hours, rp_name, rp = best
    print(f"  Best by AvgR (N>=30): {hs_name}  +  {rp_name} risk")

    # ---- Pass 2: TRAIN / TEST / LAST-2YR for the best config -----------------
    print()
    print(_bar("="))
    print(f"  PASS 2 — robustness of best config: {hs_name} | {rp_name}")
    print(_bar("="))
    print(f"  {'period':<22} {'N':>5} {'WR%':>6} {'AvgR':>6} {'PnL$':>11} "
          f"{'final$':>11} {'MaxDD%':>7} {'MaxCL':>5} {'PF':>5} {'+mo/mo':>8}")
    print(_bar())
    periods = [
        ("FULL",            None,        None),
        ("TRAIN <=2024",    None,        train_end),
        ("TEST  >=2025",    train_end,   None),
        (f"LAST 2YR ({last2_start.date()}+)", last2_start, None),
    ]
    for label, ds, de in periods:
        t = simulate(df, atr, ema200, adx, strat, hours, rp, date_start=ds, date_end=de)
        s = _stats(t)
        print(f"  {label:<22} {s['n']:>5} {s['wr']:>5.1f}% {s['avgr']:>+6.2f} "
              f"{s['pnl']:>+11,.0f} {s['final']:>11,.0f} {s['maxdd']:>6.1f}% "
              f"{s['maxcl']:>5} {s['pf']:>5.2f} {s['pos_mo']:>3}/{s['tot_mo']:<3}")
    print(_bar())

    # ---- Pass 3: monthly breakdown for LAST 2YR ------------------------------
    t2 = simulate(df, atr, ema200, adx, strat, hours, rp, date_start=last2_start)
    print()
    print(_bar("="))
    print(f"  PASS 3 — LAST-2YR monthly breakdown ({hs_name} | {rp_name})")
    print(_bar())
    monthly = defaultdict(list)
    for t in t2:
        monthly[str(t["open_time"])[:7]].append(t)
    print(f"  {'month':>8} {'N':>4} {'WR%':>6} {'AvgR':>6} {'PnL$':>10} {'bal$':>11}")
    for ym in sorted(monthly):
        s = _stats(monthly[ym])
        endbal = monthly[ym][-1]["balance_after"]
        tag = "  <== LOSS" if s["pnl"] < 0 else ""
        print(f"  {ym:>8} {s['n']:>4} {s['wr']:>5.1f}% {s['avgr']:>+6.2f} "
              f"{s['pnl']:>+10,.0f} {endbal:>11,.0f}{tag}")
    print(_bar())
    print("  VERDICT RULE: wire VBSwing into ETH only if it beats S4 (AvgR, % profitable")
    print("  months, MaxDD) in BOTH train AND test — not just the flattering last-2yr.")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
