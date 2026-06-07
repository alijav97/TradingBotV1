"""
btc_research/eth_bot/backtest/trade_autopsy.py — stick-by-stick post-mortem.

For the LOSING months of a chosen year (auto-detected from the one-position
fresh-$500 sim, exactly as forward_test counts them), walk EVERY trade the bot
actually took, candle by candle, and show:

  * WHY the signal fired   (strategy, direction, ADX, ATR, entry/SL/TP levels)
  * WHERE price went after  (each H1 bar's best/worst excursion in R, with the
                             bar where SL / TP1 / TP2 was first touched)
  * the EXIT and realised R, plus a plain-English one-line story.

R convention (per trade): 0R = entry, -1R = stop, +2R = TP1, +4R = TP2.
  long  : R_high = (high-entry)/sl_dist   R_low = (low-entry)/sl_dist
  short : R_high = (entry-low)/sl_dist     R_low = (entry-high)/sl_dist

This is forensic only — it changes nothing. It tells you whether the losers
went straight to stop, or ran toward target and reversed (a TP/exit problem),
or chopped sideways into a time-stop.

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.trade_autopsy
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_DIR     = Path(__file__).parent
DATA_DIR = _DIR / "data"
CSV      = DATA_DIR / "backtest_trades.csv"
ETH_CSV  = DATA_DIR / "ETHUSD_H1.csv"

# -- What to autopsy -----------------------------------------------------------
YEAR             = 2026          # autopsy the losing months of this year
MAX_BARS_SHOWN   = 72            # cap candle rows per trade (<= MAX_HOLD 96)
SHOW_ALL_MONTHS  = False         # True = autopsy every month, not just losers

# -- Sim params (match forward_test / live) ------------------------------------
STARTING_BALANCE = 500.0
OCT_RISK_FACTOR  = 0.5
CB_THRESHOLD     = -0.10
RISK_EARLY, RISK_TRANS, RISK_STRONG = 0.030, 0.040, 0.060
DD_THROTTLE_TRIGGER = -0.25
DD_THROTTLE_FACTOR  = 0.50

S4_SLOTS = [
    ("rsi_50",    2,  True), ("macd_adx",  6,  True), ("rsi_ema", 10, True),
    ("ema_cross", 5,  False), ("rsi_50",    7,  False), ("keltner", 14, False),
    ("ema_cross",14,  False), ("keltner",  15,  False),
]


def _bar(c: str = "-", w: int = 86) -> str:
    return c * w


def _risk_pct(adx: float) -> float:
    if adx >= 40:
        return RISK_STRONG
    if adx <= 25:
        return RISK_EARLY
    return RISK_TRANS


def _select(df: pd.DataFrame, slots) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in slots:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m
    return df[mask].sort_values("entry_time").reset_index(drop=True)


def _take_onepos(cand: pd.DataFrame) -> pd.DataFrame:
    """One-position fresh-$500 sim. Returns the TAKEN trades (full rows) with
    added pnl / balance, exactly the set forward_test counts."""
    balance = STARTING_BALANCE
    peak = balance
    last_exit = pd.Timestamp.min
    cur_month = None
    month_start_bal = balance
    month_halted = False
    taken = []

    for _, t in cand.iterrows():
        et, xt = t["entry_time"], t["exit_time"]
        mk = (et.year, et.month)
        if mk != cur_month:
            cur_month = mk
            month_start_bal = balance
            month_halted = False
        if et < last_exit or month_halted:
            continue
        rp = _risk_pct(float(t["adx"]))
        if et.month == 10:
            rp *= OCT_RISK_FACTOR
        if (balance - peak) / peak <= DD_THROTTLE_TRIGGER:
            rp *= DD_THROTTLE_FACTOR
        pnl = balance * rp * float(t["r_achieved"])
        balance += pnl
        peak = max(peak, balance)
        last_exit = xt
        row = t.to_dict()
        row["pnl"] = pnl
        row["balance"] = balance
        row["risk_pct"] = rp
        taken.append(row)
        if (balance - month_start_bal) / month_start_bal <= CB_THRESHOLD:
            month_halted = True
    return pd.DataFrame(taken)


def _walk(trade: dict, eth_idx: pd.DataFrame) -> None:
    """Print the candle-by-candle path of one trade in R units."""
    et   = trade["entry_time"]
    xt   = trade["exit_time"]
    entry = float(trade["entry"]); sl = float(trade["sl"])
    sl_dist = abs(entry - sl) or 1e-9
    is_long = str(trade["direction"]).lower() == "long"
    tp1 = float(trade.get("tp1", entry + (2 if is_long else -2) * sl_dist))
    tp2 = float(trade.get("tp2", entry + (4 if is_long else -4) * sl_dist))

    print(_bar("="))
    print(f"  {str(et)}  {str(trade['direction']).upper():<5}  {trade['strategy']:<10}  "
          f"H{int(trade['hour_utc']):02d}   risk {trade['risk_pct']*100:.1f}%   "
          f"ADX {float(trade['adx']):.1f}  ATR {float(trade['atr']):.2f}")
    print(f"  entry {entry:.2f}  SL {sl:.2f} (-1R, ${sl_dist:.2f})  "
          f"TP1 {tp1:.2f} (+2R)  TP2 {tp2:.2f} (+4R)   btc_aligned={trade.get('btc_aligned')}")
    print(f"  exit_reason {trade['exit_reason']}  exit {float(trade['exit_price']):.2f}  "
          f"r_achieved {float(trade['r_achieved']):+.2f}  bars_held {int(trade['bars_held'])}  "
          f"P&L ${trade['pnl']:+.2f} -> bal ${trade['balance']:.2f}")

    path = eth_idx[(eth_idx.index > et) & (eth_idx.index <= xt)]
    if path.empty:
        print("  (no candle data for this window)")
        return
    print(f"  {'bar':>3} {'time':<19} {'open':>9} {'high':>9} {'low':>9} {'close':>9} "
          f"{'R_best':>7} {'R_worst':>8}  event")
    mfe = -99.0; mae = 99.0; mfe_bar = mae_bar = 0
    hit_sl = hit_tp1 = hit_tp2 = None
    for i, (ts, c) in enumerate(path.iterrows(), start=1):
        hi, lo = float(c["high"]), float(c["low"])
        if is_long:
            r_hi = (hi - entry) / sl_dist; r_lo = (lo - entry) / sl_dist
        else:
            r_hi = (entry - lo) / sl_dist; r_lo = (entry - hi) / sl_dist
        if r_hi > mfe: mfe, mfe_bar = r_hi, i
        if r_lo < mae: mae, mae_bar = r_lo, i
        ev = []
        if hit_sl is None and r_lo <= -1.0: hit_sl = i; ev.append("SL touched")
        if hit_tp1 is None and r_hi >= 2.0: hit_tp1 = i; ev.append("TP1 touched")
        if hit_tp2 is None and r_hi >= 4.0: hit_tp2 = i; ev.append("TP2 touched")
        if i <= MAX_BARS_SHOWN:
            print(f"  {i:>3} {str(ts):<19} {float(c['open']):>9.2f} {hi:>9.2f} "
                  f"{lo:>9.2f} {float(c['close']):>9.2f} {r_hi:>+7.2f} {r_lo:>+8.2f}  "
                  f"{', '.join(ev)}")
    if len(path) > MAX_BARS_SHOWN:
        print(f"  ... ({len(path) - MAX_BARS_SHOWN} more bars to exit not shown)")

    # story
    first = "stop" if (hit_sl and (not hit_tp1 or hit_sl < hit_tp1)) else \
            "target" if hit_tp1 else "neither"
    print(f"  >> MFE +{mfe:.2f}R (bar {mfe_bar})  MAE {mae:.2f}R (bar {mae_bar})  "
          f"first touched: {first}"
          f"{'  | ran to TP1 then gave it back' if (hit_tp1 and float(trade['r_achieved'])<=0) else ''}")


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
    eth_idx = eth.set_index("time").sort_index()

    cand  = _select(df, S4_SLOTS)
    taken = _take_onepos(cand)
    if taken.empty:
        print("No taken trades in the data."); return
    taken["entry_time"] = pd.to_datetime(taken["entry_time"])
    yr = taken[taken["entry_time"].dt.year == YEAR].copy()
    if yr.empty:
        print(f"No taken trades in {YEAR}. CSV trades end "
              f"{df['entry_time'].max()}. Refresh data if needed."); return
    yr["month"] = yr["entry_time"].dt.to_period("M")

    # monthly P&L -> which months lost
    monthly = yr.groupby("month")["pnl"].sum()
    print()
    print(_bar("="))
    print(f"  TRADE AUTOPSY — {YEAR}  (fresh $500, one position, pure S4)")
    print(_bar("="))
    print(f"  Monthly P&L:")
    for m, p in monthly.items():
        tag = "  <== LOSING MONTH" if p < 0 else ""
        print(f"    {str(m)}  ${p:>+9.2f}{tag}")
    losing = [m for m, p in monthly.items() if p < 0]
    target = list(monthly.index) if SHOW_ALL_MONTHS else losing
    if not target:
        print("  No losing months — nothing to autopsy."); return
    print(f"  Autopsying months: {', '.join(str(m) for m in target)}")
    print()

    for m in target:
        g = yr[yr["month"] == m].sort_values("entry_time")
        print()
        print(_bar("#"))
        print(f"#  MONTH {m}   trades={len(g)}   "
              f"wins={(g['r_achieved']>0).sum()}   P&L ${monthly[m]:+.2f}")
        print(_bar("#"))
        for _, t in g.iterrows():
            _walk(t.to_dict(), eth_idx)
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
