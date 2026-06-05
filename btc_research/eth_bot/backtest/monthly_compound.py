"""
btc_research/eth_bot/backtest/monthly_compound.py

Month-by-month compound P&L simulation for the OK kill-zone strategies.

OK strategies selected from 6-year backtest (BTC-aligned, phase-2 TP1=2R/TP2=4R):
  UTC 02:xx  rsi_50 / rsi50_kz  WR=50.0%  AvgR=+0.786  PF=2.57  N=46
  UTC 06:xx  macd_adx            WR=45.0%  AvgR=+0.660  PF=2.65  N=20
  UTC 10:xx  rsi_ema             WR=48.1%  AvgR=+0.706  PF=2.36  N=27

Total live-bot trades across all three slots: ~93 over 6 years (~15/yr, ~1.3/mo)

Risk sizing (ADX-split, applied per trade with compounding):
  ADX ≤ 25  → 3%  (early trend, lower conviction)
  ADX 25-40 → 2%  (transition / dead zone)
  ADX ≥ 40  → 4%  (strong trend, high conviction)

Starting capital: $500

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.monthly_compound
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_BACKTEST_DIR = Path(__file__).parent
DATA_DIR      = _BACKTEST_DIR / "data"
TRADES_CSV    = DATA_DIR / "backtest_trades.csv"

# ── Parameters ─────────────────────────────────────────────────────────────────
STARTING_BALANCE     = 500.0

ADX_SPLIT_EARLY_MAX  = 25
ADX_SPLIT_STRONG_MIN = 40
RISK_EARLY           = 0.03   # 3% — ADX ≤ 25
RISK_TRANSITION      = 0.02   # 2% — ADX 25-40
RISK_STRONG          = 0.04   # 4% — ADX ≥ 40

# OK strategy slots (strategy_key, hour_utc, require_btc_aligned)
# rsi50_kz is the KZ-filtered version — include it if the new backtest CSV has it
OK_SLOTS = [
    ("rsi_50",   2,  True),
    ("rsi50_kz", 2,  True),   # will be empty if using old backtest CSV — no harm
    ("macd_adx", 6,  True),
    ("rsi_ema",  10, True),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _risk_pct(adx: float) -> float:
    if adx >= ADX_SPLIT_STRONG_MIN:
        return RISK_STRONG
    elif adx <= ADX_SPLIT_EARLY_MAX:
        return RISK_EARLY
    else:
        return RISK_TRANSITION


def _bar(char: str = "─", width: int = 72) -> str:
    return char * width


def _pct(val: float, total: float) -> str:
    if total == 0:
        return "  n/a"
    return f"{val / total * 100:+5.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Load ────────────────────────────────────────────────────────────────────
    if not TRADES_CSV.exists():
        print(f"ERROR: {TRADES_CSV} not found — run run_backtest.py first")
        sys.exit(1)

    df = pd.read_csv(TRADES_CSV, parse_dates=["entry_time"])

    # ── Filter OK trades ────────────────────────────────────────────────────────
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in OK_SLOTS:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m

    ok = df[mask].sort_values("entry_time").reset_index(drop=True)

    if ok.empty:
        print("ERROR: No matching trades found — check strategy names in CSV")
        sys.exit(1)

    # ── Compound simulation ─────────────────────────────────────────────────────
    balance = STARTING_BALANCE
    records = []

    for _, t in ok.iterrows():
        adx      = float(t["adx"])
        r_val    = float(t["r_achieved"])
        rp       = _risk_pct(adx)
        risk_usd = balance * rp
        pnl_usd  = risk_usd * r_val
        balance += pnl_usd

        records.append({
            "entry_time": t["entry_time"],
            "strategy":   t["strategy"],
            "hour_utc":   int(t["hour_utc"]),
            "direction":  t["direction"],
            "adx":        round(adx, 1),
            "outcome":    t["outcome"],
            "r_achieved": round(r_val, 3),
            "risk_pct":   rp * 100,
            "risk_usd":   round(risk_usd, 2),
            "pnl_usd":    round(pnl_usd, 2),
            "balance":    round(balance, 2),
        })

    sim = pd.DataFrame(records)
    sim["month"] = sim["entry_time"].dt.to_period("M")
    sim["year"]  = sim["entry_time"].dt.year

    # ══════════════════════════════════════════════════════════════════════════
    #  PRINT HEADER
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print(_bar("═"))
    print("  ETH BOT — MONTHLY COMPOUND P&L SIMULATION")
    print(f"  OK Strategies:  rsi_50 [02:xx] | macd_adx [06:xx] | rsi_ema [10:xx]")
    print(f"  Risk sizing  :  3% ADX≤25 | 2% ADX 25-40 | 4% ADX≥40  (BTC-aligned only)")
    print(f"  Starting cap :  ${STARTING_BALANCE:,.2f}   |   TP1=2R / TP2=4R")
    print(f"  Backtest     :  {sim['entry_time'].min().strftime('%Y-%m')} → "
          f"{sim['entry_time'].max().strftime('%Y-%m')}   ({len(sim)} total trades)")
    print(_bar("═"))

    # ══════════════════════════════════════════════════════════════════════════
    #  MONTHLY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("  MONTH-BY-MONTH BREAKDOWN")
    print(_bar())

    hdr = (f"  {'Month':<9}  {'N':>3}  {'W':>3}  {'L':>3}  {'WR%':>6}  "
           f"{'TotalR':>7}  {'AvgR':>6}  {'P&L $':>8}  "
           f"{'EndBal':>9}  {'MoRtn':>7}")
    print(hdr)
    print(_bar())

    prev_year     = None
    year_buf: list[dict] = []
    monthly_rows  = []

    def _flush_year(rows: list[dict]) -> None:
        """Print year subtotal."""
        if not rows:
            return
        yn   = sum(r["n"] for r in rows)
        yw   = sum(r["wins"] for r in rows)
        ytot = sum(r["pnl"] for r in rows)
        ybal = rows[-1]["end_bal"]
        ysb  = rows[0]["start_bal"]
        ypct = (ytot / ysb * 100) if ysb > 0 else 0.0
        yr_r = sum(r["tot_r"] for r in rows)
        print(_bar("·"))
        print(f"  {'YEAR':9}  {yn:>3}  {yw:>3}  {yn-yw:>3}  {yw/yn*100:>5.1f}%  "
              f"{yr_r:>+7.2f}  {'':>6}  {ytot:>+8.2f}  "
              f"{ybal:>9,.2f}  {ypct:>+6.1f}%")
        print(_bar("·"))

    for month, grp in sim.groupby("month", sort=True):
        year = month.year
        if prev_year and year != prev_year:
            _flush_year(year_buf)
            year_buf = []
            print()

        n      = len(grp)
        wins   = (grp["outcome"] == "win").sum()
        losses = n - wins
        wr     = wins / n * 100
        tot_r  = grp["r_achieved"].sum()
        avg_r  = grp["r_achieved"].mean()
        pnl    = grp["pnl_usd"].sum()
        end_b  = grp["balance"].iloc[-1]
        # start balance = balance before this month's first trade
        row_i  = grp.index[0]
        start_b = end_b - pnl

        mo_pct = pnl / start_b * 100 if start_b > 0 else 0.0

        print(f"  {str(month):<9}  {n:>3}  {wins:>3}  {losses:>3}  {wr:>5.1f}%  "
              f"{tot_r:>+7.3f}  {avg_r:>+6.3f}  {pnl:>+8.2f}  "
              f"{end_b:>9,.2f}  {mo_pct:>+6.1f}%")

        year_buf.append({
            "n": n, "wins": wins, "pnl": pnl,
            "end_bal": end_b, "start_bal": start_b,
            "tot_r": tot_r,
        })
        monthly_rows.append({
            "month": str(month), "year": year,
            "n_trades": n, "wins": wins, "losses": losses,
            "wr_pct": round(wr, 1), "total_r": round(tot_r, 3),
            "avg_r": round(avg_r, 3), "pnl_usd": round(pnl, 2),
            "end_balance": round(end_b, 2), "mo_return_pct": round(mo_pct, 2),
        })
        prev_year = year

    _flush_year(year_buf)   # flush last year

    # ══════════════════════════════════════════════════════════════════════════
    #  YEARLY SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print(_bar("═"))
    print("  YEARLY SUMMARY")
    print(_bar())
    print(f"  {'Year':<6}  {'N':>4}  {'WR%':>6}  {'Total R':>8}  "
          f"{'P&L $':>9}  {'End Bal':>10}  {'Yr Rtn':>8}  {'vs $500':>8}")
    print(_bar())

    # Regroup by year from sim
    yr_start_bal = STARTING_BALANCE
    for year, grp in sim.groupby("year"):
        n    = len(grp)
        wins = (grp["outcome"] == "win").sum()
        wr   = wins / n * 100
        tr   = grp["r_achieved"].sum()
        pnl  = grp["pnl_usd"].sum()
        eb   = grp["balance"].iloc[-1]
        yr_pct  = pnl / yr_start_bal * 100
        tot_pct = (eb - STARTING_BALANCE) / STARTING_BALANCE * 100

        print(f"  {year:<6}  {n:>4}  {wr:>5.1f}%  {tr:>+8.3f}  "
              f"{pnl:>+9.2f}  {eb:>10,.2f}  {yr_pct:>+7.1f}%  {tot_pct:>+7.1f}%")
        yr_start_bal = eb

    # ══════════════════════════════════════════════════════════════════════════
    #  OVERALL STATS
    # ══════════════════════════════════════════════════════════════════════════
    final_bal   = sim["balance"].iloc[-1]
    total_pnl   = final_bal - STARTING_BALANCE
    total_ret   = total_pnl / STARTING_BALANCE * 100
    total_r_all = sim["r_achieved"].sum()
    all_wins    = (sim["outcome"] == "win").sum()
    overall_wr  = all_wins / len(sim) * 100
    overall_avg = sim["r_achieved"].mean()

    # Months with at least one trade
    active_months = sim["month"].nunique()
    n_years       = (sim["entry_time"].max() - sim["entry_time"].min()).days / 365.25
    cagr          = ((final_bal / STARTING_BALANCE) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    print(_bar("═"))
    print()
    print("  OVERALL SUMMARY")
    print(_bar())
    print(f"  Total trades        : {len(sim)}")
    print(f"  Wins / Losses       : {all_wins} / {len(sim) - all_wins}")
    print(f"  Overall WR          : {overall_wr:.1f}%")
    print(f"  Overall AvgR        : {overall_avg:+.3f}R")
    print(f"  Total R earned      : {total_r_all:+.3f}R")
    print(f"  Starting capital    : ${STARTING_BALANCE:,.2f}")
    print(f"  Ending balance      : ${final_bal:,.2f}")
    print(f"  Total P&L           : ${total_pnl:+,.2f}  ({total_ret:+.1f}%)")
    print(f"  CAGR (annualised)   : {cagr:+.1f}%")
    print(f"  Active months       : {active_months} of {active_months + (sim['month'].max() - sim['month'].min()).n + 1 - active_months} total months")
    print(f"  Avg trades / month  : {len(sim) / active_months:.1f}")
    print(_bar())

    # ══════════════════════════════════════════════════════════════════════════
    #  DRAWDOWN ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    balances = sim["balance"]
    peak     = balances.cummax()
    dd_pct   = (balances - peak) / peak * 100
    max_dd   = dd_pct.min()
    max_dd_i = dd_pct.idxmin()

    print()
    print("  DRAWDOWN")
    print(_bar())
    print(f"  Max drawdown        : {max_dd:.1f}%  (at {sim.loc[max_dd_i, 'entry_time'].strftime('%Y-%m-%d')})")

    # Consecutive losses
    outcomes = sim["outcome"].tolist()
    max_cons_loss = cur_loss = 0
    for o in outcomes:
        cur_loss = cur_loss + 1 if o == "loss" else 0
        max_cons_loss = max(max_cons_loss, cur_loss)
    print(f"  Max consecutive L's : {max_cons_loss}")
    print(_bar())

    # ══════════════════════════════════════════════════════════════════════════
    #  PER-STRATEGY BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("  PER-STRATEGY BREAKDOWN (all BTC-aligned)")
    print(_bar())
    print(f"  {'Strategy':<14}  {'UTC':>3}  {'N':>4}  {'WR%':>6}  {'AvgR':>6}  "
          f"{'Total R':>8}  {'Total $':>9}")
    print(_bar())

    for (strat, hour), sg in sim.groupby(["strategy", "hour_utc"]):
        n  = len(sg)
        w  = (sg["outcome"] == "win").sum()
        wr = w / n * 100
        tr = sg["r_achieved"].sum()
        ar = sg["r_achieved"].mean()
        tp = sg["pnl_usd"].sum()
        print(f"  {strat:<14}  {hour:>3}  {n:>4}  {wr:>5.1f}%  {ar:>+6.3f}  "
              f"{tr:>+8.3f}  {tp:>+9.2f}")

    print(_bar())

    # ══════════════════════════════════════════════════════════════════════════
    #  SAVE TO CSV
    # ══════════════════════════════════════════════════════════════════════════
    out_monthly = DATA_DIR / "monthly_compound.csv"
    out_trades  = DATA_DIR / "compound_trades.csv"

    pd.DataFrame(monthly_rows).to_csv(out_monthly, index=False)
    sim.to_csv(out_trades, index=False)

    print()
    print(f"  Saved monthly summary → {out_monthly.name}")
    print(f"  Saved trade-level log → {out_trades.name}")
    print(_bar("═"))
    print()


if __name__ == "__main__":
    main()
