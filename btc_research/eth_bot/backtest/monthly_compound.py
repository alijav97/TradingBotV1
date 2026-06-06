"""
btc_research/eth_bot/backtest/monthly_compound.py

Full compound P&L simulation for the S4-OPTIMISED ETH Bot strategy set.

S4 STRATEGY SET (validated by backtest_optimised.py, CAGR~+170.5%, MaxDD -30%):
  UTC 02  rsi_50        BTC-aligned  Asia Night        (baseline)
  UTC 05  ema_cross     no BTC req   Asia Morning
  UTC 06  macd_adx      BTC-aligned  EU Pre-Open       (baseline)
  UTC 07  rsi_50        no BTC req   EU Open           <- S4 ADD (biggest CAGR lever)
  UTC 10  rsi_ema       BTC-aligned  EU Mid (ADX>=25)  (baseline)
  UTC 14  keltner       no BTC req   NY Pre-Open
  UTC 14  ema_cross     no BTC req   NY Pre-Open fallback
  UTC 15  keltner       no BTC req   NY Open (best slot)
  (UTC 15 macd_adx      DROPPED by S4 -- net drag)

Risk sizing (Config D - ADX-split, confirmed optimal):
  ADX <= 25  -> 2%  (early trend, weakest bucket)
  ADX 25-40  -> 3%  (sweet spot - best WR/AvgR)
  ADX >= 40  -> 5%  (strong trend, high conviction)

S4 risk overlays:
  - October risk x 0.5  (only month with negative avg return)
  - Monthly circuit breaker: halt new entries once a month draws down -10%

Scope: trades from START_YEAR (2023) onwards. Starting capital: $500

OUTPUT SECTIONS:
  1. MONTHLY breakdown (full 6-year table with year subtotals)
  2. QUARTERLY breakdown (Q1-Q4 per year)
  3. HALF-YEAR breakdown (H1/H2 per year)
  4. YEARLY summary
  5. OVERALL stats + drawdown
  6. PER-STRATEGY breakdown

== USAGE ==
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.monthly_compound
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# -- Paths ---------------------------------------------------------------------
_BACKTEST_DIR = Path(__file__).parent
DATA_DIR      = _BACKTEST_DIR / "data"
TRADES_CSV    = DATA_DIR / "backtest_trades.csv"

# -- Parameters ----------------------------------------------------------------
STARTING_BALANCE = 500.0
START_YEAR       = 2023    # scope: only trades from this year onwards

# Config D ADX-split risk (confirmed best by 6yr ETH ADX sweep)
ADX_SPLIT_EARLY_MAX  = 25
ADX_SPLIT_STRONG_MIN = 40
RISK_EARLY           = 0.02   # 2% -- ADX <= 25
RISK_TRANSITION      = 0.03   # 3% -- ADX 25-40
RISK_STRONG          = 0.05   # 5% -- ADX >= 40

# S4 risk overlays
OCT_RISK_FACTOR      = 0.5    # halve risk in October
CB_THRESHOLD         = -0.10  # halt month after -10% realised drawdown

# S4 8-slot strategy set
# Format: (strategy_key, hour_utc, require_btc_aligned)
# NOTE: rsi50_kz excluded -- fires on same bars as rsi_50 at H2 (double-count).
FINAL_SLOTS = [
    # Baseline 3 slots (BTC-aligned)
    ("rsi_50",   2,  True),    # Asia Night   - RSI 50-cross
    ("macd_adx", 6,  True),    # EU Pre-Open  - MACD+ADX
    ("rsi_ema",  10, True),    # EU Mid       - RSI+EMA (ADX>=25 in live bot)
    # Expansion + S4 (no BTC filter)
    ("ema_cross",  5,  False),  # Asia Morning - EMA 9/21 cross
    ("rsi_50",     7,  False),  # EU Open      - RSI 50-cross (S4 ADD)
    ("keltner",   14,  False),  # NY Pre-Open  - Keltner breakout
    ("ema_cross", 14,  False),  # NY Pre-Open  - EMA cross (Keltner fallback)
    ("keltner",   15,  False),  # NY Open      - Keltner breakout (best slot)
    # macd_adx[15] DROPPED by S4
]

# ---------------------------------------------------------------------------
# COMPLEMENT LEG (regime_complement.py finding)
# ---------------------------------------------------------------------------
# The "anti-correlated hedge" idea failed -- NOTHING in the data is negatively
# correlated with S4 (everything is +corr; in chop, mean-reversion bleeds too).
# BUT specific engulfing/pin_bar HOUR-slots are net-positive standalone AND
# stay positive INSIDE S4's losing months (badTotR > 0). They are an additive
# diversifier, not a hedge. Each entry below: standalone AvgR / bad-month TotR.
#   set ADD_COMPLEMENT = False to reproduce the pure-S4 baseline for comparison.
ADD_COMPLEMENT = True
COMPLEMENT_SLOTS = [
    ("engulfing", 16, False),  # +0.48 / +23.6  (best)
    ("engulfing",  1, False),  # +0.20 / +20.5
    ("pin_bar",    2, False),  # +0.36 / +19.6
    ("engulfing",  8, False),  # +0.11 / +14.4
    ("engulfing",  2, False),  # +0.41 / +13.4
    ("engulfing",  5, False),  # +0.19 / +11.3
    ("pin_bar",   19, False),  # +0.32 / +7.5
    ("pin_bar",   13, False),  # +0.10 / +6.8
    ("pin_bar",   11, False),  # +0.42 / +5.0
]

if ADD_COMPLEMENT:
    FINAL_SLOTS = FINAL_SLOTS + COMPLEMENT_SLOTS


# -- Helpers -------------------------------------------------------------------

def _risk_pct(adx: float) -> float:
    if adx >= ADX_SPLIT_STRONG_MIN:
        return RISK_STRONG
    elif adx <= ADX_SPLIT_EARLY_MAX:
        return RISK_EARLY
    else:
        return RISK_TRANSITION


def _bar(char: str = "-", width: int = 78) -> str:
    return char * width


# ==============================================================================
#  MAIN
# ==============================================================================

def main() -> None:
    if not TRADES_CSV.exists():
        print(f"ERROR: {TRADES_CSV} not found -- run run_backtest.py first")
        sys.exit(1)

    df = pd.read_csv(TRADES_CSV, parse_dates=["entry_time"])

    # -- Scope to START_YEAR onwards -------------------------------------------
    df = df[df["entry_time"].dt.year >= START_YEAR].reset_index(drop=True)
    if df.empty:
        print(f"ERROR: No trades from {START_YEAR} onwards in {TRADES_CSV.name}")
        sys.exit(1)

    # -- Filter final-slot trades ----------------------------------------------
    mask = pd.Series(False, index=df.index)
    for strat, hour, btc_req in FINAL_SLOTS:
        m = (df["strategy"] == strat) & (df["hour_utc"] == hour)
        if btc_req:
            m = m & (df["btc_aligned"] == True)
        mask = mask | m

    ok = df[mask].sort_values("entry_time").reset_index(drop=True)

    if ok.empty:
        print("ERROR: No matching trades found -- check strategy names in CSV")
        sys.exit(1)

    # -- Compound simulation (with S4 October cut + circuit breaker) -----------
    balance = STARTING_BALANCE
    records = []

    cur_month       = None
    month_start_bal = balance
    month_halted    = False
    n_halted        = 0   # trades skipped by the circuit breaker

    for _, t in ok.iterrows():
        et        = t["entry_time"]
        month_key = (et.year, et.month)

        # New month -> reset circuit-breaker state
        if month_key != cur_month:
            cur_month       = month_key
            month_start_bal = balance
            month_halted    = False

        # Circuit breaker: skip remaining trades this month
        if month_halted:
            n_halted += 1
            continue

        adx      = float(t["adx"])
        r_val    = float(t["r_achieved"])
        rp       = _risk_pct(adx)

        # October risk reduction
        if et.month == 10:
            rp *= OCT_RISK_FACTOR

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

        # Update circuit breaker after the trade
        mo_pct = (balance - month_start_bal) / month_start_bal
        if mo_pct <= CB_THRESHOLD:
            month_halted = True

    sim = pd.DataFrame(records)
    sim["month"] = sim["entry_time"].dt.to_period("M")
    sim["year"]  = sim["entry_time"].dt.year
    sim["half_label"] = sim["entry_time"].apply(
        lambda d: f"{d.year}-H{'1' if d.month <= 6 else '2'}"
    )

    n_years   = (sim["entry_time"].max() - sim["entry_time"].min()).days / 365.25
    final_bal = sim["balance"].iloc[-1]
    cagr      = ((final_bal / STARTING_BALANCE) ** (1 / n_years) - 1) * 100

    # ==========================================================================
    #  HEADER
    # ==========================================================================
    print()
    print(_bar("="))
    print("  ETH BOT -- COMPOUND P&L SIMULATION (S4 OPTIMISED, %d+)" % START_YEAR)
    print(f"  Slots  : rsi_50[02] | ema_cross[05] | macd_adx[06] | rsi_50[07]")
    print(f"         : rsi_ema[10] | keltner[14] | ema_cross[14] | keltner[15]")
    if ADD_COMPLEMENT:
        print(f"  +Compl : engulfing[01/02/05/08/16] | pin_bar[02/11/13/19]  "
              f"({len(COMPLEMENT_SLOTS)} MR slots)")
    else:
        print(f"  +Compl : OFF  (pure S4 baseline)")
    print(f"  Risk   : 2% ADX<=25 | 3% ADX 25-40 | 5% ADX>=40  (Config D)")
    print(f"  S4     : Oct risk x{OCT_RISK_FACTOR} | circuit breaker {CB_THRESHOLD*100:.0f}% monthly")
    print(f"  Capital: ${STARTING_BALANCE:,.2f}  |  TP1=2R / TP2=4R")
    print(f"  Period : {sim['entry_time'].min().strftime('%Y-%m')} -> "
          f"{sim['entry_time'].max().strftime('%Y-%m')}  "
          f"({len(sim)} trades, {len(sim) / (n_years * 12):.1f}/mo)")
    print(_bar("="))

    # ==========================================================================
    #  1. MONTHLY TABLE
    # ==========================================================================
    print()
    print(_bar("="))
    print("  1. MONTH-BY-MONTH BREAKDOWN")
    print(_bar())
    print(f"  {'Month':<9}  {'N':>3}  {'W':>3}  {'L':>3}  {'WR%':>6}  "
          f"{'TotR':>6}  {'AvgR':>6}  {'P&L $':>8}  {'EndBal':>9}  {'MoRtn':>7}")
    print(_bar())

    prev_year    = None
    year_buf: list[dict] = []
    monthly_rows = []

    def _flush_year(rows: list[dict], label: str = "YEAR") -> None:
        if not rows:
            return
        yn   = sum(r["n"] for r in rows)
        yw   = sum(r["wins"] for r in rows)
        ytot = sum(r["pnl"] for r in rows)
        ybal = rows[-1]["end_bal"]
        ysb  = rows[0]["start_bal"]
        ypct = (ytot / ysb * 100) if ysb > 0 else 0.0
        yr_r = sum(r["tot_r"] for r in rows)
        print(_bar("."))
        print(f"  {label:<9}  {yn:>3}  {yw:>3}  {yn-yw:>3}  {yw/yn*100:>5.1f}%  "
              f"{yr_r:>+6.2f}  {'':>6}  {ytot:>+8.2f}  "
              f"{ybal:>9,.2f}  {ypct:>+6.1f}%")
        print(_bar("."))

    for month, grp in sim.groupby("month", sort=True):
        year = month.year
        if prev_year and year != prev_year:
            _flush_year(year_buf, str(prev_year))
            year_buf = []
            print()

        n       = len(grp)
        wins    = (grp["outcome"] == "win").sum()
        losses  = n - wins
        wr      = wins / n * 100
        tot_r   = grp["r_achieved"].sum()
        avg_r   = grp["r_achieved"].mean()
        pnl     = grp["pnl_usd"].sum()
        end_b   = grp["balance"].iloc[-1]
        start_b = end_b - pnl
        mo_pct  = pnl / start_b * 100 if start_b > 0 else 0.0

        print(f"  {str(month):<9}  {n:>3}  {wins:>3}  {losses:>3}  {wr:>5.1f}%  "
              f"{tot_r:>+6.3f}  {avg_r:>+6.3f}  {pnl:>+8.2f}  "
              f"{end_b:>9,.2f}  {mo_pct:>+6.1f}%")

        year_buf.append({
            "n": n, "wins": wins, "pnl": pnl,
            "end_bal": end_b, "start_bal": start_b, "tot_r": tot_r,
        })
        monthly_rows.append({
            "month": str(month), "year": year,
            "n_trades": n, "wins": wins, "losses": losses,
            "wr_pct": round(wr, 1), "total_r": round(tot_r, 3),
            "avg_r": round(avg_r, 3), "pnl_usd": round(pnl, 2),
            "end_balance": round(end_b, 2), "mo_return_pct": round(mo_pct, 2),
        })
        prev_year = year

    _flush_year(year_buf, str(prev_year))

    # ==========================================================================
    #  2. QUARTERLY BREAKDOWN
    # ==========================================================================
    print()
    print(_bar("="))
    print("  2. QUARTERLY BREAKDOWN")
    print(_bar())
    print(f"  {'Quarter':<8}  {'N':>4}  {'WR%':>6}  {'TotR':>7}  "
          f"{'P&L $':>9}  {'EndBal':>10}  {'QRtn':>7}")
    print(_bar())

    quarterly_rows = []
    prev_qyr = None

    # Build quarter label manually from year + month
    sim["quarter_label"] = sim["entry_time"].apply(
        lambda d: f"{d.year}-Q{((d.month - 1) // 3) + 1}"
    )

    for qlabel, grp in sim.groupby("quarter_label", sort=True):
        qyr = int(qlabel[:4])
        if prev_qyr and qyr != prev_qyr:
            print(_bar("."))
        n    = len(grp)
        wins = (grp["outcome"] == "win").sum()
        wr   = wins / n * 100
        tr   = grp["r_achieved"].sum()
        pnl  = grp["pnl_usd"].sum()
        eb   = grp["balance"].iloc[-1]
        sb   = eb - pnl
        qpct = pnl / sb * 100 if sb > 0 else 0.0

        print(f"  {qlabel:<8}  {n:>4}  {wr:>5.1f}%  {tr:>+7.3f}  "
              f"{pnl:>+9.2f}  {eb:>10,.2f}  {qpct:>+6.1f}%")

        quarterly_rows.append({
            "quarter": qlabel, "year": qyr,
            "n_trades": n, "wins": wins, "wr_pct": round(wr, 1),
            "total_r": round(tr, 3), "pnl_usd": round(pnl, 2),
            "end_balance": round(eb, 2), "q_return_pct": round(qpct, 2),
        })
        prev_qyr = qyr

    # ==========================================================================
    #  3. HALF-YEAR BREAKDOWN
    # ==========================================================================
    print()
    print(_bar("="))
    print("  3. HALF-YEAR BREAKDOWN  (H1 = Jan-Jun | H2 = Jul-Dec)")
    print(_bar())
    print(f"  {'Half':<8}  {'N':>4}  {'WR%':>6}  {'TotR':>7}  "
          f"{'P&L $':>9}  {'EndBal':>10}  {'HRtn':>7}")
    print(_bar())

    half_rows = []
    prev_hyr  = None

    for hlabel, grp in sim.groupby("half_label", sort=True):
        hyr = int(hlabel[:4])
        if prev_hyr and hyr != prev_hyr:
            print(_bar("."))
        n    = len(grp)
        wins = (grp["outcome"] == "win").sum()
        wr   = wins / n * 100
        tr   = grp["r_achieved"].sum()
        pnl  = grp["pnl_usd"].sum()
        eb   = grp["balance"].iloc[-1]
        sb   = eb - pnl
        hpct = pnl / sb * 100 if sb > 0 else 0.0

        print(f"  {hlabel:<8}  {n:>4}  {wr:>5.1f}%  {tr:>+7.3f}  "
              f"{pnl:>+9.2f}  {eb:>10,.2f}  {hpct:>+6.1f}%")

        half_rows.append({
            "half": hlabel, "year": hyr,
            "n_trades": n, "wins": wins, "wr_pct": round(wr, 1),
            "total_r": round(tr, 3), "pnl_usd": round(pnl, 2),
            "end_balance": round(eb, 2), "h_return_pct": round(hpct, 2),
        })
        prev_hyr = hyr

    # ==========================================================================
    #  4. YEARLY SUMMARY
    # ==========================================================================
    print()
    print(_bar("="))
    print("  4. YEARLY SUMMARY")
    print(_bar())
    print(f"  {'Year':<6}  {'N':>4}  {'WR%':>6}  {'TotR':>8}  "
          f"{'P&L $':>9}  {'EndBal':>10}  {'YrRtn':>7}  {'vs $500':>8}")
    print(_bar())

    yr_start_bal = STARTING_BALANCE
    for year, grp in sim.groupby("year"):
        n       = len(grp)
        wins    = (grp["outcome"] == "win").sum()
        wr      = wins / n * 100
        tr      = grp["r_achieved"].sum()
        pnl     = grp["pnl_usd"].sum()
        eb      = grp["balance"].iloc[-1]
        yr_pct  = pnl / yr_start_bal * 100
        tot_pct = (eb - STARTING_BALANCE) / STARTING_BALANCE * 100

        print(f"  {year:<6}  {n:>4}  {wr:>5.1f}%  {tr:>+8.3f}  "
              f"{pnl:>+9.2f}  {eb:>10,.2f}  {yr_pct:>+6.1f}%  {tot_pct:>+7.1f}%")
        yr_start_bal = eb

    # ==========================================================================
    #  5. OVERALL STATS + DRAWDOWN
    # ==========================================================================
    total_pnl   = final_bal - STARTING_BALANCE
    total_ret   = total_pnl / STARTING_BALANCE * 100
    total_r_all = sim["r_achieved"].sum()
    all_wins    = (sim["outcome"] == "win").sum()
    overall_wr  = all_wins / len(sim) * 100
    overall_avg = sim["r_achieved"].mean()
    active_months = sim["month"].nunique()

    balances    = sim["balance"]
    peak        = balances.cummax()
    dd_pct      = (balances - peak) / peak * 100
    max_dd      = dd_pct.min()
    max_dd_i    = dd_pct.idxmin()
    max_dd_date = sim.loc[max_dd_i, "entry_time"].strftime("%Y-%m-%d")

    outcomes      = sim["outcome"].tolist()
    max_cons_loss = cur_loss = 0
    for o in outcomes:
        cur_loss = cur_loss + 1 if o == "loss" else 0
        max_cons_loss = max(max_cons_loss, cur_loss)

    proj_5yr = STARTING_BALANCE * ((1 + cagr / 100) ** 5)

    print()
    print(_bar("="))
    print("  5. OVERALL SUMMARY")
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
    print(f"  5yr projection      : ${proj_5yr:,.0f}  (from ${STARTING_BALANCE:,.0f})")
    print(f"  Backtest period     : {n_years:.1f} years")
    print(f"  Active months       : {active_months}")
    print(f"  Avg trades / month  : {len(sim) / active_months:.1f}")
    print(f"  Trades skipped (CB) : {n_halted}  (monthly circuit breaker)")
    print(_bar())
    print(f"  Max drawdown        : {max_dd:.1f}%  (at {max_dd_date})")
    print(f"  Max consecutive L's : {max_cons_loss}")
    print(_bar())

    # ==========================================================================
    #  6. PER-STRATEGY BREAKDOWN
    # ==========================================================================
    print()
    print(_bar("="))
    print("  6. PER-STRATEGY BREAKDOWN")
    print(_bar())
    print(f"  {'Strategy':<14}  {'UTC':>3}  {'N':>4}  {'WR%':>6}  {'AvgR':>6}  "
          f"{'TotR':>7}  {'P&L $':>9}  {'Contrib%':>9}")
    print(_bar())

    for (strat, hour), sg in sim.groupby(["strategy", "hour_utc"]):
        n      = len(sg)
        w      = (sg["outcome"] == "win").sum()
        wr     = w / n * 100
        tr     = sg["r_achieved"].sum()
        ar     = sg["r_achieved"].mean()
        tp     = sg["pnl_usd"].sum()
        contrib = tp / total_pnl * 100 if total_pnl != 0 else 0
        print(f"  {strat:<14}  {hour:>3}  {n:>4}  {wr:>5.1f}%  {ar:>+6.3f}  "
              f"{tr:>+7.3f}  {tp:>+9.2f}  {contrib:>+8.1f}%")

    print(_bar("="))

    # ==========================================================================
    #  SAVE TO CSV
    # ==========================================================================
    out_monthly   = DATA_DIR / "monthly_compound.csv"
    out_quarterly = DATA_DIR / "quarterly_compound.csv"
    out_half      = DATA_DIR / "halfyear_compound.csv"
    out_trades    = DATA_DIR / "compound_trades.csv"

    pd.DataFrame(monthly_rows).to_csv(out_monthly, index=False)
    pd.DataFrame(quarterly_rows).to_csv(out_quarterly, index=False)
    pd.DataFrame(half_rows).to_csv(out_half, index=False)
    sim.drop(columns=["quarter_label", "half_label"], errors="ignore").to_csv(
        out_trades, index=False
    )

    print()
    print(f"  Saved: monthly_compound.csv | quarterly_compound.csv | "
          f"halfyear_compound.csv | compound_trades.csv")
    print(_bar("="))
    print()


if __name__ == "__main__":
    main()
