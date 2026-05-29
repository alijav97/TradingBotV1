"""
btc_research/backtest/report.py — Format and print BTC backtest results.

Includes:
  - Overall stats (WR, R:R, expectancy, drawdown)
  - Hold-time analysis (avg hours per win/loss)
  - Direction breakdown (long vs short performance)
  - Exit reason breakdown
  - Score analysis (do higher-scoring trades actually win more?)
  - Equity curve summary
"""
from __future__ import annotations

import pandas as pd
from btc_research.settings import STARTING_BALANCE


SEP  = "=" * 65
LINE = "-" * 65


def print_report(results: dict) -> None:
    trades  = results.get("trades", [])
    balance = results.get("balance", STARTING_BALANCE)

    if not trades:
        print(SEP)
        print("BTC BACKTEST RESULTS — No trades taken.")
        print("Try lowering MIN_CONFLUENCE_SCORE (currently set in btc_research/settings.py)")
        print(SEP)
        return

    df = pd.DataFrame(trades)

    total    = len(df)
    wins_df  = df[df["pnl_usd"] > 0]
    loss_df  = df[df["pnl_usd"] <= 0]
    wins     = len(wins_df)
    losses   = len(loss_df)
    win_rate = wins / total * 100

    avg_win  = wins_df["pnl_usd"].mean()  if wins > 0   else 0.0
    avg_loss = loss_df["pnl_usd"].mean()  if losses > 0 else 0.0
    avg_r_w  = wins_df["r_multiple"].mean() if wins > 0 else 0.0
    avg_r_l  = loss_df["r_multiple"].mean() if losses > 0 else 0.0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    # Max drawdown
    equity = [STARTING_BALANCE] + df["balance_after"].tolist()
    peak   = STARTING_BALANCE
    max_dd = 0.0
    for b in equity:
        if b > peak:
            peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Hold time (H1 bars = hours)
    avg_hold_w = wins_df["bars_held"].mean() if wins > 0   else 0.0
    avg_hold_l = loss_df["bars_held"].mean() if losses > 0 else 0.0
    max_hold   = df["bars_held"].max()

    # Direction breakdown
    long_df   = df[df["direction"] == "long"]
    short_df  = df[df["direction"] == "short"]
    long_wr   = (long_df["pnl_usd"]  > 0).mean() * 100 if len(long_df)  > 0 else 0.0
    short_wr  = (short_df["pnl_usd"] > 0).mean() * 100 if len(short_df) > 0 else 0.0
    long_pnl  = long_df["pnl_usd"].sum()  if len(long_df)  > 0 else 0.0
    short_pnl = short_df["pnl_usd"].sum() if len(short_df) > 0 else 0.0

    # Exit reason breakdown
    exit_counts = df["exit_reason"].value_counts()

    # Score analysis
    avg_score_w = wins_df["score"].mean() if wins > 0   else 0.0
    avg_score_l = loss_df["score"].mean() if losses > 0 else 0.0

    # Day-of-week PnL
    df["open_date"]  = pd.to_datetime(df["open_time"])
    df["day_of_week"] = df["open_date"].dt.day_name()
    dow_stats = df.groupby("day_of_week")["pnl_usd"].agg(["count", "sum", "mean"])
    dow_wr    = df.groupby("day_of_week").apply(
        lambda x: (x["pnl_usd"] > 0).sum() / len(x) * 100
    )

    print()
    print(SEP)
    print("BTC INTER-MARKET CONFLUENCE BACKTEST — RESULTS")
    print(SEP)
    print(f"Period        : {df['open_time'].min()[:10]}  ->  {df['close_time'].max()[:10]}")
    print(f"Total trades  : {total}")
    print(f"Win rate      : {win_rate:.1f}%   ({wins} wins / {losses} losses)")
    print(f"Expectancy    : ${expectancy:+.2f} per trade")
    print()

    print("P&L")
    print(f"  Avg win     : ${avg_win:+.2f}   (avg R: +{avg_r_w:.2f})")
    print(f"  Avg loss    : ${avg_loss:+.2f}  (avg R: {avg_r_l:.2f})")
    print(f"  Net PnL     : ${balance - STARTING_BALANCE:+,.2f}")
    print(f"  Final bal   : ${balance:,.2f}   (started ${STARTING_BALANCE:,})")
    print(f"  Max drawdown: {max_dd:.1f}%")
    print()

    print("HOLD TIME (H1 bars = hours)")
    print(f"  Avg win hold : {avg_hold_w:.1f}h")
    print(f"  Avg loss hold: {avg_hold_l:.1f}h")
    print(f"  Max hold     : {max_hold}h")
    print()

    print("DIRECTION BREAKDOWN")
    print(f"  Long  : {len(long_df):3d} trades  WR={long_wr:.1f}%   PnL=${long_pnl:+,.2f}")
    print(f"  Short : {len(short_df):3d} trades  WR={short_wr:.1f}%  PnL=${short_pnl:+,.2f}")
    print()

    print("EXIT REASONS")
    for reason, count in exit_counts.items():
        pct = count / total * 100
        sub = df[df["exit_reason"] == reason]
        sub_wr = (sub["pnl_usd"] > 0).mean() * 100
        print(f"  {reason:<18}: {count:3d}  ({pct:.1f}%)  WR={sub_wr:.0f}%")
    print()

    print("CONFLUENCE SCORE ANALYSIS")
    print(f"  Avg score (winners) : {avg_score_w:.2f}")
    print(f"  Avg score (losers)  : {avg_score_l:.2f}")
    # Score quartile analysis
    df["score_q"] = pd.qcut(df["score"], q=4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    q_stats = df.groupby("score_q", observed=True).apply(
        lambda x: pd.Series({
            "trades": len(x),
            "wr_pct": (x["pnl_usd"] > 0).mean() * 100,
            "avg_pnl": x["pnl_usd"].mean(),
        })
    )
    print(f"  {'Quartile':<12} {'Trades':>7} {'WR%':>7} {'Avg PnL':>10}")
    print(f"  {LINE[:45]}")
    for q, row in q_stats.iterrows():
        print(f"  {str(q):<12} {int(row['trades']):>7} {row['wr_pct']:>6.1f}% "
              f"${row['avg_pnl']:>+9.2f}")
    print()

    print("DAY-OF-WEEK BREAKDOWN")
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    for day in day_order:
        if day not in dow_stats.index:
            continue
        n   = int(dow_stats.loc[day, "count"])
        pnl = dow_stats.loc[day, "sum"]
        wr  = dow_wr.loc[day]
        print(f"  {day:<12}: {n:3d} trades  WR={wr:.0f}%  PnL=${pnl:+,.2f}")
    print(SEP)
