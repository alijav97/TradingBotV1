"""
analyze_all_bots.py — paper-trade scoreboard for all bots (run on the VPS).

Opens every bot's SQLite journal, computes the live-paper performance of each,
and prints a side-by-side scoreboard so you can see which bot is actually most
profitable BEFORE risking real money.

It auto-discovers databases at:
    btc_research/<bot>/data/*.db
(any *.db that has a 'trades' table). WAL/SHM sidecar files are ignored.

LIVE vs BACKTEST:
  Rows whose notes contain 'backtest' are excluded from the headline stats
  (same filter the bot's own balance/throttle use). They are reported separately
  so a seeded backtest history can't masquerade as live edge.

Run on the VPS (where the bots actually run and write their DBs):
    C:\\TradingBotV2\\venv\\Scripts\\python.exe analyze_all_bots.py
or point it at specific files:
    python analyze_all_bots.py path\\to\\eth_trades.db path\\to\\btc_trades.db
"""
from __future__ import annotations

import glob
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent


def _find_dbs(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(a) for a in argv]
    found: list[Path] = []
    for p in glob.glob(str(_REPO / "btc_research" / "**" / "data" / "*.db"),
                        recursive=True):
        pp = Path(p)
        if pp.suffix == ".db" and "-wal" not in pp.name and "-shm" not in pp.name:
            found.append(pp)
    return sorted(set(found))


def _has_trades_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
    ).fetchone()
    return row is not None


def _stats(pnls: list[float], rrs: list[float]) -> dict:
    if not pnls:
        return {"n": 0}
    wins = [p for p in pnls if p > 0]
    loss = [abs(p) for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = sum(loss)
    valid_rr = [r for r in rrs if r is not None]
    return {
        "n":      len(pnls),
        "wr":     100.0 * len(wins) / len(pnls),
        "pnl":    sum(pnls),
        "avg_r":  (sum(valid_rr) / len(valid_rr)) if valid_rr else float("nan"),
        "pf":     (gross_win / gross_loss) if gross_loss > 0
                  else (float("inf") if gross_win > 0 else 0.0),
        "best":   max(pnls),
        "worst":  min(pnls),
    }


def analyze(db: Path) -> dict | None:
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return {"db": db, "error": f"open failed: {exc}"}

    try:
        if not _has_trades_table(conn):
            return {"db": db, "error": "no 'trades' table"}

        rows = conn.execute(
            "SELECT pnl_usd, rr_achieved, notes, status, open_time, close_time "
            "FROM trades WHERE status='CLOSED'"
        ).fetchall()
        open_n = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status='OPEN'"
        ).fetchone()[0]
    finally:
        conn.close()

    def is_bt(note: str | None) -> bool:
        return bool(note) and "backtest" in note.lower()

    live = [r for r in rows if not is_bt(r["notes"])]
    bt   = [r for r in rows if is_bt(r["notes"])]

    live_stats = _stats([r["pnl_usd"] or 0.0 for r in live],
                        [r["rr_achieved"] for r in live])
    bt_n = len(bt)

    times = [r["close_time"] for r in live if r["close_time"]]
    span = (f"{min(times)[:10]} … {max(times)[:10]}") if times else "—"

    return {
        "db": db, "open": open_n, "live": live_stats,
        "bt_n": bt_n, "span": span,
    }


def main() -> None:
    dbs = _find_dbs(sys.argv[1:])
    print("=" * 78)
    print("  PAPER-TRADE SCOREBOARD  (live rows only; backtest rows excluded)")
    print("=" * 78)
    if not dbs:
        print("\n  No databases found. Either no bot has recorded a trade yet,")
        print("  or you're not on the machine where the bots run.")
        print("  Pass DB paths explicitly:  python analyze_all_bots.py <db1> <db2> ...")
        return

    hdr = f"{'BOT / DB':28} {'N':>4} {'WR%':>6} {'AvgR':>6} {'PF':>6} {'PnL$':>10} {'Open':>5}"
    print("\n" + hdr)
    print("-" * 78)

    results = []
    for db in dbs:
        r = analyze(db)
        results.append(r)
        label = db.parent.parent.name + "/" + db.name  # e.g. eth_bot/eth_trades.db
        if r.get("error"):
            print(f"{label:28} {'ERR':>4}  {r['error']}")
            continue
        s = r["live"]
        if s["n"] == 0:
            print(f"{label:28} {0:>4}  (no live closed trades; {r['bt_n']} backtest rows)")
            continue
        avg_r = "  nan" if s["avg_r"] != s["avg_r"] else f"{s['avg_r']:+.2f}"
        pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
        print(f"{label:28} {s['n']:>4} {s['wr']:>6.1f} {avg_r:>6} {pf:>6} "
              f"{s['pnl']:>+10.2f} {r['open']:>5}")

    print("-" * 78)

    # Verdict
    ranked = [r for r in results
              if not r.get("error") and r["live"]["n"] > 0]
    print()
    if not ranked:
        print("  VERDICT: No bot has any LIVE paper trades yet — nothing to rank.")
        print("  Let the bots run through their kill-zones first, then re-run this.")
        return

    ranked.sort(key=lambda r: r["live"]["pnl"], reverse=True)
    best = ranked[0]
    label = best["db"].parent.parent.name
    print(f"  Most $ PnL : {label}  ({best['live']['pnl']:+.2f} over {best['live']['n']} trades)")
    print(f"  Date span  : {best['span']}")
    print()
    thin = [r for r in ranked if r["live"]["n"] < 20]
    if thin:
        print("  ⚠  CAUTION: bots with <20 live trades are statistically thin —")
        print("     the ranking is noise until each has ~20+ closed trades:")
        for r in thin:
            print(f"        {r['db'].parent.parent.name}: {r['live']['n']} trades")


if __name__ == "__main__":
    main()
