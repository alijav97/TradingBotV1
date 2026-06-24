"""
analyze_all_bots.py — paper-trade scoreboard for ALL bots (run on the VPS).

Auto-discovers every bot's SQLite database — the BTC/ETH bots in this repo AND
the separate WTI/SpotCrude bot under C:\\TradingBotV2 — then prints a side-by-side
scoreboard (N / win-rate / avg-R / profit-factor / PnL) so you can see which bot
is actually most profitable BEFORE risking real money.

It is SCHEMA-TOLERANT: the WTI bot is a different codebase, so its trade table
may use different column names. The script introspects each database, finds the
trades table and its pnl / status / R-multiple / notes columns by trying common
aliases, and computes whatever it can.

Search roots (override by passing explicit paths as arguments):
  - <repo>/btc_research/**/data/*.db        (btc_bot_1, btc_bot_2, eth_bot)
  - C:\\TradingBotV2\\**\\*.db                 (WTI / SpotCrude bot)
venv / site-packages / .git folders are skipped.

Run on the VPS:
    C:\\TradingBotV2\\venv\\Scripts\\python.exe analyze_all_bots.py
or point it at specific files:
    python analyze_all_bots.py path\\to\\eth_trades.db path\\to\\wti.db
"""
from __future__ import annotations

import glob
import os
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent

# Extra roots to scan for bot databases (the WTI bot lives outside this repo).
_EXTRA_ROOTS = [Path(r"C:\TradingBotV2"), Path(r"C:\Temp\TradingBotV2")]

_SKIP_DIRS = ("venv", "site-packages", ".git", "__pycache__", "node_modules")

# Column-name aliases (lowercased) for schema-tolerant discovery.
_PNL_COLS    = ("pnl_usd", "pnl", "profit", "profit_usd", "net_pnl",
                "realized_pnl", "pnl_dollars", "profit_loss", "p_l")
_STATUS_COLS = ("status", "state", "trade_status")
_RR_COLS     = ("rr_achieved", "rr", "r_multiple", "r_realized", "r", "rmultiple")
_NOTE_COLS   = ("notes", "note", "comment", "tag")
_CLOSE_COLS  = ("close_time", "exit_time", "closed_at", "exit_price")


def _find_dbs(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(a) for a in argv]
    found: set[Path] = set()
    patterns = [str(_REPO / "btc_research" / "**" / "data" / "*.db")]
    for root in _EXTRA_ROOTS:
        if root.exists():
            patterns.append(str(root / "**" / "*.db"))
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            pp = Path(p)
            low = str(pp).lower()
            if any(f"{os.sep}{d}{os.sep}" in low or low.endswith(f"{os.sep}{d}")
                   for d in _SKIP_DIRS):
                continue
            if "-wal" in pp.name or "-shm" in pp.name:
                continue
            found.add(pp)
    return sorted(found)


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]


def _cols(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _pick(cols_lower: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    """Return the real column name for the first alias present (case-insensitive)."""
    for a in aliases:
        if a in cols_lower:
            return cols_lower[a]
    return None


def _find_trades_table(conn: sqlite3.Connection) -> tuple[str, dict] | None:
    """Find the table that looks like a trades ledger; return (table, colmap)."""
    candidates = []
    for t in _tables(conn):
        cols = _cols(conn, t)
        lower = {c.lower(): c for c in cols}
        pnl = _pick(lower, _PNL_COLS)
        if not pnl:
            continue
        colmap = {
            "pnl":    pnl,
            "status": _pick(lower, _STATUS_COLS),
            "rr":     _pick(lower, _RR_COLS),
            "notes":  _pick(lower, _NOTE_COLS),
            "close":  _pick(lower, _CLOSE_COLS),
        }
        # Prefer a table literally named 'trades', else the one with most signal cols.
        score = sum(1 for v in colmap.values() if v) + (5 if t.lower() == "trades" else 0)
        candidates.append((score, t, colmap))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, table, colmap = candidates[0]
    return table, colmap


def _stats(pnls: list[float], rrs: list[float]) -> dict:
    if not pnls:
        return {"n": 0}
    wins = [p for p in pnls if p > 0]
    loss = [abs(p) for p in pnls if p < 0]
    gw, gl = sum(wins), sum(loss)
    valid_rr = [r for r in rrs if r is not None]
    return {
        "n":     len(pnls),
        "wr":    100.0 * len(wins) / len(pnls),
        "pnl":   sum(pnls),
        "avg_r": (sum(valid_rr) / len(valid_rr)) if valid_rr else float("nan"),
        "pf":    (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0),
    }


def analyze(db: Path) -> dict:
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return {"db": db, "error": f"open failed: {exc}"}
    try:
        found = _find_trades_table(conn)
        if not found:
            return {"db": db, "error": "no trades-like table"}
        table, cm = found

        sel = [cm["pnl"]]
        for k in ("rr", "notes", "status", "close"):
            if cm[k]:
                sel.append(cm[k])
        rows = conn.execute(f'SELECT {", ".join(set(sel))} FROM "{table}"').fetchall()
    except sqlite3.Error as exc:
        return {"db": db, "error": f"query failed: {exc}"}
    finally:
        conn.close()

    def closed(r) -> bool:
        if cm["status"]:
            return str(r[cm["status"]]).upper() == "CLOSED"
        if cm["close"]:
            return r[cm["close"]] is not None   # has an exit -> closed
        return True                              # no status info -> count all

    def is_bt(r) -> bool:
        if not cm["notes"]:
            return False
        v = r[cm["notes"]]
        return bool(v) and "backtest" in str(v).lower()

    closed_rows = [r for r in rows if closed(r)]
    live = [r for r in closed_rows if not is_bt(r)]
    bt_n = sum(1 for r in closed_rows if is_bt(r))

    s = _stats([(r[cm["pnl"]] or 0.0) for r in live],
               [r[cm["rr"]] for r in live] if cm["rr"] else [None] * len(live))
    return {"db": db, "table": table, "live": s, "bt_n": bt_n,
            "has_rr": bool(cm["rr"]), "has_status": bool(cm["status"])}


def _label(db: Path) -> str:
    # e.g. eth_bot/eth_trades.db  — or for WTI: <parent>/<file>
    parent = db.parent.parent.name if db.parent.name == "data" else db.parent.name
    return f"{parent}/{db.name}"


def main() -> None:
    dbs = _find_dbs(sys.argv[1:])
    print("=" * 82)
    print("  PAPER-TRADE SCOREBOARD   (CLOSED live rows; backtest rows excluded)")
    print("=" * 82)
    if not dbs:
        print("\n  No databases found under the repo or C:\\TradingBotV2.")
        print("  Either no bot has recorded a trade yet, or you're not on the VPS.")
        print("  Pass DB paths explicitly:  python analyze_all_bots.py <db1> <db2> ...")
        return

    print(f"\n  Found {len(dbs)} database(s):")
    for db in dbs:
        print(f"    - {db}")

    print(f"\n{'BOT / DB':34} {'N':>4} {'WR%':>6} {'AvgR':>7} {'PF':>6} {'PnL$':>11}")
    print("-" * 82)

    results = []
    for db in dbs:
        r = analyze(db)
        results.append(r)
        label = _label(db)
        if r.get("error"):
            print(f"{label:34} {'ERR':>4}   {r['error']}")
            continue
        s = r["live"]
        if s["n"] == 0:
            extra = f"(no live closed trades; {r['bt_n']} backtest rows)"
            print(f"{label:34} {0:>4}   {extra}")
            continue
        avg_r = "   n/a" if not r["has_rr"] or s["avg_r"] != s["avg_r"] else f"{s['avg_r']:+.2f}"
        pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
        print(f"{label:34} {s['n']:>4} {s['wr']:>6.1f} {avg_r:>7} {pf:>6} {s['pnl']:>+11.2f}")

    print("-" * 82)

    ranked = [r for r in results if not r.get("error") and r["live"]["n"] > 0]
    print()
    if not ranked:
        print("  VERDICT: No bot has any LIVE closed paper trades yet - nothing to rank.")
        print("  Let the bots run through their kill-zones, then re-run this.")
        return

    ranked.sort(key=lambda r: r["live"]["pnl"], reverse=True)
    best = ranked[0]
    print(f"  Most $ PnL : {_label(best['db'])}  "
          f"({best['live']['pnl']:+.2f} over {best['live']['n']} trades)")

    thin = [r for r in ranked if r["live"]["n"] < 20]
    if thin:
        print("\n  [!] CAUTION: bots with <20 live trades are statistically thin - the")
        print("     ranking is mostly noise until each has ~20+ closed trades:")
        for r in thin:
            print(f"        {_label(r['db'])}: {r['live']['n']} trades")


if __name__ == "__main__":
    main()
