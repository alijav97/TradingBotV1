"""
scripts/clear_paper_trades.py — Clear all live paper trades from the DB.

Keeps all backtest trades intact (notes='backtest').
Deletes all live paper trades (notes != 'backtest'), open or closed.
Also clears the signals_log so the next run starts with a clean history.

Run from the project root:
    python scripts/clear_paper_trades.py
"""
import sqlite3
import os
import sys

DB_PATH = r"C:\Temp\TradingBotV1\v2\data\trades.db"

if not os.path.exists(DB_PATH):
    print(f"ERROR: DB not found at {DB_PATH}")
    sys.exit(1)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# ── 1. Show what we're about to delete ───────────────────────────────────────
print("\n=== CURRENT PAPER TRADES ===")
rows = conn.execute("""
    SELECT id, symbol, direction, entry_price, stop_loss,
           exit_reason, open_time, close_time, pnl_usd, notes
    FROM trades
    WHERE notes IS NULL OR notes = '' OR notes NOT LIKE '%backtest%'
    ORDER BY open_time DESC
""").fetchall()

if not rows:
    print("  No live paper trades found.")
else:
    for r in rows:
        status = "OPEN" if not r["close_time"] else f"CLOSED ({r['exit_reason']})"
        print(
            f"  {str(r['id'])[:8]}  {r['symbol']:8s} {str(r['direction']):6s} "
            f"@ {r['entry_price']:.3f}  {status}  "
            f"pnl={r['pnl_usd'] or 0:.2f}  notes={r['notes']!r}"
        )

print(f"\nTotal paper trades to delete: {len(rows)}")

# ── 2. Show signals log ───────────────────────────────────────────────────────
try:
    sig_count = conn.execute(
        "SELECT COUNT(*) FROM signals_log"
    ).fetchone()[0]
    print(f"Signals log entries to clear: {sig_count}")
except Exception:
    sig_count = 0
    print("signals_log table not found — skipping")

# ── 3. Check for ML features linked to paper trades ──────────────────────────
try:
    ml_count = conn.execute("""
        SELECT COUNT(*) FROM ml_features
        WHERE trade_id IN (
            SELECT id FROM trades
            WHERE notes IS NULL OR notes = '' OR notes NOT LIKE '%backtest%'
        )
    """).fetchone()[0]
    print(f"ML feature rows linked to paper trades: {ml_count}")
except Exception:
    ml_count = 0

# ── 4. Confirm ────────────────────────────────────────────────────────────────
print()
confirm = input("Type YES to delete all paper trades and clear signals log: ").strip()
if confirm != "YES":
    print("Aborted — nothing deleted.")
    conn.close()
    sys.exit(0)

# ── 5. Delete ─────────────────────────────────────────────────────────────────
# Delete ML features for paper trades first (foreign key order)
if ml_count > 0:
    conn.execute("""
        DELETE FROM ml_features
        WHERE trade_id IN (
            SELECT id FROM trades
            WHERE notes IS NULL OR notes = '' OR notes NOT LIKE '%backtest%'
        )
    """)
    print(f"  Deleted {ml_count} ML feature rows")

# Delete paper trades
deleted = conn.execute("""
    DELETE FROM trades
    WHERE notes IS NULL OR notes = '' OR notes NOT LIKE '%backtest%'
""").rowcount
print(f"  Deleted {deleted} paper trade rows")

# Clear signals log
if sig_count > 0:
    try:
        conn.execute("DELETE FROM signals_log")
        print(f"  Cleared {sig_count} signals log entries")
    except Exception as e:
        print(f"  signals_log clear failed: {e}")

conn.commit()
conn.close()

print("\nDone. Database is clean. Backtest data is untouched.")
print("You can now restart the bot with a fresh slate.")
