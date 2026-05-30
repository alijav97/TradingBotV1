"""
btc_research/btc_bot_2/journal/sqlite_journal.py — SQLite trade journal for BTC Bot 2.

Fully standalone — mirrors btc_bot_1/journal/sqlite_journal.py pattern.
Changes vs v2:
  - get_paper_balance() uses btc_bot_2.settings.STARTING_BALANCE (not v2)
  - DB path defaults to btc_bot_2/data/btc2_trades.db
  - No reference to v2 anywhere
  - Includes update_trade() and get_open_trades_count() (needed by BTC2PaperTrader)

Schema is identical to v2 so analysis scripts can be shared.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS trades (
    id               TEXT PRIMARY KEY,
    symbol           TEXT NOT NULL,
    direction        TEXT NOT NULL,
    entry_price      REAL NOT NULL,
    stop_loss        REAL NOT NULL,
    tp1_price        REAL,
    tp2_price        REAL,
    lot_size         REAL NOT NULL,
    status           TEXT NOT NULL DEFAULT 'OPEN',
    open_time        TEXT NOT NULL,
    close_time       TEXT,
    exit_price       REAL,
    exit_reason      TEXT,
    pnl_usd          REAL,
    pips             REAL,
    rr_achieved      REAL,
    strategy         TEXT,
    confluence_score REAL,
    timeframe        TEXT,
    session          TEXT,
    regime           TEXT,
    news_score       REAL,
    tp1_hit          INTEGER DEFAULT 0,
    be_moved         INTEGER DEFAULT 0,
    original_sl      REAL,
    factors_json     TEXT,
    exit_regime      TEXT,
    exit_atr         REAL,
    hold_time_minutes REAL,
    notes            TEXT,
    raw_signal       TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id           TEXT PRIMARY KEY,
    symbol       TEXT NOT NULL,
    direction    TEXT NOT NULL,
    entry_price  REAL,
    stop_loss    REAL,
    tp1_price    REAL,
    score        REAL,
    strategy     TEXT,
    timeframe    TEXT,
    generated_at TEXT NOT NULL,
    taken        INTEGER DEFAULT 0,
    skip_reason  TEXT,
    raw          TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_open   ON trades(open_time);
CREATE INDEX IF NOT EXISTS idx_signals_time  ON signals(generated_at);
"""


class Journal:
    """SQLite trade journal for BTC Bot 2. Thread-safe via WAL mode."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from btc_research.btc_bot_2.settings import DB_PATH
            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("BTC2 Journal opened: %s", self.db_path)

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._apply_migrations()

    def _apply_migrations(self) -> None:
        """Add columns introduced after initial schema (safe to re-run)."""
        migrations = [
            ("trades", "original_sl",        "REAL"),
            ("trades", "factors_json",        "TEXT"),
            ("trades", "exit_regime",         "TEXT"),
            ("trades", "exit_atr",            "REAL"),
            ("trades", "hold_time_minutes",   "REAL"),
        ]
        for table, col, dtype in migrations:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

    # ── Trades ────────────────────────────────────────────────────────────────

    def open_trade(self, trade: dict) -> str:
        """
        Insert a new open trade. Returns the trade ID (UUID string).

        Required: symbol, direction, entry_price, stop_loss, lot_size
        Optional: tp1_price, tp2_price, strategy, confluence_score, timeframe,
                  session, notes, original_sl, exit_atr, raw_signal (dict or str)
        """
        tid = str(uuid.uuid4())
        now = _now()

        raw = trade.get("raw_signal", {})
        raw_signal_str = raw if isinstance(raw, str) else json.dumps(raw)

        self._conn.execute("""
            INSERT INTO trades (
                id, symbol, direction, entry_price, stop_loss, tp1_price, tp2_price,
                lot_size, status, open_time, strategy, confluence_score, timeframe,
                session, notes, raw_signal
            ) VALUES (
                :id, :symbol, :direction, :entry_price, :stop_loss, :tp1_price, :tp2_price,
                :lot_size, 'OPEN', :open_time, :strategy, :confluence_score, :timeframe,
                :session, :notes, :raw_signal
            )
        """, {
            "id":               tid,
            "symbol":           trade["symbol"],
            "direction":        trade["direction"],
            "entry_price":      float(trade["entry_price"]),
            "stop_loss":        float(trade["stop_loss"]),
            "tp1_price":        trade.get("tp1_price"),
            "tp2_price":        trade.get("tp2_price"),
            "lot_size":         float(trade.get("lot_size", 0.001)),
            "open_time":        now,
            "strategy":         trade.get("strategy", ""),
            "confluence_score": trade.get("confluence_score"),
            "timeframe":        trade.get("timeframe", "H1"),
            "session":          trade.get("session", ""),
            "notes":            trade.get("notes", ""),
            "raw_signal":       raw_signal_str,
        })
        # Set original_sl and exit_atr (ATR at open — used for trailing SL)
        original_sl = trade.get("original_sl")
        exit_atr    = trade.get("exit_atr")
        self._conn.execute(
            """UPDATE trades
               SET original_sl  = COALESCE(?, stop_loss),
                   exit_atr     = ?,
                   factors_json = ?
               WHERE id = ?""",
            (original_sl, exit_atr, json.dumps(trade.get("factors", {})), tid)
        )
        self._conn.commit()
        logger.info(
            "BTC2 Trade opened: %s %s %s @ %.2f  SL=%.2f  lot=%.4f",
            tid[:8], trade["symbol"], trade["direction"],
            trade["entry_price"], trade["stop_loss"], trade.get("lot_size", 0),
        )
        return tid

    def close_trade(
        self,
        trade_id:    str,
        exit_price:  float,
        exit_reason: str   = "MANUAL",
        pnl_usd:     float | None = None,
        pips:        float | None = None,
        rr_achieved: float | None = None,
        notes:       str   = "",
        exit_context: dict | None = None,
    ) -> bool:
        """Mark a trade as CLOSED. Returns True if found and updated."""
        now = _now()
        ctx = exit_context or {}
        try:
            with self._conn:
                cursor = self._conn.execute("""
                    UPDATE trades
                    SET status='CLOSED', close_time=:ct, exit_price=:ep,
                        exit_reason=:er, pnl_usd=:pnl, pips=:pips,
                        rr_achieved=:rr, notes=:notes,
                        exit_regime=:exit_regime, exit_atr=:exit_atr,
                        hold_time_minutes=:hold_time
                    WHERE id=:id AND status='OPEN'
                """, {
                    "ct":          now,
                    "ep":          exit_price,
                    "er":          exit_reason,
                    "pnl":         pnl_usd,
                    "pips":        pips,
                    "rr":          rr_achieved,
                    "notes":       notes,
                    "exit_regime": ctx.get("exit_regime"),
                    "exit_atr":    ctx.get("exit_atr"),
                    "hold_time":   ctx.get("hold_time_minutes"),
                    "id":          trade_id,
                })
        except Exception as exc:
            logger.error("close_trade failed for %s: %s", trade_id[:8], exc)
            return False

        updated = cursor.rowcount > 0
        if updated:
            logger.info(
                "BTC2 Trade closed: %s exit=%.2f reason=%s pnl=%s",
                trade_id[:8], exit_price, exit_reason, pnl_usd,
            )
        return updated

    def update_trade(self, trade_id: str, updates: dict) -> bool:
        """
        Generic field update for an open trade.
        Used by BTC2PaperTrader for TP1 hits, BE moves, trailing SL.

        Accepted keys: stop_loss, tp1_hit, be_moved, notes, exit_atr.
        """
        _allowed = {"stop_loss", "tp1_hit", "be_moved", "notes", "exit_atr"}
        cols = {k: v for k, v in updates.items() if k in _allowed}
        if not cols:
            return False
        set_clause = ", ".join(f"{k}=?" for k in cols)
        params     = list(cols.values()) + [trade_id]
        cursor = self._conn.execute(
            f"UPDATE trades SET {set_clause} WHERE id=? AND status='OPEN'",
            params,
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_open_trades_count(self, symbol: str | None = None) -> int:
        """Fast count query — no row fetch."""
        if symbol:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status='OPEN' AND symbol=?", (symbol,)
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status='OPEN'"
            ).fetchone()
        return row[0] if row else 0

    def get_open_trades(self, symbol: str | None = None) -> list[dict]:
        if symbol:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE status='OPEN' AND symbol=? ORDER BY open_time",
                (symbol,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE status='OPEN' ORDER BY open_time"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_trades(
        self,
        symbol: str | None = None,
        status: str | None = None,
        limit:  int = 500,
    ) -> list[dict]:
        conditions, params = [], []
        if symbol:
            conditions.append("symbol=?")
            params.append(symbol)
        if status:
            conditions.append("status=?")
            params.append(status)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM trades {where} ORDER BY open_time DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trade(self, trade_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
        return dict(row) if row else None

    def log_signal(self, signal: dict, taken: bool = False, skip_reason: str = "") -> str:
        sid = str(uuid.uuid4())
        now = _now()
        self._conn.execute("""
            INSERT INTO signals (
                id, symbol, direction, entry_price, stop_loss, tp1_price,
                score, strategy, timeframe, generated_at, taken, skip_reason, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sid,
            signal.get("symbol", ""),
            signal.get("direction", ""),
            signal.get("entry_price"),
            signal.get("stop_loss"),
            signal.get("tp1_price"),
            signal.get("adx") or signal.get("confluence_score"),
            signal.get("strategy", ""),
            signal.get("timeframe", "H1"),
            now,
            1 if taken else 0,
            skip_reason,
            json.dumps(signal),
        ))
        self._conn.commit()
        return sid

    # ── Performance ───────────────────────────────────────────────────────────

    def get_stats(self, symbol: str | None = None, days: int = 30) -> dict:
        """Compute win rate, PnL, profit factor for last N days."""
        conditions, params = ["status='CLOSED'"], []
        if symbol:
            conditions.append("symbol=?")
            params.append(symbol)
        conditions.append("close_time >= datetime('now', ?)")
        params.append(f"-{days} days")

        rows = self._conn.execute(
            "SELECT pnl_usd FROM trades WHERE " + " AND ".join(conditions),
            params,
        ).fetchall()

        if not rows:
            return {"trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "profit_factor": 0.0}

        pnls  = [r["pnl_usd"] or 0.0 for r in rows]
        wins  = [p for p in pnls if p > 0]
        loss  = [abs(p) for p in pnls if p < 0]
        wrate = len(wins) / len(pnls) if pnls else 0.0
        pf    = sum(wins) / sum(loss) if sum(loss) > 0 else float("inf")

        return {
            "trades":        len(pnls),
            "wins":          len(wins),
            "losses":        len(loss),
            "win_rate":      round(wrate * 100, 1),
            "total_pnl":     round(sum(pnls), 2),
            "avg_pnl":       round(sum(pnls) / len(pnls), 2),
            "profit_factor": round(pf, 2),
        }

    def get_paper_balance(self) -> float:
        """
        Return current compounded paper balance.
        Starts from STARTING_BALANCE and applies all closed trade PnL.
        Excludes backtest trades.
        """
        from btc_research.btc_bot_2.settings import STARTING_BALANCE
        rows = self._conn.execute(
            """SELECT pnl_usd FROM trades
               WHERE status='CLOSED'
               AND (notes IS NULL OR notes = '' OR notes NOT LIKE '%backtest%')
               ORDER BY close_time ASC""",
        ).fetchall()
        balance = STARTING_BALANCE
        for r in rows:
            pnl = r["pnl_usd"] or 0.0
            balance = max(balance + pnl, 1.0)
        return round(balance, 2)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
