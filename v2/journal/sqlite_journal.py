"""
journal/sqlite_journal.py — SQLite trade journal for TradingBotV2.

Single source of truth for all paper trades, signals, news events, and
performance summaries. Replaces the JSON-per-instrument files from V1.

Usage:
    from v2.journal.sqlite_journal import Journal
    journal = Journal()                     # uses settings.DB_PATH
    trade_id = journal.open_trade({...})
    journal.close_trade(trade_id, exit_price, exit_reason)
    df = journal.get_trades(symbol="XAUUSD")
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS trades (
    id              TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,          -- "long" | "short"
    entry_price     REAL NOT NULL,
    stop_loss       REAL NOT NULL,
    tp1_price       REAL,
    tp2_price       REAL,
    lot_size        REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'OPEN',  -- OPEN | CLOSED | CANCELLED
    open_time       TEXT NOT NULL,
    close_time      TEXT,
    exit_price      REAL,
    exit_reason     TEXT,                   -- TP1 | TP2 | SL | MANUAL | MAX_HOLD
    pnl_usd         REAL,
    pips            REAL,
    rr_achieved     REAL,
    strategy        TEXT,
    confluence_score REAL,
    timeframe       TEXT,
    session         TEXT,
    regime          TEXT,
    news_score      REAL,
    tp1_hit         INTEGER DEFAULT 0,      -- 0 | 1
    be_moved        INTEGER DEFAULT 0,      -- stop moved to breakeven
    original_sl     REAL,                   -- SL at open (preserved when moved to BE)
    factors_json    TEXT,                   -- JSON: per-factor confluence breakdown {trend:0.5, momentum:1.0, ...}
    exit_regime     TEXT,                   -- market regime at close time
    exit_atr        REAL,                   -- ATR value at close time
    hold_time_minutes REAL,                 -- minutes trade was open
    notes           TEXT,
    raw_signal      TEXT                    -- JSON dump of full signal dict
);

CREATE TABLE IF NOT EXISTS signals (
    id              TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     REAL,
    stop_loss       REAL,
    tp1_price       REAL,
    score           REAL,
    strategy        TEXT,
    timeframe       TEXT,
    generated_at    TEXT NOT NULL,
    taken           INTEGER DEFAULT 0,      -- 1 if converted to trade
    skip_reason     TEXT,                   -- why it was not taken
    raw             TEXT                    -- JSON of full signal
);

CREATE TABLE IF NOT EXISTS news_events (
    id              TEXT PRIMARY KEY,
    title           TEXT,
    country         TEXT,
    impact          TEXT,
    event_time      TEXT,
    recorded_at     TEXT NOT NULL,
    sentiment_score REAL,
    affected_symbols TEXT                   -- JSON list
);

CREATE TABLE IF NOT EXISTS performance (
    id              TEXT PRIMARY KEY,
    period_type     TEXT NOT NULL,          -- DAY | WEEK | MONTH
    period_label    TEXT NOT NULL,          -- e.g. "2026-05-25", "2026-W21"
    symbol          TEXT,                   -- NULL = all instruments
    trades_total    INTEGER DEFAULT 0,
    trades_won      INTEGER DEFAULT 0,
    trades_lost     INTEGER DEFAULT 0,
    win_rate        REAL,
    total_pnl       REAL,
    avg_rr          REAL,
    profit_factor   REAL,
    max_drawdown    REAL,
    computed_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ml_features (
    trade_id        TEXT PRIMARY KEY REFERENCES trades(id),
    features_json   TEXT NOT NULL,          -- JSON dict of 40+ features
    label           INTEGER,                -- 1 = win, 0 = loss (set on close)
    model_version   TEXT,
    predicted_prob  REAL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol  ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status  ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_open    ON trades(open_time);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_time   ON signals(generated_at);
"""


class Journal:
    """SQLite trade journal. Thread-safe via check_same_thread=False + WAL mode."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from v2.settings import DB_PATH
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
        logger.info("Journal opened: %s", self.db_path)

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

        Required keys: symbol, direction, entry_price, stop_loss, lot_size
        Optional: tp1_price, tp2_price, strategy, confluence_score, timeframe,
                  session, regime, news_score, raw_signal (dict)
        """
        tid = str(uuid.uuid4())
        now = _now()

        self._conn.execute("""
            INSERT INTO trades (
                id, symbol, direction, entry_price, stop_loss, tp1_price, tp2_price,
                lot_size, status, open_time, strategy, confluence_score, timeframe,
                session, regime, news_score, raw_signal
            ) VALUES (
                :id, :symbol, :direction, :entry_price, :stop_loss, :tp1_price, :tp2_price,
                :lot_size, 'OPEN', :open_time, :strategy, :confluence_score, :timeframe,
                :session, :regime, :news_score, :raw_signal
            )
        """, {
            "id":               tid,
            "symbol":           trade["symbol"],
            "direction":        trade["direction"],
            "entry_price":      float(trade["entry_price"]),
            "stop_loss":        float(trade["stop_loss"]),
            "tp1_price":        trade.get("tp1_price"),
            "tp2_price":        trade.get("tp2_price"),
            "lot_size":         float(trade.get("lot_size", 0.01)),
            "open_time":        now,
            "strategy":         trade.get("strategy", ""),
            "confluence_score": trade.get("confluence_score"),
            "timeframe":        trade.get("timeframe", "H1"),
            "session":          trade.get("session", ""),
            "regime":           trade.get("regime", ""),
            "news_score":       trade.get("news_score"),
            "raw_signal":       json.dumps(trade.get("raw_signal", {})),
        })
        # Preserve original SL separately so it survives BE moves
        self._conn.execute(
            "UPDATE trades SET original_sl=stop_loss, factors_json=? WHERE id=?",
            (json.dumps(trade.get("factors", {})), tid)
        )
        self._conn.commit()
        logger.info("Trade opened: %s %s %s @ %.5f", tid[:8], trade["symbol"], trade["direction"], trade["entry_price"])
        return tid

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "MANUAL",
        pnl_usd: float | None = None,
        pips: float | None = None,
        rr_achieved: float | None = None,
        notes: str = "",
        exit_context: dict | None = None,
    ) -> bool:
        """
        Mark a trade as CLOSED. All updates happen in a single transaction.

        exit_context (optional) captures market state at close time:
          {"exit_regime": str, "exit_atr": float, "hold_time_minutes": float}

        Returns True if the trade was found and updated.
        """
        now  = _now()
        ctx  = exit_context or {}
        label = (1 if pnl_usd > 0 else 0) if pnl_usd is not None else None

        # ML outcome label: TP2=1.0, TP1=0.7, other win=0.5, loss=0
        if pnl_usd is not None:
            if exit_reason == "TP2":
                label = 1
            elif exit_reason == "TP1" or pnl_usd > 0:
                label = 1
            else:
                label = 0

        try:
            with self._conn:  # single atomic transaction
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
                if label is not None:
                    self._conn.execute(
                        "UPDATE ml_features SET label=? WHERE trade_id=?",
                        (label, trade_id)
                    )
        except Exception as exc:
            logger.error("close_trade transaction failed for %s: %s", trade_id[:8], exc)
            return False

        updated = cursor.rowcount > 0
        if updated:
            logger.info("Trade closed: %s exit=%.5f reason=%s pnl=%s",
                        trade_id[:8], exit_price, exit_reason, pnl_usd)
        return updated

    def mark_tp1_hit(self, trade_id: str) -> None:
        self._conn.execute(
            "UPDATE trades SET tp1_hit=1 WHERE id=?", (trade_id,)
        )
        self._conn.commit()

    def mark_breakeven(self, trade_id: str) -> None:
        self._conn.execute(
            "UPDATE trades SET be_moved=1 WHERE id=?", (trade_id,)
        )
        self._conn.commit()

    def update_stop_loss(self, trade_id: str, new_sl: float) -> None:
        """Move stop loss to a new price (e.g., breakeven or trail)."""
        self._conn.execute(
            "UPDATE trades SET stop_loss=? WHERE id=? AND status='OPEN'",
            (new_sl, trade_id)
        )
        self._conn.commit()

    def get_open_trades(self, symbol: str | None = None) -> list[dict]:
        """Return all open trades, optionally filtered by symbol."""
        if symbol:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE status='OPEN' AND symbol=? ORDER BY open_time",
                (symbol,)
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
        limit: int = 500,
    ) -> list[dict]:
        """Return trades filtered by symbol and/or status."""
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
            params
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trade(self, trade_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
        return dict(row) if row else None

    # ── Signals ───────────────────────────────────────────────────────────────

    def log_signal(self, signal: dict, taken: bool = False, skip_reason: str = "") -> str:
        """Record a generated signal (taken or not). Returns signal ID."""
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
            signal.get("entry_price") or signal.get("entry"),
            signal.get("stop_loss"),
            signal.get("tp1_price") or signal.get("tp1"),
            signal.get("score") or signal.get("confluence_score"),
            signal.get("strategy", ""),
            signal.get("timeframe", "H1"),
            now,
            1 if taken else 0,
            skip_reason,
            json.dumps(signal),
        ))
        self._conn.commit()
        return sid

    # ── News ──────────────────────────────────────────────────────────────────

    def log_news_event(self, event: dict, sentiment_score: float = 0.0,
                       affected_symbols: list[str] | None = None) -> str:
        eid = str(uuid.uuid4())
        now = _now()
        self._conn.execute("""
            INSERT INTO news_events (
                id, title, country, impact, event_time, recorded_at,
                sentiment_score, affected_symbols
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            eid,
            event.get("title", ""),
            event.get("country", ""),
            event.get("impact", ""),
            event.get("time_gst", ""),
            now,
            sentiment_score,
            json.dumps(affected_symbols or []),
        ))
        self._conn.commit()
        return eid

    # ── ML Features ───────────────────────────────────────────────────────────

    def save_ml_features(self, trade_id: str, features: dict,
                         predicted_prob: float | None = None,
                         model_version: str = "") -> None:
        now = _now()
        self._conn.execute("""
            INSERT OR REPLACE INTO ml_features
                (trade_id, features_json, predicted_prob, model_version, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (trade_id, json.dumps(features), predicted_prob, model_version, now))
        self._conn.commit()

    def get_ml_training_data(self) -> list[dict]:
        """Return closed trades with features and labels for ML training."""
        rows = self._conn.execute("""
            SELECT f.trade_id, f.features_json, f.label
            FROM ml_features f
            JOIN trades t ON f.trade_id = t.id
            WHERE t.status='CLOSED' AND f.label IS NOT NULL
        """).fetchall()
        result = []
        for r in rows:
            try:
                features = json.loads(r["features_json"])
                features["_label"] = r["label"]
                features["_trade_id"] = r["trade_id"]
                result.append(features)
            except json.JSONDecodeError:
                continue
        return result

    # ── Performance ───────────────────────────────────────────────────────────

    def get_stats(self, symbol: str | None = None, days: int = 30) -> dict:
        """Compute win rate, PnL, and profit factor for the last N days."""
        conditions, params = ["status='CLOSED'"], []
        if symbol:
            conditions.append("symbol=?")
            params.append(symbol)
        conditions.append("close_time >= datetime('now', ?) ")
        params.append(f"-{days} days")

        rows = self._conn.execute(
            "SELECT pnl_usd FROM trades WHERE " + " AND ".join(conditions),
            params
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

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
