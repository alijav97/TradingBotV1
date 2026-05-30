"""
btc_bot_1/journal/sqlite_journal.py — SQLite trade journal for BTC Bot 1.

Adapted from v2/journal/sqlite_journal.py.
Changes:
  - get_paper_balance() uses btc_bot_1.settings.STARTING_BALANCE (not v2)
  - DB path defaults to btc_bot_1/data/btc_trades.db
  - No reference to v2 anywhere

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

CREATE TABLE IF NOT EXISTS ml_features (
    trade_id       TEXT PRIMARY KEY REFERENCES trades(id),
    features_json  TEXT NOT NULL,
    label          INTEGER,
    model_version  TEXT,
    predicted_prob REAL,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_open   ON trades(open_time);
"""


class Journal:
    """SQLite trade journal for BTC Bot 1. Thread-safe via WAL + check_same_thread=False."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from btc_research.btc_bot_1.settings import DB_PATH
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
        logger.info("BTC Journal opened: %s", self.db_path)

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Trades ────────────────────────────────────────────────────────────────

    def open_trade(self, trade: dict) -> str:
        """Insert a new open trade. Returns trade ID."""
        tid = str(uuid.uuid4())
        now = _now()
        self._conn.execute("""
            INSERT INTO trades (
                id, symbol, direction, entry_price, stop_loss, tp1_price, tp2_price,
                lot_size, status, open_time, strategy, confluence_score, timeframe,
                session, regime, news_score, raw_signal, original_sl, factors_json
            ) VALUES (
                :id, :symbol, :direction, :entry_price, :stop_loss, :tp1_price, :tp2_price,
                :lot_size, 'OPEN', :open_time, :strategy, :confluence_score, :timeframe,
                :session, :regime, :news_score, :raw_signal, :original_sl, :factors_json
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
            "session":          trade.get("session", "US Late 21-24 UTC"),
            "regime":           trade.get("regime", ""),
            "news_score":       trade.get("news_score"),
            "raw_signal":       json.dumps(trade.get("raw_signal", {})),
            "original_sl":      float(trade["stop_loss"]),
            "factors_json":     json.dumps(trade.get("factors", {})),
        })
        self._conn.commit()
        logger.info("Trade opened: %s %s %s @ %.2f",
                    tid[:8], trade["symbol"], trade["direction"], trade["entry_price"])
        return tid

    def close_trade(
        self,
        trade_id:    str,
        exit_price:  float,
        exit_reason: str = "MANUAL",
        pnl_usd:     float | None = None,
        rr_achieved: float | None = None,
        notes:       str = "",
    ) -> bool:
        """Mark a trade as CLOSED. Returns True if found and updated."""
        now = _now()
        try:
            with self._conn:
                cursor = self._conn.execute("""
                    UPDATE trades
                    SET status='CLOSED', close_time=:ct, exit_price=:ep,
                        exit_reason=:er, pnl_usd=:pnl, rr_achieved=:rr, notes=:notes
                    WHERE id=:id AND status='OPEN'
                """, {
                    "ct":    now,
                    "ep":    exit_price,
                    "er":    exit_reason,
                    "pnl":   pnl_usd,
                    "rr":    rr_achieved,
                    "notes": notes,
                    "id":    trade_id,
                })
                if pnl_usd is not None:
                    label = 1 if pnl_usd > 0 else 0
                    self._conn.execute(
                        "UPDATE ml_features SET label=? WHERE trade_id=?",
                        (label, trade_id)
                    )
        except Exception as exc:
            logger.error("close_trade failed for %s: %s", trade_id[:8], exc)
            return False
        updated = cursor.rowcount > 0
        if updated:
            logger.info("Trade closed: %s exit=%.2f reason=%s pnl=%s",
                        trade_id[:8], exit_price, exit_reason, pnl_usd)
        return updated

    def mark_tp1_hit(self, trade_id: str) -> None:
        self._conn.execute("UPDATE trades SET tp1_hit=1 WHERE id=?", (trade_id,))
        self._conn.commit()

    def mark_breakeven(self, trade_id: str) -> None:
        self._conn.execute("UPDATE trades SET be_moved=1 WHERE id=?", (trade_id,))
        self._conn.commit()

    def update_stop_loss(self, trade_id: str, new_sl: float) -> None:
        self._conn.execute(
            "UPDATE trades SET stop_loss=? WHERE id=? AND status='OPEN'",
            (new_sl, trade_id)
        )
        self._conn.commit()

    def get_open_trades(self, symbol: str | None = None) -> list[dict]:
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

    def get_trade(self, trade_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
        return dict(row) if row else None

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
            f"SELECT * FROM trades {where} ORDER BY open_time DESC LIMIT ?", params
        ).fetchall()
        return [dict(r) for r in rows]

    def log_signal(self, signal: dict, taken: bool = False, skip_reason: str = "") -> str:
        """Record a generated signal. Returns signal ID."""
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
            signal.get("stop_loss") or signal.get("sl"),
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
        """Return closed trades with features + labels for ML training."""
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
                result.append(features)
            except json.JSONDecodeError:
                continue
        return result

    def get_stats(self, days: int = 30) -> dict:
        """Win rate, PnL, profit factor for the last N days."""
        rows = self._conn.execute(
            "SELECT pnl_usd FROM trades WHERE status='CLOSED' "
            "AND close_time >= datetime('now', ?)",
            (f"-{days} days",)
        ).fetchall()
        if not rows:
            return {"trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "profit_factor": 0.0}
        pnls  = [r["pnl_usd"] or 0.0 for r in rows]
        wins  = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]
        wrate = len(wins) / len(pnls) if pnls else 0.0
        pf    = sum(wins) / sum(losses) if sum(losses) > 0 else float("inf")
        return {
            "trades":        len(pnls),
            "wins":          len(wins),
            "losses":        len(losses),
            "win_rate":      round(wrate * 100, 1),
            "total_pnl":     round(sum(pnls), 2),
            "avg_pnl":       round(sum(pnls) / len(pnls), 2),
            "profit_factor": round(pf, 2),
        }

    def get_paper_balance(self) -> float:
        """Return compounded paper balance (starts from STARTING_BALANCE)."""
        from btc_research.btc_bot_1.settings import STARTING_BALANCE
        rows = self._conn.execute(
            "SELECT pnl_usd FROM trades WHERE status='CLOSED' ORDER BY close_time ASC"
        ).fetchall()
        balance = STARTING_BALANCE
        for r in rows:
            pnl = r["pnl_usd"] or 0.0
            balance = max(balance + pnl, 1.0)
        return round(balance, 2)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
