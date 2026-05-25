"""
api/api_keys.py — API key manager for TradingBotV2.

Keys are stored in a separate table (api_keys) in the same SQLite database
used by the Journal.  Each key is a UUID-based string with an optional scope.

Scopes:
  "read"  — GET requests only
  "full"  — GET + POST (settings changes, etc.)

Usage:
    from v2.api.api_keys import APIKeyManager
    mgr = APIKeyManager(db_path)
    key = mgr.create_key("dashboard", scope="read")
    info = mgr.validate_key(key)   # dict or None
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS api_keys (
    key         TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    scope       TEXT NOT NULL DEFAULT 'read',   -- 'read' | 'full'
    created_at  TEXT NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0       -- 0 = active, 1 = revoked
);

CREATE INDEX IF NOT EXISTS idx_api_keys_key ON api_keys(key);
"""


class APIKeyManager:
    """Manages API keys stored in a SQLite table."""

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
        logger.info("APIKeyManager ready: %s", self.db_path)

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Public interface ──────────────────────────────────────────────────────

    def create_key(self, name: str, scope: str = "read") -> str:
        """
        Create a new API key.

        Args:
            name:  Human-readable label (e.g. "dashboard", "mobile_app").
            scope: "read" (GET only) or "full" (GET + POST).

        Returns:
            The raw key string (UUID4 hex, no dashes).
        """
        if scope not in ("read", "full"):
            raise ValueError(f"Invalid scope '{scope}': must be 'read' or 'full'")

        key = uuid.uuid4().hex  # 32-char hex string, no dashes
        now = _now()
        self._conn.execute(
            "INSERT INTO api_keys (key, name, scope, created_at, revoked) VALUES (?, ?, ?, ?, 0)",
            (key, name, scope, now),
        )
        self._conn.commit()
        logger.info("API key created: name=%s scope=%s key=%.8s...", name, scope, key)
        return key

    def validate_key(self, key: str) -> Optional[dict]:
        """
        Validate a key and return its info dict, or None if invalid/revoked.

        Returned dict keys: key, name, scope, created_at, revoked
        """
        if not key:
            return None
        row = self._conn.execute(
            "SELECT * FROM api_keys WHERE key=? AND revoked=0",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def revoke_key(self, key: str) -> bool:
        """
        Revoke an API key.  Returns True if a key was found and revoked.
        """
        cursor = self._conn.execute(
            "UPDATE api_keys SET revoked=1 WHERE key=? AND revoked=0",
            (key,),
        )
        self._conn.commit()
        revoked = cursor.rowcount > 0
        if revoked:
            logger.info("API key revoked: %.8s...", key)
        return revoked

    def list_keys(self) -> list[dict]:
        """Return all keys (including revoked) ordered by creation time."""
        rows = self._conn.execute(
            "SELECT key, name, scope, created_at, revoked FROM api_keys ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "APIKeyManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── FastAPI dependency functions ───────────────────────────────────────────────

def _get_key_manager() -> APIKeyManager:
    """Return the module-level singleton manager (lazily created)."""
    global _key_manager_singleton
    if _key_manager_singleton is None:
        _key_manager_singleton = APIKeyManager()
    return _key_manager_singleton


_key_manager_singleton: Optional[APIKeyManager] = None


def require_api_key(x_api_key: str = Header(...)) -> dict:
    """
    FastAPI dependency: validates the X-Api-Key header.

    Raises HTTP 401 if the key is missing or invalid.
    Returns the key info dict on success.
    """
    mgr = _get_key_manager()
    info = mgr.validate_key(x_api_key)
    if info is None:
        logger.warning("Rejected API request — invalid key: %.8s...", x_api_key or "")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
        )
    return info


def require_full_access(key_info: dict = Depends(require_api_key)) -> dict:
    """
    FastAPI dependency (stacked on require_api_key): ensures scope == 'full'.

    Raises HTTP 403 if the key only has read access.
    Returns the key info dict on success.
    """
    if key_info.get("scope") != "full":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint requires a full-access API key",
        )
    return key_info


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
