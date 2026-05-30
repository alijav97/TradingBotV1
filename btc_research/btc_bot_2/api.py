"""
btc_research/btc_bot_2/api.py — FastAPI application for BTC Bot 2.

Runs on port 8002 (separate from Bot 1 on port 8000).
All routes are protected by the same X-Api-Key header mechanism as Bot 1.

Endpoints:
  GET  /health          Status, uptime, kill-zone info
  GET  /trades          Recent trades (limit, status filter)
  GET  /trades/{id}     Single trade
  GET  /performance     Win rate, PnL, Profit Factor, per-month breakdown
  GET  /open            Currently open trade(s)
  GET  /signal/latest   Most recent signal generated (taken or not)
  POST /scan            Manually trigger a signal scan (requires full key)

Usage (standalone):
    uvicorn btc_research.btc_bot_2.api:app --host 0.0.0.0 --port 8002

Usage (from main.py):
    import uvicorn
    from btc_research.btc_bot_2.api import app, set_app_state
    set_app_state(journal=journal, engine=engine, paper_trader=pt)
    uvicorn.run(app, host="0.0.0.0", port=8002)
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
_START_TIME = time.monotonic()


# ── API key auth ──────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    return os.environ.get("BTC2_API_KEY", os.environ.get("API_KEY", ""))

def _get_full_key() -> str:
    return os.environ.get("BTC2_API_KEY_FULL", os.environ.get("API_KEY_FULL", _get_api_key()))


def require_api_key(x_api_key: str = ""):
    from fastapi import Header
    key = _get_api_key()
    if not key:
        return   # no key configured → open access (dev mode)
    if x_api_key != key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

def require_full_access(x_api_key: str = ""):
    from fastapi import Header
    key = _get_full_key()
    if not key:
        return
    if x_api_key != key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Full access key required")


# ── App state (injected by main.py) ───────────────────────────────────────────

_state: dict[str, Any] = {}


def set_app_state(
    journal:      Any = None,
    engine:       Any = None,
    paper_trader: Any = None,
) -> None:
    """Inject live components. Call before uvicorn.run()."""
    _state["journal"]      = journal
    _state["engine"]       = engine
    _state["paper_trader"] = paper_trader


# ── App creation ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "BTC Bot 2 API",
    description = "BTC Bot 2 — Asia Night + EU Open  |  VB + Swing Level Break v2",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _journal():
    j = _state.get("journal")
    if j is None:
        raise HTTPException(status_code=503, detail="Journal not available")
    return j

def _engine():
    e = _state.get("engine")
    if e is None:
        raise HTTPException(status_code=503, detail="Signal engine not available")
    return e

def _paper_trader():
    pt = _state.get("paper_trader")
    if pt is None:
        raise HTTPException(status_code=503, detail="PaperTrader not available")
    return pt


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Bot status, uptime, and kill-zone info."""
    from btc_research.btc_bot_2.settings import KZ_HOURS
    from datetime import datetime, timezone
    import math

    now     = datetime.now(timezone.utc)
    in_kz   = now.hour in KZ_HOURS
    uptime  = time.monotonic() - _START_TIME
    h       = int(uptime // 3600)
    m       = int((uptime % 3600) // 60)

    return {
        "status":        "ok",
        "bot":           "BTC Bot 2",
        "strategy":      "VB + Swing Level Break v2 [both 2xATR]",
        "kill_zone_utc": KZ_HOURS,
        "in_kill_zone":  in_kz,
        "current_hour_utc": now.hour,
        "uptime":        f"{h}h {m}m",
        "timestamp_utc": now.isoformat(),
        "api_port":      8002,
    }


@app.get("/trades")
def get_trades(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit:         int           = Query(50, ge=1, le=500),
):
    """Get recent trades. status: OPEN | CLOSED | CANCELLED (omit for all)."""
    j = _journal()
    try:
        trades = j.get_trades(status=status_filter, limit=limit)
        return {"count": len(trades), "trades": trades}
    except Exception as exc:
        logger.error("get_trades error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/trades/{trade_id}")
def get_trade(trade_id: str):
    """Get a single trade by ID."""
    j = _journal()
    try:
        trade = j.get_trade(trade_id)
        if trade is None:
            raise HTTPException(status_code=404, detail="Trade not found")
        return trade
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/open")
def get_open():
    """Get currently open trade(s) and their status."""
    j  = _journal()
    pt = _state.get("paper_trader")
    try:
        trades  = j.get_trades(status="OPEN", limit=10)
        summary = {"count": len(trades), "trades": trades}
        return summary
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/performance")
def get_performance(days: int = Query(30, ge=1, le=9999)):
    """Win rate, PnL, Profit Factor for the last N days."""
    j = _journal()
    try:
        stats = j.get_stats(days=days)
        return {
            "period_days":   days,
            "trades":        stats.get("trades", 0),
            "win_rate_pct":  round(stats.get("win_rate", 0.0), 1),
            "total_pnl_usd": round(stats.get("total_pnl", 0.0), 2),
            "profit_factor": round(stats.get("profit_factor", 0.0), 2),
            "max_dd_pct":    round(stats.get("max_drawdown_pct", 0.0), 1),
            "avg_rr":        round(stats.get("avg_rr", 0.0), 2),
            "current_balance": round(stats.get("current_balance", 0.0), 2),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/scan")
def manual_scan():
    """
    Manually trigger a signal scan (POST, full API key required).
    Returns the signal if one fires, or null.
    """
    e  = _engine()
    pt = _paper_trader()
    try:
        signal = e.scan()
        if signal is None:
            return {"signal": None, "message": "No signal at current conditions"}

        # Optionally open a paper trade
        trade_id = pt.open_trade(signal)
        return {
            "signal":   signal,
            "trade_id": trade_id,
            "opened":   trade_id is not None,
        }
    except Exception as exc:
        logger.error("Manual scan error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
