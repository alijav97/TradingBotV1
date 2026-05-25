"""
api/api_server.py — FastAPI application for TradingBotV2.

All routes are protected by API key authentication (X-Api-Key header).
POST routes additionally require a "full" scope key.

Endpoints:
  GET  /health          Status + uptime
  GET  /signals         Last 20 signals (taken + skipped)
  GET  /trades          Filtered trade list (symbol, status, limit)
  GET  /trades/{id}     Single trade by ID
  GET  /performance     Win rate, PnL, profit factor + per-symbol breakdown
  GET  /portfolio       Open trades + portfolio heat summary
  GET  /instruments     All 6 instrument configs
  POST /settings/risk   Update RISK_PER_TRADE_PCT / DAILY_LOSS_LIMIT (full key)

Usage:
    uvicorn v2.api.api_server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from v2.api.api_keys import require_api_key, require_full_access
from v2.instrument_config import INSTRUMENTS

logger = logging.getLogger(__name__)

_START_TIME = time.monotonic()


# ── App state ─────────────────────────────────────────────────────────────────

@dataclass
class AppState:
    """Shared application state attached to app.state."""
    journal: Any = None
    feed: Any = None
    scheduler: Any = None
    portfolio_heat: Any = None


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop background services around the application lifecycle."""
    state: AppState = app.state.bot

    # ── Startup ───────────────────────────────────────────────────────────────
    try:
        from v2.journal.sqlite_journal import Journal
        state.journal = Journal()
        logger.info("Journal opened")
    except Exception as exc:
        logger.error("Failed to open Journal: %s", exc)

    try:
        from v2.connectors.unified_data import DataFeed
        state.feed = DataFeed()
        state.feed.connect()
        logger.info("DataFeed connected")
    except Exception as exc:
        logger.warning("DataFeed unavailable (API will still serve journal data): %s", exc)

    try:
        from v2.risk.portfolio_heat import PortfolioHeat
        state.portfolio_heat = PortfolioHeat(state.journal)
    except Exception as exc:
        logger.warning("PortfolioHeat init failed: %s", exc)

    try:
        from v2.scheduler.scheduler import BotScheduler
        # Scheduler requires paper_trader + confluence — skip if not available
        from v2.trading.paper_trader import PaperTrader
        from v2.signals.confluence_engine import ConfluenceEngine
        pt = PaperTrader(state.journal, state.feed)
        ce = ConfluenceEngine()
        state.scheduler = BotScheduler(pt, ce, state.journal, state.feed)
        state.scheduler.start()
        logger.info("BotScheduler started")
    except Exception as exc:
        logger.warning("Scheduler unavailable: %s", exc)

    logger.info("TradingBotV2 API server ready")

    yield  # application runs

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if state.scheduler is not None:
        try:
            state.scheduler.stop()
        except Exception as exc:
            logger.warning("Scheduler stop error: %s", exc)

    if state.feed is not None:
        try:
            state.feed.disconnect()
        except Exception as exc:
            logger.warning("DataFeed disconnect error: %s", exc)

    if state.journal is not None:
        try:
            state.journal.close()
        except Exception as exc:
            logger.warning("Journal close error: %s", exc)

    logger.info("TradingBotV2 API server shut down")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    _app = FastAPI(
        title="TradingBotV2 API",
        version="2.0",
        description="Private REST API for TradingBotV2. All routes require X-Api-Key header.",
        lifespan=lifespan,
    )

    # Attach shared state container
    _app.state.bot = AppState()

    # CORS — private API protected by key auth, allow all origins
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return _app


app = create_app()


# ── Request/response models ───────────────────────────────────────────────────

class RiskSettingsUpdate(BaseModel):
    risk_per_trade_pct: Optional[float] = None   # e.g. 1.5
    daily_loss_limit: Optional[float] = None     # e.g. 3.0


# ── Helper ────────────────────────────────────────────────────────────────────

def _journal():
    """Return the live Journal or raise 503."""
    j = app.state.bot.journal
    if j is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Journal not available",
        )
    return j


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health(key_info: dict = Depends(require_api_key)) -> dict:
    """Server health + uptime."""
    return {
        "status": "ok",
        "version": "2.0",
        "uptime_seconds": int(time.monotonic() - _START_TIME),
    }


@app.get("/signals")
def get_signals(key_info: dict = Depends(require_api_key)) -> list[dict]:
    """Return the 20 most recent signals (both taken and skipped)."""
    journal = _journal()
    rows = journal._conn.execute(
        """
        SELECT id, symbol, direction, entry_price, stop_loss, tp1_price,
               score, strategy, timeframe, generated_at, taken, skip_reason
        FROM signals
        ORDER BY generated_at DESC
        LIMIT 20
        """
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/trades")
def get_trades(
    symbol: Optional[str] = Query(default=None, description="Filter by symbol, e.g. XAUUSD"),
    status: Optional[str] = Query(default=None, description="Filter by status: OPEN, CLOSED, CANCELLED"),
    limit: int = Query(default=50, ge=1, le=500, description="Max rows to return"),
    key_info: dict = Depends(require_api_key),
) -> list[dict]:
    """Return trades filtered by symbol and/or status."""
    journal = _journal()
    return journal.get_trades(symbol=symbol, status=status, limit=limit)


@app.get("/trades/{trade_id}")
def get_trade(trade_id: str, key_info: dict = Depends(require_api_key)) -> dict:
    """Return a single trade by ID."""
    journal = _journal()
    trade = journal.get_trade(trade_id)
    if trade is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade '{trade_id}' not found",
        )
    return trade


@app.get("/performance")
def get_performance(key_info: dict = Depends(require_api_key)) -> dict:
    """Overall stats + per-symbol breakdown for the last 30 days."""
    journal = _journal()

    overall = journal.get_stats(days=30)

    by_symbol: dict[str, dict] = {}
    for sym in INSTRUMENTS:
        sym_stats = journal.get_stats(symbol=sym, days=30)
        if sym_stats.get("trades", 0) > 0:
            by_symbol[sym] = sym_stats

    return {
        "period_days": 30,
        "overall": overall,
        "by_symbol": by_symbol,
    }


@app.get("/portfolio")
def get_portfolio(key_info: dict = Depends(require_api_key)) -> dict:
    """Current open trades and portfolio heat summary."""
    journal = _journal()
    open_trades = journal.get_open_trades()

    heat_summary: dict = {}
    ph = app.state.bot.portfolio_heat
    if ph is not None:
        try:
            heat_summary = ph.get_heat_summary()
        except Exception as exc:
            logger.warning("PortfolioHeat.get_heat_summary error: %s", exc)

    return {
        "open_trades": open_trades,
        "heat": heat_summary,
    }


@app.get("/instruments")
def get_instruments(key_info: dict = Depends(require_api_key)) -> list[dict]:
    """Return configuration for all 6 instruments."""
    result = []
    for sym, cfg in INSTRUMENTS.items():
        result.append({
            "symbol":           cfg.symbol,
            "source":           cfg.source,
            "mt5_symbol":       cfg.mt5_symbol,
            "binance_symbol":   cfg.binance_symbol,
            "pip_value_usd":    cfg.pip_value_usd,
            "pip_size":         cfg.pip_size,
            "min_lot":          cfg.min_lot,
            "lot_step":         cfg.lot_step,
            "max_leverage":     cfg.max_leverage,
            "sessions":         cfg.sessions,
            "correlated_with":  cfg.correlated_with,
            "description":      cfg.description,
        })
    return result


@app.post("/settings/risk")
def update_risk_settings(
    body: RiskSettingsUpdate,
    key_info: dict = Depends(require_full_access),
) -> dict:
    """
    Update runtime risk parameters.  Requires a 'full' scope API key.

    Changes apply immediately via os.environ; they do NOT persist across
    restarts unless the environment is set externally.
    """
    import v2.settings as s

    updated: dict[str, Any] = {}

    if body.risk_per_trade_pct is not None:
        if not (0.1 <= body.risk_per_trade_pct <= 10.0):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="risk_per_trade_pct must be between 0.1 and 10.0",
            )
        os.environ["RISK_PER_TRADE_PCT"] = str(body.risk_per_trade_pct)
        s.RISK_PER_TRADE_PCT = body.risk_per_trade_pct
        updated["risk_per_trade_pct"] = body.risk_per_trade_pct
        logger.info("RISK_PER_TRADE_PCT updated to %.2f by key %s",
                    body.risk_per_trade_pct, key_info.get("name"))

    if body.daily_loss_limit is not None:
        if not (0.5 <= body.daily_loss_limit <= 20.0):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="daily_loss_limit must be between 0.5 and 20.0",
            )
        os.environ["DAILY_LOSS_LIMIT"] = str(body.daily_loss_limit)
        s.DAILY_LOSS_LIMIT = body.daily_loss_limit
        updated["daily_loss_limit"] = body.daily_loss_limit
        logger.info("DAILY_LOSS_LIMIT updated to %.2f by key %s",
                    body.daily_loss_limit, key_info.get("name"))

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid fields provided (risk_per_trade_pct, daily_loss_limit)",
        )

    return {"updated": updated}
