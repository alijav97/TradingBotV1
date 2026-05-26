"""
signals/entry_checklist.py — 5-gate signal validation for TradingBotV2.

Every signal must pass ALL 5 checks before being converted to a paper trade.
If any check fails, the signal is rejected with an explicit reason.

Ported from V1 entry_checklist.py — cleaned up:
  - No global _FF_CACHE mutable state (V1 issue)
  - Uses V2 news_filter directly
  - Uses V2 confluence engine result dict (not V1 format)

Usage:
    from v2.signals.entry_checklist import validate_entry
    result = validate_entry(signal, df)
    if result["passed"]:
        open_trade(signal)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd

from v2.intelligence.news_filter import is_high_impact_window

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_RR           = 2.0    # minimum reward-to-risk
MIN_RR_NEWS_FADE = 1.5    # exception for news-fade trades
MIN_CONFLUENCE   = 3.0    # minimum confluence score (out of 12)
NEWS_BLOCK_MIN   = 30     # block X minutes before high-impact event
NEWS_BLOCK_AFTER = 30     # block X minutes after high-impact event

GST = timezone(timedelta(hours=4))


def validate_entry(signal: dict, df: pd.DataFrame | None = None, skip_news: bool = False) -> dict:
    """
    Run all 5 entry checks on a signal.

    Parameters
    ----------
    signal : signal dict — must contain: direction, entry_price (or entry),
             stop_loss, score (confluence score), timeframe (optional)
    df     : OHLCV DataFrame for the instrument

    Returns
    -------
    {
        "passed":    bool,
        "checks":    {check_name: {"passed": bool, "reason": str}},
        "failed_at": str | None,   # name of first failing check
        "summary":   str,
    }
    """
    checks = {}

    direction   = str(signal.get("direction", "")).lower()
    entry       = float(signal.get("entry_price") or signal.get("entry", 0) or 0)
    sl          = float(signal.get("stop_loss", 0) or 0)
    tp1         = float(signal.get("tp1_price") or signal.get("tp1") or 0)
    score       = float(signal.get("score") or signal.get("confluence_score", 0) or 0)
    is_news_fade= bool(signal.get("is_news_fade", False))
    is_divergence = bool(signal.get("is_divergence", False))
    is_breakout = bool(signal.get("is_breakout", False))

    failed_at: str | None = None

    # ── CHECK 1: Trend Alignment ──────────────────────────────────────────────
    c1 = _check_trend(signal, df, is_divergence)
    checks["Trend Alignment"] = c1
    if not c1["passed"] and failed_at is None:
        failed_at = "Trend Alignment"

    # ── CHECK 2: Minimum Confluence Score ─────────────────────────────────────
    c2_passed = score >= MIN_CONFLUENCE
    checks["Minimum Confluence"] = {
        "passed": c2_passed,
        "reason": f"Score {score:.1f}/12 >= {MIN_CONFLUENCE}" if c2_passed
                  else f"Score {score:.1f}/12 < {MIN_CONFLUENCE} required",
    }
    if not c2_passed and failed_at is None:
        failed_at = "Minimum Confluence"

    # ── CHECK 3: Risk/Reward Ratio ────────────────────────────────────────────
    c3 = _check_rr(entry, sl, tp1, direction, is_news_fade)
    checks["Risk/Reward Ratio"] = c3
    if not c3["passed"] and failed_at is None:
        failed_at = "Risk/Reward Ratio"

    # ── CHECK 4: News Safety Window ───────────────────────────────────────────
    if skip_news:
        c4 = {"passed": True, "reason": "News check skipped (backtest mode)"}
    else:
        c4 = _check_news(is_news_fade)
    checks["News Safety"] = c4
    if not c4["passed"] and failed_at is None:
        failed_at = "News Safety"

    # ── CHECK 5: Spread / Liquidity ───────────────────────────────────────────
    if skip_news:  # backtest mode — skip real-time session check too
        c5 = {"passed": True, "reason": "Session check skipped (backtest mode)"}
    else:
        c5 = _check_session()
    checks["Session Quality"] = c5
    if not c5["passed"] and failed_at is None:
        failed_at = "Session Quality"

    passed = failed_at is None

    # Summary line
    pass_count = sum(1 for c in checks.values() if c["passed"])
    summary = f"{'PASS' if passed else 'FAIL'} {pass_count}/5 checks"
    if failed_at:
        summary += f" — blocked at: {failed_at}"

    logger.debug(
        "Entry checklist %s %s: %s",
        signal.get("symbol", ""), direction, summary
    )

    return {
        "passed":    passed,
        "checks":    checks,
        "failed_at": failed_at,
        "summary":   summary,
    }


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_trend(signal: dict, df: pd.DataFrame | None, is_divergence: bool) -> dict:
    """CHECK 1: Trend alignment — price structure should support direction."""
    direction = str(signal.get("direction", "")).lower()
    is_long   = direction in ("long", "buy")

    if is_divergence:
        return {"passed": True, "reason": "Divergence trade — trend check waived"}

    if df is None or df.empty or len(df) < 50:
        return {"passed": True, "reason": "No DataFrame — trend check skipped"}

    try:
        close  = df["close"]
        ema50  = close.ewm(span=50).mean()
        price  = float(close.iloc[-1])
        ema50v = float(ema50.iloc[-1])

        # Simple check: price on correct side of EMA50
        if is_long and price > ema50v:
            return {"passed": True, "reason": f"Price {price:.5f} > EMA50 {ema50v:.5f} — bullish structure"}
        if not is_long and price < ema50v:
            return {"passed": True, "reason": f"Price {price:.5f} < EMA50 {ema50v:.5f} — bearish structure"}

        # Allow if close (within 0.5% of EMA)
        drift_pct = abs(price - ema50v) / ema50v * 100
        if drift_pct < 0.5:
            return {"passed": True, "reason": f"Price near EMA50 ({drift_pct:.2f}%) — borderline pass"}

        return {
            "passed": False,
            "reason": f"Price on wrong side of EMA50 — structure doesn't support {direction}",
        }
    except Exception:
        return {"passed": True, "reason": "Trend check errored — skipped"}


def _check_rr(entry: float, sl: float, tp1: float, direction: str, is_news_fade: bool) -> dict:
    """CHECK 3: Minimum R:R ratio."""
    min_rr = MIN_RR_NEWS_FADE if is_news_fade else MIN_RR

    if entry <= 0 or sl <= 0 or tp1 <= 0:
        return {"passed": False, "reason": "Missing entry/SL/TP — cannot calculate R:R"}

    sl_dist = abs(entry - sl)
    tp_dist = abs(entry - tp1)
    if sl_dist <= 0:
        return {"passed": False, "reason": "SL distance is zero"}

    rr = round(tp_dist / sl_dist, 2)
    passed = rr >= min_rr
    return {
        "passed": passed,
        "reason": f"R:R = 1:{rr} {'≥' if passed else '<'} 1:{min_rr} minimum",
    }


def _check_news(is_news_fade: bool) -> dict:
    """CHECK 4: No high-impact news within window."""
    if is_news_fade:
        return {"passed": True, "reason": "News fade trade — news check waived"}

    try:
        blocked, event_name = is_high_impact_window(
            minutes_before=NEWS_BLOCK_MIN,
            minutes_after=NEWS_BLOCK_AFTER,
        )
        if blocked:
            return {"passed": False, "reason": f"High-impact event nearby: {event_name}"}
        return {"passed": True, "reason": "No high-impact events in window"}
    except Exception:
        return {"passed": True, "reason": "News check unavailable — skipped"}


def _check_session() -> dict:
    """CHECK 5: Trading in a viable session (not dead hours)."""
    now = datetime.now(GST)
    hour = now.hour

    # Dead zones: late NY / off-hours (22:00–04:00 GST)
    if hour >= 22 or hour < 4:
        return {"passed": False, "reason": f"Off-hours ({hour:02d}:xx GST) — no liquidity"}

    # Good sessions
    if 12 <= hour < 21:
        session = "London" if hour < 17 else "NewYork"
        return {"passed": True, "reason": f"Active session: {session}"}

    # Borderline (Asian pre-London)
    return {"passed": True, "reason": f"Borderline session ({hour:02d}:xx GST)"}
