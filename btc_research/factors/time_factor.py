"""
btc_research/factors/time_factor.py — Session timing factor for BTC.

BTC makes its biggest directional moves during:
  13:00-17:00 UTC  US market open (kill-zone) — highest score
  13:00-15:00 UTC  First 2 hours of US session — extra bonus (opening momentum)

Weaker periods:
  17:00-21:00 UTC  US mid-session — still tradeable, lower score
  00:00-08:00 UTC  Asia session   — BTC active but less directional, no score
  Weekends         — spread wider, less reliable moves, penalty

Note: time_factor is used both to:
  1. Gate entries (only trade inside kill-zone)
  2. Add a small score bonus for optimal timing
"""
from __future__ import annotations

from datetime import timezone
import pandas as pd


def compute_time_factor(bar_time: pd.Timestamp) -> dict:
    """
    Score the session timing of bar_time.

    Returns dict:
        score       : float   (0.0 to 2.5)
        reason      : str
        in_killzone : bool    (True only for 13:00-17:00 UTC)
        utc_hour    : int
        day_of_week : str     ("Mon" … "Sun")
    """
    # Ensure UTC-aware
    if bar_time.tzinfo is None:
        bar_time = bar_time.replace(tzinfo=timezone.utc)

    utc_hour = bar_time.hour
    utc_dow  = bar_time.weekday()   # 0 = Monday, 6 = Sunday

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # ── Weekend penalty ───────────────────────────────────────────────────────
    if utc_dow in (5, 6):
        return {
            "score":       -1.0,
            "reason":      "weekend - low liquidity",
            "in_killzone": False,
            "utc_hour":    utc_hour,
            "day_of_week": day_names[utc_dow],
        }

    score   = 0.0
    reasons: list[str] = []

    # ── Kill-zone: 13:00-17:00 UTC ─────────────────────────────────────────
    if 13 <= utc_hour < 17:
        score += 2.0
        reasons.append("kill-zone 13-17 UTC")
        # Best first 2 hours — opening momentum is strongest
        if utc_hour < 15:
            score += 0.5
            reasons.append("early KZ bonus 13-15")
    else:
        # Outside kill-zone — no entry allowed
        reasons.append(f"outside kill-zone (UTC {utc_hour:02d}:xx)")

    in_killzone = (13 <= utc_hour < 17)

    return {
        "score":       round(score, 1),
        "reason":      " | ".join(reasons),
        "in_killzone": in_killzone,
        "utc_hour":    utc_hour,
        "day_of_week": day_names[utc_dow],
    }
