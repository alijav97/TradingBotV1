"""
btc_research/strategy/confluence.py — BTC inter-market confluence entry engine.

Entry logic (same structure as WTI strategy, different factors):
  1. Must be inside kill-zone (13:00-17:00 UTC)
  2. Morning range (08:00-13:00 UTC) must have >= 3 bars on current day
  3. Price must CLOSE ABOVE morning range high (long)
     or CLOSE BELOW morning range low (short)
  4. BTC own trend must align (D1 equivalent must be on-side)
  5. Combined confluence score (BTC + Gold + Nasdaq + Time) >= MIN_SCORE

Exit levels:
  - SL  : opposite side of morning range (long -> below MR low, short -> above MR high)
  - TP1 : entry + 2 * SL-distance   (1:2 R:R)
  - TP2 : entry + 5 * SL-distance   (1:5 R:R)

Factor weights in total score:
  BTC momentum  : 1.0x  (own trend — most important, never trade against it)
  Gold factor   : 0.5x  (inter-market confirmation)
  Nasdaq factor : 0.5x  (risk-on/off environment)
  Time factor   : 0.3x  (session bonus)
"""
from __future__ import annotations

from datetime import timezone
import pandas as pd

from btc_research.settings import (
    KZ_START_UTC, KZ_END_UTC,
    MR_START_UTC, MR_END_UTC,
    TP1_RR, TP2_RR,
    MIN_CONFLUENCE_SCORE,
)
from btc_research.factors.gold_factor   import compute_gold_factor
from btc_research.factors.nasdaq_factor import compute_nasdaq_factor
from btc_research.factors.btc_momentum  import compute_btc_momentum
from btc_research.factors.time_factor   import compute_time_factor


def _morning_range(df_btc: pd.DataFrame, bar_date) -> dict | None:
    """
    Compute the morning range (08:00-13:00 UTC) for bar_date.

    Returns {"high": float, "low": float, "bars": int} or None if < 3 bars.
    """
    if hasattr(bar_date, "date"):
        date_obj = bar_date.date()
    else:
        date_obj = bar_date

    times = pd.to_datetime(df_btc["time"])
    mask  = (
        (times.dt.date  == date_obj)  &
        (times.dt.hour  >= MR_START_UTC) &
        (times.dt.hour  <  MR_END_UTC)
    )
    mr_bars = df_btc[mask]

    if len(mr_bars) < 3:
        return None

    return {
        "high": float(mr_bars["high"].max()),
        "low":  float(mr_bars["low"].min()),
        "bars": len(mr_bars),
    }


def score_bar(
    bar_time:  pd.Timestamp,
    bar_close: float,
    direction: str,
    df_btc:    pd.DataFrame,
    df_gold:   pd.DataFrame,
    df_nas:    pd.DataFrame,
) -> dict:
    """
    Score a single bar / direction combination.

    Args:
        bar_time  : UTC-aware timestamp of the current bar
        bar_close : closing price of the current bar
        direction : "long" or "short"
        df_btc    : BTCUSD H1 history UP TO AND INCLUDING bar_time
        df_gold   : XAUUSD H1 history (full 2-year set, factor will slice internally)
        df_nas    : NAS100 H1 history (full 2-year set)

    Returns dict:
        signal    : bool   - True if entry conditions are met
        score     : float  - total confluence score
        entry     : float  - entry price (bar close)
        sl        : float  - stop loss
        tp1       : float  - take-profit 1
        tp2       : float  - take-profit 2
        factors   : dict   - per-factor detail (btc_momentum, gold, nasdaq, time, mr)
        blocked_by: str    - rejection reason (empty when signal=True)
    """
    result = {
        "signal":     False,
        "score":      0.0,
        "entry":      bar_close,
        "sl":         0.0,
        "tp1":        0.0,
        "tp2":        0.0,
        "factors":    {},
        "blocked_by": "",
    }

    # Ensure UTC-aware
    if bar_time.tzinfo is None:
        bar_time = bar_time.replace(tzinfo=timezone.utc)

    # ── 1. Session gate ────────────────────────────────────────────────────────
    time_f = compute_time_factor(bar_time)
    if not time_f["in_killzone"]:
        result["blocked_by"] = f"outside kill-zone (UTC {bar_time.hour:02d}:xx)"
        return result

    # ── 2. Morning range check ─────────────────────────────────────────────────
    mr = _morning_range(df_btc, bar_time)
    if mr is None:
        result["blocked_by"] = "no morning range (< 3 bars in 08:00-13:00 UTC)"
        return result

    mr_high = mr["high"]
    mr_low  = mr["low"]
    is_long = direction.lower() in ("long", "buy")

    # ── 3. Breakout condition ─────────────────────────────────────────────────
    if is_long and bar_close <= mr_high:
        result["blocked_by"] = (
            f"no long breakout: close {bar_close:.2f} <= MR high {mr_high:.2f}"
        )
        return result
    if not is_long and bar_close >= mr_low:
        result["blocked_by"] = (
            f"no short breakout: close {bar_close:.2f} >= MR low {mr_low:.2f}"
        )
        return result

    # ── 4. SL / TP levels ─────────────────────────────────────────────────────
    entry   = bar_close
    sl      = mr_low  if is_long else mr_high
    sl_dist = abs(entry - sl)

    if sl_dist <= 0:
        result["blocked_by"] = "zero SL distance"
        return result

    tp1 = entry + TP1_RR * sl_dist if is_long else entry - TP1_RR * sl_dist
    tp2 = entry + TP2_RR * sl_dist if is_long else entry - TP2_RR * sl_dist

    # ── 5. Factor scoring ─────────────────────────────────────────────────────
    btc_f  = compute_btc_momentum(df_btc, bar_time, direction)
    gold_f = compute_gold_factor(df_gold, bar_time)
    nas_f  = compute_nasdaq_factor(df_nas, bar_time)

    # Adjust gold/NAS scores by direction
    # (gold_factor is already BTC-directional, NAS_factor is already BTC-directional)
    # For shorts: invert gold/NAS scores (if gold is falling that's BTC bullish,
    #             which is bad for a short trade)
    direction_multiplier = 1.0 if is_long else -1.0

    total_score = (
        btc_f["score"]  * 1.0                              +   # BTC own trend
        gold_f["score"] * 0.5 * direction_multiplier       +   # Gold factor
        nas_f["score"]  * 0.5 * direction_multiplier       +   # Nasdaq factor
        time_f["score"] * 0.3                                   # Session timing bonus
    )

    result.update({
        "score":   round(total_score, 2),
        "entry":   round(entry, 2),
        "sl":      round(sl, 2),
        "tp1":     round(tp1, 2),
        "tp2":     round(tp2, 2),
        "factors": {
            "btc_momentum":  btc_f,
            "gold":          gold_f,
            "nasdaq":        nas_f,
            "time":          time_f,
            "morning_range": mr,
        },
    })

    if total_score < MIN_CONFLUENCE_SCORE:
        result["blocked_by"] = (
            f"score {total_score:.1f} < threshold {MIN_CONFLUENCE_SCORE}"
        )
        return result

    result["signal"] = True
    return result
