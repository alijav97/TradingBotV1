"""
reversal_hunter.py
──────────────────
Catches big reversal moves (e.g. XAUUSD $4,460→$4,560) that the main
signal engine misses by scoring 7 independent reversal conditions.

Main entry point:
    hunt_reversals(df, current_price=None) -> list[dict]

Returns signals when score >= 5 (max 11).
"""

from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
def hunt_reversals(df: Any, current_price: float | None = None) -> list[dict]:
    """
    Score 7 reversal conditions against the supplied DataFrame (H1 OHLCV
    with columns: open, high, low, close, volume, rsi, atr, ema200).

    Returns a list of signal dicts (may be empty) — never raises.
    """
    try:
        return _hunt(df, current_price)
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
def _hunt(df: Any, current_price: float | None) -> list[dict]:
    import pandas as pd  # noqa: F401 — used for type-checking only

    if df is None or len(df) < 10:
        return []

    score:          int        = 0
    direction:      str        = ""
    conditions_met: list[str]  = []
    wait_for_event: bool       = False

    # ── CONDITION 1 — RSI Exhaustion (max 2 pts) ─────────────────────────────
    try:
        rsi_now  = float(df["rsi"].iloc[-1])
        rsi_3ago = float(df["rsi"].iloc[-3])

        if rsi_now < 35 and rsi_now > rsi_3ago:
            score     += 2
            direction  = "long"
            conditions_met.append(f"RSI oversold {rsi_now:.1f} and turning up")
        elif rsi_now > 65 and rsi_now < rsi_3ago:
            score     += 2
            direction  = "short"
            conditions_met.append(f"RSI overbought {rsi_now:.1f} and turning down")
    except Exception:
        pass

    # ── CONDITION 1B — RSI Divergence (max 2 pts, independent of Cond 1) ─────
    try:
        from confluence_engine import detect_rsi_divergence as _detect_div
        div = _detect_div(df)
        if div.get("divergence_found"):
            div_dir      = div.get("signal_direction", "")
            div_strength = div.get("strength", "MODERATE")
            div_note     = div.get("note", "")

            if div_dir == "long" and div_strength == "STRONG":
                score += 2
                conditions_met.append(f"Strong bullish RSI divergence — {div_note}")
                if not direction:
                    direction = "long"
            elif div_dir == "long":
                score += 1
                conditions_met.append("Moderate bullish RSI divergence")
                if not direction:
                    direction = "long"
            elif div_dir == "short" and div_strength == "STRONG":
                score += 2
                conditions_met.append(f"Strong bearish RSI divergence — {div_note}")
                if not direction:
                    direction = "short"
            elif div_dir == "short":
                score += 1
                conditions_met.append("Moderate bearish RSI divergence")
                if not direction:
                    direction = "short"
    except Exception:
        pass


    try:
        atr        = float(df["atr"].iloc[-1])
        move_3bars = float(df["close"].iloc[-1]) - float(df["close"].iloc[-4])

        if abs(move_3bars) > atr * 2.5:
            score += 2
            if move_3bars < 0:
                conditions_met.append(
                    f"Exhaustion drop ${abs(move_3bars):.1f} in 3 bars — bounce expected"
                )
                if not direction:
                    direction = "long"
            else:
                conditions_met.append(
                    f"Exhaustion spike ${move_3bars:.1f} in 3 bars — fade expected"
                )
                if not direction:
                    direction = "short"
    except Exception:
        pass

    # ── CONDITION 3 — D1 vs H1 Conflict (max 2 pts) ──────────────────────────
    try:
        from mtf_analyzer import MultiTimeframeAnalyzer
        mta      = MultiTimeframeAnalyzer()
        htf      = mta.get_htf_bias("GC=F")
        d1_trend = htf.get("d1_trend", "ranging")

        h1_bullish = float(df["close"].iloc[-1]) > float(df["ema200"].iloc[-1])

        if d1_trend == "bullish" and not h1_bullish:
            score += 2
            conditions_met.append("D1 bullish but H1 bearish — buy the dip")
            if not direction:
                direction = "long"
        elif d1_trend == "bearish" and h1_bullish:
            score += 2
            conditions_met.append("D1 bearish but H1 bullish — sell the rally")
            if not direction:
                direction = "short"
    except Exception:
        pass

    # ── CONDITION 4 — Volume Climax (max 2 pts) ───────────────────────────────
    try:
        from volume_analyzer import VolumeAnalyzer
        climax = VolumeAnalyzer().detect_volume_climax(df)
        if climax.get("climax_detected"):
            score += 2
            if climax.get("type") == "selling_climax":
                conditions_met.append("Selling climax — institutional absorption")
                if not direction:
                    direction = "long"
            else:
                conditions_met.append("Buying climax — distribution detected")
                if not direction:
                    direction = "short"
    except Exception:
        pass

    # ── CONDITION 5 — Liquidity Pool Nearby (max 1 pt) ───────────────────────
    try:
        from smart_money import SmartMoneyAnalyzer
        liq   = SmartMoneyAnalyzer().find_liquidity_levels(df)
        price = float(df["close"].iloc[-1])

        for lvl in liq.get("sell_side_liquidity", []):
            if abs(price - float(lvl)) / price < 0.003:
                score += 1
                conditions_met.append(f"Price at SSL liquidity pool ${float(lvl):,.2f}")
                if not direction:
                    direction = "long"
                break

        for lvl in liq.get("buy_side_liquidity", []):
            if abs(price - float(lvl)) / price < 0.003:
                score += 1
                conditions_met.append(f"Price at BSL liquidity pool ${float(lvl):,.2f}")
                if not direction:
                    direction = "short"
                break
    except Exception:
        pass

    # ── CONDITION 6 — Premium/Discount Zone (max 1 pt) ───────────────────────
    try:
        from smart_money import SmartMoneyAnalyzer
        pd_zones  = SmartMoneyAnalyzer().find_premium_discount_zones(df)
        zone      = pd_zones.get("current_zone", "equilibrium")

        if zone == "discount":
            score += 1
            lo   = pd_zones.get("lowest_low",   0)
            eq   = pd_zones.get("equilibrium",  0)
            conditions_met.append(
                f"In discount zone ${lo:,.2f}-${eq:,.2f} — institutional buy area"
            )
            if not direction:
                direction = "long"
        elif zone == "premium":
            score += 1
            eq   = pd_zones.get("equilibrium",    0)
            hi   = pd_zones.get("highest_high",   0)
            conditions_met.append(
                f"In premium zone ${eq:,.2f}-${hi:,.2f} — institutional sell area"
            )
            if not direction:
                direction = "short"
    except Exception:
        pass

    # ── CONDITION 7 — News / Event Catalyst (max 1 pt) ───────────────────────
    try:
        from news_filter import get_todays_events
        events   = get_todays_events(impact_filter={"High"})
        fomc_kw  = ["fomc", "fed", "cpi", "nfp", "interest rate", "inflation"]

        for ev in events:
            if any(kw in str(ev.get("title", "")).lower() for kw in fomc_kw):
                score         += 1
                wait_for_event = True
                conditions_met.append(
                    f"High impact event: {ev['title']} at {ev.get('time_gst', ev.get('time_utc','?'))}"
                )
                break
    except Exception:
        pass

    # ── Skip weak / no-direction signals ─────────────────────────────────────
    if score < 5 or not direction:
        return []

    # ── Strength label ────────────────────────────────────────────────────────
    if score >= 7:
        strength = "STRONG"
    else:
        strength = "MODERATE"   # score 5 or 6

    # ── Entry / SL / TP via ATR ───────────────────────────────────────────────
    try:
        atr_val = float(df["atr"].iloc[-1])
    except Exception:
        atr_val = 20.0

    entry = round(float(current_price) if current_price else float(df["close"].iloc[-1]), 2)

    if direction == "long":
        recent_low = float(df["low"].tail(10).min())
        sl_price   = round(recent_low - atr_val * 0.5, 2)
    else:
        recent_high = float(df["high"].tail(10).max())
        sl_price    = round(recent_high + atr_val * 0.5, 2)

    sl_dist = round(abs(entry - sl_price), 2)
    if sl_dist == 0:
        sl_dist = atr_val

    if direction == "long":
        tp1 = round(entry + sl_dist * 2, 2)
        tp2 = round(entry + sl_dist * 3, 2)
    else:
        tp1 = round(entry - sl_dist * 2, 2)
        tp2 = round(entry - sl_dist * 3, 2)

    signal: dict = {
        "source":             "reversal_hunter",
        "type":               "reversal",
        "reversal_direction": direction,
        "reversal_strength":  strength,
        "score":              score,
        "max_score":          13,
        "conditions_met":     conditions_met,
        "pattern_name":       f"Reversal Hunter ({strength})",
        "direction":          direction,
        "asset":              "XAUUSD",
        "entry":              entry,
        "stop_loss":          sl_price,
        "take_profit":        tp2,
        "sl_distance":        sl_dist,
        "confidence":         round(score / 11 * 10, 1),
        "wait_for_event":     wait_for_event,
        "key_reason":         conditions_met[0] if conditions_met else "",
        "note":               f"Reversal signal {score}/11 — {strength}",
    }

    return [signal]
