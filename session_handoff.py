"""
session_handoff.py — Session Handoff Engine for TradingBotV1

Analyses Asian session range → detects London break (real or fake)
→ generates NY session bias that feeds directional context into
every signal.

Functions:
  get_asian_session_range(df)       -> dict
  detect_london_break(df, asian)    -> dict
  get_ny_session_bias(df)           -> dict
  format_session_handoff(handoff)   -> str
"""

from datetime import datetime, timezone, timedelta

GST = timezone(timedelta(hours=4))   # UAE / Dubai time


# ══════════════════════════════════════════════════════════════════════════════
#  get_asian_session_range
# ══════════════════════════════════════════════════════════════════════════════

def get_asian_session_range(df) -> dict:
    """
    Calculates the Asian session range from the supplied OHLCV DataFrame.
    Asian session UAE time: 04:00–10:00 (UTC 00:00–06:00).

    Expects df.index to be timezone-aware UTC datetimes or naive UTC.
    Returns dict with keys: available, asian_high, asian_low, asian_range,
    asian_mid, asian_close, asian_open, asian_bias, candle_count, display_line.
    """
    try:
        import pandas as pd

        if df is None or df.empty:
            return {"available": False, "reason": "no data"}

        # Ensure UTC-aware index for slicing
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is None:
            idx = idx.tz_localize("UTC")
        elif hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert("UTC")

        df_utc = df.copy()
        df_utc.index = idx

        now_uae = datetime.now(GST)
        today   = now_uae.date()

        # Filter today's Asian session (UTC 00:00–06:00)
        asian_candles = df_utc[
            (df_utc.index.date == today) &
            (df_utc.index.hour >= 0) &
            (df_utc.index.hour < 6)
        ]

        # Fall back to yesterday if today's session not yet available
        if asian_candles.empty:
            yesterday     = today - timedelta(days=1)
            asian_candles = df_utc[
                (df_utc.index.date == yesterday) &
                (df_utc.index.hour >= 0) &
                (df_utc.index.hour < 6)
            ]

        if asian_candles.empty:
            return {"available": False, "reason": "no asian session candles found"}

        asian_high  = float(asian_candles["high"].max())
        asian_low   = float(asian_candles["low"].min())
        asian_range = asian_high - asian_low
        asian_mid   = (asian_high + asian_low) / 2
        asian_close = float(asian_candles["close"].iloc[-1])
        asian_open  = float(asian_candles["open"].iloc[0])

        if asian_close > asian_mid:
            asian_bias = "bullish"
        elif asian_close < asian_mid:
            asian_bias = "bearish"
        else:
            asian_bias = "neutral"

        return {
            "available":    True,
            "asian_high":   asian_high,
            "asian_low":    asian_low,
            "asian_range":  round(asian_range, 2),
            "asian_mid":    round(asian_mid, 2),
            "asian_close":  asian_close,
            "asian_open":   asian_open,
            "asian_bias":   asian_bias,
            "candle_count": len(asian_candles),
            "display_line": (
                f"Asian: ${asian_low:,.2f}–${asian_high:,.2f} "
                f"(range ${asian_range:.2f}) bias:{asian_bias}"
            ),
        }

    except Exception as e:
        return {"available": False, "reason": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  detect_london_break
# ══════════════════════════════════════════════════════════════════════════════

def detect_london_break(df, asian_range: dict) -> dict:
    """
    Detects if/how London broke the Asian session range.
    London session UAE time: 12:00–20:30 (UTC 08:00–16:30).

    Returns dict with keys: break_detected, break_direction, break_type,
    break_level, break_amount, london_high, london_low, london_close,
    ny_bias, note, fake_break_alert.
    """
    try:
        import pandas as pd

        if not asian_range.get("available"):
            return {"break_detected": False, "break_type": "unknown",
                    "ny_bias": "NEUTRAL", "fake_break_alert": False}

        asian_high = asian_range["asian_high"]
        asian_low  = asian_range["asian_low"]

        # Ensure UTC-aware index
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is None:
            idx = idx.tz_localize("UTC")
        elif hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert("UTC")

        df_utc = df.copy()
        df_utc.index = idx

        today = datetime.now(GST).date()

        # UTC 08:00–16:00 (covers the active London open window)
        london_candles = df_utc[
            (df_utc.index.date == today) &
            (df_utc.index.hour >= 8) &
            (df_utc.index.hour < 16)
        ]

        if london_candles.empty:
            return {
                "break_detected": False,
                "break_type":     "no_data",
                "ny_bias":        "NEUTRAL",
                "fake_break_alert": False,
                "note":           "London session not started yet",
            }

        london_high  = float(london_candles["high"].max())
        london_low   = float(london_candles["low"].min())
        london_open  = float(london_candles["open"].iloc[0])
        london_close = float(london_candles["close"].iloc[-1])

        broke_high = london_high > asian_high
        broke_low  = london_low  < asian_low

        # Default (in case neither branch runs)
        break_level  = None
        break_amount = 0.0

        if broke_high and not broke_low:
            break_direction = "UPWARD"
            break_level     = asian_high
            break_amount    = london_high - asian_high
            if london_close > asian_high:
                break_type = "REAL_BREAK"
                ny_bias    = "BULLISH"
                note = (
                    f"London broke AND closed above Asian high "
                    f"${asian_high:,.2f} — real breakout, "
                    f"NY likely continuation LONG"
                )
            else:
                break_type = "FAKE_BREAK"
                ny_bias    = "BEARISH"
                note = (
                    f"London spiked above ${asian_high:,.2f} "
                    f"but closed back inside — fake breakout, "
                    f"NY likely reversal SHORT"
                )

        elif broke_low and not broke_high:
            break_direction = "DOWNWARD"
            break_level     = asian_low
            break_amount    = asian_low - london_low
            if london_close < asian_low:
                break_type = "REAL_BREAK"
                ny_bias    = "BEARISH"
                note = (
                    f"London broke AND closed below Asian low "
                    f"${asian_low:,.2f} — real breakdown, "
                    f"NY likely continuation SHORT"
                )
            else:
                break_type = "FAKE_BREAK"
                ny_bias    = "BULLISH"
                note = (
                    f"London spiked below ${asian_low:,.2f} "
                    f"but closed back inside — fake breakdown, "
                    f"NY likely reversal LONG"
                )

        elif broke_high and broke_low:
            break_direction = "BOTH"
            break_type      = "VOLATILE"
            ny_bias         = "NEUTRAL"
            note = "London broke both sides — high volatility, no clear NY bias"

        else:
            break_direction = "NONE"
            break_type      = "INSIDE_RANGE"
            ny_bias         = "NEUTRAL"
            note = (
                "London trading inside Asian range — "
                "wait for break direction before NY"
            )

        return {
            "break_detected":   broke_high or broke_low,
            "break_direction":  break_direction,
            "break_type":       break_type,
            "break_level":      break_level,
            "break_amount":     break_amount,
            "london_high":      london_high,
            "london_low":       london_low,
            "london_open":      london_open,
            "london_close":     london_close,
            "ny_bias":          ny_bias,
            "note":             note,
            "fake_break_alert": break_type == "FAKE_BREAK",
        }

    except Exception as e:
        return {
            "break_detected": False,
            "break_type":     "error",
            "ny_bias":        "NEUTRAL",
            "fake_break_alert": False,
            "note":           str(e),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  get_ny_session_bias
# ══════════════════════════════════════════════════════════════════════════════

def get_ny_session_bias(df) -> dict:
    """
    Combines Asian session range + London break detection into a full
    NY session bias package.

    Returns dict with keys: asian_range, london_break, ny_bias, confidence,
    confidence_score, recommendation, action, fake_break_alert, summary.
    """
    asian  = get_asian_session_range(df)
    london = detect_london_break(df, asian)

    ny_bias    = london.get("ny_bias",    "NEUTRAL")
    break_type = london.get("break_type", "unknown")

    # Confidence tier
    if break_type == "REAL_BREAK":
        confidence       = "HIGH"
        confidence_score = 8.0
    elif break_type == "FAKE_BREAK":
        confidence       = "HIGH"
        confidence_score = 7.5
    elif break_type == "INSIDE_RANGE":
        confidence       = "LOW"
        confidence_score = 4.0
    else:
        confidence       = "MODERATE"
        confidence_score = 6.0

    # Trading recommendation
    if ny_bias == "BULLISH" and confidence == "HIGH":
        recommendation = "LOOK FOR LONGS in NY session"
        action         = "long"
    elif ny_bias == "BEARISH" and confidence == "HIGH":
        recommendation = "LOOK FOR SHORTS in NY session"
        action         = "short"
    elif ny_bias == "BULLISH":
        recommendation = "Slight LONG bias — confirm with signals"
        action         = "long_cautious"
    elif ny_bias == "BEARISH":
        recommendation = "Slight SHORT bias — confirm with signals"
        action         = "short_cautious"
    else:
        recommendation = "NEUTRAL — wait for London to break range"
        action         = "wait"

    summary = (
        f"Asian: ${asian.get('asian_low', 0):,.2f}–"
        f"${asian.get('asian_high', 0):,.2f} | "
        f"London: {break_type} | "
        f"NY bias: {ny_bias} ({confidence})"
    )

    return {
        "asian_range":      asian,
        "london_break":     london,
        "ny_bias":          ny_bias,
        "confidence":       confidence,
        "confidence_score": confidence_score,
        "recommendation":   recommendation,
        "action":           action,
        "fake_break_alert": london.get("fake_break_alert", False),
        "summary":          summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  format_session_handoff
# ══════════════════════════════════════════════════════════════════════════════

def format_session_handoff(handoff: dict) -> str:
    """Return a formatted session handoff analysis string."""
    SEP  = "═" * 36
    DASH = "─" * 36

    asian  = handoff.get("asian_range", {})
    london = handoff.get("london_break", {})

    asian_low   = asian.get("asian_low", 0)
    asian_high  = asian.get("asian_high", 0)
    asian_range = asian.get("asian_range", 0)
    asian_bias  = asian.get("asian_bias", "unknown").upper()

    break_type  = london.get("break_type", "UNKNOWN")
    break_dir   = london.get("break_direction", "NONE")
    note        = london.get("note", "")
    fake_alert  = london.get("fake_break_alert", False)

    ny_bias     = handoff.get("ny_bias", "NEUTRAL")
    confidence  = handoff.get("confidence", "—")
    recommend   = handoff.get("recommendation", "—")

    lines = [
        SEP,
        "  SESSION HANDOFF ANALYSIS",
        SEP,
        "",
        "  ASIAN SESSION (04:00–10:00 UAE):",
    ]

    if asian.get("available"):
        lines += [
            f"  Range: ${asian_low:,.2f} – ${asian_high:,.2f}",
            f"  Size:  ${asian_range:.2f} | Bias: {asian_bias}",
        ]
    else:
        lines.append(f"  Asian data unavailable: {asian.get('reason', '')}")

    lines += [
        "",
        "  LONDON BREAK (12:00–20:30 UAE):",
        f"  {break_type}: {break_dir}",
    ]

    if note:
        lines.append(f"  {note}")

    if fake_alert:
        lines.append("  ⚠ FAKE BREAK ALERT — consider reversal setup")

    lines += [
        "",
        "  NY SESSION BIAS:",
        f"  Direction:  {ny_bias}",
        f"  Confidence: {confidence}",
        "",
        f"  → {recommend}",
        DASH,
    ]

    return "\n".join(lines)
