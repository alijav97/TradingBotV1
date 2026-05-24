"""
session_profiler.py — Session Volatility Profiling for TradingBotV1
─────────────────────────────────────────────────────────────────────
Analyses historical OHLCV data and builds per-session volatility
profiles used to scale lot size, SL and TP targets.

Usage:
    from session_profiler import (
        build_session_profiles,
        get_current_session_profile,
        get_session_adjusted_position,
    )
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR     = os.path.join(_BASE_DIR, "data")
_PROFILES_FILE = os.path.join(_DATA_DIR, "session_profiles.json")

# ── Session hour bands (UTC) ───────────────────────────────────────────────────
_SESSION_HOURS: dict[str, range] = {
    "Asian":    range(0,  7),
    "London":   range(7,  12),
    "Overlap":  range(12, 15),
    "NewYork":  range(15, 21),
    "OffHours": range(21, 24),
}

# ── Baseline multipliers per session ──────────────────────────────────────────
_BASE_MULT: dict[str, dict[str, float]] = {
    "London":   {"lot": 1.0, "sl": 1.0, "tp": 1.0},
    "Overlap":  {"lot": 1.1, "sl": 1.0, "tp": 1.1},
    "NewYork":  {"lot": 1.0, "sl": 1.1, "tp": 1.0},
    "Asian":    {"lot": 0.7, "sl": 1.3, "tp": 0.8},
    "OffHours": {"lot": 0.3, "sl": 1.5, "tp": 0.7},
}

# ── Fallback profile (returned when no json file exists) ─────────────────────
def _fallback_profile(session: str) -> dict:
    m = _BASE_MULT.get(session, _BASE_MULT["London"])
    note_map = {
        "London":   "Prime session — full position",
        "Overlap":  "High volatility overlap — slightly larger targets",
        "NewYork":  "Volatile open — widen SL slightly",
        "Asian":    "Thin liquidity — reduced size, wider SL",
        "OffHours": "Off-hours — avoid or minimise exposure",
    }
    return {
        "session":                    session,
        "avg_atr":                    0.0,
        "avg_range":                  0.0,
        "avg_volume":                 0.0,
        "directional_bias":           50.0,
        "avg_move_pips":              0.0,
        "best_rr_achieved":           2.0,
        "win_rate_long":              50.0,
        "win_rate_short":             50.0,
        "recommended_lot_multiplier": m["lot"],
        "recommended_sl_multiplier":  m["sl"],
        "recommended_tp_multiplier":  m["tp"],
        "session_grade":              "B",
        "session_note":               note_map.get(session, ""),
        "source":                     "default",
    }


def _session_from_hour(hour: int) -> str:
    """Map UTC hour → session name."""
    for name, rng in _SESSION_HOURS.items():
        if hour in rng:
            return name
    return "OffHours"


def _compute_atr(df: pd.DataFrame) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  build_session_profiles
# ══════════════════════════════════════════════════════════════════════════════

def build_session_profiles(df: pd.DataFrame) -> dict:
    """
    Analyse historical OHLCV data and build a volatility profile for each
    trading session.  Saves result to data/session_profiles.json.

    Parameters
    ----------
    df : DataFrame with at minimum high/low/close/open/volume columns.
         Index or a 'datetime' column is used for hour extraction.

    Returns
    -------
    dict keyed by session name, each entry contains profile metrics and
    recommended lot/SL/TP multipliers.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ── Ensure ATR column ─────────────────────────────────────────────────────
    if "atr" not in df.columns or df["atr"].isna().all():
        df["atr"] = _compute_atr(df)

    # ── Extract hour ──────────────────────────────────────────────────────────
    if "hour" not in df.columns:
        if "datetime" in df.columns:
            df["hour"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.hour
        elif pd.api.types.is_datetime64_any_dtype(df.index):
            df["hour"] = df.index.hour
        else:
            df["hour"] = 8   # assume London if no time info

    df["hour"] = df["hour"].fillna(8).astype(int)
    df["session_label"] = df["hour"].apply(_session_from_hour)

    # ── London stats (baseline for ATR comparison) ────────────────────────────
    london_atr = df.loc[df["session_label"] == "London", "atr"].mean()
    if pd.isna(london_atr) or london_atr <= 0:
        london_atr = df["atr"].mean()
    if pd.isna(london_atr) or london_atr <= 0:
        london_atr = 1.0

    profiles: dict[str, dict] = {}

    for session in ("Asian", "London", "Overlap", "NewYork", "OffHours"):
        mask    = df["session_label"] == session
        sub     = df[mask].dropna(subset=["high", "low", "close", "open"])
        n       = len(sub)

        if n < 5:
            profiles[session] = _fallback_profile(session)
            profiles[session]["sample_size"] = n
            continue

        # ── Core metrics ─────────────────────────────────────────────────────
        avg_atr    = float(sub["atr"].mean()) if "atr" in sub else 0.0
        avg_range  = float((sub["high"] - sub["low"]).mean())
        avg_vol    = float(sub["volume"].mean()) if "volume" in sub else 0.0
        bullish    = (sub["close"] > sub["open"]).sum()
        dir_bias   = round(float(bullish) / n * 100, 1)
        avg_move   = float((sub["close"] - sub["open"]).abs().mean())
        avg_pips   = round(avg_move / 0.1, 1)

        # ── Best R:R from large moves (> 1.5x ATR) ───────────────────────────
        big_moves  = (sub["close"] - sub["open"]).abs()
        big_rr     = big_moves[big_moves > avg_atr * 1.5] / avg_atr if avg_atr > 0 else pd.Series([2.0])
        best_rr    = round(float(big_rr.median()) if not big_rr.empty else 2.0, 2)

        # ── Win-rate: buy/sell open, check close 24 bars later ───────────────
        future_close = sub["close"].shift(-24)
        long_wins    = (future_close > sub["open"]).sum()
        short_wins   = (future_close < sub["open"]).sum()
        valid        = future_close.notna().sum()
        win_long     = round(float(long_wins)  / max(valid, 1) * 100, 1)
        win_short    = round(float(short_wins) / max(valid, 1) * 100, 1)

        # ── Session grade ─────────────────────────────────────────────────────
        if avg_pips > 15 and dir_bias > 55:
            grade = "A"
        elif avg_pips > 10 or dir_bias > 52:
            grade = "B"
        else:
            grade = "C"

        # ── Multipliers — start from baseline then apply historical overrides ─
        base  = _BASE_MULT.get(session, _BASE_MULT["London"])
        lot_m = base["lot"]
        sl_m  = base["sl"]
        tp_m  = base["tp"]

        # ATR override: if this session is significantly more volatile, widen SL
        if avg_atr > london_atr * 1.3:
            sl_m = round(sl_m + 0.2, 2)

        # Win-rate overrides
        if win_long > 60:
            lot_m = round(lot_m + 0.1, 2)
        elif win_long < 40:
            lot_m = round(lot_m - 0.1, 2)

        # Build note
        note_parts = []
        if session == "OffHours":
            note_parts.append("Off-hours — avoid trading")
        elif session == "Asian":
            note_parts.append("Thin liquidity")
        if avg_atr > london_atr * 1.3:
            note_parts.append(f"high ATR ({avg_atr:.2f} vs London {london_atr:.2f})")
        if win_long > 60:
            note_parts.append(f"strong long edge ({win_long:.0f}%)")
        elif win_long < 40:
            note_parts.append(f"poor long rate ({win_long:.0f}%)")
        note = " | ".join(note_parts) if note_parts else f"{session} session"

        profiles[session] = {
            "session":                    session,
            "avg_atr":                    round(avg_atr, 4),
            "avg_range":                  round(avg_range, 4),
            "avg_volume":                 round(avg_vol, 2),
            "directional_bias":           dir_bias,
            "avg_move_pips":              avg_pips,
            "best_rr_achieved":           best_rr,
            "win_rate_long":              win_long,
            "win_rate_short":             win_short,
            "recommended_lot_multiplier": round(lot_m, 2),
            "recommended_sl_multiplier":  round(sl_m, 2),
            "recommended_tp_multiplier":  round(tp_m, 2),
            "session_grade":              grade,
            "session_note":               note,
            "sample_size":                n,
            "source":                     "historical",
        }

    # ── Save to disk ──────────────────────────────────────────────────────────
    os.makedirs(_DATA_DIR, exist_ok=True)
    out = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sessions":  profiles,
    }
    with open(_PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return profiles


# ══════════════════════════════════════════════════════════════════════════════
#  get_current_session_profile
# ══════════════════════════════════════════════════════════════════════════════

def get_current_session_profile() -> dict:
    """
    Returns the profile dict for the current UTC session plus convenience keys:
        current_session, session_grade, trading_recommended,
        lot_multiplier, sl_multiplier, tp_multiplier, session_note
    """
    hour    = datetime.now(timezone.utc).hour
    session = _session_from_hour(hour)

    # Try loading from disk
    profiles: dict = {}
    try:
        if os.path.exists(_PROFILES_FILE):
            with open(_PROFILES_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            profiles = raw.get("sessions", raw)   # support flat or nested
    except Exception:
        pass

    profile = profiles.get(session)
    if not profile:
        profile = _fallback_profile(session)

    grade   = profile.get("session_grade", "B")
    lot_m   = float(profile.get("recommended_lot_multiplier", _BASE_MULT.get(session, {}).get("lot", 1.0)))
    sl_m    = float(profile.get("recommended_sl_multiplier",  _BASE_MULT.get(session, {}).get("sl",  1.0)))
    tp_m    = float(profile.get("recommended_tp_multiplier",  _BASE_MULT.get(session, {}).get("tp",  1.0)))

    # trading_recommended: False for OffHours, or Asian with grade C
    trading_recommended = True
    if session == "OffHours":
        trading_recommended = False
    elif session == "Asian" and grade == "C":
        trading_recommended = False

    return {
        **profile,
        "current_session":      session,
        "session_grade":        grade,
        "trading_recommended":  trading_recommended,
        "lot_multiplier":       lot_m,
        "sl_multiplier":        sl_m,
        "tp_multiplier":        tp_m,
        "session_note":         profile.get("session_note", f"{session} session"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  get_session_adjusted_position
# ══════════════════════════════════════════════════════════════════════════════

def get_session_adjusted_position(
    base_lots:         float,
    base_sl_distance:  float,
    base_tp_distance:  float,
    session_profile:   dict,
) -> dict:
    """
    Apply session multipliers to base position sizing.

    Parameters
    ----------
    base_lots         : raw lots from calculate_position()
    base_sl_distance  : raw SL distance in $
    base_tp_distance  : raw TP distance in $
    session_profile   : dict from get_current_session_profile()

    Returns
    -------
    dict with adjusted_lots, adjusted_sl_distance, adjusted_tp_distance,
    lot_change (str), sl_change (str), tp_change (str), session_note (str).
    """
    lot_m = float(session_profile.get("lot_multiplier", 1.0))
    sl_m  = float(session_profile.get("sl_multiplier",  1.0))
    tp_m  = float(session_profile.get("tp_multiplier",  1.0))
    note  = session_profile.get("session_note", "")

    raw_lots = base_lots * lot_m

    # Hard caps
    raw_lots = max(0.01, raw_lots)
    raw_lots = min(raw_lots, base_lots * 1.5)   # never > 50% above base

    adj_lots  = round(raw_lots, 2)
    adj_sl    = round(base_sl_distance * sl_m, 4)
    adj_tp    = round(base_tp_distance * tp_m, 4)

    def _pct_str(mult: float) -> str:
        diff = round((mult - 1.0) * 100)
        return f"+{diff}%" if diff >= 0 else f"{diff}%"

    return {
        "adjusted_lots":         adj_lots,
        "adjusted_sl_distance":  adj_sl,
        "adjusted_tp_distance":  adj_tp,
        "lot_change":            _pct_str(lot_m),
        "sl_change":             _pct_str(sl_m),
        "tp_change":             _pct_str(tp_m),
        "session_note":          note,
        "lot_multiplier":        lot_m,
        "sl_multiplier":         sl_m,
        "tp_multiplier":         tp_m,
    }


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("Loading historical data...")
    try:
        _df = pd.read_csv(os.path.join(_DATA_DIR, "historical_xauusd.csv"))
        profiles = build_session_profiles(_df)
        print(f"\nSession profiles built ({len(profiles)} sessions):\n")
        for sess, p in profiles.items():
            g     = p["session_grade"]
            lot_m = p["recommended_lot_multiplier"]
            sl_m  = p["recommended_sl_multiplier"]
            tp_m  = p["recommended_tp_multiplier"]
            pips  = p["avg_move_pips"]
            wl    = p["win_rate_long"]
            n     = p.get("sample_size", "?")
            print(
                f"  {sess:<10} grade={g}  lot×{lot_m:.1f}  SL×{sl_m:.1f}  TP×{tp_m:.1f}"
                f"  pips={pips:.1f}  wr_long={wl:.0f}%  n={n}"
            )
        print()
        cur = get_current_session_profile()
        print(f"Current session: {cur['current_session']}  grade={cur['session_grade']}")
        print(f"  lot×{cur['lot_multiplier']}  SL×{cur['sl_multiplier']}  TP×{cur['tp_multiplier']}")
        print(f"  Trading recommended: {cur['trading_recommended']}")
        print(f"  Note: {cur['session_note']}")

        print()
        adj = get_session_adjusted_position(0.10, 20.0, 60.0, cur)
        print(f"Position adjustment (base 0.10 lots, $20 SL, $60 TP):")
        print(f"  Adjusted lots : {adj['adjusted_lots']}  ({adj['lot_change']})")
        print(f"  Adjusted SL   : ${adj['adjusted_sl_distance']:.2f}  ({adj['sl_change']})")
        print(f"  Adjusted TP   : ${adj['adjusted_tp_distance']:.2f}  ({adj['tp_change']})")

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
