"""
entry_checklist.py
──────────────────
Every signal — from rules.json or strategy_playbooks.py — must pass this
5-point checklist before appearing in morning_briefing.py.

If a signal fails ANY single check it is REJECTED immediately.

Usage:
    from entry_checklist import validate_entry, print_checklist, quick_score

    result = validate_entry(signal, df)
    print_checklist(signal, result)

Signal dict expected keys (all optional — missing keys are handled gracefully):
    direction        : "long" | "short"
    entry            : float
    stop_loss        : float  (price)
    take_profit      : float  (price)
    pattern_name     : str
    asset            : str
    confidence_score : float  (0–10)
    is_divergence    : bool   (exception flag for CHECK 1)
    is_news_fade     : bool   (exception flag for CHECK 3)
    is_breakout      : bool   (exception flag for CHECK 5)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ── Local imports ─────────────────────────────────────────────────────────────
try:
    from confluence_engine import score_confluences
    _CONFLUENCE_AVAILABLE = True
except ImportError:
    _CONFLUENCE_AVAILABLE = False

try:
    from news_filter import fetch_ff_calendar
    _NEWS_AVAILABLE = True
except ImportError:
    _NEWS_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_RR_DEFAULT      = 2.0    # minimum reward-to-risk ratio
MIN_RR_NEWS_FADE    = 1.5    # exception for news-fade trades
MIN_CONFLUENCE      = 3      # minimum confluence score required
NEWS_BLOCK_BEFORE   = 90     # minutes before high-impact event = FAIL
NEWS_BLOCK_AFTER    = 30     # minutes after  high-impact event = FAIL

# Cache FF calendar for the session (avoid rate-limiting on repeated calls)
_FF_CACHE: list[dict] | None = None
_FF_CACHE_DATE: str = ""

SESSIONS_PASS = {
    "London":   (7,  12),
    "NewYork":  (13, 17),
    "Overlap":  (12, 15),
}
SESSIONS_FAIL = {
    "Asian":    (0,  7),
}

CHECK_NAMES = [
    "Trend Alignment",
    "Minimum Confluence Score",
    "Risk/Reward Ratio",
    "News Safety Window",
    "Session Quality",
]

W = 45   # display width


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: enrich df with indicator columns if missing
# ══════════════════════════════════════════════════════════════════════════════

def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "open" not in df.columns:
        df["open"] = df["close"].shift(1).fillna(df["close"])
    if "ema50"  not in df.columns:
        df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    if "ema200" not in df.columns:
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    if "rsi" not in df.columns:
        delta       = df["close"].diff()
        gain        = delta.clip(lower=0).rolling(14).mean()
        loss        = (-delta.clip(upper=0)).rolling(14).mean()
        rs          = gain / loss.replace(0, np.nan)
        df["rsi"]   = 100 - (100 / (1 + rs))
    if "atr" not in df.columns:
        hl          = df["high"] - df["low"]
        hc          = (df["high"] - df["close"].shift()).abs()
        lc          = (df["low"]  - df["close"].shift()).abs()
        df["atr"]   = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    if "macd" not in df.columns:
        e12               = df["close"].ewm(span=12, adjust=False).mean()
        e26               = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"]        = e12 - e26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df.dropna(subset=["ema200", "rsi", "atr"])


def _sig(signal: dict, key: str, default: Any = None) -> Any:
    return signal.get(key, default)


# ══════════════════════════════════════════════════════════════════════════════
#  EntryChecklist
# ══════════════════════════════════════════════════════════════════════════════

class EntryChecklist:
    """
    Runs 5 independent gate checks on a trading signal.
    ALL 5 must pass — a single failure rejects the signal.
    """

    # ── CHECK 1 — Trend Alignment ─────────────────────────────────────────────
    def check_trend_alignment(
        self,
        signal: dict,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Is the trade in the direction of the H4 EMA200 trend?

        Long  → needs price above EMA200.
        Short → needs price below EMA200.
        Exception: RSI divergence signals can trade counter-trend (pass with -1 confidence).
        """
        direction    = str(_sig(signal, "direction", "long")).lower()
        is_long      = direction in ("long", "buy")
        is_divergence= bool(_sig(signal, "is_divergence", False))

        row    = df.iloc[-1]
        price  = float(row["close"])
        ema200 = float(row.get("ema200", float("nan")))

        if math.isnan(ema200):
            return {
                "passed":     False,
                "check_name": "Trend Alignment",
                "detail":     "EMA200 unavailable — cannot verify trend",
                "h4_trend":   "unknown",
                "confidence_adj": 0,
            }

        is_bullish = price > ema200
        aligned    = (is_long and is_bullish) or (not is_long and not is_bullish)
        h4_trend   = "BULLISH" if is_bullish else "BEARISH"
        trade_dir  = "LONG" if is_long else "SHORT"

        if aligned:
            return {
                "passed":     True,
                "check_name": "Trend Aligned",
                "detail":     f"H4 trend: {h4_trend} | Trade: {trade_dir} ✓",
                "h4_trend":   h4_trend,
                "confidence_adj": 0,
            }
        elif is_divergence:
            return {
                "passed":     True,   # EXCEPTION: divergence can trade counter-trend
                "check_name": "Trend Aligned (Counter-Trend Exception)",
                "detail":     (
                    f"H4 trend: {h4_trend} | Trade: {trade_dir} ← COUNTER-TREND\n"
                    f"    Exception: RSI divergence signal — confidence −1"
                ),
                "h4_trend":   h4_trend,
                "confidence_adj": -1,
            }
        else:
            return {
                "passed":     False,
                "check_name": "Trend Alignment FAILED",
                "detail":     f"H4 trend: {h4_trend} | Trade: {trade_dir} ✗ — against trend",
                "h4_trend":   h4_trend,
                "confidence_adj": 0,
            }

    # ── CHECK 2 — Minimum Confluence Score ────────────────────────────────────
    def check_confluence_score(
        self,
        signal: dict,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Requires >= 3 of 6 confluences from confluence_engine.py.
        """
        direction = str(_sig(signal, "direction", "long")).lower()

        if not _CONFLUENCE_AVAILABLE:
            return {
                "passed":     False,
                "check_name": "Confluence Score FAILED",
                "detail":     "confluence_engine.py not available",
                "score":      0,
                "met":        [],
                "missed":     [],
                "confidence_adj": 0,
            }

        try:
            result      = score_confluences(df, direction)
            met_dicts   = result.get("confluences_met",    [])
            failed_dicts= result.get("confluences_failed", [])
            # Use raw count of agreements, not net score (net can be 0 if penalties cancel)
            num_met     = len([m for m in met_dicts if m.get("result") != "neutral"])
            passed      = num_met >= MIN_CONFLUENCE

            met_names   = [m.get("check", "?") for m in met_dicts if m.get("result") != "neutral"]
            missed_names= [f.get("check", "?") for f in failed_dicts]
            net_score   = result.get("confluence_score", 0)

            return {
                "passed":     passed,
                "check_name": "Confluence Score" + (" ✓" if passed else " FAILED"),
                "detail":     (
                    f"{num_met}/6 confluences met"
                    + (" ✓" if passed else f" — need {MIN_CONFLUENCE}")
                ),
                "score":      num_met,
                "met":        met_names,
                "missed":     missed_names,
                "confidence_adj": 0,
            }
        except Exception as exc:
            return {
                "passed":     False,
                "check_name": "Confluence Score FAILED",
                "detail":     f"Error running confluence check: {exc}",
                "score":      0,
                "met":        [],
                "missed":     [],
                "confidence_adj": 0,
            }

    # ── CHECK 3 — Risk/Reward Ratio ───────────────────────────────────────────
    def check_risk_reward(
        self,
        signal: dict,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Reward must be >= 2.0x risk (1.5x exception for news-fade trades).
        """
        is_news_fade = bool(_sig(signal, "is_news_fade", False))
        min_rr       = MIN_RR_NEWS_FADE if is_news_fade else MIN_RR_DEFAULT

        entry = float(_sig(signal, "entry",       0.0) or 0.0)
        sl    = float(_sig(signal, "stop_loss",   0.0) or 0.0)
        tp    = float(_sig(signal, "take_profit", 0.0) or 0.0)

        # Fallback: infer from df if signal didn't supply prices
        if entry == 0.0 and not df.empty:
            entry = float(df.iloc[-1]["close"])
            atr   = float(df.iloc[-1].get("atr", entry * 0.003))
            direction = str(_sig(signal, "direction", "long")).lower()
            if sl == 0.0:
                sl = entry - atr * 1.5 if direction in ("long", "buy") else entry + atr * 1.5
            if tp == 0.0:
                tp = entry + atr * 3.0 if direction in ("long", "buy") else entry - atr * 3.0

        risk   = abs(entry - sl)
        reward = abs(entry - tp)

        if risk == 0:
            return {
                "passed":     False,
                "check_name": "Risk/Reward FAILED",
                "detail":     "Stop loss equals entry — cannot calculate RR",
                "entry":      entry,
                "stop_loss":  sl,
                "take_profit":tp,
                "risk_pips":  0,
                "reward_pips":reward,
                "rr_ratio":   0.0,
                "confidence_adj": 0,
            }

        rr_ratio = reward / risk
        passed   = rr_ratio >= (min_rr - 0.01)   # epsilon for float precision

        direction  = str(_sig(signal, "direction", "long")).lower()
        pip_size   = 0.1   # XAUUSD
        risk_pips   = round(risk   / pip_size, 1)
        reward_pips = round(reward / pip_size, 1)

        exception_note = " (news-fade exception: 1.5x)" if is_news_fade and passed and min_rr == MIN_RR_NEWS_FADE else ""

        rr_str = f"1:{rr_ratio:.1f}"
        return {
            "passed":     passed,
            "check_name": "Risk/Reward" + (" ✓" if passed else " FAILED"),
            "detail":     (
                f"Entry: ${entry:,.0f} | SL: ${sl:,.0f} | TP: ${tp:,.0f}\n"
                f"    Risk: {risk_pips:.0f} pips | Reward: {reward_pips:.0f} pips\n"
                f"    RR Ratio: {rr_str}"
                + (" ✓" if passed else f" — need 1:{min_rr:.1f}")
                + exception_note
            ),
            "rejection_summary": f"RR ratio {rr_str} below minimum 1:{min_rr:.1f}",
            "entry":       entry,
            "stop_loss":   sl,
            "take_profit": tp,
            "risk_pips":   risk_pips,
            "reward_pips": reward_pips,
            "rr_ratio":    round(rr_ratio, 2),
            "confidence_adj": 0,
        }

    # ── CHECK 4 — News Safety Window ──────────────────────────────────────────
    def check_news_safety(
        self,
        signal: dict,
        now_utc: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Fails if a HIGH impact event is within NEWS_BLOCK_BEFORE min or
        within NEWS_BLOCK_AFTER min after.
        """
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        if not _NEWS_AVAILABLE:
            return {
                "passed":     True,   # no data = don't block
                "check_name": "News Safety (no feed)",
                "detail":     "News feed unavailable — proceeding with caution",
                "blocking_event": None,
                "confidence_adj": 0,
            }

        try:
            global _FF_CACHE, _FF_CACHE_DATE
            today_str = now_utc.strftime("%m-%d-%Y")
            if _FF_CACHE is None or _FF_CACHE_DATE != today_str:
                _FF_CACHE      = fetch_ff_calendar()
                _FF_CACHE_DATE = today_str
            all_events = _FF_CACHE
        except Exception as exc:
            return {
                "passed":     True,
                "check_name": "News Safety (feed error)",
                "detail":     f"Could not fetch calendar: {exc}",
                "blocking_event": None,
                "confidence_adj": 0,
            }

        today_str   = now_utc.strftime("%m-%d-%Y")
        high_events = [
            e for e in all_events
            if e.get("date") == today_str and e.get("impact") == "High"
        ]

        blocking_event = None
        minutes_away   = None

        for ev in high_events:
            time_str = ev.get("time_utc", "")   # "HH:MM UTC"
            try:
                t = datetime.strptime(
                    f"{today_str} {time_str.replace(' UTC', '')}",
                    "%m-%d-%Y %H:%M"
                ).replace(tzinfo=timezone.utc)
                diff_min = (t - now_utc).total_seconds() / 60

                if -NEWS_BLOCK_AFTER <= diff_min <= NEWS_BLOCK_BEFORE:
                    blocking_event = ev
                    minutes_away   = round(diff_min)
                    break
            except ValueError:
                continue

        if blocking_event:
            title = blocking_event.get("title", "High impact event")
            if minutes_away >= 0:
                timing = f"in {minutes_away} minutes"
            else:
                timing = f"{abs(minutes_away)} minutes ago"
            rec_wait = NEWS_BLOCK_BEFORE + 30

            return {
                "passed":     False,
                "check_name": "News Safety FAILED",
                "detail":     (
                    f"⚠ {title} {timing}\n"
                    f"    FAILED — too close to high impact news"
                ),
                "blocking_event": blocking_event,
                "minutes_away":   minutes_away,
                "recommendation": f"Wait {rec_wait} minutes after {title} before entering",
                "confidence_adj": 0,
            }

        next_ev = _next_event_summary(high_events, now_utc)
        return {
            "passed":     True,
            "check_name": "News Safety ✓",
            "detail":     f"No high impact events nearby ✓\n    {next_ev}",
            "blocking_event": None,
            "confidence_adj": 0,
        }

    # ── CHECK 5 — Session Quality ─────────────────────────────────────────────
    def check_session_quality(
        self,
        signal: dict,
        now_utc: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Passes for London/NY/Overlap sessions. Fails for Asian session.
        Breakout strategies get an exception for Asian session.
        Weekend always fails.
        Uses world_sessions (UAE-accurate) with UTC-hour fallback.
        """
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        is_breakout = bool(_sig(signal, "is_breakout", False))
        weekday     = now_utc.weekday()   # 0=Mon … 4=Fri; 5=Sat, 6=Sun

        if weekday >= 5:
            return {
                "passed":     False,
                "check_name": "Session Quality FAILED",
                "detail":     f"Weekend — no trading (day {weekday})",
                "session":    "Weekend",
                "confidence_adj": 0,
            }

        # ── Try world_sessions (UAE-accurate) ─────────────────────────────────
        session = None
        quality = None
        try:
            from world_sessions import get_active_sessions
            from datetime import timedelta
            _gst_tz = timezone(timedelta(hours=4))
            _now_uae = now_utc.astimezone(_gst_tz)
            active = get_active_sessions(_now_uae)
            keys   = {s["key"] for s in active}

            if "london" in keys and "newyork" in keys:
                session, quality = "London/NY Overlap", "HIGH"
            elif "newyork" in keys:
                session, quality = "New York", "HIGH"
            elif "london" in keys:
                session, quality = "London", "HIGH"
            elif keys & {"tokyo", "hongkong", "shanghai"}:
                session, quality = "Asian", "LOW"
            else:
                session, quality = "Off-Hours", "LOW"
        except Exception:
            pass

        if session is None:
            # Fallback: UTC-hour approximation
            hour = now_utc.hour + now_utc.minute / 60.0
            if 12.0 <= hour < 15.0:
                session, quality = "London/NY Overlap", "HIGH"
            elif 7.0 <= hour < 12.0:
                session, quality = "London", "HIGH"
            elif 13.0 <= hour < 17.0:
                session, quality = "New York", "HIGH"
            elif 0.0 <= hour < 7.0:
                session, quality = "Asian", "LOW"
            else:
                session, quality = "Off-Hours", "LOW"

        high_quality = quality == "HIGH"

        if high_quality:
            return {
                "passed":     True,
                "check_name": "Session Quality ✓",
                "detail":     f"{session} session active ✓",
                "session":    session,
                "confidence_adj": 0,
            }
        elif is_breakout and session == "Asian":
            return {
                "passed":     True,
                "check_name": "Session Quality ✓ (Breakout Exception)",
                "detail":     (
                    f"Asian session — normally FAIL\n"
                    f"    Exception: breakout strategy active ✓"
                ),
                "session":    session,
                "confidence_adj": 0,
            }
        else:
            return {
                "passed":     False,
                "check_name": "Session Quality FAILED",
                "detail":     f"{session} — low-probability session for trading",
                "session":    session,
                "confidence_adj": 0,
            }


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 6 — SL Quality (standalone helper, also called by validate_entry)
# ══════════════════════════════════════════════════════════════════════════════

def sl_quality_check(df: pd.DataFrame, entry: float, sl: float, direction: str) -> dict[str, Any]:
    """
    Check 6: Validate stop-loss placement quality.

    Sub-checks:
      A) Not inside market noise floor (< ATR×0.3)
      B) Placed at / near a technical swing level (within 0.5%)
      C) No significant S/R level between entry and SL
      D) Spread buffer — SL is at least $1.00 beyond the nearest swing level

    Returns a dict with keys:
      passed         : bool   (True if noise + spread checks pass)
      checks         : dict   (per sub-check results)
      warnings       : list   (non-fatal observations)
      adjusted_sl    : float  (may differ from input sl if buffer applied)
      confidence_penalty : float (0.0 unless a warning warrants a penalty)
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "atr" not in df.columns:
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"]  - df["close"].shift()).abs()
        df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    result: dict[str, Any] = {
        "passed": True,
        "check_name": "SL Quality",
        "checks": {},
        "warnings": [],
        "adjusted_sl": sl,
        "confidence_penalty": 0.0,
        "confidence_adj": 0,
    }

    atr        = float(df["atr"].iloc[-1]) if not df["atr"].isna().all() else 5.0
    noise_floor = atr * 0.3
    sl_distance = abs(entry - sl)
    is_long     = str(direction).lower() in ("long", "buy")

    # ── CHECK A — noise floor ────────────────────────────────────────────────
    if sl_distance < noise_floor:
        result["passed"] = False
        result["checks"]["noise"] = {
            "passed": False,
            "detail": f"SL too close (${sl_distance:.2f}) < noise floor (${noise_floor:.2f})",
            "min_sl_dist": round(noise_floor, 2),
        }
    else:
        result["checks"]["noise"] = {
            "passed": True,
            "detail": f"Outside noise floor ✓ (${sl_distance:.2f} > ${noise_floor:.2f})",
        }

    # ── CHECK B — near swing level ───────────────────────────────────────────
    lookback   = min(50, len(df))
    highs      = df["high"].iloc[-lookback:]
    lows       = df["low"].iloc[-lookback:]
    swing_highs = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    swing_lows  = lows[(lows.shift(1) > lows)   & (lows.shift(-1) > lows)]
    relevant    = list(swing_lows) if is_long else list(swing_highs)
    tolerance   = sl * 0.005
    near_level  = any(abs(lvl - sl) <= tolerance for lvl in relevant)

    if near_level:
        result["checks"]["structure"] = {"passed": True,  "detail": "SL at swing level ✓"}
    else:
        result["checks"]["structure"] = {"passed": False, "detail": "No swing level near SL — may lack structure"}
        result["warnings"].append("SL not anchored to a key level — consider adjusting")
        result["confidence_penalty"] += 0.5
        result["confidence_adj"]      = -0  # warning only, not a hard fail

    # ── CHECK C — no S/R between entry and SL ───────────────────────────────
    all_levels   = list(swing_highs) + list(swing_lows)
    sl_min       = min(entry, sl)
    sl_max       = max(entry, sl)
    lvls_between = [l for l in all_levels if sl_min < l < sl_max]
    if lvls_between:
        nearest = min(lvls_between, key=lambda l: abs(l - entry))
        result["checks"]["sr_between"] = {
            "passed": False,
            "detail": f"S/R at ${nearest:,.2f} between entry and SL — may cause early stop",
        }
        result["warnings"].append(f"S/R level ${nearest:,.2f} sits between entry and SL")
    else:
        result["checks"]["sr_between"] = {"passed": True, "detail": "No S/R between entry and SL ✓"}

    # ── CHECK D — spread buffer ──────────────────────────────────────────────
    SPREAD_BUFFER = 1.00
    if near_level and sl_distance < (noise_floor + SPREAD_BUFFER):
        adjusted = round(sl - SPREAD_BUFFER if is_long else sl + SPREAD_BUFFER, 2)
        result["adjusted_sl"] = adjusted
        result["checks"]["spread"] = {
            "passed": True,
            "detail": f"SL auto-adjusted by ${SPREAD_BUFFER:.2f} spread buffer → ${adjusted:,.2f}",
        }
    else:
        result["checks"]["spread"] = {"passed": True, "detail": "Spread buffer adequate ✓"}

    # ── CHECK E — Volume climax nearby (advisory only) ──────────────────────
    try:
        from volume_analyzer import VolumeAnalyzer
        climax = VolumeAnalyzer().detect_volume_climax(df)
        if climax.get("climax_detected"):
            result["warnings"].append(
                "Volume climax detected within entry window. "
                "Exhaustion signal present. High reversal risk."
            )
            result["confidence_penalty"] = result.get("confidence_penalty", 0) + 1.5
            result["checks"]["volume_climax"] = {
                "passed": False,
                "detail": (
                    f"Volume climax: {climax.get('type','?')} — "
                    f"{climax.get('warning','')} (advisory only)"
                ),
            }
        else:
            result["checks"]["volume_climax"] = {"passed": True, "detail": "No volume climax ✓"}
    except Exception:
        result["checks"]["volume_climax"] = {"passed": True, "detail": "Volume climax check skipped"}

    # ── CHECK F — Dynamic ATR validation (advisory only, never hard-fails) ───
    try:
        from atr_sl_engine import calculate_dynamic_sl as _cds_f
        _dyn_f      = _cds_f(df, direction, entry)
        _rec_dist   = _dyn_f["sl_distance"]
        result["recommended_sl"]  = _dyn_f["sl_price"]
        result["sl_breakdown"]    = _dyn_f["sl_breakdown"]
        result["volatility_state"]= _dyn_f["volatility_state"]
        result["atr_percentile"]  = _dyn_f["atr_percentile"]
        if sl_distance < _rec_dist * 0.7:
            result["warnings"].append(
                f"SL tighter than ATR recommendation — consider widening to "
                f"${_dyn_f['sl_price']:,.2f}"
            )
        elif sl_distance > _rec_dist * 1.5:
            result["warnings"].append(
                f"SL wider than ATR recommendation — check if setup is still valid"
            )
        result["checks"]["dynamic_atr"] = {
            "passed": True,
            "detail": (
                f"Dynamic ATR SL: ${_dyn_f['sl_price']:,.2f} "
                f"({_dyn_f['sl_breakdown']})"
            ),
        }
    except Exception:
        result["checks"]["dynamic_atr"] = {"passed": True, "detail": "Dynamic ATR check skipped"}

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  validate_entry — main public function
# ══════════════════════════════════════════════════════════════════════════════

def validate_entry(
    signal: dict[str, Any],
    df: pd.DataFrame,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """
    Run all 5 checks. Signal must pass ALL 5 to be approved.

    Parameters
    ----------
    signal  : dict with at minimum 'direction'; optionally entry/SL/TP/confidence
    df      : OHLCV DataFrame (indicators auto-computed if missing)
    now_utc : override current time (for testing)

    Returns
    -------
    passed            : bool   — True only if all 5 checks pass
    checks_passed     : int    — number of checks passed (must be 5/5)
    check_results     : dict   — per-check result dicts keyed 1–5
    rejection_reason  : str    — first failed check reason (empty if passed)
    final_confidence  : float  — signal's original confidence ± adjustments
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    df = _enrich(df)

    # ── CHECK 0 — Risk feasibility (hard-fail before all other checks) ──────
    _entry_p = float(_sig(signal, "entry",     0) or 0)
    _sl_p    = float(_sig(signal, "stop_loss", 0) or 0)
    if _entry_p and _sl_p:
        try:
            from settings import load_settings as _lset, calculate_position as _cpos
            _pos = _cpos(_entry_p, _sl_p, _lset())
            if not _pos.get("tradeable", True):
                _reason = _pos.get("reason", "Setup not tradeable with current risk settings")
                return {
                    "passed":           False,
                    "checks_passed":    0,
                    "check_results":    {0: {"passed": False, "check_name": "Risk Feasibility",
                                            "reason": _reason, "hard_fail": True}},
                    "rejection_reason": _reason,
                    "final_confidence": float(_sig(signal, "confidence_score", 5.0) or 5.0),
                    "hard_fail":        True,
                }
        except Exception:
            pass
    # ── END CHECK 0 ──────────────────────────────────────────────────────────

    checker  = EntryChecklist()
    base_conf= float(_sig(signal, "confidence_score", 5.0) or 5.0)
    conf_adj = 0.0

    r1 = checker.check_trend_alignment(signal, df)
    r2 = checker.check_confluence_score(signal, df)
    r3 = checker.check_risk_reward(signal, df)
    r4 = checker.check_news_safety(signal, now_utc)
    r5 = checker.check_session_quality(signal, now_utc)

    # CHECK 6 — SL quality (warning only — does not hard-fail the signal,
    # but confidence penalty is applied and adjusted_sl is stored)
    entry_p = float(_sig(signal, "entry",     0) or 0)
    sl_p    = float(_sig(signal, "stop_loss", 0) or 0)
    direction = str(_sig(signal, "direction", "long")).lower()
    r6: dict[str, Any] = {"passed": True, "check_name": "SL Quality", "confidence_adj": 0}
    if entry_p and sl_p:
        try:
            r6 = sl_quality_check(df, entry_p, sl_p, direction)
        except Exception:
            pass

    results   = {1: r1, 2: r2, 3: r3, 4: r4, 5: r5, 6: r6}
    # Only checks 1–5 can hard-fail the signal; check 6 is advisory
    core_results = {k: v for k, v in results.items() if k <= 5}
    num_passed  = sum(1 for r in core_results.values() if r["passed"])
    all_passed  = all(r["passed"] for r in core_results.values())

    for r in results.values():
        conf_adj += r.get("confidence_adj", 0)
    # Apply SL confidence penalty (non-fatal)
    conf_adj -= r6.get("confidence_penalty", 0.0)

    final_confidence = round(min(10.0, max(0.0, base_conf + conf_adj)), 1)

    rejection_reason = ""
    if not all_passed:
        for idx in range(1, 6):
            r = core_results[idx]
            if not r["passed"]:
                # Prefer a clean summary line if the check provided one
                rejection_reason = r.get(
                    "rejection_summary",
                    r["detail"].split("\n")[0]
                )
                break

    return {
        "passed":           all_passed,
        "checks_passed":    num_passed,
        "total_checks":     5,       # hard gates are still 5/5
        "check_results":    results,
        "sl_quality":       r6,
        "adjusted_sl":      r6.get("adjusted_sl", sl_p),
        "rejection_reason": rejection_reason,
        "final_confidence": final_confidence,
        "confidence_adj":   conf_adj,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  quick_score — lightweight pre-filter
# ══════════════════════════════════════════════════════════════════════════════

def quick_score(
    signal: dict[str, Any],
    df: pd.DataFrame,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """
    Fast pre-filter — returns just pass/fail + score + confidence.
    Runs all 5 checks but skips full confluence engine for speed.

    Returns
    -------
    passed     : bool
    score      : str  e.g. "4/5"
    confidence : float
    """
    result = validate_entry(signal, df, now_utc)
    return {
        "passed":     result["passed"],
        "score":      f"{result['checks_passed']}/5",
        "confidence": result["final_confidence"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  print_checklist — formatted terminal output
# ══════════════════════════════════════════════════════════════════════════════

def print_checklist(
    signal: dict[str, Any],
    results: dict[str, Any],
) -> None:
    """
    Print a formatted 5-point checklist report for a signal.

    Parameters
    ----------
    signal  : the signal dict passed to validate_entry
    results : the dict returned by validate_entry
    """
    asset     = str(_sig(signal, "asset",     "XAUUSD"))
    direction = str(_sig(signal, "direction", "long")).upper()
    pattern   = str(_sig(signal, "pattern_name", "Signal"))

    CHECK = "\u2713"   # ✓
    CROSS = "\u2717"   # ✗

    title_line1 = "ENTRY CHECKLIST"
    title_line2 = f"{asset} {direction}"

    inner  = W - 2
    pad1   = (inner - len(title_line1)) // 2
    pad2   = (inner - len(title_line2)) // 2

    print(f"\n  \u2554{'═' * inner}\u2557")
    print(f"  \u2551{' ' * pad1}{title_line1}{' ' * (inner - pad1 - len(title_line1))}\u2551")
    print(f"  \u2551{' ' * pad2}{title_line2}{' ' * (inner - pad2 - len(title_line2))}\u2551")
    print(f"  \u255a{'═' * inner}\u255d")

    check_labels = {
        1: "CHECK 1 — Trend Alignment",
        2: "CHECK 2 — Confluence Score",
        3: "CHECK 3 — Risk/Reward",
        4: "CHECK 4 — News Safety",
        5: "CHECK 5 — Session Quality",
    }

    for idx in range(1, 6):
        r      = results["check_results"][idx]
        passed = r["passed"]
        icon   = CHECK if passed else CROSS
        label  = check_labels[idx]
        lines  = r["detail"].split("\n")

        print(f"\n  {icon} {label}")

        # Handle CHECK 2 specially (show met/missed lists)
        if idx == 2:
            print(f"    {lines[0]}")
            met_names    = r.get("met",    [])
            missed_names = r.get("missed", [])
            if met_names:
                print(f"    Met: {', '.join(met_names)}")
            if missed_names:
                print(f"    Missed: {', '.join(missed_names)}")
        else:
            for line in lines:
                print(f"    {line.strip()}")

    # ── Final result block ────────────────────────────────────────────────────
    all_passed = results["passed"]
    conf       = results["final_confidence"]
    reason     = results["rejection_reason"]

    print(f"\n  {'═' * inner}")

    if all_passed:
        print(f"  RESULT: SIGNAL APPROVED {CHECK}")
        print(f"  CONFIDENCE: {conf}/10")
        print(f"  RECOMMENDATION: Trade meets all entry requirements")
    else:
        print(f"  RESULT: SIGNAL REJECTED {CROSS}")
        print(f"  REASON: {reason}")

        # Check if we have a recommendation from the blocking news check
        r4 = results["check_results"][4]
        rec = r4.get("recommendation", "")
        if rec and not results["check_results"][4]["passed"]:
            # Wrap long recommendation
            words  = rec.split()
            line   = ""
            first  = True
            prefix = "  RECOMMENDATION: "
            cont   = " " * len(prefix)
            out    = []
            for w in words:
                if len(line) + len(w) + 1 <= (W - len(prefix)):
                    line = (line + " " + w).strip()
                else:
                    out.append(line)
                    line = w
            if line:
                out.append(line)
            print(f"{prefix}{out[0]}")
            for l in out[1:]:
                print(f"{cont}{l}")
        elif not all_passed:
            print(f"  RECOMMENDATION: Do not enter this trade")

    print(f"  {'═' * inner}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _next_event_summary(high_events: list[dict], now_utc: datetime) -> str:
    """Return a short string describing the next high-impact event today."""
    today_str = now_utc.strftime("%m-%d-%Y")
    future = []
    for ev in high_events:
        time_str = ev.get("time_utc", "")
        try:
            t = datetime.strptime(
                f"{today_str} {time_str.replace(' UTC', '')}",
                "%m-%d-%Y %H:%M"
            ).replace(tzinfo=timezone.utc)
            if t > now_utc:
                future.append((t, ev))
        except ValueError:
            continue

    if not future:
        return "No more high impact events today"
    future.sort(key=lambda x: x[0])
    t, ev = future[0]
    mins  = round((t - now_utc).total_seconds() / 60)
    return f"Next: {ev['title']} ({ev.get('country','?')}) in {mins} min"


# ══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, sys

    HIST_CSV = os.path.join("data", "historical_xauusd.csv")

    print("\n  Loading price data...")
    try:
        raw = pd.read_csv(HIST_CSV, index_col=0)
        df  = _enrich(raw)
        print(f"  Candles: {len(df)} | Last close: ${df.iloc[-1]['close']:,.2f}\n")
    except FileNotFoundError:
        print("  data/historical_xauusd.csv not found. Run setup.py first.")
        sys.exit(1)

    c = df.iloc[-1]

    # ── Test A: well-formed SHORT signal ─────────────────────────────────────
    signal_short = {
        "pattern_name":    "Gravestone Doji",
        "asset":           "XAUUSD",
        "direction":       "short",
        "entry":           float(c["close"]),
        "stop_loss":       float(c["close"]) + float(c["atr"]) * 1.5,
        "take_profit":     float(c["close"]) - float(c["atr"]) * 3.0,
        "confidence_score": 7.0,
        "is_divergence":   False,
        "is_news_fade":    False,
        "is_breakout":     False,
    }

    print("  ── TEST A: Well-formed SHORT signal ──")
    result_a = validate_entry(signal_short, df)
    print_checklist(signal_short, result_a)
    qs_a = quick_score(signal_short, df)
    print(f"  Quick score: {qs_a['score']} passed | Confidence: {qs_a['confidence']}/10\n")

    # ── Test B: poor RR + counter-trend LONG ─────────────────────────────────
    signal_bad = {
        "pattern_name":    "Test Signal",
        "asset":           "XAUUSD",
        "direction":       "long",
        "entry":           float(c["close"]),
        "stop_loss":       float(c["close"]) - float(c["atr"]) * 3.0,
        "take_profit":     float(c["close"]) + float(c["atr"]) * 0.5,   # bad RR
        "confidence_score": 5.0,
        "is_divergence":   False,
        "is_news_fade":    False,
        "is_breakout":     False,
    }

    print("  ── TEST B: Bad RR + counter-trend LONG ──")
    result_b = validate_entry(signal_bad, df)
    print_checklist(signal_bad, result_b)
    qs_b = quick_score(signal_bad, df)
    print(f"  Quick score: {qs_b['score']} passed | Confidence: {qs_b['confidence']}/10\n")
