"""
instrument_confluence.py — Instrument-aware confluence layer for TradingBotV1
Wraps confluence_engine.py and adds per-instrument scoring weights,
hard trading rules, sector rotation, open interest and macro signals.
"""
from __future__ import annotations
from datetime import datetime

try:
    from macro_scorer    import MacroScorer
    _macro_scorer    = MacroScorer()
    _MACRO_OK        = True
except Exception:
    _MACRO_OK        = False
    _macro_scorer    = None  # type: ignore[assignment]

try:
    from sector_rotation import SectorRotation
    _sector_rotation = SectorRotation()
    _SR_OK           = True
except Exception:
    _SR_OK           = False
    _sector_rotation = None  # type: ignore[assignment]

try:
    from open_interest import OpenInterestAnalyzer
    _oi_analyzer     = OpenInterestAnalyzer()
    _OI_OK           = True
except Exception:
    _OI_OK           = False
    _oi_analyzer     = None  # type: ignore[assignment]

try:
    from instrument_data import get_market_context as _get_market_context
    _ID_OK = True
except Exception:
    _ID_OK = False
    def _get_market_context(i):  return {}  # type: ignore[misc]

# ── Per-instrument confluence weight adjustments ──────────────────────────────
INSTRUMENT_WEIGHTS: dict = {
    "XAUUSD": {
        "technical":   0.35,
        "fundamental": 0.25,   # COT, DXY, geo risk
        "session":     0.15,
        "ml":          0.15,
        "macro":       0.10,
    },
    "NAS100": {
        "technical":   0.30,
        "fundamental": 0.15,   # earnings, fed
        "session":     0.20,   # NY session critical
        "ml":          0.15,
        "macro":       0.10,
        "sector":      0.10,   # sector rotation
    },
    "US30": {
        "technical":   0.30,
        "fundamental": 0.20,
        "session":     0.20,
        "ml":          0.15,
        "macro":       0.05,
        "sector":      0.10,
    },
    "GBPUSD": {
        "technical":   0.25,
        "fundamental": 0.10,
        "session":     0.25,   # London session key
        "ml":          0.15,
        "macro":       0.25,   # macro is huge for forex
    },
    "EURUSD": {
        "technical":   0.25,
        "fundamental": 0.10,
        "session":     0.20,
        "ml":          0.15,
        "macro":       0.30,   # macro most important
    },
    "WTI": {
        "technical":   0.30,
        "fundamental": 0.30,   # EIA, OPEC huge
        "session":     0.15,
        "ml":          0.15,
        "macro":       0.10,
    },
}

# ── Per-instrument hard entry rules ───────────────────────────────────────────
# Violations BLOCK signal generation for that instrument.
INSTRUMENT_HARD_RULES: dict = {
    "NAS100": [
        "no_trade_if_vix_above_30",
        "ny_session_only",
        "no_trade_30min_before_fed",
    ],
    "US30": [
        "no_trade_if_vix_above_35",
        "ny_session_only",
    ],
    "GBPUSD": [
        "london_session_preferred",
        "no_trade_during_boe_meeting",
        "spread_must_be_below_2_pips",
    ],
    "EURUSD": [
        "london_frankfurt_session_preferred",
        "no_trade_during_ecb_meeting",
        "spread_must_be_below_1_5_pips",
    ],
    "WTI": [
        "avoid_30min_before_eia_wednesday",
        "ny_session_only",
    ],
    "XAUUSD": [],   # existing rules completely unchanged
}


class InstrumentConfluence:

    def __init__(self, instrument: str = "XAUUSD") -> None:
        self.instrument = instrument
        self.weights    = INSTRUMENT_WEIGHTS.get(
            instrument, INSTRUMENT_WEIGHTS["XAUUSD"])
        self.hard_rules = INSTRUMENT_HARD_RULES.get(instrument, [])

    # ── Hard rule checker ─────────────────────────────────────────────────────

    def check_hard_rules(self, context: dict) -> dict:
        """
        Check instrument-specific hard rules.
        Returns blocked=True if any rule is violated.
        """
        try:
            violations: list[str] = []
            now_utc   = datetime.utcnow()
            hour_gst  = (now_utc.hour + 4) % 24
            weekday   = now_utc.weekday()   # 0=Mon, 6=Sun

            for rule in self.hard_rules:

                if rule == "no_trade_if_vix_above_30":
                    vix = float(context.get("vix") or 0)
                    if vix > 30:
                        violations.append(
                            f"VIX {vix:.1f} > 30 — extreme fear, no NAS100 trade")

                elif rule == "no_trade_if_vix_above_35":
                    vix = float(context.get("vix") or 0)
                    if vix > 35:
                        violations.append(
                            f"VIX {vix:.1f} > 35 — extreme fear, no US30 trade")

                elif rule == "ny_session_only":
                    if not (13 <= hour_gst <= 22):
                        violations.append(
                            f"Hour {hour_gst}:00 GST is outside NY session "
                            f"(13:00–22:00 GST)")

                elif rule == "london_session_preferred":
                    if not (8 <= hour_gst <= 17):
                        violations.append(
                            f"Hour {hour_gst}:00 GST is outside London session "
                            f"(08:00–17:00 GST)")

                elif rule == "london_frankfurt_session_preferred":
                    if not (7 <= hour_gst <= 17):
                        violations.append(
                            f"Hour {hour_gst}:00 GST is outside EU session "
                            f"(07:00–17:00 GST)")

                elif rule == "avoid_30min_before_eia_wednesday":
                    # EIA report: every Wednesday ~18:30 GST
                    if weekday == 2 and (18 <= hour_gst <= 19):
                        violations.append(
                            "EIA report window (Wed 18:00–19:00 GST) — "
                            "no WTI trade during this period")

                elif rule == "spread_must_be_below_2_pips":
                    spread = float(context.get("spread_pips") or 0)
                    if spread > 2:
                        violations.append(
                            f"Spread {spread:.1f} pips > 2 — too wide for "
                            f"{self.instrument}")

                elif rule == "spread_must_be_below_1_5_pips":
                    spread = float(context.get("spread_pips") or 0)
                    if spread > 1.5:
                        violations.append(
                            f"Spread {spread:.1f} pips > 1.5 — too wide for EURUSD")

                # Soft-check rules (no live data available — skipped silently)
                elif rule in ("no_trade_30min_before_fed",
                              "no_trade_during_boe_meeting",
                              "no_trade_during_ecb_meeting"):
                    pass   # calendar events — not checked in real-time yet

            return {
                "blocked":       len(violations) > 0,
                "violations":    violations,
                "rules_checked": len(self.hard_rules),
            }
        except Exception as e:
            return {"blocked": False, "violations": [], "error": str(e)}

    # ── Extra confluence score ────────────────────────────────────────────────

    def get_extra_confluence_score(self) -> dict:
        """
        Additional confluence score from sector rotation, open interest,
        and macro — these are NOT in the original confluence_engine.
        Returns extra_score (±points) and human-readable factors list.
        """
        extra_score = 0.0
        factors: list[str] = []

        # ── Sector rotation (indices & commodities) ───────────────────────────
        if self.instrument in ("NAS100", "US30", "WTI") and _SR_OK and _sector_rotation is not None:
            try:
                sb = _sector_rotation.get_instrument_bias(self.instrument)
                bias = sb.get("bias", "")
                if "BULLISH" in bias:
                    extra_score += 8
                    factors.append(f"✅ Sector flow: {bias}")
                elif "BEARISH" in bias:
                    extra_score -= 8
                    factors.append(f"❌ Sector flow: {bias}")
                elif "CAUTION" in bias:
                    extra_score -= 4
                    factors.append(f"⚠️ Sector flow: {bias}")
                else:
                    factors.append("➡️ Sector flow: NEUTRAL")
            except Exception:
                pass

        # ── Open interest / volume signal (all instruments) ───────────────────
        if _OI_OK and _oi_analyzer is not None:
            try:
                oi      = _oi_analyzer.get_volume_analysis(self.instrument)
                oi_bias = oi.get("bias", "NEUTRAL")
                if oi_bias == "BULLISH":
                    extra_score += 7
                    factors.append("✅ Volume: Strong institutional buying confirmed")
                elif oi_bias == "BEARISH":
                    extra_score -= 7
                    factors.append("❌ Volume: Strong institutional selling confirmed")
                elif oi_bias == "CAUTION":
                    extra_score -= 3
                    factors.append("⚠️ Volume: Weak move — shorts covering only")
                elif oi_bias == "REVERSAL WATCH":
                    factors.append("🔄 Volume: Trend exhaustion — watch for reversal")
            except Exception:
                pass

        # ── Macro score (forex pairs only) ────────────────────────────────────
        if self.instrument in ("GBPUSD", "EURUSD") and _MACRO_OK and _macro_scorer is not None:
            try:
                macro      = _macro_scorer.score_pair(self.instrument)
                mb         = macro.get("bias", "NEUTRAL")
                score_diff = macro.get("score_diff", 0)
                if "LONG" in mb:
                    bonus        = min(15.0, abs(score_diff) * 0.5)
                    extra_score += bonus
                    factors.append(
                        f"✅ Macro: {mb} (score diff: {score_diff:+.0f})")
                elif "SHORT" in mb:
                    bonus        = min(15.0, abs(score_diff) * 0.5)
                    extra_score -= bonus
                    factors.append(
                        f"❌ Macro: {mb} (score diff: {score_diff:+.0f})")
                else:
                    factors.append("➡️ Macro: NEUTRAL — no edge from fundamentals")
            except Exception:
                pass

        return {
            "extra_score": round(extra_score, 1),
            "factors":     factors,
            "instrument":  self.instrument,
        }

    # ── Full signal context ───────────────────────────────────────────────────

    def get_full_signal_context(self) -> dict:
        """
        Full signal context for this instrument.
        Combines hard rule check + extra confluence.
        Used by bot_chat.py before showing any signal.
        """
        try:
            ctx   = _get_market_context(self.instrument)
            rules = self.check_hard_rules(ctx)
            extra = self.get_extra_confluence_score()

            return {
                "instrument":      self.instrument,
                "hard_rules":      rules,
                "blocked":         rules["blocked"],
                "violations":      rules["violations"],
                "extra_confluence": extra,
                "extra_score":     extra["extra_score"],
                "extra_factors":   extra["factors"],
                "market_context":  ctx,
                "weights":         self.weights,
            }
        except Exception as e:
            return {
                "instrument":  self.instrument,
                "blocked":     False,
                "extra_score": 0,
                "violations":  [],
                "extra_factors": [],
                "error":       str(e),
            }
