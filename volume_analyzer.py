"""
volume_analyzer.py — Volume Analysis Engine for TradingBotV1
=============================================================
Provides volume-based confluence signals, climax detection, and
strategy-specific volume filtering.

Usage:
    from volume_analyzer import VolumeAnalyzer, check_volume_confluence
    va     = VolumeAnalyzer()
    profile = va.get_volume_profile(df)
    conf   = check_volume_confluence(df, "long", "London Breakout")
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR                 = os.path.dirname(os.path.abspath(__file__))
JOURNAL_FILE             = os.path.join(BASE_DIR, "data", "trade_journal.json")
VOLUME_STATS_FILE        = os.path.join(BASE_DIR, "data", "volume_strategy_stats.json")
GST                      = timezone(timedelta(hours=4))

# ── Debug logger (graceful) ───────────────────────────────────────────────────
try:
    from debug_logger import log_info, log_error
except ImportError:
    def log_info(m: str) -> None: pass
    def log_error(**kw) -> None: pass


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vol_col(df: pd.DataFrame) -> str:
    """Return the name of the volume column present in df."""
    for col in ("tick_volume", "volume", "Volume", "Tick_volume"):
        if col in df.columns:
            return col
    return "volume"   # default — caller gets KeyError if truly absent


def _load_journal() -> list[dict]:
    if not os.path.exists(JOURNAL_FILE):
        return []
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _load_volume_stats() -> dict:
    if not os.path.exists(VOLUME_STATS_FILE):
        return {}
    try:
        with open(VOLUME_STATS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_volume_stats(stats: dict) -> None:
    os.makedirs(os.path.dirname(VOLUME_STATS_FILE), exist_ok=True)
    with open(VOLUME_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════════════
#  VolumeAnalyzer class
# ═════════════════════════════════════════════════════════════════════════════

class VolumeAnalyzer:
    """
    All volume analysis methods for XAUUSD H1 data.
    Handles both 'tick_volume' (MT5) and 'volume' (yfinance) columns.
    """

    # ── get_volume_profile ────────────────────────────────────────────────────
    def get_volume_profile(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Classify current volume relative to the 20-period rolling average.

        Returns
        -------
        current_volume    : float
        avg_volume        : float
        volume_ratio      : float   (current / avg)
        volume_class      : str     (exceptional|high|normal|low|very_low)
        note              : str
        suitable_for_trading : bool
        """
        try:
            vcol = _vol_col(df)
            avg_volume     = df[vcol].rolling(20).mean().iloc[-1]
            current_volume = float(df[vcol].iloc[-1])

            if avg_volume == 0 or np.isnan(avg_volume):
                return {
                    "current_volume": current_volume,
                    "avg_volume":     0.0,
                    "volume_ratio":   1.0,
                    "volume_class":   "normal",
                    "note":           "Volume average unavailable — assuming normal",
                    "suitable_for_trading": True,
                }

            volume_ratio = current_volume / avg_volume

            if volume_ratio > 2.0:
                volume_class = "exceptional"
                note         = "Institutional activity detected"
            elif volume_ratio > 1.5:
                volume_class = "high"
                note         = "Strong participation"
            elif volume_ratio > 0.8:
                volume_class = "normal"
                note         = "Average market conditions"
            elif volume_ratio > 0.5:
                volume_class = "low"
                note         = "Weak participation — caution"
            else:
                volume_class = "very_low"
                note         = "Avoid trading — no conviction"

            return {
                "current_volume":      round(current_volume, 0),
                "avg_volume":          round(float(avg_volume), 0),
                "volume_ratio":        round(volume_ratio, 2),
                "volume_class":        volume_class,
                "note":                note,
                "suitable_for_trading": volume_ratio >= 0.8,
            }
        except Exception as exc:
            return {
                "current_volume": 0, "avg_volume": 0, "volume_ratio": 1.0,
                "volume_class": "normal", "note": f"Error: {exc}",
                "suitable_for_trading": True,
            }

    # ── check_volume_confirmation ─────────────────────────────────────────────
    def check_volume_confirmation(
        self,
        df: pd.DataFrame,
        direction: str,
    ) -> dict[str, Any]:
        """
        Check whether the last 3 candles confirm direction via volume pattern.

        BULLISH: up-candles should have higher volume; down-candles lower.
        BEARISH: down-candles should have higher volume; up-candles lower.

        Returns
        -------
        confirmed     : bool
        score         : int   (0–3)
        detail        : str   e.g. "2/3 candles confirm"
        volume_trend  : str   increasing | decreasing | mixed
        """
        try:
            vcol    = _vol_col(df)
            is_long = str(direction).lower() in ("long", "buy")
            last3   = df.tail(3)

            score = 0
            for _, row in last3.iterrows():
                is_up_candle  = float(row["close"]) >= float(row["open"])
                candle_vol    = float(row[vcol])
                avg_vol_20    = float(df[vcol].rolling(20).mean().iloc[-1])

                if is_long:
                    # Want up candles with high vol, down candles with low vol
                    if is_up_candle and candle_vol > avg_vol_20:
                        score += 1
                    elif not is_up_candle and candle_vol < avg_vol_20:
                        score += 1
                else:
                    # Want down candles with high vol, up candles with low vol
                    if not is_up_candle and candle_vol > avg_vol_20:
                        score += 1
                    elif is_up_candle and candle_vol < avg_vol_20:
                        score += 1

            # Volume trend over last 3 candles
            vols = list(last3[vcol].astype(float))
            if len(vols) >= 2:
                diffs = [vols[i+1] - vols[i] for i in range(len(vols)-1)]
                if all(d > 0 for d in diffs):
                    volume_trend = "increasing"
                elif all(d < 0 for d in diffs):
                    volume_trend = "decreasing"
                else:
                    volume_trend = "mixed"
            else:
                volume_trend = "mixed"

            return {
                "confirmed":     score >= 2,
                "score":         score,
                "detail":        f"{score}/3 candles confirm",
                "volume_trend":  volume_trend,
            }
        except Exception as exc:
            return {
                "confirmed": True, "score": 2,
                "detail": f"Error: {exc}", "volume_trend": "mixed",
            }

    # ── detect_volume_climax ──────────────────────────────────────────────────
    def detect_volume_climax(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Detect a volume climax: very high volume + tiny candle body.
        This indicates exhaustion / institutional distribution or accumulation.

        Returns
        -------
        climax_detected   : bool
        type              : "buying_climax" | "selling_climax" | None
        warning           : str | None
        confidence_impact : float  (−1.5 if climax, else 0)
        """
        try:
            vcol        = _vol_col(df)
            last_volume = float(df[vcol].iloc[-1])
            avg_volume  = float(df[vcol].rolling(20).mean().iloc[-1])
            last_body   = abs(float(df["close"].iloc[-1]) - float(df["open"].iloc[-1]))
            avg_body    = df.apply(
                lambda x: abs(float(x["close"]) - float(x["open"])), axis=1
            ).rolling(20).mean().iloc[-1]

            is_climax = (
                avg_volume > 0
                and last_volume > avg_volume * 2.0
                and avg_body > 0
                and last_body < avg_body * 0.5
            )

            climax_type = None
            warning     = None
            if is_climax:
                is_up_candle = float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])
                climax_type  = "buying_climax" if is_up_candle else "selling_climax"
                warning      = "Exhaustion — trend may end"

            return {
                "climax_detected":  is_climax,
                "type":             climax_type,
                "warning":          warning,
                "confidence_impact": -1.5 if is_climax else 0,
                "volume_ratio":     round(last_volume / avg_volume, 2) if avg_volume > 0 else 1.0,
                "body_ratio":       round(last_body / avg_body, 2)    if avg_body   > 0 else 1.0,
            }
        except Exception as exc:
            return {
                "climax_detected": False, "type": None,
                "warning": None,          "confidence_impact": 0,
                "volume_ratio": 1.0,      "body_ratio": 1.0,
            }

    # ── get_volume_at_levels ──────────────────────────────────────────────────
    def get_volume_at_levels(
        self,
        df: pd.DataFrame,
        key_levels: list[float],
    ) -> list[dict[str, Any]]:
        """
        For each price level, find historical candles that touched it and
        compute average volume vs overall market average.

        Returns list of dicts with level, touches, avg_volume_at_level,
        vs_market_avg ("+X%"), institutional_interest (bool), note.
        """
        try:
            vcol        = _vol_col(df)
            market_avg  = float(df[vcol].mean())
            if market_avg == 0:
                market_avg = 1.0

            results = []
            for level in key_levels:
                tolerance = level * 0.002   # 0.2% proximity
                touching  = df[
                    (df["low"]  <= level + tolerance) &
                    (df["high"] >= level - tolerance)
                ]
                if touching.empty:
                    results.append({
                        "level":                level,
                        "touches":              0,
                        "avg_volume_at_level":  0.0,
                        "vs_market_avg":        "0%",
                        "institutional_interest": False,
                        "note":                 "No historical touches found",
                    })
                    continue

                avg_vol_at_level = float(touching[vcol].mean())
                ratio            = avg_vol_at_level / market_avg
                pct_str          = f"+{(ratio-1)*100:.0f}%" if ratio >= 1 else f"{(ratio-1)*100:.0f}%"
                institutional    = ratio > 1.5
                note = (
                    f"Strong institutional activity at ${level:,.2f}"
                    if institutional else
                    f"Normal activity at ${level:,.2f}"
                )

                results.append({
                    "level":                round(level, 2),
                    "touches":              len(touching),
                    "avg_volume_at_level":  round(avg_vol_at_level, 0),
                    "vs_market_avg":        pct_str,
                    "institutional_interest": institutional,
                    "note":                 note,
                })
            return results
        except Exception:
            return []

    # ── track_volume_by_strategy ─────────────────────────────────────────────
    def track_volume_by_strategy(self) -> dict[str, Any]:
        """
        Load trade_journal.json, group by playbook_name, compute volume stats
        per strategy (wins vs losses), save to data/volume_strategy_stats.json.

        Returns dict of playbook → volume profile.
        """
        journal = _load_journal()
        if not journal:
            return {}

        # Group trades by playbook
        by_playbook: dict[str, list[dict]] = {}
        for t in journal:
            pb = (t.get("pattern_name") or "unknown").strip()
            by_playbook.setdefault(pb, []).append(t)

        stats: dict[str, Any] = {}
        for pb, trades in by_playbook.items():
            wins   = [t for t in trades if (t.get("outcome") or "").lower() == "win"]
            losses = [t for t in trades if (t.get("outcome") or "").lower() == "loss"]

            # Volume ratio from journal (may have been saved by morning_briefing)
            win_ratios  = [float(t["volume_ratio"]) for t in wins   if "volume_ratio" in t]
            loss_ratios = [float(t["volume_ratio"]) for t in losses if "volume_ratio" in t]

            avg_vol_wins   = round(float(np.mean(win_ratios)),  2) if win_ratios  else 1.5
            avg_vol_losses = round(float(np.mean(loss_ratios)), 2) if loss_ratios else 0.9

            # Optimal range: mean ± 0.5 of win ratios
            if win_ratios:
                opt_min = round(max(0.5, avg_vol_wins - 0.5), 1)
                opt_max = round(avg_vol_wins + 0.5, 1)
                vol_filter = round(max(0.5, avg_vol_wins - 0.3), 1)
            else:
                opt_min, opt_max = 1.2, 2.5
                vol_filter = 1.2

            # Insight
            loss_below = (
                sum(1 for r in loss_ratios if r < vol_filter) /
                max(len(loss_ratios), 1)
            ) * 100

            insight = (
                f"Works best at {opt_min}x-{opt_max}x. "
                f"Below {vol_filter}x = {loss_below:.0f}% loss rate."
            )

            stats[pb] = {
                "avg_volume_wins":   avg_vol_wins,
                "avg_volume_losses": avg_vol_losses,
                "optimal_range":     f"{opt_min}x - {opt_max}x",
                "volume_filter":     vol_filter,
                "insight":           insight,
                "sample_wins":       len(wins),
                "sample_losses":     len(losses),
            }

        try:
            _save_volume_stats(stats)
            log_info(f"[volume_analyzer] Volume stats saved for {len(stats)} playbooks")
        except Exception as exc:
            log_error(module="volume_analyzer", function="track_volume_by_strategy",
                      error=str(exc))

        return stats

    # ── get_volume_filter_for_playbook ────────────────────────────────────────
    def get_volume_filter_for_playbook(
        self,
        playbook_name: str,
        current_ratio: float,
    ) -> dict[str, Any]:
        """
        Compare current volume ratio against the strategy's optimal range.

        Returns pass/fail + confidence adjustment.
        """
        try:
            stats  = _load_volume_stats()
            pb_key = playbook_name.strip()
            # Fuzzy match
            if pb_key not in stats:
                for k in stats:
                    if k.lower() in pb_key.lower() or pb_key.lower() in k.lower():
                        pb_key = k
                        break

            if pb_key not in stats:
                # No data — default pass
                return {
                    "pass":  True,
                    "boost": 0.0,
                    "note":  "No volume profile for this strategy — neutral",
                }

            pb_stats   = stats[pb_key]
            opt_range  = pb_stats.get("optimal_range", "1.2x - 2.5x")
            vol_filter = float(pb_stats.get("volume_filter", 1.2))

            # Parse optimal range string "X.Xx - Y.Yx"
            try:
                parts   = opt_range.replace("x", "").split("-")
                opt_min = float(parts[0].strip())
                opt_max = float(parts[1].strip())
            except Exception:
                opt_min, opt_max = 1.2, 2.5

            in_range = opt_min <= current_ratio <= opt_max

            if in_range:
                return {
                    "pass":  True,
                    "boost": 0.5,
                    "note":  f"Volume {current_ratio:.1f}x is optimal for {playbook_name} ({opt_range})",
                }
            elif current_ratio >= vol_filter:
                return {
                    "pass":  True,
                    "boost": 0.0,
                    "note":  f"Volume {current_ratio:.1f}x acceptable (above filter {vol_filter}x)",
                }
            else:
                return {
                    "pass":    False,
                    "penalty": -1.0,
                    "note":    (
                        f"Volume {current_ratio:.1f}x below filter {vol_filter}x. "
                        f"{pb_stats.get('insight', '')}"
                    ),
                }
        except Exception as exc:
            return {
                "pass": True, "boost": 0.0,
                "note": f"Volume filter error: {exc}",
            }


# ═════════════════════════════════════════════════════════════════════════════
#  check_volume_confluence — public API, used by confluence_engine + backtest
# ═════════════════════════════════════════════════════════════════════════════

def check_volume_confluence(
    df: pd.DataFrame,
    direction: str,
    playbook_name: str = "unknown",
) -> dict[str, Any]:
    """
    Run all volume checks and return a combined confluence score.

    Scoring:
      +0.5  high or exceptional volume
      +0.3  direction confirmed by last 3 candles
      +0.2  no climax detected
      +0.5  strategy optimal volume range
      −1.5  climax detected (override)
      −1.0  strategy sub-optimal volume

    Returns
    -------
    score            : float   (total, can be negative)
    details          : list[str]
    volume_ratio     : float
    volume_class     : str
    climax           : bool
    strategy_optimal : bool
    confirmation_score : int (0-3)
    volume_profile   : dict
    """
    va     = VolumeAnalyzer()
    score  = 0.0
    details: list[str] = []

    # ── Profile ───────────────────────────────────────────────────────────────
    profile = va.get_volume_profile(df)
    ratio   = profile.get("volume_ratio", 1.0)
    vclass  = profile.get("volume_class", "normal")
    note    = profile.get("note", "")

    if vclass in ("exceptional", "high"):
        score += 0.5
        details.append(f"✓ Volume {ratio:.2f}x avg — {vclass} ({note})  +0.5")
    elif vclass == "normal":
        details.append(f"~ Volume {ratio:.2f}x avg — normal  +0.0")
    else:
        details.append(f"✗ Volume {ratio:.2f}x avg — {vclass} ({note})  +0.0")

    # ── Direction confirmation ─────────────────────────────────────────────────
    confirm = va.check_volume_confirmation(df, direction)
    conf_score = confirm.get("score", 0)
    if confirm.get("confirmed", False):
        score += 0.3
        details.append(f"✓ Direction confirmed {confirm['detail']}  +0.3")
    else:
        details.append(f"✗ Direction not confirmed {confirm['detail']}  +0.0")

    # ── Climax check ─────────────────────────────────────────────────────────
    climax_data = va.detect_volume_climax(df)
    is_climax   = climax_data.get("climax_detected", False)
    if is_climax:
        score -= 1.5
        details.append(
            f"✗ Volume climax ({climax_data.get('type','?')}) — "
            f"{climax_data.get('warning', '')}  −1.5"
        )
    else:
        score += 0.2
        details.append(f"✓ No climax signal  +0.2")

    # ── Strategy optimal filter ───────────────────────────────────────────────
    filt = va.get_volume_filter_for_playbook(playbook_name, ratio)
    strategy_optimal = filt.get("pass", True)
    if strategy_optimal:
        boost = filt.get("boost", 0.0)
        score += boost
        if boost > 0:
            details.append(f"✓ {filt.get('note','Strategy optimal')}  +{boost:.1f}")
        else:
            details.append(f"~ {filt.get('note','Volume acceptable')}  +0.0")
    else:
        penalty = abs(filt.get("penalty", 1.0))
        score  -= penalty
        details.append(f"✗ {filt.get('note','Strategy suboptimal')}  −{penalty:.1f}")

    return {
        "score":               round(score, 2),
        "details":             details,
        "volume_ratio":        ratio,
        "volume_class":        vclass,
        "climax":              is_climax,
        "climax_type":         climax_data.get("type"),
        "strategy_optimal":    strategy_optimal,
        "confirmation_score":  conf_score,
        "volume_profile":      profile,
        "climax_data":         climax_data,
        "filter_result":       filt,
    }
