"""
atr_sl_engine.py — Dynamic ATR-Based Stop Loss Calculator
──────────────────────────────────────────────────────────
Replaces static ATR × 1.5 SL with a smart, context-aware SL that adapts
to live volatility, trading session, market regime, and geo risk.

Usage:
    from atr_sl_engine import calculate_dynamic_sl
    result = calculate_dynamic_sl(df, "long", entry=3300.0, session="London",
                                   regime="RANGING", geo_multiplier=1.0,
                                   strategy_name="EMA Trend Continuation")
"""

from __future__ import annotations

import math
import pandas as pd

# ── Session base multipliers ──────────────────────────────────────────────────
_SESSION_MULT: dict[str, float] = {
    "London":    1.5,
    "LondonNY":  1.5,   # Overlap
    "Overlap":   1.5,
    "NewYork":   1.5,
    "Asian":     2.0,
    "Off-Hours": 2.5,
    "OffHours":  2.5,
}

# ── Regime multipliers ────────────────────────────────────────────────────────
_REGIME_MULT: dict[str, float] = {
    "TRENDING_STRONG":    0.9,
    "TRENDING_WEAK":      1.0,
    "TRENDING":           1.0,
    "RANGING":            1.2,
    "VOLATILE_EXPANDING": 1.5,
    "VOLATILE":           1.5,
    "SQUEEZE_BUILDING":   0.8,
    "SQUEEZE":            0.8,
}

# ── Strategy-specific minimum multipliers ─────────────────────────────────────
_STRATEGY_MIN: dict[str, float] = {
    "london breakout":          1.0,
    "rsi oversold bounce":      0.8,
    "rsi overbought reversal":  0.8,
    "news spike fade":          1.5,
    "ema trend continuation":   1.0,
}
_STRATEGY_MIN_DEFAULT = 0.8

# ── Hard caps (in ATR multiples) ──────────────────────────────────────────────
_ATR_MIN_MULT = 0.5
_ATR_MAX_MULT = 3.0


def calculate_dynamic_sl(
    df: pd.DataFrame,
    direction: str,
    entry: float,
    session: str = "London",
    regime: str = "RANGING",
    geo_multiplier: float = 0.0,
    strategy_name: str = "",
) -> dict:
    """
    Calculate a dynamic, context-aware stop-loss distance and price.

    Parameters
    ----------
    df             : enriched DataFrame with 'atr' column
    direction      : "long" or "short" (case-insensitive)
    entry          : entry price
    session        : current trading session name
    regime         : market regime string from market_context.detect_gold_regime()
    geo_multiplier : extra ATR buffer from geo risk layer (sl_atr_multiplier)
    strategy_name  : playbook / strategy name for minimum-SL lookup

    Returns
    -------
    Full breakdown dict — see module docstring for all keys.
    """
    # ── Settings ──────────────────────────────────────────────────────────────
    try:
        from settings import load_settings
        _s     = load_settings()
        min_rr = float(_s.get("min_rr", 3.0))
    except Exception:
        min_rr = 3.0

    # ── Live price staleness check ────────────────────────────────────────────
    try:
        from mt5_sync import get_live_price as _glp_sl
        _lp_sl = _glp_sl()
        if _lp_sl.get("price") and _lp_sl["price"] > 0:
            _live_mid = _lp_sl["price"]
            _drift    = abs(entry - _live_mid)
            if _drift > 20:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    f"calculate_dynamic_sl: entry price ${entry:,.2f} differs from "
                    f"live ${_live_mid:,.2f} by ${_drift:,.2f} — verify entry before placing trade"
                )
    except Exception:
        pass

    # ── STEP 1 — ATR and volatility percentile ────────────────────────────────
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    if "atr" not in df.columns or df["atr"].isna().all():
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"]  - df["close"].shift()).abs()
        df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    atr_series = df["atr"].dropna()
    atr_value  = float(atr_series.iloc[-1]) if not atr_series.empty else float(entry * 0.005)
    if atr_value <= 0:
        atr_value = float(entry * 0.005)

    # Percentile of current ATR vs last 50 bars
    window = min(50, len(atr_series))
    recent = atr_series.iloc[-window:]
    n_below = (recent < atr_value).sum()
    atr_percentile = round(float(n_below) / max(window - 1, 1) * 100, 1)

    if atr_percentile > 80:
        volatility_state = "high_volatility"
        vol_multiplier   = 1.3
    elif atr_percentile < 20:
        volatility_state = "low_volatility"
        vol_multiplier   = 0.9
    else:
        volatility_state = "normal_volatility"
        vol_multiplier   = 1.0

    # ── STEP 2 — Session base multiplier ─────────────────────────────────────
    session_mult = _SESSION_MULT.get(session, 1.5)

    # ── STEP 3 — Regime adjustment ────────────────────────────────────────────
    regime_upper = str(regime).upper()
    regime_mult  = _REGIME_MULT.get(regime_upper, 1.0)
    # Fuzzy match: TRENDING_STRONG / TRENDING_WEAK both start with TRENDING
    if regime_mult == 1.0 and regime_upper not in _REGIME_MULT:
        for key, val in _REGIME_MULT.items():
            if regime_upper.startswith(key.split("_")[0]):
                regime_mult = val
                break

    # ── STEP 4 — Combined base distance (before geo) ─────────────────────────
    raw_multiplier = session_mult * regime_mult * vol_multiplier

    # ── STEP 5 — Strategy minimum ────────────────────────────────────────────
    strat_key  = strategy_name.lower().strip()
    strat_min  = _STRATEGY_MIN.get(strat_key, _STRATEGY_MIN_DEFAULT)
    # If the computed multiplier falls below strategy minimum, raise it
    raw_multiplier = max(raw_multiplier, strat_min)

    # ── STEP 6 — Geo risk buffer ──────────────────────────────────────────────
    geo_buffer      = float(geo_multiplier) * atr_value
    base_sl_dist    = raw_multiplier * atr_value
    final_sl_dist   = base_sl_dist + geo_buffer

    # ── STEP 7 — Hard caps in ATR units ──────────────────────────────────────
    final_sl_dist = max(final_sl_dist, _ATR_MIN_MULT * atr_value)
    final_sl_dist = min(final_sl_dist, _ATR_MAX_MULT * atr_value)

    # Re-derive the effective total multiplier after capping
    final_multiplier = round(final_sl_dist / atr_value, 3) if atr_value > 0 else raw_multiplier

    # ── STEP 8 — SL price ────────────────────────────────────────────────────
    is_long  = str(direction).lower() in ("long", "buy")
    sl_price = round(entry - final_sl_dist, 2) if is_long else round(entry + final_sl_dist, 2)

    # ── STEP 9 — TP prices ────────────────────────────────────────────────────
    if is_long:
        tp1_price = round(entry + final_sl_dist * 2.0,    2)
        tp2_price = round(entry + final_sl_dist * min_rr,  2)
    else:
        tp1_price = round(entry - final_sl_dist * 2.0,    2)
        tp2_price = round(entry - final_sl_dist * min_rr,  2)

    # ── R:R ───────────────────────────────────────────────────────────────────
    rr_at_tp1 = round(abs(tp1_price - entry) / final_sl_dist, 2) if final_sl_dist > 0 else 0.0
    rr_at_tp2 = round(abs(tp2_price - entry) / final_sl_dist, 2) if final_sl_dist > 0 else 0.0

    # ── Quality label ─────────────────────────────────────────────────────────
    if final_multiplier <= 1.0:
        quality = "tight"
    elif final_multiplier <= 2.0:
        quality = "normal"
    else:
        quality = "wide"

    # ── Breakdown string ──────────────────────────────────────────────────────
    vol_label    = {"high_volatility": "high vol", "low_volatility": "low vol",
                    "normal_volatility": "normal vol"}.get(volatility_state, "normal vol")
    geo_part     = f" + {geo_multiplier:.1f}×(geo)" if geo_multiplier > 0 else ""
    breakdown    = (
        f"{session_mult:.1f}×({session}) × {regime_mult:.1f}×({regime.title()}) "
        f"× {vol_multiplier:.1f}×({vol_label})"
        f"{geo_part} = {final_multiplier:.2f}× ATR = ${final_sl_dist:.2f}"
    )

    return {
        "sl_price":          sl_price,
        "tp1_price":         tp1_price,
        "tp2_price":         tp2_price,
        "sl_distance":       round(final_sl_dist, 4),
        "atr_value":         round(atr_value, 4),
        "atr_percentile":    atr_percentile,
        "volatility_state":  volatility_state,
        "session_multiplier": session_mult,
        "regime_multiplier":  regime_mult,
        "vol_multiplier":     vol_multiplier,
        "geo_buffer":         round(geo_buffer, 4),
        "final_multiplier":   final_multiplier,
        "sl_breakdown":       breakdown,
        "quality":            quality,
        "rr_at_tp1":          rr_at_tp1,
        "rr_at_tp2":          rr_at_tp2,
    }


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json as _json
    import numpy as np

    # Build a minimal test DataFrame
    np.random.seed(42)
    n = 100
    close = 3300.0 + np.cumsum(np.random.randn(n) * 5)
    high  = close + np.abs(np.random.randn(n) * 3)
    low   = close - np.abs(np.random.randn(n) * 3)
    _df   = pd.DataFrame({"high": high, "low": low, "close": close,
                          "open": close, "volume": np.ones(n) * 1000})
    hl    = _df["high"] - _df["low"]
    hc    = (_df["high"] - _df["close"].shift()).abs()
    lc    = (_df["low"]  - _df["close"].shift()).abs()
    _df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    result = calculate_dynamic_sl(
        _df, "long", entry=float(_df["close"].iloc[-1]),
        session="London", regime="RANGING",
        geo_multiplier=1.0, strategy_name="EMA Trend Continuation",
    )
    print(_json.dumps(result, indent=2))
