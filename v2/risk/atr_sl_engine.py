"""
risk/atr_sl_engine.py — Dynamic ATR-based stop loss calculator for TradingBotV2.

Context-aware SL that adapts to volatility, session, regime, and geo risk.
Cleaned up from V1: removed V1-specific mt5_sync + settings imports.

Usage:
    from v2.risk.atr_sl_engine import calculate_dynamic_sl
    result = calculate_dynamic_sl(df, "long", entry=3310.0,
                                   session="London", regime="RANGING")
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Multiplier tables ─────────────────────────────────────────────────────────

_SESSION_MULT: dict[str, float] = {
    "London":    1.0,
    "LondonNY":  1.0,
    "Overlap":   1.0,
    "NewYork":   0.9,
    "NY":        0.9,
    "Asian":     1.2,
    "Off-Hours": 1.5,
    "OffHours":  1.5,
}

_REGIME_MULT: dict[str, float] = {
    "TRENDING_STRONG":    0.8,
    "TRENDING_WEAK":      0.9,
    "TRENDING":           0.8,
    "RANGING":            1.2,
    "VOLATILE_EXPANDING": 1.5,
    "VOLATILE":           1.5,
    "SQUEEZE_BUILDING":   1.0,
    "SQUEEZE":            1.0,
}

_STRATEGY_MIN: dict[str, float] = {
    "london breakout":         1.0,
    "rsi oversold bounce":     0.8,
    "rsi overbought reversal": 0.8,
    "news spike fade":         1.5,
    "ema trend continuation":  1.0,
    "smc order block":         1.0,
    "fvg fill":                0.9,
    "liquidity sweep":         1.1,
}
_STRATEGY_MIN_DEFAULT = 0.8

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
    min_rr: float = 2.0,
) -> dict:
    """
    Calculate a context-aware stop-loss and TP prices.

    Parameters
    ----------
    df             : OHLCV DataFrame (needs high/low/close; atr optional)
    direction      : "long" | "short"
    entry          : entry price
    session        : trading session name
    regime         : market regime string
    geo_multiplier : extra ATR buffer from geo risk layer
    strategy_name  : strategy name for minimum-SL lookup
    min_rr         : minimum RR for TP2 (default 2.0 → 1:2 at TP2)

    Returns
    -------
    dict with: sl_price, tp1_price, tp2_price, sl_distance, atr_value,
               final_multiplier, quality, rr_at_tp1, rr_at_tp2, breakdown
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ── ATR ───────────────────────────────────────────────────────────────────
    if "atr" not in df.columns or df["atr"].isna().all():
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"]  - df["close"].shift()).abs()
        df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    atr_series = df["atr"].dropna()
    atr        = float(atr_series.iloc[-1]) if not atr_series.empty else float(entry * 0.005)
    atr        = atr if atr > 0 else float(entry * 0.005)

    # Volatility percentile (last 50 bars)
    window  = min(50, len(atr_series))
    recent  = atr_series.iloc[-window:]
    pct     = round(float((recent < atr).sum()) / max(window - 1, 1) * 100, 1)

    if   pct > 80: vol_state, vol_mult = "high_volatility",   1.3
    elif pct < 20: vol_state, vol_mult = "low_volatility",    0.9
    else:          vol_state, vol_mult = "normal_volatility",  1.0

    # ── Multipliers ───────────────────────────────────────────────────────────
    sess_mult   = _SESSION_MULT.get(session, 1.5)
    reg_upper   = str(regime).upper()
    reg_mult    = _REGIME_MULT.get(reg_upper, 1.0)
    if reg_mult == 1.0 and reg_upper not in _REGIME_MULT:
        for k, v in _REGIME_MULT.items():
            if reg_upper.startswith(k.split("_")[0]):
                reg_mult = v
                break

    strat_min   = _STRATEGY_MIN.get(strategy_name.lower().strip(), _STRATEGY_MIN_DEFAULT)
    raw_mult    = max(sess_mult * reg_mult * vol_mult, strat_min)

    # ── SL distance ───────────────────────────────────────────────────────────
    geo_buf   = float(geo_multiplier) * atr
    sl_dist   = raw_mult * atr + geo_buf
    sl_dist   = max(sl_dist, _ATR_MIN_MULT * atr)
    sl_dist   = min(sl_dist, _ATR_MAX_MULT * atr)
    final_mult = round(sl_dist / atr, 3) if atr > 0 else raw_mult

    # ── Price levels ──────────────────────────────────────────────────────────
    is_long = str(direction).lower() in ("long", "buy")
    if is_long:
        sl_price  = round(entry - sl_dist, 5)
        tp1_price = round(entry + sl_dist * 1.5, 5)
        tp2_price = round(entry + sl_dist * min_rr, 5)
    else:
        sl_price  = round(entry + sl_dist, 5)
        tp1_price = round(entry - sl_dist * 1.5, 5)
        tp2_price = round(entry - sl_dist * min_rr, 5)

    rr1 = round(abs(tp1_price - entry) / sl_dist, 2) if sl_dist > 0 else 0.0
    rr2 = round(abs(tp2_price - entry) / sl_dist, 2) if sl_dist > 0 else 0.0

    quality = "tight" if final_mult <= 1.0 else "normal" if final_mult <= 2.0 else "wide"

    breakdown = (
        f"{sess_mult:.1f}×({session}) × {reg_mult:.1f}×({regime}) "
        f"× {vol_mult:.1f}×({vol_state.replace('_', ' ')})"
        + (f" + {geo_multiplier:.1f}×(geo)" if geo_multiplier > 0 else "")
        + f" = {final_mult:.2f}× ATR"
    )

    return {
        "sl_price":           sl_price,
        "tp1_price":          tp1_price,
        "tp2_price":          tp2_price,
        "sl_distance":        round(sl_dist, 5),
        "atr_value":          round(atr, 5),
        "atr_percentile":     pct,
        "volatility_state":   vol_state,
        "session_multiplier": sess_mult,
        "regime_multiplier":  reg_mult,
        "vol_multiplier":     vol_mult,
        "geo_buffer":         round(geo_buf, 5),
        "final_multiplier":   final_mult,
        "sl_breakdown":       breakdown,
        "quality":            quality,
        "rr_at_tp1":          rr1,
        "rr_at_tp2":          rr2,
    }
