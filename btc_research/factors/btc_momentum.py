"""
btc_research/factors/btc_momentum.py — BTC's own trend & momentum factor.

This is the HIGHEST-WEIGHT factor. We never trade against BTC's own trend.

Multi-timeframe analysis on H1 data:
  - D1 equivalent trend : EMA(96)  on H1  (~4 days)
  - H4 equivalent trend : EMA(20)  on H1  (~1 day H4 period)
  - H1 momentum         : EMA(8) vs EMA(20) short-term alignment
  - RSI(14)             : avoid chasing extremes
  - ATR volatility      : skip if market is flat/choppy

Score convention (direction-relative):
  Positive = current direction IS aligned with trend
  Negative = current direction IS AGAINST trend
"""
from __future__ import annotations

import pandas as pd


def compute_btc_momentum(
    df_btc:    pd.DataFrame,
    bar_time:  pd.Timestamp,
    direction: str,
    lookback:  int = 120,
) -> dict:
    """
    Compute BTC trend/momentum factor at bar_time for the given direction.

    Args:
        df_btc    : BTCUSD H1 DataFrame
        bar_time  : current bar timestamp (UTC-aware)
        direction : "long" or "short"
        lookback  : H1 bars of BTC history to use

    Returns dict:
        score        : float  (+ve = direction aligned, -ve = direction opposed)
        bias         : str    ("bullish"/"bearish"/"neutral")
        reason       : str
    """
    if df_btc.empty:
        return {"score": 0.0, "bias": "neutral", "reason": "no BTC data"}

    mask   = df_btc["time"] <= bar_time
    window = df_btc[mask].tail(lookback)

    if len(window) < 50:
        return {"score": 0.0, "bias": "neutral", "reason": "insufficient BTC data"}

    close   = window["close"].astype(float).reset_index(drop=True)
    high    = window["high"].astype(float).reset_index(drop=True)
    low     = window["low"].astype(float).reset_index(drop=True)
    is_long = direction.lower() in ("long", "buy")
    current = float(close.iloc[-1])

    # ── Multi-timeframe EMAs on H1 ────────────────────────────────────────────
    ema8  = float(close.ewm(span=8,  adjust=False).mean().iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])   # ~H4 trend
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema96 = float(close.ewm(span=96, adjust=False).mean().iloc[-1])   # ~D1 trend

    # ── RSI(14) ───────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta).clip(lower=0).rolling(14).mean()
    rsi   = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-10)))

    # ── ATR(14) as % of price ─────────────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr     = float(tr.rolling(14).mean().iloc[-1])
    atr_pct = atr / current * 100 if current > 0 else 0.0

    score   = 0.0
    reasons: list[str] = []

    # ── D1-equivalent trend (EMA96) — highest weight ─────────────────────────
    # This is the macro trend; trading against it has low edge.
    if is_long:
        if current > ema96:
            score += 2.0
            reasons.append("D1 trend bull (long+)")
        else:
            score -= 2.0
            reasons.append("D1 trend bear (long-)")
    else:
        if current < ema96:
            score += 2.0
            reasons.append("D1 trend bear (short+)")
        else:
            score -= 2.0
            reasons.append("D1 trend bull (short-)")

    # ── H4-equivalent trend (EMA20 vs EMA50) ─────────────────────────────────
    if is_long:
        if ema20 > ema50:
            score += 1.0
            reasons.append("H4 bull (long+)")
        else:
            score -= 1.0
            reasons.append("H4 bear (long-)")
    else:
        if ema20 < ema50:
            score += 1.0
            reasons.append("H4 bear (short+)")
        else:
            score -= 1.0
            reasons.append("H4 bull (short-)")

    # ── H1 short-term momentum (EMA8 vs EMA20) ───────────────────────────────
    if is_long and ema8 > ema20:
        score += 0.5
        reasons.append("H1 momentum bull")
    elif not is_long and ema8 < ema20:
        score += 0.5
        reasons.append("H1 momentum bear")

    # ── RSI — avoid chasing overbought/oversold extremes ─────────────────────
    if is_long and rsi > 75:
        score -= 1.0
        reasons.append(f"BTC RSI {rsi:.0f} overbought (long-)")
    elif not is_long and rsi < 25:
        score -= 1.0
        reasons.append(f"BTC RSI {rsi:.0f} oversold (short-)")

    # ── Volatility filter — skip flat/choppy markets ─────────────────────────
    if atr_pct < 0.25:
        score -= 1.0
        reasons.append(f"low volatility ATR={atr_pct:.2f}% (skip)")

    bias = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")
    return {
        "score":       round(score, 1),
        "bias":        bias,
        "reason":      " | ".join(reasons) or "neutral",
        "btc_ema8":    round(ema8, 2),
        "btc_ema20":   round(ema20, 2),
        "btc_ema96":   round(ema96, 2),
        "btc_rsi":     round(rsi, 1),
        "btc_atr_pct": round(atr_pct, 3),
        "current":     round(current, 2),
    }
