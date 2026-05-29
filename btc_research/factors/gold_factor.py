"""
btc_research/factors/gold_factor.py — Gold-BTC inverse correlation factor.

Key insight: Gold and BTC often move inversely.
- When gold is RISING strongly  → risk-off / safe-haven flows → BTC bearish signal
- When gold is FALLING          → risk-on environment        → BTC bullish signal

This is not always true (both can rise as inflation hedges), so we use this as
one factor among several rather than a standalone signal.

Score range: roughly -2.5 to +2.5
  Positive = gold bearish = BTC bullish
  Negative = gold bullish = BTC bearish
"""
from __future__ import annotations

import pandas as pd


def compute_gold_factor(
    df_gold:  pd.DataFrame,
    bar_time: pd.Timestamp,
    lookback: int = 60,
) -> dict:
    """
    Compute the gold factor at bar_time.

    Args:
        df_gold  : XAUUSD H1 DataFrame with 'time' and 'close' columns
        bar_time : current bar timestamp (UTC-aware)
        lookback : number of bars to include in calculation

    Returns dict:
        score    : float  (+ve = BTC bullish, -ve = BTC bearish)
        bias     : str    ("bullish" / "bearish" / "neutral")
        reason   : str
        gold_ema20, gold_ema50, gold_rsi, gold_pct5 : supporting values
    """
    if df_gold.empty:
        return {"score": 0.0, "bias": "neutral", "reason": "no gold data"}

    mask   = df_gold["time"] <= bar_time
    window = df_gold[mask].tail(lookback)

    if len(window) < 20:
        return {"score": 0.0, "bias": "neutral", "reason": "insufficient gold data"}

    close = window["close"].astype(float).reset_index(drop=True)

    # ── EMA trend ─────────────────────────────────────────────────────────────
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

    # ── Short-term momentum: % change over last 5 bars ─────────────────────────
    pct_5 = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
             if len(close) >= 5 else 0.0)

    # ── RSI(14) ───────────────────────────────────────────────────────────────
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta).clip(lower=0).rolling(14).mean()
    rsi    = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-10)))

    score   = 0.0
    reasons: list[str] = []

    # EMA trend direction
    if ema20 < ema50 * 0.9997:          # Gold trending DOWN  -> BTC+
        score += 1.0
        reasons.append("gold trend DN (BTC+)")
    elif ema20 > ema50 * 1.0003:        # Gold trending UP    -> BTC-
        score -= 1.0
        reasons.append("gold trend UP (BTC-)")

    # Short-term momentum (strong moves have more weight)
    if pct_5 > 1.0:                     # Gold surging hard   -> BTC-
        score -= 1.0
        reasons.append(f"gold +{pct_5:.1f}% (5bar) BTC-")
    elif pct_5 > 0.4:                   # Gold creeping up    -> slight BTC-
        score -= 0.5
        reasons.append(f"gold +{pct_5:.1f}% (5bar) mild BTC-")
    elif pct_5 < -1.0:                  # Gold dumping hard   -> BTC+
        score += 1.0
        reasons.append(f"gold {pct_5:.1f}% (5bar) BTC+")
    elif pct_5 < -0.4:                  # Gold drifting down  -> slight BTC+
        score += 0.5
        reasons.append(f"gold {pct_5:.1f}% (5bar) mild BTC+")

    # RSI extremes — potential mean reversion signal
    if rsi > 72:                        # Gold overbought -> likely to cool  -> BTC+
        score += 0.5
        reasons.append(f"gold RSI {rsi:.0f} overbought")
    elif rsi < 28:                      # Gold oversold   -> likely to bounce -> BTC-
        score -= 0.5
        reasons.append(f"gold RSI {rsi:.0f} oversold")

    bias = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")
    return {
        "score":      round(score, 1),
        "bias":       bias,
        "reason":     " | ".join(reasons) or "gold neutral",
        "gold_ema20": round(ema20, 2),
        "gold_ema50": round(ema50, 2),
        "gold_rsi":   round(rsi, 1),
        "gold_pct5":  round(pct_5, 2),
    }
