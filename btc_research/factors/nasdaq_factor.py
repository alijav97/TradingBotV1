"""
btc_research/factors/nasdaq_factor.py — NAS100 risk-on/off factor for BTC.

Key insight: BTC behaves like a leveraged tech stock — it is highly correlated
with the Nasdaq.
- NAS100 trending UP   → risk-on   → BTC bullish
- NAS100 trending DOWN → risk-off  → BTC bearish

Important: NAS100 has market-hours gaps (roughly closes 21:00-00:00 UTC on
weekdays, and all weekend). We look back to the most recent available bar
regardless of gap, but we penalise confidence when data is stale.

Score range: roughly -2.5 to +2.5
"""
from __future__ import annotations

import pandas as pd


def compute_nasdaq_factor(
    df_nas:   pd.DataFrame,
    bar_time: pd.Timestamp,
    lookback: int = 60,
) -> dict:
    """
    Compute the NAS100 risk-on/off factor at bar_time.

    Args:
        df_nas   : NAS100 H1 DataFrame with 'time' and 'close' columns
        bar_time : current bar timestamp (UTC-aware)
        lookback : bars of NAS100 history to use

    Returns dict:
        score        : float  (+ve = risk-on -> BTC+, -ve = risk-off -> BTC-)
        bias         : str
        reason       : str
        data_gap_h   : hours since last NAS100 bar (0 when market open)
    """
    if df_nas.empty:
        return {"score": 0.0, "bias": "neutral", "reason": "no NAS data"}

    mask   = df_nas["time"] <= bar_time
    window = df_nas[mask].tail(lookback)

    if len(window) < 15:
        return {"score": 0.0, "bias": "neutral", "reason": "insufficient NAS data"}

    close = window["close"].astype(float).reset_index(drop=True)

    # ── EMA trend ─────────────────────────────────────────────────────────────
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

    # ── Short-term momentum ───────────────────────────────────────────────────
    pct_5 = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
             if len(close) >= 5 else 0.0)

    # ── RSI(14) ───────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta).clip(lower=0).rolling(14).mean()
    rsi   = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-10)))

    # ── Stale data detection ─────────────────────────────────────────────────
    last_nas_time = window["time"].iloc[-1]
    if hasattr(last_nas_time, "to_pydatetime"):
        last_nas_time = pd.Timestamp(last_nas_time)
    data_gap_h = (bar_time - last_nas_time).total_seconds() / 3600

    # If NAS100 data is > 16h old (overnight gap or weekend), halve score weight
    # to reflect reduced confidence in the stale reading
    conf = 0.5 if data_gap_h > 16 else 1.0

    score   = 0.0
    reasons: list[str] = []

    if data_gap_h > 16:
        reasons.append(f"NAS stale ({data_gap_h:.0f}h gap, 0.5x weight)")

    # EMA trend direction
    if ema20 > ema50 * 1.001:           # NAS trending UP   -> risk-on  -> BTC+
        score += 1.0 * conf
        reasons.append("NAS trend UP (BTC+)")
    elif ema20 < ema50 * 0.999:         # NAS trending DOWN -> risk-off -> BTC-
        score -= 1.0 * conf
        reasons.append("NAS trend DN (BTC-)")

    # Short-term momentum
    if pct_5 > 1.0:                     # NAS up >1% in 5 bars -> strong risk-on
        score += 1.0 * conf
        reasons.append(f"NAS +{pct_5:.1f}% (5bar) BTC+")
    elif pct_5 > 0.3:
        score += 0.5 * conf
        reasons.append(f"NAS +{pct_5:.1f}% (5bar) mild BTC+")
    elif pct_5 < -1.0:                  # NAS down >1% -> risk-off -> BTC-
        score -= 1.0 * conf
        reasons.append(f"NAS {pct_5:.1f}% (5bar) BTC-")
    elif pct_5 < -0.3:
        score -= 0.5 * conf
        reasons.append(f"NAS {pct_5:.1f}% (5bar) mild BTC-")

    # RSI
    if rsi > 65:
        score += 0.5 * conf
        reasons.append(f"NAS RSI {rsi:.0f} strong")
    elif rsi < 35:
        score -= 0.5 * conf
        reasons.append(f"NAS RSI {rsi:.0f} weak")

    bias = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")
    return {
        "score":      round(score, 1),
        "bias":       bias,
        "reason":     " | ".join(reasons) or "NAS neutral",
        "nas_ema20":  round(ema20, 2),
        "nas_ema50":  round(ema50, 2),
        "nas_rsi":    round(rsi, 1),
        "nas_pct5":   round(pct_5, 2),
        "data_gap_h": round(data_gap_h, 1),
    }
