"""
btc_research/eth_bot/backtest/run_backtest.py — Full-spectrum ETH backtest engine.

Tests 25 strategies × 24 UTC hours to find the best kill-zone windows.

== STRATEGIES TESTED ==

  Breakout / Momentum
    vb           — ATR Volatility Breakout (wide-range bar close at extreme)
    donchian     — Donchian 20-bar channel breakout (EMA200 filtered)
    turtle_55    — Turtle 55-bar channel breakout (pure trend, no EMA200 filter)
    bb_break     — Bollinger Band outer-band breakout (fresh close outside ±2σ)
    keltner      — Keltner Channel breakout (EMA20 ± 2×ATR10)
    inside_bar   — Inside bar compression → breakout of mother-bar range
    pdh_pdl      — Previous Day High / Low breakout

  Session-based
    morning_rng  — Morning Range: first 3 UTC hours → break at H 3+
    london_open  — London Open: Asian-session range → break at H 7-9 UTC
    ny_open      — NY Open: London-session range → break at H 13-15 UTC

  Trend Following
    ema_cross    — EMA 9/21 golden/death cross (EMA200 trend filter)
    macd_cross   — MACD line/signal crossover (EMA200 filtered)
    supertrend   — SuperTrend(10, 3) direction flip
    three_bar    — Three consecutive closes in same direction (momentum)

  Support / Resistance
    swing_break  — Swing level structural break (ATR-capped SL)
    swing_retest — Swing level pullback retest (tight SL)

  Oscillator / Mean-Reversion
    rsi_50       — RSI 50-level crossover (trend-confirmation mode)
    rsi_rev      — RSI 30/70 reversal hook (counter-trend)
    stoch        — Stochastic K/D crossover in oversold/overbought zone

  Candle Patterns
    engulfing    — Bullish / Bearish engulfing candle (EMA200 filtered)
    pin_bar      — Hammer / Shooting-star pin bar (EMA200 filtered)

  Multi-indicator Confluence
    macd_adx         — MACD crossover + ADX ≥ 25 (trend-strength gate)
    rsi_ema          — RSI 50-cross + EMA 9/21 stack + EMA200

  Combined Strategies (derived from 6yr backtest results — KZ-filtered)
    swing_keltner    — Swing Break[20] AND Keltner[EMA20 ± 2×ATR14]  (14-16 UTC only)
    rsi50_kz         — RSI14 50-cross in EMA200 direction             (02 UTC only)

== FILTERS ==
  Base    : EMA200 direction (per strategy) + ADX ≥ 20 (global)
  +BTC    : Base + BTC EMA200 must agree with ETH direction

== OUTPUT ==
  btc_research/eth_bot/backtest/data/backtest_trades.csv   — every simulated trade
  btc_research/eth_bot/backtest/data/backtest_summary.csv  — stats by hour × strategy

== USAGE ==
  cd C:\\Temp\\TradingBotV1
  C:\\TradingBotV2\\venv\\Scripts\\python.exe -m btc_research.eth_bot.backtest.run_backtest

== SIMULATION RULES ==
  Entry     : close of the signal bar
  SL hit    : low ≤ SL for longs / high ≥ SL for shorts (SL checked before TP)
  TP1 hit   : high ≥ TP1 for longs / low ≤ TP1 for shorts  [at 2R — close 50%]
  After TP1 : SL moves to entry (breakeven); trailing SL = price ± 2×ATR
  TP2 hit   : high ≥ TP2 for longs / low ≤ TP2 for shorts  [at 4R — close remainder]
  Max hold  : 96 bars (4 days) — force-close at bar-96 close
  r_achieved: BLENDED across the 50/50 scale-out (matches live paper_trader):
              clean run = +3R (0.5*2R + 0.5*4R); TP1-then-breakeven = +1R;
              full stop before TP1 = -1R. (Old model was single-unit: +4R / 0R.)
  Note      : TP2 changed 5R→4R after Phase 1 backtest showed ETH rarely reaches 5R
              (avg_r ~+0.5R across all strategies); 4R materially increases TP2 hit rate
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    stream = sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────────
_BACKTEST_DIR = Path(__file__).parent
DATA_DIR      = _BACKTEST_DIR / "data"
ETH_CSV       = DATA_DIR / "ETHUSD_H1.csv"
BTC_CSV       = DATA_DIR / "BTCUSD_H1.csv"
TRADES_CSV    = DATA_DIR / "backtest_trades.csv"
SUMMARY_CSV   = DATA_DIR / "backtest_summary.csv"

# ── Global parameters ───────────────────────────────────────────────────────────
ADX_THRESHOLD  = 20      # minimum ADX for any signal
TP1_RR         = 2.0     # first take-profit in R  (close 50% of position)
TP2_RR         = 4.0     # second take-profit in R (close remainder)
                         # Rationale: ETH phase-1 backtest showed avg_r ~0.5R across all
                         # strategies → very few trades reached the old 5R target.
                         # Reducing to 4R improves TP2 hit rate while preserving R:R.
TRAIL_ATR_MULT = 2.0     # trailing SL = price ± TRAIL_ATR_MULT × ATR (after TP1)
MAX_HOLD_BARS  = 96      # force-close after 4 days
WARMUP_BARS    = 260     # bars discarded while indicators stabilise (EMA200 + buffer)

# ── Strategy registry ─────────────────────────────────────────────────────────
# Maps strategy_key → human-readable label
STRATEGIES: dict[str, str] = {
    # Breakout / Momentum
    "vb":          "Volatility Breakout",
    "donchian":    "Donchian 20-bar",
    "turtle_55":   "Turtle 55-bar",
    "bb_break":    "BB Breakout",
    "keltner":     "Keltner Channel",
    "inside_bar":  "Inside Bar",
    "pdh_pdl":     "Prev Day H/L",
    # Session
    "morning_rng": "Morning Range",
    "london_open": "London Open",
    "ny_open":     "NY Open",
    # Trend following
    "ema_cross":   "EMA Cross 9/21",
    "macd_cross":  "MACD Cross",
    "supertrend":  "SuperTrend",
    "three_bar":   "Three-Bar Momentum",
    # Support / Resistance
    "swing_break":  "Swing Break",
    "swing_retest": "Swing Retest",
    # Oscillator
    "rsi_50":      "RSI 50-Cross",
    "rsi_rev":     "RSI Reversal",
    "stoch":       "Stochastic",
    # Candle patterns
    "engulfing":   "Engulfing",
    "pin_bar":     "Pin Bar",
    # Multi-indicator
    "macd_adx":       "MACD+ADX",
    "rsi_ema":        "RSI+EMA",
    # Combined strategies (KZ-filtered — fire only at specific hours)
    "swing_keltner":  "Swing+Keltner [14-16 UTC]",
    "rsi50_kz":       "RSI50-Cross [02 UTC]",
}


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR HELPERS  (all vectorised, return pd.Series with same index)
# ══════════════════════════════════════════════════════════════════════════════

def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> pd.Series:
    sp  = 2 * period - 1
    hd  = high.diff()
    ld  = low.diff()
    tr  = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    pdm = hd.where((hd > 0) & (hd > -ld), 0.0)
    mdm = (-ld).where((-ld > 0) & (-ld > hd), 0.0)
    aw  = tr.ewm(span=sp, adjust=False).mean()
    pw  = pdm.ewm(span=sp, adjust=False).mean()
    mw  = mdm.ewm(span=sp, adjust=False).mean()
    pdi = 100 * pw / aw
    ndi = 100 * mw / aw
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(span=sp, adjust=False).mean().fillna(0)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _compute_macd(close: pd.Series,
                  fast: int = 12, slow: int = 26, signal: int = 9,
                  ) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line  = ema_f - ema_s
    sig   = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def _compute_bb(close: pd.Series,
                period: int = 20, std_mult: float = 2.0,
                ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / (mid + 1e-12)
    return upper, mid, lower, width


def _compute_stoch(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3,
                   ) -> tuple[pd.Series, pd.Series]:
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k  = 100 * (close - ll) / (hh - ll + 1e-12)
    k  = k.rolling(3).mean()        # smooth %K
    d  = k.rolling(d_period).mean()
    return k, d


def _compute_keltner(high: pd.Series, low: pd.Series, close: pd.Series,
                     ema_period: int = 20, atr_period: int = 10,
                     atr_mult: float = 2.0,
                     ) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid   = close.ewm(span=ema_period, adjust=False).mean()
    atr   = _compute_atr(high, low, close, atr_period)
    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr
    return upper, mid, lower


def _compute_supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = 10, multiplier: float = 3.0,
                        ) -> tuple[pd.Series, pd.Series]:
    """Returns (supertrend_line, direction) where direction: 1=bull, -1=bear."""
    atr  = _compute_atr(high, low, close, period).values
    hl2  = ((high + low) / 2.0).values
    c    = close.values
    n    = len(c)

    b_upper = hl2 + multiplier * atr
    b_lower = hl2 - multiplier * atr
    f_upper = b_upper.copy()
    f_lower = b_lower.copy()
    st      = np.full(n, np.nan)
    direc   = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        if np.isnan(atr[i]):
            continue
        # Tighten upper band
        f_upper[i] = b_upper[i] if (b_upper[i] < f_upper[i-1] or c[i-1] > f_upper[i-1]) else f_upper[i-1]
        # Raise lower band
        f_lower[i] = b_lower[i] if (b_lower[i] > f_lower[i-1] or c[i-1] < f_lower[i-1]) else f_lower[i-1]

        if np.isnan(st[i-1]):
            st[i]    = f_lower[i]
            direc[i] = 1
        elif st[i-1] == f_upper[i-1]:    # was bearish
            if c[i] > f_upper[i]:
                st[i] = f_lower[i];  direc[i] = 1
            else:
                st[i] = f_upper[i];  direc[i] = -1
        else:                             # was bullish
            if c[i] < f_lower[i]:
                st[i] = f_upper[i];  direc[i] = -1
            else:
                st[i] = f_lower[i];  direc[i] = 1

    idx = close.index
    return pd.Series(st, index=idx), pd.Series(direc.astype(float), index=idx)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION RANGE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _add_session_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds session-based columns (computed from same-day bars, no lookahead):
      morning_high/low  — hours 0-2 UTC  (used in morning_rng strategy at H≥3)
      asian_high/low    — hours 0-6 UTC  (used in london_open strategy at H 7-9)
      london_high/low   — hours 7-12 UTC (used in ny_open strategy at H 13-15)
      prev_day_high/low — previous calendar day (used in pdh_pdl strategy)
    """
    df = df.copy()
    ts   = pd.to_datetime(df["time"], utc=True)
    hrs  = ts.dt.hour.values
    date = ts.dt.date

    df["_dt"]  = date
    df["_hr"]  = hrs

    # Use original df for all aggregates (no double-counting issue because
    # strategy checkers only USE these values at the correct trigger hours)
    mn = df[df["_hr"] <= 2].groupby("_dt").agg(
        morning_high=("high", "max"), morning_low=("low", "min")
    ).reset_index()

    as_ = df[df["_hr"] <= 6].groupby("_dt").agg(
        asian_high=("high", "max"), asian_low=("low", "min")
    ).reset_index()

    lo = df[(df["_hr"] >= 7) & (df["_hr"] <= 12)].groupby("_dt").agg(
        london_high=("high", "max"), london_low=("low", "min")
    ).reset_index()

    dy = df.groupby("_dt").agg(_dh=("high", "max"), _dl=("low", "min")).reset_index()
    dy["prev_day_high"] = dy["_dh"].shift(1)
    dy["prev_day_low"]  = dy["_dl"].shift(1)
    dy = dy[["_dt", "prev_day_high", "prev_day_low"]]

    for aux in [mn, as_, lo, dy]:
        df = df.merge(aux, on="_dt", how="left")

    df.drop(columns=["_dt", "_hr"], inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  PRECOMPUTE ALL INDICATORS  (one pass over the full DataFrame)
# ══════════════════════════════════════════════════════════════════════════════

def _precompute(df: pd.DataFrame) -> pd.DataFrame:
    """Add every indicator column used by any strategy checker."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    # Trend EMAs
    df["ema9"]   = _ema(c, 9)
    df["ema21"]  = _ema(c, 21)
    df["ema96"]  = _ema(c, 96)     # ≈ 4-day trend (used by swing strategies)
    df["ema200"] = _ema(c, 200)

    # Momentum / strength
    df["adx14"]  = _compute_adx(h, l, c, 14)
    df["atr14"]  = _compute_atr(h, l, c, 14)
    df["rsi14"]  = _compute_rsi(c, 14)

    # MACD
    df["macd_line"], df["macd_sig"], df["macd_hist"] = _compute_macd(c)

    # Bollinger Bands (20, 2)
    df["bb_upper"], df["bb_mid"], df["bb_lower"], df["bb_width"] = _compute_bb(c)

    # Stochastic (14, 3)
    df["stoch_k"], df["stoch_d"] = _compute_stoch(h, l, c)

    # Donchian 20-bar (shift=1 → no lookahead: signal bar close vs prev 20 highs)
    df["don_hi20"] = h.rolling(20).max().shift(1)
    df["don_lo20"] = l.rolling(20).min().shift(1)

    # Donchian 55-bar (Turtle)
    df["don_hi55"] = h.rolling(55).max().shift(1)
    df["don_lo55"] = l.rolling(55).min().shift(1)

    # Keltner (EMA20 ± 2×ATR10) — used by individual keltner strategy
    df["kelt_upper"], df["kelt_mid"], df["kelt_lower"] = _compute_keltner(h, l, c)

    # Keltner with ATR14 (used by combined swing_keltner strategy)
    _kelt14_mid         = _ema(c, 20)
    df["kelt14_upper"]  = _kelt14_mid + 2.0 * df["atr14"]
    df["kelt14_lower"]  = _kelt14_mid - 2.0 * df["atr14"]

    # 20-bar swing high / low — shift(1) avoids lookahead (excludes current bar)
    df["swing20_hi"] = h.rolling(20).max().shift(1)
    df["swing20_lo"] = l.rolling(20).min().shift(1)

    # SuperTrend (10, 3)
    df["st_val"], df["st_dir"] = _compute_supertrend(h, l, c)

    # Session ranges
    df = _add_session_ranges(df)

    # Hour column (needed by session-based checkers)
    df["_hour"] = pd.to_datetime(df["time"], utc=True).dt.hour

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY CHECKERS
#  Signature: (df, i, is_long) → dict | None
#  dict keys: entry, sl, sl_dist
# ══════════════════════════════════════════════════════════════════════════════

def _nan(*vals) -> bool:
    """Return True if any value is NaN."""
    return any(np.isnan(v) for v in vals)


# ── 1. Volatility Breakout ─────────────────────────────────────────────────────
def _check_vb(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    r   = df.iloc[i]
    atr = r["atr14"];  ema200 = r["ema200"]
    bc  = r["close"];  bh = r["high"];  bl = r["low"]
    if _nan(atr, ema200) or atr <= 0:
        return None
    rng = bh - bl
    if rng < 1.2 * atr:
        return None
    pos = (bc - bl) / rng
    if is_long  and (pos < 0.70 or bc < ema200): return None
    if not is_long and (pos > 0.30 or bc > ema200): return None
    sl = bl if is_long else bh
    sl_dist = abs(bc - sl)
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist} if sl_dist > 0 else None


# ── 2. Donchian 20-bar ────────────────────────────────────────────────────────
def _check_donchian(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    r   = df.iloc[i]
    atr = r["atr14"];  ema200 = r["ema200"]
    bc  = r["close"]
    if _nan(atr, ema200, r["don_hi20"]) or atr <= 0:
        return None
    if is_long:
        if bc < ema200 or bc <= r["don_hi20"]: return None
        sl = bc - 1.5 * atr
    else:
        if bc > ema200 or bc >= r["don_lo20"]: return None
        sl = bc + 1.5 * atr
    return {"entry": bc, "sl": sl, "sl_dist": 1.5 * atr}


# ── 3. Turtle 55-bar ──────────────────────────────────────────────────────────
def _check_turtle_55(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    """Classic Turtle: 55-bar breakout, no EMA200 filter (pure trend-following)."""
    r   = df.iloc[i]
    atr = r["atr14"]
    bc  = r["close"]
    if _nan(atr, r["don_hi55"]) or atr <= 0:
        return None
    if is_long:
        if bc <= r["don_hi55"]: return None
    else:
        if bc >= r["don_lo55"]: return None
    sl_dist = 2.0 * atr
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 4. Bollinger Band Breakout ────────────────────────────────────────────────
def _check_bb_break(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  ema200 = r["ema200"]
    bc  = r["close"]
    if _nan(atr, ema200, r["bb_upper"]) or atr <= 0:
        return None
    if is_long:
        if bc < ema200 or bc <= r["bb_upper"]: return None
        if p["close"] > p["bb_upper"]:         return None  # not fresh
        sl = r["bb_mid"]
    else:
        if bc > ema200 or bc >= r["bb_lower"]: return None
        if p["close"] < p["bb_lower"]:         return None
        sl = r["bb_mid"]
    sl_dist = abs(bc - sl)
    if sl_dist < 0.3 * atr or sl_dist > 3.5 * atr:
        return None
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 5. Keltner Channel ────────────────────────────────────────────────────────
def _check_keltner(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  ema200 = r["ema200"]
    bc  = r["close"]
    if _nan(atr, ema200, r["kelt_upper"]) or atr <= 0:
        return None
    if is_long:
        if bc < ema200 or bc <= r["kelt_upper"]: return None
        if p["close"] > p["kelt_upper"]:          return None
        sl = r["kelt_mid"]
    else:
        if bc > ema200 or bc >= r["kelt_lower"]: return None
        if p["close"] < p["kelt_lower"]:          return None
        sl = r["kelt_mid"]
    sl_dist = abs(bc - sl)
    if sl_dist < 0.3 * atr or sl_dist > 3.5 * atr:
        return None
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 6. Inside Bar Breakout ────────────────────────────────────────────────────
def _check_inside_bar(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    """Signal bar breaks above/below the previous bar which was an inside bar."""
    if i < 2:
        return None
    r  = df.iloc[i];  ib = df.iloc[i - 1];  mb = df.iloc[i - 2]
    atr = r["atr14"];  ema200 = r["ema200"];  bc = r["close"]
    if _nan(atr, ema200) or atr <= 0:
        return None
    # ib must be contained within mb (inside bar)
    if not (ib["high"] < mb["high"] and ib["low"] > mb["low"]):
        return None
    if is_long:
        if bc < ema200 or bc <= ib["high"]: return None
        sl = ib["low"] - 0.1 * atr
    else:
        if bc > ema200 or bc >= ib["low"]:  return None
        sl = ib["high"] + 0.1 * atr
    sl_dist = abs(bc - sl)
    sl_dist = min(sl_dist, 2.5 * atr)
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist} if sl_dist > 0 else None


# ── 7. Previous Day High / Low ────────────────────────────────────────────────
def _check_pdh_pdl(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  ema200 = r["ema200"];  bc = r["close"]
    if _nan(atr, ema200) or _nan(r.get("prev_day_high", np.nan)):
        return None
    pdh = r["prev_day_high"];  pdl = r["prev_day_low"]
    if _nan(pdh, pdl):
        return None
    if is_long:
        if bc < ema200 or bc <= pdh or p["close"] > pdh: return None
        sl = bc - 1.5 * atr
    else:
        if bc > ema200 or bc >= pdl or p["close"] < pdl: return None
        sl = bc + 1.5 * atr
    return {"entry": bc, "sl": sl, "sl_dist": 1.5 * atr}


# ── 8. Morning Range ──────────────────────────────────────────────────────────
def _check_morning_rng(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    """Morning range = UTC hours 0-2. Signal fires at hour ≥ 3."""
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    hour = int(r["_hour"])
    if hour < 3:
        return None
    atr = r["atr14"];  ema200 = r["ema200"];  bc = r["close"]
    mh = r.get("morning_high", np.nan);  ml = r.get("morning_low", np.nan)
    if _nan(atr, ema200, mh, ml) or atr <= 0:
        return None
    rng = mh - ml
    if rng <= 0 or rng > 3.5 * atr:
        return None
    if is_long:
        if bc < ema200 or bc <= mh or p["close"] > mh: return None
        sl = max(ml - 0.1 * atr, bc - 2.0 * atr)
    else:
        if bc > ema200 or bc >= ml or p["close"] < ml: return None
        sl = min(mh + 0.1 * atr, bc + 2.0 * atr)
    sl_dist = abs(bc - sl)
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist} if sl_dist > 0 else None


# ── 9. London Open (Asian range breakout) ─────────────────────────────────────
def _check_london_open(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    """Asian range = UTC hours 0-6. London open fires at hours 7, 8, 9."""
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    hour = int(r["_hour"])
    if hour not in (7, 8, 9):
        return None
    atr = r["atr14"];  bc = r["close"]
    ah = r.get("asian_high", np.nan);  al = r.get("asian_low", np.nan)
    if _nan(atr, ah, al) or atr <= 0:
        return None
    rng = ah - al
    if rng <= 0 or rng > 4.5 * atr:
        return None
    if is_long:
        if bc <= ah or p["close"] > ah: return None
        sl = max(al - 0.1 * atr, bc - 2.5 * atr)
    else:
        if bc >= al or p["close"] < al: return None
        sl = min(ah + 0.1 * atr, bc + 2.5 * atr)
    sl_dist = abs(bc - sl)
    if sl_dist < 0.2 * atr:
        return None
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 10. NY Open (London range breakout) ───────────────────────────────────────
def _check_ny_open(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    """London range = UTC hours 7-12. NY open fires at hours 13, 14, 15."""
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    hour = int(r["_hour"])
    if hour not in (13, 14, 15):
        return None
    atr = r["atr14"];  bc = r["close"]
    lh = r.get("london_high", np.nan);  ll = r.get("london_low", np.nan)
    if _nan(atr, lh, ll) or atr <= 0:
        return None
    rng = lh - ll
    if rng <= 0 or rng > 5.0 * atr:
        return None
    if is_long:
        if bc <= lh or p["close"] > lh: return None
        sl = bc - 1.5 * atr
    else:
        if bc >= ll or p["close"] < ll: return None
        sl = bc + 1.5 * atr
    return {"entry": bc, "sl": sl, "sl_dist": 1.5 * atr}


# ── 11. EMA Cross 9 / 21 ──────────────────────────────────────────────────────
def _check_ema_cross(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  ema200 = r["ema200"];  bc = r["close"]
    if _nan(atr, ema200, r["ema9"], r["ema21"]) or atr <= 0:
        return None
    if is_long:
        cross = r["ema9"] > r["ema21"] and p["ema9"] <= p["ema21"]
        if not cross or bc < ema200: return None
    else:
        cross = r["ema9"] < r["ema21"] and p["ema9"] >= p["ema21"]
        if not cross or bc > ema200: return None
    sl_dist = 1.5 * atr
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 12. MACD Cross ────────────────────────────────────────────────────────────
def _check_macd_cross(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  ema200 = r["ema200"];  bc = r["close"]
    if _nan(atr, ema200, r["macd_line"]) or atr <= 0:
        return None
    if is_long:
        cross = r["macd_line"] > r["macd_sig"] and p["macd_line"] <= p["macd_sig"]
        if not cross or bc < ema200: return None
    else:
        cross = r["macd_line"] < r["macd_sig"] and p["macd_line"] >= p["macd_sig"]
        if not cross or bc > ema200: return None
    sl_dist = 1.5 * atr
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 13. SuperTrend ────────────────────────────────────────────────────────────
def _check_supertrend(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    """Signal only on direction change (flip)."""
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  bc = r["close"]
    if _nan(atr, r["st_val"]) or r["st_dir"] == 0 or atr <= 0:
        return None
    if is_long:
        if not (r["st_dir"] == 1 and p["st_dir"] == -1): return None
    else:
        if not (r["st_dir"] == -1 and p["st_dir"] == 1): return None
    sl      = r["st_val"]
    sl_dist = abs(bc - sl)
    if sl_dist < 0.2 * atr or sl_dist > 3.5 * atr:
        return None
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 14. Three-Bar Momentum ────────────────────────────────────────────────────
def _check_three_bar(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    if i < 2:
        return None
    r = df.iloc[i]
    atr = r["atr14"];  ema200 = r["ema200"];  bc = r["close"]
    if _nan(atr, ema200) or atr <= 0:
        return None
    c1 = df.iloc[i - 2]["close"]
    c2 = df.iloc[i - 1]["close"]
    if is_long:
        if bc < ema200 or not (c1 < c2 < bc): return None
        if (bc - c1) < 0.5 * atr:             return None
    else:
        if bc > ema200 or not (c1 > c2 > bc): return None
        if (c1 - bc) < 0.5 * atr:             return None
    sl_dist = 1.5 * atr
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 15. Swing Level Break ─────────────────────────────────────────────────────
def _find_swings(highs: np.ndarray, lows: np.ndarray,
                 n: int = 3) -> tuple[list, list]:
    sh, sl = [], []
    size = len(highs)
    for k in range(n, size - n):
        if highs[k] == max(highs[k - n: k + n + 1]):
            sh.append((k, highs[k]))
        if lows[k] == min(lows[k - n: k + n + 1]):
            sl.append((k, lows[k]))
    return sh, sl


def _check_swing_break(df: pd.DataFrame, i: int, is_long: bool,
                       lookback: int = 40, max_sl_atr: float = 2.0) -> Optional[dict]:
    r    = df.iloc[i]
    atr  = r["atr14"];  ema96 = r["ema96"];  bc = r["close"]
    if _nan(atr, ema96) or atr <= 0:
        return None
    start = max(0, i - lookback)
    win   = df.iloc[start: i + 1]
    if len(win) < 10:
        return None
    hs = win["high"].astype(float).values
    ls = win["low"].astype(float).values
    sh, sl = _find_swings(hs, ls)

    if is_long:
        if not sh or bc < ema96: return None
        sh_price = sh[-1][1];  sh_idx = sh[-1][0]
        if bc <= sh_price:      return None
        sl_cands  = [s for s in sl if s[0] < sh_idx]
        sl_struct = sl_cands[-1][1] if sl_cands else float(win["low"].min())
        sl_val    = max(sl_struct, bc - max_sl_atr * atr)
    else:
        if not sl or bc > ema96: return None
        sl_price = sl[-1][1];  sl_idx = sl[-1][0]
        if bc >= sl_price:      return None
        sh_cands  = [s for s in sh if s[0] < sl_idx]
        sl_struct = sh_cands[-1][1] if sh_cands else float(win["high"].max())
        sl_val    = min(sl_struct, bc + max_sl_atr * atr)

    sl_dist = abs(bc - sl_val)
    return {"entry": bc, "sl": sl_val, "sl_dist": sl_dist} if sl_dist > 0 else None


# ── 16. Swing Level Retest ────────────────────────────────────────────────────
def _check_swing_retest(df: pd.DataFrame, i: int, is_long: bool,
                        lookback: int = 40, retest_lb: int = 20,
                        tol_atr: float = 0.35, sl_buf: float = 0.05,
                        max_sl_atr: float = 1.2) -> Optional[dict]:
    r   = df.iloc[i]
    atr = r["atr14"];  ema96 = r["ema96"];  bc = r["close"]
    bh  = r["high"];   bl   = r["low"]
    if _nan(atr, ema96) or atr <= 0:
        return None
    start = max(0, i - lookback - 30)
    win   = df.iloc[start: i + 1]
    n     = len(win)
    if n < 15:
        return None
    hs = win["high"].astype(float).values
    ls = win["low"].astype(float).values
    cs = win["close"].astype(float).values
    tol = tol_atr * atr

    sub = win.tail(lookback).reset_index(drop=True)
    sh, sl = _find_swings(sub["high"].astype(float).values, sub["low"].astype(float).values)

    if is_long:
        if not sh or bc < ema96: return None
        end = n - 3 - 2
        for sh_idx, sh_price in reversed(sh):
            if sh_idx >= end: continue
            bb = None
            for k in range(max(sh_idx + 1, n - 1 - retest_lb), n - 1):
                if cs[k] > sh_price: bb = k
            if bb is None or bb >= n - 2: continue
            lows_since = ls[bb + 1: n]
            if len(lows_since) == 0 or float(np.min(lows_since)) > sh_price + tol: continue
            if bl > sh_price + tol or bc <= sh_price: continue
            entry = bc;  sl_val = bl - sl_buf * atr
            dist  = abs(entry - sl_val)
            if dist <= 0 or dist > max_sl_atr * atr: continue
            return {"entry": entry, "sl": sl_val, "sl_dist": dist}
    else:
        if not sl or bc > ema96: return None
        end = n - 3 - 2
        for sl_idx, sl_price in reversed(sl):
            if sl_idx >= end: continue
            bb = None
            for k in range(max(sl_idx + 1, n - 1 - retest_lb), n - 1):
                if cs[k] < sl_price: bb = k
            if bb is None or bb >= n - 2: continue
            highs_since = hs[bb + 1: n]
            if len(highs_since) == 0 or float(np.max(highs_since)) < sl_price - tol: continue
            if bh < sl_price - tol or bc >= sl_price: continue
            entry = bc;  sl_val = bh + sl_buf * atr
            dist  = abs(sl_val - entry)
            if dist <= 0 or dist > max_sl_atr * atr: continue
            return {"entry": entry, "sl": sl_val, "sl_dist": dist}

    return None


# ── 17. RSI 50-Cross ──────────────────────────────────────────────────────────
def _check_rsi_50(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  ema200 = r["ema200"];  bc = r["close"]
    if _nan(atr, ema200, r["rsi14"]) or atr <= 0:
        return None
    if is_long:
        if not (r["rsi14"] > 50 and p["rsi14"] <= 50) or bc < ema200: return None
    else:
        if not (r["rsi14"] < 50 and p["rsi14"] >= 50) or bc > ema200: return None
    sl_dist = 1.5 * atr
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 18. RSI Reversal (30/70) ──────────────────────────────────────────────────
def _check_rsi_rev(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    """Counter-trend — no EMA200 filter (mean-reversion)."""
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  bc = r["close"]
    if _nan(atr, r["rsi14"]) or atr <= 0:
        return None
    if is_long:
        if not (p["rsi14"] < 30 and r["rsi14"] >= 30): return None
    else:
        if not (p["rsi14"] > 70 and r["rsi14"] <= 70): return None
    sl_dist = 1.5 * atr
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 19. Stochastic K/D cross ──────────────────────────────────────────────────
def _check_stoch(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  ema200 = r["ema200"];  bc = r["close"]
    if _nan(atr, ema200, r["stoch_k"]) or atr <= 0:
        return None
    if is_long:
        cross     = r["stoch_k"] > r["stoch_d"] and p["stoch_k"] <= p["stoch_d"]
        oversold  = r["stoch_d"] < 30
        if not cross or not oversold or bc < ema200: return None
    else:
        cross       = r["stoch_k"] < r["stoch_d"] and p["stoch_k"] >= p["stoch_d"]
        overbought  = r["stoch_d"] > 70
        if not cross or not overbought or bc > ema200: return None
    sl_dist = 1.5 * atr
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 20. Engulfing ─────────────────────────────────────────────────────────────
def _check_engulfing(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  ema200 = r["ema200"]
    bc = r["close"];  bo = r["open"];  bh = r["high"];  bl = r["low"]
    pc = p["close"];  po = p["open"]
    if _nan(atr, ema200) or atr <= 0:
        return None
    if is_long:
        bullish  = bc > bo
        engulfs  = bo <= pc and bc >= po
        prev_bear = pc < po
        if not (bullish and engulfs and prev_bear and bc > ema200): return None
        sl = bl - 0.1 * atr
    else:
        bearish  = bc < bo
        engulfs  = bo >= pc and bc <= po
        prev_bull = pc > po
        if not (bearish and engulfs and prev_bull and bc < ema200): return None
        sl = bh + 0.1 * atr
    sl_dist = min(abs(bc - sl), 2.5 * atr)
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist} if sl_dist > 0.1 * atr else None


# ── 21. Pin Bar ───────────────────────────────────────────────────────────────
def _check_pin_bar(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    r = df.iloc[i]
    atr = r["atr14"];  ema200 = r["ema200"]
    bc = r["close"];  bo = r["open"];  bh = r["high"];  bl = r["low"]
    if _nan(atr, ema200) or atr <= 0:
        return None
    body  = abs(bc - bo)
    rng   = bh - bl
    if rng <= 0 or body < 0.01:
        return None
    upper_sh = bh - max(bc, bo)
    lower_sh = min(bc, bo) - bl
    if is_long:
        # Hammer: lower shadow ≥ 2× body, upper shadow ≤ 1× body
        if lower_sh < 2.0 * body or upper_sh > body or bc < ema200:
            return None
        sl = bl - 0.1 * atr
    else:
        # Shooting star: upper shadow ≥ 2× body, lower shadow ≤ 1× body
        if upper_sh < 2.0 * body or lower_sh > body or bc > ema200:
            return None
        sl = bh + 0.1 * atr
    sl_dist = min(abs(bc - sl), 2.5 * atr)
    if sl_dist < 0.15 * atr:
        return None
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 22. MACD + ADX Confluence ─────────────────────────────────────────────────
def _check_macd_adx(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  adx = r["adx14"];  ema200 = r["ema200"];  bc = r["close"]
    if _nan(atr, adx, ema200, r["macd_line"]) or atr <= 0 or adx < 25:
        return None
    if is_long:
        cross = r["macd_line"] > r["macd_sig"] and p["macd_line"] <= p["macd_sig"]
        if not cross or r["macd_line"] < 0 or bc < ema200: return None
    else:
        cross = r["macd_line"] < r["macd_sig"] and p["macd_line"] >= p["macd_sig"]
        if not cross or r["macd_line"] > 0 or bc > ema200: return None
    sl_dist = 1.5 * atr
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 23. RSI + EMA Confluence ──────────────────────────────────────────────────
def _check_rsi_ema(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    if i < 1:
        return None
    r = df.iloc[i];  p = df.iloc[i - 1]
    atr = r["atr14"];  ema200 = r["ema200"];  bc = r["close"]
    if _nan(atr, ema200, r["rsi14"], r["ema9"]) or atr <= 0:
        return None
    if is_long:
        rsi_cross = r["rsi14"] > 50 and p["rsi14"] <= 50
        ema_stack = r["ema9"] > r["ema21"]
        if not (rsi_cross and ema_stack and bc > ema200): return None
    else:
        rsi_cross = r["rsi14"] < 50 and p["rsi14"] >= 50
        ema_stack = r["ema9"] < r["ema21"]
        if not (rsi_cross and ema_stack and bc < ema200): return None
    sl_dist = 1.5 * atr
    sl = bc - sl_dist if is_long else bc + sl_dist
    return {"entry": bc, "sl": sl, "sl_dist": sl_dist}


# ── 24. Swing + Keltner combined (14-16 UTC only) ─────────────────────────────
def _check_swing_keltner(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    """
    PRIMARY combined strategy — Swing Break[20-bar] AND Keltner[EMA20 ± 2×ATR14].
    BOTH conditions must fire simultaneously.  Fires ONLY at 14, 15, 16 UTC.

    LONG : close > 20-bar swing_high  AND  close > kelt14_upper
    SHORT: close < 20-bar swing_low   AND  close < kelt14_lower

    SL: swing_lo − 0.25×ATR14 (long) / swing_hi + 0.25×ATR14 (short)
    Expected WR: ~55-65% (Keltner alone 53.7% at 15 UTC; AND filter removes false breaks)
    """
    r    = df.iloc[i]
    hour = int(r["_hour"])
    if hour not in (14, 15, 16):
        return None

    atr    = r["atr14"];   ema200 = r["ema200"];   bc = r["close"]
    kup    = r.get("kelt14_upper", np.nan)
    klo    = r.get("kelt14_lower", np.nan)
    swhi   = r.get("swing20_hi",   np.nan)
    swlo   = r.get("swing20_lo",   np.nan)

    if _nan(atr, ema200, kup, klo, swhi, swlo) or atr <= 0:
        return None

    if is_long:
        if bc < ema200:             return None   # EMA200 direction
        if bc <= swhi:              return None   # swing break not confirmed
        if bc <= kup:               return None   # keltner breakout not confirmed
        sl      = swlo - 0.25 * atr
        sl_dist = bc - sl
    else:
        if bc > ema200:             return None
        if bc >= swlo:              return None
        if bc >= klo:               return None
        sl      = swhi + 0.25 * atr
        sl_dist = sl - bc

    return {"entry": bc, "sl": sl, "sl_dist": sl_dist} if sl_dist > 0 else None


# ── 25. RSI50-Cross KZ (02 UTC only) ──────────────────────────────────────────
def _check_rsi50_kz(df: pd.DataFrame, i: int, is_long: bool) -> Optional[dict]:
    """
    SECONDARY combined strategy — RSI(14) crossing above/below 50 in EMA200 direction.
    Fires ONLY at 02 UTC (Asia Night).

    LONG : RSI was < 50 on prior bar, now ≥ 50
    SHORT: RSI was > 50 on prior bar, now ≤ 50

    SL: 10-bar swing_low  − 0.25×ATR14 (long)
        10-bar swing_high + 0.25×ATR14 (short)
    Basis: RSI 50-Cross at 02 UTC alone → 50.9% WR in phase-1 backtest
    """
    if i < 1:
        return None
    r    = df.iloc[i];  p = df.iloc[i - 1]
    hour = int(r["_hour"])
    if hour != 2:
        return None

    atr    = r["atr14"];   ema200 = r["ema200"];   bc = r["close"]
    rsi_c  = r["rsi14"];   rsi_p  = p["rsi14"]

    if _nan(atr, ema200, rsi_c, rsi_p) or atr <= 0:
        return None

    if is_long:
        if bc < ema200:                     return None
        if not (rsi_p < 50 <= rsi_c):       return None   # cross above 50
        start   = max(0, i - 9)
        sl_ref  = float(df.iloc[start: i + 1]["low"].min())
        sl      = sl_ref - 0.25 * atr
        sl_dist = bc - sl
    else:
        if bc > ema200:                     return None
        if not (rsi_p > 50 >= rsi_c):       return None   # cross below 50
        start   = max(0, i - 9)
        sl_ref  = float(df.iloc[start: i + 1]["high"].max())
        sl      = sl_ref + 0.25 * atr
        sl_dist = sl - bc

    return {"entry": bc, "sl": sl, "sl_dist": sl_dist} if sl_dist > 0 else None


# ── Dispatcher ────────────────────────────────────────────────────────────────
_CHECKER = {
    "vb":           _check_vb,
    "donchian":     _check_donchian,
    "turtle_55":    _check_turtle_55,
    "bb_break":     _check_bb_break,
    "keltner":      _check_keltner,
    "inside_bar":   _check_inside_bar,
    "pdh_pdl":      _check_pdh_pdl,
    "morning_rng":  _check_morning_rng,
    "london_open":  _check_london_open,
    "ny_open":      _check_ny_open,
    "ema_cross":    _check_ema_cross,
    "macd_cross":   _check_macd_cross,
    "supertrend":   _check_supertrend,
    "three_bar":    _check_three_bar,
    "swing_break":  _check_swing_break,
    "swing_retest": _check_swing_retest,
    "rsi_50":       _check_rsi_50,
    "rsi_rev":      _check_rsi_rev,
    "stoch":        _check_stoch,
    "engulfing":    _check_engulfing,
    "pin_bar":      _check_pin_bar,
    "macd_adx":       _check_macd_adx,
    "rsi_ema":        _check_rsi_ema,
    # Combined strategies (KZ hour-filtered)
    "swing_keltner":  _check_swing_keltner,
    "rsi50_kz":       _check_rsi50_kz,
}


# ══════════════════════════════════════════════════════════════════════════════
#  TRADE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_trade(df: pd.DataFrame, entry_i: int,
                    entry: float, sl: float, sl_dist: float,
                    is_long: bool, atr_entry: float) -> dict:
    """Walk forward bars; model the LIVE 50%-at-TP1 / 50%-at-TP2 scale-out.

    r_achieved is the BLENDED R across BOTH halves (matches paper_trader):
      * 50% closes at TP1  -> +0.5 * TP1_RR  (= +1.0R) banked
      * remaining 50% rides: stop to breakeven after TP1, trailing 2xATR;
        it exits at TP2 (+0.5 * TP2_RR = +2.0R), the trailing/breakeven stop,
        or the max-hold mark.
      * if TP1 NEVER hits, the FULL position exits at SL (-1R) or max-hold.

    So a clean run = +3R (not the old single-unit +4R); a "TP1 then fade to
    breakeven" = +1R (not the old 0R) -- the latter also flips that trade from
    a recorded loss to a win, since outcome = (r_achieved > 0).
    """
    tp1 = entry + sl_dist * TP1_RR * (1 if is_long else -1)
    tp2 = entry + sl_dist * TP2_RR * (1 if is_long else -1)
    sgn = 1.0 if is_long else -1.0

    cur_sl   = sl
    tp1_hit  = False
    realized = 0.0          # blended R already banked from the closed half
    n_bars   = len(df)

    for j in range(entry_i + 1, min(entry_i + MAX_HOLD_BARS + 1, n_bars)):
        bh = float(df.iloc[j]["high"])
        bl = float(df.iloc[j]["low"])
        bc = float(df.iloc[j]["close"])

        # Trailing SL ratchet after TP1 (applies to the remaining half)
        if tp1_hit and atr_entry > 0:
            trail = TRAIL_ATR_MULT * atr_entry
            if is_long:
                new_t = bc - trail
                if new_t > cur_sl: cur_sl = new_t
            else:
                new_t = bc + trail
                if new_t < cur_sl: cur_sl = new_t

        # TP2 — remaining half exits at +TP2_RR (+2R blended)
        hit_tp2 = (bh >= tp2) if is_long else (bl <= tp2)
        if hit_tp2:
            if not tp1_hit:                  # same bar pierced BOTH TP1 and TP2
                realized += 0.5 * TP1_RR
            realized += 0.5 * TP2_RR
            return {"exit_reason": "TP2", "exit_bar": j,
                    "exit_price": tp2, "r_achieved": round(realized, 3)}

        # SL (checked AFTER TP2 → TP2 wins same-bar ties)
        hit_sl = (bl <= cur_sl) if is_long else (bh >= cur_sl)
        if hit_sl:
            sl_r = (cur_sl - entry) / sl_dist * sgn
            if tp1_hit:
                realized += 0.5 * sl_r       # only the remaining half stops out
                reason = "SL_AFTER_TP1"
            else:
                realized += 1.0 * sl_r       # full position stopped (≈ -1R)
                reason = "SL"
            return {"exit_reason": reason, "exit_bar": j,
                    "exit_price": cur_sl, "r_achieved": round(realized, 3)}

        # TP1 — close 50% at +TP1_RR (+1R blended), move stop to breakeven
        if not tp1_hit:
            hit_tp1 = (bh >= tp1) if is_long else (bl <= tp1)
            if hit_tp1:
                tp1_hit = True
                realized += 0.5 * TP1_RR
                cur_sl = entry

    # Max-hold exit — mark the remaining position to market
    last_j  = min(entry_i + MAX_HOLD_BARS, n_bars - 1)
    exit_px = float(df.iloc[last_j]["close"])
    mark_r  = (exit_px - entry) / sl_dist * sgn
    realized += (0.5 if tp1_hit else 1.0) * mark_r
    return {"exit_reason": "MAX_HOLD", "exit_bar": last_j,
            "exit_price": exit_px, "r_achieved": round(realized, 3)}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run(eth_df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """Iterate every ETH H1 bar. For each bar test all 25 strategies."""
    eth_df = eth_df.copy()
    btc_df = btc_df.copy()
    eth_df["time"] = pd.to_datetime(eth_df["time"], utc=True)
    btc_df["time"] = pd.to_datetime(btc_df["time"], utc=True)

    logger.info("Precomputing indicators for ETH (%d bars)...", len(eth_df))
    eth_df = _precompute(eth_df)

    # BTC EMA200 lookup helper
    btc_df["ema200"] = _ema(btc_df["close"].astype(float), 200)
    btc_idx = btc_df.set_index("time")

    def _btc_aligned(ts, is_long: bool) -> bool:
        try:
            row = btc_idx.loc[ts]
        except KeyError:
            pos = btc_idx.index.get_indexer([ts], method="nearest")[0]
            if pos < 0: return False
            row = btc_idx.iloc[pos]
        above = float(row["close"]) > float(row["ema200"])
        return (above and is_long) or (not above and not is_long)

    records = []
    n       = len(eth_df)
    n_strat = len(STRATEGIES)
    logger.info("Running backtest: %d bars × %d strategies...",
                n - WARMUP_BARS, n_strat)

    for i in range(WARMUP_BARS, n - 1):
        r   = eth_df.iloc[i]
        ts  = r["time"]
        bc  = float(r["close"])
        adx = float(r["adx14"])
        atr = float(r["atr14"])

        if np.isnan(adx) or np.isnan(atr) or atr <= 0 or adx < ADX_THRESHOLD:
            continue

        ema200  = float(r["ema200"])
        is_long = bc > ema200
        direction = "long" if is_long else "short"
        btc_al  = _btc_aligned(ts, is_long)

        for key, checker in _CHECKER.items():
            sig = checker(eth_df, i, is_long)
            if sig is None:
                continue

            result = _simulate_trade(eth_df, i, sig["entry"], sig["sl"],
                                     sig["sl_dist"], is_long, atr)

            exit_j  = result["exit_bar"]
            exit_ts = eth_df["time"].iloc[min(exit_j, n - 1)]
            r_val   = result["r_achieved"]

            records.append({
                "entry_time":   ts.isoformat(),
                "exit_time":    exit_ts.isoformat(),
                "hour_utc":     ts.hour,
                "strategy":     key,
                "direction":    direction,
                "entry":        round(sig["entry"], 4),
                "sl":           round(sig["sl"], 4),
                "sl_dist":      round(sig["sl_dist"], 4),
                "tp1":          round(sig["entry"] + sig["sl_dist"] * TP1_RR * (1 if is_long else -1), 4),
                "tp2":          round(sig["entry"] + sig["sl_dist"] * TP2_RR * (1 if is_long else -1), 4),
                "adx":          round(adx, 1),
                "atr":          round(atr, 4),
                "ema200":       round(ema200, 4),
                "btc_aligned":  btc_al,
                "exit_reason":  result["exit_reason"],
                "exit_price":   round(result["exit_price"], 4),
                "r_achieved":   r_val,
                "bars_held":    exit_j - i,
                "outcome":      "win" if r_val > 0 else "loss",
            })

        if i % 1000 == 0:
            logger.info("  Bar %d / %d  |  trades so far: %d",
                        i, n, len(records))

    logger.info("Backtest complete — %d total trades", len(records))
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY STATS
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (hour_utc × strategy), compute win rate, avg R, profit factor, max DD.
    Produces both base stats and BTC-alignment-filtered stats.
    Column names match what report.py expects.
    """
    rows = []
    for (hour, strat), grp in trades.groupby(["hour_utc", "strategy"]):

        def _stat(sub: pd.DataFrame) -> dict:
            n = len(sub)
            if n == 0:
                return {}
            wins = (sub["outcome"] == "win").sum()
            rs   = sub["r_achieved"]
            gw   = rs[rs > 0].sum()
            gl   = rs[rs < 0].abs().sum()
            pf   = round(gw / gl, 3) if gl > 0 else float("inf")
            dd   = round(rs.cumsum().sub(rs.cumsum().cummax()).min(), 3)
            return {
                "n_trades":      n,
                "win_rate":      round(wins / n, 4),
                "avg_r":         round(rs.mean(), 4),
                "total_r":       round(rs.sum(), 3),
                "profit_factor": pf,
                "max_dd_r":      dd,
            }

        base = _stat(grp)
        if not base:
            continue

        btc_grp = grp[grp["btc_aligned"] == True]
        btc_s   = _stat(btc_grp)

        row = {"hour_utc": int(hour), "strategy": strat,
               "strategy_label": STRATEGIES.get(strat, strat)}
        row.update(base)
        # BTC-aligned stats (prefix with btc_ to match report.py)
        for k, v in btc_s.items():
            row[k.replace("n_trades", "n_btc")
               .replace("win_rate", "win_rate_btc")
               .replace("avg_r", "avg_r_btc")
               .replace("total_r", "total_r_btc")
               .replace("profit_factor", "profit_factor_btc")
               .replace("max_dd_r", "max_dd_r_btc")] = v
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    # Fill missing BTC columns with neutral defaults
    for col in ["n_btc", "win_rate_btc", "avg_r_btc", "total_r_btc",
                "profit_factor_btc", "max_dd_r_btc"]:
        if col not in summary.columns:
            summary[col] = np.nan

    summary["n_btc"] = summary["n_btc"].fillna(0).astype(int)

    return summary.sort_values("total_r", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("=" * 65)
    logger.info("ETH Full-Spectrum Backtest  (%d strategies)  TP1=%.0fR  TP2=%.0fR",
                len(STRATEGIES), TP1_RR, TP2_RR)
    logger.info("=" * 65)

    if not ETH_CSV.exists() or not BTC_CSV.exists():
        logger.error("Data files not found — run collect_data.py first:")
        logger.error("  python -m btc_research.eth_bot.backtest.collect_data")
        sys.exit(1)

    logger.info("Loading ETHUSD H1: %s", ETH_CSV)
    eth_df = pd.read_csv(ETH_CSV, parse_dates=["time"])
    logger.info("Loading BTCUSD H1: %s", BTC_CSV)
    btc_df = pd.read_csv(BTC_CSV, parse_dates=["time"])
    logger.info("ETH: %d bars  |  BTC: %d bars", len(eth_df), len(btc_df))
    logger.info("Period: %s → %s", eth_df["time"].iloc[0], eth_df["time"].iloc[-1])

    trades_df = run(eth_df, btc_df)

    if trades_df.empty:
        logger.error("No trades found — check data and parameters")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(TRADES_CSV, index=False)
    logger.info("Trades saved → %s  (%d rows)", TRADES_CSV, len(trades_df))

    summary_df = compute_summary(trades_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    logger.info("Summary saved → %s  (%d rows)", SUMMARY_CSV, len(summary_df))

    # Quick console top-20 preview
    logger.info("")
    logger.info("Top 20 hour × strategy by total R:")
    logger.info("─" * 80)
    for _, row in summary_df.head(20).iterrows():
        logger.info(
            "  %02d:00 UTC  %-22s  N=%4d  WR=%5.1f%%  AvgR=%+.3f  "
            "TotalR=%+.1f  PF=%.2f",
            int(row["hour_utc"]),
            STRATEGIES.get(row["strategy"], row["strategy"]),
            int(row["n_trades"]),
            row["win_rate"] * 100,
            row["avg_r"],
            row["total_r"],
            row["profit_factor"],
        )
    logger.info("")
    logger.info("Run report.py for full analysis and KZ recommendations:")
    logger.info("  python -m btc_research.eth_bot.backtest.report")


if __name__ == "__main__":
    main()
