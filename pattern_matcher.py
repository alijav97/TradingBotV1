"""
pattern_matcher.py - Historical Pattern Matching for TradingBotV1

Compares current XAUUSD market conditions against 2 years of historical
data to find similar setups and analyse what happened next each time.
"""

import os
import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# Reuse the data-download and indicator functions from backtest.py
from backtest import get_historical_data, SYMBOL, PERIOD, INTERVAL, SESSIONS

# ── Constants ─────────────────────────────────────────────────────────────────

PIP_SIZE       = 0.1      # $0.10 per pip for gold
MOVE_THRESHOLD = 50.0     # pips; minimum move to count as UP or DOWN
LOOKAHEAD      = 24       # candles to look ahead after a match

# RSI zone boundaries
RSI_OVERSOLD   = 35
RSI_OVERBOUGHT = 65


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rsi_zone(rsi: float) -> str:
    if rsi < RSI_OVERSOLD:
        return "oversold"
    if rsi > RSI_OVERBOUGHT:
        return "overbought"
    return "neutral"


def _session_now() -> str:
    h = datetime.now(timezone.utc).hour
    for name, (start, end) in SESSIONS.items():
        if start <= end:
            if start <= h < end:
                return name
        else:
            if h >= start or h < end:
                return name
    return "Off-hours"


def _candle_label(o: float, h: float, l: float, c: float) -> str:
    """Return a simple one-word label for a single OHLC candle."""
    body      = abs(c - o)
    rng       = h - l
    body_rat  = body / rng if rng > 0 else 0
    upper_w   = h - max(o, c)
    lower_w   = min(o, c) - l
    bullish   = c > o

    if body_rat < 0.1:
        return "doji"
    if lower_w > body * 2 and upper_w < body:
        return "hammer" if bullish else "hanging_man"
    if upper_w > body * 2 and lower_w < body:
        return "shooting_star"
    if body_rat > 0.75:
        return "marubozu_bull" if bullish else "marubozu_bear"
    return "candle_bull" if bullish else "candle_bear"


def _patterns_from_df(df: pd.DataFrame, idx: int, n: int = 5) -> list[str]:
    """Return candle labels for the n candles ending at idx (inclusive)."""
    start = max(0, idx - n + 1)
    labels = []
    for i in range(start, idx + 1):
        row = df.iloc[i]
        labels.append(_candle_label(row["open"], row["high"], row["low"], row["close"]))
    return labels


def _at_support_resistance(row: pd.Series) -> str:
    """Cheap proxy: compare close to Bollinger bands."""
    if row["close"] <= row["bb_lower"] * 1.002:
        return "support"
    if row["close"] >= row["bb_upper"] * 0.998:
        return "resistance"
    return "mid-range"


# ── 1. get_current_snapshot ───────────────────────────────────────────────────

def get_current_snapshot(df: pd.DataFrame) -> dict:
    """
    Capture the current (most-recent) market state from a historical DataFrame.

    In a live trading system you would fetch real-time data; here we use the
    last available row of the DataFrame as the "current" candle.

    Args:
        df: Full historical DataFrame from get_historical_data().

    Returns:
        Dict describing the current market snapshot.
    """
    row = df.iloc[-1]
    idx = len(df) - 1

    price          = float(row["close"])
    ema50          = float(row["ema50"])
    ema200         = float(row["ema200"])
    rsi            = float(row["rsi"])
    atr            = float(row["atr"])
    macd           = float(row["macd"])
    macd_signal    = float(row["macd_signal"])

    ema50_pct  = (price - ema50)  / ema50  * 100
    ema200_pct = (price - ema200) / ema200 * 100

    candle_patterns = _patterns_from_df(df, idx, n=5)
    sr_zone         = _at_support_resistance(row)
    session         = _session_now()
    day_of_week     = datetime.now(timezone.utc).strftime("%A")

    snapshot = {
        "price":              price,
        "ema50":              ema50,
        "ema200":             ema200,
        "ema50_pct":          round(ema50_pct,  3),   # % price is above/below EMA50
        "ema200_pct":         round(ema200_pct, 3),   # % price is above/below EMA200
        "above_ema50":        price > ema50,
        "above_ema200":       price > ema200,
        "rsi":                round(rsi, 1),
        "rsi_zone":           _rsi_zone(rsi),
        "atr":                round(atr, 3),
        "atr_pips":           round(atr / PIP_SIZE, 1),
        "macd_bullish":       macd > macd_signal,
        "candle_patterns":    candle_patterns,
        "sr_zone":            sr_zone,
        "session":            session,
        "day_of_week":        day_of_week,
        "snapshot_datetime":  str(row.get("datetime", "unknown")),
    }
    return snapshot


# ── 2. find_similar_historical ────────────────────────────────────────────────

def find_similar_historical(
    current_snapshot: dict,
    df: pd.DataFrame,
    top_n: int = 10,
) -> list[dict]:
    """
    Score every historical candle against the current snapshot and return
    the top_n most similar moments.

    Similarity is computed from four equally-weighted components (25% each):
        1. RSI level match
        2. EMA position match (both EMA50 and EMA200 relative to price)
        3. Candle pattern match (most recent candle label)
        4. Volatility (ATR) match

    Args:
        current_snapshot: Dict from get_current_snapshot().
        df:               Full historical DataFrame.
        top_n:            How many top matches to return.

    Returns:
        List of dicts with keys: index, datetime, similarity, row_data.
    """
    curr_rsi        = current_snapshot["rsi"]
    curr_ema50_pct  = current_snapshot["ema50_pct"]
    curr_ema200_pct = current_snapshot["ema200_pct"]
    curr_atr_pips   = current_snapshot["atr_pips"]
    curr_pattern    = current_snapshot["candle_patterns"][-1]  # most recent candle

    # Exclude last LOOKAHEAD candles so we always have a full outcome window
    scan_end = len(df) - LOOKAHEAD - 1
    scores: list[tuple[float, int]] = []

    for i in range(200, scan_end):   # skip first 200 warmup candles
        row = df.iloc[i]

        # ── RSI similarity (25%) ───────────────────────────────────────────
        rsi_diff   = abs(float(row["rsi"]) - curr_rsi)
        rsi_score  = max(0.0, 1.0 - rsi_diff / 50.0)

        # ── EMA position similarity (25%) ──────────────────────────────────
        price      = float(row["close"])
        e50_pct    = (price - float(row["ema50"]))  / float(row["ema50"])  * 100
        e200_pct   = (price - float(row["ema200"])) / float(row["ema200"]) * 100
        ema50_diff = abs(e50_pct  - curr_ema50_pct)
        ema200_diff= abs(e200_pct - curr_ema200_pct)
        ema_score  = max(0.0, 1.0 - (ema50_diff + ema200_diff) / 4.0)

        # ── Candle pattern similarity (25%) ────────────────────────────────
        hist_pattern = _candle_label(
            float(row["open"]), float(row["high"]),
            float(row["low"]),  float(row["close"])
        )
        # Exact match = 1.0; same bullish/bearish family = 0.5; else = 0.0
        if hist_pattern == curr_pattern:
            pattern_score = 1.0
        elif (("bull" in hist_pattern and "bull" in curr_pattern) or
              ("bear" in hist_pattern and "bear" in curr_pattern)):
            pattern_score = 0.5
        else:
            pattern_score = 0.0

        # ── ATR / volatility similarity (25%) ──────────────────────────────
        hist_atr_pips = float(row["atr"]) / PIP_SIZE
        atr_ratio     = min(hist_atr_pips, curr_atr_pips) / max(hist_atr_pips, curr_atr_pips) \
                        if max(hist_atr_pips, curr_atr_pips) > 0 else 1.0
        atr_score = atr_ratio  # 1.0 = identical volatility, 0.0 = totally different

        # ── Weighted total ─────────────────────────────────────────────────
        total = (rsi_score + ema_score + pattern_score + atr_score) / 4.0
        scores.append((total, i))

    # Sort descending and return top_n
    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:top_n]

    results = []
    for similarity, idx in top:
        row = df.iloc[idx]
        results.append({
            "index":      idx,
            "datetime":   row.get("datetime", "unknown"),
            "similarity": round(similarity * 100, 1),  # percent
            "rsi":        round(float(row["rsi"]), 1),
            "atr_pips":   round(float(row["atr"]) / PIP_SIZE, 1),
            "close":      round(float(row["close"]), 2),
            "session":    str(row.get("session", "")),
        })
    return results


# ── 3. analyse_outcomes ───────────────────────────────────────────────────────

def analyse_outcomes(
    similar_moments: list[dict],
    df: pd.DataFrame,
) -> dict:
    """
    For each similar historical moment, measure what happened in the next
    LOOKAHEAD candles (24 hours on H1 data).

    Args:
        similar_moments: Output of find_similar_historical().
        df:              Full historical DataFrame.

    Returns:
        Outcome summary dict.
    """
    up_moves:   list[float] = []
    down_moves: list[float] = []
    best_up_detail:   dict | None = None   # largest single up move
    best_down_detail: dict | None = None   # largest single down move

    for moment in similar_moments:
        idx       = moment["index"]
        entry     = float(df.iloc[idx]["close"])
        end_idx   = min(idx + LOOKAHEAD, len(df) - 1)

        # Max move up and down over the lookahead window
        future    = df.iloc[idx + 1 : end_idx + 1]
        if future.empty:
            continue

        max_high = float(future["high"].max())
        min_low  = float(future["low"].min())

        up_pips   = (max_high - entry) / PIP_SIZE
        down_pips = (entry - min_low)  / PIP_SIZE

        # Determine dominant direction for this instance
        if up_pips >= MOVE_THRESHOLD and up_pips > down_pips * 1.2:
            up_moves.append(up_pips)
            if best_up_detail is None or up_pips > best_up_detail["pips"]:
                best_up_detail = {
                    "pips":     round(up_pips, 1),
                    "datetime": moment["datetime"],
                    "rsi":      moment["rsi"],
                    "sim":      moment["similarity"],
                    "close":    moment["close"],
                }
        elif down_pips >= MOVE_THRESHOLD and down_pips > up_pips * 1.2:
            down_moves.append(down_pips)
            if best_down_detail is None or down_pips > best_down_detail["pips"]:
                best_down_detail = {
                    "pips":     round(down_pips, 1),
                    "datetime": moment["datetime"],
                    "rsi":      moment["rsi"],
                    "sim":      moment["similarity"],
                    "close":    moment["close"],
                }
        else:
            # Ambiguous — assign to whichever side is larger
            if up_pips >= down_pips:
                up_moves.append(up_pips)
            else:
                down_moves.append(down_pips)

    total = len(up_moves) + len(down_moves)
    if total == 0:
        return {"error": "No outcomes could be measured."}

    up_count   = len(up_moves)
    down_count = len(down_moves)
    up_pct     = up_count   / total * 100
    down_pct   = down_count / total * 100

    avg_up   = float(np.mean(up_moves))   if up_moves   else 0.0
    avg_down = float(np.mean(down_moves)) if down_moves else 0.0
    max_up   = float(max(up_moves))       if up_moves   else 0.0
    max_down = float(max(down_moves))     if down_moves else 0.0

    if up_pct >= 60:
        verdict    = "BULLISH BIAS"
        confidence = min(10, int(up_pct / 10))
        expected_pips = avg_up
    elif down_pct >= 60:
        verdict    = "BEARISH BIAS"
        confidence = min(10, int(down_pct / 10))
        expected_pips = -avg_down
    else:
        verdict    = "NEUTRAL / MIXED"
        confidence = max(1, 10 - abs(int((up_pct - down_pct) / 10)))
        expected_pips = avg_up if up_pct >= down_pct else -avg_down

    return {
        "total_matches":  total,
        "up_count":       up_count,
        "down_count":     down_count,
        "up_pct":         round(up_pct,   1),
        "down_pct":       round(down_pct, 1),
        "avg_up_pips":    round(avg_up,   1),
        "avg_down_pips":  round(avg_down, 1),
        "max_up_pips":    round(max_up,   1),
        "max_down_pips":  round(max_down, 1),
        "verdict":        verdict,
        "confidence":     confidence,
        "expected_pips":  round(expected_pips, 1),
        "best_up_setup":  best_up_detail,
        "best_down_setup":best_down_detail,
    }


# ── 4. print_historical_analysis ─────────────────────────────────────────────

def print_historical_analysis(df: pd.DataFrame | None = None) -> None:
    """
    Run the full pipeline and print the pattern-match analysis report.

    If df is None the function downloads fresh data automatically.
    """
    if df is None:
        print("  Downloading historical data...")
        df = get_historical_data()
        if df.empty:
            print("  [ERROR] Could not load data.")
            return

    print("\n  Capturing current market snapshot...")
    snapshot = get_current_snapshot(df)

    print(f"  Current price : {snapshot['price']}")
    print(f"  RSI           : {snapshot['rsi']} ({snapshot['rsi_zone']})")
    print(f"  ATR           : {snapshot['atr_pips']} pips")
    print(f"  vs EMA50      : {'+' if snapshot['above_ema50']  else ''}{snapshot['ema50_pct']}%")
    print(f"  vs EMA200     : {'+' if snapshot['above_ema200'] else ''}{snapshot['ema200_pct']}%")
    print(f"  S/R zone      : {snapshot['sr_zone']}")
    print(f"  Session       : {snapshot['session']}")
    print(f"  Last candles  : {', '.join(snapshot['candle_patterns'])}")

    print("\n  Scanning 2 years of history for similar setups...")
    similar = find_similar_historical(snapshot, df, top_n=10)

    outcomes = analyse_outcomes(similar, df)

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("=== HISTORICAL PATTERN ANALYSIS ===")
    print("=" * 55)

    n = outcomes.get("total_matches", 0)
    print(f"\nCurrent market conditions match {n} similar")
    print(f"historical setups from the past 2 years.")

    print(f"\nWHAT HAPPENED NEXT in those {n} cases:")

    up_count  = outcomes.get("up_count",    0)
    dn_count  = outcomes.get("down_count",  0)
    up_pct    = outcomes.get("up_pct",    0.0)
    dn_pct    = outcomes.get("down_pct",  0.0)
    avg_up    = outcomes.get("avg_up_pips",   0.0)
    avg_dn    = outcomes.get("avg_down_pips", 0.0)
    max_up    = outcomes.get("max_up_pips",   0.0)
    max_dn    = outcomes.get("max_down_pips", 0.0)

    print(f"  → Price went UP  : {up_count} times ({up_pct}%)")
    print(f"     Average move  : +{avg_up} pips")
    print(f"     Largest move  : +{max_up} pips")
    print(f"  → Price went DOWN: {dn_count} times ({dn_pct}%)")
    print(f"     Average move  : -{avg_dn} pips")
    print(f"     Largest move  : -{max_dn} pips")

    print(f"\n  HISTORICAL VERDICT : {outcomes.get('verdict', '—')}")
    exp = outcomes.get("expected_pips", 0)
    sign = "+" if exp >= 0 else ""
    print(f"  EXPECTED MOVE SIZE : {sign}{exp} pips average")
    print(f"  CONFIDENCE         : {outcomes.get('confidence', '—')}/10")

    # Best similar setup narrative
    best = outcomes.get("best_up_setup") or outcomes.get("best_down_setup")
    if best:
        pips       = best["pips"]
        direction  = "bounced" if outcomes.get("best_up_setup") else "dropped"
        hours      = LOOKAHEAD
        dt_raw     = best["datetime"]
        try:
            dt_label = pd.Timestamp(dt_raw).strftime("%B %d %Y")
        except Exception:
            dt_label = str(dt_raw)[:16]

        print(f"\n  MOST SIMILAR PAST SETUP:")
        print(f"    Date      : {dt_label}")
        print(
            f"    Similarity: {best['sim']}%  |  "
            f"RSI at the time: {best['rsi']}  |  "
            f"Price: {best['close']}"
        )
        print(
            f"    What happened: Price {direction} {pips} pips "
            f"over the next {hours} hours"
        )

    print("\n  Top 10 similar historical moments:")
    print(f"  {'#':<3} {'Date':<22} {'Sim%':>5} {'RSI':>6} {'ATR(pips)':>10} {'Session':<12}")
    print("  " + "-" * 60)
    for i, m in enumerate(similar, start=1):
        try:
            dt_str = pd.Timestamp(m["datetime"]).strftime("%Y-%m-%d %H:%M")
        except Exception:
            dt_str = str(m["datetime"])[:16]
        print(
            f"  {i:<3} {dt_str:<22} {m['similarity']:>4.1f}%"
            f"  {m['rsi']:>6.1f}  {m['atr_pips']:>9.1f}  {m['session']:<12}"
        )

    print("=" * 55)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  TradingBotV1 — Historical Pattern Matcher")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 55)

    print_historical_analysis()
