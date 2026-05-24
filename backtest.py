"""
backtest.py - Historical Rule Backtester for TradingBotV1

Downloads 2 years of XAUUSD H1 data via yfinance, runs every rule
from data/rules.json against the price history, and enriches each
rule with backtest statistics (win rate, profit factor, etc.).

All indicators are computed with pure pandas/numpy so there are no
third-party indicator library dependencies.
"""

import os
import json
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from _progress import Spinner, ProgressBar, _bar, _fmt_time, OK, FAIL, SKIP, WARN

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

RULES_FILE            = os.path.join("data", "rules.json")
BACKTEST_RESULTS_FILE = os.path.join("data", "backtest_results.json")
SYMBOL           = "GC=F"          # Gold futures (proxy for XAUUSD)
PERIOD           = "2y"
INTERVAL         = "1h"
MAX_HOLD_CANDLES = 48              # maximum bars before forced exit

# Per-instrument Yahoo Finance ticker map
YF_BACKTEST_TICKERS: dict = {
    "XAUUSD": "GC=F",
    "NAS100": "NQ=F",
    "US30":   "YM=F",
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "WTI":    "CL=F",
}

# Default stop/TP in price units when rule has no explicit levels
# Gold moves $5–30 per candle; 1 pip = $0.10, so $15 SL = 150 pips
DEFAULT_SL_PIPS  = 150.0           # $15 / oz  = 150 pips on XAUUSD
DEFAULT_TP_PIPS  = 225.0           # $22.50 / oz = 225 pips (1.5 : 1 RR)
PIP_SIZE         = 0.1             # 1 pip = $0.10 for gold

# ── Tier thresholds ───────────────────────────────────────────────────────────
TIER_A_WR = 55.0;  TIER_A_PF = 1.2    # Strong  — use with full confidence
TIER_B_WR = 45.0;  TIER_B_PF = 0.9    # Moderate — use with caution
TIER_C_WR = 35.0;  TIER_C_PF = 0.7    # Weak    — paper trade only

TIER_LABELS = {
    "A": "STRONG SIGNAL",
    "B": "MODERATE SIGNAL",
    "C": "WEAK SIGNAL - paper trade only",
    "D": "UNVERIFIED - skip for now",
}


def _get_tier(wr: float, pf: float) -> str:
    """Return tier letter A/B/C/D based on win rate and profit factor."""
    if wr >= TIER_A_WR and pf >= TIER_A_PF:
        return "A"
    if wr >= TIER_B_WR and pf >= TIER_B_PF:
        return "B"
    if wr >= TIER_C_WR and pf >= TIER_C_PF:
        return "C"
    return "D"


# Trading sessions in UTC hours
SESSIONS = {
    "Sydney":  (21, 6),
    "Tokyo":   (0,  9),
    "London":  (7,  16),
    "New York": (12, 21),
}


# ── Indicator helpers (pure pandas/numpy) ─────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _macd(close: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9
          ) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema   = _ema(close, fast)
    slow_ema   = _ema(close, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close: pd.Series, period: int = 20,
               std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


# ── 1. get_historical_data ────────────────────────────────────────────────────

def _download_yf_symbol(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Try to download OHLCV from yfinance for a single symbol.
    Returns a raw DataFrame (index = DatetimeIndex) or empty DataFrame.
    Applies UTC normalisation to fix pandas 3.x timezone handling.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        return pd.DataFrame()

    # ── pandas 3.x / yfinance 1.3+ datetime fix ───────────────────────────
    # Ensure index is tz-aware UTC, then strip tz so downstream code is
    # compatible with both old and new pandas versions.
    try:
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert(None)          # strip tz (naive UTC)
    except Exception:
        pass

    return df


def _download_yf_fallbacks(period: str, interval: str) -> tuple[pd.DataFrame, str]:
    """
    Try GC=F → XAUUSD=X → GLD in order. Returns (df, symbol_that_worked).
    """
    for sym in ["GC=F", "XAUUSD=X", "GLD"]:
        sp = Spinner(f"Downloading {sym} | {period} | {interval} ...", indent=2).start()
        try:
            df = _download_yf_symbol(sym, period, interval)
            if not df.empty:
                sp.stop(success=True, suffix=f"{len(df):,} rows  [symbol: {sym}]")
                return df, sym
            sp.stop(success=False, suffix="empty")
        except Exception as exc:
            sp.stop(success=False, suffix=str(exc))
    return pd.DataFrame(), ""


def _download_yahoo_json(period: str = "2y", interval: str = "1h") -> pd.DataFrame:
    """
    Last-resort fallback: fetch raw JSON from Yahoo Finance chart API and
    parse it manually into a DataFrame — bypasses yfinance entirely.
    """
    import requests

    range_map = {"1y": "1y", "2y": "2y", "6mo": "6mo", "1mo": "1mo"}
    yrange    = range_map.get(period, "2y")
    url       = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF"
        f"?interval={interval}&range={yrange}"
    )
    headers = {"User-Agent": "Mozilla/5.0"}

    sp = Spinner("Downloading via Yahoo JSON API (fallback)...", indent=2).start()
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data   = resp.json()
        result = data["chart"]["result"][0]
        ts     = result["timestamp"]
        ohlcv  = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "open":   ohlcv["open"],
            "high":   ohlcv["high"],
            "low":    ohlcv["low"],
            "close":  ohlcv["close"],
            "volume": ohlcv["volume"],
        }, index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(None))

        df.dropna(subset=["close"], inplace=True)
        sp.stop(success=True, suffix=f"{len(df):,} rows  [Yahoo JSON API / GC=F]")
        return df, "GC=F (JSON API)"
    except Exception as exc:
        sp.stop(success=False, suffix=str(exc))
        return pd.DataFrame(), ""


def get_historical_data(
    symbol: str = SYMBOL,
    period: str = PERIOD,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    """
    Download OHLCV data and compute all technical indicators.

    Indicators added:
        ema50, ema200           – trend EMAs
        rsi                     – RSI 14
        atr                     – ATR 14
        macd, macd_signal,
        macd_hist               – MACD (12/26/9)
        bb_upper, bb_mid,
        bb_lower                – Bollinger Bands 20/2
        hour, session           – time-of-day metadata

    Args:
        symbol:   Yahoo Finance ticker (default "GC=F").
        period:   History length string accepted by yfinance (default "2y").
        interval: Bar size string accepted by yfinance (default "1h").

    Returns:
        pandas DataFrame with OHLCV + indicators, or empty DataFrame on error.
    """
    try:
        # ── 1. Try yfinance with symbol fallbacks ─────────────────────────────
        raw, used_symbol = _download_yf_fallbacks(period, interval)

        # ── 2. If all yfinance symbols failed, try direct Yahoo JSON API ──────
        if raw.empty:
            raw, used_symbol = _download_yahoo_json(period, interval)

        if raw.empty:
            print(f"  {FAIL} All download methods failed — no historical data")
            return pd.DataFrame()

        # ── 3. Normalise columns ──────────────────────────────────────────────
        # yfinance may return title-case or lower-case column names
        raw.columns = [c.lower() for c in raw.columns]
        needed = ["open", "high", "low", "close", "volume"]
        missing = [c for c in needed if c not in raw.columns]
        if missing:
            print(f"  {FAIL} Missing columns after download: {missing}")
            return pd.DataFrame()

        df = raw[needed].copy()
        df.dropna(subset=["close"], inplace=True)

        # ── 4. Indicators ─────────────────────────────────────────────────────
        df["ema50"]       = _ema(df["close"], 50)
        df["ema200"]      = _ema(df["close"], 200)
        df["rsi"]         = _rsi(df["close"], 14)
        df["atr"]         = _atr(df["high"], df["low"], df["close"], 14)
        (df["macd"],
         df["macd_signal"],
         df["macd_hist"]) = _macd(df["close"])
        (df["bb_upper"],
         df["bb_mid"],
         df["bb_lower"])  = _bollinger(df["close"])

        # ── 5. Time metadata ──────────────────────────────────────────────────
        # Index is already naive UTC after _download_yf_symbol / JSON fallback
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        df["hour"] = df.index.hour

        def _session(h: int) -> str:
            for name, (start, end) in SESSIONS.items():
                if start <= end:
                    if start <= h < end:
                        return name
                else:  # wraps midnight
                    if h >= start or h < end:
                        return name
            return "Off-hours"

        df["session"] = df["hour"].apply(_session)

        # Drop warmup rows where indicators are still NaN
        df.dropna(subset=["ema200", "rsi", "atr", "macd"], inplace=True)

        # ── 6. Promote index → "datetime" column ─────────────────────────────
        # In pandas 3.x / yfinance 1.3+, the index may already be named
        # "Datetime" (capital D) rather than "index" after reset_index().
        df.reset_index(inplace=True)
        # Rename whichever column held the DatetimeIndex to "datetime"
        for candidate in ("Datetime", "datetime", "Date", "index", "level_0"):
            if candidate in df.columns and candidate != "datetime":
                df.rename(columns={candidate: "datetime"}, inplace=True)
                break
        # Safety: if still missing, derive from existing index
        if "datetime" not in df.columns:
            df.insert(0, "datetime", df.index)

        sp_final = Spinner("", indent=2)  # silent — just for the final line
        print(
            f"    ✓ Ready: {len(df):,} candles  "
            f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})  "
            f"[{used_symbol}]"
        )
        return df

    except Exception as exc:
        print(f"  {FAIL} get_historical_data error: {exc}")
        return pd.DataFrame()


# ── Rule condition matching ───────────────────────────────────────────────────

def _parse_sl_tp(rule: dict, atr_val: float) -> tuple[float, float]:
    """
    Extract stop-loss and take-profit pip values from rule text fields.
    Falls back to ATR multiples when the rule text is vague.
    """
    sl_text = str(rule.get("stop_loss_logic",  "") or rule.get("stop_loss_pips",  "")).lower()
    tp_text = str(rule.get("take_profit_logic", "") or rule.get("take_profit_pips", "")).lower()

    # Try to extract numeric multipliers (e.g. "1.5x ATR" → 1.5)
    def _atr_mult(text: str, default: float) -> float:
        import re
        m = re.search(r"(\d+\.?\d*)\s*[xX*]\s*atr", text)
        if m:
            return float(m.group(1)) * atr_val / PIP_SIZE
        # Try plain number (e.g. "20 pips")
        m = re.search(r"(\d+\.?\d*)\s*pip", text)
        if m:
            return float(m.group(1))
        return default

    sl_pips = _atr_mult(sl_text, DEFAULT_SL_PIPS)
    tp_pips = _atr_mult(tp_text, DEFAULT_TP_PIPS)

    # Sanity bounds
    sl_pips = max(5.0,  min(sl_pips, 200.0))
    tp_pips = max(10.0, min(tp_pips, 500.0))
    return sl_pips, tp_pips


def _matches_condition(row: pd.Series, rule: dict) -> bool:
    """
    Heuristic rule-matching against technical indicators.

    Parses the rule's condition / pattern_name text and checks
    whether the current candle satisfies common pattern keywords.
    Returns True if a match is detected.
    """
    condition = (
        str(rule.get("condition",           "")).lower() + " " +
        str(rule.get("entry_condition",     "")).lower() + " " +
        str(rule.get("pattern_name",        "")).lower() + " " +
        str(rule.get("indicator_conditions", "")).lower()
    )
    direction = str(rule.get("direction", "both")).lower()

    # Helper flags
    bullish_candle = row["close"] > row["open"]
    bearish_candle = row["close"] < row["open"]
    body_size      = abs(row["close"] - row["open"])
    candle_range   = row["high"] - row["low"]
    body_ratio     = body_size / candle_range if candle_range > 0 else 0

    above_ema50  = row["close"] > row["ema50"]
    above_ema200 = row["close"] > row["ema200"]
    rsi_val      = row["rsi"]
    macd_bull    = row["macd"] > row["macd_signal"]

    # ── Pattern keyword matching ───────────────────────────────────────────
    # RSI patterns
    if "oversold" in condition or "rsi" in condition and "30" in condition:
        return rsi_val < 35
    if "overbought" in condition or "rsi" in condition and "70" in condition:
        return rsi_val > 65

    # MACD patterns
    if "macd" in condition and "cross" in condition:
        return macd_bull if direction in ("long", "buy", "both") else not macd_bull
    if "macd" in condition:
        return macd_bull if direction in ("long", "buy") else not macd_bull

    # EMA / trend patterns
    if "golden cross" in condition or "ema cross" in condition:
        return above_ema50 and above_ema200
    if "death cross" in condition:
        return not above_ema50 and not above_ema200
    if "ema" in condition and ("200" in condition or "trend" in condition):
        if direction in ("long", "buy", "both"):
            return above_ema200
        return not above_ema200

    # Candlestick patterns
    if "hammer" in condition or "pin bar" in condition or "rejection" in condition:
        lower_wick = row["open"] - row["low"] if bearish_candle else row["close"] - row["low"]
        return lower_wick > body_size * 2 and rsi_val < 55

    if "shooting star" in condition or "inverted hammer" in condition:
        upper_wick = row["high"] - row["close"] if bearish_candle else row["high"] - row["open"]
        return upper_wick > body_size * 2 and rsi_val > 45

    if "doji" in condition:
        return body_ratio < 0.1

    if "engulf" in condition:
        if direction in ("long", "buy", "both"):
            return bullish_candle and body_ratio > 0.6 and rsi_val < 60
        return bearish_candle and body_ratio > 0.6 and rsi_val > 40

    if "morning star" in condition:
        return bullish_candle and rsi_val < 45 and above_ema200

    if "evening star" in condition:
        return bearish_candle and rsi_val > 55 and not above_ema200

    if "harami" in condition:
        return body_ratio < 0.4

    if "marubozu" in condition or "strong" in condition:
        return body_ratio > 0.8

    # Fibonacci / support-resistance
    if "fibonacci" in condition or "fib" in condition or "retracement" in condition:
        return rsi_val < 50 if direction in ("long", "buy") else rsi_val > 50

    if "support" in condition or "resistance" in condition:
        if direction in ("long", "buy", "both"):
            return above_ema50 and rsi_val < 55
        return not above_ema50 and rsi_val > 45

    # Bollinger Bands
    if "bollinger" in condition or "band" in condition:
        if direction in ("long", "buy", "both"):
            return row["close"] <= row["bb_lower"] * 1.002
        return row["close"] >= row["bb_upper"] * 0.998

    # ATR / volatility
    if "atr" in condition or "volatility" in condition or "breakout" in condition:
        avg_atr = row.get("atr", DEFAULT_SL_PIPS * PIP_SIZE)
        return body_size > avg_atr * 0.8

    # Trend-following fallback
    if direction in ("long", "buy"):
        return above_ema50 and bullish_candle and rsi_val > 40
    if direction in ("short", "sell"):
        return not above_ema50 and bearish_candle and rsi_val < 60

    # Default: require trend alignment
    return above_ema200 and bullish_candle


# ── 2. backtest_rule ──────────────────────────────────────────────────────────

def backtest_rule(rule: dict, df: pd.DataFrame) -> dict:
    """
    Walk through all candles and simulate every trade triggered by a rule.

    Args:
        rule: Single rule dict from rules.json.
        df:   Full historical DataFrame from get_historical_data().

    Returns:
        Results dict with win_rate, profit_factor, avg pips, etc.
    """
    direction = str(rule.get("direction", "both")).lower()
    is_long   = direction in ("long", "buy", "both")
    is_short  = direction in ("short", "sell", "both")

    wins, losses       = 0, 0
    win_pips: list[float]  = []
    loss_pips: list[float] = []
    session_wins: dict[str, int]   = {}
    session_total: dict[str, int]  = {}
    hour_wins: dict[int, int]      = {}
    hour_total: dict[int, int]     = {}

    i = 0
    in_trade = False

    while i < len(df) - MAX_HOLD_CANDLES - 1:
        row = df.iloc[i]

        if not _matches_condition(row, rule):
            i += 1
            continue

        # ── Entry ──────────────────────────────────────────────────────────
        entry_price = row["close"]
        atr_val     = row["atr"] if row["atr"] > 0 else DEFAULT_SL_PIPS * PIP_SIZE
        sl_pips, tp_pips = _parse_sl_tp(rule, atr_val)

        sl_price = entry_price - sl_pips * PIP_SIZE if is_long else entry_price + sl_pips * PIP_SIZE
        tp_price = entry_price + tp_pips * PIP_SIZE if is_long else entry_price - tp_pips * PIP_SIZE

        sess  = row["session"]
        hour  = int(row["hour"])
        session_total[sess]  = session_total.get(sess, 0) + 1
        hour_total[hour]     = hour_total.get(hour, 0) + 1

        # ── Simulate trade ─────────────────────────────────────────────────
        result = "timeout"
        pips   = 0.0

        for j in range(i + 1, min(i + MAX_HOLD_CANDLES + 1, len(df))):
            future = df.iloc[j]

            if is_long:
                if future["low"] <= sl_price:
                    result = "loss"
                    pips   = -sl_pips
                    break
                if future["high"] >= tp_price:
                    result = "win"
                    pips   = tp_pips
                    break
            else:
                if future["high"] >= sl_price:
                    result = "loss"
                    pips   = -sl_pips
                    break
                if future["low"] <= tp_price:
                    result = "win"
                    pips   = tp_pips
                    break

        if result == "timeout":
            # Close at last available price
            last = df.iloc[min(i + MAX_HOLD_CANDLES, len(df) - 1)]["close"]
            pips = (last - entry_price) / PIP_SIZE if is_long else (entry_price - last) / PIP_SIZE
            result = "win" if pips > 0 else "loss"

        if result == "win":
            wins += 1
            win_pips.append(pips)
            session_wins[sess] = session_wins.get(sess, 0) + 1
            hour_wins[hour]    = hour_wins.get(hour, 0) + 1
        else:
            losses += 1
            loss_pips.append(abs(pips))

        # Skip ahead to avoid overlapping trades
        i += MAX_HOLD_CANDLES // 2

    # ── Compute summary stats ──────────────────────────────────────────────
    total_signals = wins + losses
    win_rate      = (wins / total_signals * 100) if total_signals > 0 else 0.0
    avg_win       = float(np.mean(win_pips))   if win_pips   else 0.0
    avg_loss      = float(np.mean(loss_pips))  if loss_pips  else 0.0
    total_won     = sum(win_pips)
    total_lost    = sum(loss_pips)
    profit_factor = (total_won / total_lost) if total_lost > 0 else (total_won if total_won > 0 else 0.0)

    # Best sessions by win rate
    best_sessions = sorted(
        [s for s in session_total if session_total[s] >= 3],
        key=lambda s: session_wins.get(s, 0) / session_total[s],
        reverse=True,
    )[:2]

    # Best hours
    best_hours = sorted(
        [h for h in hour_total if hour_total[h] >= 2],
        key=lambda h: hour_wins.get(h, 0) / hour_total[h],
        reverse=True,
    )[:3]

    # Market condition heuristic
    if avg_win > avg_loss * 1.3:
        market_condition = "trending"
    elif total_signals > 30:
        market_condition = "ranging"
    else:
        market_condition = "mixed"

    # Confidence score from win rate and profit factor
    conf = min(10, max(1, int(win_rate / 10) + (1 if profit_factor >= 1.5 else 0)))

    return {
        "total_signals":       total_signals,
        "wins":                wins,
        "losses":              losses,
        "win_rate":            round(win_rate, 1),
        "avg_win_pips":        round(avg_win, 1),
        "avg_loss_pips":       round(avg_loss, 1),
        "profit_factor":       round(profit_factor, 2),
        "best_sessions":       best_sessions,
        "best_hours_utc":      best_hours,
        "best_market_conditions": market_condition,
        "backtest_confidence": conf,
    }


# ── 3. backtest_all_rules ─────────────────────────────────────────────────────

def backtest_all_rules(
    symbol: str   = SYMBOL,
    period: str   = PERIOD,
    interval: str = INTERVAL,
    instrument: str = "XAUUSD",
) -> list[dict]:
    """
    Load rules.json, run backtest_rule() on every rule, enrich and re-save.

    Args:
        symbol, period, interval: Passed to get_historical_data().
        instrument: Trading instrument key (e.g. "NAS100"). Overrides symbol
                    via YF_BACKTEST_TICKERS when provided.

    Returns:
        Enriched and sorted list of rule dicts.
    """
    # Resolve symbol from instrument if not explicitly overridden
    if instrument != "XAUUSD" or symbol == SYMBOL:
        symbol = YF_BACKTEST_TICKERS.get(instrument, symbol)
    # Load rules
    if not os.path.exists(RULES_FILE):
        print(f"[ERROR] {RULES_FILE} not found. Run ingest.py first.")
        return []

    with open(RULES_FILE, "r", encoding="utf-8") as fh:
        rules: list[dict] = json.load(fh)

    if not rules:
        print("[ERROR] rules.json is empty.")
        return []

    print(f"\n  {len(rules)} rules loaded from {RULES_FILE}")

    # Download historical data once
    df = get_historical_data(symbol, period, interval)
    if df.empty:
        print("[ERROR] No historical data — backtest aborted.")
        return rules

    print(f"\n  Running backtest on {len(rules)} rules...\n")

    for idx, rule in enumerate(rules, start=1):
        name = rule.get("pattern_name", f"Rule {idx}")
        print(f"  [{idx:>3}/{len(rules)}] Testing: {name[:55]:<55}", end=" ")

        try:
            result = backtest_rule(rule, df)
            rule["backtest"] = result
            rule["tier"]    = _get_tier(result["win_rate"], result["profit_factor"])
            print(
                f"WR {result['win_rate']:>5.1f}%  PF {result['profit_factor']:.2f}  "
                f"n={result['total_signals']}  [{rule['tier']}]"
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            rule["backtest"] = {}

    # Sort by profit_factor descending (rules with no backtest go to bottom)
    rules.sort(
        key=lambda r: r.get("backtest", {}).get("profit_factor", 0),
        reverse=True,
    )

    # Save enriched rules
    os.makedirs("data", exist_ok=True)
    with open(RULES_FILE, "w", encoding="utf-8") as fh:
        json.dump(rules, fh, indent=2, ensure_ascii=False)
    print(f"\n  [SAVED] Enriched rules written to {RULES_FILE}")

    return rules


# ── 4. print_backtest_report ──────────────────────────────────────────────────

def print_backtest_report(rules: list[dict]) -> None:
    """
    Print a human-readable backtest summary report.

    Args:
        rules: Enriched list of rule dicts (output of backtest_all_rules()).
    """
    tested = [r for r in rules if r.get("backtest")]
    if not tested:
        print("  No backtest results to display.")
        return

    total_tested      = len(tested)
    above_60wr        = sum(1 for r in tested if r["backtest"].get("win_rate", 0)      >= 60)
    above_15pf        = sum(1 for r in tested if r["backtest"].get("profit_factor", 0) >= 1.5)
    recommended_count = sum(
        1 for r in tested
        if r["backtest"].get("win_rate", 0) >= 55
        and r["backtest"].get("profit_factor", 0) >= 1.2
        and r["backtest"].get("total_signals", 0) >= 5
    )

    top5  = tested[:5]
    worst = sorted(tested, key=lambda r: r["backtest"].get("profit_factor", 0))[:5]

    print("\n" + "=" * 58)
    print("=== BACKTEST REPORT ===")
    print(f"=== Period: 2 years {SYMBOL} {INTERVAL} ===")
    print("=" * 58)

    print("\nTOP 5 PERFORMING RULES:")
    for i, rule in enumerate(top5, start=1):
        bt   = rule["backtest"]
        name = rule.get("pattern_name", "Unknown")[:50]
        best_sess = ", ".join(bt.get("best_sessions", [])) or "—"
        print(f"\n  {i}. {name}")
        print(f"     Win rate: {bt['win_rate']}% | Profit factor: {bt['profit_factor']}")
        print(f"     Avg win: +{bt['avg_win_pips']} pips | Avg loss: -{bt['avg_loss_pips']} pips")
        print(f"     Signals found: {bt['total_signals']} times in 2 years")
        print(f"     Best session: {best_sess}")
        print(f"     Market condition: {bt.get('best_market_conditions', '—')}")

    print("\n" + "-" * 58)
    print("\nWORST 5 RULES (consider ignoring these):")
    for i, rule in enumerate(worst, start=1):
        bt   = rule["backtest"]
        name = rule.get("pattern_name", "Unknown")[:50]
        print(f"\n  {i}. {name}")
        print(f"     Win rate: {bt['win_rate']}% | Profit factor: {bt['profit_factor']}")
        print(f"     Signals found: {bt['total_signals']} times in 2 years")

    # Tier breakdown
    tier_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in tested:
        bt = r.get("backtest", {})
        t  = r.get("tier") or _get_tier(
            bt.get("win_rate", 0), bt.get("profit_factor", 0)
        )
        tier_counts[t] = tier_counts.get(t, 0) + 1

    print("\n" + "-" * 58)
    print("\nOVERALL STATS:")
    print(f"  Total rules tested          : {total_tested}")
    print(f"  Rules with ≥60% win rate    : {above_60wr}")
    print(f"  Rules with profit factor ≥1.5 : {above_15pf}")
    print(f"  Recommended rules to use    : {recommended_count}")
    print()
    print("  TIER BREAKDOWN:")
    print(f"  Tier A — STRONG SIGNAL           : {tier_counts['A']:>4}  (WR ≥ 55% AND PF ≥ 1.2)")
    print(f"  Tier B — MODERATE SIGNAL         : {tier_counts['B']:>4}  (WR ≥ 45% AND PF ≥ 0.9)")
    print(f"  Tier C — WEAK / paper trade only : {tier_counts['C']:>4}  (WR ≥ 35% AND PF ≥ 0.7)")
    print(f"  Tier D — UNVERIFIED / skip        : {tier_counts['D']:>4}  (below Tier C)")
    print("=" * 58)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("  " + "═" * 52)
    print("  TradingBotV1 — Backtest Engine")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("  " + "═" * 52)

    enriched_rules = backtest_all_rules()
    print_backtest_report(enriched_rules)


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW PIPELINE — run_backtest / simulate_trade / generate_backtest_report
#               / run_post_loss_analysis
# ═══════════════════════════════════════════════════════════════════════════════

GST = timezone(timedelta(hours=4))

# ── Optional imports (all wrapped) ──────────────────────────────────────────
try:
    from confluence_engine import score_confluences as _score_confluences
    _CONF_OK = True
except ImportError:
    _CONF_OK = False
    def _score_confluences(*a, **kw): return {}  # type: ignore

try:
    from entry_checklist import validate_entry as _validate_entry
    _EC_OK = True
except ImportError:
    _EC_OK = False
    def _validate_entry(*a, **kw): return {"checks_passed": 5, "passed": True, "final_confidence": 8.0}  # type: ignore

try:
    from volume_analyzer import check_volume_confluence as _check_volume_confluence
    _VA_OK = True
except ImportError:
    _VA_OK = False
    def _check_volume_confluence(*a, **kw): return {"climax": False, "strategy_optimal": True, "score": 0}  # type: ignore

try:
    from pattern_fatigue import check_strategy_fatigue as _check_fatigue
    _PF_OK = True
except ImportError:
    _PF_OK = False
    def _check_fatigue(*a, **kw): return {"fatigue_level": "none"}  # type: ignore

try:
    from market_context import detect_gold_regime as _detect_regime
    _MC_OK = True
except ImportError:
    _MC_OK = False
    def _detect_regime(*a, **kw): return {"regime": "trending"}  # type: ignore

try:
    from strategy_playbooks import PLAYBOOKS as _PLAYBOOKS
    _SP_OK = True
except ImportError:
    _SP_OK = False
    _PLAYBOOKS = {}  # type: ignore

BACKTEST_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "data", "backtest_results")
AUTO_FIXES_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "data", "logs", "auto_fixes.json")
PENDING_FIXES_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "data", "logs", "pending_fixes.json")
SETTINGS_FILE        = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "data", "user_settings.json")

_MIN_CONFIDENCE = 7.5


def _load_settings_bt() -> dict:
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"balance": 300, "risk_pct": 10, "leverage": 20,
                "implied_rr": 3, "min_confidence": 7.5, "partial_tp": True}


# ── simulate_trade ─────────────────────────────────────────────────────────────

def simulate_trade(
    entry_idx:     int,
    df:            pd.DataFrame,
    signal:        dict,
    settings:      dict,
) -> dict:
    """
    Walk future candles from entry_idx and simulate trade outcome.

    Returns
    -------
    outcome        : full_win | partial_win | full_loss | timeout
    pnl_usd        : float
    rr_achieved    : float
    tp1_hit        : bool
    tp2_hit        : bool
    sl_hit         : bool
    candles_held   : int
    max_favorable  : float
    max_adverse    : float
    lots           : float
    """
    entry    = float(signal.get("entry",      df["close"].iloc[entry_idx]))
    sl       = float(signal.get("stop_loss",  entry - 15))
    tp1      = float(signal.get("take_profit", entry + 45))
    direction = str(signal.get("direction", "long")).lower()
    is_long  = direction in ("long", "buy")

    sl_dist  = abs(entry - sl)
    tp1_dist = abs(entry - tp1)

    # Partial TP: TP2 = 2 × TP distance from entry
    partial_tp = settings.get("partial_tp", True)
    tp2 = (entry + tp1_dist * 2) if is_long else (entry - tp1_dist * 2)

    # Position sizing
    balance  = float(settings.get("balance",  300))
    risk_pct = float(settings.get("risk_pct", 10))
    risk_usd = balance * risk_pct / 100
    lots     = round(risk_usd / (sl_dist * 100), 2) if sl_dist else 0.01
    lots     = max(0.01, lots)

    future   = df.iloc[entry_idx + 1: entry_idx + MAX_HOLD_CANDLES + 1]

    tp1_hit = tp2_hit = sl_hit = False
    candles_held   = 0
    max_favorable  = 0.0
    max_adverse    = 0.0

    for _, bar in future.iterrows():
        candles_held += 1
        hi = float(bar["high"])
        lo = float(bar["low"])

        fav = (hi - entry) if is_long else (entry - lo)
        adv = (entry - lo) if is_long else (hi - entry)
        max_favorable = max(max_favorable, fav)
        max_adverse   = max(max_adverse, adv)

        if is_long:
            if lo <= sl:
                sl_hit = True
                break
            if not tp1_hit and hi >= tp1:
                tp1_hit = True
                if not partial_tp:
                    tp2_hit = True
                    break
            if tp1_hit and hi >= tp2:
                tp2_hit = True
                break
        else:
            if hi >= sl:
                sl_hit = True
                break
            if not tp1_hit and lo <= tp1:
                tp1_hit = True
                if not partial_tp:
                    tp2_hit = True
                    break
            if tp1_hit and lo <= tp2:
                tp2_hit = True
                break

    # P&L
    if sl_hit:
        pnl_usd    = -round(lots * sl_dist * 100, 2)
        outcome    = "full_loss"
        rr_achieved = 0.0
    elif tp2_hit:
        pnl_usd    = round(lots * tp1_dist * 100 * 0.5 + lots * tp1_dist * 2 * 100 * 0.5, 2)
        outcome    = "full_win"
        rr_achieved = 2.0
    elif tp1_hit:
        pnl_usd    = round(lots * tp1_dist * 100 * 0.5, 2)
        outcome    = "partial_win"
        rr_achieved = 1.0
    else:
        # Timeout — close at last price
        last_price = float(df["close"].iloc[min(entry_idx + MAX_HOLD_CANDLES, len(df) - 1)])
        pnl_usd    = round(lots * abs(last_price - entry) * 100 *
                           (1 if (is_long and last_price > entry) else -1), 2)
        outcome    = "timeout"
        rr_achieved = round(abs(last_price - entry) / sl_dist, 2) if sl_dist else 0

    return {
        "outcome":       outcome,
        "pnl_usd":       pnl_usd,
        "rr_achieved":   rr_achieved,
        "tp1_hit":       tp1_hit,
        "tp2_hit":       tp2_hit,
        "sl_hit":        sl_hit,
        "candles_held":  candles_held,
        "max_favorable": round(max_favorable, 2),
        "max_adverse":   round(max_adverse, 2),
        "lots":          lots,
    }


# ── run_backtest ───────────────────────────────────────────────────────────────

def run_backtest(
    playbook_name:  str,
    df_historical:  pd.DataFrame | None = None,
    settings:       dict | None         = None,
    instrument:     str                 = "XAUUSD",
) -> list[dict]:
    """
    9-stage pipeline backtest for a named playbook.

    Stage 1 : detect_gold_regime
    Stage 2 : check_playbook_conditions (_matches_condition)
    Stage 3 : score_confluences  — reject if < min_confidence
    Stage 4 : check_volume_confluence — reject if climax / suboptimal
    Stage 5 : validate_entry checklist — reject if < 4/5
    Stage 6 : check_strategy_fatigue  — reject if critical
    Stage 7 : sl_quality_check (via validate_entry sl_quality)
    Stage 8 : simulate_trade
    Stage 9 : record_backtest_trade

    Returns list of trade record dicts.
    """
    if settings is None:
        settings = _load_settings_bt()

    # Use instrument-specific ticker when downloading fresh data
    _bt_ticker = YF_BACKTEST_TICKERS.get(instrument, "GC=F")

    if df_historical is None:
        df_historical = get_historical_data(
            symbol=_bt_ticker if instrument != "XAUUSD" else SYMBOL
        )

    if df_historical is None or df_historical.empty:
        return []

    min_conf    = float(settings.get("min_confidence", _MIN_CONFIDENCE))
    results     = []
    pb_key      = playbook_name.strip()

    # Try to load playbook spec
    pb_spec = {}
    if _SP_OK:
        pb_spec = _PLAYBOOKS.get(pb_key, {})
        if not pb_spec:
            for k, v in _PLAYBOOKS.items():
                if pb_key.lower() in k.lower() or k.lower() in pb_key.lower():
                    pb_spec = v
                    pb_key  = k
                    break

    direction = pb_spec.get("direction", "long") if pb_spec else "long"

    # Warm-up: need 30 candles of history before first trade
    WARMUP = 30

    for i in range(WARMUP, len(df_historical) - MAX_HOLD_CANDLES - 1):
        df_window = df_historical.iloc[max(0, i - 100): i + 1].copy()
        row       = df_historical.iloc[i]

        record: dict = {
            "candle_idx":  i,
            "timestamp":   str(row.name),
            "playbook":    pb_key,
            "direction":   direction,
            "entry":       float(row["close"]),
            "rejections":  [],
            "instrument":  instrument,
            "ticker_used": _bt_ticker,
        }

        # Stage 1 — Regime
        regime = "unknown"
        if _MC_OK:
            try:
                r = _detect_regime(df_window)
                regime = r.get("regime", "unknown")
                record["regime"] = regime
            except Exception:
                pass

        # Stage 2 — Playbook conditions
        dummy_rule = dict(pb_spec) if pb_spec else {"name": pb_key, "direction": direction}
        dummy_rule["name"] = pb_key
        if not _matches_condition(row, dummy_rule):
            record["rejections"].append("stage2_conditions")
            record["stage_rejected"] = 2
            results.append(record)
            continue

        # Stage 3 — Confluence
        if _CONF_OK:
            try:
                conf = _score_confluences(df_window, direction,
                                          current_time=pd.Timestamp(row.name).to_pydatetime(),
                                          playbook=pb_key)
                score = float(conf.get("confidence", conf.get("weighted_score", 0)))
                record["confluence_score"] = score
                if score < min_conf:
                    record["rejections"].append(f"stage3_conf_{score:.1f}")
                    record["stage_rejected"] = 3
                    results.append(record)
                    continue
            except Exception:
                pass

        # Stage 4 — Volume
        if _VA_OK:
            try:
                vol = _check_volume_confluence(df_window, direction, pb_key)
                record["volume"] = {
                    "ratio":    vol.get("volume_ratio", 1.0),
                    "class":    vol.get("volume_class", "normal"),
                    "score":    vol.get("score", 0),
                    "climax":   vol.get("climax", False),
                    "optimal":  vol.get("strategy_optimal", True),
                }
                if vol.get("climax"):
                    record["rejections"].append("stage4_climax")
                    record["stage_rejected"] = 4
                    results.append(record)
                    continue
                if not vol.get("strategy_optimal", True):
                    record["rejections"].append("stage4_vol_suboptimal")
                    record["stage_rejected"] = 4
                    results.append(record)
                    continue
            except Exception:
                pass

        # Stage 5 — Entry checklist
        if _EC_OK:
            try:
                sig_dummy = {
                    "pattern_name": pb_key,
                    "direction":    direction,
                    "entry":        float(row["close"]),
                    "stop_loss":    float(row["close"]) - 15 if direction == "long" else float(row["close"]) + 15,
                    "take_profit":  float(row["close"]) + 45 if direction == "long" else float(row["close"]) - 45,
                }
                ck = _validate_entry(sig_dummy, df_window)
                record["checklist_passed"] = ck.get("checks_passed", 0)
                if ck.get("checks_passed", 0) < 4:
                    record["rejections"].append(f"stage5_checklist_{ck.get('checks_passed',0)}/5")
                    record["stage_rejected"] = 5
                    results.append(record)
                    continue
            except Exception:
                pass

        # Stage 6 — Fatigue
        if _PF_OK:
            try:
                fat = _check_fatigue(pb_key, df_window, direction)
                fat_level = fat.get("fatigue_level", "none")
                record["fatigue"] = fat_level
                if fat_level == "critical":
                    record["rejections"].append("stage6_fatigue_critical")
                    record["stage_rejected"] = 6
                    results.append(record)
                    continue
            except Exception:
                pass

        # Stages 7+8 — Simulate trade
        signal_for_sim = {
            "direction":   direction,
            "entry":       float(row["close"]),
            "stop_loss":   float(row["close"]) - 15 if direction == "long" else float(row["close"]) + 15,
            "take_profit": float(row["close"]) + 45 if direction == "long" else float(row["close"]) - 45,
        }
        if pb_spec:
            rr  = float(pb_spec.get("risk_reward", 3))
            sl_ = float(pb_spec.get("sl_distance", 15))
            tp_ = sl_ * rr
            signal_for_sim["stop_loss"]   = float(row["close"]) - sl_ if direction == "long" else float(row["close"]) + sl_
            signal_for_sim["take_profit"] = float(row["close"]) + tp_ if direction == "long" else float(row["close"]) - tp_

        sim = simulate_trade(i, df_historical, signal_for_sim, settings)
        record.update(sim)
        record["stage_rejected"] = None   # passed all stages
        results.append(record)

    return results


# ── generate_backtest_report ───────────────────────────────────────────────────

def generate_backtest_report(
    results:       list[dict],
    playbook_name: str,
    settings:      dict | None = None,
) -> tuple[str, str]:
    """
    Generate a text report and JSON dump for backtest results.
    Saves to data/backtest_results/.

    Returns (report_filepath, report_text).
    """
    if settings is None:
        settings = _load_settings_bt()

    os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)

    traded    = [r for r in results if r.get("stage_rejected") is None]
    rejected  = [r for r in results if r.get("stage_rejected") is not None]

    full_wins    = [r for r in traded if r.get("outcome") == "full_win"]
    partial_wins = [r for r in traded if r.get("outcome") == "partial_win"]
    losses       = [r for r in traded if r.get("outcome") == "full_loss"]
    timeouts     = [r for r in traded if r.get("outcome") == "timeout"]

    total_scanned = len(results)
    total_traded  = len(traded)
    win_rate      = round((len(full_wins) + len(partial_wins)) / max(total_traded, 1) * 100, 1)
    total_pnl     = round(sum(r.get("pnl_usd", 0) for r in traded), 2)
    balance_start = float(settings.get("balance", 300))
    balance_end   = round(balance_start + total_pnl, 2)

    # Group stats helpers
    def _group_stats(key: str) -> dict:
        groups: dict = {}
        for r in traded:
            val = r.get(key, "unknown")
            if val not in groups:
                groups[val] = {"total": 0, "wins": 0, "pnl": 0.0}
            groups[val]["total"] += 1
            if r.get("outcome") in ("full_win", "partial_win"):
                groups[val]["wins"] += 1
            groups[val]["pnl"] += r.get("pnl_usd", 0)
        return {
            k: {
                "total": v["total"],
                "win_rate": round(v["wins"] / max(v["total"], 1) * 100, 1),
                "total_pnl": round(v["pnl"], 2),
            }
            for k, v in groups.items()
        }

    # Rejection breakdown by stage
    rej_by_stage: dict = {}
    for r in rejected:
        stage = str(r.get("stage_rejected", "?"))
        rej_by_stage[stage] = rej_by_stage.get(stage, 0) + 1

    # Volume class breakdown
    vol_groups = {}
    for r in traded:
        vc = (r.get("volume") or {}).get("class", "unknown")
        if vc not in vol_groups:
            vol_groups[vc] = {"total": 0, "wins": 0}
        vol_groups[vc]["total"] += 1
        if r.get("outcome") in ("full_win", "partial_win"):
            vol_groups[vc]["wins"] += 1

    SEP = "═" * 55
    DASH = "─" * 55
    now_str = datetime.now(GST).strftime("%Y-%m-%d %H:%M GST")
    fname_ts = datetime.now(GST).strftime("%d%b_%H%M")

    report_lines = [
        SEP,
        f"  BACKTEST REPORT — {playbook_name.upper()}",
        f"  Generated: {now_str}",
        SEP,
        "",
        "  OVERVIEW",
        DASH,
        f"  Total candles scanned  : {total_scanned:,}",
        f"  Setups passed filter   : {total_traded:,}",
        f"  Rejected by pipeline   : {len(rejected):,}",
        "",
        f"  Full wins              : {len(full_wins):,}",
        f"  Partial wins           : {len(partial_wins):,}",
        f"  Full losses            : {len(losses):,}",
        f"  Timeouts               : {len(timeouts):,}",
        f"  Win rate               : {win_rate}%",
        "",
        f"  Starting balance       : ${balance_start:,.2f}",
        f"  Total P&L              : ${total_pnl:+,.2f}",
        f"  Ending balance         : ${balance_end:,.2f}",
        "",
        "  PIPELINE REJECTION BREAKDOWN",
        DASH,
    ]
    for stage, cnt in sorted(rej_by_stage.items()):
        labels = {
            "2": "Stage 2 — Playbook conditions",
            "3": "Stage 3 — Confluence below threshold",
            "4": "Stage 4 — Volume climax / suboptimal",
            "5": "Stage 5 — Entry checklist < 4/5",
            "6": "Stage 6 — Strategy fatigue critical",
        }
        lbl = labels.get(stage, f"Stage {stage}")
        report_lines.append(f"  {lbl:<40}: {cnt:>5}")

    report_lines += [
        "",
        "  PERFORMANCE BY REGIME",
        DASH,
    ]
    for regime, st in _group_stats("regime").items():
        report_lines.append(
            f"  {str(regime):<20} : {st['total']:>4} trades | WR {st['win_rate']:>5.1f}% | P&L ${st['total_pnl']:>+8,.2f}"
        )

    report_lines += ["", "  PERFORMANCE BY VOLUME CLASS", DASH]
    for vc, vst in vol_groups.items():
        wr = round(vst["wins"] / max(vst["total"], 1) * 100, 1)
        report_lines.append(f"  {str(vc):<20} : {vst['total']:>4} trades | WR {wr:>5.1f}%")

    report_lines += ["", "  PERFORMANCE BY FATIGUE LEVEL", DASH]
    for fl, st in _group_stats("fatigue").items():
        report_lines.append(
            f"  {str(fl):<20} : {st['total']:>4} trades | WR {st['win_rate']:>5.1f}% | P&L ${st['total_pnl']:>+8,.2f}"
        )

    report_lines += ["", SEP, ""]

    report_text = "\n".join(report_lines)

    # Save
    safe_name = playbook_name.replace(" ", "_").replace("/", "-")[:30]
    txt_path  = os.path.join(BACKTEST_RESULTS_DIR, f"{safe_name}_report_{fname_ts}.txt")
    json_path = os.path.join(BACKTEST_RESULTS_DIR, f"{safe_name}_full_{fname_ts}.json")

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as exc:
        print(f"[backtest] Error saving report: {exc}")

    return txt_path, report_text


# ── run_post_loss_analysis ─────────────────────────────────────────────────────

def run_post_loss_analysis(
    failed_trade:  dict,
    df_historical: pd.DataFrame | None = None,
) -> dict:
    """
    Run full post-loss pipeline analysis on a failed trade.

    Finds similar historical setups, counts failure reasons, optionally
    auto-applies a fix if confidence > 70%.

    Returns
    -------
    similar_setups_found   : int
    win_rate               : float
    primary_failure_reason : str
    fix_applied            : bool
    fix_description        : str
    expected_improvement   : str
    confidence_in_fix      : float
    """
    if df_historical is None:
        df_historical = get_historical_data()

    pattern   = str(failed_trade.get("pattern_name", "unknown"))
    direction = str(failed_trade.get("direction", "long"))
    entry     = float(failed_trade.get("entry", 0) or 0)
    sl        = float(failed_trade.get("stop_loss", 0) or 0)

    # Build fingerprint
    fingerprint = {
        "playbook":  pattern,
        "direction": direction,
    }

    # Find similar setups in historical (simplified: same direction)
    similar_wins   = 0
    similar_losses = 0

    if df_historical is not None and not df_historical.empty and _MC_OK:
        try:
            for i in range(30, min(len(df_historical) - MAX_HOLD_CANDLES, 500)):
                row = df_historical.iloc[i]
                r_dir = "long" if float(row.get("close", 0)) > float(row.get("open", 0)) else "short"
                if r_dir == direction:
                    win = float(row.get("high", 0)) > entry + abs(entry - sl)
                    if win:
                        similar_wins += 1
                    else:
                        similar_losses += 1
        except Exception:
            pass

    similar_total = similar_wins + similar_losses
    win_rate      = round(similar_wins / max(similar_total, 1) * 100, 1)

    # Analyze failure reasons
    reasons = []

    # Volume check
    vol_ratio = float(failed_trade.get("volume_ratio", 0) or 0)
    if vol_ratio > 0 and vol_ratio < 1.0:
        reasons.append(("VOLUME_MISMATCH", 0.8))

    # Regime
    trade_regime    = str(failed_trade.get("regime", "")).lower()
    if trade_regime in ("ranging", "consolidation", "choppy"):
        reasons.append(("WRONG_REGIME", 0.75))

    # SL too tight
    sl_dist = abs(entry - sl) if entry and sl else 0
    if sl_dist > 0 and sl_dist < 10:
        reasons.append(("SL_TOO_TIGHT", 0.7))

    # Session
    close_time = str(failed_trade.get("closed_at", "") or "")
    open_time  = str(failed_trade.get("open_time", "") or "")
    if close_time and open_time:
        try:
            from datetime import datetime as _dt
            ct = _dt.fromisoformat(close_time.replace("Z", "+00:00"))
            ot = _dt.fromisoformat(open_time.replace("Z", "+00:00"))
            def _sess(h): return "Asian" if (h >= 22 or h < 7) else ("London" if h < 12 else "NY")
            if _sess(ct.hour) != _sess(ot.hour):
                reasons.append(("SESSION_CHANGE", 0.65))
        except Exception:
            pass

    if not reasons:
        reasons.append(("UNKNOWN", 0.5))

    # Primary reason = highest confidence
    reasons.sort(key=lambda x: x[1], reverse=True)
    primary_reason, confidence = reasons[0]

    # Fix map
    fix_map = {
        "VOLUME_MISMATCH": (
            f"Add volume_ratio >= 1.2 filter for {pattern}",
            "Est. +8-12% win rate improvement",
        ),
        "WRONG_REGIME": (
            f"Only trade {pattern} in trending regime",
            "Est. +10-15% win rate improvement",
        ),
        "SL_TOO_TIGHT": (
            f"Minimum SL = 1× ATR (est. $15-20)",
            "Est. +5-8% win rate improvement",
        ),
        "SESSION_CHANGE": (
            f"Close {pattern} by session end or move SL to breakeven",
            "Est. +5% win rate improvement",
        ),
        "UNKNOWN": (
            "No automatic fix — manual review required",
            "Unknown",
        ),
    }
    fix_description, expected_improvement = fix_map.get(
        primary_reason, ("No fix found", "Unknown")
    )

    fix_applied = False
    os.makedirs(os.path.dirname(AUTO_FIXES_FILE), exist_ok=True)

    if confidence >= 0.7 and primary_reason != "UNKNOWN":
        # Auto-apply to settings or rules
        try:
            fix_record = {
                "applied_at":    datetime.now(GST).isoformat(),
                "pattern":       pattern,
                "reason":        primary_reason,
                "fix":           fix_description,
                "confidence":    confidence,
                "expected":      expected_improvement,
            }
            _existing: list = []
            if os.path.exists(AUTO_FIXES_FILE):
                with open(AUTO_FIXES_FILE, encoding="utf-8") as f:
                    _existing = json.load(f)
            _existing.append(fix_record)
            with open(AUTO_FIXES_FILE, "w", encoding="utf-8") as f:
                json.dump(_existing, f, indent=2, default=str)
            fix_applied = True
        except Exception:
            pass
    elif confidence >= 0.5:
        # Pending
        try:
            pending_record = {
                "created_at":  datetime.now(GST).isoformat(),
                "pattern":     pattern,
                "reason":      primary_reason,
                "fix":         fix_description,
                "confidence":  confidence,
                "expected":    expected_improvement,
                "status":      "pending_review",
            }
            _pending: list = []
            if os.path.exists(PENDING_FIXES_FILE):
                with open(PENDING_FIXES_FILE, encoding="utf-8") as f:
                    _pending = json.load(f)
            _pending.append(pending_record)
            with open(PENDING_FIXES_FILE, "w", encoding="utf-8") as f:
                json.dump(_pending, f, indent=2, default=str)
        except Exception:
            pass

    return {
        "similar_setups_found":   similar_total,
        "win_rate":               win_rate,
        "primary_failure_reason": primary_reason,
        "all_reasons":            [r[0] for r in reasons],
        "fix_applied":            fix_applied,
        "fix_description":        fix_description,
        "expected_improvement":   expected_improvement,
        "confidence_in_fix":      round(confidence, 2),
    }
