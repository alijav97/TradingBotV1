"""
dxy_correlation.py
──────────────────
DXY (US Dollar Index) correlation filter for Gold trading.

DXY and Gold have a strong inverse correlation:
  • Rising DXY → Dollar strengthening → Gold tends to fall
  • Falling DXY → Dollar weakening    → Gold tends to rise

Class
-----
  DXYCorrelation
    .get_dxy_data()                          — fetch DXY OHLCV + indicators
    .dxy_gold_alignment(dxy_dir, gold_dir)   — correlation alignment check
    .dxy_momentum(dxy_df)                    — RSI/EMA momentum reading

Standalone helpers
------------------
  get_dxy_context()                          — one-call summary dict
  print_dxy_report(gold_direction, dxy_ctx)  — formatted console output
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# ── optional yfinance ─────────────────────────────────────────────────────────
try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

# ── DXY component weights (official ICE basket) ───────────────────────────────
_DXY_COMPONENTS: dict[str, dict] = {
    "EURUSD=X":  {"weight": 0.576, "inverse": True},   # EUR (inverse: 1/EUR)
    "USDJPY=X":  {"weight": 0.136, "inverse": False},  # JPY
    "GBPUSD=X":  {"weight": 0.119, "inverse": True},   # GBP (inverse)
    "USDCAD=X":  {"weight": 0.091, "inverse": False},  # CAD
    "USDSEK=X":  {"weight": 0.042, "inverse": False},  # SEK
    "USDCHF=X":  {"weight": 0.036, "inverse": False},  # CHF
}

_DIRECT_TICKER = "DX-Y.NYB"  # CME/ICE direct DXY futures


# ══════════════════════════════════════════════════════════════════════════════
#  DXYCorrelation
# ══════════════════════════════════════════════════════════════════════════════

class DXYCorrelation:
    """Fetch and analyse the US Dollar Index (DXY) relative to Gold."""

    # ------------------------------------------------------------------
    # 1. get_dxy_data
    # ------------------------------------------------------------------
    def get_dxy_data(
        self,
        period: str = "60d",
        interval: str = "1d",
    ) -> dict[str, Any]:
        """
        Fetch DXY OHLCV data and enrich it with EMA20, RSI14, ATR14.

        Strategy
        --------
        1. Try the direct DX-Y.NYB ticker (ICE futures).
        2. Fall back to reconstructing DXY from currency pair basket.
        3. If only EURUSD is available, approximate direction from inverse.

        Returns
        -------
        dict with keys:
          dxy_df        : pd.DataFrame | None
          dxy_close     : float | None   (latest close)
          ema20         : float | None
          rsi           : float | None
          source        : "direct" | "reconstructed" | "eurusd_approx" | "unavailable"
          available     : bool
          error         : str | None
        """
        if not _YF_OK:
            return self._unavailable("yfinance not installed")

        # ── Attempt 1: direct DXY ─────────────────────────────────────────────
        df = self._fetch_ticker(_DIRECT_TICKER, period=period, interval=interval)
        if df is not None and len(df) >= 20:
            df = self._enrich(df)
            return self._build_result(df, source="direct")

        # ── Attempt 2: reconstruct from basket ────────────────────────────────
        df = self._reconstruct_dxy(period=period, interval=interval)
        if df is not None and len(df) >= 20:
            df = self._enrich(df)
            return self._build_result(df, source="reconstructed")

        # ── Attempt 3: EURUSD approximation (57.6% of DXY) ──────────────────
        eur_raw = self._fetch_ticker("EURUSD=X", period=period, interval=interval)
        if eur_raw is not None and len(eur_raw) >= 20:
            # DXY ≈ C × (1 / EURUSD)^0.576  — simplified as direction proxy
            df = pd.DataFrame(index=eur_raw.index)
            df["close"] = 100.0 / eur_raw["close"]  # inverted proxy
            df["high"]  = 100.0 / eur_raw["low"]
            df["low"]   = 100.0 / eur_raw["high"]
            df["open"]  = 100.0 / eur_raw["open"]
            df["volume"] = eur_raw.get("volume", 0)
            df = self._enrich(df)
            return self._build_result(df, source="eurusd_approx")

        return self._unavailable("all DXY data sources failed")

    # ------------------------------------------------------------------
    # 2. dxy_gold_alignment
    # ------------------------------------------------------------------
    def dxy_gold_alignment(
        self,
        dxy_direction: str,
        gold_signal_direction: str,
    ) -> dict[str, Any]:
        """
        Assess whether the current DXY direction aligns with a Gold trade.

        Parameters
        ----------
        dxy_direction         : "up" | "down" | "sideways"
        gold_signal_direction : "long" | "short"

        Returns
        -------
        aligned           : bool
        correlation_note  : str
        conflict_severity : "none" | "minor" | "major"
        """
        dxy = dxy_direction.lower().strip()
        gld = gold_signal_direction.lower().strip()

        if dxy == "sideways" or dxy == "ranging":
            return {
                "aligned": True,
                "correlation_note": "DXY ranging — no correlation pressure",
                "conflict_severity": "none",
            }

        if dxy == "up" and gld == "short":
            return {
                "aligned": True,
                "correlation_note": "DXY rising confirms Gold SHORT OK",
                "conflict_severity": "none",
            }

        if dxy == "down" and gld == "long":
            return {
                "aligned": True,
                "correlation_note": "DXY falling confirms Gold LONG OK",
                "conflict_severity": "none",
            }

        if dxy == "up" and gld == "long":
            return {
                "aligned": False,
                "correlation_note": "DXY rising — CONFLICT with Gold LONG X",
                "conflict_severity": "major",
            }

        if dxy == "down" and gld == "short":
            return {
                "aligned": False,
                "correlation_note": "DXY falling — CONFLICT with Gold SHORT X",
                "conflict_severity": "major",
            }

        # Unknown / missing direction
        return {
            "aligned": True,
            "correlation_note": f"DXY direction '{dxy}' unclear — neutral",
            "conflict_severity": "none",
        }

    # ------------------------------------------------------------------
    # 3. dxy_momentum
    # ------------------------------------------------------------------
    def dxy_momentum(self, dxy_df: pd.DataFrame) -> dict[str, Any]:
        """
        Determine the current DXY momentum state.

        Uses EMA20 slope and RSI14 to classify strength.

        Returns
        -------
        dxy_rsi          : float
        dxy_trend        : "up" | "down" | "sideways"
        momentum_strength: "strong" | "weak"
        ema20_value      : float
        detail           : str
        """
        if dxy_df is None or dxy_df.empty or "close" not in dxy_df.columns:
            return {
                "dxy_rsi": 50.0,
                "dxy_trend": "sideways",
                "momentum_strength": "weak",
                "ema20_value": None,
                "detail": "no data",
            }

        close = dxy_df["close"].dropna()

        # EMA20
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema20_val = float(ema20.iloc[-1])
        ema20_slope = float(ema20.iloc[-1] - ema20.iloc[-3]) if len(ema20) >= 3 else 0.0

        # RSI14
        rsi_val = self._rsi(close, period=14)

        # Trend from EMA slope + price position
        last_close = float(close.iloc[-1])
        if ema20_slope > 0.05 and last_close > ema20_val:
            trend = "up"
        elif ema20_slope < -0.05 and last_close < ema20_val:
            trend = "down"
        else:
            trend = "sideways"

        # Strength
        if trend == "up" and rsi_val > 60:
            strength = "strong"
            detail   = f"EMA rising + RSI {rsi_val:.1f} — strong dollar"
        elif trend == "down" and rsi_val < 40:
            strength = "strong"
            detail   = f"EMA falling + RSI {rsi_val:.1f} — weak dollar"
        else:
            strength = "weak"
            detail   = f"EMA flat or mixed, RSI {rsi_val:.1f} — indecisive"

        return {
            "dxy_rsi":           round(rsi_val, 1),
            "dxy_trend":         trend,
            "momentum_strength": strength,
            "ema20_value":       round(ema20_val, 3),
            "detail":            detail,
        }

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _fetch_ticker(
        self, ticker: str, period: str = "60d", interval: str = "1d"
    ) -> pd.DataFrame | None:
        """Download OHLCV for a single ticker, return None on failure."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
            if raw is None or raw.empty:
                return None
            # Flatten multi-level columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw = raw.rename(columns={"adj close": "close"})
            for col in ("open", "high", "low", "close"):
                if col not in raw.columns:
                    return None
            return raw[["open", "high", "low", "close", "volume"]].copy()
        except Exception:
            return None

    def _reconstruct_dxy(
        self, period: str = "60d", interval: str = "1d"
    ) -> pd.DataFrame | None:
        """
        Build a DXY approximation from the basket of currency pairs.
        Uses a weighted geometric mean of daily % changes.
        """
        closes: dict[str, pd.Series] = {}
        for ticker, cfg in _DXY_COMPONENTS.items():
            df = self._fetch_ticker(ticker, period=period, interval=interval)
            if df is not None and not df.empty:
                s = df["close"].copy()
                if cfg["inverse"]:
                    s = 1.0 / s
                closes[ticker] = s

        if not closes:
            return None

        # Align all series on common dates
        combined = pd.DataFrame(closes).dropna()
        if len(combined) < 20:
            return None

        # Weighted daily % change sum
        weighted_chg = pd.Series(0.0, index=combined.index)
        total_weight = 0.0
        for ticker, cfg in _DXY_COMPONENTS.items():
            if ticker not in combined.columns:
                continue
            pct = combined[ticker].pct_change()
            weighted_chg += pct * cfg["weight"]
            total_weight  += cfg["weight"]

        if total_weight < 0.5:
            return None

        # Rebuild index level starting at 100
        idx = (1 + weighted_chg.fillna(0)).cumprod() * 100.0
        idx.iloc[0] = 100.0

        df_out = pd.DataFrame({
            "open":   idx.shift(1).bfill(),
            "high":   idx * 1.001,
            "low":    idx * 0.999,
            "close":  idx,
            "volume": 0,
        })
        return df_out

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ema20, rsi, atr columns to a DXY DataFrame."""
        df = df.copy()
        close = df["close"]

        df["ema20"] = close.ewm(span=20, adjust=False).mean()

        # RSI14
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - 100 / (1 + rs)

        # ATR14
        if "high" in df.columns and "low" in df.columns:
            hl  = df["high"] - df["low"]
            hpc = (df["high"] - close.shift(1)).abs()
            lpc = (df["low"]  - close.shift(1)).abs()
            tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
            df["atr"] = tr.rolling(14).mean()
        else:
            df["atr"] = np.nan

        return df

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> float:
        """Return the latest RSI value for a price series."""
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - 100 / (1 + rs)
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0

    @staticmethod
    def _build_result(df: pd.DataFrame, source: str) -> dict[str, Any]:
        last = df.iloc[-1]
        return {
            "dxy_df":    df,
            "dxy_close": round(float(last["close"]), 3),
            "ema20":     round(float(last["ema20"]), 3) if "ema20" in df.columns else None,
            "rsi":       round(float(last["rsi"]),   1) if "rsi"   in df.columns else None,
            "atr":       round(float(last["atr"]),   3) if "atr"   in df.columns and not np.isnan(last.get("atr", np.nan)) else None,
            "source":    source,
            "available": True,
            "error":     None,
        }

    @staticmethod
    def _unavailable(reason: str) -> dict[str, Any]:
        return {
            "dxy_df":    None,
            "dxy_close": None,
            "ema20":     None,
            "rsi":       None,
            "atr":       None,
            "source":    "unavailable",
            "available": False,
            "error":     reason,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_dxy_context() -> dict[str, Any]:
    """
    One-call convenience wrapper.

    Returns
    -------
    dict with keys:
      dxy_df, dxy_trend, dxy_rsi, momentum_strength,
      available, display_line, dxy_data (full raw result),
      gold_aligned (bool — True if DXY direction aligns with gold signal direction)
    """
    dxy = DXYCorrelation()
    data = dxy.get_dxy_data()

    if not data["available"] or data["dxy_df"] is None:
        return {
            "dxy_df":           None,
            "dxy_trend":        "sideways",
            "dxy_rsi":          50.0,
            "momentum_strength": "weak",
            "available":        False,
            "gold_aligned":     True,   # neutral fallback
            "display_line":     f"DXY STATUS: Unavailable ({data.get('error', '?')})",
            "dxy_data":         data,
        }

    mom = dxy.dxy_momentum(data["dxy_df"])

    trend_word = {
        "up":       "Rising",
        "down":     "Falling",
        "sideways": "Ranging",
    }.get(mom["dxy_trend"], "Unknown")

    display_line = (
        f"DXY STATUS: {trend_word} "
        f"(RSI {mom['dxy_rsi']}) "
        f"[{mom['momentum_strength'].upper()} — {data['source']}]"
    )

    # gold_aligned: DXY falling = good for longs (most common use-case default)
    gold_aligned = mom["dxy_trend"] != "up"   # falling/sideways → aligned with longs

    return {
        "dxy_df":            data["dxy_df"],
        "dxy_trend":         mom["dxy_trend"],
        "dxy_rsi":           mom["dxy_rsi"],
        "momentum_strength": mom["momentum_strength"],
        "ema20":             data["ema20"],
        "dxy_close":         data["dxy_close"],
        "available":         True,
        "gold_aligned":      gold_aligned,
        "display_line":      display_line,
        "dxy_data":          data,
        "momentum":          mom,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  US 10-Year Treasury Yield context
# ══════════════════════════════════════════════════════════════════════════════

def get_yields_context() -> dict[str, Any]:
    """
    Fetch US 10-Year Treasury yield via yfinance (^TNX) and derive
    directional bias for gold.

    Returns
    -------
    dict with keys:
      available         : bool
      current_yield     : float   (e.g. 4.32 — percent)
      yield_change_1d   : float   (basis points)
      yield_change_5d   : float   (basis points)
      yield_trend       : "rising" | "falling" | "sideways"
      yield_momentum    : "strong" | "weak"
      gold_bias_from_yields : "bullish" | "bearish" | "neutral"
      display_line      : str
    """
    _UNAVAILABLE: dict[str, Any] = {
        "available":            False,
        "current_yield":        None,
        "yield_change_1d":      None,
        "yield_change_5d":      None,
        "yield_trend":          "sideways",
        "yield_momentum":       "weak",
        "gold_bias_from_yields": "neutral",
        "display_line":         "US10Y: unavailable",
    }

    if not _YF_OK:
        return {**_UNAVAILABLE, "display_line": "US10Y: yfinance not installed"}

    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            raw = yf.download(
                "^TNX",
                period="30d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

        if raw is None or raw.empty or len(raw) < 6:
            return {**_UNAVAILABLE, "display_line": "US10Y: no data returned"}

        # Flatten multi-level columns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        close = raw["close"].dropna()
        if len(close) < 6:
            return {**_UNAVAILABLE, "display_line": "US10Y: insufficient history"}

        current_yield  = float(close.iloc[-1])
        prev_1d_yield  = float(close.iloc[-2])
        prev_5d_yield  = float(close.iloc[-6]) if len(close) >= 6 else float(close.iloc[0])

        # Changes in basis points (1% = 100 bps; ^TNX is in percent already)
        change_1d = round((current_yield - prev_1d_yield) * 100, 1)   # bps
        change_5d = round((current_yield - prev_5d_yield) * 100, 1)   # bps

        # Trend
        if change_5d > 5:
            yield_trend = "rising"
        elif change_5d < -5:
            yield_trend = "falling"
        else:
            yield_trend = "sideways"

        # Momentum
        yield_momentum = "strong" if abs(change_5d) > 15 else "weak"

        # Gold bias from yields
        # Rising yields + strong → bearish gold (dollar is attractive)
        # Falling yields + strong → bullish gold (safe haven)
        # Anything else → neutral
        if yield_trend == "rising" and yield_momentum == "strong":
            gold_bias = "bearish"
        elif yield_trend == "falling" and yield_momentum == "strong":
            gold_bias = "bullish"
        else:
            gold_bias = "neutral"

        # Direction arrows
        arrow = "↑" if yield_trend == "rising" else ("↓" if yield_trend == "falling" else "→")
        bias_str = {"bullish": "Bullish gold", "bearish": "Bearish gold",
                    "neutral": "Neutral"}.get(gold_bias, "Neutral")
        display_line = (
            f"US10Y: {current_yield:.2f}% {arrow} "
            f"{yield_trend.capitalize()} ({change_5d:+.0f}bps/5d) — {bias_str}"
        )

        return {
            "available":            True,
            "current_yield":        round(current_yield, 3),
            "yield_change_1d":      change_1d,
            "yield_change_5d":      change_5d,
            "yield_trend":          yield_trend,
            "yield_momentum":       yield_momentum,
            "gold_bias_from_yields": gold_bias,
            "display_line":         display_line,
        }

    except Exception as exc:
        return {**_UNAVAILABLE, "display_line": f"US10Y: error — {exc}"}


# ══════════════════════════════════════════════════════════════════════════════
#  Combined macro filter — DXY + US10Y Yields
# ══════════════════════════════════════════════════════════════════════════════

def get_macro_context(gold_direction: str) -> dict[str, Any]:
    """
    Combined macro alignment filter for a given gold trade direction.

    Scores DXY and yields alignment, returns a combined macro_score,
    macro_bias, confidence_adjustment, and human-readable summary.

    Parameters
    ----------
    gold_direction : "long" | "short"

    Returns
    -------
    dict with keys:
      macro_score           : float  (-2.0 to +2.0)
      macro_bias            : str    (strongly_bullish / bullish / neutral /
                                      bearish / strongly_bearish)
      macro_confirmed       : bool   (macro_score >= 1.0)
      macro_opposed         : bool   (macro_score <= -1.0)
      dxy                   : dict   (get_dxy_context result)
      yields                : dict   (get_yields_context result)
      summary               : str
      confidence_adjustment : float  (+1.0 / 0.0 / -1.0)
      display_line          : str
      available             : bool
    """
    is_long = gold_direction.lower().strip() == "long"

    dxy    = get_dxy_context()
    yields = get_yields_context()

    # ── DXY score ──────────────────────────────────────────────────────────────
    dxy_trend = dxy.get("dxy_trend", "sideways")
    if dxy_trend == "sideways":
        dxy_score = 0.0
        dxy_label = "DXY ranging"
    elif (is_long and dxy_trend == "down") or (not is_long and dxy_trend == "up"):
        dxy_score = 1.0
        dxy_label = f"DXY {'falling' if dxy_trend == 'down' else 'rising'} ✓"
    else:
        dxy_score = -1.0
        dxy_label = f"DXY {'rising' if dxy_trend == 'up' else 'falling'} ✗"

    # ── Yields score ───────────────────────────────────────────────────────────
    y_trend = yields.get("yield_trend", "sideways")
    if y_trend == "sideways":
        y_score = 0.0
        y_label = "US10Y sideways"
    elif (is_long and y_trend == "falling") or (not is_long and y_trend == "rising"):
        y_score = 1.0
        y_label = f"US10Y {'falling' if y_trend == 'falling' else 'rising'} ✓"
    else:
        y_score = -1.0
        y_label = f"US10Y {'rising' if y_trend == 'rising' else 'falling'} ✗"

    macro_score = dxy_score + y_score

    # ── Macro bias label ───────────────────────────────────────────────────────
    if macro_score >= 2.0:
        macro_bias = "strongly_bullish" if is_long else "strongly_bearish"
    elif macro_score >= 1.0:
        macro_bias = "bullish" if is_long else "bearish"
    elif macro_score <= -2.0:
        macro_bias = "strongly_bearish" if is_long else "strongly_bullish"
    elif macro_score <= -1.0:
        macro_bias = "bearish" if is_long else "bullish"
    else:
        macro_bias = "neutral"

    macro_confirmed = macro_score >= 1.0
    macro_opposed   = macro_score <= -1.0

    # ── Confidence adjustment ──────────────────────────────────────────────────
    if macro_confirmed:
        conf_adj = 1.0
    elif macro_opposed:
        conf_adj = -1.0
    else:
        conf_adj = 0.0

    # ── Summary string ─────────────────────────────────────────────────────────
    parts = [dxy_label, y_label]
    direction_str = "bullish gold" if is_long else "bearish gold"
    bias_words = {
        "strongly_bullish": "strongly bullish gold",
        "bullish":          "bullish gold",
        "neutral":          "neutral",
        "bearish":          "bearish gold",
        "strongly_bearish": "strongly bearish gold",
    }
    summary = " + ".join(parts) + f" → {bias_words.get(macro_bias, macro_bias)}"

    # ── Display line ────────────────────────────────────────────────────────────
    y_cur  = yields.get("current_yield")
    y_str  = f"{y_cur:.2f}%" if y_cur else "N/A"
    y_arr  = {"rising": "↑", "falling": "↓", "sideways": "→"}.get(y_trend, "→")
    dxy_tr = {"up": "Rising ▲", "down": "Falling ▼", "sideways": "Ranging ─"}.get(dxy_trend, "—")
    display_line = (
        f"DXY: {dxy_tr} (RSI {dxy.get('dxy_rsi', '—')})  |  "
        f"US10Y: {y_str} {y_arr}  |  "
        f"Macro: {macro_bias.replace('_', ' ').title()}"
    )

    return {
        "macro_score":          macro_score,
        "macro_bias":           macro_bias,
        "macro_confirmed":      macro_confirmed,
        "macro_opposed":        macro_opposed,
        "dxy":                  dxy,
        "yields":               yields,
        "summary":              summary,
        "confidence_adjustment": conf_adj,
        "display_line":         display_line,
        "available":            dxy.get("available", False) or yields.get("available", False),
        # Expose DXY fields at top level for backward compatibility
        "dxy_trend":            dxy_trend,
        "dxy_rsi":              dxy.get("dxy_rsi", 50.0),
        "momentum_strength":    dxy.get("momentum_strength", "weak"),
        "dxy_close":            dxy.get("dxy_close"),
        "dxy_df":               dxy.get("dxy_df"),
    }


def print_dxy_report(gold_direction: str, dxy_ctx: dict | None = None) -> None:
    """
    Print a formatted DXY correlation report.

    Parameters
    ----------
    gold_direction : "long" | "short"
    dxy_ctx        : result from get_dxy_context(), or None
    """
    print()
    print("  ╔" + "═" * 44 + "╗")
    print("  ║" + "  DXY CORRELATION REPORT  ".center(44) + "║")
    print("  ╚" + "═" * 44 + "╝")

    if dxy_ctx is None or not dxy_ctx.get("available"):
        reason = (dxy_ctx or {}).get("display_line", "DXY data unavailable")
        print(f"  {reason}")
        print()
        return

    trend     = dxy_ctx.get("dxy_trend", "sideways")
    rsi_val   = dxy_ctx.get("dxy_rsi", 50.0)
    strength  = dxy_ctx.get("momentum_strength", "weak")
    close_val = dxy_ctx.get("dxy_close")

    trend_word = {"up": "Rising ▲", "down": "Falling ▼", "sideways": "Ranging ─"}.get(trend, trend)
    close_str  = f"  |  Close: {close_val:.3f}" if close_val else ""

    print(f"  DXY STATUS   : {trend_word} (RSI {rsi_val}){close_str}")
    print(f"  STRENGTH     : {strength.upper()}")

    # Alignment check
    dxy_obj   = DXYCorrelation()
    alignment = dxy_obj.dxy_gold_alignment(trend, gold_direction)

    icon = "✅" if alignment["aligned"] else "⚠️ "
    print(f"  CORRELATION  : {icon}  {alignment['correlation_note']}")

    # Recommendation
    if not alignment["aligned"]:
        gold_up = "longs" if gold_direction == "long" else "shorts"
        dxy_act = "weakens" if trend == "up" else "strengthens"
        print(f"  RECOMMENDATION: Avoid {gold_up} until DXY {dxy_act}")
    elif trend == "sideways":
        print( "  RECOMMENDATION: No DXY pressure — trade on own merit")
    else:
        print( "  RECOMMENDATION: DXY confirms trade direction — proceed")

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Self-test (run directly: python dxy_correlation.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  DXY CORRELATION — SELF TEST")
    print("=" * 55)

    # --- Alignment logic (no network needed) ---
    dxy = DXYCorrelation()

    cases = [
        ("up",       "short",  True),
        ("down",     "long",   True),
        ("up",       "long",   False),
        ("down",     "short",  False),
        ("sideways", "long",   True),
        ("sideways", "short",  True),
    ]

    print("\nAlignment logic tests:")
    all_pass = True
    for dxy_dir, gold_dir, expected in cases:
        result = dxy.dxy_gold_alignment(dxy_dir, gold_dir)
        status = "OK" if result["aligned"] == expected else "X FAIL"
        if result["aligned"] != expected:
            all_pass = False
        print(f"  DXY {dxy_dir:9} + Gold {gold_dir:5} -> aligned={result['aligned']}  {status}")
        print(f"    {result['correlation_note']}")

    assert all_pass, "Alignment logic tests failed!"
    print("\n  All alignment tests OK")

    # --- Momentum logic ---
    print("\nMomentum test (synthetic data):")
    dates  = pd.date_range("2024-01-01", periods=60, freq="D")
    prices = pd.Series(100 + np.linspace(0, 5, 60) + np.random.randn(60) * 0.2, index=dates)
    df_syn = pd.DataFrame({"close": prices, "high": prices * 1.01, "low": prices * 0.99})
    df_syn = dxy._enrich(df_syn)
    mom    = dxy.dxy_momentum(df_syn)
    print(f"  Trend: {mom['dxy_trend']}  |  RSI: {mom['dxy_rsi']}  |  Strength: {mom['momentum_strength']}")
    assert mom["dxy_trend"] in ("up", "down", "sideways")

    # --- Live data (network) ---
    print("\nLive DXY data fetch:")
    ctx = get_dxy_context()
    print(f"  {ctx['display_line']}")
    print(f"  Available: {ctx['available']}")
    if ctx["available"]:
        print(f"  Trend: {ctx['dxy_trend']}  |  RSI: {ctx['dxy_rsi']}")
        print_dxy_report("long", ctx)

    print("\n  Self-test complete OK")

