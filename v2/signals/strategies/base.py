"""
signals/strategies/base.py — Base class and result type for all strategy modules.

Every strategy inherits StrategyBase and implements evaluate().
StrategyResult is the standard return type shared across the whole system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategyResult:
    """Standardised result returned by every strategy evaluate() call."""
    signal:        bool
    strategy_name: str
    symbol:        str
    direction:     str
    score:         float        # 0–10 quality score
    entry_price:   float
    stop_loss:     float
    tp1_price:     float
    tp2_price:     float
    reasons:       list[str] = field(default_factory=list)
    blocked_by:    str = ""
    factors:       dict = field(default_factory=dict)

    def to_signal_dict(self) -> dict:
        """Convert to the signal dict format used by PaperTrader / Checklist."""
        return {
            "symbol":           self.symbol,
            "direction":        self.direction,
            "entry_price":      self.entry_price,
            "stop_loss":        self.stop_loss,
            "tp1_price":        self.tp1_price,
            "tp2_price":        self.tp2_price,
            "score":            self.score,
            "confluence_score": self.score,
            "strategy":         self.strategy_name,
            "factors":          self.factors,
            "reasons":          self.reasons,
        }


class StrategyBase:
    """
    Abstract base for all strategy modules.

    Subclasses set:
        name        : str  — unique strategy identifier
        instruments : list — empty = all instruments
        timeframes  : list — which timeframes this strategy applies to
        min_df_bars : int  — minimum bars needed in df_h1

    And implement:
        evaluate(symbol, direction, df_h1, df_h4, df_d1, context) -> StrategyResult
    """
    name:        str       = "base"
    instruments: list[str] = []
    timeframes:  list[str] = ["H1"]
    min_df_bars: int       = 50

    def evaluate(
        self,
        symbol:    str,
        direction: str,
        df_h1:     pd.DataFrame,
        df_h4:     pd.DataFrame | None = None,
        df_d1:     pd.DataFrame | None = None,
        context:   dict | None = None,
    ) -> StrategyResult:
        raise NotImplementedError

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _no_signal(self, symbol: str, direction: str, reason: str) -> StrategyResult:
        return StrategyResult(
            signal=False, strategy_name=self.name, symbol=symbol,
            direction=direction, score=0.0,
            entry_price=0.0, stop_loss=0.0, tp1_price=0.0, tp2_price=0.0,
            blocked_by=reason,
        )

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            hl = df["high"] - df["low"]
            hc = (df["high"] - df["close"].shift()).abs()
            lc = (df["low"]  - df["close"].shift()).abs()
            return float(pd.concat([hl, hc, lc], axis=1).max(axis=1)
                         .rolling(period).mean().iloc[-1])
        except Exception:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1])

    def _ema(self, df: pd.DataFrame, span: int) -> float:
        return float(df["close"].ewm(span=span, adjust=False).mean().iloc[-1])

    def _calc_tps(
        self, entry: float, sl: float, direction: str,
        rr1: float = 2.0, rr2: float = 4.0,
    ) -> tuple[float, float]:
        dist = abs(entry - sl)
        if direction.lower() in ("long", "buy"):
            return round(entry + dist * rr1, 5), round(entry + dist * rr2, 5)
        return round(entry - dist * rr1, 5), round(entry - dist * rr2, 5)

    def _htf_bias(
        self,
        df_h4: pd.DataFrame | None,
        df_d1: pd.DataFrame | None,
        direction: str,
    ) -> tuple[bool, str]:
        """Return (aligned, reason). True if at least one HTF agrees."""
        is_long = direction.lower() in ("long", "buy")
        aligned = 0
        reasons: list[str] = []
        for label, df in [("H4", df_h4), ("D1", df_d1)]:
            if df is None or df.empty or len(df) < 30:
                continue
            ema50 = float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
            price = float(df["close"].iloc[-1])
            if is_long and price > ema50:
                aligned += 1
                reasons.append(f"{label} bullish")
            elif not is_long and price < ema50:
                aligned += 1
                reasons.append(f"{label} bearish")
        has_data = any(
            df is not None and not df.empty and len(df) >= 30
            for df in [df_h4, df_d1]
        )
        if not has_data:
            return True, "No HTF data — check skipped"
        ok = aligned > 0
        return ok, " | ".join(reasons) if reasons else f"HTF opposes {direction}"

    def _adx(self, df: pd.DataFrame, period: int = 14) -> dict:
        try:
            from v2.analysis.indicators import get_adx
            return get_adx(df, period)
        except Exception:
            return {"adx": 0.0, "trending": False, "bias": "neutral"}

    def _macd(self, df: pd.DataFrame) -> dict:
        try:
            from v2.analysis.indicators import get_macd
            return get_macd(df)
        except Exception:
            return {"bias": "neutral", "bullish_cross": False, "bearish_cross": False}
