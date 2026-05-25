"""
analysis/mtf_analyzer.py — Multi-timeframe alignment analyzer for TradingBotV2.

Determines whether D1 / H4 / H1 trend direction agrees.
Used by confluence engine as the HTF alignment factor.

Usage:
    from v2.analysis.mtf_analyzer import MTFAnalyzer
    mtf = MTFAnalyzer(feed)
    result = mtf.analyze("XAUUSD", "long")
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from v2.analysis.indicators import get_adx, get_supertrend, get_macd, get_ichimoku

if TYPE_CHECKING:
    from v2.connectors.unified_data import DataFeed

logger = logging.getLogger(__name__)


class MTFAnalyzer:
    """Checks trend alignment across D1, H4, and H1 timeframes."""

    TIMEFRAMES = ["D1", "H4", "H1"]

    def __init__(self, feed: "DataFeed") -> None:
        self._feed = feed

    def analyze(self, symbol: str, direction: str, count: int = 200) -> dict:
        """
        Fetch D1/H4/H1 and check whether all three align with direction.

        Returns
        -------
        {
            "symbol":     str,
            "direction":  str,
            "aligned":    bool,     # all 3 TFs agree
            "score":      float,    # 0.0–3.0 (1 per TF in alignment)
            "per_tf":     dict,     # per-timeframe breakdown
            "summary":    str,
        }
        """
        is_long = direction.lower() in ("long", "buy")
        per_tf: dict[str, dict] = {}
        aligned_count = 0

        for tf in self.TIMEFRAMES:
            df = self._feed.get_ohlcv(symbol, tf, count)
            if df.empty or len(df) < 30:
                per_tf[tf] = {"aligned": False, "reason": "insufficient data"}
                continue

            tf_result = self._check_tf(df, direction)
            per_tf[tf] = tf_result
            if tf_result["aligned"]:
                aligned_count += 1

        all_aligned = aligned_count == len(self.TIMEFRAMES)
        score       = float(aligned_count)
        summary     = (
            f"{aligned_count}/{len(self.TIMEFRAMES)} timeframes aligned with {direction.upper()}"
        )

        return {
            "symbol":    symbol,
            "direction": direction,
            "aligned":   all_aligned,
            "score":     score,
            "per_tf":    per_tf,
            "summary":   summary,
        }

    def _check_tf(self, df: pd.DataFrame, direction: str) -> dict:
        """Return alignment dict for a single timeframe."""
        is_long = direction.lower() in ("long", "buy")
        signals: list[bool] = []
        reasons: list[str]  = []

        try:
            adx = get_adx(df)
            if adx.get("trending"):
                adx_aligned = adx.get("bias") == ("bullish" if is_long else "bearish")
                signals.append(adx_aligned)
                if adx_aligned:
                    reasons.append(f"ADX {adx.get('adx',0):.0f} trending {'bull' if is_long else 'bear'}")
        except Exception:
            pass

        try:
            st = get_supertrend(df)
            st_aligned = st.get("trend") == ("bullish" if is_long else "bearish")
            signals.append(st_aligned)
            if st_aligned:
                reasons.append(f"Supertrend {st.get('trend','')}")
        except Exception:
            pass

        try:
            macd = get_macd(df)
            macd_aligned = macd.get("bias", "") in (
                ("bullish", "strongly_bullish") if is_long
                else ("bearish", "strongly_bearish")
            )
            signals.append(macd_aligned)
            if macd_aligned:
                reasons.append(f"MACD {macd.get('bias','')}")
        except Exception:
            pass

        if not signals:
            return {"aligned": False, "reason": "no indicators computed"}

        aligned = sum(signals) > len(signals) / 2   # majority vote
        return {
            "aligned": aligned,
            "reason":  ", ".join(reasons) if reasons else "not aligned",
            "votes":   f"{sum(signals)}/{len(signals)}",
        }
