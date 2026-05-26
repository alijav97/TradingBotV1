"""
strategies/fvg_fill.py — Fair Value Gap Fill strategy.

Price frequently returns to fill price inefficiency gaps. This strategy:
  1. Finds active unfilled FVGs aligned with the trade direction
  2. Price is currently entering (at or near) the FVG
  3. Market structure supports the fill direction
  4. MACD doesn't strongly contradict

Applies to all instruments. FVGs are strongest on XAUUSD and BTCUSDT.
"""
from __future__ import annotations

import logging

import pandas as pd

from v2.signals.strategies.base import StrategyBase, StrategyResult
from v2.analysis.smart_money import SmartMoneyAnalyzer

logger = logging.getLogger(__name__)
_SMC = SmartMoneyAnalyzer()

FVG_ENTRY_PCT = 0.015   # price must be within 1.5% of FVG midpoint


class FVGFillStrategy(StrategyBase):
    name        = "fvg_fill"
    instruments = []   # all instruments
    timeframes  = ["H1"]
    min_df_bars = 50

    def evaluate(
        self,
        symbol:    str,
        direction: str,
        df_h1:     pd.DataFrame,
        df_h4:     pd.DataFrame | None = None,
        df_d1:     pd.DataFrame | None = None,
        context:   dict | None = None,
    ) -> StrategyResult:

        if len(df_h1) < self.min_df_bars:
            return self._no_signal(symbol, direction, "Insufficient H1 bars")

        is_long = direction.lower() in ("long", "buy")

        # ── HTF check ─────────────────────────────────────────────────────────
        htf_ok, htf_reason = self._htf_bias(df_h4, df_d1, direction)

        # ── Find unfilled FVG ─────────────────────────────────────────────────
        fvgs       = _SMC.find_fair_value_gaps(df_h1)
        target_fvg = "bullish" if is_long else "bearish"
        price      = float(df_h1["close"].iloc[-1])

        candidates = [
            f for f in fvgs
            if f["fvg_type"] == target_fvg
            and not f["filled"]
            and abs(price - f["fvg_midpoint"]) / max(f["fvg_midpoint"], 1e-9) <= FVG_ENTRY_PCT
        ]

        if not candidates:
            return self._no_signal(symbol, direction,
                                   f"No active unfilled {target_fvg} FVG within {FVG_ENTRY_PCT*100:.1f}% of price")

        best_fvg = candidates[0]

        # ── Market structure check ────────────────────────────────────────────
        struct    = _SMC.detect_market_structure(df_h1)
        bias      = struct.get("bias", "neutral")
        struct_ok = (is_long and bias in ("bullish", "neutral")) or \
                    (not is_long and bias in ("bearish", "neutral"))

        # ── MACD check — don't fight a strong momentum signal ─────────────────
        macd = self._macd(df_h1)
        macd_strongly_against = (
            (is_long  and macd.get("bias") in ("strongly_bearish",)) or
            (not is_long and macd.get("bias") in ("strongly_bullish",))
        )
        if macd_strongly_against:
            return self._no_signal(symbol, direction,
                                   f"Strong MACD momentum against direction — skip FVG fill")

        # ── Entry, SL, TP ─────────────────────────────────────────────────────
        atr   = self._atr(df_h1)
        entry = best_fvg["fvg_midpoint"]

        if is_long:
            stop_loss = round(best_fvg["fvg_bottom"] - atr * 0.3, 5)
        else:
            stop_loss = round(best_fvg["fvg_top"] + atr * 0.3, 5)

        tp1, tp2 = self._calc_tps(entry, stop_loss, direction, rr1=2.0, rr2=4.0)

        # ── Score ─────────────────────────────────────────────────────────────
        gap_size = best_fvg["fvg_top"] - best_fvg["fvg_bottom"]
        score = 0.0
        score += 3.0                                    # FVG found and price at it
        score += 2.0 if htf_ok else 0.5
        score += 1.5 if struct_ok else 0.0
        score += 1.5 if not struct.get("last_choch") else 2.0   # CHoCH = stronger
        score += min(gap_size / max(atr, 1e-9) * 1.5, 2.0)      # bigger gap = better

        reasons = [
            f"{target_fvg.capitalize()} FVG {best_fvg['fvg_bottom']:.5f}–{best_fvg['fvg_top']:.5f} "
            f"(mid {best_fvg['fvg_midpoint']:.5f}, unfilled)",
            htf_reason,
            struct.get("detail", f"Structure bias: {bias}")[:80],
        ]

        return StrategyResult(
            signal=True,
            strategy_name=self.name,
            symbol=symbol,
            direction=direction,
            score=round(min(score, 10.0), 1),
            entry_price=round(entry, 5),
            stop_loss=stop_loss,
            tp1_price=tp1,
            tp2_price=tp2,
            reasons=[r for r in reasons if r],
            factors={
                "fvg_top":    best_fvg["fvg_top"],
                "fvg_bottom": best_fvg["fvg_bottom"],
                "fvg_mid":    best_fvg["fvg_midpoint"],
                "htf_ok":     htf_ok,
                "struct_bias": bias,
            },
        )
