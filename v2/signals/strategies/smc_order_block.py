"""
strategies/smc_order_block.py — SMC Order Block Retest strategy.

Fires when:
  1. An active untested Order Block aligned with direction exists
  2. Market structure (BOS) confirms direction
  3. Price is currently at or entering the OB zone
  4. HTF (H4 or D1) agrees

Best instruments: XAUUSD, GBPJPY (most reactive to institutional OBs)
"""
from __future__ import annotations

import logging

import pandas as pd

from v2.signals.strategies.base import StrategyBase, StrategyResult
from v2.analysis.smart_money import SmartMoneyAnalyzer

logger = logging.getLogger(__name__)
_SMC = SmartMoneyAnalyzer()

OB_PROX_PCT = 0.008   # price within 0.8% of OB level = "at the OB"


class SMCOrderBlockStrategy(StrategyBase):
    name        = "smc_order_block"
    instruments = ["XAUUSD", "GBPJPY", "WTI", "NAS100", "BTCUSDT", "ETHUSDT"]
    timeframes  = ["H1"]
    min_df_bars = 60

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
        if not htf_ok:
            return self._no_signal(symbol, direction, f"HTF opposing: {htf_reason}")

        # ── Find active OBs ───────────────────────────────────────────────────
        obs = _SMC.find_order_blocks(df_h1)
        target_type = "bullish" if is_long else "bearish"
        price = float(df_h1["close"].iloc[-1])

        # Filter: correct type, active (price near OB), untested preferred
        matching = [
            ob for ob in obs
            if ob["ob_type"] == target_type
            and abs(price - ob["ob_level"]) / max(ob["ob_level"], 1e-9) <= OB_PROX_PCT
        ]

        if not matching:
            return self._no_signal(symbol, direction, f"No active {target_type} OB within {OB_PROX_PCT*100:.1f}% of price")

        best_ob = matching[0]  # already sorted: active+untested+strong first

        # ── Market structure check ────────────────────────────────────────────
        struct = _SMC.detect_market_structure(df_h1)
        bias   = struct.get("bias", "neutral")
        struct_ok = (is_long and bias in ("bullish", "neutral")) or \
                    (not is_long and bias in ("bearish", "neutral"))

        if not struct_ok:
            return self._no_signal(symbol, direction,
                                   f"Structure bias {bias} opposes {direction}")

        # ── ADX — avoid entering OBs in choppy markets ─────────────────────
        adx = self._adx(df_h1)
        if adx.get("adx", 0) < 15:
            return self._no_signal(symbol, direction, f"ADX {adx.get('adx', 0):.0f} too weak — market choppy")

        # ── Entry, SL, TP ─────────────────────────────────────────────────────
        atr = self._atr(df_h1)

        entry = best_ob["ob_level"]
        if is_long:
            stop_loss = round(best_ob["ob_low"] - atr * 0.3, 5)
        else:
            stop_loss = round(best_ob["ob_high"] + atr * 0.3, 5)

        tp1, tp2 = self._calc_tps(entry, stop_loss, direction, rr1=2.0, rr2=4.0)

        # ── Score ─────────────────────────────────────────────────────────────
        score = 0.0
        score += 2.0 if htf_ok else 0.0
        score += 2.5 if best_ob["untested"] else 1.5
        score += 1.5 if best_ob["ob_strength"] == "strong" else 1.0
        score += 1.5 if struct_ok and struct["last_bos"] else 1.0
        score += 1.5 if adx.get("trending") else 0.5
        score += 1.0 if struct.get("last_choch") else 0.5

        liq = _SMC.find_liquidity_levels(df_h1)
        score += 1.0 if liq["swept_recently"] else 0.0

        reasons = [
            f"{target_type.capitalize()} OB at {best_ob['ob_level']:.2f} "
            f"[{best_ob['ob_low']:.2f}–{best_ob['ob_high']:.2f}] "
            f"({'untested' if best_ob['untested'] else 'tested'}, {best_ob['ob_strength']})",
            htf_reason,
            struct.get("detail", f"Structure: {struct['structure']}")[:80],
            f"ADX={adx.get('adx', 0):.0f} ({adx.get('strength', 'n/a')})",
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
                "ob_level":    best_ob["ob_level"],
                "ob_strength": best_ob["ob_strength"],
                "ob_untested": best_ob["untested"],
                "htf_ok":      htf_ok,
                "struct_bias": bias,
                "adx":         adx.get("adx", 0),
            },
        )
