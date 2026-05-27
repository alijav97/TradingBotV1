"""
strategies/ict_gold.py — ICT Gold Charter Model for XAUUSD.

5-step entry model:
  1. D1 HTF Bias      — D1 trend determines allowed direction
  2. H4 Confirmation  — H4 structure confirms D1 bias
  3. Liquidity Sweep  — H1 shows stop hunt (correct-side sweep + reversal confirmed)
  4. FVG / OB Entry   — H1 FVG or Order Block provides precise entry zone
  5. Low-Hanging TP   — next liquidity cluster defines TP1

Only fires on XAUUSD. Direction must match D1+H4 bias.

Fixes vs v1:
  - Sweep direction validated (sell-side for longs, buy-side for shorts)
  - Neutral CHoCH bias no longer accepted
  - Entry proximity check: price must be within 1.5×ATR of FVG/OB zone
"""
from __future__ import annotations

import logging

import pandas as pd

from v2.signals.strategies.base import StrategyBase, StrategyResult
from v2.analysis.smart_money import SmartMoneyAnalyzer

logger = logging.getLogger(__name__)

_SMC = SmartMoneyAnalyzer()

HTF_EMA_SPAN   = 50
H1_BARS_NEEDED = 80
SWEEP_LOOKBACK = 10


class ICTGoldStrategy(StrategyBase):
    name        = "ict_gold"
    instruments = ["XAUUSD"]
    timeframes  = ["H1"]
    min_df_bars = H1_BARS_NEEDED

    def evaluate(
        self,
        symbol:    str,
        direction: str,
        df_h1:     pd.DataFrame,
        df_h4:     pd.DataFrame | None = None,
        df_d1:     pd.DataFrame | None = None,
        context:   dict | None = None,
    ) -> StrategyResult:

        if symbol != "XAUUSD":
            return self._no_signal(symbol, direction, "ICT Gold only for XAUUSD")
        if len(df_h1) < H1_BARS_NEEDED:
            return self._no_signal(symbol, direction, "Insufficient H1 bars")

        is_long = direction.lower() in ("long", "buy")

        # ── STEP 1: D1 HTF Bias ────────────────────────────────────────────────
        d1_bias = "neutral"
        if df_d1 is not None and len(df_d1) >= 50:
            d1_ema   = float(df_d1["close"].ewm(span=HTF_EMA_SPAN, adjust=False).mean().iloc[-1])
            d1_price = float(df_d1["close"].iloc[-1])
            d1_bias  = "bullish" if d1_price > d1_ema else "bearish"

        if d1_bias == "neutral":
            return self._no_signal(symbol, direction, "D1 bias unclear — need 50+ D1 bars")
        if is_long and d1_bias != "bullish":
            return self._no_signal(symbol, direction, "D1 bearish — no longs in ICT Gold model")
        if not is_long and d1_bias != "bearish":
            return self._no_signal(symbol, direction, "D1 bullish — no shorts in ICT Gold model")

        # ── STEP 2: H4 Confirmation ────────────────────────────────────────────
        h4_ok     = False
        h4_reason = "No H4 data"
        if df_h4 is not None and len(df_h4) >= 30:
            h4_ema   = float(df_h4["close"].ewm(span=HTF_EMA_SPAN, adjust=False).mean().iloc[-1])
            h4_price = float(df_h4["close"].iloc[-1])
            h4_ok    = (is_long and h4_price > h4_ema) or (not is_long and h4_price < h4_ema)
            h4_reason = f"H4 {'confirms' if h4_ok else 'opposes'} {direction}"

        if not h4_ok:
            return self._no_signal(symbol, direction, h4_reason)

        # ── STEP 3: Directional Liquidity Sweep ───────────────────────────────
        # FIX: validate that the CORRECT side was swept, not just any sweep.
        # Long  → sell-side liquidity (equal lows) must have been taken, then reversed up
        # Short → buy-side liquidity (equal highs) must have been taken, then reversed down
        liq    = _SMC.find_liquidity_levels(df_h1)
        struct = _SMC.detect_market_structure(df_h1)

        atr   = self._atr(df_h1)
        price = float(df_h1["close"].iloc[-1])

        recent = df_h1.iloc[-SWEEP_LOOKBACK:]

        if is_long:
            ssl = liq.get("sell_side_liquidity", [])
            if not ssl:
                return self._no_signal(symbol, direction, "No sell-side liquidity levels found")
            # Nearest sell-side level to current price
            nearest_ssl = min(ssl, key=lambda x: abs(x - price))
            # At least one bar in lookback must have wicked below this level
            price_swept_ssl = any(float(row["low"]) < nearest_ssl for _, row in recent.iterrows())
            if not price_swept_ssl:
                return self._no_signal(
                    symbol, direction,
                    f"No sell-side sweep below {nearest_ssl:.2f} in last {SWEEP_LOOKBACK} bars"
                )
            # Current close must be ABOVE the swept level (reversal confirmed)
            if price < nearest_ssl:
                return self._no_signal(symbol, direction,
                                       "Price still below swept level — reversal not confirmed")
        else:
            bsl = liq.get("buy_side_liquidity", [])
            if not bsl:
                return self._no_signal(symbol, direction, "No buy-side liquidity levels found")
            nearest_bsl = min(bsl, key=lambda x: abs(x - price))
            price_swept_bsl = any(float(row["high"]) > nearest_bsl for _, row in recent.iterrows())
            if not price_swept_bsl:
                return self._no_signal(
                    symbol, direction,
                    f"No buy-side sweep above {nearest_bsl:.2f} in last {SWEEP_LOOKBACK} bars"
                )
            if price > nearest_bsl:
                return self._no_signal(symbol, direction,
                                       "Price still above swept level — reversal not confirmed")

        # FIX: require clear directional bias, not neutral
        bias = struct.get("bias", "neutral")
        if is_long and bias != "bullish":
            return self._no_signal(symbol, direction,
                                   f"Structure bias '{bias}' — need bullish CHoCH after sweep")
        if not is_long and bias != "bearish":
            return self._no_signal(symbol, direction,
                                   f"Structure bias '{bias}' — need bearish CHoCH after sweep")

        choch = struct.get("last_choch")

        # ── STEP 4: FVG / OB Entry Zone ───────────────────────────────────────
        fvgs = _SMC.find_fair_value_gaps(df_h1)
        obs  = _SMC.find_order_blocks(df_h1)

        target_fvg  = "bullish" if is_long else "bearish"
        active_fvgs = [f for f in fvgs if f["active"] and not f["filled"]
                       and f["fvg_type"] == target_fvg]
        target_ob   = "bullish" if is_long else "bearish"
        active_obs  = [ob for ob in obs if ob["active"] and ob["ob_type"] == target_ob]

        if not active_fvgs and not active_obs:
            return self._no_signal(symbol, direction, "No active FVG or OB for entry")

        if active_fvgs:
            best         = active_fvgs[0]
            entry        = best["fvg_midpoint"]
            entry_source = f"FVG {best['fvg_bottom']:.2f}–{best['fvg_top']:.2f}"
        else:
            best         = active_obs[0]
            entry        = best["ob_level"]
            entry_source = f"OB {best['ob_low']:.2f}–{best['ob_high']:.2f}"

        # FIX: entry zone proximity check — skip if price is too far from zone
        if abs(price - entry) > atr * 1.5:
            return self._no_signal(
                symbol, direction,
                f"Price {price:.2f} too far from entry zone {entry:.2f} "
                f"(gap {abs(price - entry):.2f} > 1.5×ATR {atr * 1.5:.2f})"
            )

        # Use current price as entry (live tick will override in scheduler anyway)
        entry = price

        # ── STEP 5: SL below swept level, TP at next liquidity ────────────────
        if is_long:
            swept_level = nearest_ssl
            stop_loss   = round(swept_level - atr * 0.5, 2)
            bsl_targets = liq.get("buy_side_liquidity", [])
            tp1_target  = bsl_targets[0] if bsl_targets else None
        else:
            swept_level = nearest_bsl
            stop_loss   = round(swept_level + atr * 0.5, 2)
            ssl_targets = liq.get("sell_side_liquidity", [])
            tp1_target  = ssl_targets[0] if ssl_targets else None

        dist = abs(entry - stop_loss)
        if dist < atr * 0.5:
            # SL too tight — pad it
            dist      = atr * 1.0
            stop_loss = round(entry - dist, 2) if is_long else round(entry + dist, 2)

        tp1 = (
            round(tp1_target, 2)
            if tp1_target and abs(tp1_target - entry) >= dist
            else round(entry + dist * 2, 2) if is_long else round(entry - dist * 2, 2)
        )
        tp2 = round(entry + dist * 4, 2) if is_long else round(entry - dist * 4, 2)

        # ── Score (0–10) ───────────────────────────────────────────────────────
        score = 0.0
        score += 2.0                                 # D1 bias confirmed
        score += 2.0                                 # H4 confirmation
        score += 2.0                                 # directional sweep confirmed
        score += 1.5 if active_fvgs else 1.0        # FVG preferred over OB
        score += 1.0 if choch else 0.0              # CHoCH present
        score += 1.5 if struct.get("structure") in ("trending_up", "trending_down") else 0.5

        reasons = [
            f"D1 {d1_bias} HTF bias confirmed",
            h4_reason,
            f"{'Sell' if is_long else 'Buy'}-side liquidity swept — stop hunt complete",
            f"Entry zone: {entry_source}",
            struct.get("detail", "")[:80],
        ]

        return StrategyResult(
            signal=True,
            strategy_name=self.name,
            symbol=symbol,
            direction=direction,
            score=round(min(score, 10.0), 1),
            entry_price=round(entry, 2),
            stop_loss=stop_loss,
            tp1_price=tp1,
            tp2_price=tp2,
            reasons=[r for r in reasons if r],
            factors={
                "d1_bias":  d1_bias,
                "h4_ok":    h4_ok,
                "swept":    True,
                "has_fvg":  bool(active_fvgs),
                "has_ob":   bool(active_obs),
                "choch":    bool(choch),
            },
        )
