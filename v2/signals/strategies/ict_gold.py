"""
strategies/ict_gold.py — ICT Gold Charter Model for XAUUSD.

5-step entry model:
  1. D1 HTF Bias      — D1 trend determines allowed direction
  2. H4 Confirmation  — H4 structure confirms D1 bias
  3. Liquidity Sweep  — H1 shows stop hunt (equal high/low taken then reversed)
  4. FVG / OB Entry   — H1 FVG or Order Block provides precise entry zone
  5. Low-Hanging TP   — next liquidity cluster defines TP1

Only fires on XAUUSD. Direction must match D1+H4 bias.
"""
from __future__ import annotations

import logging

import pandas as pd

from v2.signals.strategies.base import StrategyBase, StrategyResult
from v2.analysis.smart_money import SmartMoneyAnalyzer

logger = logging.getLogger(__name__)

_SMC = SmartMoneyAnalyzer()

HTF_EMA_SPAN   = 50     # EMA for D1 / H4 bias
H1_BARS_NEEDED = 80
SWEEP_LOOKBACK = 10     # bars to look back for sweep


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

        # ── STEP 1: D1 HTF Bias ───────────────────────────────────────────────
        d1_bias = "neutral"
        if df_d1 is not None and len(df_d1) >= 50:
            d1_ema  = float(df_d1["close"].ewm(span=HTF_EMA_SPAN, adjust=False).mean().iloc[-1])
            d1_price = float(df_d1["close"].iloc[-1])
            d1_bias  = "bullish" if d1_price > d1_ema else "bearish"

        if d1_bias == "neutral":
            return self._no_signal(symbol, direction, "D1 bias unclear — need 50+ D1 bars")

        if is_long and d1_bias != "bullish":
            return self._no_signal(symbol, direction, f"D1 bearish — no longs in ICT Gold model")
        if not is_long and d1_bias != "bearish":
            return self._no_signal(symbol, direction, f"D1 bullish — no shorts in ICT Gold model")

        # ── STEP 2: H4 Confirmation ───────────────────────────────────────────
        h4_ok = False
        h4_reason = "No H4 data"
        if df_h4 is not None and len(df_h4) >= 30:
            h4_ema   = float(df_h4["close"].ewm(span=HTF_EMA_SPAN, adjust=False).mean().iloc[-1])
            h4_price = float(df_h4["close"].iloc[-1])
            h4_ok    = (is_long and h4_price > h4_ema) or (not is_long and h4_price < h4_ema)
            h4_reason = f"H4 {'confirms' if h4_ok else 'opposes'} {direction} (price {'>' if h4_price > h4_ema else '<'} EMA50)"

        if not h4_ok:
            return self._no_signal(symbol, direction, h4_reason)

        # ── STEP 3: Liquidity Sweep on H1 ────────────────────────────────────
        liq    = _SMC.find_liquidity_levels(df_h1)
        struct = _SMC.detect_market_structure(df_h1)
        swept  = liq["swept_recently"]

        # For a long setup: we want price to have swept sell-side liquidity
        # (equal lows / prior lows), then reversed → CHoCH bullish
        # For a short: swept buy-side (equal highs), then reversed → CHoCH bearish
        if not swept:
            return self._no_signal(symbol, direction, "No liquidity sweep in last 5 bars")

        choch = struct.get("last_choch")
        bias  = struct.get("bias", "neutral")
        if is_long and bias not in ("bullish", "neutral"):
            return self._no_signal(symbol, direction, f"CHoCH not bullish after sweep ({bias})")
        if not is_long and bias not in ("bearish", "neutral"):
            return self._no_signal(symbol, direction, f"CHoCH not bearish after sweep ({bias})")

        # ── STEP 4: FVG / OB Entry Zone ──────────────────────────────────────
        fvgs = _SMC.find_fair_value_gaps(df_h1)
        obs  = _SMC.find_order_blocks(df_h1)

        target_fvg = "bullish" if is_long else "bearish"
        active_fvgs = [f for f in fvgs if f["active"] and not f["filled"] and f["fvg_type"] == target_fvg]
        target_ob  = "bullish" if is_long else "bearish"
        active_obs  = [ob for ob in obs  if ob["active"] and ob["ob_type"]  == target_ob]

        if not active_fvgs and not active_obs:
            return self._no_signal(symbol, direction, "No active FVG or OB for entry")

        # Prefer FVG entry over OB; use midpoint
        atr   = self._atr(df_h1)
        price = float(df_h1["close"].iloc[-1])

        if active_fvgs:
            best = active_fvgs[0]
            entry = best["fvg_midpoint"]
            entry_source = f"FVG {best['fvg_bottom']:.2f}–{best['fvg_top']:.2f}"
        else:
            best = active_obs[0]
            entry = best["ob_level"]
            entry_source = f"OB {best['ob_low']:.2f}–{best['ob_high']:.2f}"

        # ── STEP 5: SL below swept level + TP at next liquidity ───────────────
        if is_long:
            # SL = below sell-side liquidity (the level that was swept)
            ssl = liq["sell_side_liquidity"]
            swept_level = ssl[0] if ssl else (entry - atr * 2)
            stop_loss = round(swept_level - atr * 0.5, 2)
        else:
            bsl = liq["buy_side_liquidity"]
            swept_level = bsl[0] if bsl else (entry + atr * 2)
            stop_loss = round(swept_level + atr * 0.5, 2)

        # TP1 = next liquidity cluster in direction; TP2 = 4× SL distance
        if is_long:
            bsl = liq["buy_side_liquidity"]
            tp1_target = bsl[0] if bsl else None
        else:
            ssl = liq["sell_side_liquidity"]
            tp1_target = ssl[0] if ssl else None

        dist = abs(entry - stop_loss)
        tp1  = round(tp1_target, 2) if tp1_target and abs(tp1_target - entry) >= dist else round(entry + dist * 2, 2) if is_long else round(entry - dist * 2, 2)
        tp2  = round(entry + dist * 4, 2) if is_long else round(entry - dist * 4, 2)

        # ── Score (0–10) ──────────────────────────────────────────────────────
        score = 0.0
        score += 2.0                                # D1 bias confirmed
        score += 2.0                                # H4 confirmation
        score += 2.0                                # liquidity sweep
        score += 1.5 if active_fvgs else 1.0       # FVG > OB for entry
        score += 1.0 if choch else 0.5             # CHoCH confirmation
        score += 1.5 if struct["structure"] in ("trending_up", "trending_down") else 0.5

        reasons = [
            f"D1 {d1_bias} (HTF bias confirmed)",
            h4_reason,
            "Liquidity swept — stop hunt complete",
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
                "d1_bias": d1_bias,
                "h4_ok": h4_ok,
                "swept": swept,
                "has_fvg": bool(active_fvgs),
                "has_ob": bool(active_obs),
            },
        )
