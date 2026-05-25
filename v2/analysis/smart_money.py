"""
smart_money.py
──────────────
Smart Money Concepts (SMC) — the framework institutional traders use.

Adds four tools retail bots miss entirely:
  1. Order Blocks   — last opposing candle before an explosive move
  2. Fair Value Gaps — price inefficiency gaps price returns to fill
  3. Liquidity Levels — equal high/low clusters where stops pool
  4. Market Structure — BOS (continuation) and CHoCH (reversal signal)

Combined into smc_score() for use in confluence_engine.py.

Usage:
    from smart_money import SmartMoneyAnalyzer
    sma   = SmartMoneyAnalyzer()
    score = sma.smc_score(df, "long")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
OB_PROXIMITY_PCT    = 0.005   # 0.5% — order block is "active" if price within this
FVG_LOOKBACK        = 50      # only flag FVGs from last N candles
EQ_LEVEL_TOLERANCE  = 0.001   # 0.1% — "equal" high or low within this band
STRONG_MOVE_MULT    = 1.5     # candle must be > ATR * this to count as strong move
SWING_LOOKBACK      = 20      # bars to scan for swing highs / lows in structure


# ══════════════════════════════════════════════════════════════════════════════
#  SmartMoneyAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class SmartMoneyAnalyzer:
    """
    Detects Smart Money Concepts on a standard OHLCV + indicator DataFrame.

    Expected columns: open, high, low, close, (atr optional — calculated if absent)
    """

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with guaranteed open/high/low/close columns."""
        df = df.copy()
        if "open" not in df.columns:
            df["open"] = df["close"].shift(1).fillna(df["close"])
        for col in ("high", "low", "close", "open"):
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")
        return df

    def _atr_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Return ATR series (uses existing column if present)."""
        if "atr" in df.columns:
            return df["atr"].bfill()
        hl  = df["high"] - df["low"]
        hcp = (df["high"] - df["close"].shift()).abs()
        lcp = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
        return tr.rolling(period).mean().fillna(tr)

    def _candle_body(self, df: pd.DataFrame) -> pd.Series:
        return (df["close"] - df["open"]).abs()

    # ══════════════════════════════════════════════════════════════════════════
    #  1. Order Blocks
    # ══════════════════════════════════════════════════════════════════════════

    def find_order_blocks(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Identify order blocks: the last opposing candle before a strong impulsive move.

        Bearish OB  — last *bullish* candle before a strong bearish impulse.
        Bullish OB  — last *bearish* candle before a strong bullish impulse.

        Returns list of dicts:
            ob_type     : "bullish" | "bearish"
            ob_level    : midpoint of the candle (reference price)
            ob_high     : candle high
            ob_low      : candle low
            ob_strength : "strong" | "weak"  (based on impulse size vs ATR)
            untested    : True if price has not re-entered the OB range since creation
            active      : True if current price is within OB_PROXIMITY_PCT of ob_level
            bar_index   : position in df
        """
        df      = self._ensure_ohlc(df)
        atr     = self._atr_series(df)
        price   = float(df["close"].iloc[-1])
        results = []

        # Need at least 5 bars
        if len(df) < 5:
            return results

        # Scan all bars (skip first 2 and last 1 — need context before and after)
        for i in range(2, len(df) - 1):
            curr_open  = float(df["open"].iloc[i])
            curr_close = float(df["close"].iloc[i])
            curr_high  = float(df["high"].iloc[i])
            curr_low   = float(df["low"].iloc[i])
            curr_atr   = float(atr.iloc[i]) if not np.isnan(float(atr.iloc[i])) else 1.0

            # Measure the impulse starting from the NEXT bar
            # Look forward up to 3 bars for a strong move
            for look in range(1, 4):
                if i + look >= len(df):
                    break

                future_close = float(df["close"].iloc[i + look])
                future_open  = float(df["open"].iloc[i + look])
                impulse_size = abs(future_close - curr_close)

                if impulse_size < curr_atr * STRONG_MOVE_MULT:
                    continue  # not a strong impulsive move

                # ── Bearish OB: curr candle is bullish, next move is bearish ──
                if (curr_close > curr_open                 # curr candle is up
                        and future_close < curr_close      # next move goes down
                        and (future_close - future_open) < 0):  # bearish impulse bar

                    ob_level = (curr_high + curr_low) / 2
                    strength = "strong" if impulse_size > curr_atr * 2.5 else "weak"

                    # Check untested: has price returned inside OB range after i?
                    future_slice = df.iloc[i + 1:]
                    re_entered   = (
                        (future_slice["low"] <= curr_high) &
                        (future_slice["high"] >= curr_low)
                    ).any()

                    results.append({
                        "ob_type":   "bearish",
                        "ob_level":  round(ob_level, 2),
                        "ob_high":   round(curr_high, 2),
                        "ob_low":    round(curr_low, 2),
                        "ob_strength": strength,
                        "untested":  not re_entered,
                        "active":    abs(price - ob_level) / max(ob_level, 1e-9) <= OB_PROXIMITY_PCT,
                        "bar_index": i,
                    })
                    break  # one OB per starting candle

                # ── Bullish OB: curr candle is bearish, next move is bullish ──
                elif (curr_close < curr_open                # curr candle is down
                        and future_close > curr_close       # next move goes up
                        and (future_close - future_open) > 0):  # bullish impulse bar

                    ob_level = (curr_high + curr_low) / 2
                    strength = "strong" if impulse_size > curr_atr * 2.5 else "weak"

                    future_slice = df.iloc[i + 1:]
                    re_entered   = (
                        (future_slice["low"] <= curr_high) &
                        (future_slice["high"] >= curr_low)
                    ).any()

                    results.append({
                        "ob_type":   "bullish",
                        "ob_level":  round(ob_level, 2),
                        "ob_high":   round(curr_high, 2),
                        "ob_low":    round(curr_low, 2),
                        "ob_strength": strength,
                        "untested":  not re_entered,
                        "active":    abs(price - ob_level) / max(ob_level, 1e-9) <= OB_PROXIMITY_PCT,
                        "bar_index": i,
                    })
                    break

        # Return most recent first, prioritise active + untested + strong
        results.sort(
            key=lambda x: (
                x["active"],
                x["untested"],
                x["ob_strength"] == "strong",
                x["bar_index"],
            ),
            reverse=True,
        )
        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  2. Fair Value Gaps
    # ══════════════════════════════════════════════════════════════════════════

    def find_fair_value_gaps(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Detect Fair Value Gaps (FVGs) — price inefficiency zones.

        Bullish FVG : candle[i-2].high < candle[i].low   (gap above)
        Bearish FVG : candle[i-2].low  > candle[i].high  (gap below)

        Only checks the last FVG_LOOKBACK candles.
        Marks filled=True if price has closed inside the gap after creation.

        Returns list of dicts:
            fvg_type    : "bullish" | "bearish"
            fvg_top     : upper boundary of gap
            fvg_bottom  : lower boundary of gap
            fvg_midpoint: middle of gap
            filled      : True if subsequently closed inside gap
            active      : True if price is currently in or very near the gap
            bar_index   : centre candle index (i-1)
        """
        df      = self._ensure_ohlc(df)
        price   = float(df["close"].iloc[-1])
        n       = len(df)
        start   = max(2, n - FVG_LOOKBACK)
        results = []

        for i in range(start, n):
            prev2_high = float(df["high"].iloc[i - 2])
            prev2_low  = float(df["low"].iloc[i - 2])
            curr_high  = float(df["high"].iloc[i])
            curr_low   = float(df["low"].iloc[i])

            # ── Bullish FVG: gap between candle[i-2] high and candle[i] low ──
            if prev2_high < curr_low:
                top    = round(curr_low,   2)
                bottom = round(prev2_high, 2)
                mid    = round((top + bottom) / 2, 2)
                gap    = top - bottom

                if gap <= 0:
                    continue

                # Check if filled — any subsequent close inside gap
                future = df.iloc[i + 1:] if i + 1 < n else df.iloc[0:0]
                filled = ((future["close"] >= bottom) & (future["close"] <= top)).any()

                near   = abs(price - mid) / max(mid, 1e-9) <= OB_PROXIMITY_PCT * 3
                inside = bottom <= price <= top

                results.append({
                    "fvg_type":    "bullish",
                    "fvg_top":     top,
                    "fvg_bottom":  bottom,
                    "fvg_midpoint": mid,
                    "filled":      bool(filled),
                    "active":      inside or near,
                    "bar_index":   i - 1,
                })

            # ── Bearish FVG: gap between candle[i-2] low and candle[i] high ──
            elif prev2_low > curr_high:
                top    = round(prev2_low,  2)
                bottom = round(curr_high,  2)
                mid    = round((top + bottom) / 2, 2)
                gap    = top - bottom

                if gap <= 0:
                    continue

                future = df.iloc[i + 1:] if i + 1 < n else df.iloc[0:0]
                filled = ((future["close"] >= bottom) & (future["close"] <= top)).any()

                near   = abs(price - mid) / max(mid, 1e-9) <= OB_PROXIMITY_PCT * 3
                inside = bottom <= price <= top

                results.append({
                    "fvg_type":    "bearish",
                    "fvg_top":     top,
                    "fvg_bottom":  bottom,
                    "fvg_midpoint": mid,
                    "filled":      bool(filled),
                    "active":      inside or near,
                    "bar_index":   i - 1,
                })

        # Most recent first, unfilled active gaps first
        results.sort(key=lambda x: (x["active"], not x["filled"], x["bar_index"]), reverse=True)
        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  3. Liquidity Levels
    # ══════════════════════════════════════════════════════════════════════════

    def find_liquidity_levels(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Identify where retail stop losses are clustered.

        Equal highs / equal lows: 2+ candles touching the same level ±0.1%.
        These are stop clusters — institutions hunt them before reversing.

        Also flags:
          prior_day_high / prior_day_low
          session_high   / session_low  (last 24 bars as proxy)

        Returns:
            buy_side_liquidity  : list of price levels (above price — sell stops)
            sell_side_liquidity : list of price levels (below price — buy stops)
            swept_recently      : True if last 5 bars broke a key level then reversed
        """
        df    = self._ensure_ohlc(df)
        price = float(df["close"].iloc[-1])
        highs = df["high"].values.tolist()
        lows  = df["low"].values.tolist()
        n     = len(df)

        buy_liq  = []   # above current price (old highs = sell side liquidity)
        sell_liq = []   # below current price (old lows  = buy side liquidity)

        # ── Equal highs / lows ────────────────────────────────────────────────
        # Cluster highs and lows that are within EQ_LEVEL_TOLERANCE of each other
        def _cluster_levels(levels: list[float], n_min: int = 2) -> list[float]:
            """Group nearby levels; return representative level for clusters ≥ n_min."""
            sorted_lvls = sorted(set(round(v, 1) for v in levels))
            clusters    = []
            used        = [False] * len(sorted_lvls)
            for idx, base in enumerate(sorted_lvls):
                if used[idx]:
                    continue
                grp = [base]
                for j in range(idx + 1, len(sorted_lvls)):
                    if abs(sorted_lvls[j] - base) / max(base, 1e-9) <= EQ_LEVEL_TOLERANCE:
                        grp.append(sorted_lvls[j])
                        used[j] = True
                if len(grp) >= n_min:
                    clusters.append(round(sum(grp) / len(grp), 2))
            return clusters

        eq_highs = _cluster_levels(highs[-100:], n_min=2)
        eq_lows  = _cluster_levels(lows[-100:],  n_min=2)

        for lvl in eq_highs:
            if lvl > price * (1 + EQ_LEVEL_TOLERANCE):
                buy_liq.append(lvl)
        for lvl in eq_lows:
            if lvl < price * (1 - EQ_LEVEL_TOLERANCE):
                sell_liq.append(lvl)

        # ── Prior day high / low (approx last 24 bars) ────────────────────────
        day_slice = df.iloc[max(0, n - 24): n - 1]
        if len(day_slice) > 0:
            pd_high = round(float(day_slice["high"].max()), 2)
            pd_low  = round(float(day_slice["low"].min()),  2)
            if pd_high > price and pd_high not in buy_liq:
                buy_liq.append(pd_high)
            if pd_low < price and pd_low not in sell_liq:
                sell_liq.append(pd_low)

        # ── Session high / low (last 8 bars as intraday proxy) ────────────────
        sess_slice = df.iloc[max(0, n - 8):]
        s_high = round(float(sess_slice["high"].max()), 2)
        s_low  = round(float(sess_slice["low"].min()),  2)
        if s_high > price and s_high not in buy_liq:
            buy_liq.append(s_high)
        if s_low < price and s_low not in sell_liq:
            sell_liq.append(s_low)

        # ── Swept recently ────────────────────────────────────────────────────
        # A "sweep" = last 5 bars broke a liquidity level then closed back inside it
        swept = False
        recent = df.iloc[-5:]
        r_high = float(recent["high"].max())
        r_low  = float(recent["low"].min())
        r_close_last = float(df["close"].iloc[-1])

        for lvl in buy_liq:
            if r_high >= lvl and r_close_last < lvl:
                swept = True
                break
        if not swept:
            for lvl in sell_liq:
                if r_low <= lvl and r_close_last > lvl:
                    swept = True
                    break

        return {
            "buy_side_liquidity":  sorted(set(buy_liq),  reverse=True)[:6],
            "sell_side_liquidity": sorted(set(sell_liq))[:6],
            "swept_recently":      swept,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  4. Market Structure
    # ══════════════════════════════════════════════════════════════════════════

    def detect_market_structure(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Determine market structure using swing highs/lows.

        Break of Structure (BOS):
          Uptrend  — price forms higher high  → BOS bullish (continuation)
          Downtrend — price forms lower low   → BOS bearish (continuation)

        Change of Character (CHoCH):
          Uptrend fails to make new high AND breaks below last swing low → bearish CHoCH
          Downtrend fails to make new low AND breaks above last swing high → bullish CHoCH

        Returns:
            structure     : "trending_up" | "trending_down" | "ranging"
            last_bos      : price of last Break of Structure (or None)
            last_choch    : price of last Change of Character (or None)
            bias          : "bullish" | "bearish" | "neutral"
            detail        : plain English explanation
            swing_highs   : list of recent swing high prices
            swing_lows    : list of recent swing low prices
        """
        df    = self._ensure_ohlc(df)
        n     = len(df)
        lb    = min(SWING_LOOKBACK, n // 3)

        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values

        # ── Find swing highs and lows ─────────────────────────────────────────
        swing_highs = []
        swing_lows  = []
        window      = max(2, lb // 4)

        for i in range(window, n - window):
            local_high = highs[i - window: i + window + 1]
            local_low  = lows[i - window: i + window + 1]
            if highs[i] == max(local_high):
                swing_highs.append((i, round(float(highs[i]), 2)))
            if lows[i] == min(local_low):
                swing_lows.append((i, round(float(lows[i]), 2)))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {
                "structure":   "ranging",
                "last_bos":    None,
                "last_choch":  None,
                "bias":        "neutral",
                "detail":      "Insufficient swing data to determine structure",
                "swing_highs": [v for _, v in swing_highs[-4:]],
                "swing_lows":  [v for _, v in swing_lows[-4:]],
            }

        # Last 2 swing highs / lows
        sh_vals = [v for _, v in swing_highs[-4:]]
        sl_vals = [v for _, v in swing_lows[-4:]]

        last_sh  = sh_vals[-1]
        prev_sh  = sh_vals[-2] if len(sh_vals) >= 2 else last_sh
        last_sl  = sl_vals[-1]
        prev_sl  = sl_vals[-2] if len(sl_vals) >= 2 else last_sl

        price    = float(closes[-1])

        hh = last_sh > prev_sh   # higher high
        lh = last_sh < prev_sh   # lower high
        hl = last_sl > prev_sl   # higher low
        ll = last_sl < prev_sl   # lower low

        # ── Determine structure ───────────────────────────────────────────────
        last_bos   = None
        last_choch = None
        bias       = "neutral"
        structure  = "ranging"
        detail     = ""

        if hh and hl:
            structure = "trending_up"
            last_bos  = last_sh
            bias      = "bullish"
            detail    = f"Higher High ({last_sh}) + Higher Low ({last_sl}) — Uptrend BOS confirmed"

        elif ll and lh:
            structure = "trending_down"
            last_bos  = last_sl
            bias      = "bearish"
            detail    = f"Lower Low ({last_sl}) + Lower High ({last_sh}) — Downtrend BOS confirmed"

        elif hh and ll:
            structure  = "ranging"
            bias       = "neutral"
            detail     = f"HH ({last_sh}) but LL ({last_sl}) — expanding range, no clear bias"

        elif lh and hl:
            structure  = "ranging"
            bias       = "neutral"
            detail     = f"LH ({last_sh}) but HL ({last_sl}) — contracting range"

        # ── Check for CHoCH (early reversal) ─────────────────────────────────
        if structure == "trending_up" and price < last_sl:
            last_choch = last_sl
            bias       = "bearish"
            detail     += f" | ⚠️ CHoCH: broke below last HL ({last_sl}) — possible reversal"

        elif structure == "trending_down" and price > last_sh:
            last_choch = last_sh
            bias       = "bullish"
            detail     += f" | ⚠️ CHoCH: broke above last LH ({last_sh}) — possible reversal"

        return {
            "structure":   structure,
            "last_bos":    last_bos,
            "last_choch":  last_choch,
            "bias":        bias,
            "detail":      detail.strip(" |"),
            "swing_highs": sh_vals,
            "swing_lows":  sl_vals,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  5. SMC Score
    # ══════════════════════════════════════════════════════════════════════════

    def smc_score(self, df: pd.DataFrame, direction: str) -> dict[str, Any]:
        """
        Combined SMC score for use in confluence_engine.py as a 7th check.

        Scores one point each (0–4 total):
          1. active_order_block  — price is at an OB aligned with direction
          2. fvg_nearby          — unfilled FVG nearby aligned with direction
          3. liquidity_swept     — recent liquidity sweep (trap + reversal)
          4. structure_aligned   — BOS or CHoCH confirms direction

        Parameters
        ----------
        df        : enriched OHLCV DataFrame
        direction : "long" | "short"

        Returns
        -------
        score              : int 0–4
        active_order_block : bool
        fvg_nearby         : bool
        liquidity_swept    : bool
        structure_aligned  : bool
        summary            : plain English explanation
        ob_detail          : str — closest active OB description
        fvg_detail         : str — closest active FVG description
        structure_detail   : str — structure description
        liquidity_detail   : str — liquidity description
        """
        direction = direction.lower().strip()
        is_long   = direction in ("long", "buy")
        price     = float(df["close"].iloc[-1])

        notes = []

        # ── 1. Order Block ────────────────────────────────────────────────────
        obs          = self.find_order_blocks(df)
        target_type  = "bullish" if is_long else "bearish"
        active_obs   = [ob for ob in obs if ob["active"] and ob["ob_type"] == target_type]
        strong_obs   = [ob for ob in active_obs if ob["ob_strength"] == "strong"]
        best_ob      = (strong_obs or active_obs or [None])[0]

        active_order_block = best_ob is not None
        if best_ob:
            ob_detail = (
                f"{'Bullish' if is_long else 'Bearish'} OB at ${best_ob['ob_level']:,.2f} "
                f"[{best_ob['ob_low']:,.2f}–{best_ob['ob_high']:,.2f}] "
                f"({'untested' if best_ob['untested'] else 'tested'}, "
                f"{best_ob['ob_strength']})"
            )
            notes.append(f"✓ OB: {ob_detail}")
        else:
            ob_detail = f"No active {'bullish' if is_long else 'bearish'} OB near ${price:,.2f}"
            notes.append(f"✗ OB: {ob_detail}")

        # ── 2. Fair Value Gap ─────────────────────────────────────────────────
        fvgs        = self.find_fair_value_gaps(df)
        target_fvg  = "bullish" if is_long else "bearish"
        active_fvgs = [f for f in fvgs if f["active"] and not f["filled"] and f["fvg_type"] == target_fvg]
        best_fvg    = active_fvgs[0] if active_fvgs else None

        fvg_nearby = best_fvg is not None
        if best_fvg:
            fvg_detail = (
                f"{'Bullish' if is_long else 'Bearish'} FVG "
                f"${best_fvg['fvg_bottom']:,.2f}–${best_fvg['fvg_top']:,.2f} "
                f"(mid ${best_fvg['fvg_midpoint']:,.2f}, unfilled)"
            )
            notes.append(f"✓ FVG: {fvg_detail}")
        else:
            fvg_detail = f"No unfilled {'bullish' if is_long else 'bearish'} FVG nearby"
            notes.append(f"✗ FVG: {fvg_detail}")

        # ── 3. Liquidity Swept ────────────────────────────────────────────────
        liq               = self.find_liquidity_levels(df)
        liquidity_swept   = liq["swept_recently"]
        buy_liq           = liq["buy_side_liquidity"]
        sell_liq          = liq["sell_side_liquidity"]

        if liquidity_swept:
            liq_detail = "Liquidity swept recently — trap + reversal expected"
            notes.append(f"✓ Liq: {liq_detail}")
        else:
            nearest_buy  = buy_liq[0]  if buy_liq  else None
            nearest_sell = sell_liq[0] if sell_liq else None
            targets = []
            if nearest_buy:
                targets.append(f"BSL target ${nearest_buy:,.2f}")
            if nearest_sell:
                targets.append(f"SSL target ${nearest_sell:,.2f}")
            liq_detail = "No recent sweep — " + (", ".join(targets) or "no key levels nearby")
            notes.append(f"✗ Liq: {liq_detail}")

        # ── 4. Structure Aligned ──────────────────────────────────────────────
        ms                = self.detect_market_structure(df)
        ms_bias           = ms["bias"]
        ms_has_choch      = ms["last_choch"] is not None

        # For a long: bias should be bullish OR CHoCH just triggered bullish
        # For a short: bias should be bearish OR CHoCH just triggered bearish
        if is_long:
            structure_aligned = ms_bias == "bullish"
        else:
            structure_aligned = ms_bias == "bearish"

        structure_detail = ms["detail"] or f"Structure: {ms['structure']} | Bias: {ms_bias}"
        if structure_aligned:
            notes.append(f"✓ Structure: {structure_detail}")
        else:
            notes.append(f"✗ Structure: {structure_detail} (need {'bullish' if is_long else 'bearish'})")

        # ── Final score ───────────────────────────────────────────────────────
        score = sum([
            active_order_block,
            fvg_nearby,
            liquidity_swept,
            structure_aligned,
        ])

        quality = {4: "INSTITUTIONAL", 3: "HIGH", 2: "MODERATE", 1: "LOW", 0: "NONE"}
        summary = (
            f"SMC {quality.get(score, 'NONE')} quality ({score}/4) | "
            + " | ".join(notes)
        )

        return {
            "score":               score,
            "active_order_block":  active_order_block,
            "fvg_nearby":          fvg_nearby,
            "liquidity_swept":     liquidity_swept,
            "structure_aligned":   structure_aligned,
            "summary":             summary,
            "ob_detail":           ob_detail,
            "fvg_detail":          fvg_detail,
            "structure_detail":    structure_detail,
            "liquidity_detail":    liq_detail,
            "order_blocks":        obs[:5],
            "fair_value_gaps":     fvgs[:5],
            "liquidity_levels":    liq,
            "market_structure":    ms,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  6. Breaker Blocks
    # ══════════════════════════════════════════════════════════════════════════

    def find_breaker_blocks(self, df: pd.DataFrame, direction: str = "long") -> list[dict[str, Any]]:
        """
        Find breaker blocks — failed order blocks that now act as support/resistance.

        A bullish breaker: bearish OB was broken upward → now support.
        A bearish breaker: bullish OB was broken downward → now resistance.

        Returns list of dicts:
            breaker_type  : "bullish" | "bearish"
            breaker_level : float (midpoint of the OB)
            breaker_high  : float
            breaker_low   : float
            active        : True if price within 0.5% of breaker_level
            bar_index     : int
        """
        df      = self._ensure_ohlc(df)
        price   = float(df["close"].iloc[-1])
        obs     = self.find_order_blocks(df)
        results = []

        for ob in obs:
            if ob["untested"]:
                continue   # only test OBs that price re-entered

            ob_high  = ob["ob_high"]
            ob_low   = ob["ob_low"]
            ob_level = ob["ob_level"]
            ob_type  = ob["ob_type"]
            bar_i    = ob["bar_index"]

            # Look at price action after the OB bar
            if bar_i + 1 >= len(df):
                continue
            after = df.iloc[bar_i + 1:]

            if ob_type == "bullish":
                # Bullish OB broken downward → bearish breaker
                if (after["close"] < ob_low).any():
                    results.append({
                        "breaker_type":  "bearish",
                        "breaker_level": round(ob_level, 2),
                        "breaker_high":  round(ob_high, 2),
                        "breaker_low":   round(ob_low, 2),
                        "active":        abs(price - ob_level) / max(ob_level, 1e-9) <= 0.005,
                        "bar_index":     bar_i,
                    })
            elif ob_type == "bearish":
                # Bearish OB broken upward → bullish breaker
                if (after["close"] > ob_high).any():
                    results.append({
                        "breaker_type":  "bullish",
                        "breaker_level": round(ob_level, 2),
                        "breaker_high":  round(ob_high, 2),
                        "breaker_low":   round(ob_low, 2),
                        "active":        abs(price - ob_level) / max(ob_level, 1e-9) <= 0.005,
                        "bar_index":     bar_i,
                    })

        # Most recent active breakers first
        results.sort(key=lambda x: (x["active"], x["bar_index"]), reverse=True)
        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  7. Premium / Discount Zones
    # ══════════════════════════════════════════════════════════════════════════

    def find_premium_discount_zones(self, df: pd.DataFrame, lookback: int = 100) -> dict[str, Any]:
        """
        Identify premium and discount zones using the Fibonacci 50% equilibrium.

        Scans last `lookback` bars:
          equilibrium   = (highest_high + lowest_low) / 2
          premium_zone  = above equilibrium (institutional selling area)
          discount_zone = below equilibrium (institutional buying area)

        Returns:
            highest_high  : float
            lowest_low    : float
            equilibrium   : float
            current_price : float
            current_zone  : "premium" | "discount" | "equilibrium"
            premium_start : float   (= equilibrium)
            discount_end  : float   (= equilibrium)
            zone_bias     : "long" if discount, "short" if premium, "neutral"
            fib_levels    : dict of fib retracements on the full range
        """
        df    = self._ensure_ohlc(df)
        n     = len(df)
        slice_ = df.iloc[max(0, n - lookback):]

        highest = round(float(slice_["high"].max()), 2)
        lowest  = round(float(slice_["low"].min()),  2)
        rng     = highest - lowest
        equil   = round((highest + lowest) / 2, 2)
        price   = round(float(df["close"].iloc[-1]), 2)

        eq_band = rng * 0.005   # ±0.5% of range = equilibrium band

        if abs(price - equil) <= eq_band:
            zone = "equilibrium"
            bias = "neutral"
        elif price > equil:
            zone = "premium"
            bias = "short"
        else:
            zone = "discount"
            bias = "long"

        fib_levels: dict[str, float] = {}
        if rng > 0:
            for level, pct in [("0.236", 0.236), ("0.382", 0.382), ("0.5", 0.5),
                                ("0.618", 0.618), ("0.786", 0.786)]:
                fib_levels[level] = round(highest - rng * pct, 2)

        return {
            "highest_high":  highest,
            "lowest_low":    lowest,
            "equilibrium":   equil,
            "current_price": price,
            "current_zone":  zone,
            "premium_start": equil,
            "discount_end":  equil,
            "zone_bias":     bias,
            "fib_levels":    fib_levels,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  8. Master SMC Context
    # ══════════════════════════════════════════════════════════════════════════

    def get_smc_context(self, df: pd.DataFrame, direction: str) -> dict[str, Any]:
        """
        Master function combining ALL SMC signals into one context dict.

        Entry quality:
          A: OB active + FVG nearby + structure aligned + zone aligned
          B: any 2 of (OB active, FVG nearby, structure aligned)
          C: any 1 of the above
          D: none

        confidence_adjustment:
          Grade A → +1.0
          Grade B → +0.5
          Grade C →  0.0
          Grade D → -0.5

        Returns full context dict including all sub-results.
        """
        direction = direction.lower().strip()
        is_long   = direction in ("long", "buy")

        # ── Run all sub-functions ─────────────────────────────────────────────
        base        = self.smc_score(df, direction)
        breakers    = self.find_breaker_blocks(df, direction)
        pd_zones    = self.find_premium_discount_zones(df)

        ob_active      = base["active_order_block"]
        fvg_nearby     = base["fvg_nearby"]
        str_aligned    = base["structure_aligned"]
        zone_bias      = pd_zones.get("zone_bias", "neutral")
        zone_aligned   = (is_long and zone_bias == "long") or \
                         (not is_long and zone_bias == "short")

        # Active breakers aligned with direction
        target_breaker = "bullish" if is_long else "bearish"
        active_breakers = [b for b in breakers if b["active"] and b["breaker_type"] == target_breaker]

        # ── Grade ─────────────────────────────────────────────────────────────
        criteria = [ob_active, fvg_nearby, str_aligned]
        met_count = sum(criteria)

        if met_count >= 3 or (met_count >= 2 and zone_aligned):
            grade = "A"
        elif met_count >= 2:
            grade = "B"
        elif met_count >= 1:
            grade = "C"
        else:
            grade = "D"

        adj_map = {"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5}
        conf_adj = adj_map[grade]

        # ── Reasons ───────────────────────────────────────────────────────────
        entry_reasons = []
        avoid_reasons = []

        if ob_active:
            entry_reasons.append(base["ob_detail"])
        else:
            avoid_reasons.append("No active order block aligned with direction")

        if fvg_nearby:
            entry_reasons.append(base["fvg_detail"])
        else:
            avoid_reasons.append("No unfilled FVG nearby")

        if str_aligned:
            entry_reasons.append(f"Structure {base['structure_detail'][:60]}")
        else:
            avoid_reasons.append(f"Structure not aligned — {base['structure_detail'][:60]}")

        if zone_aligned:
            entry_reasons.append(
                f"Price in {'discount' if is_long else 'premium'} zone "
                f"(eq=${pd_zones['equilibrium']:,.2f})"
            )
        else:
            avoid_reasons.append(
                f"Price in {pd_zones['current_zone']} zone — "
                f"{'avoid longs at premium' if is_long else 'avoid shorts at discount'}"
            )

        if active_breakers:
            entry_reasons.append(
                f"Breaker block at ${active_breakers[0]['breaker_level']:,.2f} "
                f"({active_breakers[0]['breaker_type']})"
            )

        # Entry quality label
        quality_labels = {
            "A": "A — OB + FVG + Structure + Zone aligned",
            "B": "B — 2 of 3 SMC criteria met",
            "C": "C — 1 of 3 SMC criteria met",
            "D": "D — no SMC confluence",
        }

        return {
            "smc_score":            base["score"],
            "order_blocks":         [ob for ob in base["order_blocks"] if ob["active"]],
            "fair_value_gaps":      [f for f in base["fair_value_gaps"] if f["active"] and not f["filled"]],
            "breaker_blocks":       active_breakers,
            "liquidity":            base["liquidity_levels"],
            "structure":            base["market_structure"],
            "premium_discount":     pd_zones,
            "zone_aligned":         zone_aligned,
            "entry_quality":        grade,
            "entry_quality_label":  quality_labels[grade],
            "entry_reasons":        entry_reasons,
            "avoid_reasons":        avoid_reasons,
            "confidence_adjustment": conf_adj,
            # detailed sub-results for display
            "ob_detail":            base["ob_detail"],
            "fvg_detail":           base["fvg_detail"],
            "structure_detail":     base["structure_detail"],
            "liquidity_detail":     base["liquidity_detail"],
            "active_order_block":   ob_active,
            "fvg_nearby":           fvg_nearby,
            "structure_aligned":    str_aligned,
            "liquidity_swept":      base["liquidity_swept"],
            "liquidity_map":        self.get_liquidity_map(df),
        }

    def get_liquidity_map(self, df: pd.DataFrame) -> dict:
        """
        Build a liquidity map for this dataframe using liquidity_map.py.
        Returns the map dict, or a minimal dict if unavailable.
        """
        try:
            from liquidity_map import build_liquidity_map as _blm
            price = float(df["close"].iloc[-1])
            return _blm(df, price)
        except Exception:
            return {"available": False, "clusters_above": [], "clusters_below": [],
                    "poc": 0.0, "va_high": 0.0, "va_low": 0.0, "voids": [],
                    "likely_move": "NEUTRAL", "likely_reason": "unavailable",
                    "largest_cluster": None}


# ══════════════════════════════════════════════════════════════════════════════
#  Quick print helper
# ══════════════════════════════════════════════════════════════════════════════

def print_smc_report(df: pd.DataFrame, direction: str = "long") -> None:
    """Print a formatted SMC report to stdout."""
    sma    = SmartMoneyAnalyzer()
    result = sma.smc_score(df, direction)
    w      = 68
    price  = float(df["close"].iloc[-1])

    print("╔" + "═" * w + "╗")
    print(f"║{'  SMART MONEY CONCEPTS ANALYSIS':^{w}}║")
    print(f"║{'  XAUUSD  |  Direction: ' + direction.upper() + '  |  Price: $' + f'{price:,.2f}':^{w}}║")
    print("╠" + "═" * w + "╣")

    score = result["score"]
    bar   = "█" * score + "░" * (4 - score)
    print(f"║  SMC Score : {score}/4  [{bar}]{'':>{w - 24}}║")
    print("╠" + "═" * w + "╣")

    labels = [
        ("Order Block",    result["active_order_block"],  result["ob_detail"]),
        ("FVG Nearby",     result["fvg_nearby"],          result["fvg_detail"]),
        ("Liq Swept",      result["liquidity_swept"],     result["liquidity_detail"]),
        ("Structure",      result["structure_aligned"],   result["structure_detail"]),
    ]
    for name, passed, detail in labels:
        icon   = "✓" if passed else "✗"
        detail = detail[:w - 18] + "…" if len(detail) > w - 18 else detail
        print(f"║  {icon} {name:<14} {detail:<{w - 18}}║")

    print("╠" + "═" * w + "╣")
    ms = result["market_structure"]
    print(f"║  Structure : {ms['structure'].replace('_', ' ').upper():<20} Bias: {ms['bias'].upper():<10}{'':>{w-50}}║")
    obs  = [ob for ob in result["order_blocks"] if ob["active"]]
    fvgs = [f  for f  in result["fair_value_gaps"] if f["active"] and not f["filled"]]
    print(f"║  Active OBs: {len(obs):<4}  Active FVGs: {len(fvgs):<4}{'':>{w-28}}║")
    print("╚" + "═" * w + "╝")


# ══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, sys

    DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "historical_xauusd.csv")
    try:
        df = pd.read_csv(DATA_CSV, index_col=0)
        df.columns = [c.lower() for c in df.columns]
        if "open" not in df.columns:
            df["open"] = df["close"].shift(1).fillna(df["close"])
        df = df.tail(200)
        print(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"CSV not found ({e}), using synthetic data")
        rng   = np.random.default_rng(42)
        n     = 200
        close = 3500 + np.cumsum(rng.normal(0, 5, n))
        df = pd.DataFrame({
            "open":   close - rng.uniform(0, 8, n),
            "high":   close + rng.uniform(2, 15, n),
            "low":    close - rng.uniform(2, 15, n),
            "close":  close,
            "volume": rng.integers(1000, 9000, n).astype(float),
        })

    sma = SmartMoneyAnalyzer()

    print("\n─── Order Blocks ───────────────────────────────────────")
    obs = sma.find_order_blocks(df)
    active_obs = [o for o in obs if o["active"]]
    print(f"Total found: {len(obs)}  |  Active: {len(active_obs)}")
    for ob in obs[:3]:
        print(f"  {ob['ob_type'].upper():8} OB @ ${ob['ob_level']:,.2f} "
              f"[{ob['ob_low']:,.2f}–{ob['ob_high']:,.2f}]  "
              f"{'untested' if ob['untested'] else 'tested':8}  "
              f"{'ACTIVE' if ob['active'] else '      '}")

    print("\n─── Fair Value Gaps ────────────────────────────────────")
    fvgs = sma.find_fair_value_gaps(df)
    unfilled = [f for f in fvgs if not f["filled"]]
    print(f"Total found: {len(fvgs)}  |  Unfilled: {len(unfilled)}")
    for f in fvgs[:3]:
        status = "filled" if f["filled"] else "OPEN  "
        print(f"  {f['fvg_type'].upper():8} FVG ${f['fvg_bottom']:,.2f}–${f['fvg_top']:,.2f} "
              f"mid ${f['fvg_midpoint']:,.2f}  {status}  "
              f"{'ACTIVE' if f['active'] else '      '}")

    print("\n─── Liquidity Levels ───────────────────────────────────")
    liq = sma.find_liquidity_levels(df)
    print(f"Buy-side (above):  {liq['buy_side_liquidity']}")
    print(f"Sell-side (below): {liq['sell_side_liquidity']}")
    print(f"Swept recently:    {liq['swept_recently']}")

    print("\n─── Market Structure ───────────────────────────────────")
    ms = sma.detect_market_structure(df)
    print(f"Structure : {ms['structure']}")
    print(f"Bias      : {ms['bias']}")
    print(f"Last BOS  : {ms['last_bos']}")
    print(f"Last CHoCH: {ms['last_choch']}")
    print(f"Detail    : {ms['detail']}")

    print()
    print_smc_report(df, "long")
    print()
    result = sma.smc_score(df, "short")
    print(f"\nShort SMC score: {result['score']}/4")
    print("Self-test complete ✓")
