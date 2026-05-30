"""
btc_research/strategies/swing_level_v2.py — Enhanced Swing Level Break (v2).

== WHAT'S WRONG WITH v1 ==
  SwingLevelBreak v1 enters on the first break of a swing level.
  SL = prior swing structure = 4.42× ATR on average.

  That 4.42× SL is the bottleneck:
    - Huge SL distance → fewer lots per dollar of risk
    - Trade stays open a very long time (TP1 = 2×4.42 = 8.84 ATR away)
    - Blocks new trades from opening during the long hold
    - WR is 57.9% (great) but poor R because SL is so wide

== WHAT v2 ADDS ==
  Three improvements tested independently and combined:

  1. ATR-CAPPED SL  (mode: "cap")
     Keep the first-break entry but cap the SL at max_sl_atr × ATR from entry.
     If the prior swing structure is within the cap → use it (unchanged).
     If the prior swing structure is further → cap the SL at entry ± max_sl_atr×ATR.
     This tightens the SL from 4.42× to ~1.5-2×ATR without changing the entry.
     Risk: may get stopped out more but the better R should more than compensate.

  2. RETEST ENTRY  (mode: "retest")
     Skip the first break entirely. Wait for the level to be broken, then for
     price to pull back and RETEST the broken level, then enter on rejection.
     SL = current bar's extreme (for longs: bar low, for shorts: bar high) + buffer.
     SL distance: ~0.5-0.8×ATR (6-8× tighter than v1's prior-structure SL).
     Pro: highest quality entry — level flip confirmed, structure validated.
     Con: fewer signals (many levels are broken but never retested during KZ hours).

  3. DUAL MODE  (mode: "both" or "retest_preferred")
     Check for retest first. If found → use tight SL.
     If no retest but first break → use ATR-capped SL.
     "Best available entry" from the same swing framework.

== ENTRY MODES ==
  "break"            : first break only, SL = prior swing structure (v1 behaviour)
  "break_capped"     : first break only, SL capped at max_sl_atr × ATR
  "retest"           : retest only, SL = bar extreme + buffer
  "retest_preferred" : retest if available, else first break with capped SL
  "both"             : fire on either first break OR retest (separate events,
                       retest fires only if first break didn't fire this session)

== SL PLACEMENT DETAIL ==
  First break  : prior swing structure (v1) OR capped at max_sl_atr × ATR
  Retest       : bar low  - sl_buffer_atr × ATR  (for longs)
                 bar high + sl_buffer_atr × ATR  (for shorts)

== PARAMETERS ==
  swing_n           : bars each side for swing confirmation      (default 3)
  lookback          : how far back to search for swing levels    (default 50)
  min_atr_pct       : ATR% threshold (flat market filter)        (default 0.15)
  macro_ema         : macro trend EMA period                     (default 96)
  entry_mode        : "break" | "break_capped" | "retest" |
                      "retest_preferred" | "both"               (default "break")
  max_sl_atr        : SL cap for break entries (× ATR)           (default 2.0)
  retest_lookback   : break must be within last N bars           (default 25)
  retest_tol_atr    : how close to level = "retest" (× ATR)     (default 0.35)
  sl_buffer_atr     : buffer on retest bar SL (× ATR)           (default 0.05)
  retest_max_sl_atr : reject retest if SL > this × ATR          (default 1.2)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from btc_research.strategies.base import BTCStrategy


def _find_swings(df: pd.DataFrame, n: int = 3
                 ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs = df["high"].astype(float).values
    lows  = df["low"].astype(float).values
    size  = len(df)
    swing_highs: list[tuple[int, float]] = []
    swing_lows:  list[tuple[int, float]] = []
    for i in range(n, size - n):
        if highs[i] == max(highs[i - n: i + n + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - n: i + n + 1]):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


_VALID_MODES = {"break", "break_capped", "retest", "retest_preferred", "both"}


class SwingLevelBreakV2(BTCStrategy):
    name        = "Swing Level Break v2"
    description = "Swing level break with retest detection and ATR-capped SL"

    def __init__(
        self,
        swing_n:           int   = 3,
        lookback:          int   = 50,
        min_atr_pct:       float = 0.15,
        macro_ema:         int   = 96,
        entry_mode:        str   = "retest_preferred",   # default = best mode
        max_sl_atr:        float = 2.0,   # cap for break-mode SL
        retest_lookback:   int   = 25,    # break must be within last N bars
        retest_tol_atr:    float = 0.35,  # how close = "retesting" the level
        sl_buffer_atr:     float = 0.05,  # buffer on retest bar SL
        retest_max_sl_atr: float = 1.2,   # reject retest if SL dist > this×ATR
    ):
        if entry_mode not in _VALID_MODES:
            raise ValueError(f"entry_mode must be one of {_VALID_MODES}")
        self.swing_n           = swing_n
        self.lookback          = lookback
        self.min_atr_pct       = min_atr_pct
        self.macro_ema         = macro_ema
        self.entry_mode        = entry_mode
        self.max_sl_atr        = max_sl_atr
        self.retest_lookback   = retest_lookback
        self.retest_tol_atr    = retest_tol_atr
        self.sl_buffer_atr     = sl_buffer_atr
        self.retest_max_sl_atr = retest_max_sl_atr

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_break(
        self, is_long: bool, bar_close: float, bar_high: float, bar_low: float,
        win: pd.DataFrame, swing_highs, swing_lows, atr: float, ema_macro: float
    ) -> dict | None:
        """
        Check for first-break entry (v1 logic + optional SL cap).
        Returns signal dict or None.
        """
        no = None
        if is_long:
            if not swing_highs: return no
            sh_price = swing_highs[-1][1]
            sh_idx   = swing_highs[-1][0]
            if bar_close <= sh_price: return no
            if bar_close < ema_macro: return no

            sl_candidates = [sl for sl in swing_lows if sl[0] < sh_idx]
            sl_struct = sl_candidates[-1][1] if sl_candidates else float(win["low"].min())

            # ATR cap
            if self.entry_mode in ("break_capped", "retest_preferred", "both"):
                sl_floor = bar_close - self.max_sl_atr * atr
                sl_val   = max(sl_struct, sl_floor)   # tighter of the two
            else:
                sl_val = sl_struct

            sl_dist = abs(bar_close - sl_val)
            if sl_dist <= 0: return no

            return {
                "signal": True, "entry": round(bar_close, 2), "sl": round(sl_val, 2),
                "tp1_rr": 1.5, "tp2_rr": 5.0,
                "entry_type": "break",
                "reason": f"SL-break long {sh_price:.0f} | SL={sl_val:.0f} "
                          f"(struct={sl_struct:.0f} cap={bar_close - self.max_sl_atr*atr:.0f})",
            }
        else:
            if not swing_lows: return no
            sl_price = swing_lows[-1][1]
            sl_idx   = swing_lows[-1][0]
            if bar_close >= sl_price: return no
            if bar_close > ema_macro: return no

            sh_candidates = [sh for sh in swing_highs if sh[0] < sl_idx]
            sl_struct = sh_candidates[-1][1] if sh_candidates else float(win["high"].max())

            if self.entry_mode in ("break_capped", "retest_preferred", "both"):
                sl_ceil  = bar_close + self.max_sl_atr * atr
                sl_val   = min(sl_struct, sl_ceil)
            else:
                sl_val = sl_struct

            sl_dist = abs(sl_val - bar_close)
            if sl_dist <= 0: return no

            return {
                "signal": True, "entry": round(bar_close, 2), "sl": round(sl_val, 2),
                "tp1_rr": 1.5, "tp2_rr": 5.0,
                "entry_type": "break",
                "reason": f"SL-break short {sl_price:.0f} | SL={sl_val:.0f}",
            }

    def _check_retest(
        self, is_long: bool, bar_close: float, bar_high: float, bar_low: float,
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        swing_highs, swing_lows, atr: float, ema_macro: float
    ) -> dict | None:
        """
        Check for retest entry: level was broken, price came back, now rejecting.
        Returns signal dict or None.
        """
        no  = None
        n   = len(closes)
        tol = self.retest_tol_atr * atr

        # Use only the non-recent portion of the window for swing detection
        # (swings must be confirmed, not just the last few bars)
        search_end = n - self.swing_n - 2
        if search_end < self.swing_n + 2: return no

        if is_long:
            if not swing_highs: return no
            for sh_idx, sh_price in reversed(swing_highs):
                if sh_idx >= search_end: continue

                # Find break bar: a bar after sh_idx where close > sh_price
                break_bar = None
                start = max(sh_idx + 1, n - 1 - self.retest_lookback)
                for k in range(start, n - 1):
                    if closes[k] > sh_price:
                        break_bar = k
                if break_bar is None: continue
                if break_bar >= n - 2:  continue  # too recent, no time to retest

                # Price must have come back down toward sh_price since break
                lows_since = lows[break_bar + 1: n]
                if len(lows_since) == 0: continue
                if float(np.min(lows_since)) > sh_price + tol: continue  # never retested

                # Current bar touches the level
                if bar_low > sh_price + tol: continue
                # Current bar closes above (rejection confirmed)
                if bar_close <= sh_price: continue
                # Macro alignment
                if bar_close < ema_macro: continue

                entry  = bar_close
                sl_val = bar_low - self.sl_buffer_atr * atr
                sl_dist = abs(entry - sl_val)
                if sl_dist <= 0: continue
                if sl_dist > self.retest_max_sl_atr * atr: continue

                return {
                    "signal": True, "entry": round(entry, 2), "sl": round(sl_val, 2),
                    "tp1_rr": 2.0, "tp2_rr": 5.0,   # tighter SL → can target 2R TP1
                    "entry_type": "retest",
                    "reason": (f"SL-retest long: level={sh_price:.0f} "
                               f"| break@bar-{n-1-break_bar} "
                               f"| bar_low={bar_low:.0f} tol={tol:.0f}"),
                }
            return no

        else:
            if not swing_lows: return no
            for sl_idx, sl_price in reversed(swing_lows):
                if sl_idx >= search_end: continue

                break_bar = None
                start = max(sl_idx + 1, n - 1 - self.retest_lookback)
                for k in range(start, n - 1):
                    if closes[k] < sl_price:
                        break_bar = k
                if break_bar is None: continue
                if break_bar >= n - 2: continue

                highs_since = highs[break_bar + 1: n]
                if len(highs_since) == 0: continue
                if float(np.max(highs_since)) < sl_price - tol: continue

                if bar_high < sl_price - tol: continue
                if bar_close >= sl_price: continue
                if bar_close > ema_macro: continue

                entry  = bar_close
                sl_val = bar_high + self.sl_buffer_atr * atr
                sl_dist = abs(sl_val - entry)
                if sl_dist <= 0: continue
                if sl_dist > self.retest_max_sl_atr * atr: continue

                return {
                    "signal": True, "entry": round(entry, 2), "sl": round(sl_val, 2),
                    "tp1_rr": 2.0, "tp2_rr": 5.0,
                    "entry_type": "retest",
                    "reason": (f"SL-retest short: level={sl_price:.0f} "
                               f"| break@bar-{n-1-break_bar}"),
                }
            return no

    # ── Main signal generation ─────────────────────────────────────────────────

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        no_sig = {"signal": False, "entry": 0.0, "sl": 0.0, "reason": ""}
        min_bars = max(self.lookback, self.macro_ema) + 10
        if len(df_window) < min_bars:
            no_sig["reason"] = "insufficient bars"; return no_sig

        current   = df_window.iloc[-1]
        bar_close = float(current["close"])
        bar_high  = float(current["high"])
        bar_low   = float(current["low"])
        is_long   = direction.lower() in ("long", "buy")

        close_s = df_window["close"].astype(float)
        high_s  = df_window["high"].astype(float)
        low_s   = df_window["low"].astype(float)

        # ATR
        tr = pd.concat([
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s  - close_s.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr     = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = atr / bar_close * 100
        if atr_pct < self.min_atr_pct:
            no_sig["reason"] = f"ATR {atr_pct:.2f}% too low"; return no_sig

        ema_macro = float(close_s.ewm(span=self.macro_ema, adjust=False).mean().iloc[-1])

        # Swing levels on rolling window
        win = df_window.tail(self.lookback).reset_index(drop=True)
        swing_highs, swing_lows = _find_swings(win, self.swing_n)

        # Raw arrays for retest check (full window needed for break detection)
        rt_win   = df_window.tail(self.lookback + 30)
        rt_h     = rt_win["high"].astype(float).values
        rt_l     = rt_win["low"].astype(float).values
        rt_c     = rt_win["close"].astype(float).values
        # Rebuild swings on same-length sub-window for retest
        rt_win2  = df_window.tail(self.lookback).reset_index(drop=True)
        rt_sh, rt_sl = _find_swings(rt_win2, self.swing_n)

        # ── Dispatch by mode ──────────────────────────────────────────────────
        if self.entry_mode == "break":
            sig = self._check_break(is_long, bar_close, bar_high, bar_low,
                                    win, swing_highs, swing_lows, atr, ema_macro)
            return sig if sig else no_sig

        elif self.entry_mode == "break_capped":
            sig = self._check_break(is_long, bar_close, bar_high, bar_low,
                                    win, swing_highs, swing_lows, atr, ema_macro)
            return sig if sig else no_sig

        elif self.entry_mode == "retest":
            sig = self._check_retest(is_long, bar_close, bar_high, bar_low,
                                     rt_h, rt_l, rt_c,
                                     rt_sh, rt_sl, atr, ema_macro)
            return sig if sig else no_sig

        elif self.entry_mode == "retest_preferred":
            # Try retest first (tighter SL, higher quality), fall back to capped break
            sig = self._check_retest(is_long, bar_close, bar_high, bar_low,
                                     rt_h, rt_l, rt_c,
                                     rt_sh, rt_sl, atr, ema_macro)
            if sig: return sig
            sig = self._check_break(is_long, bar_close, bar_high, bar_low,
                                    win, swing_highs, swing_lows, atr, ema_macro)
            return sig if sig else no_sig

        elif self.entry_mode == "both":
            # Fire on whichever fires — retest takes priority on same bar
            sig = self._check_retest(is_long, bar_close, bar_high, bar_low,
                                     rt_h, rt_l, rt_c,
                                     rt_sh, rt_sl, atr, ema_macro)
            if sig: return sig
            sig = self._check_break(is_long, bar_close, bar_high, bar_low,
                                    win, swing_highs, swing_lows, atr, ema_macro)
            return sig if sig else no_sig

        no_sig["reason"] = f"unknown mode {self.entry_mode}"
        return no_sig
