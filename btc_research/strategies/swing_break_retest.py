"""
btc_research/strategies/swing_break_retest.py — Swing Break Retest (SBR) strategy.

== CONCEPT ==
  The images showed: "The first breakout is usually the trap.
  The second breakout after retest is usually the real move."

  This strategy waits for the CONFIRMATION step that the original Swing Level
  Break skips. Instead of entering on the first close beyond a swing level,
  we wait for:

    1. A swing level to be BROKEN  (price closes beyond it — happens a few bars ago)
    2. Price RETESTS that level    (pulls back toward it after the break)
    3. The retest REJECTS          (current bar confirms the level has flipped)

  This is the "Retest of Broken Swing Level" entry from smart-money playbooks:
    - For longs: old resistance → confirmed new support
    - For shorts: old support   → confirmed new resistance

== SL PLACEMENT ==
  Entry: bar close (retest rejection confirmed)
  SL   : just below the current bar's LOW (for longs)
          just above the current bar's HIGH (for shorts)

  WHY THIS SL IS TIGHTER:
    Original Swing Level Break SL = prior swing structure = 1-3× ATR away
    Swing Break Retest SL        = retest bar extreme    = 0.3-0.8× ATR away
    Result: 2-4× better R per trade at the same TP levels

== DETECTION LOGIC ==
  Window: last 80 bars

  For LONG setup:
    1. Find swing HIGH in bars [-80 to -5] (confirmed: n bars each side)
    2. Scan bars AFTER the swing high for a "break bar" — a bar that closed
       ABOVE the swing high level (break occurred)
    3. The break must have happened within the last `break_lookback` bars
    4. After the break, price must have come BACK DOWN to the level
       (current bar low ≤ swing_high + retest_tol × ATR)
    5. Current bar closes ABOVE the swing_high (rejection confirmed)
    → LONG signal

  For SHORT setup: mirror image

== PARAMETERS ==
  swing_n         : bars each side to confirm a swing high/low  (default 3)
  swing_lookback  : how far back to search for swing levels     (default 80)
  break_lookback  : break must be within last N bars            (default 25)
  retest_tol_atr  : how close to level = "retest" (× ATR)      (default 0.35)
  sl_buffer_atr   : SL buffer beyond retest bar extreme (× ATR) (default 0.05)
  max_sl_atr      : reject if SL distance > this × ATR          (default 1.2)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from btc_research.strategies.base import BTCStrategy


def _find_swings(highs: np.ndarray, lows: np.ndarray, n: int = 3
                 ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Local swing detection confirmed by N bars each side."""
    size = len(highs)
    swing_highs: list[tuple[int, float]] = []
    swing_lows:  list[tuple[int, float]] = []
    for i in range(n, size - n):
        if highs[i] == max(highs[i - n: i + n + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - n: i + n + 1]):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


class SwingBreakRetest(BTCStrategy):
    name        = "Swing Break Retest"
    description = ("Break of swing level confirmed by pullback retest  |  "
                   "tighter SL = 2-4× better R than first-break entry")

    def __init__(
        self,
        swing_n:        int   = 3,    # bars each side for swing confirmation
        swing_lookback: int   = 80,   # how far back to search for levels
        break_lookback: int   = 25,   # break must be within last N bars
        retest_tol_atr: float = 0.35, # price within this × ATR = "retesting"
        sl_buffer_atr:  float = 0.05, # extra buffer on SL beyond bar extreme
        max_sl_atr:     float = 1.2,  # reject if SL dist > this × ATR (too wide)
    ):
        self.swing_n        = swing_n
        self.swing_lookback = swing_lookback
        self.break_lookback = break_lookback
        self.retest_tol_atr = retest_tol_atr
        self.sl_buffer_atr  = sl_buffer_atr
        self.max_sl_atr     = max_sl_atr

    def generate_signal(
        self,
        df_window: pd.DataFrame,
        bar_time:  pd.Timestamp,
        direction: str,
    ) -> dict:
        no_sig = {"signal": False, "entry": 0.0, "sl": 0.0, "reason": ""}
        min_bars = self.swing_lookback + self.swing_n + 5
        if len(df_window) < min_bars:
            no_sig["reason"] = "insufficient bars"
            return no_sig

        # Use a rolling window to keep things fast
        win    = df_window.tail(self.swing_lookback + self.swing_n + 5)
        highs  = win["high"].astype(float).values
        lows   = win["low"].astype(float).values
        closes = win["close"].astype(float).values
        n_bars = len(win)

        curr_close = closes[-1]
        curr_high  = highs[-1]
        curr_low   = lows[-1]

        # ATR(14) on the full window
        high_s  = win["high"].astype(float)
        low_s   = win["low"].astype(float)
        close_s = win["close"].astype(float)
        tr = pd.concat([
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s  - close_s.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-2])
        if atr <= 0:
            no_sig["reason"] = "ATR zero"
            return no_sig

        tol = self.retest_tol_atr * atr

        # ── Find swings in the non-recent portion of the window ───────────────
        # Exclude last (swing_n+2) bars so swings are confirmed
        search_end = n_bars - self.swing_n - 2
        if search_end < self.swing_n + 2:
            no_sig["reason"] = "window too short for swing search"
            return no_sig

        swing_h, swing_l = _find_swings(
            highs[:search_end], lows[:search_end], self.swing_n
        )

        is_long = direction.lower() in ("long", "buy")

        if is_long:
            # ── LONG: look for swing high that was broken, now retesting ──────
            if not swing_h:
                no_sig["reason"] = "no swing highs found"
                return no_sig

            # Search most-recent swing highs first (reverse order)
            for sh_idx, sh_price in reversed(swing_h):
                # Step 1: find a "break bar" after the swing high where close > sh_price
                # The break must be within last break_lookback bars from current
                break_bar_idx = None
                search_start  = max(sh_idx + 1, n_bars - 1 - self.break_lookback)
                for k in range(search_start, n_bars - 1):  # exclude current bar
                    if closes[k] > sh_price:
                        break_bar_idx = k
                        # Don't break — we want the MOST RECENT break bar
                if break_bar_idx is None:
                    continue  # this swing high was never broken, try older one

                # Step 2: break must have happened before current bar
                # (not happening right now — we want to see the retest)
                if break_bar_idx >= n_bars - 2:
                    continue  # break is too recent (current or prev bar), need retest time

                # Step 3: since the break, price must have come back DOWN
                # toward the swing high level. Check min low since break bar.
                lows_since_break = lows[break_bar_idx + 1: n_bars]
                if len(lows_since_break) == 0:
                    continue
                min_low_since_break = float(np.min(lows_since_break))
                if min_low_since_break > sh_price + tol:
                    continue  # price never came back down to retest the level

                # Step 4: current bar is retesting — low is near the level
                if curr_low > sh_price + tol:
                    continue  # current bar didn't touch the level

                # Step 5: current bar CLOSES above the level (rejection confirmed)
                if curr_close <= sh_price:
                    continue  # close is still below — no rejection

                # ── Found a valid retest setup ─────────────────────────────────
                entry  = curr_close
                sl_val = curr_low - self.sl_buffer_atr * atr

                sl_dist = abs(entry - sl_val)
                if sl_dist <= 0:
                    continue
                if sl_dist > self.max_sl_atr * atr:
                    continue  # SL too wide

                return {
                    "signal": True,
                    "entry":  round(entry, 2),
                    "sl":     round(sl_val, 2),
                    "reason": (f"SBR long: retest of broken swing_H {sh_price:.0f} "
                               f"| break@bar-{n_bars - 1 - break_bar_idx} "
                               f"| curr_low={curr_low:.0f} tol={tol:.0f}"),
                    "tp1_rr": 2.0,
                    "tp2_rr": 5.0,
                }

            no_sig["reason"] = "no broken swing high retest found"
            return no_sig

        else:
            # ── SHORT: swing low broken, now retesting from below ─────────────
            if not swing_l:
                no_sig["reason"] = "no swing lows found"
                return no_sig

            for sl_idx, sl_price in reversed(swing_l):
                break_bar_idx = None
                search_start  = max(sl_idx + 1, n_bars - 1 - self.break_lookback)
                for k in range(search_start, n_bars - 1):
                    if closes[k] < sl_price:
                        break_bar_idx = k
                if break_bar_idx is None:
                    continue
                if break_bar_idx >= n_bars - 2:
                    continue

                highs_since_break = highs[break_bar_idx + 1: n_bars]
                if len(highs_since_break) == 0:
                    continue
                max_high_since_break = float(np.max(highs_since_break))
                if max_high_since_break < sl_price - tol:
                    continue

                if curr_high < sl_price - tol:
                    continue

                if curr_close >= sl_price:
                    continue

                entry  = curr_close
                sl_val = curr_high + self.sl_buffer_atr * atr

                sl_dist = abs(sl_val - entry)
                if sl_dist <= 0:
                    continue
                if sl_dist > self.max_sl_atr * atr:
                    continue

                return {
                    "signal": True,
                    "entry":  round(entry, 2),
                    "sl":     round(sl_val, 2),
                    "reason": (f"SBR short: retest of broken swing_L {sl_price:.0f} "
                               f"| break@bar-{n_bars - 1 - break_bar_idx} "
                               f"| curr_high={curr_high:.0f} tol={tol:.0f}"),
                    "tp1_rr": 2.0,
                    "tp2_rr": 5.0,
                }

            no_sig["reason"] = "no broken swing low retest found"
            return no_sig
