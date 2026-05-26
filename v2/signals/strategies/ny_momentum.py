"""
strategies/ny_momentum.py — New York Session Momentum strategy.

For WTI and NAS100 which only trade well during NY session.
Also applies as secondary strategy for XAUUSD/GBPJPY during NY.

Entry model:
  1. NY session active (12:00–17:00 UTC)
  2. ADX > 25 — strong trend
  3. Supertrend aligned with direction
  4. EMA21 pullback OR strong momentum candle after open
  5. Previous session high/low acts as magnet (TP target)

Logic: NY open creates momentum moves. Trade the first strong pullback.
"""
from __future__ import annotations

import logging
from datetime import timezone

import pandas as pd

from v2.signals.strategies.base import StrategyBase, StrategyResult

logger = logging.getLogger(__name__)

NY_OPEN_UTC    = 12   # 12:00 UTC (08:00 ET)
NY_TRADE_UTC   = 17   # Stop entering new NY trades after 17:00 UTC
ADX_STRONG     = 25


class NYMomentumStrategy(StrategyBase):
    name        = "ny_momentum"
    instruments = ["WTI", "NAS100", "XAUUSD", "GBPJPY"]
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

        if symbol not in self.instruments:
            return self._no_signal(symbol, direction, "NY momentum: wrong instrument")
        if len(df_h1) < self.min_df_bars:
            return self._no_signal(symbol, direction, "Insufficient H1 bars")

        is_long = direction.lower() in ("long", "buy")

        # ── Session timing ────────────────────────────────────────────────────
        current_hour = None
        if "time" in df_h1.columns:
            try:
                last_time = pd.to_datetime(df_h1["time"].iloc[-1], utc=True)
                current_hour = last_time.hour
            except Exception:
                pass

        if current_hour is not None and not (NY_OPEN_UTC <= current_hour < NY_TRADE_UTC):
            return self._no_signal(symbol, direction,
                                   f"Not NY session (UTC {current_hour:02d}:xx, need {NY_OPEN_UTC:02d}–{NY_TRADE_UTC:02d})")

        # ── ADX strong trend ──────────────────────────────────────────────────
        adx = self._adx(df_h1)
        if adx.get("adx", 0) < ADX_STRONG:
            return self._no_signal(symbol, direction,
                                   f"ADX {adx.get('adx', 0):.1f} < {ADX_STRONG} — need strong trend for NY momentum")

        adx_agrees = adx.get("bias") == ("bullish" if is_long else "bearish")
        if not adx_agrees:
            return self._no_signal(symbol, direction,
                                   f"ADX directional bias {adx.get('bias')} opposes {direction}")

        # ── Supertrend check ──────────────────────────────────────────────────
        st_ok = False
        try:
            from v2.analysis.indicators import get_supertrend
            st = get_supertrend(df_h1)
            st_ok = st.get("bias") == ("bullish" if is_long else "bearish")
        except Exception:
            st_ok = True  # skip if unavailable

        if not st_ok:
            return self._no_signal(symbol, direction, "Supertrend opposes direction")

        # ── HTF check ─────────────────────────────────────────────────────────
        htf_ok, htf_reason = self._htf_bias(df_h4, df_d1, direction)

        # ── Pullback or momentum candle ───────────────────────────────────────
        close = df_h1["close"]
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        price = float(close.iloc[-1])
        prox_to_ema = abs(price - ema21) / max(ema21, 1e-9)
        at_pullback = prox_to_ema <= 0.008

        # Strong momentum candle: body > 60% of range
        o = float(df_h1["open"].iloc[-1])
        c = float(df_h1["close"].iloc[-1])
        h = float(df_h1["high"].iloc[-1])
        lo = float(df_h1["low"].iloc[-1])
        body_pct = abs(c - o) / max(h - lo, 1e-9)
        momentum_candle = body_pct > 0.6 and ((is_long and c > o) or (not is_long and c < o))

        if not at_pullback and not momentum_candle:
            return self._no_signal(symbol, direction,
                                   f"No pullback to EMA21 ({prox_to_ema*100:.1f}% away) and no momentum candle")

        # ── Entry, SL, TP ─────────────────────────────────────────────────────
        atr       = self._atr(df_h1)
        entry     = price
        stop_loss = round(price - atr * 1.8, 5) if is_long else round(price + atr * 1.8, 5)
        tp1, tp2  = self._calc_tps(entry, stop_loss, direction, rr1=2.0, rr2=3.5)

        # ── Score ─────────────────────────────────────────────────────────────
        score = 0.0
        score += 2.5                                       # NY session active
        score += min(adx.get("adx", 0) / 10.0, 2.5)      # ADX strength
        score += 1.5 if st_ok else 0.0
        score += 1.5 if htf_ok else 0.5
        score += 1.5 if momentum_candle else 0.5
        score += 0.5 if at_pullback else 0.0

        reasons = [
            f"NY session momentum — ADX={adx.get('adx',0):.1f} ({adx.get('strength','')})",
            f"Supertrend {'aligned' if st_ok else 'weak'}",
            "Momentum candle confirmed" if momentum_candle else f"EMA21 pullback ({prox_to_ema*100:.1f}%)",
            htf_reason,
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
                "adx":             adx.get("adx", 0),
                "supertrend_ok":   st_ok,
                "at_pullback":     at_pullback,
                "momentum_candle": momentum_candle,
                "htf_ok":          htf_ok,
                "hour_utc":        current_hour,
            },
        )
