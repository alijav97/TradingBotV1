"""
ml/feature_engineer.py — Feature extraction for TradingBotV2 ML pipeline.

Extracts 40+ numeric features from a closed trade dict (from SQLite journal)
plus an optional OHLCV DataFrame. All missing values fall back to 0.0 so the
resulting dict is always safe to pass directly to a scikit-learn / LightGBM
estimator.

Usage:
    from v2.ml.feature_engineer import FeatureEngineer
    fe = FeatureEngineer(journal=journal)
    features = fe.extract(trade_row, df=ohlcv_df)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from v2.journal.sqlite_journal import Journal

logger = logging.getLogger(__name__)

# ── Encoding maps ──────────────────────────────────────────────────────────────

_DIRECTION_MAP: dict[str, float] = {"long": 1.0, "buy": 1.0, "short": -1.0, "sell": -1.0}

_TIMEFRAME_MAP: dict[str, float] = {
    "M1": 1.0, "M5": 5.0, "M15": 15.0, "M30": 30.0,
    "H1": 60.0, "H4": 240.0, "D1": 1440.0, "W1": 10080.0,
}

_SESSION_MAP: dict[str, float] = {
    "asian": 1.0, "london": 2.0, "new york": 3.0, "ny": 3.0,
    "overlap": 4.0, "london/ny": 4.0, "london_ny": 4.0, "off": 0.0,
}

_REGIME_MAP: dict[str, float] = {
    "trending_bull": 1.0, "trending bull": 1.0, "bull": 1.0,
    "trending_bear": 2.0, "trending bear": 2.0, "bear": 2.0,
    "trending": 3.0,
    "ranging": 4.0, "range": 4.0,
    "volatile": 5.0, "volatile/ranging": 5.0,
    "unknown": 0.0,
}

# Trading session windows (UTC hours, inclusive start)
_SESSION_UTC: dict[str, tuple[int, int]] = {
    "asian":   (23, 8),   # 23:00–08:00 UTC  (wraps midnight)
    "london":  (7, 16),   # 07:00–16:00 UTC
    "ny":      (12, 21),  # 12:00–21:00 UTC
}


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert a value to float; return default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _encode_map(raw: str | None, mapping: dict[str, float], default: float = 0.0) -> float:
    """Lower-strip a string and look it up in a mapping dict."""
    if not raw:
        return default
    return mapping.get(str(raw).strip().lower(), default)


def _is_session_active(hour_utc: int, session: str) -> float:
    """Return 1.0 if the given UTC hour falls within the named session."""
    bounds = _SESSION_UTC.get(session)
    if bounds is None:
        return 0.0
    start, end = bounds
    if start < end:
        return 1.0 if start <= hour_utc < end else 0.0
    # Wraps midnight (Asian)
    return 1.0 if (hour_utc >= start or hour_utc < end) else 0.0


# ── FeatureEngineer ────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Converts a closed trade dict + optional OHLCV DataFrame into a flat dict
    of numeric features suitable for ML training / inference.
    """

    def __init__(self, journal: "Journal | None" = None) -> None:
        self._journal = journal

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, trade: dict, df: pd.DataFrame | None = None) -> dict:
        """
        Extract 40+ numeric features from a closed trade row and optional OHLCV df.

        Parameters
        ----------
        trade:
            A dict matching the ``trades`` table schema from sqlite_journal.py.
        df:
            Optional OHLCV DataFrame with columns: open, high, low, close, volume.
            Index should be datetime-like. Needed for price-action features.

        Returns
        -------
        dict
            All values are float. Missing / uncomputable values default to 0.0.
        """
        features: dict[str, float] = {}

        try:
            features.update(self._time_features(trade))
            features.update(self._trade_features(trade))
            features.update(self._market_structure_features(trade))
            features.update(self._price_action_features(df))
            features.update(self._context_features(trade))
            features.update(self._historical_features(trade))
        except Exception as exc:  # broad catch only at the outermost level
            logger.error("feature extraction failed for trade %s: %s",
                         trade.get("id", "?")[:8], exc, exc_info=True)

        return features

    # ── Feature groups ─────────────────────────────────────────────────────────

    def _time_features(self, trade: dict) -> dict[str, float]:
        """Time-based features derived from the trade's open_time."""
        result: dict[str, float] = {}
        open_time_str: str = trade.get("open_time") or ""
        try:
            dt = datetime.fromisoformat(open_time_str)
        except ValueError:
            logger.debug("Cannot parse open_time '%s', using UTC now", open_time_str)
            dt = datetime.now(timezone.utc)

        hour = float(dt.hour)
        dow  = float(dt.weekday())   # 0=Mon … 6=Sun

        result["hour_of_day"]      = hour
        result["day_of_week"]      = dow
        result["is_asian_session"] = _is_session_active(dt.hour, "asian")
        result["is_london_session"]= _is_session_active(dt.hour, "london")
        result["is_ny_session"]    = _is_session_active(dt.hour, "ny")
        # Encode hour cyclically so midnight ≈ 23:00
        result["hour_sin"]         = float(np.sin(2 * np.pi * dt.hour / 24))
        result["hour_cos"]         = float(np.cos(2 * np.pi * dt.hour / 24))
        result["dow_sin"]          = float(np.sin(2 * np.pi * dt.weekday() / 7))
        result["dow_cos"]          = float(np.cos(2 * np.pi * dt.weekday() / 7))
        return result

    def _trade_features(self, trade: dict) -> dict[str, float]:
        """Core trade-level numeric features."""
        direction_raw: str = trade.get("direction") or ""
        dir_enc = _encode_map(direction_raw, _DIRECTION_MAP, default=0.0)

        tf_raw: str = trade.get("timeframe") or "H1"
        tf_enc = _encode_map(tf_raw, _TIMEFRAME_MAP, default=60.0)

        entry   = _safe_float(trade.get("entry_price"))
        sl      = _safe_float(trade.get("stop_loss"))
        tp1     = _safe_float(trade.get("tp1_price"))
        tp2     = _safe_float(trade.get("tp2_price"))
        lot     = _safe_float(trade.get("lot_size"))
        conf    = _safe_float(trade.get("confluence_score"))

        # Risk/reward (entry → TP1 vs entry → SL)
        sl_dist = abs(entry - sl) if entry and sl else 0.0
        tp1_dist = abs(tp1 - entry) if tp1 and entry else 0.0
        tp2_dist = abs(tp2 - entry) if tp2 and entry else 0.0
        rr1 = tp1_dist / sl_dist if sl_dist > 0 else 0.0
        rr2 = tp2_dist / sl_dist if sl_dist > 0 else 0.0

        return {
            "direction_encoded":  dir_enc,
            "timeframe_encoded":  tf_enc,
            "lot_size":           lot,
            "confluence_score":   conf,
            "entry_price":        _safe_float(entry),
            "sl_distance_pips":   sl_dist,
            "tp1_distance_pips":  tp1_dist,
            "tp2_distance_pips":  tp2_dist,
            "rr_ratio_tp1":       rr1,
            "rr_ratio_tp2":       rr2,
            "tp1_hit":            float(int(trade.get("tp1_hit") or 0)),
            "be_moved":           float(int(trade.get("be_moved") or 0)),
        }

    def _market_structure_features(self, trade: dict) -> dict[str, float]:
        """Session, regime, and news score features."""
        session_raw: str = trade.get("session") or ""
        regime_raw:  str = trade.get("regime")  or ""
        news_score       = _safe_float(trade.get("news_score"))

        return {
            "session_encoded": _encode_map(session_raw, _SESSION_MAP),
            "regime_encoded":  _encode_map(regime_raw, _REGIME_MAP),
            "news_score":      news_score,
        }

    def _price_action_features(self, df: pd.DataFrame | None) -> dict[str, float]:
        """ATR, ADX, RSI, distance to EMA50, volume ratio — from OHLCV DataFrame."""
        zero: dict[str, float] = {
            "atr":               0.0,
            "atr_percentile":    0.0,
            "adx":               0.0,
            "rsi":               50.0,
            "distance_to_ema50": 0.0,
            "volume_ratio":      1.0,
        }
        if df is None or df.empty:
            return zero

        required = {"high", "low", "close"}
        if not required.issubset({c.lower() for c in df.columns}):
            logger.debug("OHLCV DataFrame missing required columns; skipping price-action features")
            return zero

        # Normalise column names
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        result: dict[str, float] = {}

        # ── ATR (14) ───────────────────────────────────────────────────────────
        try:
            atr_series = self._calc_atr(df, period=14)
            atr_val = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0.0
            result["atr"] = atr_val
            # ATR percentile: position of current ATR in last 100 values
            recent_atr = atr_series.dropna().tail(100)
            if len(recent_atr) >= 5:
                pct = float((recent_atr < atr_val).mean()) * 100.0
            else:
                pct = 50.0
            result["atr_percentile"] = pct
        except Exception as exc:
            logger.debug("ATR calculation error: %s", exc)
            result["atr"] = 0.0
            result["atr_percentile"] = 0.0

        # ── ADX (14) ───────────────────────────────────────────────────────────
        try:
            result["adx"] = self._calc_adx(df, period=14)
        except Exception as exc:
            logger.debug("ADX calculation error: %s", exc)
            result["adx"] = 0.0

        # ── RSI (14) ───────────────────────────────────────────────────────────
        try:
            result["rsi"] = self._calc_rsi(df, period=14)
        except Exception as exc:
            logger.debug("RSI calculation error: %s", exc)
            result["rsi"] = 50.0

        # ── Distance to EMA 50 ─────────────────────────────────────────────────
        try:
            close = df["close"]
            ema50 = close.ewm(span=50, adjust=False).mean()
            last_close = float(close.iloc[-1])
            last_ema50 = float(ema50.iloc[-1])
            # Normalise by ATR so it's scale-invariant
            atr_v = result.get("atr", 0.0) or 1e-9
            result["distance_to_ema50"] = (last_close - last_ema50) / atr_v
        except Exception as exc:
            logger.debug("EMA50 distance error: %s", exc)
            result["distance_to_ema50"] = 0.0

        # ── Volume ratio ───────────────────────────────────────────────────────
        try:
            if "volume" in df.columns:
                vol = df["volume"].replace(0, np.nan)
                rolling_avg = vol.rolling(20).mean()
                last_vol = float(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0.0
                avg_vol  = float(rolling_avg.iloc[-1]) if not pd.isna(rolling_avg.iloc[-1]) else 1.0
                result["volume_ratio"] = last_vol / avg_vol if avg_vol > 0 else 1.0
            else:
                result["volume_ratio"] = 1.0
        except Exception as exc:
            logger.debug("Volume ratio error: %s", exc)
            result["volume_ratio"] = 1.0

        return result

    def _context_features(self, trade: dict) -> dict[str, float]:
        """DXY trend, geo score, COT bias — parsed from raw_signal JSON."""
        raw_signal: dict = {}
        raw_str = trade.get("raw_signal")
        if raw_str:
            try:
                raw_signal = json.loads(raw_str) if isinstance(raw_str, str) else raw_str
            except json.JSONDecodeError as exc:
                logger.debug("raw_signal JSON decode error: %s", exc)

        # DXY trend: encode as numeric (1=bullish, -1=bearish, 0=neutral)
        dxy_raw = str(raw_signal.get("dxy_trend", raw_signal.get("dxy_trend_at_entry", "neutral"))).lower()
        dxy_map = {"bullish": 1.0, "bull": 1.0, "up": 1.0, "bearish": -1.0, "bear": -1.0, "down": -1.0}
        dxy_enc = dxy_map.get(dxy_raw, 0.0)

        geo_score = _safe_float(raw_signal.get("geo_score", raw_signal.get("geo_risk_score", 0.0)))

        # COT bias: net speculative positioning direction
        cot_raw = str(raw_signal.get("cot_bias", raw_signal.get("cot_net_position", "neutral"))).lower()
        cot_map = {"bullish": 1.0, "bull": 1.0, "long": 1.0, "bearish": -1.0, "bear": -1.0, "short": -1.0}
        cot_enc = cot_map.get(cot_raw, 0.0)

        # Extra context scalars stored in raw_signal
        spread_pips  = _safe_float(raw_signal.get("spread_pips", raw_signal.get("spread_usd", 0.0)))
        checklist    = _safe_float(raw_signal.get("checklist_score", raw_signal.get("checklist_passed", 0.0)))
        counter_trend = float(bool(raw_signal.get("counter_trend", raw_signal.get("is_counter_trend", False))))
        adx_from_sig = _safe_float(raw_signal.get("adx", 0.0))
        in_killzone  = float(bool(raw_signal.get("in_killzone", False)))

        return {
            "dxy_trend_at_entry":  dxy_enc,
            "geo_score":           geo_score,
            "cot_bias":            cot_enc,
            "spread_pips":         spread_pips,
            "checklist_score":     checklist,
            "counter_trend":       counter_trend,
            "adx_signal":          adx_from_sig,
            "in_killzone":         in_killzone,
        }

    def _historical_features(self, trade: dict) -> dict[str, float]:
        """Per-instrument and per-strategy win rates from the last 20 closed trades."""
        if self._journal is None:
            return {
                "instrument_win_rate_last20": 0.0,
                "strategy_win_rate_last20":   0.0,
            }

        symbol:   str = trade.get("symbol", "")
        strategy: str = trade.get("strategy", "")
        trade_id: str = trade.get("id", "")

        instrument_wr = self._calc_historical_wr(symbol=symbol, strategy=None,
                                                  exclude_id=trade_id, limit=20)
        strategy_wr   = self._calc_historical_wr(symbol=symbol, strategy=strategy,
                                                  exclude_id=trade_id, limit=20)

        return {
            "instrument_win_rate_last20": instrument_wr,
            "strategy_win_rate_last20":   strategy_wr,
        }

    def _calc_historical_wr(
        self,
        symbol: str,
        strategy: str | None,
        exclude_id: str,
        limit: int,
    ) -> float:
        """Query journal for the last ``limit`` closed trades and return win rate."""
        try:
            trades = self._journal.get_trades(symbol=symbol, status="CLOSED", limit=limit + 1)
            # Exclude the trade being evaluated to avoid data leakage
            trades = [t for t in trades if t.get("id") != exclude_id][:limit]
            if not trades:
                return 0.0
            if strategy:
                trades = [t for t in trades if t.get("strategy") == strategy]
            if not trades:
                return 0.0
            wins = sum(1 for t in trades if _safe_float(t.get("pnl_usd")) > 0)
            return round(wins / len(trades), 4)
        except Exception as exc:
            logger.debug("Historical win-rate query failed: %s", exc)
            return 0.0

    # ── Indicator helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """True Range → EMA-smoothed ATR."""
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _calc_rsi(df: pd.DataFrame, period: int = 14) -> float:
        """Wilder RSI on close prices."""
        close = df["close"]
        if len(close) < period + 1:
            return 50.0
        delta  = close.diff()
        gain   = delta.clip(lower=0)
        loss   = (-delta).clip(lower=0)
        avg_g  = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_l  = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs     = avg_g / avg_l.replace(0, np.nan)
        rsi    = 100 - (100 / (1 + rs))
        val    = rsi.iloc[-1]
        return float(val) if not pd.isna(val) else 50.0

    @staticmethod
    def _calc_adx(df: pd.DataFrame, period: int = 14) -> float:
        """Average Directional Index (Wilder smoothing)."""
        if len(df) < period * 2:
            return 0.0
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        # When +DM < -DM or vice versa, zero out the weaker
        mask = plus_dm < minus_dm
        plus_dm[mask]  = 0.0
        mask2 = minus_dm < plus_dm
        minus_dm[mask2] = 0.0

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr14    = tr.ewm(span=period, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr14.replace(0, np.nan)

        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(span=period, adjust=False).mean()
        val = adx.iloc[-1]
        return float(val) if not pd.isna(val) else 0.0
