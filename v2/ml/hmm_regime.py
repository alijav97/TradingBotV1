"""
ml/hmm_regime.py — Hidden Markov Model market-regime detector for TradingBotV2.

Uses ``hmmlearn.hmm.GaussianHMM`` with 4 hidden states mapped to:
  TRENDING_BULL, TRENDING_BEAR, RANGING, VOLATILE

If hmmlearn is not installed a simple rule-based fallback is used instead.

Usage:
    from v2.ml.hmm_regime import RegimeDetector
    detector = RegimeDetector()
    detector.fit(ohlcv_df)
    regime = detector.predict(ohlcv_df)          # e.g. "TRENDING_BULL"
    probs  = detector.get_regime_probs(ohlcv_df) # {"TRENDING_BULL": 0.85, ...}
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    _GaussianHMM = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# ── Regime constants ───────────────────────────────────────────────────────────

REGIME_NAMES = ["TRENDING_BULL", "TRENDING_BEAR", "RANGING", "VOLATILE"]
N_STATES = 4

# ── Feature extraction ─────────────────────────────────────────────────────────


def _build_hmm_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build a (T × 4) observation matrix from OHLCV data.

    Columns:
      0 — log return
      1 — ATR / close  (normalised volatility)
      2 — rolling 10-bar return std  (realised vol)
      3 — volume ratio  (bar vol / 20-bar avg vol)
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)

    # Log returns
    log_ret = np.log(close / close.shift(1)).fillna(0.0)

    # ATR / close
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr     = tr.ewm(span=14, adjust=False).mean()
    atr_pct = (atr / close.replace(0, np.nan)).fillna(0.0)

    # Rolling realised volatility
    roll_vol = log_ret.rolling(10).std().fillna(0.0)

    # Volume ratio
    if "volume" in df.columns:
        vol       = df["volume"].astype(float).replace(0, np.nan)
        avg_vol   = vol.rolling(20).mean()
        vol_ratio = (vol / avg_vol.replace(0, np.nan)).fillna(1.0)
    else:
        vol_ratio = pd.Series(1.0, index=df.index)

    obs = np.column_stack([
        log_ret.values,
        atr_pct.values,
        roll_vol.values,
        vol_ratio.values,
    ])
    # Replace any remaining NaN / Inf
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return obs.astype(np.float64)


def _assign_regime_labels(model: Any) -> list[str]:
    """
    Map HMM hidden states → regime names by inspecting the learned means.

    State assignment heuristics (based on log-return mean and ATR/close mean):
      - Highest positive  log-return mean → TRENDING_BULL
      - Most  negative    log-return mean → TRENDING_BEAR
      - Lowest ATR/close  mean            → RANGING
      - Remaining state                   → VOLATILE
    """
    means = model.means_          # shape (N_STATES, n_features)
    ret_means = means[:, 0]       # log-return means
    atr_means = means[:, 1]       # ATR/close means

    assigned: list[str | None] = [None] * N_STATES
    available = set(range(N_STATES))

    bull_idx  = int(np.argmax(ret_means))
    assigned[bull_idx] = "TRENDING_BULL"
    available.discard(bull_idx)

    bear_idx  = int(np.argmin(ret_means[list(available)] + 1e9 * np.array(
        [1 if i not in available else 0 for i in range(N_STATES)]
    )))
    # Re-derive: minimum log-return among remaining states
    rem_ret = {i: ret_means[i] for i in available}
    bear_idx = min(rem_ret, key=rem_ret.get)  # type: ignore[arg-type]
    assigned[bear_idx] = "TRENDING_BEAR"
    available.discard(bear_idx)

    rem_atr = {i: atr_means[i] for i in available}
    range_idx = min(rem_atr, key=rem_atr.get)  # type: ignore[arg-type]
    assigned[range_idx] = "RANGING"
    available.discard(range_idx)

    vol_idx = next(iter(available))
    assigned[vol_idx] = "VOLATILE"

    # Ensure no None entries (safety net)
    return [r if r is not None else "RANGING" for r in assigned]


# ── Fallback rules ─────────────────────────────────────────────────────────────

def _rule_based_regime(df: pd.DataFrame) -> str:
    """
    Simple heuristic regime detection when hmmlearn is unavailable.

    Rules:
      - ADX > 25 + positive EMA slope → TRENDING_BULL
      - ADX > 25 + negative EMA slope → TRENDING_BEAR
      - Realised vol > 2× median      → VOLATILE
      - Else                           → RANGING
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    close = df["close"].astype(float)

    # ADX
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask_p = plus_dm < minus_dm; plus_dm[mask_p] = 0.0
    mask_m = minus_dm < plus_dm; minus_dm[mask_m] = 0.0
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14    = tr.ewm(span=14, adjust=False).mean().replace(0, np.nan)
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14
    dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = float(dx.ewm(span=14, adjust=False).mean().iloc[-1])

    # EMA slope
    ema50    = close.ewm(span=50, adjust=False).mean()
    ema_now  = float(ema50.iloc[-1])
    ema_prev = float(ema50.iloc[-6]) if len(ema50) >= 6 else ema_now
    ema_slope = ema_now - ema_prev

    # Realised volatility
    log_ret  = np.log(close / close.shift(1)).dropna()
    roll_vol = float(log_ret.rolling(10).std().iloc[-1]) if len(log_ret) >= 10 else 0.0
    med_vol  = float(log_ret.rolling(50).std().median()) if len(log_ret) >= 50 else roll_vol

    if adx > 25:
        return "TRENDING_BULL" if ema_slope > 0 else "TRENDING_BEAR"
    if med_vol > 0 and roll_vol > 2.0 * med_vol:
        return "VOLATILE"
    return "RANGING"


# ── RegimeDetector ─────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Fit a 4-state Gaussian HMM on OHLCV data and decode market regimes.

    Falls back to rule-based detection if hmmlearn is not installed.
    """

    def __init__(self, n_states: int = N_STATES, random_state: int = 42) -> None:
        self._n_states    = n_states
        self._random_state = random_state
        self._model: Any  = None
        self._regime_map: list[str] = REGIME_NAMES[:]
        self._fitted      = False
        self._use_hmm     = _HMM_AVAILABLE

        if not self._use_hmm:
            logger.warning(
                "hmmlearn is not installed; RegimeDetector will use rule-based fallback. "
                "Install with: pip install hmmlearn"
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """
        Train the HMM (or prepare fallback) on an OHLCV DataFrame.

        Parameters
        ----------
        df:
            OHLCV DataFrame with at least columns: high, low, close.
            Longer history → more stable state estimates.

        Returns
        -------
        self
        """
        if df is None or df.empty:
            raise ValueError("fit() requires a non-empty OHLCV DataFrame")

        if not self._use_hmm:
            logger.info("Skipping HMM fit (hmmlearn unavailable); rule-based fallback active")
            self._fitted = True
            return self

        obs = _build_hmm_features(df)
        if len(obs) < self._n_states * 5:
            logger.warning(
                "Too few observations (%d) to reliably fit %d-state HMM; "
                "falling back to rule-based detection",
                len(obs), self._n_states,
            )
            self._use_hmm = False
            self._fitted  = True
            return self

        try:
            model = _GaussianHMM(
                n_components=self._n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=self._random_state,
                verbose=False,
            )
            model.fit(obs)
            self._model      = model
            self._regime_map = _assign_regime_labels(model)
            self._fitted     = True
            logger.info("HMM fitted; state→regime map: %s", self._regime_map)
        except Exception as exc:
            logger.error("HMM fit failed (%s); falling back to rule-based detection", exc)
            self._use_hmm = False
            self._fitted  = True

        return self

    def predict(self, df: pd.DataFrame) -> str:
        """
        Decode the most likely current regime from the last OHLCV bars.

        Parameters
        ----------
        df:
            OHLCV DataFrame (same schema as used for fit).

        Returns
        -------
        str
            One of: ``TRENDING_BULL``, ``TRENDING_BEAR``, ``RANGING``, ``VOLATILE``.
        """
        if df is None or df.empty:
            return "RANGING"

        if not self._fitted:
            logger.debug("RegimeDetector not fitted yet; calling fit() automatically")
            self.fit(df)

        if not self._use_hmm or self._model is None:
            return _rule_based_regime(df)

        try:
            obs = _build_hmm_features(df)
            states = self._model.predict(obs)
            last_state = int(states[-1])
            return self._regime_map[last_state]
        except Exception as exc:
            logger.error("HMM predict failed (%s); using rule-based fallback", exc)
            return _rule_based_regime(df)

    def get_regime_probs(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Return the posterior state probabilities for the last bar.

        Parameters
        ----------
        df:
            OHLCV DataFrame.

        Returns
        -------
        dict
            Mapping of regime name → probability, summing to 1.0.
            Falls back to one-hot if HMM posterior is unavailable.
        """
        if df is None or df.empty:
            return {name: 0.25 for name in REGIME_NAMES}

        if not self._fitted:
            self.fit(df)

        if not self._use_hmm or self._model is None:
            # Rule-based: one-hot with soft neighbours
            dominant = _rule_based_regime(df)
            probs = {name: 0.05 for name in REGIME_NAMES}
            remainder = 1.0 - sum(probs.values()) + probs.get(dominant, 0.05)
            probs[dominant] = round(remainder + probs[dominant], 4)
            # Normalise
            total = sum(probs.values())
            return {k: round(v / total, 4) for k, v in probs.items()}

        try:
            obs = _build_hmm_features(df)
            # posteriors: shape (T, N_STATES)
            _log_prob, posteriors = self._model.score_samples(obs)
            last_posteriors = posteriors[-1]        # shape (N_STATES,)
            return {
                self._regime_map[i]: round(float(last_posteriors[i]), 4)
                for i in range(self._n_states)
            }
        except Exception as exc:
            logger.error("get_regime_probs failed (%s); returning uniform probs", exc)
            return {name: round(1.0 / N_STATES, 4) for name in REGIME_NAMES}
