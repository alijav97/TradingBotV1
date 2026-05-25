"""
signals/signal_ranker.py — Composite signal ranker for TradingBotV2.

Ranks a list of signals by a weighted composite score built from four
components:

    composite = (confluence * 0.4) + (win_rate * 0.3) + (ml * 0.2) + (rr * 0.1)

Each component is first normalised to [0, 1] before weighting.

Component definitions
─────────────────────
    confluence_score : raw score from ConfluenceEngine (0 – 12)
    win_rate_score   : historical win rate from Journal.get_stats() (0 – 100 %)
    ml_score         : LightGBM win-probability (0.0 – 1.0); defaults to 0.5
                       when no trained model is available
    rr_score         : based on R:R ratio derived from (tp1_price, stop_loss,
                       entry_price); clipped and normalised to [0, 1]

Usage:
    from v2.signals.signal_ranker import SignalRanker
    from v2.journal.sqlite_journal import Journal

    journal = Journal()
    ranker  = SignalRanker()

    ranked = ranker.rank(signals, journal)
    best   = ranked[0]   # highest composite_score first
"""
from __future__ import annotations

import logging
from typing import Any

from v2.journal.sqlite_journal import Journal

logger = logging.getLogger(__name__)

# ── Weighting constants ────────────────────────────────────────────────────────
_W_CONFLUENCE = 0.4
_W_WIN_RATE   = 0.3
_W_ML         = 0.2
_W_RR         = 0.1

# Normalisation bounds
_CONFLUENCE_MAX = 12.0   # ConfluenceEngine max_score
_WIN_RATE_MAX   = 100.0  # percentage (0 – 100)
_RR_MIN         = 1.0    # below 1:1 → score 0
_RR_MAX         = 5.0    # 5:1 or above → score 1
_ML_DEFAULT     = 0.5    # fallback when model is untrained


class SignalRanker:
    """
    Ranks trading signals by composite score.

    Parameters
    ----------
    journal_days :
        Look-back window (days) passed to ``Journal.get_stats()`` when
        computing the historical win-rate component.  Default: 30.
    """

    def __init__(self, journal_days: int = 30) -> None:
        self.journal_days = journal_days

    # ── Public API ─────────────────────────────────────────────────────────────

    def rank(
        self,
        signals:   list[dict],
        journal:   Journal,
        ml_model:  Any = None,
    ) -> list[dict]:
        """
        Score every signal and return a list sorted by composite_score descending.

        Parameters
        ----------
        signals   : list of signal dicts (each must have at minimum a ``symbol`` key)
        journal   : open Journal instance (used to pull win-rate stats)
        ml_model  : fitted LGBMClassifier (or a LGBMTrainer instance); optional.
                    When ``None``, ml_score defaults to 0.5 for all signals.

        Returns
        -------
        list[dict] — same dicts as input, each augmented with
                     ``composite_score`` and ``rank_reason``, sorted desc.
        """
        if not signals:
            return []

        scored: list[dict] = []
        for sig in signals:
            try:
                scored.append(self.score_one(sig, journal, ml_model))
            except Exception as exc:
                logger.error(
                    "signal_ranker: score_one failed for signal %s: %s",
                    sig.get("symbol", "?"), exc,
                    exc_info=True,
                )
                # Still include the signal but with a zero composite score so
                # it sorts to the bottom rather than being silently dropped.
                fallback = dict(sig)
                fallback["composite_score"] = 0.0
                fallback["rank_reason"]     = f"scoring_error: {exc}"
                scored.append(fallback)

        scored.sort(key=lambda s: s["composite_score"], reverse=True)
        logger.debug(
            "signal_ranker: ranked %d signals; top score=%.3f",
            len(scored),
            scored[0]["composite_score"] if scored else 0.0,
        )
        return scored

    def score_one(
        self,
        signal:   dict,
        journal:  Journal,
        ml_model: Any = None,
    ) -> dict:
        """
        Compute the composite score for a single signal dict.

        The input dict is shallow-copied before being augmented so the
        caller's original dict is not mutated.

        Parameters
        ----------
        signal    : signal dict — expected keys (all optional with defaults):
                      symbol, confluence_score, entry_price, stop_loss,
                      tp1_price, ml_features (dict of features for ml_model)
        journal   : open Journal instance
        ml_model  : fitted model (LGBMClassifier or LGBMTrainer); optional

        Returns
        -------
        dict — shallow copy of ``signal`` with added keys:
            composite_score : float  (0.0 – 1.0)
            rank_reason     : str    (human-readable breakdown)
            _score_components : dict (raw + normalised values per component)
        """
        result = dict(signal)

        # ── 1. Confluence component ────────────────────────────────────────────
        raw_confluence  = float(signal.get("confluence_score", signal.get("score", 0)) or 0)
        norm_confluence = _clamp(raw_confluence / _CONFLUENCE_MAX)

        # ── 2. Win-rate component ──────────────────────────────────────────────
        symbol           = signal.get("symbol", None)
        norm_win_rate, raw_win_rate = self._get_win_rate_norm(journal, symbol)

        # ── 3. ML score component ─────────────────────────────────────────────
        ml_raw, ml_source = self._get_ml_score(signal, ml_model)
        norm_ml = _clamp(ml_raw)  # already 0–1

        # ── 4. R:R component ──────────────────────────────────────────────────
        rr_ratio, norm_rr = self._get_rr_norm(signal)

        # ── Composite ─────────────────────────────────────────────────────────
        composite = (
            norm_confluence * _W_CONFLUENCE
            + norm_win_rate * _W_WIN_RATE
            + norm_ml       * _W_ML
            + norm_rr       * _W_RR
        )
        composite = round(_clamp(composite), 4)

        # ── Human-readable explanation ─────────────────────────────────────────
        rank_reason = (
            f"composite={composite:.3f} "
            f"[confluence={norm_confluence:.2f}×{_W_CONFLUENCE}"
            f" + win_rate={norm_win_rate:.2f}×{_W_WIN_RATE}"
            f" + ml={norm_ml:.2f}×{_W_ML}({ml_source})"
            f" + rr={norm_rr:.2f}×{_W_RR}(rr={rr_ratio:.2f})]"
        )

        result["composite_score"] = composite
        result["rank_reason"]     = rank_reason
        result["_score_components"] = {
            "confluence_raw":  raw_confluence,
            "confluence_norm": round(norm_confluence, 4),
            "win_rate_raw":    raw_win_rate,
            "win_rate_norm":   round(norm_win_rate, 4),
            "ml_raw":          round(ml_raw, 4),
            "ml_norm":         round(norm_ml, 4),
            "ml_source":       ml_source,
            "rr_ratio":        round(rr_ratio, 4),
            "rr_norm":         round(norm_rr, 4),
        }

        logger.debug(
            "signal_ranker: %s %s composite=%.3f "
            "(conf=%.2f wr=%.2f ml=%.2f rr=%.2f)",
            signal.get("symbol", "?"),
            signal.get("direction", "?"),
            composite,
            norm_confluence, norm_win_rate, norm_ml, norm_rr,
        )
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_win_rate_norm(
        self,
        journal: Journal,
        symbol:  str | None,
    ) -> tuple[float, float]:
        """
        Query Journal.get_stats() and return (normalised_win_rate, raw_pct).

        If the journal has no closed trades yet, returns (0.5, 50.0) —
        a neutral prior so new setups are not penalised.
        """
        try:
            stats = journal.get_stats(symbol=symbol, days=self.journal_days)
            trades = stats.get("trades", 0)
            if trades == 0:
                # No history yet — neutral prior
                return 0.5, 50.0
            raw = float(stats.get("win_rate", 0.0))
            return _clamp(raw / _WIN_RATE_MAX), raw
        except Exception as exc:
            logger.warning(
                "signal_ranker: could not fetch win-rate stats for %s: %s",
                symbol, exc,
            )
            return 0.5, 50.0

    def _get_ml_score(
        self,
        signal:   dict,
        ml_model: Any,
    ) -> tuple[float, str]:
        """
        Get ML win-probability from the model.

        Supports two calling conventions:
          • LGBMTrainer instance  → calls trainer.predict(model_obj, features)
            (expects ml_model to be a tuple (trainer, model_obj) or a
             LGBMTrainer with a loaded model attached as .model)
          • Raw LGBMClassifier    → calls model.predict_proba(X) directly via
            our internal helper
          • None                  → returns default 0.5

        In all error cases returns (_ML_DEFAULT, "default").
        """
        if ml_model is None:
            return _ML_DEFAULT, "default"

        features: dict = signal.get("ml_features") or signal.get("features") or {}

        # Case 1: caller passed a (LGBMTrainer, fitted_model) tuple
        if isinstance(ml_model, tuple) and len(ml_model) == 2:
            trainer, model_obj = ml_model
            try:
                prob = float(trainer.predict(model_obj, features))
                return _clamp(prob), "lgbm_trainer"
            except Exception as exc:
                logger.warning("signal_ranker: trainer.predict failed: %s", exc)
                return _ML_DEFAULT, "default"

        # Case 2: caller passed a LGBMTrainer whose .model attribute holds the
        #         fitted classifier (convenience pattern)
        if hasattr(ml_model, "predict") and hasattr(ml_model, "train"):
            # Looks like a LGBMTrainer — check for attached model
            inner_model = getattr(ml_model, "model", None) or getattr(ml_model, "_model", None)
            if inner_model is not None:
                try:
                    prob = float(ml_model.predict(inner_model, features))
                    return _clamp(prob), "lgbm_trainer"
                except Exception as exc:
                    logger.warning("signal_ranker: trainer.predict (inner) failed: %s", exc)
                    return _ML_DEFAULT, "default"

        # Case 3: caller passed a raw sklearn-compatible classifier
        if hasattr(ml_model, "predict_proba"):
            try:
                import numpy as np
                feature_names: list[str] = (
                    getattr(ml_model, "_v2_feature_names", None)
                    or list(getattr(ml_model, "feature_name_", None) or [])
                    or list(features.keys())
                )
                row = np.zeros((1, len(feature_names)), dtype=np.float32)
                for j, k in enumerate(feature_names):
                    val = features.get(k, 0.0)
                    try:
                        row[0, j] = float(val) if val is not None else 0.0
                    except (TypeError, ValueError):
                        row[0, j] = 0.0
                proba = ml_model.predict_proba(row)
                prob  = float(np.clip(proba[0, 1], 0.0, 1.0))
                return prob, "lgbm_raw"
            except Exception as exc:
                logger.warning("signal_ranker: predict_proba failed: %s", exc)
                return _ML_DEFAULT, "default"

        logger.warning(
            "signal_ranker: ml_model type %s is not supported; using default",
            type(ml_model).__name__,
        )
        return _ML_DEFAULT, "default"

    @staticmethod
    def _get_rr_norm(signal: dict) -> tuple[float, float]:
        """
        Compute R:R ratio from signal fields and normalise to [0, 1].

        R:R = |tp1_price - entry_price| / |stop_loss - entry_price|

        Clipped: below _RR_MIN → 0.0, above _RR_MAX → 1.0.
        Returns (rr_ratio, norm_rr).
        """
        try:
            entry = float(signal.get("entry_price") or signal.get("entry") or 0)
            sl    = float(signal.get("stop_loss")   or signal.get("sl")    or 0)
            tp1   = float(signal.get("tp1_price")   or signal.get("tp1")   or 0)

            if entry <= 0 or sl <= 0 or tp1 <= 0:
                return 0.0, 0.0

            risk   = abs(entry - sl)
            reward = abs(tp1   - entry)

            if risk < 1e-12:
                return 0.0, 0.0

            rr_ratio = reward / risk

            if rr_ratio <= _RR_MIN:
                return rr_ratio, 0.0
            if rr_ratio >= _RR_MAX:
                return rr_ratio, 1.0

            norm = (rr_ratio - _RR_MIN) / (_RR_MAX - _RR_MIN)
            return rr_ratio, _clamp(norm)

        except (TypeError, ValueError, ZeroDivisionError) as exc:
            logger.debug("signal_ranker: R:R calculation failed: %s", exc)
            return 0.0, 0.0


# ── Module-level helper ────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clip ``value`` to [lo, hi]."""
    return max(lo, min(hi, value))
