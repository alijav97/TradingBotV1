"""
ml/ml_engine.py — ML orchestration layer for TradingBotV2.

MLEngine wires together FeatureEngineer, LGBMTrainer, and RegimeDetector
into a single facade used by the rest of the trading system.

Usage:
    from v2.ml.ml_engine import MLEngine
    engine = MLEngine(journal=journal)

    # Check / trigger retraining
    if engine.should_retrain():
        result = engine.retrain()

    # Score a live signal before taking a trade
    prob = engine.get_signal_confidence(signal_dict, df=ohlcv_df)

    # Current regime for a symbol
    regime = engine.get_regime("XAUUSD")
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from v2.ml.feature_engineer import FeatureEngineer
from v2.ml.lightgbm_trainer import LGBMTrainer
from v2.ml.hmm_regime import RegimeDetector

if TYPE_CHECKING:
    from v2.journal.sqlite_journal import Journal

logger = logging.getLogger(__name__)

# ── MLEngine ───────────────────────────────────────────────────────────────────


class MLEngine:
    """
    Orchestrates feature engineering, model training, and inference.

    Parameters
    ----------
    journal:
        A :class:`v2.journal.sqlite_journal.Journal` instance used for
        loading training data and historical win-rate features.
    feed:
        Optional data-feed / connector object.  If provided, ``get_regime``
        will attempt to fetch OHLCV data for the requested symbol.
        When ``None`` the caller must supply a DataFrame directly.
    """

    def __init__(
        self,
        journal: "Journal",
        feed: Any | None = None,
    ) -> None:
        self._journal          = journal
        self._feed             = feed
        self._feature_engineer = FeatureEngineer(journal=journal)
        self._trainer          = LGBMTrainer()
        self._regime_detector  = RegimeDetector()
        self._model: Any | None = self._trainer.load_model()

        # Track how many trades existed at last retrain
        self._trades_at_last_retrain: int = self._count_closed_trades()
        logger.info(
            "MLEngine initialised; model_loaded=%s, closed_trades=%d",
            self._model is not None,
            self._trades_at_last_retrain,
        )

    # ── Retraining ─────────────────────────────────────────────────────────────

    def should_retrain(self) -> bool:
        """
        Return ``True`` if at least ``ML_RETRAIN_INTERVAL`` new closed trades
        have accumulated since the last retrain.
        """
        from v2 import settings
        current = self._count_closed_trades()
        delta   = current - self._trades_at_last_retrain
        needs   = delta >= settings.ML_RETRAIN_INTERVAL
        logger.debug(
            "should_retrain: current=%d last=%d delta=%d threshold=%d → %s",
            current, self._trades_at_last_retrain,
            delta, settings.ML_RETRAIN_INTERVAL, needs,
        )
        return needs

    def retrain(self) -> dict:
        """
        Load labelled training data from the journal, fit a new LightGBM
        classifier, and refresh the in-memory model.

        Returns
        -------
        dict with keys:
            ``trained``      — bool, True if a model was successfully fitted.
            ``samples``      — int, number of labelled rows used.
            ``val_accuracy`` — float, validation accuracy (NaN if not enough val data).
        """
        result: dict = {"trained": False, "samples": 0, "val_accuracy": float("nan")}

        training_data = self._journal.get_ml_training_data()
        result["samples"] = len(training_data)

        if len(training_data) < 10:
            logger.warning(
                "Not enough labelled trades to retrain (need 10, have %d)",
                len(training_data),
            )
            return result

        try:
            model = self._trainer.train(training_data)
            self._model = model
            self._trades_at_last_retrain = self._count_closed_trades()
            result["trained"]      = True
            result["val_accuracy"] = float(getattr(model, "_v2_val_accuracy", float("nan")))
            logger.info(
                "Retrain complete: samples=%d val_accuracy=%.3f",
                result["samples"], result["val_accuracy"],
            )
        except (ValueError, RuntimeError) as exc:
            logger.error("Retrain failed: %s", exc, exc_info=True)

        return result

    # ── Inference ──────────────────────────────────────────────────────────────

    def get_signal_confidence(
        self,
        signal: dict,
        df: pd.DataFrame | None = None,
    ) -> float:
        """
        Estimate win probability for a potential trade signal (0.0–1.0).

        The signal dict should follow the same schema as a trade row or
        a pre-trade signal from the confluence engine.  When no trained
        model is available a heuristic fallback is used.

        Parameters
        ----------
        signal:
            Dict containing signal/trade fields (direction, timeframe,
            session, regime, confluence_score, etc.).
        df:
            Optional OHLCV DataFrame for price-action features.

        Returns
        -------
        float
            Win probability clamped to [0.0, 1.0].
        """
        if self._model is None:
            logger.debug("No trained model; using heuristic confidence")
            return self._heuristic_confidence(signal)

        try:
            features = self._feature_engineer.extract(signal, df=df)
            prob = self._trainer.predict(self._model, features)
            logger.debug(
                "ML confidence for %s %s: %.3f",
                signal.get("symbol", "?"), signal.get("direction", "?"), prob,
            )
            return prob
        except Exception as exc:
            logger.error("get_signal_confidence failed: %s", exc, exc_info=True)
            return self._heuristic_confidence(signal)

    def get_regime(
        self,
        symbol: str,
        df: pd.DataFrame | None = None,
    ) -> str:
        """
        Return the current market regime for a symbol.

        Parameters
        ----------
        symbol:
            Instrument ticker, e.g. ``"XAUUSD"``.
        df:
            Optional OHLCV DataFrame.  If ``None`` and a feed is configured,
            the engine will attempt to fetch data automatically.

        Returns
        -------
        str
            One of: ``TRENDING_BULL``, ``TRENDING_BEAR``, ``RANGING``, ``VOLATILE``.
        """
        ohlcv = df

        # Attempt to fetch data from feed if not provided
        if ohlcv is None and self._feed is not None:
            try:
                ohlcv = self._feed.get_ohlcv(symbol, timeframe="H1", limit=200)
            except Exception as exc:
                logger.warning("Feed fetch failed for %s: %s", symbol, exc)

        if ohlcv is None or (isinstance(ohlcv, pd.DataFrame) and ohlcv.empty):
            logger.debug("No OHLCV data for regime detection of %s; returning RANGING", symbol)
            return "RANGING"

        try:
            if not self._regime_detector._fitted:
                self._regime_detector.fit(ohlcv)
            return self._regime_detector.predict(ohlcv)
        except Exception as exc:
            logger.error("get_regime failed for %s: %s", symbol, exc, exc_info=True)
            return "RANGING"

    # ── Convenience wrappers ───────────────────────────────────────────────────

    def get_regime_probs(
        self,
        symbol: str,
        df: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Return regime posterior probability dict for a symbol."""
        ohlcv = df
        if ohlcv is None and self._feed is not None:
            try:
                ohlcv = self._feed.get_ohlcv(symbol, timeframe="H1", limit=200)
            except Exception as exc:
                logger.warning("Feed fetch for regime probs failed for %s: %s", symbol, exc)

        if ohlcv is None or (isinstance(ohlcv, pd.DataFrame) and ohlcv.empty):
            return {name: 0.25 for name in ["TRENDING_BULL", "TRENDING_BEAR", "RANGING", "VOLATILE"]}

        try:
            return self._regime_detector.get_regime_probs(ohlcv)
        except Exception as exc:
            logger.error("get_regime_probs failed: %s", exc, exc_info=True)
            return {name: 0.25 for name in ["TRENDING_BULL", "TRENDING_BEAR", "RANGING", "VOLATILE"]}

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance from the current model (empty dict if no model)."""
        if self._model is None:
            return {}
        feature_names: list[str] = getattr(self._model, "_v2_feature_names", None) or []
        return self._trainer.get_feature_importance(self._model, feature_names)

    def extract_and_store_features(
        self,
        trade: dict,
        df: pd.DataFrame | None = None,
        predicted_prob: float | None = None,
    ) -> None:
        """
        Extract features for a trade and persist them in the journal's
        ``ml_features`` table.  Should be called when a trade is opened.
        """
        trade_id = trade.get("id")
        if not trade_id:
            logger.warning("extract_and_store_features: trade has no 'id', skipping")
            return
        try:
            features = self._feature_engineer.extract(trade, df=df)
            self._journal.save_ml_features(
                trade_id=trade_id,
                features=features,
                predicted_prob=predicted_prob,
                model_version=self._model_version(),
            )
        except Exception as exc:
            logger.error(
                "Failed to store ML features for trade %s: %s",
                str(trade_id)[:8], exc, exc_info=True,
            )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _count_closed_trades(self) -> int:
        """Return total number of closed trades in the journal."""
        try:
            trades = self._journal.get_trades(status="CLOSED", limit=100_000)
            return len(trades)
        except Exception as exc:
            logger.debug("_count_closed_trades error: %s", exc)
            return 0

    @staticmethod
    def _heuristic_confidence(signal: dict) -> float:
        """
        Rule-based confidence estimate when no ML model is available.
        Uses confluence_score normalised to 0–1.
        """
        raw_score = signal.get("confluence_score") or signal.get("confidence") or 0.0
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        # Assume confluence_score is on a 0–10 scale; clamp to [0, 1]
        return float(min(max(score / 10.0, 0.0), 1.0))

    def _model_version(self) -> str:
        """Return a short version string for the current model."""
        if self._model is None:
            return "none"
        n = getattr(self._model, "_v2_n_samples", 0)
        return f"lgbm_n{n}"
