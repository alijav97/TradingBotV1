"""
ml/lightgbm_trainer.py — LightGBM classifier for trade win-probability.

Wraps LGBMClassifier with 80/20 train/val split, model persistence
(pickle), and feature-importance reporting.

Usage:
    from v2.ml.lightgbm_trainer import LGBMTrainer
    trainer = LGBMTrainer()
    model   = trainer.train(training_data)          # list[dict] with "_label" key
    prob    = trainer.predict(model, features)      # float 0.0–1.0
    trainer.save_model(model)
    model   = trainer.load_model()
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Internal helpers ───────────────────────────────────────────────────────────

_PRIVATE_KEYS = frozenset({"_label", "_trade_id"})


def _prepare_matrix(
    data: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Convert list-of-feature-dicts into (X, y, feature_names).

    Keys starting with ``_`` are stripped (they are metadata, not features).
    Missing values are filled with 0.0.  All dicts must share the same feature
    set; extra keys in later rows are silently dropped.
    """
    if not data:
        raise ValueError("training_data is empty")

    # Collect the union of feature keys (excluding private keys) in insertion order
    feature_names: list[str] = []
    seen: set[str] = set()
    for row in data:
        for k in row:
            if k not in _PRIVATE_KEYS and k not in seen:
                feature_names.append(k)
                seen.add(k)

    labels = np.array([float(row.get("_label", 0)) for row in data], dtype=np.float32)
    matrix = np.zeros((len(data), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(data):
        for j, k in enumerate(feature_names):
            val = row.get(k, 0.0)
            try:
                matrix[i, j] = float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                matrix[i, j] = 0.0

    return matrix, labels, feature_names


# ── LGBMTrainer ───────────────────────────────────────────────────────────────

class LGBMTrainer:
    """
    Train, persist, and serve a LightGBM binary classifier.

    Parameters
    ----------
    model_dir:
        Directory to read/write ``lgbm_model.pkl``.
        Defaults to ``settings.MODEL_DIR``.
    """

    MODEL_FILENAME = "lgbm_model.pkl"

    def __init__(self, model_dir: str | Path | None = None) -> None:
        if model_dir is None:
            from v2 import settings
            model_dir = settings.MODEL_DIR
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model_path = self._model_dir / self.MODEL_FILENAME

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self, training_data: list[dict]) -> Any:
        """
        Fit a LGBMClassifier on ``training_data``.

        Parameters
        ----------
        training_data:
            List of feature dicts; each must contain a ``"_label"`` key
            (1 = win, 0 = loss).  ``"_trade_id"`` is ignored.

        Returns
        -------
        Fitted LGBMClassifier model.
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed. Run: pip install lightgbm"
            ) from exc

        if len(training_data) < 10:
            raise ValueError(
                f"Need at least 10 labelled trades to train; got {len(training_data)}"
            )

        X, y, feature_names = _prepare_matrix(training_data)

        # 80 / 20 train / validation split (chronological — no shuffle)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        logger.info(
            "Training LGBMClassifier on %d samples (val=%d), features=%d",
            len(X_train), len(X_val), len(feature_names),
        )

        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            feature_name=feature_names,
            eval_set=[(X_val, y_val)],
        )

        # Validation accuracy
        if len(X_val) > 0:
            y_pred = model.predict(X_val)
            val_acc = float((y_pred == y_val).mean())
            logger.info("Validation accuracy: %.3f (%d samples)", val_acc, len(X_val))
        else:
            val_acc = float("nan")
            logger.warning("Validation set is empty; accuracy not computed")

        # Attach metadata so callers can inspect without retraining
        model._v2_feature_names = feature_names  # type: ignore[attr-defined]
        model._v2_val_accuracy  = val_acc          # type: ignore[attr-defined]
        model._v2_n_samples     = len(X)           # type: ignore[attr-defined]

        self.save_model(model)
        return model

    def predict(self, model: Any, features: dict) -> float:
        """
        Return win probability (0.0–1.0) for a single feature dict.

        Parameters
        ----------
        model:
            A fitted model returned by :meth:`train` or :meth:`load_model`.
        features:
            Feature dict (same schema as training rows, without ``_label``).
        """
        feature_names: list[str] = getattr(model, "_v2_feature_names", None) or []
        if not feature_names:
            # Fall back to model's booster feature names
            try:
                feature_names = list(model.feature_name_)
            except AttributeError:
                logger.warning("Model has no stored feature names; prediction may be unreliable")
                feature_names = list(features.keys())

        row = np.zeros((1, len(feature_names)), dtype=np.float32)
        for j, k in enumerate(feature_names):
            val = features.get(k, 0.0)
            try:
                row[0, j] = float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                row[0, j] = 0.0

        try:
            proba = model.predict_proba(row)
            # LGBMClassifier returns [[p_loss, p_win]]
            return float(np.clip(proba[0, 1], 0.0, 1.0))
        except Exception as exc:
            logger.error("predict_proba failed: %s", exc, exc_info=True)
            return 0.5

    def load_model(self) -> Any | None:
        """
        Load model from ``MODEL_DIR/lgbm_model.pkl``.
        Returns ``None`` if the file does not exist.
        """
        if not self._model_path.exists():
            logger.debug("No saved model found at %s", self._model_path)
            return None
        try:
            with open(self._model_path, "rb") as fh:
                model = pickle.load(fh)
            logger.info("Loaded LightGBM model from %s", self._model_path)
            return model
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            logger.error("Failed to load model from %s: %s", self._model_path, exc)
            return None

    def save_model(self, model: Any) -> None:
        """Persist model to ``MODEL_DIR/lgbm_model.pkl``."""
        try:
            with open(self._model_path, "wb") as fh:
                pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved LightGBM model to %s", self._model_path)
        except OSError as exc:
            logger.error("Failed to save model to %s: %s", self._model_path, exc)
            raise

    def get_feature_importance(self, model: Any, feature_names: list[str]) -> dict[str, float]:
        """
        Return a dict mapping feature name → normalized importance score (0–1).

        Parameters
        ----------
        model:
            Fitted LGBMClassifier.
        feature_names:
            Ordered list of feature names matching the training columns.
            If the model already stores ``_v2_feature_names``, that list
            takes precedence.
        """
        stored_names: list[str] = getattr(model, "_v2_feature_names", None) or feature_names

        try:
            importances: np.ndarray = model.feature_importances_
        except AttributeError as exc:
            logger.error("Model does not expose feature_importances_: %s", exc)
            return {}

        if len(importances) != len(stored_names):
            logger.warning(
                "Mismatch: %d importances vs %d feature names",
                len(importances), len(stored_names),
            )
            min_len = min(len(importances), len(stored_names))
            importances   = importances[:min_len]
            stored_names  = stored_names[:min_len]

        total = float(importances.sum()) or 1.0
        return {
            name: round(float(imp) / total, 6)
            for name, imp in sorted(
                zip(stored_names, importances),
                key=lambda t: t[1],
                reverse=True,
            )
        }
