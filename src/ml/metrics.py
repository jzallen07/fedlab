"""Evaluation metric helpers for HF trainer loops."""

from __future__ import annotations

from typing import Any, Protocol

import evaluate
import numpy as np
from transformers import EvalPrediction


class AccuracyMetric(Protocol):
    """Protocol for accuracy metric implementations."""

    def compute(self, *, predictions: np.ndarray, references: np.ndarray) -> dict[str, float]:
        """Compute accuracy metric."""


class _FallbackAccuracyMetric:
    """Fallback local metric when evaluate loading is unavailable."""

    def compute(self, *, predictions: np.ndarray, references: np.ndarray) -> dict[str, float]:
        preds = np.asarray(predictions)
        refs = np.asarray(references)
        if preds.shape != refs.shape:
            raise ValueError("predictions and references must have matching shapes")
        return {"accuracy": float((preds == refs).mean())}


def load_accuracy_metric() -> AccuracyMetric:
    """Load HF evaluate accuracy metric, with local fallback."""

    try:
        metric = evaluate.load("accuracy")
    except Exception:
        return _FallbackAccuracyMetric()

    return metric  # type: ignore[return-value]


def build_compute_metrics_fn(metric: AccuracyMetric | None = None):
    """Build a Trainer-compatible classification metric callback."""

    accuracy_metric = metric if metric is not None else load_accuracy_metric()
    fallback_metric = _FallbackAccuracyMetric()

    def _compute(eval_prediction: EvalPrediction) -> dict[str, float]:
        predictions: Any = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predicted_labels = np.argmax(np.asarray(predictions), axis=-1)
        references = np.asarray(eval_prediction.label_ids)
        try:
            result = accuracy_metric.compute(predictions=predicted_labels, references=references)
        except Exception:
            result = fallback_metric.compute(predictions=predicted_labels, references=references)
        accuracy = float(result.get("accuracy", 0.0))
        return {"accuracy": accuracy}

    return _compute
