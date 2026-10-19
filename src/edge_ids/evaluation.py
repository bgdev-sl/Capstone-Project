"""Evaluation metrics used by the capstone experiment."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred) -> dict[str, Any]:
    """Return overall binary metrics plus the confusion matrix and class report."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
    }


def safe_confidence(model, transformed_X, predictions) -> np.ndarray:
    """Return a confidence-like score when the estimator exposes probabilities."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_X)
        return probabilities.max(axis=1)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(transformed_X)
        if np.ndim(scores) == 1:
            # Convert signed scores to a simple [0, 1] confidence-like quantity.
            return 1.0 / (1.0 + np.exp(-np.abs(scores)))
        return np.max(scores, axis=1)

    return np.ones(len(predictions), dtype=float)
