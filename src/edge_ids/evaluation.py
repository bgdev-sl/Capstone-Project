# Evaluate metrics

from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Return overall binary metrics, confusion matrix, and class report
def classification_metrics(y_true, y_pred) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': report
    }

# Return confidence score
def safe_confidence(model, transformed_X, predictions) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(transformed_X)
        return probabilities.max(axis=1)

    if hasattr(model, 'decision_function'):
        scores = model.decision_function(transformed_X)
        if np.ndim(scores) == 1:
            return 1.0 / (1.0 + np.exp(-np.abs(scores)))
        return np.max(scores, axis=1)
    return np.ones(len(predictions), dtype=float)
