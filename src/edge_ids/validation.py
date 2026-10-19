"""Validation helpers for experiment outputs.

These checks are intentionally conservative: they validate that a benchmark
result file is structurally complete and contains values in plausible ranges,
without interpreting which model is preferable for the research question.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_BENCHMARK_COLUMNS = {
    "model",
    "feature_reduction",
    "model_version",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "latency_ms_per_sample",
    "cpu_percent",
    "rss_after_mb",
    "rss_delta_mb",
    "output_feature_count",
    "train_rows",
    "test_rows",
    "model_size_bytes",
}


class OutputValidationError(ValueError):
    """Raised when an experiment output does not satisfy the expected schema."""


def validate_metric_range(name: str, value: float) -> None:
    """Validate a classification metric that should be within [0, 1]."""
    if not 0.0 <= float(value) <= 1.0:
        raise OutputValidationError(f"{name} must be between 0 and 1; received {value!r}.")


def validate_benchmark_frame(
    frame: pd.DataFrame,
    expected_models: Iterable[str] = ("decision_tree", "random_forest", "xgboost", "mlp"),
) -> None:
    """Validate the schema and basic value ranges of a benchmark DataFrame."""
    missing = REQUIRED_BENCHMARK_COLUMNS.difference(frame.columns)
    if missing:
        raise OutputValidationError(f"Benchmark output is missing columns: {sorted(missing)}")

    if frame.empty:
        raise OutputValidationError("Benchmark output contains no experiment rows.")

    expected = set(expected_models)
    observed = set(frame["model"].dropna().astype(str))
    unknown = observed.difference(expected)
    if unknown:
        raise OutputValidationError(f"Unknown model names in benchmark output: {sorted(unknown)}")

    for column in ("accuracy", "precision", "recall", "f1"):
        for value in frame[column].dropna():
            validate_metric_range(column, float(value))

    for column in ("latency_ms_per_sample", "rss_after_mb", "model_size_bytes"):
        if (pd.to_numeric(frame[column], errors="coerce") < 0).any():
            raise OutputValidationError(f"{column} contains a negative value.")

    if (pd.to_numeric(frame["output_feature_count"], errors="coerce") <= 0).any():
        raise OutputValidationError("output_feature_count must be greater than zero.")

    if (pd.to_numeric(frame["train_rows"], errors="coerce") <= 0).any():
        raise OutputValidationError("train_rows must be greater than zero.")

    if (pd.to_numeric(frame["test_rows"], errors="coerce") <= 0).any():
        raise OutputValidationError("test_rows must be greater than zero.")


def validate_benchmark_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a benchmark CSV, returning the validated frame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark output not found: {path}")
    frame = pd.read_csv(path)
    validate_benchmark_frame(frame)
    return frame
