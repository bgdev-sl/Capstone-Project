"""Dataset loading and train/test preparation.

The research design uses CICIoT2023 as the primary dataset and ToN-IoT as an
optional secondary test set. This module therefore avoids hard-coding a single
CSV schema and instead supports configurable label and excluded columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_CANDIDATES = (
    "label",
    "Label",
    "attack",
    "Attack",
    "target",
    "Target",
    "class",
    "Class",
)


@dataclass
class DatasetSplit:
    """Container for a reproducible train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def infer_label_column(frame: pd.DataFrame, configured_label: str | None = None) -> str:
    """Determine the label column from configuration or common dataset names."""
    if configured_label:
        if configured_label not in frame.columns:
            raise ValueError(
                f"Configured label column '{configured_label}' was not found in the dataset. "
                f"Available columns include: {list(frame.columns)[:20]}"
            )
        return configured_label

    for candidate in LABEL_CANDIDATES:
        if candidate in frame.columns:
            return candidate

    raise ValueError(
        "Unable to infer the label column. Set data.label_column in configs/config.yaml."
    )


def normalize_binary_labels(labels: pd.Series) -> pd.Series:
    """Normalize common benign/malicious labels to a binary 0/1 target.

    A label containing 'benign' is mapped to 0. Any other string label is mapped
    to 1. Numeric labels are preserved when they are already binary; otherwise
    values greater than zero are treated as malicious.
    """
    # Numeric labels are sometimes stored as strings ("0" / "1") in CSVs.
    # Recognize those before applying the benign/malicious string convention.
    numeric = pd.to_numeric(labels, errors="coerce")
    if pd.api.types.is_numeric_dtype(labels) or numeric.notna().all():
        numeric = numeric.fillna(0)
        unique = set(pd.unique(numeric))
        if unique.issubset({0, 1}):
            return numeric.astype(int)
        return (numeric > 0).astype(int)

    text = labels.astype(str).str.strip().str.lower()
    return (~text.str.contains("benign", regex=False)).astype(int)


def _reservoir_sample_csv(
    path: Path,
    sample_rows: int,
    random_state: int = 42,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Uniformly sample at most ``sample_rows`` from a large CSV without loading it all.

    Independent random priorities are assigned to each row; the sample retains
    the ``sample_rows`` smallest priorities. Memory is bounded by one input chunk
    plus the requested sample. This is a sampled-data path, not full-data training.
    """
    if sample_rows < 1:
        raise ValueError("max_rows must be positive when supplied.")
    if chunk_size < 1:
        raise ValueError("sampling_chunk_rows must be a positive integer.")
    rng = np.random.default_rng(random_state)
    reservoir = pd.DataFrame()
    priorities = np.empty(0, dtype=np.float64)

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        if chunk.empty:
            continue
        chunk_priorities = rng.random(len(chunk))
        combined = pd.concat([reservoir, chunk], ignore_index=True)
        combined_priorities = np.concatenate([priorities, chunk_priorities])
        if len(combined) > sample_rows:
            selected = np.argpartition(combined_priorities, sample_rows - 1)[:sample_rows]
            reservoir = combined.iloc[selected].reset_index(drop=True)
            priorities = combined_priorities[selected]
        else:
            reservoir = combined
            priorities = combined_priorities

    return reservoir.reset_index(drop=True)


def load_dataset(
    path: str | Path,
    label_column: str | None = None,
    drop_columns: Iterable[str] = (),
    max_rows: int | None = None,
    random_state: int = 42,
    sampling_chunk_rows: int = 10_000,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load a CSV for batch fitting, optionally selecting a bounded reservoir sample.

    When ``max_rows`` is set, CSV chunks are used only to select a representative
    in-memory sample. The selected sample is then passed to ordinary batch training;
    this function does not provide incremental/online model fitting.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if max_rows is not None:
        # Do not load the complete multi-gigabyte CSV merely to sample it. The
        # reservoir sampler keeps memory bounded by one input chunk + max_rows.
        frame = _reservoir_sample_csv(
            path,
            max_rows,
            random_state=random_state,
            chunk_size=sampling_chunk_rows,
        )
    else:
        frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError("The supplied dataset is empty.")

    label_name = infer_label_column(frame, label_column)
    y = normalize_binary_labels(frame[label_name])

    excluded = set(drop_columns)
    excluded.discard(label_name)
    X = frame.drop(columns=[label_name, *[c for c in excluded if c in frame.columns]])

    # Remove all-empty columns because they have no predictive value and can cause
    # downstream imputers to behave unexpectedly.
    X = X.dropna(axis=1, how="all")

    return X, y, label_name


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DatasetSplit:
    """Create a stratified, reproducible train/test split."""
    if y.nunique() < 2:
        raise ValueError("The target must contain at least two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
