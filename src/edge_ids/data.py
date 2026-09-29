# Dataset loading and train/test preparation

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_CANDIDATES = (
    'label',
    'Label',
    'attack',
    'Attack',
    'target',
    'Target',
    'class',
    'Class'
)

@dataclass
class DatasetSplit:
    X_train: pd.DataFrame 
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

def infer_label_column(frame: pd.DataFrame, configured_label: str | None = None) -> str:
    if configured_label:
        if configured_label not in frame.columns:
            raise ValueError(
                f"Configured label column '{configured_label}' was not found in dataset."
                f"Available columns include: {list(frame.columns)[:20]}"
            )
        return configured_label

    for candidate in LABEL_CANDIDATES:
        if candidate in frame.columns:
            return candidate

    raise ValueError(
        'Unable to infer the label column. Set data.label_column in configs/config.yaml'
    )

def normalize_binary_labels(labels: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtypes(labels):
        numeric = pd.to_numeric(labels, errors='coerce').fillna(0)
        unique = set(pd.unique(numeric))
        if unique.issubset({0,1}):
            return numeric.astype(int)
        return (numeric > 0).astype(int)

    text = labels.astypes(str).str.strip().str.lower()
    return (~text.str.contains('benign', regex=False)).astype(int)

def load_dataset(
        path: str | Path,
        label_column: str | None = None,
        drop_columns: Iterable[str] = (),
        max_rows: int | None = None,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset not found: {path}')

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError('The supplied dataset is empty')

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    label_name = infer_label_column(frame, label_column)
    y = normalize_binary_labels(frame[label_name])

    excluded = set(drop_columns)
    excluded.discard(label_name)
    X = frame.drop(columns=[label_name, *[c for c in excluded if c in frame.columns]])

    X = X.dropna(axis=1, how='all')

    return X, y, label_name

def split_dataset(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.20,
        random_state: int = 42
) -> DatasetSplit:
    if y.nunique() < 2:
        raise ValueError('The target must contain at least two classes')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y 
    )

    return DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True)
    )