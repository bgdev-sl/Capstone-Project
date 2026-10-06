# Validating benchmark result file:
#   - structurally complete
#   - values are plausible

from __future__ import annotations
from pathlib import Path 
from typing import Iterable 
import pandas as pd 

REQUIRED_BENCHMARK_COLUMNS = {
    'model',
    'feature_reduction',
    'model_version',
    'accuracy',
    'precision',
    'recall',
    'f1',
    'latency_ms_per_sample',
    'cpu_percent',
    'rss_after_mb',
    'rss_delta_mb',
    'output_feature_count',
    'train_rows',
    'test_rows',
    'model_size_bytes'
}

# Raise when output does not satisfy expected schema
class OutputValidationError(ValueError):
    pass

    # validate classification metric (should be between 0 and 1)
def validate_metric_range(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise OutputValidationError(f'{name} must be between 0 and 1; received {value!r}')

    # Validate schema and basic value ranges
def validate_benchmark_frame(frame: pd.DataFrame, expected_models: Iterable[str] = ('decision_tree', 'random_forest', 'xgboost', 'mlp')) -> None:
    missing = REQUIRED_BENCHMARK_COLUMNS.difference(frame.columns)
        
    if missing:
        raise OutputValidationError(f'Benchmark output is missing columns: {sorted(missing)}')

    if frame.empty:
        raise OutputValidationError(f'Benchmark output contains no experiement rows.')

    expected = set(expected_models)
    observed = set(frame['model'].dropna().astype(str))
    unknown = observed.difference(expected)

    if unknown:
        raise OutputValidationError(f'Unknown model names in benchmark output: {sorted(unknown)}')

    for column in ('accuracy', 'precision', 'recall', 'f1'):
        for value in frame[column].dropna():
            validate_metric_range(column, float(value))

    for column in ('latency_ms_per_sample', 'rss_after_mb', 'model_size_bytes'):
        if (pd.to_numeric(frame[column], errors='coerce') < 0).any():
            raise OutputValidationError(f'{column} contains a negative value.')

    if (pd.to_numeric(frame['output_feature_count'], errors='coerce') <= 0).any():
        raise OutputValidationError('output_feature_count must be greater than zero.')

    if (pd.to_numeric(frame['train_rows'], errors='coerce') <= 0).any():
        raise OutputValidationError('train_rows must be greater than zero.')

    if (pd.to_numeric(frame['test_rows'], errors='coerce') <= 0).any():
        raise OutputValidationError('test_rows must be greater than zero.')

# load and validate benchmark CSV and return validated frame
def validate_benchmark_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Benchmark output not found: {path}')
    frame = pd.read_csv(path)
    validate_benchmark_frame(frame)
    return frame 