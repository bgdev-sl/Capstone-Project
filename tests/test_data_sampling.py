"""Tests for bounded CSV sampling used by batch-only training."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.data import load_dataset, normalize_binary_labels


def _write_dataset(path: Path, rows: int = 500) -> pd.DataFrame:
    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "label": np.where(index % 2 == 0, "Benign", "DDoS"),
            "row_id": index,
            "Flow_ID": [f"flow-{i:06d}" for i in index],
            "duration": (index % 13).astype(float),
            "packet_count": ((index * 3) % 17).astype(float),
            "protocol": np.where(index % 3 == 0, "TCP", np.where(index % 3 == 1, "UDP", "ICMP")),
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_normalize_numeric_string_labels():
    assert normalize_binary_labels(pd.Series(["0", "1", "1", "0"])).tolist() == [0, 1, 1, 0]


def test_load_dataset_max_rows_uses_bounded_reservoir_sample(tmp_path):
    path = tmp_path / "flows.csv"
    _write_dataset(path, rows=500)
    X, y, label = load_dataset(
        path,
        label_column="label",
        drop_columns=["Flow_ID"],
        max_rows=60,
        random_state=42,
        sampling_chunk_rows=37,
    )
    assert len(X) == 60
    assert len(y) == 60
    assert label == "label"
    assert set(y.unique()) == {0, 1}
    # The sample should represent rows across the file, not just its first 60 rows.
    assert X["row_id"].max() > 60


def test_reservoir_sample_is_reproducible_for_fixed_seed(tmp_path):
    path = tmp_path / "flows.csv"
    _write_dataset(path, rows=500)
    first_X, first_y, _ = load_dataset(
        path, label_column="label", max_rows=80, random_state=19, sampling_chunk_rows=41
    )
    second_X, second_y, _ = load_dataset(
        path, label_column="label", max_rows=80, random_state=19, sampling_chunk_rows=41
    )
    pd.testing.assert_frame_equal(first_X, second_X)
    pd.testing.assert_series_equal(first_y, second_y)


def test_reservoir_sampler_rejects_nonpositive_chunk_size(tmp_path):
    path = tmp_path / "flows.csv"
    _write_dataset(path, rows=100)
    import pytest

    with pytest.raises(ValueError, match="sampling_chunk_rows must be a positive integer"):
        load_dataset(path, label_column="label", max_rows=20, sampling_chunk_rows=0)


def test_batch_cli_exposes_sampling_controls_but_not_incremental_mode():
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_models.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--max-rows" in completed.stdout
    assert "--sampling-chunk-rows" in completed.stdout
    assert "--incremental" not in completed.stdout
    assert "--chunksize" not in completed.stdout
