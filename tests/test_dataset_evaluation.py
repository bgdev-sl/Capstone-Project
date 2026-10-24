"""Tests for large multi-dataset model evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.experiment_evaluation import (
    BinaryMetricsAccumulator,
    discover_datasets,
    evaluate_dataset,
)
from edge_ids.preprocessing import FeaturePipeline


def test_binary_metrics_accumulator_matches_known_confusion_counts():
    accumulator = BinaryMetricsAccumulator()
    accumulator.update([0, 0, 1], [0, 1, 1])
    accumulator.update([1, 1, 0, 1], [0, 0, 0, 1])
    result = accumulator.compute()

    assert result["tn"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 2
    assert result["tp"] == 2
    assert result["accuracy"] == pytest.approx(4 / 7)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1 / 2)
    assert result["f1"] == pytest.approx(4 / 7)
    assert result["specificity"] == pytest.approx(2 / 3)
    assert result["f1_benign"] == pytest.approx(4 / 7)
    assert result["f1_malicious"] == pytest.approx(4 / 7)


def _make_test_bundle(rows: int = 60):
    idx = np.arange(rows)
    X = pd.DataFrame({
        "flow_bytes": ((idx * 17) % 101).astype(float),
        "packet_count": ((idx * 3) % 19).astype(float),
        "protocol": np.where(idx % 2 == 0, "TCP", "UDP"),
    })
    y = pd.Series(((X["flow_bytes"].to_numpy() > 48) | (X["packet_count"].to_numpy() > 12)).astype(int))
    features = FeaturePipeline().fit(X, y)
    transformed = features.transform(X)
    model = DecisionTreeClassifier(random_state=7).fit(transformed, y)
    metadata = {
        "model_name": "decision_tree",
        "model_version": "test_tree",
        "feature_reduction": False,
        "preprocessing_metadata": {
            "numeric_columns": features.metadata.numeric_columns,
            "categorical_columns": features.metadata.categorical_columns,
        },
    }
    return X, y, {"model": model, "features": features, "metadata": metadata}, {
        "model_version": "test_tree", "metadata": metadata,
    }


def test_evaluate_dataset_processes_multiple_batches_and_reports_metrics(tmp_path):
    X, y, bundle, manifest = _make_test_bundle(61)
    frame = X.copy()
    frame["Label"] = np.where(y.to_numpy() == 1, "DDoS", "Benign")
    path = tmp_path / "dataset_one.csv"
    frame.to_csv(path, index=False)

    result = evaluate_dataset(path, bundle, manifest, batch_size=13, label_column="Label")

    assert result["status"] == "ok"
    assert result["rows_processed"] == 61
    assert result["labeled_rows"] == 61
    assert result["batches_processed"] == 5
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["throughput_samples_per_second"] > 0
    assert result["sampled_peak_rss_mb"] >= result["rss_before_evaluation_mb"]


def test_evaluate_dataset_supports_unlabeled_dataset_when_enabled(tmp_path):
    X, _, bundle, manifest = _make_test_bundle(30)
    path = tmp_path / "unlabeled.csv"
    X.to_csv(path, index=False)

    result = evaluate_dataset(path, bundle, manifest, batch_size=7, allow_unlabeled=True)
    assert result["status"] == "ok_unlabeled"
    assert result["rows_processed"] == 30
    assert result["labeled_rows"] == 0
    assert result["accuracy"] is None


def test_evaluate_dataset_rejects_missing_required_features(tmp_path):
    X, y, bundle, manifest = _make_test_bundle(12)
    path = tmp_path / "missing_feature.csv"
    pd.DataFrame({"Label": np.where(y == 1, "DDoS", "Benign"), "flow_bytes": X["flow_bytes"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing model feature columns"):
        evaluate_dataset(path, bundle, manifest, label_column="Label", batch_size=4)


def test_discover_datasets_accepts_a_directory_glob(tmp_path):
    (tmp_path / "one.csv").write_text("label,x\n0,1\n", encoding="utf-8")
    (tmp_path / "two.csv").write_text("label,x\n1,2\n", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("not csv", encoding="utf-8")
    datasets = discover_datasets(dataset_dir=str(tmp_path), pattern="*.csv")
    assert [p.name for p in datasets] == ["one.csv", "two.csv"]


def test_resume_file_preserves_successes_and_removes_prior_errors(tmp_path):
    import csv
    from edge_ids.experiment_evaluation import CSV_FIELDS, prepare_resume_file, write_csv_row

    output = tmp_path / "model.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow({"status": "ok", "model_version": "m1", "dataset_path": "/data/a.csv", "dataset": "a.csv"})
        writer.writerow({"status": "error", "model_version": "m1", "dataset_path": "/data/b.csv", "dataset": "b.csv"})
        writer.writerow({"status": "ok", "model_version": "m1", "dataset_path": "/data/a.csv", "dataset": "a.csv"})

    completed = prepare_resume_file(output)
    assert completed == {"/data/a.csv"}
    with output.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["dataset_path"] == "/data/a.csv"
    assert rows[0]["status"] == "ok"
