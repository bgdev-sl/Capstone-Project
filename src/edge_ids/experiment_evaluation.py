"""Memory-bounded evaluation of registered models across large CSV datasets.

This module performs chunked *batch inference* and accumulates classification
metrics without retaining every row or prediction in memory. It does not train
models incrementally and does not change the existing batch-training workflow.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import psutil

from .data import infer_label_column, normalize_binary_labels
from .registry import ModelRegistry


CSV_FIELDS = [
    "status", "model_version", "model_name", "feature_reduction", "dataset",
    "dataset_path", "label_column", "rows_processed", "labeled_rows",
    "benign_support", "malicious_support", "predicted_benign", "predicted_malicious",
    "tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1",
    "precision_benign", "recall_benign", "f1_benign",
    "precision_malicious", "recall_malicious", "f1_malicious",
    "specificity", "balanced_accuracy", "false_positive_rate", "false_negative_rate",
    "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "mcc",
    "model_load_seconds", "dataset_read_seconds", "preprocess_seconds", "predict_seconds",
    "inference_seconds", "total_elapsed_seconds", "latency_ms_per_sample",
    "batch_latency_p50_ms", "batch_latency_p95_ms", "throughput_samples_per_second",
    "end_to_end_throughput_samples_per_second", "cpu_seconds_inference",
    "cpu_utilization_percent", "rss_before_evaluation_mb", "rss_after_dataset_mb",
    "rss_delta_mb", "sampled_peak_rss_mb", "batch_size", "batches_processed",
    "error_type", "error_message",
]


class BinaryMetricsAccumulator:
    """Accumulate binary classification counts across multiple prediction batches."""

    def __init__(self) -> None:
        self.tn = self.fp = self.fn = self.tp = 0

    def update(self, y_true: Iterable[Any], y_pred: Iterable[Any]) -> None:
        truth = np.asarray(y_true).reshape(-1)
        pred = np.asarray(y_pred).reshape(-1)
        if len(truth) != len(pred):
            raise ValueError("y_true and y_pred must have the same number of rows")
        if len(truth) == 0:
            return
        truth = truth.astype(np.int8, copy=False)
        pred = pred.astype(np.int8, copy=False)
        if not np.isin(truth, (0, 1)).all() or not np.isin(pred, (0, 1)).all():
            raise ValueError("Binary metrics require labels and predictions encoded as 0/1")
        self.tn += int(np.count_nonzero((truth == 0) & (pred == 0)))
        self.fp += int(np.count_nonzero((truth == 0) & (pred == 1)))
        self.fn += int(np.count_nonzero((truth == 1) & (pred == 0)))
        self.tp += int(np.count_nonzero((truth == 1) & (pred == 1)))

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    def compute(self) -> dict[str, Any]:
        tn, fp, fn, tp = self.tn, self.fp, self.fn, self.tp
        n = tn + fp + fn + tp
        if n == 0:
            return {key: None for key in (
                "benign_support", "malicious_support", "predicted_benign", "predicted_malicious",
                "tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1",
                "precision_benign", "recall_benign", "f1_benign",
                "precision_malicious", "recall_malicious", "f1_malicious",
                "specificity", "balanced_accuracy", "false_positive_rate", "false_negative_rate",
                "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "mcc",
            )}

        precision_pos = self._ratio(tp, tp + fp)
        recall_pos = self._ratio(tp, tp + fn)
        f1_pos = self._ratio(2 * precision_pos * recall_pos, precision_pos + recall_pos)
        precision_neg = self._ratio(tn, tn + fn)
        recall_neg = self._ratio(tn, tn + fp)  # specificity for benign class
        f1_neg = self._ratio(2 * precision_neg * recall_neg, precision_neg + recall_neg)
        specificity = recall_neg
        mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = self._ratio(tp * tn - fp * fn, mcc_denominator)
        support_neg, support_pos = tn + fp, tp + fn

        return {
            "benign_support": support_neg,
            "malicious_support": support_pos,
            "predicted_benign": tn + fn,
            "predicted_malicious": tp + fp,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": self._ratio(tn + tp, n),
            # Backward-compatible headline metrics use malicious (positive) class.
            "precision": precision_pos,
            "recall": recall_pos,
            "f1": f1_pos,
            "precision_benign": precision_neg,
            "recall_benign": recall_neg,
            "f1_benign": f1_neg,
            "precision_malicious": precision_pos,
            "recall_malicious": recall_pos,
            "f1_malicious": f1_pos,
            "specificity": specificity,
            "balanced_accuracy": (recall_pos + specificity) / 2,
            "false_positive_rate": self._ratio(fp, fp + tn),
            "false_negative_rate": self._ratio(fn, fn + tp),
            "precision_macro": (precision_neg + precision_pos) / 2,
            "recall_macro": (recall_neg + recall_pos) / 2,
            "f1_macro": (f1_neg + f1_pos) / 2,
            "f1_weighted": self._ratio(f1_neg * support_neg + f1_pos * support_pos, n),
            "mcc": mcc,
        }


def discover_datasets(
    datasets: list[str] | None = None,
    dataset_dir: str | None = None,
    pattern: str = "*.csv",
) -> list[Path]:
    """Resolve input CSVs from explicit paths or a directory glob."""
    if datasets:
        paths = [Path(p).expanduser().resolve() for p in datasets]
    elif dataset_dir:
        folder = Path(dataset_dir).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"Dataset directory does not exist: {folder}")
        paths = sorted(p.resolve() for p in folder.glob(pattern) if p.is_file())
    else:
        raise ValueError("Supply --datasets or --dataset-dir")
    if not paths:
        raise FileNotFoundError("No CSV datasets matched the supplied paths/pattern")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Dataset file(s) not found: " + ", ".join(missing))
    # Dedupe but preserve first-seen order.
    return list(dict.fromkeys(paths))


def discover_model_versions(model_dir: str | Path) -> list[str]:
    """Return registered model versions having both a manifest and artifact."""
    folder = Path(model_dir)
    versions: list[str] = []
    for manifest_path in sorted(folder.glob("*.json")):
        if (folder / f"{manifest_path.stem}.joblib").is_file():
            versions.append(manifest_path.stem)
    return versions


def _select_features(bundle: dict[str, Any], chunk: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the raw input features required by a trained bundle."""
    features = bundle.get("features")
    metadata = getattr(features, "metadata", None)
    if metadata is not None:
        expected = list(dict.fromkeys(metadata.numeric_columns + metadata.categorical_columns))
    else:
        saved = bundle.get("metadata", {}).get("preprocessing_metadata", {})
        expected = list(dict.fromkeys(
            saved.get("numeric_columns", []) + saved.get("categorical_columns", [])
        ))
    if not expected:
        raise ValueError("Model bundle does not declare its required raw feature columns")
    missing = [column for column in expected if column not in chunk.columns]
    if missing:
        raise ValueError(
            "Dataset is missing model input feature(s): " + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )
    selected = chunk.loc[:, expected].copy(deep=False)
    numeric_columns = getattr(metadata, "numeric_columns", None) if metadata is not None else None
    if numeric_columns is None:
        saved = bundle.get("metadata", {}).get("preprocessing_metadata", {})
        numeric_columns = saved.get("numeric_columns", [])
    # Enforce the numeric schema learned at training time. Malformed strings in
    # numeric columns become missing values and are handled by the fitted imputer.
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected


def _resolve_label_column(columns: Iterable[str], configured_label: str | None,
                          allow_unlabeled: bool) -> str | None:
    header = pd.DataFrame(columns=list(columns))
    try:
        return infer_label_column(header, configured_label)
    except ValueError:
        if allow_unlabeled and not configured_label:
            return None
        raise


def evaluate_dataset(
    dataset_path: str | Path,
    bundle: dict[str, Any],
    manifest: dict[str, Any],
    *,
    label_column: str | None = None,
    allow_unlabeled: bool = False,
    batch_size: int = 10_000,
    model_load_seconds: float = 0.0,
) -> dict[str, Any]:
    """Evaluate one model on one CSV in bounded row batches.

    Preprocessing and model prediction are timed separately. Classification
    metrics are accumulated from per-batch confusion counts rather than storing
    the entire labels/predictions arrays in memory.
    """
    path = Path(dataset_path).resolve()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    start_total = time.perf_counter()
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024 * 1024)
    sampled_peak = rss_before
    read_seconds = preprocess_seconds = predict_seconds = inference_seconds = cpu_seconds = 0.0
    batch_times_ms: list[float] = []
    accumulator = BinaryMetricsAccumulator()
    rows_processed = labeled_rows = batches_processed = 0

    header = pd.read_csv(path, nrows=0, low_memory=False)
    target_column = _resolve_label_column(header.columns, label_column, allow_unlabeled)
    if target_column and target_column not in header.columns:
        raise ValueError(f"Label column '{target_column}' is not present in {path.name}")

    # Check that required raw features exist before the potentially long pass.
    model_features = bundle["features"]
    prep_meta = getattr(model_features, "metadata", None)
    if prep_meta is not None:
        required = set(prep_meta.numeric_columns + prep_meta.categorical_columns)
    else:
        saved = bundle.get("metadata", {}).get("preprocessing_metadata", {})
        required = set(saved.get("numeric_columns", []) + saved.get("categorical_columns", []))
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"Dataset {path.name} is missing model feature columns: {missing[:20]}")

    read_started = time.perf_counter()
    reader = pd.read_csv(path, chunksize=batch_size, low_memory=False)
    while True:
        try:
            chunk = next(reader)
        except StopIteration:
            read_seconds += time.perf_counter() - read_started
            break
        read_seconds += time.perf_counter() - read_started
        if chunk.empty:
            read_started = time.perf_counter()
            continue

        y = None
        labelled_mask = None
        if target_column is not None:
            label_values = chunk[target_column]
            label_present = label_values.notna()
            if pd.api.types.is_object_dtype(label_values.dtype) or pd.api.types.is_string_dtype(label_values.dtype):
                label_present &= label_values.astype(str).str.strip().ne("")
            labelled_mask = label_present.to_numpy()
            if labelled_mask.any():
                # Normalize only non-null label values; truly unlabeled rows can
                # still be predicted, but are not counted as evaluation examples.
                y = normalize_binary_labels(label_values.loc[labelled_mask]).to_numpy(dtype=np.int8)

        X = _select_features(bundle, chunk)
        wall_before = time.perf_counter()
        cpu_before = process.cpu_times()
        transform_start = time.perf_counter()
        transformed = model_features.transform(X)
        preprocess_seconds += time.perf_counter() - transform_start
        sampled_peak = max(sampled_peak, process.memory_info().rss / (1024 * 1024))

        predict_start = time.perf_counter()
        prediction = np.asarray(bundle["model"].predict(transformed)).reshape(-1)
        predict_seconds += time.perf_counter() - predict_start
        sampled_peak = max(sampled_peak, process.memory_info().rss / (1024 * 1024))
        cpu_after = process.cpu_times()
        wall_after = time.perf_counter()

        chunk_wall = wall_after - wall_before
        inference_seconds += chunk_wall
        cpu_seconds += (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
        batch_times_ms.append(chunk_wall * 1000.0)
        if len(prediction) != len(chunk):
            raise RuntimeError(
                f"Prediction count mismatch for {path.name}: got {len(prediction)} for {len(chunk)} rows"
            )
        if y is not None and labelled_mask is not None:
            accumulator.update(y, prediction[labelled_mask])
            labeled_rows += len(y)
        rows_processed += len(chunk)
        batches_processed += 1
        del chunk, X, transformed, prediction, y, labelled_mask
        sampled_peak = max(sampled_peak, process.memory_info().rss / (1024 * 1024))
        read_started = time.perf_counter()

    total_seconds = time.perf_counter() - start_total
    rss_after = process.memory_info().rss / (1024 * 1024)
    metric_values = accumulator.compute()
    meta = manifest.get("metadata", {})
    result: dict[str, Any] = {
        "status": "ok" if target_column is not None and labeled_rows else "ok_unlabeled",
        "model_version": manifest.get("model_version", path.stem),
        "model_name": meta.get("model_name", "unknown"),
        "feature_reduction": meta.get("feature_reduction", "unknown"),
        "dataset": path.name,
        "dataset_path": str(path),
        "label_column": target_column or "",
        "rows_processed": rows_processed,
        "labeled_rows": labeled_rows,
        **metric_values,
        "model_load_seconds": model_load_seconds,
        "dataset_read_seconds": read_seconds,
        "preprocess_seconds": preprocess_seconds,
        "predict_seconds": predict_seconds,
        "inference_seconds": inference_seconds,
        "total_elapsed_seconds": total_seconds,
        "latency_ms_per_sample": (inference_seconds * 1000 / rows_processed) if rows_processed else None,
        "batch_latency_p50_ms": float(np.percentile(batch_times_ms, 50)) if batch_times_ms else None,
        "batch_latency_p95_ms": float(np.percentile(batch_times_ms, 95)) if batch_times_ms else None,
        "throughput_samples_per_second": (rows_processed / inference_seconds) if inference_seconds else None,
        "end_to_end_throughput_samples_per_second": (rows_processed / total_seconds) if total_seconds else None,
        "cpu_seconds_inference": cpu_seconds,
        "cpu_utilization_percent": (100.0 * cpu_seconds / inference_seconds) if inference_seconds else None,
        "rss_before_evaluation_mb": rss_before,
        "rss_after_dataset_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
        "sampled_peak_rss_mb": sampled_peak,
        "batch_size": batch_size,
        "batches_processed": batches_processed,
        "error_type": "",
        "error_message": "",
    }
    return result


def error_result(model_version: str, dataset: Path, error: Exception, batch_size: int) -> dict[str, Any]:
    """Build a CSV row for a failed model/dataset pair so batch jobs can continue."""
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "status": "error",
        "model_version": model_version,
        "dataset": dataset.name,
        "dataset_path": str(dataset.resolve()),
        "batch_size": batch_size,
        "error_type": type(error).__name__,
        "error_message": str(error).replace("\n", " ")[:2000],
    })
    return row


def write_csv_row(handle, row: dict[str, Any]) -> None:
    """Write one result row with a stable schema and flush it as a checkpoint."""
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})
    handle.flush()
    os.fsync(handle.fileno())


def prepare_resume_file(csv_path: Path) -> set[str]:
    """Keep one successful row per dataset and discard failed rows before retrying.

    This prevents duplicate dataset rows when a previous run was resumed after
    partial failures. Successful rows are preserved; failed/malformed rows are
    retried in the new worker pass.
    """
    if not csv_path.exists():
        return set()
    successful: dict[str, dict[str, str]] = {}
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = row.get("dataset_path", "")
                if key and row.get("status") in {"ok", "ok_unlabeled"}:
                    successful[key] = row
    except (OSError, csv.Error):
        successful = {}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in successful.values():
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
    return set(successful)


def read_completed_paths(csv_path: Path) -> set[str]:
    """Return successful paths in a result CSV (including an empty set on failure)."""
    completed: set[str] = set()
    if not csv_path.exists():
        return completed
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") in {"ok", "ok_unlabeled"} and row.get("dataset_path"):
                    completed.add(row["dataset_path"])
    except (OSError, csv.Error):
        return set()
    return completed
