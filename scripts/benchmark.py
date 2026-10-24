#!/usr/bin/env python3
"""Benchmark the four in-scope models with and without feature reduction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.config import load_config, ensure_directories  # noqa: E402
from edge_ids.data import load_dataset  # noqa: E402
from edge_ids.models import SUPPORTED_MODELS  # noqa: E402
from edge_ids.registry import ModelRegistry  # noqa: E402
from edge_ids.training import train_one  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--sampling-chunk-rows",
        type=int,
        default=None,
        help="Rows read per CSV chunk while selecting --max-rows (default: data.sampling_chunk_rows, 10000).",
    )
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS))
    parser.add_argument("--reduced-k", type=int, default=None)
    parser.add_argument("--output", default="results/benchmark.csv")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-reduced", action="store_true")
    parser.add_argument(
        "--allow-large-batch",
        action="store_true",
        help="Override the safety guard for benchmarking a CSV larger than 2 GiB without --max-rows (may exhaust RAM).",
    )
    return parser.parse_args()

def next_model_version(registry, model_name: str, suffix: str) -> str:
    """
    Return the next unused version identifier for a model/configuration.

    Existing model artifacts and manifests are preserved. Version numbering
    is independent for each model and feature configuration.
    """
    existing_versions = {
        str(item.get("model_version"))
        for item in registry.list_models()
        if item.get("model_version")
    }

    version_num = 1

    while True:
        candidate = f"{model_name}_v{version_num}_{suffix}"

        # Check both files in case an earlier run left a partial artifact.
        artifact_path = registry.model_dir / f"{candidate}.joblib"
        manifest_path = registry.model_dir / f"{candidate}.json"

        if (
            candidate not in existing_versions
            and not artifact_path.exists()
            and not manifest_path.exists()
        ):
            return candidate

        version_num += 1

def main():
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config, ROOT)

    data_cfg = config.get("data", {})
    max_rows = args.max_rows if args.max_rows is not None else data_cfg.get("max_rows")
    dataset_path = Path(args.dataset)
    dataset_size = dataset_path.stat().st_size if dataset_path.exists() else 0
    if max_rows is None and dataset_size > 2 * 1024**3 and not args.allow_large_batch:
        raise SystemExit(
            f"Dataset is {dataset_size / (1024**3):.2f} GiB. The four-model benchmark uses batch "
            "estimators and may exhaust RAM. Pass --max-rows N to benchmark on a bounded reservoir "
            "sample (use the same persisted sample for reproducible comparisons), or explicitly "
            "override with --allow-large-batch only if your RAM is sufficient."
        )
    X, y, label_column = load_dataset(
        args.dataset,
        label_column=data_cfg.get("label_column"),
        drop_columns=data_cfg.get("drop_columns", []),
        max_rows=max_rows,
        random_state=config.get("project", {}).get("random_state", 42),
        sampling_chunk_rows=(
            args.sampling_chunk_rows
            if args.sampling_chunk_rows is not None
            else int(data_cfg.get("sampling_chunk_rows", 10_000))
        ),
    )

    reduction_cfg = config.get("feature_reduction", {})
    reduced_k = args.reduced_k if args.reduced_k is not None else reduction_cfg.get("k", 64)

    registry = ModelRegistry(ROOT / config.get("paths", {}).get("model_dir", "models"))
    rows = []

    experiments = []
    if not args.skip_full:
        experiments.append((False, None))
    if not args.skip_reduced:
        experiments.append((True, reduced_k))

    for model_name in args.models:
        for reduced, k in experiments:
            suffix = f"reduced{k}" if reduced else "full"
            # Find an unused version so existing models are not overwritten.
            version = next_model_version(
                registry,
                model_name,
                suffix,
            )
            result = train_one(
                model_name,
                X,
                y,
                config,
                version,
                feature_reduction=reduced,
                feature_k=k or reduced_k,
                registry=registry,
            )
            row = {
                "model": model_name,
                "feature_reduction": reduced,
                "feature_k": k,
                "model_version": version,
                "accuracy": result["metrics"]["accuracy"],
                "precision": result["metrics"]["precision"],
                "recall": result["metrics"]["recall"],
                "f1": result["metrics"]["f1"],
                "latency_ms_per_sample": result["inference_resource_metrics"]["latency_ms_per_sample"],
                "throughput_samples_per_second": result["inference_resource_metrics"]["throughput_samples_per_second"],
                "cpu_percent": result["inference_resource_metrics"]["cpu_percent"],
                "rss_after_mb": result["inference_resource_metrics"]["rss_after_mb"],
                "rss_delta_mb": result["inference_resource_metrics"]["rss_delta_mb"],
                "mean_confidence": result["confidence_mean"],
                "output_feature_count": result["output_feature_count"],
                "train_rows": result["train_rows"],
                "test_rows": result["test_rows"],
                "inference_repeats": result.get("inference_repeats", config.get("evaluation", {}).get("inference_repeats", 20)),
                "model_size_bytes": result["manifest"]["artifact_size_bytes"],
                "label_column": label_column,
            }
            rows.append(row)
            print(f"Completed {version}")

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Benchmark results written to {output}")


if __name__ == "__main__":
    main()
