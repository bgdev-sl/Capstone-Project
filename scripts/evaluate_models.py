#!/usr/bin/env python3
"""Batch-evaluate every registered model over many labeled CSV datasets.

Default operation isolates one model version per subprocess. This keeps model
memory measurements more comparable than loading all eight estimators at once.
Inference is chunked to conserve RAM; model training remains ordinary batch
training and no predictions are written per-row unless a separate export is added.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Support running the script from a source checkout without requiring an editable
# install, while remaining compatible with the normal pip-install workflow.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edge_ids.config import load_config
from edge_ids.experiment_evaluation import (
    CSV_FIELDS,
    discover_datasets,
    discover_model_versions,
    error_result,
    evaluate_dataset,
    prepare_resume_file,
    read_completed_paths,
    write_csv_row,
)
from edge_ids.registry import ModelRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate registered Edge IDS models on many CSV datasets using bounded batch inference."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--datasets", nargs="+", help="Explicit dataset CSV paths")
    source.add_argument("--dataset-dir", help="Directory containing evaluation CSV files")
    parser.add_argument("--pattern", default="*.csv", help="Glob used with --dataset-dir (default: *.csv)")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path")
    parser.add_argument("--model-dir", default=None, help="Model directory (default: from config)")
    parser.add_argument("--output-dir", default="results/multi_dataset_evaluation", help="One output CSV per model")
    parser.add_argument("--models", nargs="+", default=None, help="Model version names; default: all registered models")
    parser.add_argument("--label-column", default=None, help="Label column; if omitted, config and common names are checked")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Rows per inference batch (default: 10000)")
    parser.add_argument("--allow-unlabeled", action="store_true", help="Allow feature-only datasets; classification metrics will be blank")
    parser.add_argument("--resume", action="store_true", help="Skip already successful dataset rows in each model CSV")
    parser.add_argument("--fail-fast", action="store_true", help="Stop processing the current model after the first dataset error")
    parser.add_argument("--_worker-model", help=argparse.SUPPRESS)
    parser.add_argument("--_resolved-datasets-json", help=argparse.SUPPRESS)
    return parser.parse_args()


def _load_datasets(args: argparse.Namespace) -> list[Path]:
    if args._resolved_datasets_json:
        return [Path(p) for p in json.loads(args._resolved_datasets_json)]
    return discover_datasets(args.datasets, args.dataset_dir, args.pattern)


def _model_dir(config: dict, override: str | None) -> Path:
    raw = override or config.get("paths", {}).get("model_dir", "models")
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _config_path(args: argparse.Namespace) -> Path:
    path = Path(args.config).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_worker(args: argparse.Namespace) -> int:
    config_path = _config_path(args)
    config = load_config(config_path)
    model_dir = _model_dir(config, args.model_dir)
    registry = ModelRegistry(model_dir)
    datasets = _load_datasets(args)
    version = args._worker_model
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{version}.csv"

    if args.resume and output_path.exists():
        completed = prepare_resume_file(output_path)
    else:
        completed = set()
    append_mode = args.resume and output_path.exists()
    handle = output_path.open("a" if append_mode else "w", newline="", encoding="utf-8")
    if not append_mode:
        csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()
        handle.flush()

    label_column = args.label_column or config.get("data", {}).get("label_column")
    try:
        model_load_start = time.perf_counter()
        bundle, manifest = registry.load(version, verify_integrity=True)
        model_load_seconds = time.perf_counter() - model_load_start
        print(f"\n[{version}] loaded and integrity-verified in {model_load_seconds:.3f}s", flush=True)
    except Exception as exc:
        print(f"[{version}] model load failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        for dataset in datasets:
            if str(dataset.resolve()) in completed:
                continue
            write_csv_row(handle, error_result(version, dataset, exc, args.batch_size))
        handle.close()
        return 1

    errors = 0
    for index, dataset in enumerate(datasets, start=1):
        dataset_key = str(dataset.resolve())
        if dataset_key in completed:
            print(f"[{version}] [{index}/{len(datasets)}] SKIP already complete: {dataset.name}", flush=True)
            continue
        print(f"[{version}] [{index}/{len(datasets)}] evaluating {dataset.name}", flush=True)
        try:
            result = evaluate_dataset(
                dataset,
                bundle,
                manifest,
                label_column=label_column,
                allow_unlabeled=args.allow_unlabeled,
                batch_size=args.batch_size,
                model_load_seconds=model_load_seconds,
            )
            write_csv_row(handle, result)
            latency_value = result.get("latency_ms_per_sample")
            throughput_value = result.get("throughput_samples_per_second")
            latency_text = f"{latency_value:.6f}" if latency_value is not None else "n/a"
            throughput_text = f"{throughput_value:.1f}" if throughput_value is not None else "n/a"
            print(
                f"[{version}] {dataset.name}: rows={result['rows_processed']}, "
                f"f1={result.get('f1')}, latency_ms/sample={latency_text}, "
                f"throughput={throughput_text}/s, "
                f"peak_rss={result['sampled_peak_rss_mb']:.1f} MiB",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            row = error_result(version, dataset, exc, args.batch_size)
            write_csv_row(handle, row)
            print(f"[{version}] ERROR {dataset.name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                break
    handle.close()
    del bundle, manifest
    return 1 if errors else 0


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        print("ERROR: --batch-size must be positive", file=sys.stderr)
        return 2
    config_path = _config_path(args)
    config = load_config(config_path)
    model_dir = _model_dir(config, args.model_dir)
    datasets = _load_datasets(args)
    all_versions = discover_model_versions(model_dir)
    if not all_versions:
        print(f"ERROR: no model manifests plus artifacts found in {model_dir}", file=sys.stderr)
        return 2
    versions = args.models or all_versions
    unknown = sorted(set(versions).difference(all_versions))
    if unknown:
        print(f"ERROR: unregistered model version(s): {unknown}. Available: {all_versions}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets: {len(datasets)}", flush=True)
    print(f"Models: {len(versions)}", flush=True)
    print(f"Batch size: {args.batch_size:,} rows", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print("Inference is processed in bounded batches; models are not retrained.", flush=True)

    start = time.perf_counter()
    exit_codes: dict[str, int] = {}
    for version in versions:
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--datasets", *[str(p) for p in datasets],
            "--config", str(config_path),
            "--model-dir", str(model_dir),
            "--output-dir", str(output_dir),
            "--batch-size", str(args.batch_size),
            "--_worker-model", version,
            "--_resolved-datasets-json", json.dumps([str(p) for p in datasets]),
        ]
        if args.label_column:
            cmd.extend(["--label-column", args.label_column])
        if args.allow_unlabeled:
            cmd.append("--allow-unlabeled")
        if args.resume:
            cmd.append("--resume")
        if args.fail_fast:
            cmd.append("--fail-fast")
        print(f"\n=== Starting isolated worker: {version} ===", flush=True)
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        exit_codes[version] = completed.returncode
        print(f"=== Finished {version}: exit={completed.returncode} ===", flush=True)

    manifest_path = output_dir / "evaluation_run_manifest.json"
    manifest_data = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "model_dir": str(model_dir),
        "datasets": [str(p) for p in datasets],
        "model_versions": versions,
        "batch_size": args.batch_size,
        "allow_unlabeled": args.allow_unlabeled,
        "resume": args.resume,
        "per_model_csv_files": {v: f"{v}.csv" for v in versions},
        "worker_exit_codes": exit_codes,
        "elapsed_seconds": time.perf_counter() - start,
        "interpretation_note": (
            "Inference is processed in batches; training is not performed. CPU is process CPU time divided by "
            "inference wall time and can exceed 100% on multi-core systems. RSS peak is sampled at batch boundaries, "
            "not an OS-level absolute peak. Classification metrics are computed only for rows with a non-missing label."
        ),
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    # Count row statuses in generated files to provide an honest final summary.
    total_errors = 0
    for version in versions:
        result_path = output_dir / f"{version}.csv"
        if result_path.exists():
            with result_path.open("r", newline="", encoding="utf-8") as handle:
                total_errors += sum(1 for row in csv.DictReader(handle) if row.get("status") == "error")
    print(f"\nEvaluation finished in {manifest_data['elapsed_seconds']:.1f}s", flush=True)
    print(f"Per-model CSV files: {output_dir}", flush=True)
    print(f"Dataset/model rows with errors: {total_errors}", flush=True)
    print(f"Run manifest: {manifest_path}", flush=True)
    return 1 if total_errors or any(exit_codes.values()) else 0


if __name__ == "__main__":
    args = parse_args()
    if args._worker_model:
        raise SystemExit(run_worker(args))
    raise SystemExit(main())
