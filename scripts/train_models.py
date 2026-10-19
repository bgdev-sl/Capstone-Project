#!/usr/bin/env python3
"""Train a selected model against the configured dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.config import load_config, ensure_directories  # noqa: E402
from edge_ids.data import load_dataset  # noqa: E402
from edge_ids.registry import ModelRegistry  # noqa: E402
from edge_ids.models import SUPPORTED_MODELS  # noqa: E402
from edge_ids.training import train_one  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", required=True, help="CSV dataset path")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--version", required=True, help="Version string, e.g. dt_v1_reduced")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Train batch estimators on a uniform reservoir sample of at most N rows; avoids loading the full CSV into memory.",
    )
    parser.add_argument(
        "--sampling-chunk-rows",
        type=int,
        default=None,
        help="Rows read per CSV chunk while selecting --max-rows (default: data.sampling_chunk_rows, 10000).",
    )
    parser.add_argument("--no-feature-reduction", action="store_true")
    parser.add_argument("--feature-k", type=int, default=None)
    parser.add_argument(
        "--allow-large-batch",
        action="store_true",
        help="Override the safety guard for batch training from a CSV larger than 2 GiB (may exhaust RAM).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config, ROOT)

    data_cfg = config.get("data", {})
    reduction_cfg = config.get("feature_reduction", {})
    reduction_enabled = reduction_cfg.get("enabled", True) and not args.no_feature_reduction
    feature_k = args.feature_k if args.feature_k is not None else reduction_cfg.get("k", 64)
    registry = ModelRegistry(ROOT / config.get("paths", {}).get("model_dir", "models"))

    max_rows = args.max_rows if args.max_rows is not None else data_cfg.get("max_rows")
    dataset_size = Path(args.dataset).stat().st_size if Path(args.dataset).exists() else 0
    if (
        max_rows is None
        and dataset_size > 2 * 1024**3
        and not args.allow_large_batch
    ):
        raise SystemExit(
            f"Dataset is {dataset_size / (1024**3):.2f} GiB. Batch training reads the selected data "
            "and transformed feature matrix into memory and may exhaust RAM. Use --max-rows N "
            "to select a bounded reservoir sample for batch training. If you have verified sufficient "
            "RAM, --allow-large-batch overrides this guard."
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

    result = train_one(
        model_name=args.model,
        X=X,
        y=y,
        config=config,
        model_version=args.version,
        feature_reduction=reduction_enabled,
        feature_k=feature_k,
        registry=registry,
    )
    result["dataset"] = str(args.dataset)
    result["label_column"] = label_column

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
