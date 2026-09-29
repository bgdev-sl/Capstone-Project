# Train a selected model

from __future__ import annotations
import argparse
import json
from pathlib import Path 
import sys 

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from edge_ids.config import load_config, ensure_directories
from edge_ids.data import load_dataset
from edge_ids.registry import ModelRegistry
from edge_ids.models import SUPPORTED_MODELS
from edge_ids.training import train_one

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", required=True, help="CSV dataset path")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--version", required=True, help="Version string, e.g. dt_v1_reduced")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-feature-reduction", action="store_true")
    parser.add_argument("--feature-k", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config, ROOT)

    data_cfg = config.get("data", {})
    max_rows = args.max_rows if args.max_rows is not None else data_cfg.get("max_rows")
    X, y, label_column = load_dataset(
        args.dataset,
        label_column=data_cfg.get("label_column"),
        drop_columns=data_cfg.get("drop_columns", []),
        max_rows=max_rows,
        random_state=config.get("project", {}).get("random_state", 42),
    )

    reduction_cfg = config.get("feature_reduction", {})
    reduction_enabled = reduction_cfg.get("enabled", True) and not args.no_feature_reduction
    feature_k = args.feature_k if args.feature_k is not None else reduction_cfg.get("k", 64)

    registry = ModelRegistry(ROOT / config.get("paths", {}).get("model_dir", "models"))
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