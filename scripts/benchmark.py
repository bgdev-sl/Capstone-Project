# Benchmark the models with and without feature reduction

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from edge_ids.config import load_config, ensure_directories
from edge_ids.data import load_dataset
from edge_ids.models import SUPPORTED_MODELS
from edge_ids.registry import ModelRegistry
from edge_ids.training import train_one

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--models', nargs='+', choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS))
    parser.add_argument('--reduced-k', type=int, default=None)
    parser.add_argument('--output', default='results/benchmark.csv')
    parser.add_argument('--skip-full', action='store_true')
    parser.add_argument('--skip-reduced', action='store-true')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config, ROOT)

    data_cfg = config.get('data', {})
    X, y, label_column = load_dataset(
        args.dataset,
        label_column=data_cfg.get('label_column'),
        drop_columns=data_cfg.get('drop_columns', []),
        max_rows=args.max_rows if args.max_rows is not None else data_cfg.get('max_rows'),
        random_state=config.get('project', {}).get('random_state', 42)
    )

    reduction_cfg = config.get('feature_reduction', {})
    reduced_k = args.reduced_k if args.reduced_k is not None else reduction_cfg.get('k', 64)

    registry = ModelRegistry(ROOT / config.get('paths', {}).get('model_dir', 'models'))
    rows = []

    experiments = []
    if not args.skip_full:
        experiments.append((False, None))
    if not args.skip_reduced:
        experiments.append((True, reduced_k))

    for model_name in args.models:
        for reduced, k in experiments:
            suffix = f'reduced{k}' if reduced else 'full'
            version = f'{model_name}_v1_{suffix}'
            result = train_one(
                model_name,
                X,
                y,
                config,
                version,
                feature_reduction=reduced,
                feature_k=k or reduced_k,
                registry=registry
            )
            row = {
                'model': model_name,
                'feature_reduction': reduced,
                'feature_k': k,
                'model_version': version,
                'accuracy': result['metrics']['accuracy'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'f1': result['metrics']['f1'],
                'latency_ms_per_sample': result['inference_resource_metrics']['latency_ms_per_sample'],
                'cpu_percent': result['inference_resource_metrics']['cpu_percent'],
                'rss_after_mb': result['inference_resource_metrics']['rss_after_mb'],
                'rss_delta_mb': result['inference_resource_metrics']['rss_delta_mb'],
                'mean_confidence': result['confidence_mean'],
                'output_feature_count': result['output_feature_count'],
                'train_rows': result['train_rows'],
                'test_rows': result['test_rows'],
                'model_size_bytes': result['manifest']['artifact_size_bytes'],
                'label_column': label_column
            }
            rows.append(row)
            print(f'Completed {version}')

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f'Benchmark results written to {output}')

if __name__ == '__main__':
    main()

