#!/usr/bin/env python3

# Plot the observed F1/latency trade-off from a benchmark CSV

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/integrated_benchmark.csv')
    parser.add_argument('--output', default='results/performance_tradeoff.png')
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {'model', 'feature_reduction', 'f1', 'latency_ms_per_sample'}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f'Benchmark CSV is missing columns: {sorted(missing)}')

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in frame.iterrows():
        label = f"{row['model']} {'reduced' if bool(row['feature_reduction']) else 'full'}"
        ax.scatter(row['latency_ms_per_sample'], row['f1'], s=70)
        # Separate labels for near-overlapping points while keeping the plot readable.
        offsets = {
            ('decision_tree', False): (5, 9),
            ('decision_tree', True): (5, 8),
            ('mlp', False): (5, 9),
            ('mlp', True): (5, -16),
            ('random_forest', False): (5, 8),
            ('random_forest', True): (5, 8),
            ('xgboost', False): (5, 8),
            ('xgboost', True): (5, 8),
        }
        key = (str(row['model']), bool(row['feature_reduction']))
        ax.annotate(label, (row['latency_ms_per_sample'], row['f1']),
                    xytext=offsets.get(key, (5, 4)), textcoords='offset points', fontsize=8)

    ax.set_xscale('log')
    ax.set_xlabel('Warmed batch inference latency (ms per sample, log scale)')
    ax.set_ylabel('F1 score (malicious class)')
    ax.set_title('Edge IDS prototype: detection quality vs inference latency')
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f'Saved visualization to {output}')


if __name__ == '__main__':
    main()
