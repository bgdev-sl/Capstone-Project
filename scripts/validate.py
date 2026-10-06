# validate the structure and value ranges

from __future__ import annotations
import argparse
from pathlib import Path 
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from edge_ids.validation import validate_benchmark_csv

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', help='Path to a benchmark result CSV')
    args = parser.parse_args()

    frame = validate_benchmark_csv(ROOT / args.csv)
    print(f'VALID: {len(frame)} benchmark rows passed schema and range validation.')

if __name__ == '__main__':
    main()
    