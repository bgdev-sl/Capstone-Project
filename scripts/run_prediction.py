# Run inference against CSV 

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.config import load_config  
from edge_ids.pipeline import EdgeIDSApplication

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--input", required=True, help="Feature CSV")
    parser.add_argument("--output", default="results/predictions.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    app = EdgeIDSApplication(config, ROOT)
    predictions, resources = app.predict_csv(args.model_version, args.input)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Predictions written to {output}")
    print("Resource metrics:")
    for key, value in resources.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()