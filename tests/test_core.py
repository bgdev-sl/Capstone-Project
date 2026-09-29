from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.data import normalize_binary_labels
from edge_ids.integrity import sha256_file, verify_sha256
from edge_ids.preprocessing import FeaturePipeline

def test_normalize_binary_labels():
    labels = pd.Series(["Benign", "DDoS", "Scanning"])
    assert normalize_binary_labels(labels).tolist() == [0, 1, 1]

def test_feature_pipeline_reduces_features():
    X = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [5, 4, 3, 2, 1, 0],
            "protocol": ["TCP", "TCP", "UDP", "UDP", "TCP", "UDP"],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])
    pipeline = FeaturePipeline(feature_reduction_enabled=True, k=2)
    transformed = pipeline.fit_transform(X, y)
    assert transformed.shape[1] == 2

def test_sha256_verification(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    digest = sha256_file(path)
    assert verify_sha256(path, digest)
