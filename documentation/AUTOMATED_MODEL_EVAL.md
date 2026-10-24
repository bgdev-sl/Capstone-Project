# Automated multi-dataset prediction and evaluation

The `scripts/evaluate_models.py` utility evaluates every registered model version (normally four algorithms × full/reduced feature sets = eight model bundles) across many labeled CSV files. It produces **one CSV per model version**, with one row per dataset. It does not train or update models.

## Why inference is chunked

Each CSV is read in batches (10,000 rows by default). The existing fitted preprocessor and model are applied to each batch, and confusion-matrix counts are accumulated across batches. Predictions and input rows are not stored after the current batch. This is **bounded batch inference**, not incremental/streaming training; the project's training path remains batch-based.

## Organize the inputs

Place the evaluation dataset CSVs in a dedicated folder, for example:

```text
data/evaluation_datasets/
  dataset_01.csv
  dataset_02.csv
  ...
  dataset_20.csv
```

Each labeled dataset must contain the raw feature columns expected by the saved model bundles and a target column. By default, the script reads `data.label_column` from `configs/config.yaml`, or tries common names (`label`, `Label`, `attack`, `Attack`, `target`, `Target`, `class`, `Class`). Use `--label-column Label` if necessary. Binary labels are normalized using the project's current convention: benign → 0 and malicious/non-benign → 1. Ensure each target column represents the same benign/malicious definition.

## Run all model versions over the 20 datasets

From the repository root with the virtual environment activated:

```bash
python -m pip install -e ".[all]"
python scripts/evaluate_models_on_datasets.py \
  --dataset-dir data/evaluation_datasets \
  --pattern "*.csv" \
  --config configs/config.yaml \
  --output-dir results/multi_dataset_evaluation \
  --batch-size 10000
```

You can also pass explicit paths:

```bash
python scripts/evaluate_models_on_datasets.py \
  --datasets data/evaluation_datasets/dataset_01.csv data/evaluation_datasets/dataset_02.csv \
  --output-dir results/multi_dataset_evaluation \
  --batch-size 5000
```

To test only two model variants before starting the full run:

```bash
python scripts/evaluate_models_on_datasets.py \
  --dataset-dir data/evaluation_datasets \
  --models xgboost_v1_full xgboost_v1_reduced8 \
  --output-dir results/multi_dataset_evaluation_smoke \
  --batch-size 5000
```

To resume after interruption, reuse the same output directory with `--resume`. Successful dataset/model rows are skipped; rows previously marked `error` are retried. `--allow-unlabeled` permits feature-only files, but classification metrics are blank for those datasets. Because accuracy, precision, recall, F1, and confusion matrix require ground-truth labels, keep it off for the research comparison whenever possible.

## Outputs

The output directory contains:

- `decision_tree_v1_full.csv`
- `decision_tree_v1_reduced8.csv`
- `random_forest_v1_full.csv`
- `random_forest_v1_reduced8.csv`
- `xgboost_v1_full.csv`
- `xgboost_v1_reduced8.csv`
- `mlp_v1_full.csv`
- `mlp_v1_reduced8.csv`
- `evaluation_run_manifest.json`

The exact list is driven by the registered model artifacts found in the configured model directory. Each model CSV has one result row per dataset, including failed dataset/model pairs so an entire run can complete and report errors. Rows are flushed after every dataset, and `--resume` can continue successful work.

## Metrics per dataset/model row

- Detection: accuracy; malicious-class precision/recall/F1; separate benign and malicious precision/recall/F1; specificity; balanced accuracy; false-positive rate; false-negative rate; macro precision/recall/F1; weighted F1; Matthews correlation coefficient; TN/FP/FN/TP; and benign/malicious support.
- Timing: preprocessing seconds, model prediction seconds, inference seconds, mean latency per sample, p50/p95 batch latency, inference throughput, end-to-end throughput, data-read time, and total elapsed time.
- Resources: process CPU seconds during preprocessing/prediction, CPU utilization derived from CPU-time/wall-time, RSS before/after evaluation, RSS delta, and sampled peak RSS.
- Traceability: model version, feature-reduction setting, dataset name/path, label column, row counts, batch size, batch count, status, and error details.

`cpu_utilization_percent` can exceed 100% when native libraries use multiple CPU cores. `sampled_peak_rss_mb` is sampled between batches and at transformation/prediction boundaries, so it is an observed sample rather than an OS-level maximum. RSS includes the Python runtime, loaded model, preprocessing libraries, and current batch—not just the estimator object. CSV read time is reported separately; inference throughput measures preprocessing plus model prediction, whereas end-to-end throughput includes file reading and metric aggregation.

The script launches an isolated subprocess per model version to make model memory measurements more comparable. This means each CSV is parsed once per model (eight passes over each dataset when all eight models run). Benchmarking 20 million rows across eight models is a substantial workload; start with two models and one dataset, then scale up. Use `--batch-size 2000` or `5000` if RAM is constrained. Larger batch sizes can improve throughput but increase transient memory use.

## Research-validity notes

1. These are **evaluation datasets**, not training sets. The models remain frozen. Each dataset must use compatible feature meanings, units, labels, and preprocessing assumptions.
2. The metrics are valid only if the labels are available and use the same benign/malicious definition. An absent label yields inference resource metrics only.
3. Full-feature and reduced-feature models may have been fitted with different feature selectors; the script always uses the preprocessing object packaged with each model.
4. Model artifacts must match their JSON SHA-256 manifests. The existing registry verifies integrity on load.
5. Resource results depend on hardware, Python/library versions, batch size, competing load, and CSV storage performance. Record these alongside the CSVs when reporting results.
6. For statistical conclusions, run repeat evaluations where practical and report variability; this script reports one measured pass per dataset/model pair.

## Streamlit dashboard workflow

Version 0.7.1 adds an ADMIN-only UI for this evaluator. The UI launches `scripts/evaluate_models.py` in a background subprocess using the same Python interpreter and project configuration, so longer runs do not block the Streamlit request. The UI lets an administrator select the dataset directory and glob, models, optional label column, batch size, output directory, resume behavior, and fail-fast behavior. Results are aggregated from the per-model CSV files for viewing and download; the canonical individual CSVs and run manifest remain on disk. Use the Refresh evaluation status button to poll a running job and display the tail of its log.
