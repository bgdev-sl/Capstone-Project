# ML-Based IDS Testing for Edge Devices - MSIT Capstone

This package is designed to build ML-based IDS models and evaluate their performance and resource consumption on IoT/Edge devices. The package offers functionality for the following: 

- traffic collection / flow input
- preprocessing and feature extraction
- implement a selection of ML algorithms for intrusion detection on IoT/edge devices
    - Decision Tree
    - Random Forest
    - XGBoost
    - MLP
- integrate an alert system and JSONL logging
- local authentication with RBAC roles (ADMIN and ANALYST)
- versioned model storage and SHA-256 integrity verification
- CPU, RAM, and latency measurements
- optional feature reduction (as an experiemental variable)
- Streamlit dashboard
- optional Scapy packet metadata collection

## 1. Prototype Scope

The prototype treats the primary experiment as binary intrusion detection: benign (0) vs. potentially malicious (1). This matches the report's system requirement that the models classify traffic as benign or malicious while evaluating accuracy, precision, recall, F1-score, confusion matrices, CPU, RAM, latency, model size, and energy where practical.

Feature reduction is implemented as `SelectKBest(f_classif)` so the benchmark can compare full vs. reduced feature sets. The reduction method is configurable and is not presented as the only valid research methodology.

## 2. File Organization

```text
edge_ids_prototype/
├── .github/workflow/
│   └── ci.yml
├── app/
│   └── dashboard.py
├── configs/
│   ├── config_edge.yaml
│   └── config.yaml
├── data/
├── logs/
├── models/
├── results/
├── scripts/
│   ├── benchmark.py
│   ├── capture.py
│   ├── create_user.py
│   ├── evaluate_models.py
│   ├── make_dataset.py
│   ├── plot_results.py
│   ├── run_prediction.py
│   ├── run_system_acceptance.py
│   ├── train_models.py
│   └── validate.py
├── src/edge_ids/
│   ├── alerts.py
│   ├── api.py
│   ├── auth.py
│   ├── config_manager.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── experiment_evaluation.py
│   ├── inference.py
│   ├── integrity.py
│   ├── models.py
│   ├── monitoring.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── registry.py
│   ├── traffic.py
│   ├── training.py
│   └── validation.py
├── tests/
│   ├── test_api.py
│   ├── test_core.py
│   └── test_data_sampling.py
├── compose.yaml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## 3. Testing Prototype

### (a) Install virtual environment

```bash
python -m venv .venv 

# Linux/macOS
source .venv/bin/activate

# Windows Powershell
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

>---
> **Note**
> - Packet capture requires **Scapy**, which needs appropriate *OS permissions*. 
> - The *dashboard app* requires **Streamlit**.
>---


### (b) Smoke test with small generated dataset

Any relevant dataset that fits the labeling configuration can be used to train models, but the package also includes a `make_dataset.py` script that allows the user to generate a small demo dataset to observe the functionalities. To generate a demo dataset: 

```bash
python scripts/make_dataset.py \
  --rows 2500 \
  --output data/demo_flows.csv
```

To benchmark all four algorithms with and without feature reduction:

```bash
python scripts/benchmark.py \
  --config configs/config_edge.yaml \
  --dataset data/demo_flows.csv \
  --max-rows 2500 \
  --reduced-k 10 \
  --output results/demo_benchmark.csv
```

The resulting CSV will contain the following for each model/feature config:

- accuracy
- precision
- recall
- F1
- latency per sample
- throughput (samples per second)
- process CPU percentage
- resident memory after inference
- resident memory delta
- feature count
- mean confidence
- train rows
- test rows
- inference repeats
- model size (bytes)

## 4. Implementing and accessing the dashboard

1. Create a local account. For example:
    ```bash
    python scripts/create_user.py --username admin --role ADMIN
    # OR
    python scripts/create_user.py --username analyst --role ANALYST
    ```
    >---
    > **Note**
    >
    > Passwords are stored locally using *PBKDF2-HMAC-SHA256 with a random salt*
    >
    >---

2. Start the **Streamlit** dashboard:
    ```bash
    streamlit run app/dashboard.py
    # OR:
    python -m streamlit run app/dashboard.py
    ```
    >---
    > **Note**
    > 
    > The dashboard is intentionally local to allow for secure experimentation boundaries and to minimize security risks
    >
    >---

## 5. Security Factors

While the authentication module demonstrates the use of password authentication and RBAC architecture, it does not replace a production-ready system that integrates enterprise identity verification.

The model registry utilizes SHA-256 verification to detect artifact tampering.

Logs are written with rotation to account for limited local storage capabilities.



## 6. Working with Large Datasets

V 0.5.0 introduces dataset sampling for working with large datasets. The following example demonstrates how this feature can be used:

### Running a bounded benchmark

```bash
python scripts/benchmark.py \
  --config configs/config_edge.yaml \
  --dataset data/CICIoT2023.csv \
  --max-rows 500000 \
  --sampling-chunk-rows 5000 \
  --reduced-k 24 \
  --output results/benchmark_sample-500_000.csv
```
This produces the eight model configurations (four full-feature models and four reduced-feature models). Using the benchmark script ensures all models are trained on the same data sampling, which can help support evaluation of model performance and resource consumption metrics.

The resulting CSV contains the following metrics for each model/feature config:

- accuracy
- precision
- recall
- F1
- inference latency per sample
- process CPU percentage
- resident memory after inference
- resident memory delta
- feature count
- mean confidence

### Memory and Sampling Guidance:

- Begin with `--max-rows 10000` or `25000` to validate the real CSV schema and estimate RAM use.
- The sampler keeps the requested sample and one CSV chunk in memory. Reduce `data.sampling_chunk_rows` in YAML if one source chunk is too large for your machine.
- Training then materializes the sampled features and transformed matrices in memory. `--max-rows` must be chosen based on RAM available, feature cardinality, and the model being trained.
- Tree and MLP model fitting remain batch operations; lowering `--max-rows` is the primary memory control in this release.
- Avoid `--allow-large-batch` unless you have verified enough memory for the full dataset and transformed feature matrices.

## 7. Core Testing and Validation

To run the PyTest suite:

```bash
python -m pytest -q
```

To validate benchmark output:

```bash
python scripts/validate.py results/benchmark_sample-500_000.csv
```

To evaluate API functionality:

```bash
# ==========================
# In the first terminal:
# ==========================
# Create test users:
python scripts/create_user.py \
  --username acceptance_analyst \
  --role ANALYST

  python scripts/create_user.py \
  --username acceptance_admin \
  --role ADMIN

# Launch the API service
uvicorn edge_ids.api:app \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8000

# ==========================
# In a second terminal:
# ==========================
# Define API variables for test-case
export EDGE_IDS_ACCEPTANCE_URL="http://127.0.0.1:8000"

export EDGE_IDS_ACCEPTANCE_ANALYST_USER="acceptance_analyst"
read -rsp "Analyst password: " EDGE_IDS_ACCEPTANCE_ANALYST_PASSWORD
echo
export EDGE_IDS_ACCEPTANCE_ANALYST_PASSWORD

export EDGE_IDS_ACCEPTANCE_ADMIN_USER="acceptance_admin"
read -rsp "Admin password: " EDGE_IDS_ACCEPTANCE_ADMIN_PASSWORD
echo
export EDGE_IDS_ACCEPTANCE_ADMIN_PASSWORD

export EDGE_IDS_ACCEPTANCE_THRESHOLD="0.81"

# Run acceptance test:
python scripts/run_system_acceptance.py
```

Testing covers dataset handling, label normalization, train/test splitting, preprocessing and feature reduction, model factories, evaluation metrics, confidence calculation, model integrity, model registry behavior, authentication/RBAC, alert generation, controlled configuration changes, and benchmark-output validation.

## 8. Evaluating Model Performance After Training

### Running a Single Inference

This prints all predictions to a CSV file

```bash
python scripts/run_prediction.py \
  --model-version random_forest_v1_reduced \
  --input data/demo_flows.csv \
  --output results/predictions.csv
```

### Multi-Model Prediction and Evaluation

`scripts/evaluate_models.py` can be used to evaluate all models against new dataset(s) and evaluate the performance and resource metrics for each dataset run. The following is an example use case for running this script:

```bash
python scripts/evaluate_models.py \
  --dataset-dir data/evaluation_datasets \
  --pattern "*.csv" \
  --output-dir results/multi_dataset_evaluation \
  --batch-size 10000
```

This automates the evaluation process and prints results to CSV files, one for each model configuration. 
