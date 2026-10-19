# ML-Based IDS Testing for Edge Devices - MSIT Capstone

This design is a prototype architecture for a system that attempts to achieve the following:

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
- optional feature reduction
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
│   └── test_core.py
├── compose.yaml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## 3. Testing Prototype

### (a) Install virtual environment

```bash
python -m venv .venv 

source .venv/bin/activate

pip install -r requirements.txt
```

>---
> **Note**
> - Packet capture requires **Scapy**, which needs appropriate *OS permissions*. 
> - The *dashboard app* requires **Streamlit**.
>---


### (b) Smoke test with small generated dataset

For the prototype, a small dataset is generate. For the full experiment, **CICIoT2023** will be used for training models.

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

## 4. Future plans

1. Download the prepared CICIoT2023 CSV in `data/` 
2. Make label column configurable in `config/config.yaml` if necessary
3. Train Models
    - A single model should be trained as:

        ```bash
        python scripts/train_models.py \
          --dataset data/CICIoT2023.csv \
          --model random_forest \
          --version random_forest_v1_reduced \
          --feature-k 64
        ```
    - Train all models and run a full-vs-reduced comparison
        ```bash
        python scripts/benchmark.py \
          --dataset data/CICIoT2023.csv \
          --models decision_tree random_forest xgboost mlp \
          --reduced-k 64 \
          --output results/ciciot2023_benchmark.csv
        ```
4. After models are registered and verified, they can be loaded and tested using `run_prediction.py`. for example:
    ```bash
    python scripts/run_prediction.py \
      --model-version random_forest_v1_reduced \
      --input data/demo_flow.csv \
      --output results/predictions.csv
    ```

>---
> **Note**
>
> To maintain a controlled experiment, the same *dataset split*, *preprocessing config*, and *environment* should be maintained across all models.
>
>---

## 5. Implementing and accessing the dashboard

1. Create a local account:
    ```bash
    python scripts/create_user.py --username admin --role ADMIN
    # OR
    python scripts/create_user.py --username analyst --role ANALYST
    ```
    >---
    > **Note**
    >
    > This script will prompt the user to create a password, which will be stored in `configs/users.json` using *PBKDF2-HMAC-SHA256 with a random salt* for user authentication
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
    > The dashboard is intentionally local to allow for secure experimentation boundaries and minimize security risks
    >
    >---

## 6. Notes on current prototype

### (a) Packet collection

**Scapy** can be used for to collect packet traffic data. `scripts/capture.py` can be used for this. It only collects and stores packet metadata, omitting payload bytes and minimizing data stored.

This version should only be used in an isolated environment for security purposes. It can be used as follows: 

```bash
python scripts/capture.py --count 100 --output data/packet_metadata.jsonl
```

> ---
> **Important**
>
> A dedicated flow-feature adapter is still needed before live packet data can be classified by a trained model.
>
> ---

### (b) Security factors

While the authentication module demonstrates the use of password authentication and RBAC architecture, it does not replace a production-ready system that integrates enterprise identity verification.

The model registry utilizes SHA-256 verification to detect artifact tampering.

Logs are written with rotation to account for limited local storage capabilities.

## 7. Automated GitHub Testing

The repository includes `.github/workflows/ci.yml`, which runs the PyTest suite on pushes and pull requests targeting `main` and `develop`. This provides a basic quality gate before milestone changes are merged.

## 8. Integrated REST API and deployment

A FastAPI adapter now exposes the same `EdgeIDSApplication` service used by the Streamlit UI. It provides `GET /health`, `GET /api/v1/models`, `POST /api/v1/predict`, `GET /api/v1/alerts`, and an ADMIN-only `PATCH /api/v1/admin/config`. Non-health routes use the local PBKDF2-hashed user store and HTTP Basic authentication. Keep the API local/isolated; HTTP Basic must be protected by TLS before any remote deployment.

Example request: `examples/api_predict_request.json`. Run a local API with `uvicorn edge_ids.api:app --app-dir src --host 127.0.0.1 --port 8000`; interactive docs are at `/docs`. The dashboard calls the same service layer directly and supports prediction download.

The repository also contains `Dockerfile`, `compose.yaml`, `.env.example`, and `.dockerignore` for an integration deployment with API and dashboard containers. Copy `.env.example` to `.env` and follow `SYSTEM_INTEGRATION_EVALUATION.md` for the setup, measured synthetic-data baseline, qualitative assessment, and deployment guidance. `results/benchmark_environment.json` records the synthetic dataset and edge-config hashes, exact runtime versions, and timing policy.

## 9. V 0.5.0 EXAMPLE IMPLIMENTATION

Create a virtual environemnt and install the dependencies:

```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

Benchmark all algorithms with and without feature reduction:

```bash
python scripts/benchmark.py \
  --config configs/config_edge.yaml \
  --dataset data/CICIoT2023.csv \
  --max-rows 500000 \
  --sampling-chunk-rows 5000 \
  --reduced-k 24 \
  --output results/benchmark_sample-500_000.csv
```

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

## 10. Core Testing and Validation

To run the testing suite:

```bash
python -m pytest -q
```

To validate benchmark output:

```bash
python scripts/validate.py results/benchmark_sample-500_000.csv
```

Testing covers dataset handling, label normalization, train/test splitting, preprocessing and feature reduction, model factories, evaluation metrics, confidence calculation, model integrity, model registry behavior, authentication/RBAC, alert generation, controlled configuration changes, and benchmark-output validation.


## 11. Futher Plans

1. CICIoT2023.csv will be split into two seperate datasets. 
2. The first half of the dataset will be used to model training and benchmarking. 
3. The latter half of the dataset will be used for evaluating artifact performance within the IoT environment.
4. Model performance within the IoT environment will be evaluated to assess performance and resource consumption.