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
├── app/
│   └── dashboard.py
├── configs/
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
│   ├── run_prediction.py
│   ├── train_models.py
│   └── validate.py
├── src/edge_ids/
│   ├── alerts.py
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
│   └── test_core.py
├── pyproject.toml
└── requirements.txt
```

## 3. Testing Prototype

### (a) Install virtual environment

```bash
python -m venv .venv 

.venv\\Scripts\\Acitivate.ps1

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
- process CPU percentage
- resident memory after inference
- resident memory delta
- feature count
- mean confidence

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

### (c) Other considerations

This current software prototype does not implement energy measurement. To achieve this, a third-party system monitoring tool will be implemented during testing within the edge environment. While a specific tool has not yet been identified, **PowerJoular** is a possible option for energy consumption monitoring.

## 7. Next steps

1. Prepare CICIoT2023 and identify the final label and excluded metadata columns
2. Freeze a preprocessing config before model comparison.
3. Run all models on the same train/test split and the same feature-reduction method.
4. Deploy the registered model artifacts into the Fedora IoT 44 VM.
5. Implement PowerJoular (or other power monitoring tool) and designate monitoring for the IoT VM.
6. Run the same inference workload repeatedly and record median/p95 latency, CPU utilization, RAM usage, and energy consumption.
7. Export results for statistical analysis and visualization.

## V 0.2 Notes:

- Adds extensive unit testing in `tests/test_core.py`
- Adds validation for benchmark results in `scripts/validate.py` and `src/validation.py` 
- Adds controls to limit configuration modification via the dashboard with `src/config_manager.py`
- Adds CI testing with `workflows/ci.yml` that runs unit tests before committing changes.