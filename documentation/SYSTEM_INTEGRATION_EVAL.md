# System Integration and Evaluation — Current Baseline

**Prototype:** Edge IDS v0.3.0, final-stage hardening snapshot  
**Date:** 15 October 2026

## Integrated application

The application comprises a Streamlit dashboard, FastAPI REST adapter, shared `EdgeIDSApplication` service, preprocessing pipeline, four-model ML interface, versioned model registry, SHA-256 verification, resource monitor, local authentication/RBAC, alert JSONL logger, YAML configuration, and CSV benchmark workflow.

The REST interface exposes:

| Method / path | Purpose | Access |
|---|---|---|
| `GET /health` | Liveness | Public, no sensitive details |
| `GET /api/v1/models` | Registered model versions | Authenticated |
| `POST /api/v1/predict` | Batch inference and resource metrics | Authenticated |
| `GET /api/v1/alerts` | Recent alerts | Authenticated |
| `PATCH /api/v1/admin/config` | Modify whitelisted configuration | ADMIN only |

For REST predictions, the response contains classification metadata only; the service does not echo caller-supplied flow features. The local dashboard may show its uploaded feature table and prediction table to the logged-in user.

## Validation results

- PyTest: **28 passed**.
- Live API acceptance: **11/11 passed**.
- Source compilation: passed.
- Benchmark validation: 8/8 rows passed output schema/range checks.
- Synthetic smoke-test highest malicious-class F1: XGBoost/full features, **0.9833**.

## Example local setup

```bash
python -m venv .venv
# Linux/Fedora
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[all]"
python -m pytest -q
```

To run services locally, create an ADMIN and ANALYST user interactively with `python scripts/create_user.py`, then start the API and dashboard in separate terminals:

```bash
uvicorn edge_ids.api:app --app-dir src --host 127.0.0.1 --port 8000
streamlit run app/dashboard.py --server.address 127.0.0.1 --server.port 8501
```

The API uses local HTTP Basic authentication, so it must remain loopback/isolated until protected by TLS and a stronger identity/session layer. Use Docker Compose only after validating the build on a machine with Docker or Podman; no image build was possible in this test environment.

## Performance boundary

The included benchmark uses the synthetic demo dataset and an 80/20 split with 19 full features or 8 selected features. It records detection metrics, model size, latency, throughput, CPU, and process RSS. It is a software-validation smoke test, not a basis for selecting the final research model. The controlled CICIoT2023/Fedora IoT 44 experiment, repeated trials, statistical analysis, and energy measurement where feasible remain outstanding.
