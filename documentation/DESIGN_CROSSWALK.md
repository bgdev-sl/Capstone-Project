# Design Crosswalk: Report / Diagram → Prototype

This crosswalk shows how the software prototype maps to the architecture and requirements in the technical report.

| Report / diagram element | Prototype implementation |
|---|---|
| 1. Traffic Collection Module | `src/edge_ids/traffic.py` — CSV flow input plus optional Scapy packet metadata capture |
| 2. Preprocessing & Feature Extraction | `src/edge_ids/preprocessing.py` — imputation, scaling, categorical encoding, configurable feature reduction |
| 3. ML Interface Module | `src/edge_ids/models.py`, `training.py`, `inference.py` — Decision Tree, Random Forest, XGBoost, MLP |
| 4. Alert & Logging Module | `src/edge_ids/alerts.py` — confidence threshold, JSONL alerts, rotating log files |
| Authentication & Authorization | `src/edge_ids/auth.py` — local authentication and ADMIN / ANALYST roles |
| Model / Configuration Integrity | `src/edge_ids/registry.py`, `integrity.py` — model versioning, metadata, SHA-256 verification |
| Resource Monitoring | `src/edge_ids/monitoring.py` — elapsed latency, CPU percentage, RSS before/after, RSS delta |
| Public datasets | Configurable CSV loader; CICIoT2023 is expected to be the primary research dataset |
| Feature reduction experiment | `SelectKBest(f_classif)` with configurable `k`; benchmark runs full and reduced feature sets |
| Secure dashboard | `app/dashboard.py` — authenticated local Streamlit interface with prediction CSV download and structured alert view |
| REST API | `src/edge_ids/api.py` — authenticated model listing, batch prediction, alert retrieval, and ADMIN-only allowlisted configuration updates |
| UI/backend linkage | Streamlit and FastAPI both call `EdgeIDSApplication`; inference and alert logic remain in shared services |
| Research outputs | `results/integrated_benchmark.csv` contains model, feature configuration, classification metrics, warmed end-to-end software latency, throughput, CPU, RSS, confidence, feature count, and model size |
| Deployment packaging | `Dockerfile`, `compose.yaml`, and `.env.example` support local API/dashboard integration; official measurements should use a consistent target environment |
| Isolated test environment | Software supports local/VM execution; production deployment is deliberately out of scope |

## Requirement mapping

### Functional requirements

- **FR1:** traffic / flow ingestion is represented by CSV flow loading and optional Scapy metadata collection.
- **FR2:** validation, imputation, encoding, scaling, and feature reduction are implemented in one shared transformation object.
- **FR3:** four selectable supervised models are implemented.
- **FR4:** malicious classifications can produce alert records and logs.
- **FR5:** dashboard access is protected by local authentication and role checks.
- **FR6:** model artifacts are versioned and integrity checked before loading.
- **FR7:** model performance and resource metrics are recorded during training/inference and exported to CSV.

### Non-functional requirements

- **Performance:** batch inference latency is measured per sample.
- **Usability:** the dashboard exposes model metadata, upload, inference, and alert visibility.
- **Reliability:** input validation and integrity verification reject empty/missing or tampered artifacts.
- **Scalability:** model creation and preprocessing are configuration-driven so additional estimators/datasets can be added without redesigning the whole application.
- **Security:** local RBAC, password hashing, data minimization in the packet collector, integrity checking, and log rotation are implemented as prototype controls.
