# Streamlit dashboard — Multi-Dataset Evaluation (v0.7.1)

## What changed

- Model metadata and integrity information is now contained in a Streamlit expander with `expanded=False`, so the section starts collapsed and can be opened as needed.
- The dashboard adds a **Multi-Dataset Model Evaluation** section for ADMIN users.
- An administrator can select an input directory and filename glob, model versions, optional ground-truth label column, batch size, output directory, resume behavior, and fail-fast behavior.
- The dashboard starts `scripts/evaluate_models.py` using the same Python interpreter as Streamlit and redirects its output to `dashboard_evaluation.log`. Evaluation runs in the background rather than blocking the interactive UI.
- The user can refresh status, inspect the end of the run log, review output CSV rows, download a combined CSV, and download individual per-model CSV files. A descriptive unweighted mean-F1 chart is shown for successful labeled evaluations.

## Run the dashboard

Install the project and optional dashboard dependencies from the project root:

```bash
python -m pip install -e ".[all]"
```

Create the evaluation dataset directory and place the labeled CSVs inside it:

```text
data/evaluation_datasets/
├── dataset_01.csv
├── dataset_02.csv
└── ...
```

Create the local ADMIN account if one does not already exist:

```bash
python scripts/create_user.py --username admin --role ADMIN
```

Run Streamlit:

```bash
streamlit run app/dashboard.py
```

Sign in with the ADMIN account, expand **Multi-Dataset Model Evaluation**, check the dataset directory and filename pattern, select models, and start the run. By default, results are written under `results/multi_dataset_evaluation/`. Output paths are constrained to subdirectories of the project root.

## Resource and safety behavior

- Evaluation is batch inference, not model training. Source datasets are processed in the evaluator's configurable row batches.
- The dashboard starts one background evaluator; the evaluator itself runs isolated worker processes per model version.
- Large runs can take a long time. The dashboard remains usable while the worker runs; click **Refresh evaluation status** to update the displayed process status and log excerpt.
- Use `--resume` through the checkbox to skip successfully completed dataset rows from previous runs. Existing per-model CSVs may be rewritten when resume is disabled.
- Dataset paths are inputs, while output paths must remain within the project directory. The section that launches evaluation is ADMIN-only because it can consume substantial compute and write result files.
- The UI reads the per-model CSVs created by the CLI and does not retain individual million-row predictions in Streamlit session memory.

## Output files

For every selected model version, the output directory contains `<model_version>.csv` with one row per dataset, plus `evaluation_run_manifest.json` and `dashboard_evaluation.log`. The dashboard can show a combined result table and offers a combined CSV download without replacing the canonical per-model CSVs.

## Validation notes

The dashboard support helpers have unit tests for command construction, output path validation, combined CSV loading, and log tailing. The full PyTest suite and a CLI smoke run using the exact command builder used by the dashboard passed. The smoke run used the bundled synthetic demo data, not the research datasets. The Streamlit runtime was not installed in the validation container, so a live browser-driven dashboard test remains to be run in the user's local environment after installing `.[all]`.
