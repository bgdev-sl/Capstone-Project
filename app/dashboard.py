"""Local Streamlit dashboard for the ML-based edge IDS prototype.

This is a research dashboard, not a production security console. It demonstrates
RBAC, model integrity verification, inference, alerts, and resource measurements
from the architecture diagram.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

ROOT = Path(os.getenv("EDGE_IDS_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
CONFIG_PATH = Path(os.getenv("EDGE_IDS_CONFIG", str(ROOT / "configs" / "config.yaml"))).resolve()
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.auth import authenticate  # noqa: E402
from edge_ids.config import load_config  # noqa: E402
from edge_ids.config_manager import update_config_value  # noqa: E402
from edge_ids.dashboard_support import (  # noqa: E402
    build_evaluation_command,
    load_result_csvs,
    resolve_project_output_dir,
    tail_text_file,
)
from edge_ids.experiment_evaluation import discover_datasets  # noqa: E402
from edge_ids.pipeline import EdgeIDSApplication  # noqa: E402

CONFIG = load_config(CONFIG_PATH)
APP = EdgeIDSApplication(CONFIG, ROOT)
USER_FILE = ROOT / CONFIG.get("paths", {}).get("user_file", "configs/users.json")

st.set_page_config(page_title="Edge ML IDS", layout="wide")
st.title("ML-Based IDS — Edge Prototype")
st.caption("Experimental environment only; not a production IDS.")


if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.subheader("Authentication")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        identity = authenticate(USER_FILE, username, password)
        if identity is None:
            st.error("Invalid credentials.")
        else:
            st.session_state.user = identity
            st.rerun()
    st.stop()

user = st.session_state.user
with st.sidebar:
    st.write(f"Signed in as **{user['username']}** ({user['role']})")
    if st.button("Sign out"):
        st.session_state.user = None
        st.rerun()

models = APP.available_models()
if not models:
    st.warning("No registered models found. Train a model with scripts/train_models.py first.")
    st.stop()

model_versions = [item["model_version"] for item in models]
if user["role"] == "ADMIN":
    selected = st.sidebar.selectbox("Model version", model_versions)
else:
    selected = model_versions[-1]
    st.sidebar.caption(f"Analyst model selection is restricted to the registered default: {selected}")

manifest = next(item for item in models if item["model_version"] == selected)

with st.expander("Model Metadata and Integrity", expanded=False):
    # Metadata is available on demand, while the default view remains focused
    # on the user's main task rather than a large manifest block.
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", manifest["metadata"]["model_name"])
    col2.metric("Features", manifest["metadata"]["output_feature_count"])
    col3.metric("Feature reduction", str(manifest["metadata"]["feature_reduction"]))
    st.code(json.dumps(manifest, indent=2), language="json")

st.subheader("Inference")
upload = st.file_uploader("Upload a CSV containing model input features", type=["csv"])
if upload is not None:
    frame = pd.read_csv(upload)
    st.write("Input preview")
    st.dataframe(frame.head(10), use_container_width=True)

    if st.button("Run inference"):
        try:
            # Pass the uploaded frame directly through the shared service layer;
            # avoid writing potentially sensitive input traffic to a temporary CSV.
            predictions, resources = APP.predict_frame(selected, frame)
            st.success("Inference complete.")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Rows", len(predictions))
            m2.metric("Malicious", int(predictions["prediction"].sum()))
            m3.metric("Latency ms/sample", f"{resources['latency_ms_per_sample']:.4f}")
            m4.metric("Throughput rows/sec", f"{resources['throughput_samples_per_second']:.1f}")
            m5.metric("Process RSS MB", f"{resources['rss_after_mb']:.1f}")
            st.dataframe(predictions, use_container_width=True)
            st.download_button(
                "Download predictions as CSV",
                data=predictions.to_csv(index=False).encode("utf-8"),
                file_name="edge_ids_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Inference failed: {exc}")

st.subheader("Multi-Dataset Model Evaluation")
st.caption(
    "Run registered model versions against a directory of labeled CSV datasets. "
    "Evaluation uses bounded batch inference; it does not retrain models. "
    "Large runs execute in a background process and write one CSV per model."
)

# Keep run handles and selected output context in this session so progress can
# be checked without blocking the dashboard while the experiment is running.
for _state_key, _default in (
    ("evaluation_process", None),
    ("evaluation_log_path", ""),
    ("evaluation_output_dir", ""),
    ("evaluation_models", []),
    ("evaluation_command", []),
):
    if _state_key not in st.session_state:
        st.session_state[_state_key] = _default

evaluation_process = st.session_state.evaluation_process
evaluation_running = evaluation_process is not None and evaluation_process.poll() is None
registered_model_versions = [item["model_version"] for item in models]

default_dataset_dir = str((ROOT / "data" / "evaluation_datasets").resolve())
default_output_dir = "results/multi_dataset_evaluation"

if user["role"] == "ADMIN":
    with st.form("multi_dataset_evaluation_form"):
        eval_col1, eval_col2 = st.columns(2)
        with eval_col1:
            dataset_dir_input = st.text_input(
                "Evaluation dataset directory",
                value=default_dataset_dir,
                help="Folder containing the CSV files to evaluate. The data is read in bounded batches.",
            )
            dataset_pattern = st.text_input("Filename pattern", value="*.csv")
            label_column_input = st.text_input(
                "Ground-truth label column (optional)",
                value=str(CONFIG.get("data", {}).get("label_column") or ""),
                help="Leave blank to use the configured label column or infer a common label name.",
            )
        with eval_col2:
            eval_output_dir_input = st.text_input("Output directory", value=default_output_dir)
            eval_batch_size = st.number_input(
                "Inference batch size (rows)", min_value=100, max_value=200_000,
                value=10_000, step=1_000,
                help="Lower values reduce per-batch memory use, but can increase runtime.",
            )
            eval_models_input = st.multiselect(
                "Models to evaluate",
                options=registered_model_versions,
                default=registered_model_versions,
                help="By default, all registered model versions are evaluated.",
            )
        eval_resume = st.checkbox(
            "Resume existing results (skip successful dataset/model rows)", value=False
        )
        eval_fail_fast = st.checkbox("Stop each model after its first dataset error", value=False)
        launch_evaluation = st.form_submit_button(
            "Start multi-dataset evaluation", disabled=evaluation_running
        )

    # Preview the input selection before starting the potentially long run.
    try:
        preview_paths = discover_datasets(
            dataset_dir=dataset_dir_input, pattern=dataset_pattern
        ) if Path(dataset_dir_input).expanduser().is_dir() else []
        if preview_paths:
            st.caption(f"Input selection: {len(preview_paths)} CSV file(s) matched.")
        elif Path(dataset_dir_input).expanduser().is_dir():
            st.warning("No datasets match this directory and filename pattern.")
        else:
            st.caption("Choose an existing directory containing your evaluation CSV files.")
    except Exception as exc:
        st.warning(f"Unable to inspect dataset selection: {exc}")

    if launch_evaluation:
        try:
            if evaluation_running:
                raise RuntimeError("An evaluation job is already running in this dashboard session.")
            dataset_paths = discover_datasets(
                dataset_dir=dataset_dir_input, pattern=dataset_pattern
            )
            if not eval_models_input:
                raise ValueError("Select at least one registered model to evaluate.")
            output_path = resolve_project_output_dir(eval_output_dir_input, ROOT)
            output_path.mkdir(parents=True, exist_ok=True)
            log_path = output_path / "dashboard_evaluation.log"
            evaluator_script = ROOT / "scripts" / "evaluate_models_on_datasets.py"
            command = build_evaluation_command(
                python_executable=sys.executable,
                script_path=evaluator_script,
                config_path=CONFIG_PATH,
                dataset_dir=dataset_dir_input,
                pattern=dataset_pattern,
                output_dir=output_path,
                batch_size=int(eval_batch_size),
                model_versions=eval_models_input,
                label_column=label_column_input,
                resume=eval_resume,
                fail_fast=eval_fail_fast,
            )
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            st.session_state.evaluation_process = process
            st.session_state.evaluation_log_path = str(log_path)
            st.session_state.evaluation_output_dir = str(output_path)
            st.session_state.evaluation_models = list(eval_models_input)
            st.session_state.evaluation_command = command
            st.success(
                f"Evaluation started for {len(eval_models_input)} model(s) across "
                f"{len(dataset_paths)} dataset(s). Process ID: {process.pid}."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not start multi-dataset evaluation: {exc}")
else:
    st.info("Only ADMIN accounts can launch batch evaluations. Existing results can be viewed below.")

# A refresh reruns the script, polls the background process, and updates the log
# excerpt without keeping the Streamlit request blocked for the full experiment.
evaluation_process = st.session_state.get("evaluation_process")
evaluation_log_path = st.session_state.get("evaluation_log_path", "")
if evaluation_process is not None:
    return_code = evaluation_process.poll()
    if return_code is None:
        st.info(f"Multi-dataset evaluation is running (PID {evaluation_process.pid}).")
    elif return_code == 0:
        st.success("Multi-dataset evaluation process finished successfully.")
    else:
        st.error(f"Multi-dataset evaluation process exited with code {return_code}. Check the log below.")
    st.button("Refresh evaluation status", key="refresh_evaluation_status")
    if evaluation_log_path:
        with st.expander("Evaluation log", expanded=(return_code is None)):
            st.code(tail_text_file(evaluation_log_path), language="text")

# Results are read from the per-model files created by the evaluator. The
# combined table remains small (roughly models × datasets rows), even when each
# source dataset contains a million records.
result_dir_value = st.session_state.get("evaluation_output_dir") or str(
    (ROOT / default_output_dir).resolve()
)
result_models = st.session_state.get("evaluation_models") or registered_model_versions
result_directory = Path(result_dir_value)
if result_directory.is_dir():
    combined_results = load_result_csvs(result_directory, result_models)
    if not combined_results.empty:
        st.markdown("**Evaluation results**")
        status_counts = combined_results.get("status", pd.Series(dtype=str)).value_counts().to_dict()
        status_text = ", ".join(f"{key}: {value}" for key, value in status_counts.items()) or "No statuses"
        st.caption(f"Results from `{result_directory}` · {len(combined_results)} dataset/model row(s) · {status_text}")
        display_columns = [
            "status", "model_version", "dataset", "rows_processed", "accuracy",
            "precision", "recall", "f1", "f1_macro", "latency_ms_per_sample",
            "throughput_samples_per_second", "cpu_utilization_percent",
            "sampled_peak_rss_mb", "total_elapsed_seconds", "error_message",
        ]
        visible_columns = [column for column in display_columns if column in combined_results.columns]
        st.dataframe(combined_results[visible_columns], use_container_width=True)
        st.download_button(
            "Download combined evaluation results",
            data=combined_results.to_csv(index=False).encode("utf-8"),
            file_name="multi_dataset_evaluation_combined.csv",
            mime="text/csv",
            key="download_combined_evaluation_results",
        )
        successful = combined_results[
            combined_results.get("status", pd.Series(index=combined_results.index, dtype=str)).eq("ok")
            & combined_results.get("f1", pd.Series(index=combined_results.index, dtype=float)).notna()
        ]
        if not successful.empty and "model_version" in successful and "f1" in successful:
            average_f1 = (
                successful.groupby("model_version", as_index=False)["f1"]
                .mean()
                .rename(columns={"f1": "mean_f1_across_datasets"})
            )
            st.caption("Unweighted mean F1 by model across successful dataset evaluations (descriptive only).")
            st.bar_chart(average_f1.set_index("model_version")["mean_f1_across_datasets"])
        with st.expander("Download individual per-model CSV files"):
            for version in result_models:
                csv_path = result_directory / f"{version}.csv"
                if csv_path.is_file():
                    st.download_button(
                        f"Download {version}.csv",
                        data=csv_path.read_bytes(),
                        file_name=csv_path.name,
                        mime="text/csv",
                        key=f"download_eval_{version}",
                    )

st.subheader("Administration")
if user["role"] == "ADMIN":
    current_threshold = float(CONFIG.get("security", {}).get("alert_confidence_threshold", 0.80))
    new_threshold = st.number_input(
        "Alert confidence threshold", min_value=0.0, max_value=1.0, value=current_threshold, step=0.01
    )
    if st.button("Save threshold"):
        try:
            update_config_value(CONFIG_PATH, "security.alert_confidence_threshold", float(new_threshold))
            st.success("Configuration updated. Restarting the dashboard state.")
            st.rerun()
        except Exception as exc:
            st.error(f"Configuration update failed: {exc}")
else:
    st.info("ANALYST: inference and monitoring views are available; administrative configuration and model selection are restricted.")

st.subheader("Recent Alert Log")
alert_path = ROOT / CONFIG.get("paths", {}).get("log_dir", "logs") / "alerts.jsonl"
if alert_path.exists():
    alert_rows = []
    with alert_path.open("r", encoding="utf-8") as handle:
        for line in handle.readlines()[-20:]:
            try:
                alert_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if alert_rows:
        st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)
    else:
        st.caption("The alert log has no valid records yet.")
else:
    st.caption("No alerts have been generated yet.")
