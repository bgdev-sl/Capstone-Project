from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(os.getenv("EDGE_IDS_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
CONFIG_PATH = Path(os.getenv("EDGE_IDS_CONFIG", str(ROOT / "configs" / "config.yaml"))).resolve()
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.auth import authenticate  # noqa: E402
from edge_ids.config import load_config  # noqa: E402
from edge_ids.config_manager import update_config_value  # noqa: E402
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

st.subheader("Model Integrity and Metadata")
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
