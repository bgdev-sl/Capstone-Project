# Basic Streamlit Dashboard

from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.auth import authenticate  # noqa: E402
from edge_ids.config import load_config  # noqa: E402
from edge_ids.config_manager import update_config_value  # noqa: E402
from edge_ids.pipeline import EdgeIDSApplication  # noqa: E402

CONFIG = load_config(ROOT / "configs/config.yaml")
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
        temp_input = ROOT / "results" / "_dashboard_input.csv"
        temp_input.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(temp_input, index=False)
        try:
            predictions, resources = APP.predict_csv(selected, temp_input)
            st.success("Inference complete.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", len(predictions))
            m2.metric("Malicious", int(predictions["prediction"].sum()))
            m3.metric("Latency ms/sample", f"{resources['latency_ms_per_sample']:.3f}")
            m4.metric("CPU %", f"{resources['cpu_percent']:.1f}")
            st.dataframe(predictions, use_container_width=True)
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
            update_config_value(ROOT / "configs/config.yaml", "security.alert_confidence_threshold", float(new_threshold))
            st.success("Configuration updated. Restarting the dashboard state.")
            st.rerun()
        except Exception as exc:
            st.error(f"Configuration update failed: {exc}")
else:
    st.info("ANALYST: inference and monitoring views are available; administrative configuration and model selection are restricted.")

st.subheader("Recent Alert Log")
alert_path = ROOT / CONFIG.get("paths", {}).get("log_dir", "logs") / "alerts.jsonl"
if alert_path.exists():
    with alert_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()[-20:]
    if lines:
        st.code("".join(lines), language="json")
else:
    st.caption("No alerts have been generated yet.")
