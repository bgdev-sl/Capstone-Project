"""Small, testable helpers used by the Streamlit dashboard.

These functions intentionally avoid importing Streamlit so they can be covered
by ordinary unit tests and used safely from CLI/test environments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def resolve_project_output_dir(value: str | Path, project_root: str | Path) -> Path:
    """Resolve an output directory and prevent writes outside the project root."""
    root = Path(project_root).expanduser().resolve()
    candidate = Path(value).expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError("Output directory must be a subdirectory of the project root")
    return resolved


def build_evaluation_command(
    *,
    python_executable: str,
    script_path: str | Path,
    config_path: str | Path,
    dataset_dir: str | Path,
    pattern: str,
    output_dir: str | Path,
    batch_size: int,
    model_versions: Sequence[str],
    label_column: str = "",
    resume: bool = False,
    fail_fast: bool = False,
) -> list[str]:
    """Build the evaluator CLI command without invoking a shell."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if not model_versions:
        raise ValueError("Select at least one registered model")

    command = [
        python_executable,
        str(Path(script_path).resolve()),
        "--dataset-dir", str(Path(dataset_dir).expanduser().resolve()),
        "--pattern", pattern,
        "--config", str(Path(config_path).expanduser().resolve()),
        "--output-dir", str(Path(output_dir).expanduser().resolve()),
        "--batch-size", str(batch_size),
        "--models", *[str(version) for version in model_versions],
    ]
    if label_column.strip():
        command.extend(["--label-column", label_column.strip()])
    if resume:
        command.append("--resume")
    if fail_fast:
        command.append("--fail-fast")
    return command


def tail_text_file(path: str | Path, max_bytes: int = 24_000) -> str:
    """Return the end of a log file without reading an unbounded file into RAM."""
    log_path = Path(path)
    if not log_path.is_file():
        return "Log file has not been created yet."
    size = log_path.stat().st_size
    with log_path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(-max_bytes, 2)
        data = handle.read()
    text = data.decode("utf-8", errors="replace")
    if size > max_bytes:
        text = "[Earlier log content omitted... ]\n" + text
    return text.strip() or "Log file is currently empty."


def load_result_csvs(output_dir: str | Path, model_versions: Sequence[str]) -> pd.DataFrame:
    """Combine per-model result CSV files for a compact dashboard summary."""
    root = Path(output_dir)
    frames: list[pd.DataFrame] = []
    for version in model_versions:
        path = root / f"{version}.csv"
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
