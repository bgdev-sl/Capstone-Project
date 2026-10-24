from __future__ import annotations

import pandas as pd
import pytest

from edge_ids.dashboard_support import (
    build_evaluation_command,
    load_result_csvs,
    resolve_project_output_dir,
    tail_text_file,
)


def test_resolve_project_output_dir_accepts_relative_child(tmp_path):
    resolved = resolve_project_output_dir("results/evaluation", tmp_path)
    assert resolved == (tmp_path / "results" / "evaluation").resolve()


def test_resolve_project_output_dir_rejects_path_outside_project(tmp_path):
    with pytest.raises(ValueError, match="subdirectory of the project root"):
        resolve_project_output_dir("../outside", tmp_path)


def test_build_evaluation_command_includes_models_and_optional_flags(tmp_path):
    command = build_evaluation_command(
        python_executable="python",
        script_path=tmp_path / "scripts" / "evaluate.py",
        config_path=tmp_path / "configs" / "config.yaml",
        dataset_dir=tmp_path / "data",
        pattern="*.csv",
        output_dir=tmp_path / "results",
        batch_size=5000,
        model_versions=["tree_full", "mlp_reduced"],
        label_column="Label",
        resume=True,
        fail_fast=True,
    )
    assert "--models" in command
    models_index = command.index("--models")
    assert command[models_index + 1:models_index + 3] == ["tree_full", "mlp_reduced"]
    assert command[command.index("--batch-size") + 1] == "5000"
    assert "--resume" in command
    assert "--fail-fast" in command
    assert command[command.index("--label-column") + 1] == "Label"


def test_build_evaluation_command_rejects_invalid_batch_or_empty_models(tmp_path):
    shared = dict(
        python_executable="python",
        script_path=tmp_path / "evaluate.py",
        config_path=tmp_path / "config.yaml",
        dataset_dir=tmp_path / "data",
        pattern="*.csv",
        output_dir=tmp_path / "results",
        model_versions=["tree_full"],
    )
    with pytest.raises(ValueError, match="positive"):
        build_evaluation_command(**shared, batch_size=0)
    with pytest.raises(ValueError, match="at least one"):
        build_evaluation_command(**{**shared, "model_versions": []}, batch_size=100)


def test_load_result_csvs_combines_selected_model_files(tmp_path):
    pd.DataFrame([{"model_version": "tree", "dataset": "a.csv", "f1": 0.9}]).to_csv(
        tmp_path / "tree.csv", index=False
    )
    pd.DataFrame([{"model_version": "mlp", "dataset": "b.csv", "f1": 0.8}]).to_csv(
        tmp_path / "mlp.csv", index=False
    )
    combined = load_result_csvs(tmp_path, ["tree", "mlp", "missing"])
    assert len(combined) == 2
    assert set(combined["model_version"]) == {"tree", "mlp"}


def test_tail_text_file_reads_only_tail_and_handles_missing(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("0123456789", encoding="utf-8")
    assert "56789" in tail_text_file(log, max_bytes=5)
    assert "Earlier log content omitted" in tail_text_file(log, max_bytes=5)
    assert "not been created" in tail_text_file(tmp_path / "missing.log")
