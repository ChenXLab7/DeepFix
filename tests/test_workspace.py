from __future__ import annotations

import json

import pytest

from deepfix.workspace import (
    WorkspaceBaselineError,
    WorkspaceFactory,
    compute_code_state_hash,
)


def test_workspace_copy_preserves_source_and_records_frozen_baseline(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "bug.py").write_text("VALUE = 1\n", encoding="utf-8")
    factory = WorkspaceFactory(tmp_path / "deepfix-workspaces")

    workspace = factory.create("task-1", source)

    assert (workspace.root / "bug.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert workspace.baseline.source_root == str(source.resolve())
    assert workspace.baseline.workspace_root == str(workspace.root)
    assert workspace.baseline.baseline_id
    assert compute_code_state_hash(workspace.root) == workspace.baseline.code_state_hash

    (workspace.root / "bug.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert (source / "bug.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert compute_code_state_hash(workspace.root) != workspace.baseline.code_state_hash


def test_reopening_same_task_validates_baseline(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "x.py").write_text("x = 1\n", encoding="utf-8")
    factory = WorkspaceFactory(tmp_path / "deepfix-workspaces")
    first = factory.create("task-1", source)

    second = factory.load("task-1")

    assert second.baseline == first.baseline
    assert second.root == first.root


def test_workspace_excludes_runtime_and_repository_metadata(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    for relative in (
        ".git/config",
        ".deepfix/state.json",
        ".deepfix-artifacts/debug/log.jsonl",
        "__pycache__/value.pyc",
        ".pytest_cache/state",
        ".venv/pyvenv.cfg",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("runtime\n", encoding="utf-8")

    workspace = WorkspaceFactory(tmp_path / "workspaces").create("task-1", source)

    assert workspace.baseline.managed_file_hashes.keys() == {"value.py"}
    assert not (workspace.root / ".git").exists()
    assert not (workspace.root / ".deepfix").exists()
    assert ".deepfix-baseline.json" not in workspace.baseline.managed_file_hashes


def test_tampered_baseline_identity_is_rejected(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "x.py").write_text("x = 1\n", encoding="utf-8")
    factory = WorkspaceFactory(tmp_path / "workspaces")
    workspace = factory.create("task-1", source)
    baseline_path = workspace.root / ".deepfix-baseline.json"
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload["baseline_id"] = "tampered"
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WorkspaceBaselineError, match="identity"):
        factory.load("task-1")


def test_unsafe_task_id_cannot_escape_workspace_root(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    factory = WorkspaceFactory(tmp_path / "workspaces")

    with pytest.raises(ValueError, match="task_id"):
        factory.create("../outside", source)
