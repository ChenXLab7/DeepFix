from __future__ import annotations

import subprocess

import pytest

from deepfix.evaluation.provenance import _git_dirty, build_provenance_batch


def test_git_dirty_includes_untracked_files(tmp_path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "untracked.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert _git_dirty(tmp_path) is True


def test_git_dirty_does_not_treat_query_failure_as_clean(tmp_path) -> None:
    with pytest.raises(ValueError, match="Git working tree status"):
        _git_dirty(tmp_path)


def test_runtime_provenance_reports_pre_call_all_role_accounting(
    tmp_path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    python = tmp_path / "python.exe"
    python.touch()
    monkeypatch.setattr(
        "deepfix.evaluation.provenance._git_root",
        lambda path: path,
    )
    monkeypatch.setattr(
        "deepfix.evaluation.provenance._git_value",
        lambda _path, *args: (
            "a" * 40 if args[-1] == "HEAD" else "https://example.invalid/repo"
        ),
    )
    monkeypatch.setattr("deepfix.evaluation.provenance._git_dirty", lambda _path: False)
    monkeypatch.setattr(
        "deepfix.evaluation.provenance._source_tree_sha256",
        lambda _path: "b" * 64,
    )
    monkeypatch.setattr(
        "deepfix.evaluation.provenance._python_version",
        lambda _path: "Python 3.12",
    )
    monkeypatch.setattr(
        "deepfix.evaluation.provenance._file_sha256",
        lambda _path: "c" * 64,
    )
    monkeypatch.setattr(
        "deepfix.evaluation.provenance._endpoint_fingerprint",
        lambda _value: "d" * 64,
    )

    provenance = build_provenance_batch(project, python, ["run-1"])

    assert provenance.budget_enforcement == "pre_call_reservation"
    assert provenance.model_accounting == "all_model_roles"
