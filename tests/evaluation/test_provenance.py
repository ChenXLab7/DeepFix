from __future__ import annotations

import subprocess

import pytest

from deepfix.evaluation.provenance import _git_dirty


def test_git_dirty_includes_untracked_files(tmp_path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "untracked.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert _git_dirty(tmp_path) is True


def test_git_dirty_does_not_treat_query_failure_as_clean(tmp_path) -> None:
    with pytest.raises(ValueError, match="Git working tree status"):
        _git_dirty(tmp_path)
