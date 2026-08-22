import sys
from pathlib import Path

import pytest

from deepfix.config import ApprovalMode, load_config


def test_load_config_resolves_project_and_state_paths(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    state_home = tmp_path / "state"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(state_home))

    config = load_config(project, ApprovalMode.MANUAL)

    assert config.project_root == project.resolve()
    assert config.database_path == state_home.resolve() / "deepfix.sqlite3"
    assert config.artifacts_path == state_home.resolve() / "artifacts"
    assert config.artifacts_path.is_dir()
    assert config.model_name == "deepseek-chat"
    assert config.approval_mode is ApprovalMode.MANUAL


def test_load_config_rejects_missing_project(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    with pytest.raises(ValueError, match="项目目录不存在"):
        load_config(tmp_path / "missing", ApprovalMode.GUARDED)


def test_load_config_requires_deepseek_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_config(tmp_path, ApprovalMode.MANUAL)


def test_load_config_uses_current_interpreter_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))

    config = load_config(tmp_path, ApprovalMode.MANUAL)

    assert config.project_python == Path(sys.executable).resolve()


def test_load_config_resolves_explicit_project_interpreter(tmp_path, monkeypatch):
    project_python = tmp_path / "venv" / "python.exe"
    project_python.parent.mkdir()
    project_python.touch()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))

    config = load_config(
        tmp_path,
        ApprovalMode.MANUAL,
        project_python=project_python,
    )

    assert config.project_python == project_python.resolve()


def test_load_config_rejects_missing_project_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    with pytest.raises(ValueError, match="Python 解释器不存在"):
        load_config(
            tmp_path,
            ApprovalMode.MANUAL,
            project_python=tmp_path / "missing-python",
        )


def test_load_config_enables_tavily_only_when_key_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("DEEPFIX_SEARCH_PROVIDER", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    without_key = load_config(tmp_path, ApprovalMode.MANUAL)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-secret")
    with_key = load_config(tmp_path, ApprovalMode.MANUAL)

    assert without_key.search_provider is None
    assert with_key.search_provider == "tavily"


def test_load_config_rejects_unknown_search_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_SEARCH_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="DEEPFIX_SEARCH_PROVIDER"):
        load_config(tmp_path, ApprovalMode.MANUAL)
