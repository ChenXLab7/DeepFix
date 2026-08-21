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
