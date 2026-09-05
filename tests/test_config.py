import os
import sys
from pathlib import Path

import pytest
from pydantic import SecretStr

from deepfix.config import ApprovalMode, load_config, redact_config_secrets

_MODEL_ENV_NAMES = (
    "DEEPSEEK_API_KEY",
    "DEEPFIX_MAIN_API_KEY",
    "DEEPFIX_COMPACTION_API_KEY",
    "DEEPFIX_MAIN_MODEL",
    "DEEPFIX_COMPACTION_MODEL",
    "DEEPSEEK_BASE_URL",
    "DEEPFIX_DIAGNOSTIC_TIMEOUT_SECONDS",
    "DEEPFIX_VERIFICATION_TIMEOUT_SECONDS",
    "DEEPFIX_MAIN_REQUEST_TIMEOUT_SECONDS",
    "DEEPFIX_COMPACTION_REQUEST_TIMEOUT_SECONDS",
    "DEEPFIX_MAX_GRAPH_STEPS",
)


@pytest.fixture(autouse=True)
def isolate_model_environment(monkeypatch, tmp_path):
    missing = object()
    original = {name: os.environ.get(name, missing) for name in _MODEL_ENV_NAMES}
    for name in _MODEL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    yield
    for name, value in original.items():
        if value is missing:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def load_isolated(project, mode, tmp_path, **kwargs):
    return load_config(
        project,
        mode,
        env_file=tmp_path / "missing.env",
        **kwargs,
    )


def test_load_config_resolves_project_and_state_paths(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    state_home = tmp_path / "state"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(state_home))

    config = load_isolated(project, ApprovalMode.MANUAL, tmp_path)

    assert config.project_root == project.resolve()
    assert config.database_path == state_home.resolve() / "deepfix.sqlite3"
    assert config.artifacts_path == state_home.resolve() / "artifacts"
    assert config.artifacts_path.is_dir()
    assert config.workspaces_path == state_home.resolve() / "workspaces"
    assert config.workspaces_path.is_dir()
    assert config.main_model.model_name == "deepseek-v4-pro"
    assert config.compaction_model.model_name == "deepseek-v4-flash"
    assert config.approval_mode is ApprovalMode.MANUAL
    assert config.diagnostic_timeout_seconds == 10
    assert config.verification_timeout_seconds == 120
    assert config.main_model.request_timeout_seconds == 120
    assert config.compaction_model.request_timeout_seconds == 120
    assert config.max_graph_steps == 80


def test_load_config_accepts_separate_shell_timeout_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_DIAGNOSTIC_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("DEEPFIX_VERIFICATION_TIMEOUT_SECONDS", "45")

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)

    assert config.diagnostic_timeout_seconds == 3
    assert config.verification_timeout_seconds == 45


def test_load_config_accepts_separate_model_request_timeouts(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_MAIN_REQUEST_TIMEOUT_SECONDS", "75")
    monkeypatch.setenv("DEEPFIX_COMPACTION_REQUEST_TIMEOUT_SECONDS", "45")

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)

    assert config.main_model.request_timeout_seconds == 75
    assert config.compaction_model.request_timeout_seconds == 45


def test_load_config_accepts_graph_step_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_MAX_GRAPH_STEPS", "24")

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)

    assert config.max_graph_steps == 24


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DEEPFIX_DIAGNOSTIC_TIMEOUT_SECONDS", "0"),
        ("DEEPFIX_DIAGNOSTIC_TIMEOUT_SECONDS", "abc"),
        ("DEEPFIX_VERIFICATION_TIMEOUT_SECONDS", "-1"),
        ("DEEPFIX_MAIN_REQUEST_TIMEOUT_SECONDS", "0"),
        ("DEEPFIX_COMPACTION_REQUEST_TIMEOUT_SECONDS", "abc"),
    ],
)
def test_load_config_rejects_invalid_shell_timeout_limits(
    tmp_path,
    monkeypatch,
    name,
    value,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)


def test_load_config_rejects_missing_project(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    with pytest.raises(ValueError, match="项目目录不存在"):
        load_isolated(tmp_path / "missing", ApprovalMode.GUARDED, tmp_path)


def test_load_config_requires_deepseek_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)


def test_load_config_uses_current_interpreter_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)

    assert config.project_python == Path(sys.executable).resolve()


def test_load_config_resolves_explicit_project_interpreter(tmp_path, monkeypatch):
    project_python = tmp_path / "venv" / "python.exe"
    project_python.parent.mkdir()
    project_python.touch()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))

    config = load_isolated(
        tmp_path,
        ApprovalMode.MANUAL,
        tmp_path,
        project_python=project_python,
    )

    assert config.project_python == project_python.resolve()


def test_load_config_rejects_missing_project_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    with pytest.raises(ValueError, match="Python 解释器不存在"):
        load_isolated(
            tmp_path,
            ApprovalMode.MANUAL,
            tmp_path,
            project_python=tmp_path / "missing-python",
        )


def test_load_config_enables_tavily_only_when_key_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("DEEPFIX_SEARCH_PROVIDER", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    without_key = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-secret")
    with_key = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)

    assert without_key.search_provider is None
    assert with_key.search_provider == "tavily"


def test_load_config_rejects_unknown_search_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPFIX_SEARCH_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="DEEPFIX_SEARCH_PROVIDER"):
        load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)


def test_model_roles_use_v4_defaults_and_common_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "common-secret")

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)

    assert config.main_model.model_name == "deepseek-v4-pro"
    assert config.compaction_model.model_name == "deepseek-v4-flash"
    assert config.main_model.api_key.get_secret_value() == "common-secret"
    assert config.compaction_model.api_key.get_secret_value() == "common-secret"
    assert isinstance(config.main_model.api_key, SecretStr)


def test_compaction_key_falls_back_to_main_role_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "main-secret")

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)

    assert config.main_model.api_key.get_secret_value() == "main-secret"
    assert config.compaction_model.api_key.get_secret_value() == "main-secret"


def test_role_specific_keys_and_models_remain_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "common-secret")
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "main-secret")
    monkeypatch.setenv("DEEPFIX_COMPACTION_API_KEY", "compact-secret")
    monkeypatch.setenv("DEEPFIX_MAIN_MODEL", "main-model")
    monkeypatch.setenv("DEEPFIX_COMPACTION_MODEL", "compact-model")

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)

    assert config.main_model.model_name == "main-model"
    assert config.compaction_model.model_name == "compact-model"
    assert config.main_model.api_key.get_secret_value() == "main-secret"
    assert config.compaction_model.api_key.get_secret_value() == "compact-secret"


def test_package_env_file_does_not_override_process_environment(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=file-common\n"
        "DEEPFIX_MAIN_API_KEY=file-main\n"
        "DEEPFIX_MAIN_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "process-main")

    config = load_config(tmp_path, ApprovalMode.MANUAL, env_file=env_file)

    assert config.main_model.api_key.get_secret_value() == "process-main"
    assert config.compaction_model.api_key.get_secret_value() == "process-main"
    assert config.main_model.model_name == "file-model"


@pytest.mark.parametrize(
    "name",
    ["DEEPFIX_MAIN_MODEL", "DEEPFIX_COMPACTION_MODEL"],
)
def test_blank_model_name_is_rejected(tmp_path, monkeypatch, name):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv(name, "   ")

    with pytest.raises(ValueError, match=name):
        load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)


@pytest.mark.parametrize(
    "url",
    ["", "http://api.deepseek.com", "not-a-url"],
)
def test_base_url_requires_https(tmp_path, monkeypatch, url):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", url)

    with pytest.raises(ValueError, match="DEEPSEEK_BASE_URL"):
        load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)


def test_config_repr_and_redaction_never_expose_role_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "main-secret-value")
    monkeypatch.setenv("DEEPFIX_COMPACTION_API_KEY", "compact-secret-value")

    config = load_isolated(tmp_path, ApprovalMode.MANUAL, tmp_path)
    rendered = repr(config)
    redacted = redact_config_secrets(
        "main-secret-value / compact-secret-value",
        config,
    )

    assert "main-secret-value" not in rendered
    assert "compact-secret-value" not in rendered
    assert redacted == "[REDACTED] / [REDACTED]"
