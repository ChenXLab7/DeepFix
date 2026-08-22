from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import SecretStr

_DEFAULT_ENV_FILE = Path(__file__).with_name(".env")
_USE_DEFAULT_ENV_FILE = object()


class ApprovalMode(StrEnum):
    MANUAL = "manual"
    GUARDED = "guarded"


@dataclass(frozen=True)
class ModelRoleConfig:
    model_name: str
    api_key: SecretStr
    base_url: str


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    database_path: Path
    artifacts_path: Path
    main_model: ModelRoleConfig
    compaction_model: ModelRoleConfig
    approval_mode: ApprovalMode
    project_python: Path
    search_provider: str | None = None
    shell_timeout_seconds: int = 120
    max_shell_calls: int = 20
    max_changed_files: int = 10
    max_agent_invocations: int = 30
    max_consecutive_test_failures: int = 3


def load_config(
    project_root: str | Path,
    approval_mode: ApprovalMode,
    *,
    project_python: str | Path | None = None,
    env_file: str | Path | None | object = _USE_DEFAULT_ENV_FILE,
) -> AppConfig:
    resolved_env_file = _DEFAULT_ENV_FILE if env_file is _USE_DEFAULT_ENV_FILE else env_file
    if resolved_env_file is not None:
        load_dotenv(Path(resolved_env_file), override=False, encoding="utf-8")

    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"项目目录不存在: {project}")

    main_key = _configured("DEEPFIX_MAIN_API_KEY") or _configured("DEEPSEEK_API_KEY")
    if main_key is None:
        raise ValueError("缺少 DEEPFIX_MAIN_API_KEY 或 DEEPSEEK_API_KEY")
    compaction_key = (
        _configured("DEEPFIX_COMPACTION_API_KEY")
        or _configured("DEEPFIX_MAIN_API_KEY")
        or _configured("DEEPSEEK_API_KEY")
    )
    if compaction_key is None:
        raise ValueError(
            "缺少 DEEPFIX_COMPACTION_API_KEY、DEEPFIX_MAIN_API_KEY 或 DEEPSEEK_API_KEY"
        )

    base_url = _base_url()
    main_model_name = _model_name("DEEPFIX_MAIN_MODEL", "deepseek-v4-pro")
    compaction_model_name = _model_name(
        "DEEPFIX_COMPACTION_MODEL", "deepseek-v4-flash"
    )
    python_executable = Path(project_python or sys.executable).expanduser().resolve()
    if not python_executable.is_file():
        raise ValueError(f"Python 解释器不存在: {python_executable}")
    search_provider = _load_search_provider()
    database_path = state_database_path()
    artifacts_path = database_path.parent / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        project_root=project,
        database_path=database_path,
        artifacts_path=artifacts_path,
        main_model=ModelRoleConfig(
            model_name=main_model_name,
            api_key=SecretStr(main_key),
            base_url=base_url,
        ),
        compaction_model=ModelRoleConfig(
            model_name=compaction_model_name,
            api_key=SecretStr(compaction_key),
            base_url=base_url,
        ),
        approval_mode=approval_mode,
        project_python=python_executable,
        search_provider=search_provider,
    )


def _configured(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _model_name(name: str, default: str) -> str:
    if name in os.environ and not os.environ[name].strip():
        raise ValueError(f"{name} 不能为空")
    return _configured(name) or default


def _base_url() -> str:
    name = "DEEPSEEK_BASE_URL"
    if name in os.environ and not os.environ[name].strip():
        raise ValueError(f"{name} 不能为空")
    value = _configured(name) or "https://api.deepseek.com"
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{name} 必须是有效的 HTTPS URL")
    return value.rstrip("/")


def redact_config_secrets(value: str, config: AppConfig) -> str:
    result = value
    for role in (config.main_model, config.compaction_model):
        secret = role.api_key.get_secret_value()
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result


def _load_search_provider() -> str | None:
    provider = os.environ.get("DEEPFIX_SEARCH_PROVIDER", "").strip().lower()
    if not provider:
        return None
    if provider != "tavily":
        raise ValueError(f"不支持的 DEEPFIX_SEARCH_PROVIDER: {provider}")
    if not os.environ.get("TAVILY_API_KEY", "").strip():
        return None
    return provider


def state_database_path() -> Path:
    default_home = Path.home() / ".deepfix"
    state_home = Path(os.environ.get("DEEPFIX_HOME", default_home)).expanduser().resolve()
    state_home.mkdir(parents=True, exist_ok=True)
    return state_home / "deepfix.sqlite3"
