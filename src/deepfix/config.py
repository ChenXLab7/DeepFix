from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ApprovalMode(StrEnum):
    MANUAL = "manual"
    GUARDED = "guarded"


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    database_path: Path
    artifacts_path: Path
    model_name: str
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
) -> AppConfig:
    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"项目目录不存在: {project}")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("缺少环境变量 DEEPSEEK_API_KEY")
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
        model_name="deepseek-chat",
        approval_mode=approval_mode,
        project_python=python_executable,
        search_provider=search_provider,
    )


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
