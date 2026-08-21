from __future__ import annotations

import os
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
    model_name: str
    approval_mode: ApprovalMode
    shell_timeout_seconds: int = 120
    max_shell_calls: int = 20
    max_changed_files: int = 10
    max_agent_invocations: int = 30
    max_consecutive_test_failures: int = 3


def load_config(project_root: str | Path, approval_mode: ApprovalMode) -> AppConfig:
    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"项目目录不存在: {project}")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("缺少环境变量 DEEPSEEK_API_KEY")
    return AppConfig(
        project_root=project,
        database_path=state_database_path(),
        model_name="deepseek-chat",
        approval_mode=approval_mode,
    )


def state_database_path() -> Path:
    default_home = Path.home() / ".deepfix"
    state_home = Path(os.environ.get("DEEPFIX_HOME", default_home)).expanduser().resolve()
    state_home.mkdir(parents=True, exist_ok=True)
    return state_home / "deepfix.sqlite3"
