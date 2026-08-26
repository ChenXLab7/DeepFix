from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import sys
import threading
from enum import StrEnum
from pathlib import Path
from typing import Literal

from deepagents.backends.protocol import ExecuteResponse

from deepfix.backend import GuardedLocalShellBackend
from deepfix.compaction.models import StrictModel
from deepfix.workspace import TaskWorkspace, WorkspacePathPolicy, WorkspaceScopeError

_SHELL_OPERATORS = re.compile(r"(?:&&|\|\||[\r\n|;&<>()`])")
_SHELL_EXPANSION = re.compile(r"(?:%[^%]+%|\$\(|\$\{|\$env:)", re.IGNORECASE)
_PYTHON_COMMAND = re.compile(r"(?i)^python(?:\.exe)?$")
_READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {"diff", "log", "rev-parse", "show", "status"}
)
_SHELL_MUTATORS = frozenset(
    {
        "copy",
        "cp",
        "del",
        "erase",
        "move",
        "mv",
        "remove-item",
        "ren",
        "rename",
        "rm",
        "rmdir",
    }
)


class ConfinementLevel(StrEnum):
    GUARDED_LOCAL = "guarded_local"
    STRICT = "strict"


class CommandDecision(StrictModel):
    allowed: bool
    reason: str
    confinement_level: ConfinementLevel
    requires_approval: bool


class ApprovalGrant(StrictModel):
    grant_id: str
    task_id: str
    command_hash: str
    source: Literal["approval_record", "trusted_corpus"]


class WorkspaceCommandPolicy:
    def __init__(self, workspace: TaskWorkspace) -> None:
        self.workspace = workspace
        self.paths = WorkspacePathPolicy(workspace.root)

    def evaluate(self, command: str) -> CommandDecision:
        if not isinstance(command, str) or not command.strip():
            return self._deny("command must be a non-empty string")
        if _SHELL_OPERATORS.search(command):
            return self._deny("compound shell commands are not allowed")
        if _SHELL_EXPANSION.search(command):
            return self._deny("shell variable expansion is not allowed")
        try:
            tokens = _command_tokens(command)
        except ValueError:
            return self._deny("command quoting is invalid")
        if not tokens:
            return self._deny("command must be a non-empty string")

        if Path(tokens[0]).is_absolute() or "/" in tokens[0] or "\\" in tokens[0]:
            return self._deny("explicit executable paths are not allowed")
        executable = tokens[0].lower()
        if executable in _SHELL_MUTATORS:
            return self._deny("shell filesystem mutation is not allowed")
        if executable == "git":
            return self._evaluate_git(tokens)
        if _PYTHON_COMMAND.fullmatch(executable):
            return self._evaluate_python(tokens)
        if executable in {"rg", "rg.exe"}:
            if not self._paths_are_scoped(tokens[2:]):
                return self._deny("search references a path outside the workspace")
            return self._allow("read-only repository search", requires_approval=False)
        return self._deny("command is not classified for guarded local execution")

    def _evaluate_git(self, tokens: list[str]) -> CommandDecision:
        lowered = [token.lower() for token in tokens[1:]]
        if "--global" in lowered or "--system" in lowered:
            return self._deny("global or system Git configuration is not allowed")
        if not lowered or lowered[0] not in _READ_ONLY_GIT_SUBCOMMANDS:
            return self._deny("Git command is not read-only")
        if not self._paths_are_scoped(tokens[2:]):
            return self._deny("Git command references a path outside the workspace")
        return self._allow("read-only Git diagnostic", requires_approval=False)

    def _evaluate_python(self, tokens: list[str]) -> CommandDecision:
        lowered = [token.lower() for token in tokens]
        if lowered[1:3] == ["-m", "pip"]:
            return self._deny("package installation is not allowed")
        if lowered[1:3] != ["-m", "pytest"]:
            return self._deny("arbitrary Python execution requires strict confinement")
        if not self._paths_are_scoped(tokens[3:]):
            return self._deny("pytest references a path outside the workspace")
        return self._allow("scoped pytest verification", requires_approval=True)

    def _paths_are_scoped(self, tokens: list[str]) -> bool:
        for token in tokens:
            candidate = _path_argument(token)
            if candidate is None:
                continue
            try:
                self.paths.resolve_allowed(candidate)
            except WorkspaceScopeError:
                return False
        return True

    @staticmethod
    def _allow(reason: str, *, requires_approval: bool) -> CommandDecision:
        return CommandDecision(
            allowed=True,
            reason=reason,
            confinement_level=ConfinementLevel.GUARDED_LOCAL,
            requires_approval=requires_approval,
        )

    @staticmethod
    def _deny(reason: str) -> CommandDecision:
        return CommandDecision(
            allowed=False,
            reason=reason,
            confinement_level=ConfinementLevel.GUARDED_LOCAL,
            requires_approval=False,
        )


class WorkspaceCommandRunner:
    def __init__(
        self,
        workspace: TaskWorkspace,
        policy: WorkspaceCommandPolicy,
        *,
        project_python: str | Path = sys.executable,
        max_output_bytes: int = 100_000,
    ) -> None:
        self.workspace = workspace
        self.policy = policy
        self.project_python = Path(project_python).expanduser().resolve(strict=True)
        self._used_grant_ids: set[str] = set()
        self._grant_lock = threading.Lock()
        self._environment = task_scoped_environment(workspace, self.project_python)
        self._backend = GuardedLocalShellBackend(
            root_dir=workspace.root,
            virtual_mode=True,
            diagnostic_timeout_seconds=300,
            verification_timeout_seconds=300,
            project_python=str(self.project_python),
            max_output_bytes=max_output_bytes,
            env=self._environment,
            inherit_env=False,
        )

    def execute(
        self,
        command: str,
        *,
        timeout: int,
        approval_grant: ApprovalGrant | None,
    ) -> ExecuteResponse:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        decision = self.policy.evaluate(command)
        if not decision.allowed:
            return _denied(decision.reason)
        if decision.requires_approval:
            denial = self._consume_grant(command, approval_grant)
            if denial is not None:
                return _denied(denial)
        bound_command = _bind_project_python(command, self.project_python)
        return self._backend.execute(bound_command, timeout=timeout)

    def _consume_grant(
        self,
        command: str,
        grant: ApprovalGrant | None,
    ) -> str | None:
        if grant is None:
            return "approval required"
        if grant.task_id != self.workspace.task_id:
            return "approval grant belongs to another task"
        if grant.command_hash != normalized_command_hash(command):
            return "approval grant does not match command"
        with self._grant_lock:
            if grant.grant_id in self._used_grant_ids:
                return "approval grant already consumed"
            self._used_grant_ids.add(grant.grant_id)
        return None


def normalized_command_hash(command: str) -> str:
    normalized = re.sub(r"\s+", " ", command.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def task_scoped_environment(
    workspace: TaskWorkspace,
    project_python: Path,
) -> dict[str, str]:
    runtime = workspace.root / ".deepfix-runtime"
    locations = {
        "home": runtime / "home",
        "cache": runtime / "cache",
        "config": runtime / "config",
        "tmp": runtime / "tmp",
        "pip-cache": runtime / "pip-cache",
    }
    for path in locations.values():
        path.mkdir(parents=True, exist_ok=True)
    pip_config = runtime / "pip.conf"
    git_config = runtime / "gitconfig"
    pip_config.touch(exist_ok=True)
    git_config.touch(exist_ok=True)

    environment = {
        name: value
        for name in ("COMSPEC", "PATHEXT", "SYSTEMROOT", "WINDIR")
        if (value := os.environ.get(name)) is not None
    }
    environment.update(
        {
            "HOME": str(locations["home"]),
            "USERPROFILE": str(locations["home"]),
            "XDG_CACHE_HOME": str(locations["cache"]),
            "XDG_CONFIG_HOME": str(locations["config"]),
            "TEMP": str(locations["tmp"]),
            "TMP": str(locations["tmp"]),
            "PIP_CACHE_DIR": str(locations["pip-cache"]),
            "PIP_CONFIG_FILE": str(pip_config),
            "GIT_CONFIG_GLOBAL": str(git_config),
            "GIT_CONFIG_NOSYSTEM": "1",
            "PYTHONNOUSERSITE": "1",
            "PATH": os.pathsep.join(
                item
                for item in (str(project_python.parent), os.environ.get("PATH", ""))
                if item
            ),
        }
    )
    return environment


def _command_tokens(command: str) -> list[str]:
    normalized = command.replace("\\", "/")
    return [token.strip('"\'') for token in shlex.split(normalized, posix=True)]


def _path_argument(token: str) -> str | None:
    value = token.split("=", 1)[1] if token.startswith("--") and "=" in token else token
    value = value.split("::", 1)[0]
    if not value or value.startswith("-"):
        return None
    if value.isdigit():
        return None
    if (
        "/" in value
        or "\\" in value
        or value.startswith(".")
        or Path(value).suffix.lower() in {".py", ".toml", ".ini", ".cfg"}
    ):
        return value
    return None


def _bind_project_python(command: str, project_python: Path) -> str:
    stripped = command.lstrip()
    match = re.match(r"(?i)python(?:\.exe)?(?=\s|$)", stripped)
    if match is None:
        return command
    executable = subprocess.list2cmdline([str(project_python)])
    return f"{executable}{stripped[match.end():]}"


def _denied(reason: str) -> ExecuteResponse:
    return ExecuteResponse(
        output=f"Denied: {reason}",
        exit_code=126,
        truncated=False,
    )
