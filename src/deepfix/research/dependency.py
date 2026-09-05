from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from deepfix.research.models import DependencyFinding

_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_ENVIRONMENT_VARIABLES = (
    "COMSPEC",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "WINDIR",
)
_VERSION_PROBE = """
import json
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    payload = {"installed_version": version(sys.argv[1]), "diagnostic": None}
except PackageNotFoundError:
    payload = {"installed_version": None, "diagnostic": "package not installed"}
print(json.dumps(payload))
""".strip()

Runner = Callable[..., subprocess.CompletedProcess[str]]


class DependencyInspector:
    def __init__(
        self,
        project_root: str | Path,
        project_python: str | Path,
        *,
        runner: Runner = subprocess.run,
        timeout_seconds: int = 10,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.project_python = Path(project_python).expanduser().resolve()
        self.runner = runner
        self.timeout_seconds = timeout_seconds

    def inspect(self, package_name: str) -> DependencyFinding:
        if not _PACKAGE_NAME.fullmatch(package_name.strip()):
            raise ValueError(f"依赖包名称无效: {package_name}")
        normalized_name = canonicalize_name(package_name.strip())
        constraints: list[str] = []
        source_files: list[str] = []

        readers = (
            ("pyproject.toml", self._read_pyproject),
            ("requirements.txt", self._read_requirements),
            ("poetry.lock", self._read_lock),
            ("uv.lock", self._read_lock),
        )
        for filename, reader in readers:
            path = self.project_root / filename
            if not path.is_file():
                continue
            found = reader(path, normalized_name)
            if not found:
                continue
            constraints.extend(found)
            source_files.append(filename)

        installed_version, diagnostic = self._probe_installed_version(normalized_name)
        return DependencyFinding(
            package_name=normalized_name,
            declared_constraints=list(dict.fromkeys(constraints)),
            installed_version=installed_version,
            python_executable=str(self.project_python),
            source_files=source_files,
            diagnostic=diagnostic,
        )

    @staticmethod
    def _read_pyproject(path: Path, package_name: str) -> list[str]:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        project = payload.get("project", {})
        dependencies = project.get("dependencies", []) if isinstance(project, dict) else []
        if not isinstance(dependencies, list):
            return []
        return _matching_requirements(dependencies, package_name)

    @staticmethod
    def _read_requirements(path: Path, package_name: str) -> list[str]:
        lines = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = re.split(r"\s+#", raw_line, maxsplit=1)[0].strip()
            if line and not line.startswith(("#", "-")):
                lines.append(line)
        return _matching_requirements(lines, package_name)

    @staticmethod
    def _read_lock(path: Path, package_name: str) -> list[str]:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        packages = payload.get("package", [])
        if not isinstance(packages, list):
            return []
        versions: list[str] = []
        for package in packages:
            if not isinstance(package, dict):
                continue
            name = package.get("name")
            version = package.get("version")
            if (
                isinstance(name, str)
                and canonicalize_name(name) == package_name
                and isinstance(version, str)
                and version.strip()
            ):
                versions.append(f"=={version.strip()}")
        return versions

    def _probe_installed_version(self, package_name: str) -> tuple[str | None, str | None]:
        environment = {
            name: value
            for name in _SAFE_ENVIRONMENT_VARIABLES
            if (value := os.environ.get(name)) is not None
        }
        environment["PYTHONIOENCODING"] = "utf-8"
        command = [
            str(self.project_python),
            "-c",
            _VERSION_PROBE,
            package_name,
        ]
        try:
            result = self.runner(
                command,
                shell=False,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired:
            return None, f"检查已安装版本超时（{self.timeout_seconds} 秒）"
        except OSError as exc:
            return None, _bounded_diagnostic(f"无法启动目标 Python: {exc}")

        if result.returncode != 0:
            detail = (result.stderr or "").strip()
            return None, _bounded_diagnostic(
                f"目标 Python 返回 exit_code={result.returncode}: {detail}"
            )
        try:
            payload: Any = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return None, "目标 Python 返回无效 JSON"
        if not isinstance(payload, dict):
            return None, "目标 Python 返回无效 JSON"
        installed_version = payload.get("installed_version")
        diagnostic = payload.get("diagnostic")
        if installed_version is not None and not isinstance(installed_version, str):
            return None, "目标 Python 返回无效版本信息"
        if diagnostic is not None and not isinstance(diagnostic, str):
            return None, "目标 Python 返回无效诊断信息"
        return installed_version, diagnostic


def _matching_requirements(values: list[object], package_name: str) -> list[str]:
    matches: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        try:
            requirement = Requirement(value)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) == package_name:
            matches.append(value.strip())
    return matches


def _bounded_diagnostic(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"
