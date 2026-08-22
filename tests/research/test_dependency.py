from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from deepfix.research.dependency import DependencyInspector


@dataclass
class RecordingRunner:
    result: subprocess.CompletedProcess[str] | None = None
    error: Exception | None = None
    calls: list[tuple[list[str], dict[str, object]]] = field(default_factory=list)

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), dict(kwargs)))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _completed(
    *,
    installed_version: str | None = "2.8.4",
    diagnostic: str | None = None,
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    stdout = json.dumps(
        {
            "installed_version": installed_version,
            "diagnostic": diagnostic,
        }
    )
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def _inspector(
    project: Path,
    runner: RecordingRunner,
    project_python: Path | None = None,
) -> DependencyInspector:
    return DependencyInspector(
        project_root=project,
        project_python=project_python or Path("C:/project/.venv/Scripts/python.exe"),
        runner=runner,
    )


def test_inspect_accumulates_supported_declarations_without_duplicates(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
dependencies = [
  "pydantic>=2,<3",
  "httpx>=0.27; python_version >= '3.11'",
]
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text(
        "pydantic>=2,<3  # same constraint\npytest>=8\n",
        encoding="utf-8",
    )
    (tmp_path / "poetry.lock").write_text(
        '[[package]]\nname = "pydantic"\nversion = "2.8.4"\n',
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "pydantic"\nversion = "2.8.4"\n',
        encoding="utf-8",
    )
    runner = RecordingRunner(result=_completed())

    finding = _inspector(tmp_path, runner).inspect("pydantic")

    assert finding.package_name == "pydantic"
    assert finding.declared_constraints == ["pydantic>=2,<3", "==2.8.4"]
    assert finding.source_files == [
        "pyproject.toml",
        "requirements.txt",
        "poetry.lock",
        "uv.lock",
    ]
    assert finding.installed_version == "2.8.4"


def test_inspect_normalizes_distribution_names(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["pydantic-settings>=2"]\n',
        encoding="utf-8",
    )
    runner = RecordingRunner(result=_completed(installed_version="2.4.0"))

    finding = _inspector(tmp_path, runner).inspect("Pydantic_Settings")

    assert finding.package_name == "pydantic-settings"
    assert finding.declared_constraints == ["pydantic-settings>=2"]
    assert finding.installed_version == "2.4.0"


def test_missing_package_does_not_use_deepfix_environment_version(tmp_path):
    (tmp_path / "requirements.txt").write_text("httpx>=0.27\n", encoding="utf-8")
    runner = RecordingRunner(
        result=_completed(installed_version=None, diagnostic="package not installed")
    )

    finding = _inspector(tmp_path, runner).inspect("pydantic")

    assert finding.declared_constraints == []
    assert finding.source_files == []
    assert finding.installed_version is None
    assert finding.diagnostic == "package not installed"


def test_probe_uses_fixed_script_argument_list_and_sanitized_environment(
    tmp_path,
    monkeypatch,
):
    project_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    runner = RecordingRunner(result=_completed())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-leak")
    monkeypatch.setenv("DATABASE_URL", "must-not-leak")
    for name in ("COMSPEC", "PATHEXT", "TEMP", "TMP", "WINDIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SYSTEMROOT", "C:/Windows")

    finding = _inspector(tmp_path, runner, project_python).inspect("Pydantic")

    args, kwargs = runner.calls[0]
    assert args[0] == str(project_python.resolve())
    assert args[1] == "-c"
    assert "pydantic" not in args[2].lower()
    assert args[3] == "pydantic"
    assert kwargs["shell"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    assert kwargs["timeout"] == 10
    assert kwargs["env"] == {
        "PYTHONIOENCODING": "utf-8",
        "SYSTEMROOT": "C:/Windows",
    }
    assert finding.python_executable == str(project_python.resolve())


def test_probe_timeout_returns_diagnostic(tmp_path):
    runner = RecordingRunner(
        error=subprocess.TimeoutExpired(cmd=["python"], timeout=10)
    )

    finding = _inspector(tmp_path, runner).inspect("pydantic")

    assert finding.installed_version is None
    assert "超时" in str(finding.diagnostic)


def test_probe_nonzero_exit_returns_bounded_diagnostic(tmp_path):
    runner = RecordingRunner(
        result=_completed(returncode=3, stderr="broken environment" * 100)
    )

    finding = _inspector(tmp_path, runner).inspect("pydantic")

    assert finding.installed_version is None
    assert "exit_code=3" in str(finding.diagnostic)
    assert len(str(finding.diagnostic)) <= 500


def test_probe_malformed_json_returns_diagnostic(tmp_path):
    runner = RecordingRunner(
        result=subprocess.CompletedProcess([], 0, "not-json", "")
    )

    finding = _inspector(tmp_path, runner).inspect("pydantic")

    assert finding.installed_version is None
    assert "无效 JSON" in str(finding.diagnostic)


def test_probe_rejects_invalid_package_name_before_execution(tmp_path):
    runner = RecordingRunner(result=_completed())

    with pytest.raises(ValueError, match="依赖包名称无效"):
        _inspector(tmp_path, runner).inspect("pydantic; import os")

    assert runner.calls == []
