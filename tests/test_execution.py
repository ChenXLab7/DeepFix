from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from deepfix.execution import (
    ApprovalGrant,
    ConfinementLevel,
    WorkspaceCommandPolicy,
    WorkspaceCommandRunner,
    normalized_command_hash,
)
from deepfix.workspace import WorkspaceFactory


@pytest.fixture
def workspace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    return WorkspaceFactory(tmp_path / "workspaces").create("task-1", source)


@pytest.mark.parametrize(
    "command",
    [
        "rm ../outside.txt",
        "python C:/outside/script.py",
        "git config --global user.name agent",
        "python -m pip install demo",
        "python -m pytest ../outside/test_value.py -q",
        "python -m pytest tests/test_value.py -q | more",
        "C:/OtherPython/python.exe -m pytest tests/test_value.py -q",
        "rg secret C:/outside",
        "git status --short & git config --global user.name agent",
        "git status --short\nrm value.py",
        "python -m pytest %SYSTEMROOT%/outside.py -q",
    ],
)
def test_guarded_policy_rejects_workspace_escape_or_global_side_effect(
    command,
    workspace,
) -> None:
    decision = WorkspaceCommandPolicy(workspace).evaluate(command)

    assert decision.allowed is False
    assert decision.confinement_level is ConfinementLevel.GUARDED_LOCAL


def test_guarded_policy_allows_scoped_pytest_but_requires_approval(workspace) -> None:
    decision = WorkspaceCommandPolicy(workspace).evaluate(
        "python -m pytest tests/test_value.py -q"
    )

    assert decision.allowed is True
    assert decision.confinement_level is ConfinementLevel.GUARDED_LOCAL
    assert decision.requires_approval is True


def test_guarded_policy_allows_read_only_repository_diagnostic(workspace) -> None:
    decision = WorkspaceCommandPolicy(workspace).evaluate("git status --short")

    assert decision.allowed is True
    assert decision.requires_approval is False


def test_runner_denies_scoped_pytest_without_matching_approval(workspace) -> None:
    command = "python -m pytest tests/test_value.py -q"
    runner = WorkspaceCommandRunner(
        workspace,
        WorkspaceCommandPolicy(workspace),
        project_python=Path(sys.executable),
    )

    result = runner.execute(command, timeout=30, approval_grant=None)

    assert result.exit_code == 126
    assert "approval required" in result.output.lower()


def test_runner_redirects_user_state_and_consumes_approval_once(workspace) -> None:
    test_file = workspace.root / "test_task_environment.py"
    output_file = workspace.root / "environment.json"
    test_file.write_text(
        """
import json
import os
from pathlib import Path

def test_environment_is_task_scoped():
    Path('environment.json').write_text(json.dumps({
        name: os.environ[name]
        for name in ('HOME', 'USERPROFILE', 'TEMP', 'TMP', 'PIP_CONFIG_FILE', 'GIT_CONFIG_GLOBAL', 'PYTHONNOUSERSITE')
    }), encoding='utf-8')
""".lstrip(),
        encoding="utf-8",
    )
    command = "python -m pytest test_task_environment.py -q"
    grant = ApprovalGrant(
        grant_id="grant-1",
        task_id=workspace.task_id,
        command_hash=normalized_command_hash(command),
        source="approval_record",
    )
    runner = WorkspaceCommandRunner(
        workspace,
        WorkspaceCommandPolicy(workspace),
        project_python=Path(sys.executable),
    )

    first = runner.execute(command, timeout=30, approval_grant=grant)
    second = runner.execute(command, timeout=30, approval_grant=grant)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 126
    assert "already consumed" in second.output.lower()
    environment = json.loads(output_file.read_text(encoding="utf-8"))
    assert environment.pop("PYTHONNOUSERSITE") == "1"
    assert all(
        _is_relative_to(Path(value), workspace.root)
        for value in environment.values()
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def test_configured_wrapper_uses_same_policy_as_python_alias(workspace):
    wrapper = workspace.root.parent / "Python Runtime" / "pytest-wrapper.exe"
    policy = WorkspaceCommandPolicy(workspace, project_python=wrapper)
    for arguments in (
        "-m pytest test_value.py -q",
        "-m pytest ../outside.py",
        "-m pip install demo",
        '-c "print(1)"',
        "-m pytest test_value.py | more",
    ):
        assert policy.evaluate(f'"{wrapper}" {arguments}') == policy.evaluate(
            f"python {arguments}"
        )
    assert policy.evaluate(f'"{wrapper}" -m pytest test_value.py').allowed
    assert not policy.evaluate(f'"{wrapper.parent / "other.exe"}" -m pytest').allowed


def test_runner_accepts_explicit_configured_interpreter(workspace):
    (workspace.root / "test_value.py").write_text("def test_value(): assert True\n")
    command = f'"{sys.executable}" -m pytest test_value.py -q'
    runner = WorkspaceCommandRunner(
        workspace, WorkspaceCommandPolicy(workspace), project_python=sys.executable
    )
    grant = ApprovalGrant(
        grant_id="explicit", task_id=workspace.task_id,
        command_hash=normalized_command_hash(command), source="approval_record",
    )
    result = runner.execute(command, timeout=30, approval_grant=grant)
    assert result.exit_code == 0, result.output


def test_runner_keeps_configured_symlink_path(workspace, monkeypatch):
    # Simulate a venv symlink on Windows without requiring symlink privileges.
    configured = workspace.root / "venv" / "bin" / "python"
    configured.parent.mkdir(parents=True)
    configured.touch()
    original_resolve = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda path, **kw: (
        Path(sys.executable) if path == configured else original_resolve(path, **kw)
    ))
    runner = WorkspaceCommandRunner(
        workspace, WorkspaceCommandPolicy(workspace), project_python=configured
    )
    assert runner.project_python == configured


def test_runner_defaults_to_policy_interpreter_without_mutating_policy(workspace):
    configured = workspace.root / "wrapper"
    configured.touch()
    policy = WorkspaceCommandPolicy(workspace, project_python=configured)
    runner = WorkspaceCommandRunner(workspace, policy)
    assert runner.project_python == configured
    WorkspaceCommandRunner(workspace, policy, project_python=sys.executable)
    assert policy.project_python == configured


@pytest.mark.parametrize("arguments", [
    "-m pytest ../outside.py", "-m pip install demo",
    "-m pytest test_value.py && echo unsafe",
])
def test_approval_cannot_override_configured_interpreter_denials(workspace, arguments):
    command = f'"{sys.executable}" {arguments}'
    runner = WorkspaceCommandRunner(workspace, WorkspaceCommandPolicy(workspace))
    grant = ApprovalGrant(
        grant_id="denied", task_id=workspace.task_id,
        command_hash=normalized_command_hash(command), source="approval_record",
    )
    result = runner.execute(command, timeout=30, approval_grant=grant)
    assert result.exit_code == 126


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX venv symlink integration")
def test_runner_invokes_venv_symlink_without_switching_to_base_python(workspace):
    import subprocess

    from deepfix.execution import _bind_project_python

    venv = workspace.root / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    configured = venv / "bin" / "python"
    runner = WorkspaceCommandRunner(
        workspace, WorkspaceCommandPolicy(workspace), project_python=configured,
    )
    command = _bind_project_python('python -c "import sys; print(sys.prefix)"',
                                   runner.project_python)
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    assert Path(result.stdout.strip()) == venv
