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
