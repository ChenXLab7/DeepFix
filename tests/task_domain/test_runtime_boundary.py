from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from deepfix.approval import ApprovalPolicy
from deepfix.cli import print_task_list
from deepfix.config import ApprovalMode, load_config
from deepfix.persistence import TaskRepository
from deepfix.service import BugfixService
from deepfix.task_domain.models import TaskDefinition, TaskLifecycleStatus
from deepfix.task_domain.outcome import RepairOutcomeCandidate
from deepfix.task_domain.runtime import TaskRuntime

FORBIDDEN_FACT_FIELDS = {
    "conversation",
    "evidence",
    "hypotheses",
    "changed_files",
    "test_results",
    "approvals",
    "context_metrics",
    "recovery",
    "repair_plan",
    "final_summary",
}


def test_task_runtime_is_a_minimal_non_authoritative_result() -> None:
    assert set(TaskRuntime.model_fields) == {
        "task_id",
        "lifecycle",
        "pending_actions",
        "latest_decision_id",
        "pause_reason",
    }
    assert FORBIDDEN_FACT_FIELDS.isdisjoint(TaskRuntime.model_fields)

    runtime = TaskRuntime(
        task_id="task-a",
        lifecycle=TaskLifecycleStatus.WAITING_APPROVAL,
        pending_actions=[{"name": "edit_file", "args": {"file_path": "parser.py"}}],
    )

    assert runtime.pending_actions[0]["name"] == "edit_file"


@pytest.mark.parametrize("field", sorted(FORBIDDEN_FACT_FIELDS))
def test_task_runtime_rejects_fact_authority_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        TaskRuntime.model_validate(
            {
                "task_id": "task-a",
                "lifecycle": "running",
                field: [],
            }
        )


class _NeedsInputAgent:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, value, config):
        self.calls += 1
        return {
            "messages": [],
            "structured_response": RepairOutcomeCandidate(
                status="needs_input",
                question="Which Python version?",
                summary="Need one environment detail",
            ),
        }


def test_service_start_returns_runtime_projection_not_legacy_aggregate(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    config = load_config(tmp_path, ApprovalMode.MANUAL)
    tasks = TaskRepository(config.database_path)
    service = BugfixService(
        _NeedsInputAgent(),
        tasks,
        ApprovalPolicy(config.approval_mode),
        config,
    )

    result = service.start("parser fails")

    assert isinstance(result, TaskRuntime)
    assert result.lifecycle is TaskLifecycleStatus.PAUSED
    assert result.pause_reason == "Which Python version?"
    assert tasks.get_definition(result.task_id).original_problem == "parser fails"
    with tasks.database.connection() as connection:
        legacy = connection.execute(
            "SELECT 1 FROM legacy_task_projection WHERE task_id = ?",
            (result.task_id,),
        ).fetchone()
    assert legacy is None


def test_service_continue_returns_runtime_projection(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    config = load_config(tmp_path, ApprovalMode.MANUAL)
    tasks = TaskRepository(config.database_path)
    agent = _NeedsInputAgent()
    service = BugfixService(
        agent,
        tasks,
        ApprovalPolicy(config.approval_mode),
        config,
    )
    for method_name in ("_load_current_task", "_save", "_sync_context"):
        assert not hasattr(service, method_name)
    started = service.start("parser fails")

    continued = service.continue_task(started.task_id, "Python 3.12")

    assert isinstance(continued, TaskRuntime)
    assert continued.task_id == started.task_id
    assert agent.calls == 2


def test_pause_uses_lifecycle_without_legacy_task_projection(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    config = load_config(tmp_path, ApprovalMode.MANUAL)
    tasks = TaskRepository(config.database_path)
    tasks.create_definition(
        TaskDefinition(
            task_id="task-runtime-only",
            original_message_id="message-runtime-only",
            original_problem="parser fails",
            approval_mode="manual",
            source_project_root=str(tmp_path),
            workspace_root=str(tmp_path),
            project_python=str(config.project_python),
            confinement_level="legacy_local",
            created_at="2026-08-31T00:00:00+00:00",
        )
    )
    tasks.transition_lifecycle(
        "task-runtime-only",
        TaskLifecycleStatus.RUNNING,
        expected_version=1,
    )
    service = BugfixService(
        _NeedsInputAgent(),
        tasks,
        ApprovalPolicy(config.approval_mode),
        config,
    )

    result = service.pause_task("task-runtime-only", "user paused")

    assert result == TaskRuntime(
        task_id="task-runtime-only",
        lifecycle=TaskLifecycleStatus.PAUSED,
        pause_reason="user paused",
    )
    assert tasks.get_lifecycle("task-runtime-only").reason == "user paused"


class _CheckpointAgent(_NeedsInputAgent):
    def get_state(self, config):
        interrupt = SimpleNamespace(
            value={
                "action_requests": [
                    {
                        "name": "edit_file",
                        "args": {"file_path": "parser.py"},
                        "description": "edit parser",
                    }
                ],
                "review_configs": [
                    {"action_name": "edit_file", "allowed_decisions": ["approve"]}
                ],
            }
        )
        return SimpleNamespace(tasks=[SimpleNamespace(interrupts=(interrupt,))])


def test_pending_actions_are_read_from_graph_checkpoint_not_legacy_payload(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPFIX_HOME", str(tmp_path / "state"))
    config = load_config(tmp_path, ApprovalMode.MANUAL)
    tasks = TaskRepository(config.database_path)
    tasks.create_definition(
        TaskDefinition(
            task_id="task-approval",
            original_message_id="message-approval",
            original_problem="parser fails",
            approval_mode="manual",
            source_project_root=str(tmp_path),
            workspace_root=str(tmp_path),
            project_python=str(config.project_python),
            confinement_level="legacy_local",
            created_at="2026-08-31T00:00:00+00:00",
        )
    )
    tasks.transition_lifecycle(
        "task-approval", TaskLifecycleStatus.RUNNING, expected_version=1
    )
    tasks.transition_lifecycle(
        "task-approval", TaskLifecycleStatus.WAITING_APPROVAL, expected_version=2
    )
    service = BugfixService(
        _CheckpointAgent(),
        tasks,
        ApprovalPolicy(config.approval_mode),
        config,
    )

    actions = service.pending_actions("task-approval")

    assert [(item["name"], item["args"]) for item in actions] == [
        ("edit_file", {"file_path": "parser.py"})
    ]

    resumed = service.decide("task-approval", ["approve"])

    assert isinstance(resumed, TaskRuntime)
    assert [item.decision for item in service.repositories.execution.list_approvals(
        "task-approval"
    )] == ["approve"]


def test_cli_task_list_reads_definition_and_lifecycle_not_legacy_payload(
    tmp_path, monkeypatch
) -> None:
    tasks = TaskRepository(tmp_path / "deepfix.sqlite3")
    tasks.create_definition(
        TaskDefinition(
            task_id="task-list",
            original_message_id="message-list",
            original_problem="parser fails",
            approval_mode="manual",
            source_project_root=str(tmp_path),
            workspace_root=str(tmp_path / "workspace"),
            project_python="python",
            confinement_level="guarded_local",
            created_at="2026-08-31T00:00:00+00:00",
        )
    )
    tasks.transition_lifecycle(
        "task-list", TaskLifecycleStatus.RUNNING, expected_version=1
    )
    assert not hasattr(tasks, "list_recent")
    output: list[str] = []

    print_task_list(tasks, output_fn=output.append)

    assert output == [f"task-list\trunning\t{tmp_path}\tparser fails"]
