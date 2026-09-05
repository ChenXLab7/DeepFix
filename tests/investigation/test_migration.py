from __future__ import annotations

import sys

from langchain_core.messages import AIMessage, ToolMessage

from deepfix.compaction.models import SystemTestEvidence
from deepfix.domain_repositories import DomainRepositories
from deepfix.investigation.migration import InvestigationMigrator
from deepfix.task_domain.models import TaskDefinition


def _fixture(tmp_path):
    repositories = DomainRepositories.create(tmp_path / "deepfix.sqlite3")
    definition = TaskDefinition(
        task_id="task-a",
        original_message_id="message-user",
        original_problem="legacy bug",
        approval_mode="manual",
        source_project_root=str(tmp_path),
        workspace_root=str(tmp_path),
        workspace_baseline_id=None,
        project_python=sys.executable,
        confinement_level="legacy_local",
        created_at="2026-08-31T00:00:00+00:00",
    )
    repositories.tasks.create_definition(definition)
    return repositories, InvestigationMigrator(
        tasks=repositories.tasks,
        store=repositories.investigation,
        evidence_repository=repositories.evidence,
    )


def test_migrator_restores_paired_test_tool_unit_from_current_evidence(tmp_path):
    repositories, migrator = _fixture(tmp_path)
    evidence = SystemTestEvidence(
        evidence_id="evidence-test",
        command=f'"{sys.executable}" -m pytest -q',
        exit_code=1,
        summary="1 failed",
        tool_call_id="call-test",
        source_message_id="message-tool",
    )
    repositories.evidence.record_deterministic(
        "task-a",
        evidence,
        provenance_root_ids=[evidence.evidence_id],
    )
    messages = [
        AIMessage(
            id="message-ai",
            content="run test",
            tool_calls=[
                {
                    "id": "call-test",
                    "name": "execute",
                    "args": {"command": "python -m pytest -q"},
                }
            ],
        ),
        ToolMessage(
            id="message-tool",
            tool_call_id="call-test",
            content="1 failed\nExit code: 1",
            artifact={"exit_code": 1},
        ),
    ]

    state = migrator.migrate("task-a", messages)

    assert state.migration_version == 1
    assert state.test_evidence_ids == ["evidence-test"]
    assert repositories.investigation.has_event(
        "task-a",
        next(
            event.event_id
            for event in repositories.investigation.list_events("task-a")
            if event.event_type.value == "test_observed"
        ),
    )


def test_migrator_ignores_unpaired_tool_call_and_is_idempotent(tmp_path):
    repositories, migrator = _fixture(tmp_path)
    messages = [
        AIMessage(
            id="message-ai",
            content="run test",
            tool_calls=[
                {
                    "id": "call-test",
                    "name": "execute",
                    "args": {"command": "python -m pytest -q"},
                }
            ],
        )
    ]

    first = migrator.migrate("task-a", messages)
    second = migrator.migrate("task-a", messages)

    assert first == second
    assert first.test_evidence_ids == []
    assert repositories.investigation.last_sequence("task-a") == 1
