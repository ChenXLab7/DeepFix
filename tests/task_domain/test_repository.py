from __future__ import annotations

import pytest

from deepfix.database import SQLiteDatabase
from deepfix.task_domain.models import (
    AdjudicationDecision,
    AdjudicationDecisionConflict,
    TaskDefinition,
    TaskDefinitionConflict,
    TaskLifecycleConflict,
    TaskLifecycleStatus,
)
from deepfix.task_domain.repository import TaskRepository
from deepfix.verification import (
    OracleConflictRule,
    VerificationOracle,
    VerificationPolicy,
    VerificationPolicyConflict,
)


@pytest.fixture
def repository(tmp_path) -> TaskRepository:
    return TaskRepository(SQLiteDatabase(tmp_path / "deepfix.db"))


@pytest.fixture
def definition() -> TaskDefinition:
    return TaskDefinition(
        task_id="task-1",
        original_message_id="message-1",
        original_problem="修复排序错误",
        approval_mode="manual",
        source_project_root="C:/repo",
        workspace_root="C:/workspaces/task-1",
        workspace_baseline_id="baseline-1",
        project_python="C:/Python/python.exe",
        confinement_level="guarded_local",
        created_at="2026-08-29T00:00:00+00:00",
    )


@pytest.fixture
def policy() -> VerificationPolicy:
    required = VerificationOracle(
        oracle_id="oracle-targeted",
        origin="user_specified",
        command="python -m pytest tests/test_value.py -q",
        scope="targeted",
        role="required",
        relevant_paths=["tests/test_value.py"],
    )
    return VerificationPolicy(
        policy_id="policy-1",
        task_id="task-1",
        version=1,
        required_oracles=[required],
        supplemental_oracles=[],
        conflict_rules=[
            OracleConflictRule(
                rule_id="related-failure",
                description="相关回归失败阻止完成",
                blocking_scopes=["module", "full_suite"],
            )
        ],
    )


def test_create_definition_is_idempotent_but_rejects_mutation(
    repository,
    definition,
) -> None:
    assert repository.create_definition(definition) == definition
    assert repository.create_definition(definition) == definition

    changed = definition.model_copy(update={"original_problem": "changed"})
    with pytest.raises(TaskDefinitionConflict, match="immutable"):
        repository.create_definition(changed)


def test_original_message_cannot_define_two_tasks(repository, definition) -> None:
    repository.create_definition(definition)
    conflicting = definition.model_copy(update={"task_id": "task-2"})

    with pytest.raises(TaskDefinitionConflict, match="original message"):
        repository.create_definition(conflicting)


def test_definition_and_lifecycle_are_separate_rows(repository, definition) -> None:
    repository.create_definition(definition)

    assert repository.get_definition(definition.task_id) == definition
    lifecycle = repository.get_lifecycle(definition.task_id)
    assert lifecycle.status is TaskLifecycleStatus.CREATED
    assert lifecycle.version == 1
    assert lifecycle.paused_from is None


def test_lifecycle_uses_optimistic_version_and_transition_rules(
    repository,
    definition,
) -> None:
    repository.create_definition(definition)
    running = repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.RUNNING,
        expected_version=1,
    )

    assert running.version == 2
    assert running.status is TaskLifecycleStatus.RUNNING
    with pytest.raises(TaskLifecycleConflict, match="version"):
        repository.transition_lifecycle(
            definition.task_id,
            TaskLifecycleStatus.COMPLETED,
            expected_version=1,
        )


def test_pause_records_business_status_and_resume_source(repository, definition) -> None:
    repository.create_definition(definition)
    repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.RUNNING,
    )

    paused = repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.PAUSED,
        reason="需要人工输入",
    )
    resumed = repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.RUNNING,
    )

    assert paused.paused_from is TaskLifecycleStatus.RUNNING
    assert paused.reason == "需要人工输入"
    assert resumed.paused_from is None
    assert resumed.reason is None


def test_replaying_current_lifecycle_is_idempotent(repository, definition) -> None:
    repository.create_definition(definition)
    first = repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.RUNNING,
    )

    replay = repository.transition_lifecycle(
        definition.task_id,
        TaskLifecycleStatus.RUNNING,
    )

    assert replay == first


def test_normalized_task_tables_exclude_other_domains(repository) -> None:
    forbidden = {
        "messages",
        "todos",
        "hypotheses",
        "evidence",
        "test_results",
        "changed_files",
        "operations",
        "receipts",
        "approvals",
        "snapshots",
        "artifacts",
        "summary",
        "report",
    }

    with repository.checkpoint_connection() as connection:
        for table in (
            "task_definitions",
            "task_lifecycle",
            "verification_policies",
            "adjudication_decisions",
            "token_budgets",
            "token_reservations",
        ):
            columns = {
                row[1]
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            assert columns
            assert columns.isdisjoint(forbidden)


def test_unknown_task_raises_key_error(repository) -> None:
    with pytest.raises(KeyError, match="missing-task"):
        repository.get_definition("missing-task")
    with pytest.raises(KeyError, match="missing-task"):
        repository.get_lifecycle("missing-task")


def test_task_repository_loads_latest_and_requested_policy_version(
    repository,
    policy,
) -> None:
    repository.save_verification_policy(policy)
    second = policy.model_copy(update={"policy_id": "policy-2", "version": 2})
    repository.save_verification_policy(second)

    assert repository.load_verification_policy(policy.task_id) == second
    assert repository.load_verification_policy(policy.task_id, version=1) == policy
    assert repository.load_verification_policy("missing-task") is None


def test_task_repository_rejects_required_oracle_downgrade(
    repository,
    policy,
) -> None:
    repository.save_verification_policy(policy)
    downgraded = policy.model_copy(
        update={
            "policy_id": "policy-2",
            "version": 2,
            "required_oracles": [],
            "supplemental_oracles": policy.required_oracles,
        }
    )

    with pytest.raises(VerificationPolicyConflict, match="cannot be removed"):
        repository.save_verification_policy(downgraded)


def test_adjudication_persists_only_supporting_ids(repository, definition) -> None:
    repository.create_definition(definition)
    decision = AdjudicationDecision(
        decision_id="decision-1",
        task_id=definition.task_id,
        outcome="fixed",
        evidence_ids=["evidence-1"],
        operation_ids=["operation-1"],
        decided_at="2026-08-29T00:00:00+00:00",
    )

    repository.record_adjudication(decision)

    assert repository.latest_adjudication(definition.task_id) == decision
    with repository.checkpoint_connection() as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(adjudication_decisions)"
            )
        }
    assert columns == {
        "decision_id",
        "task_id",
        "outcome",
        "evidence_ids_json",
        "operation_ids_json",
        "decided_at",
    }


def test_adjudication_replay_is_idempotent_but_changed_content_conflicts(
    repository,
    definition,
) -> None:
    repository.create_definition(definition)
    decision = AdjudicationDecision(
        decision_id="decision-1",
        task_id=definition.task_id,
        outcome="fixed",
        evidence_ids=["evidence-1"],
        operation_ids=[],
        decided_at="2026-08-29T00:00:00+00:00",
    )
    repository.record_adjudication(decision)

    repository.record_adjudication(decision)
    changed = decision.model_copy(update={"outcome": "failed"})

    with pytest.raises(AdjudicationDecisionConflict, match="identity"):
        repository.record_adjudication(changed)
