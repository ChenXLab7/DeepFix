from __future__ import annotations

from pathlib import Path

from deepfix.compaction.models import FileChangeEvidence, SystemTestEvidence
from deepfix.domain_repositories.evidence import EvidenceRepository
from deepfix.domain_repositories.execution import ExecutionIntegrity
from deepfix.task_domain.adjudication import (
    OutcomeAdjudicationInput,
    OutcomeAdjudicator,
)
from deepfix.task_domain.models import TaskDefinition, TaskLifecycle, TaskLifecycleStatus
from deepfix.verification import (
    OracleConflictRule,
    VerificationOracle,
    VerificationPolicy,
)

TASK_ID = "task-adjudication"
TARGET_COMMAND = "python -m pytest tests/test_value.py -q"


def test_fixed_requires_required_oracle_change_and_clean_execution(tmp_path: Path) -> None:
    evidence = EvidenceRepository(tmp_path / "deepfix.db")
    change = _change("change-1")
    test = _test("target-pass", TARGET_COMMAND, exit_code=0)
    _record(evidence, change)
    _record(evidence, test)

    result = OutcomeAdjudicator().decide(
        _input(
            evidence,
            successful_change_evidence_ids=[change.evidence_id],
        )
    )

    assert result.outcome == "fixed"
    assert result.decision is not None
    assert result.decision.outcome == "fixed"
    assert result.decision.evidence_ids == ["change-1", "target-pass"]
    assert result.decision.operation_ids == []
    assert not hasattr(result.decision, "evidence")


def test_higher_authority_repository_conflict_blocks_fixed(tmp_path: Path) -> None:
    evidence = EvidenceRepository(tmp_path / "deepfix.db")
    _record(evidence, _change("change-1"))
    _record(evidence, _test("target-pass", TARGET_COMMAND, exit_code=0))
    _record(
        evidence,
        _test(
            "suite-failure",
            "python -m pytest -q",
            exit_code=1,
            origin="repository_existing",
            scope="full_suite",
        ),
    )

    result = OutcomeAdjudicator().decide(
        _input(evidence, successful_change_evidence_ids=["change-1"])
    )

    assert result.outcome == "review"
    assert result.decision is None
    assert result.blocking_evidence_ids == ["suite-failure"]


def test_agent_generated_test_or_executor_claim_cannot_create_fixed(
    tmp_path: Path,
) -> None:
    evidence = EvidenceRepository(tmp_path / "deepfix.db")
    _record(evidence, _change("change-1"))
    _record(
        evidence,
        _test(
            "agent-pass",
            TARGET_COMMAND,
            exit_code=0,
            origin="agent_generated",
        ),
    )

    result = OutcomeAdjudicator().decide(
        _input(evidence, successful_change_evidence_ids=["change-1"])
    )

    assert result.outcome == "continue"
    assert result.decision is None
    assert result.reason == "required verification is incomplete"


def test_incomplete_execution_pauses_with_operation_ids(tmp_path: Path) -> None:
    evidence = EvidenceRepository(tmp_path / "deepfix.db")
    _record(evidence, _change("change-1"))
    _record(evidence, _test("target-pass", TARGET_COMMAND, exit_code=0))

    result = OutcomeAdjudicator().decide(
        _input(
            evidence,
            successful_change_evidence_ids=["change-1"],
            execution=ExecutionIntegrity(
                status_counts={"unknown": 1},
                receipt_count=1,
                approval_count=0,
                incomplete_operation_ids=["operation-unknown"],
                unknown_operation_ids=["operation-unknown"],
            ),
        )
    )

    assert result.outcome == "paused"
    assert result.decision is not None
    assert result.decision.operation_ids == ["operation-unknown"]


def test_scope_violation_blocks_fixed_without_persistable_decision(
    tmp_path: Path,
) -> None:
    evidence = EvidenceRepository(tmp_path / "deepfix.db")
    _record(evidence, _change("change-1"))
    _record(evidence, _test("target-pass", TARGET_COMMAND, exit_code=0))

    result = OutcomeAdjudicator().decide(
        _input(
            evidence,
            successful_change_evidence_ids=["change-1"],
            scope_violation_evidence_ids=["scope-violation"],
        )
    )

    assert result.outcome == "review"
    assert result.decision is None
    assert result.blocking_evidence_ids == ["scope-violation"]


def test_not_reproduced_requires_passing_user_baseline_and_no_change(
    tmp_path: Path,
) -> None:
    evidence = EvidenceRepository(tmp_path / "deepfix.db")
    _record(
        evidence,
        _test(
            "baseline-pass",
            TARGET_COMMAND,
            exit_code=0,
            timing="baseline",
        ),
    )

    result = OutcomeAdjudicator().decide(
        _input(evidence, reproduction_state="not_reproduced")
    )

    assert result.outcome == "not_reproduced"
    assert result.decision is not None
    assert result.decision.evidence_ids == ["baseline-pass"]


def _input(
    evidence: EvidenceRepository,
    *,
    successful_change_evidence_ids: list[str] | None = None,
    scope_violation_evidence_ids: list[str] | None = None,
    reproduction_state: str = "reproduced",
    execution: ExecutionIntegrity | None = None,
) -> OutcomeAdjudicationInput:
    return OutcomeAdjudicationInput(
        definition=_definition(),
        lifecycle=TaskLifecycle(
            task_id=TASK_ID,
            status=TaskLifecycleStatus.RUNNING,
            version=2,
            updated_at="2026-08-31T01:00:00+00:00",
        ),
        verification_policy=_policy(),
        verification=evidence.verification_view(TASK_ID),
        execution_integrity=execution
        or ExecutionIntegrity(
            status_counts={"committed": 2},
            receipt_count=2,
            approval_count=1,
            incomplete_operation_ids=[],
            unknown_operation_ids=[],
        ),
        successful_change_evidence_ids=successful_change_evidence_ids or [],
        scope_violation_evidence_ids=scope_violation_evidence_ids or [],
        reproduction_state=reproduction_state,
    )


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task_id=TASK_ID,
        original_message_id="message-user-1",
        original_problem="fix value.py and run the target test",
        approval_mode="manual",
        source_project_root="C:/source",
        workspace_root="C:/workspace",
        workspace_baseline_id="baseline-1",
        project_python="C:/Python/python.exe",
        confinement_level="guarded_local",
        created_at="2026-08-31T00:00:00+00:00",
    )


def _policy() -> VerificationPolicy:
    return VerificationPolicy(
        policy_id="policy-1",
        task_id=TASK_ID,
        version=1,
        required_oracles=[
            VerificationOracle(
                oracle_id="oracle-target",
                origin="user_specified",
                command=TARGET_COMMAND,
                scope="targeted",
                role="required",
                required_timing="post_change",
                relevant_paths=["tests/test_value.py"],
            )
        ],
        supplemental_oracles=[],
        conflict_rules=[
            OracleConflictRule(
                rule_id="related-suite-failure",
                description="related repository suite failure blocks fixed",
                blocking_scopes=["module", "full_suite"],
            )
        ],
    )


def _change(evidence_id: str) -> FileChangeEvidence:
    return FileChangeEvidence(
        evidence_id=evidence_id,
        path="value.py",
        operation="edit",
        status="succeeded",
        tool_call_id=f"call-{evidence_id}",
        source_message_id=f"message-{evidence_id}",
    )


def _test(
    evidence_id: str,
    command: str,
    *,
    exit_code: int,
    origin: str = "user_specified",
    scope: str = "targeted",
    timing: str = "post_change",
) -> SystemTestEvidence:
    return SystemTestEvidence(
        evidence_id=evidence_id,
        command=command,
        exit_code=exit_code,
        summary="test result",
        tool_call_id=f"call-{evidence_id}",
        source_message_id=f"message-{evidence_id}",
        origin=origin,
        scope=scope,
        timing=timing,
        workspace_baseline_id="baseline-1",
        code_state_hash="code-state-1",
    )


def _record(
    repository: EvidenceRepository,
    evidence: FileChangeEvidence | SystemTestEvidence,
) -> None:
    repository.record_deterministic(
        TASK_ID,
        evidence,
        provenance_root_ids=[evidence.evidence_id],
    )
