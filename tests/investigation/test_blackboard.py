from __future__ import annotations

from dataclasses import replace

from deepfix.compaction.models import (
    CompactionSnapshot,
    DeterministicEvidenceBlock,
    ProvenancedClaim,
    ProvenanceRef,
    SystemTestEvidence,
    TaskAnchor,
)
from deepfix.domain_repositories.execution import ExecutionIntegrity
from deepfix.investigation.blackboard import CaseBlackboardBuilder
from deepfix.protected_context import ProtectedContext


def _test_evidence(exit_code: int) -> SystemTestEvidence:
    return SystemTestEvidence(
        evidence_id="evidence-1",
        command="python -m pytest tests/test_target.py -q",
        exit_code=exit_code,
        summary="authoritative test result",
        tool_call_id="call-1",
        source_message_id="message-1",
        origin="user_specified",
        scope="targeted",
        timing="baseline",
        workspace_baseline_id="baseline-1",
        code_state_hash="code-1",
        test_target_paths=["tests/test_target.py"],
    )


def _snapshot(duplicate_test: SystemTestEvidence) -> CompactionSnapshot:
    return CompactionSnapshot(
        task_id="task-1",
        version=1,
        created_at="2026-08-27T00:00:00+00:00",
        source_work_unit_ids=[],
        task_goal="Repair target",
        user_constraints=[],
        confirmed_facts=[],
        deterministic_evidence=DeterministicEvidenceBlock(tests=[duplicate_test]),
        active_hypotheses=[],
        rejected_hypotheses=[],
        confirmed_hypotheses=[],
        changed_files=[],
        experiments=[],
        test_results=[duplicate_test],
        conflicts=[],
        unresolved_questions=[],
        next_steps=[],
        artifact_references=[],
        content_hash="snapshot-hash",
    )


def _context(*, facts: tuple[ProvenancedClaim, ...] = ()) -> ProtectedContext:
    current_test = _test_evidence(exit_code=1)
    return ProtectedContext(
        task_anchor=TaskAnchor(
            task_id="task-1",
            task_goal="Repair target",
            latest_user_message_id="message-1",
            project_root="C:/workspace",
            project_python="C:/Python/python.exe",
            task_status="running",
        ),
        deterministic_evidence=DeterministicEvidenceBlock(tests=[current_test]),
        confirmed_facts=facts,
        hypotheses=(),
        unresolved_questions=(),
        execution_integrity=ExecutionIntegrity(
            receipt_count=0,
            approval_count=0,
        ),
        external_evidence=(),
        active_snapshot=_snapshot(_test_evidence(exit_code=0)),
    )


def test_blackboard_projects_same_evidence_once() -> None:
    builder = CaseBlackboardBuilder(lambda _task_id: _context())

    view = builder.build("task-1")

    assert [item.evidence_id for item in view.deterministic_evidence] == [
        "evidence-1"
    ]


def test_semantic_claim_cannot_replace_test_exit_code() -> None:
    fact = ProvenancedClaim(
        claim_id="claim-passed",
        text="pytest passed",
        sources=[
            ProvenanceRef(kind="system_evidence", ref_id="semantic-evidence-2")
        ],
    )
    builder = CaseBlackboardBuilder(lambda _task_id: _context(facts=(fact,)))

    view = builder.build("task-1")

    assert view.test_results[0].exit_code == 1
    assert view.confirmed_claims[0].text == "pytest passed"


def test_blackboard_rejects_context_from_another_task() -> None:
    context = _context()
    wrong_anchor = context.task_anchor.model_copy(update={"task_id": "other-task"})
    builder = CaseBlackboardBuilder(
        lambda _task_id: ProtectedContext(
            task_anchor=wrong_anchor,
            deterministic_evidence=context.deterministic_evidence,
            confirmed_facts=context.confirmed_facts,
            hypotheses=context.hypotheses,
            unresolved_questions=context.unresolved_questions,
            execution_integrity=context.execution_integrity,
            external_evidence=context.external_evidence,
            active_snapshot=context.active_snapshot,
        )
    )

    try:
        builder.build("task-1")
    except ValueError as exc:
        assert "task mismatch" in str(exc)
    else:
        raise AssertionError("cross-task context must be rejected")


def test_pytest_usage_error_does_not_mark_bug_reproduced() -> None:
    usage_error = _test_evidence(exit_code=4)
    context = _context()
    context = replace(
        context,
        deterministic_evidence=DeterministicEvidenceBlock(tests=[usage_error]),
        active_snapshot=None,
    )

    view = CaseBlackboardBuilder(lambda _task_id: context).build("task-1")

    assert view.reproduction_state == "unknown"
