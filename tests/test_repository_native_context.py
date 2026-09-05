from deepfix.compaction.models import (
    ApprovalEvidence,
    CompactionSnapshot,
    DeterministicEvidenceBlock,
    HypothesisRecord,
    ProvenancedClaim,
    ProvenancedText,
    SnapshotCoverage,
    SystemTestEvidence,
    TaskAnchor,
    UserConstraint,
)
from deepfix.domain_repositories.execution import ExecutionIntegrity
from deepfix.protected_context import ProtectedContext, render_protected_context


def test_current_domain_projection_wins_and_identifiers_are_displayed_once():
    constraint = UserConstraint(
        constraint_id="constraint-1", text="不要修改测试", source_user_message_id="user-1"
    )
    claim = ProvenancedClaim(claim_id="claim-1", text="失败可复现")
    hypothesis = HypothesisRecord(
        hypothesis_id="hyp-1",
        text="current rejected hypothesis",
        state="rejected",
        reason="test evidence disproved it",
        updated_in_version=1,
    )
    evidence = DeterministicEvidenceBlock(
        tests=[
            SystemTestEvidence(
                evidence_id="evidence-1",
                command="python -m pytest -q",
                exit_code=1,
                summary="1 failed",
                tool_call_id="call-1",
                source_message_id="tool-message-1",
            )
        ],
        approvals=[
            ApprovalEvidence(
                evidence_id="approval-1",
                operation="execute pytest",
                decision="approve",
                risk="L1",
            )
        ],
    )
    anchor = TaskAnchor(
        task_id="task-a",
        task_goal="修复错误",
        user_constraints=[constraint],
        latest_user_message_id="user-1",
        project_root="C:/workspace",
        project_python="C:/Python/python.exe",
        task_status="running",
    )
    snapshot = CompactionSnapshot(
        task_id="task-a",
        version=1,
        lifecycle="active",
        created_at="2026-08-31T00:00:00+00:00",
        activated_at="2026-08-31T00:00:01+00:00",
        source_work_unit_ids=["wu-1"],
        coverage=SnapshotCoverage(),
        task_goal=anchor.task_goal,
        user_constraints=[constraint],
        confirmed_facts=[claim],
        deterministic_evidence=evidence,
        active_hypotheses=[
            hypothesis.model_copy(update={"text": "stale hypothesis", "state": "active"})
        ],
        rejected_hypotheses=[],
        confirmed_hypotheses=[],
        changed_files=[],
        experiments=[],
        test_results=evidence.tests,
        conflicts=[],
        unresolved_questions=[],
        next_steps=[],
        artifact_references=[],
        content_hash="a" * 64,
    )
    context = ProtectedContext(
        task_anchor=anchor,
        deterministic_evidence=evidence,
        confirmed_facts=(claim,),
        hypotheses=(hypothesis,),
        unresolved_questions=(ProvenancedText(text="为什么只在 Windows 失败"),),
        execution_integrity=ExecutionIntegrity(
            status_counts={"committed": 1}, receipt_count=1, approval_count=1
        ),
        external_evidence=(),
        active_snapshot=snapshot,
    )

    rendered = render_protected_context(context)

    assert rendered.count("<deepfix_protected_context>") == 1
    assert rendered.count('constraint_id="constraint-1"') == 1
    assert rendered.count('evidence_id="evidence-1"') == 1
    assert rendered.count('evidence_id="approval-1"') == 1
    assert rendered.count('hypothesis_id="hyp-1"') == 1
    assert rendered.count('claim_id="claim-1"') == 1
    assert 'state="rejected"' in rendered
    assert "stale hypothesis" not in rendered
    assert "为什么只在 Windows 失败" in rendered
    assert "<receipt_count>1</receipt_count>" in rendered
