from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from deepfix.compaction.models import (
    ApprovalEvidence,
    ArtifactReference,
    FileChangeEvidence,
    ProvenancedClaim,
    ProvenanceRef,
    ResearchStatusEvidence,
    SystemTestEvidence,
)
from deepfix.domain_repositories.evidence import (
    ArtifactIntegrityError,
    EvidenceAuthority,
    EvidenceIdentityConflict,
    EvidenceKind,
    EvidenceRepository,
    EvidenceVerification,
)
from deepfix.models import Evidence
from deepfix.research.models import ExternalEvidence


def _test_evidence(*, exit_code: int = 0) -> SystemTestEvidence:
    return SystemTestEvidence(
        evidence_id="test-evidence-1",
        command="python -m pytest tests/test_value.py -q",
        exit_code=exit_code,
        summary="1 passed" if exit_code == 0 else "1 failed",
        tool_call_id="tool-call-1",
        source_message_id="tool-message-1",
        origin="user_specified",
        scope="targeted",
        timing="post_change",
        workspace_baseline_id="baseline-1",
        code_state_hash="code-state-1",
        test_target_paths=["tests/test_value.py"],
        test_content_hashes={"tests/test_value.py": "a" * 64},
    )


def _claim() -> ProvenancedClaim:
    return ProvenancedClaim(
        claim_id="claim-1",
        text="parser.py contains the semantic root cause",
        sources=[ProvenanceRef(kind="work_unit", ref_id="work-unit-1")],
    )


def _artifact_reference() -> ArtifactReference:
    return ArtifactReference(
        path="research/evidence-1.json",
        kind="research",
        content_hash="b" * 64,
        work_unit_ids=["work-unit-1"],
    )


def test_deterministic_test_is_persisted_as_immutable_system_evidence(tmp_path: Path):
    repository = EvidenceRepository(tmp_path / "deepfix.db")

    stored = repository.record_deterministic(
        "task-1",
        _test_evidence(),
        provenance_root_ids=["tool-call-1"],
    )

    assert stored.kind is EvidenceKind.TEST
    assert stored.authority is EvidenceAuthority.SYSTEM
    assert stored.verification_state is EvidenceVerification.VERIFIED
    assert stored.provenance_root_ids == ["tool-call-1"]
    assert len(stored.content_hash) == 64
    assert repository.get("task-1", stored.evidence_id) == stored
    with pytest.raises(ValidationError):
        stored.authority = EvidenceAuthority.MODEL_SEMANTIC


def test_semantic_candidate_cannot_be_elevated_to_system_authority(tmp_path: Path):
    repository = EvidenceRepository(tmp_path / "deepfix.db")

    stored = repository.record_semantic_candidate(
        "task-1",
        _claim(),
        provenance_root_ids=["receipt-1"],
    )

    assert stored.kind is EvidenceKind.SEMANTIC_CLAIM
    assert stored.authority is EvidenceAuthority.MODEL_SEMANTIC
    assert stored.verification_state is EvidenceVerification.UNVERIFIED


def test_approval_is_system_authoritative_even_when_decision_is_reject(tmp_path: Path):
    repository = EvidenceRepository(tmp_path / "deepfix.db")

    stored = repository.record_deterministic(
        "task-1",
        ApprovalEvidence(
            evidence_id="approval-1",
            operation="execute: pytest -q",
            decision="reject",
            risk="L2",
        ),
        provenance_root_ids=["interrupt-1"],
    )

    assert stored.kind is EvidenceKind.APPROVAL
    assert stored.authority is EvidenceAuthority.SYSTEM
    assert stored.verification_state is EvidenceVerification.VERIFIED


def test_external_evidence_is_research_authoritative(tmp_path: Path):
    reference = _artifact_reference()
    repository = EvidenceRepository(
        tmp_path / "deepfix.db",
        artifact_verifier=lambda candidate: candidate == reference,
    )
    evidence = ExternalEvidence(
        evidence_id="external-1",
        task_id="task-1",
        candidate_id="candidate-1",
        source_type="official_docs",
        evidence_level="E1",
        title="Official timeout documentation",
        url="https://example.test/docs",
        query="pytest timeout",
        relevant_excerpt="The timeout option stops a hanging test.",
        retrieved_at="2026-08-29T00:00:00+00:00",
        dependency_name="pytest-timeout",
        documented_version="2.4",
        project_version="2.4",
        local_verification="verified",
        local_evidence=[Evidence(source="pytest", observation="timeout reproduced")],
        linked_test_tool_call_ids=["tool-call-1"],
        verification_explanation="Observed locally",
        artifact_path=reference.path,
    )

    stored = repository.accept_external(
        evidence,
        provenance_root_ids=["url:https://example.test/docs"],
        artifact_references=[reference],
    )

    assert stored.kind is EvidenceKind.EXTERNAL_RESEARCH
    assert stored.authority is EvidenceAuthority.RESEARCH
    assert stored.verification_state is EvidenceVerification.VERIFIED


def test_same_identity_replays_but_changed_content_conflicts(tmp_path: Path):
    repository = EvidenceRepository(tmp_path / "deepfix.db")
    first = repository.record_deterministic(
        "task-1",
        _test_evidence(exit_code=0),
        provenance_root_ids=["tool-call-1"],
    )

    replay = repository.record_deterministic(
        "task-1",
        _test_evidence(exit_code=0),
        provenance_root_ids=["tool-call-1"],
    )

    assert replay == first
    with pytest.raises(EvidenceIdentityConflict):
        repository.record_deterministic(
            "task-1",
            _test_evidence(exit_code=1),
            provenance_root_ids=["tool-call-1"],
        )


def test_repository_requires_independent_provenance_roots(tmp_path: Path):
    repository = EvidenceRepository(tmp_path / "deepfix.db")

    with pytest.raises(ValueError, match="provenance"):
        repository.record_deterministic(
            "task-1",
            FileChangeEvidence(
                evidence_id="file-evidence-1",
                path="src/value.py",
                operation="edit",
                status="succeeded",
                tool_call_id="tool-call-2",
                source_message_id="tool-message-2",
            ),
            provenance_root_ids=[],
        )


def test_repository_rejects_artifact_reference_that_cannot_be_verified(tmp_path: Path):
    repository = EvidenceRepository(
        tmp_path / "deepfix.db",
        artifact_verifier=lambda _reference: False,
    )

    with pytest.raises(ArtifactIntegrityError, match="research/evidence-1.json"):
        repository.record_deterministic(
            "task-1",
            ResearchStatusEvidence(
                evidence_id="research-status-1",
                verification="verified",
                artifact_path="research/evidence-1.json",
            ),
            provenance_root_ids=["research-attempt-1"],
            artifact_references=[_artifact_reference()],
        )


def test_verified_artifact_and_provenance_are_returned_by_task_view(tmp_path: Path):
    reference = _artifact_reference()
    repository = EvidenceRepository(
        tmp_path / "deepfix.db",
        artifact_verifier=lambda candidate: candidate == reference,
    )
    stored = repository.record_deterministic(
        "task-1",
        ResearchStatusEvidence(
            evidence_id="research-status-1",
            verification="contradicted",
            artifact_path=reference.path,
        ),
        provenance_root_ids=["research-attempt-1", "research-attempt-1"],
        artifact_references=[reference],
    )

    items = repository.list_for_task("task-1")
    view = repository.verification_view("task-1")

    assert items == [stored]
    assert stored.provenance_root_ids == ["research-attempt-1"]
    assert stored.verification_state is EvidenceVerification.CONTRADICTED
    assert stored.artifact_references == [reference]
    assert view.evidence_ids == [stored.evidence_id]
    assert view.verified_evidence_ids == []
    assert view.contradicted_evidence_ids == [stored.evidence_id]
