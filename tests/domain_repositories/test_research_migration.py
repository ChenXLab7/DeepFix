from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from deepfix.compaction.models import ArtifactReference
from deepfix.domain_repositories.evidence import (
    EvidenceAuthority,
    EvidenceRepository,
    EvidenceVerification,
)
from deepfix.research.models import (
    ExternalEvidence,
    LocalEvidenceReference,
    SearchCandidate,
)


def _candidate(*, candidate_id: str = "candidate-1") -> SearchCandidate:
    return SearchCandidate(
        candidate_id=candidate_id,
        task_id="task-1",
        source_type="official_docs",
        evidence_level="E1",
        title="Official pytest timeout documentation",
        url="https://example.test/docs/timeout",
        query="pytest timeout",
        repository=None,
        created_at="2026-08-29T00:00:00+00:00",
    )


def _external(
    *,
    evidence_id: str | None = None,
    candidate_id: str = "candidate-1",
    verification: str = "unverified",
) -> ExternalEvidence:
    return ExternalEvidence(
        evidence_id=evidence_id or uuid4().hex,
        task_id="task-1",
        candidate_id=candidate_id,
        source_type="official_docs",
        evidence_level="E1",
        title="Official pytest timeout documentation",
        url="https://example.test/docs/timeout",
        query="pytest timeout",
        relevant_excerpt="The timeout option terminates hanging tests.",
        retrieved_at="2026-08-29T00:00:01+00:00",
        dependency_name="pytest-timeout",
        documented_version="2.4.0",
        project_version="2.4.0",
        local_verification=verification,
        local_evidence=[],
        linked_test_tool_call_ids=[],
        verification_explanation=None,
        artifact_path="/.deepfix-artifacts/research/task-1/evidence.md",
    )


def _reference(content: str) -> ArtifactReference:
    return ArtifactReference(
        path="/.deepfix-artifacts/research/task-1/evidence.md",
        kind="research",
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        work_unit_ids=[],
    )


def test_research_attempt_audit_redacts_provider_secrets_and_tracks_candidates(
    tmp_path: Path,
):
    repository = EvidenceRepository(tmp_path / "deepfix.db")
    attempt = repository.record_research_attempt(
        task_id="task-1",
        query_id="query-1",
        sanitized_query="pytest timeout",
        providers=["github"],
        provider_errors=["Authorization: Bearer secret-token"],
    )
    repository.record_research_candidates("query-1", [_candidate()])

    restored = repository.get_research_attempt("task-1", "query-1")

    assert attempt.query_id == "query-1"
    assert restored.provider_errors == ["Authorization: Bearer [REDACTED]"]
    assert restored.candidate_ids == ["candidate-1"]
    assert repository.get_research_candidate("task-1", "candidate-1") == _candidate()
    assert repository.research_summary("task-1") == (
        1,
        ["Authorization: Bearer [REDACTED]"],
    )


def test_research_summary_does_not_count_candidate_link_placeholders(tmp_path: Path):
    repository = EvidenceRepository(tmp_path / "deepfix.db")
    repository.record_research_attempt(
        task_id="task-1",
        query_id="placeholder-1",
        sanitized_query="pytest timeout",
        providers=[],
        provider_errors=[],
    )
    repository.record_research_candidates("placeholder-1", [_candidate()])

    assert repository.research_summary("task-1") == (0, [])


def test_external_evidence_update_preserves_immutable_revision_history(tmp_path: Path):
    content = "verified artifact body"
    reference = _reference(content)
    repository = EvidenceRepository(
        tmp_path / "deepfix.db", artifact_verifier=lambda item: item == reference
    )
    original = _external()
    first = repository.accept_external(
        original,
        provenance_root_ids=["url:https://example.test/docs/timeout"],
        artifact_references=[reference],
    )
    verified = original.model_copy(
        update={
            "local_verification": "verified",
            "local_evidence": [
                LocalEvidenceReference(
                    source="tests/test_timeout.py", observation="1 passed"
                )
            ],
            "linked_test_tool_call_ids": ["pytest-1"],
            "verification_explanation": "Local test passed",
        }
    )

    current = repository.update_external(verified)
    revisions = repository.list_revisions("task-1", original.evidence_id)

    assert first.authority is EvidenceAuthority.RESEARCH
    assert current.verification_state is EvidenceVerification.VERIFIED
    assert [item.verification_state for item in revisions] == [
        EvidenceVerification.UNVERIFIED,
        EvidenceVerification.VERIFIED,
    ]
    assert revisions[0].content_hash == first.content_hash
    assert revisions[1].content_hash == current.content_hash
