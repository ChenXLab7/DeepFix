from __future__ import annotations

import inspect

import pytest

from deepfix.compaction.models import SystemTestEvidence
from deepfix.domain_repositories.evidence import EvidenceRepository
from deepfix.domain_repositories.investigation import (
    HypothesisIdentityConflict,
    HypothesisTransitionError,
    InvestigationEvidenceMissing,
    InvestigationRepository,
)
from deepfix.investigation.models import InvestigationHypothesis, UnresolvedQuestion


def _hypothesis(
    hypothesis_id: str,
    *,
    statement: str = "parser branch mishandles empty input",
    state: str = "candidate",
    evidence_ids: list[str] | None = None,
    reopens_hypothesis_id: str | None = None,
) -> InvestigationHypothesis:
    return InvestigationHypothesis(
        hypothesis_id=hypothesis_id,
        statement=statement,
        state=state,
        evidence_ids=evidence_ids or [],
        checked_locations=[],
        reason=f"{state} reason",
        reopens_hypothesis_id=reopens_hypothesis_id,
    )


def _question(question_id: str = "q-1") -> UnresolvedQuestion:
    return UnresolvedQuestion(
        question_id=question_id,
        task_id="task-1",
        text="Why does the parser fail only on Windows?",
        status="open",
        source_ids=["message-1"],
        resolution_evidence_ids=[],
        created_at="2026-08-29T00:00:00+00:00",
    )


def _record_test_evidence(database, evidence_id: str = "e-1") -> None:
    EvidenceRepository(database).record_deterministic(
        "task-1",
        SystemTestEvidence(
            evidence_id=evidence_id,
            command="python -m pytest -q",
            exit_code=1,
            summary="1 failed",
            tool_call_id="call-1",
            source_message_id="message-2",
        ),
        provenance_root_ids=["call-1"],
    )


def test_hypothesis_transition_requires_stable_id_and_current_evidence(tmp_path):
    database = tmp_path / "deepfix.db"
    _record_test_evidence(database)
    repository = InvestigationRepository(database)
    candidate = _hypothesis("h-1")

    repository.record_hypothesis("task-1", candidate)
    supported = candidate.model_copy(update={"state": "supported", "evidence_ids": ["e-1"]})

    assert repository.record_hypothesis("task-1", supported) == supported
    with pytest.raises(HypothesisIdentityConflict):
        repository.record_hypothesis(
            "task-1", _hypothesis("h-1", statement="different statement")
        )


def test_hypothesis_transition_rejects_missing_or_invalid_evidence(tmp_path):
    repository = InvestigationRepository(tmp_path / "deepfix.db")
    candidate = _hypothesis("h-1")
    repository.record_hypothesis("task-1", candidate)

    with pytest.raises(InvestigationEvidenceMissing):
        repository.record_hypothesis(
            "task-1",
            candidate.model_copy(update={"state": "supported", "evidence_ids": ["missing"]}),
        )


def test_rejected_hypothesis_reopens_only_under_new_identity(tmp_path):
    database = tmp_path / "deepfix.db"
    _record_test_evidence(database)
    repository = InvestigationRepository(database)
    candidate = _hypothesis("h-1")
    repository.record_hypothesis("task-1", candidate)
    rejected = candidate.model_copy(update={"state": "rejected", "evidence_ids": ["e-1"]})
    repository.record_hypothesis("task-1", rejected)

    with pytest.raises(HypothesisTransitionError):
        repository.record_hypothesis(
            "task-1", rejected.model_copy(update={"state": "candidate"})
        )
    with pytest.raises(HypothesisTransitionError, match="reopens_hypothesis_id"):
        repository.record_hypothesis(
            "task-1", _hypothesis("h-2", statement=rejected.statement)
        )

    reopened = _hypothesis(
        "h-2",
        statement=rejected.statement,
        evidence_ids=["e-1"],
        reopens_hypothesis_id="h-1",
    )
    assert repository.record_hypothesis("task-1", reopened) == reopened


def test_question_is_not_todo_or_task_lifecycle_state(tmp_path):
    repository = InvestigationRepository(tmp_path / "deepfix.db")

    question = repository.open_question(_question())

    assert question.status == "open"
    assert not hasattr(question, "todo_status")
    assert not hasattr(repository, "transition_task")
    assert not hasattr(repository, "can_execute")
    assert "owner" not in inspect.signature(repository.open_question).parameters


def test_question_resolution_requires_current_evidence_id(tmp_path):
    database = tmp_path / "deepfix.db"
    repository = InvestigationRepository(database)
    repository.open_question(_question())

    with pytest.raises(InvestigationEvidenceMissing):
        repository.resolve_question("task-1", "q-1", evidence_ids=["missing"])

    _record_test_evidence(database)
    resolved = repository.resolve_question("task-1", "q-1", evidence_ids=["e-1"])

    assert resolved.status == "resolved"
    assert resolved.resolution_evidence_ids == ["e-1"]
    assert resolved.resolved_at is not None
