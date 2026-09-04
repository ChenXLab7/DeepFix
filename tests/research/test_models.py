from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepfix.research.models import (
    DependencyContext,
    DependencyFinding,
    ExternalEvidence,
    LocalEvidenceReference,
    ResearchQuery,
    SearchCandidate,
)


def test_dependency_context_preserves_project_version_sources():
    finding = DependencyFinding(
        package_name="pydantic",
        declared_constraints=[">=2", "==2.8.4"],
        installed_version="2.8.4",
        python_executable="C:/project/.venv/Scripts/python.exe",
        source_files=["pyproject.toml", "uv.lock"],
        diagnostic=None,
    )

    context = DependencyContext(
        package=finding,
        official_repository="pydantic/pydantic",
        official_domains=["docs.pydantic.dev"],
    )

    assert context.package.installed_version == "2.8.4"
    assert context.official_repository == "pydantic/pydantic"
    assert context.official_domains == ["docs.pydantic.dev"]


def test_search_candidate_accepts_approved_source_and_level():
    candidate = SearchCandidate(
        candidate_id="candidate-1",
        task_id="task-1",
        source_type="github_issue",
        evidence_level="E3",
        title="Unexpected validation behavior",
        url="https://github.com/pydantic/pydantic/issues/1",
        query="pydantic 2 validation behavior",
        repository="pydantic/pydantic",
        created_at="2026-08-22T00:00:00+00:00",
    )

    assert candidate.evidence_level == "E3"
    assert candidate.repository == "pydantic/pydantic"


@pytest.mark.parametrize(
    ("field", "value"),
    [("source_type", "blog"), ("evidence_level", "E4")],
)
def test_search_candidate_rejects_unapproved_classification(field, value):
    payload = {
        "candidate_id": "candidate-1",
        "task_id": "task-1",
        "source_type": "official_docs",
        "evidence_level": "E1",
        "title": "Pydantic models",
        "url": "https://docs.pydantic.dev/latest/concepts/models/",
        "query": "pydantic model_copy",
        "repository": None,
        "created_at": "2026-08-22T00:00:00+00:00",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        SearchCandidate.model_validate(payload)


def test_external_evidence_converts_local_evidence_and_tracks_verification():
    external = ExternalEvidence(
        evidence_id="evidence-1",
        task_id="task-1",
        candidate_id="candidate-1",
        source_type="official_docs",
        evidence_level="E1",
        title="Pydantic models",
        url="https://docs.pydantic.dev/latest/concepts/models/",
        query="pydantic model_copy",
        relevant_excerpt="model_copy accepts an update mapping",
        retrieved_at="2026-08-22T00:01:00+00:00",
        dependency_name="pydantic",
        documented_version="2.8",
        project_version="2.8.4",
        local_verification="verified",
        local_evidence=[
            {"source": "tests/test_models.py:10", "observation": "pytest passed"}
        ],
        linked_test_tool_call_ids=["call-1"],
        verification_explanation="The local regression test passes.",
        artifact_path="/.deepfix-artifacts/research/task-1/evidence-1.md",
    )

    assert external.local_evidence == [
        LocalEvidenceReference(source="tests/test_models.py:10", observation="pytest passed")
    ]
    assert external.local_verification == "verified"


def test_external_evidence_rejects_unknown_verification_status():
    with pytest.raises(ValidationError):
        ExternalEvidence(
            evidence_id="evidence-1",
            task_id="task-1",
            candidate_id="candidate-1",
            source_type="official_docs",
            evidence_level="E1",
            title="Pydantic models",
            url="https://docs.pydantic.dev/latest/concepts/models/",
            query="pydantic model_copy",
            relevant_excerpt="excerpt",
            retrieved_at="2026-08-22T00:01:00+00:00",
            dependency_name="pydantic",
            documented_version="2.8",
            project_version="2.8.4",
            local_verification="trusted",
            local_evidence=[],
            linked_test_tool_call_ids=[],
            verification_explanation=None,
            artifact_path="/.deepfix-artifacts/research/task-1/evidence-1.md",
        )


def test_research_query_records_provider_outcomes():
    query = ResearchQuery(
        query_id="query-1",
        task_id="task-1",
        sanitized_query="pydantic 2 model_copy",
        providers=["pypi", "github"],
        provider_errors=["github: rate limited"],
        created_at="2026-08-22T00:00:00+00:00",
    )

    assert query.providers == ["pypi", "github"]
    assert query.provider_errors == ["github: rate limited"]
