from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from deepfix.config import ApprovalMode
from deepfix.models import Evidence, TaskState
from deepfix.persistence import TaskRepository
from deepfix.research.models import ExternalEvidence, SearchCandidate
from deepfix.research.store import ResearchEvidenceStore


def _candidate(
    *,
    title: str = "Pydantic 模型复制",
    url: str = "https://docs.pydantic.dev/latest/concepts/models/",
) -> SearchCandidate:
    return SearchCandidate(
        candidate_id="draft",
        task_id="draft",
        source_type="official_docs",
        evidence_level="E1",
        title=title,
        url=url,
        query="draft query",
        repository="pydantic/pydantic",
        created_at="draft timestamp",
    )


def _evidence(
    *,
    task_id: str,
    candidate_id: str,
    evidence_id: str | None = None,
) -> ExternalEvidence:
    return ExternalEvidence(
        evidence_id=evidence_id or uuid4().hex,
        task_id=task_id,
        candidate_id=candidate_id,
        source_type="official_docs",
        evidence_level="E1",
        title="Pydantic 模型复制",
        url="https://docs.pydantic.dev/latest/concepts/models/",
        query="pydantic 2 model_copy",
        relevant_excerpt="model_copy 接受 update 参数。",
        retrieved_at="2026-08-22T00:01:00+00:00",
        dependency_name="pydantic",
        documented_version="2.8",
        project_version="2.8.4",
        local_verification="unverified",
        local_evidence=[],
        linked_test_tool_call_ids=[],
        verification_explanation=None,
        artifact_path=(
            f"/.deepfix-artifacts/research/{task_id}/evidence.md"
        ),
    )


def test_initialization_preserves_existing_task_rows(tmp_path):
    database = tmp_path / "deepfix.sqlite3"
    tasks = TaskRepository(database)
    task = TaskState.create(tmp_path, "修复验证错误", ApprovalMode.MANUAL)
    tasks.save(task)

    ResearchEvidenceStore(database)

    assert tasks.get(task.task_id) == task


def test_query_summary_counts_queries_and_collects_provider_errors(tmp_path):
    store = ResearchEvidenceStore(tmp_path / "deepfix.sqlite3")

    first = store.save_query(
        "task-a",
        "pydantic 2 model_copy",
        ["pypi", "github"],
        ["github: 请求受限"],
    )
    store.save_query(
        "task-a",
        "pydantic 2 validation",
        ["pypi", "github"],
        ["github: 请求受限", "pypi: 暂时超时"],
    )
    store.save_query("task-b", "other", ["pypi"], ["other error"])

    count, errors = store.query_summary("task-a")

    UUID(first.query_id)
    assert first.task_id == "task-a"
    assert first.sanitized_query == "pydantic 2 model_copy"
    assert first.providers == ["pypi", "github"]
    assert count == 2
    assert errors == ["github: 请求受限", "pypi: 暂时超时"]


def test_save_candidates_assigns_random_task_local_ids(tmp_path):
    store = ResearchEvidenceStore(tmp_path / "deepfix.sqlite3")
    source_url = "https://docs.pydantic.dev/latest/concepts/models/"

    saved = store.save_candidates(
        "task-a",
        "pydantic 2 model_copy",
        [_candidate(url=source_url), _candidate(title="第二条", url=source_url)],
    )

    assert len(saved) == 2
    assert saved[0].candidate_id != saved[1].candidate_id
    assert all(UUID(item.candidate_id) for item in saved)
    assert all(item.task_id == "task-a" for item in saved)
    assert all(item.query == "pydantic 2 model_copy" for item in saved)
    assert all(item.created_at != "draft timestamp" for item in saved)
    assert all(item.candidate_id not in source_url for item in saved)


def test_candidate_lookup_isolated_by_task(tmp_path):
    store = ResearchEvidenceStore(tmp_path / "deepfix.sqlite3")
    candidate = store.save_candidates("task-a", "query", [_candidate()])[0]

    assert store.get_candidate("task-a", candidate.candidate_id) == candidate
    with pytest.raises(KeyError):
        store.get_candidate("task-b", candidate.candidate_id)


def test_evidence_round_trip_preserves_unicode_and_nested_evidence(tmp_path):
    store = ResearchEvidenceStore(tmp_path / "deepfix.sqlite3")
    candidate = store.save_candidates("task-a", "query", [_candidate()])[0]
    external = _evidence(task_id="task-a", candidate_id=candidate.candidate_id)
    external.local_evidence = [
        Evidence("tests/test_模型.py:8", "验证结果：通过")
    ]

    store.save_evidence(external)
    restored = store.get_evidence("task-a", external.evidence_id)

    assert restored == external
    assert restored.local_evidence == [
        Evidence("tests/test_模型.py:8", "验证结果：通过")
    ]


def test_evidence_requires_uuid_id(tmp_path):
    store = ResearchEvidenceStore(tmp_path / "deepfix.sqlite3")
    external = _evidence(
        task_id="task-a",
        candidate_id=uuid4().hex,
        evidence_id="https://docs.pydantic.dev/evidence",
    )

    with pytest.raises(ValueError, match="evidence_id"):
        store.save_evidence(external)


def test_evidence_reads_and_lists_are_isolated_by_task(tmp_path):
    store = ResearchEvidenceStore(tmp_path / "deepfix.sqlite3")
    candidate_a = store.save_candidates("task-a", "query", [_candidate()])[0]
    candidate_b = store.save_candidates("task-b", "query", [_candidate()])[0]
    evidence_a = _evidence(task_id="task-a", candidate_id=candidate_a.candidate_id)
    evidence_b = _evidence(task_id="task-b", candidate_id=candidate_b.candidate_id)
    store.save_evidence(evidence_a)
    store.save_evidence(evidence_b)

    assert store.list_evidence("task-a") == [evidence_a]
    assert store.list_evidence("task-b") == [evidence_b]
    with pytest.raises(KeyError):
        store.get_evidence("task-b", evidence_a.evidence_id)


def test_update_verification_changes_only_current_task_record(tmp_path):
    store = ResearchEvidenceStore(tmp_path / "deepfix.sqlite3")
    shared_id = uuid4().hex
    candidate_a = store.save_candidates("task-a", "query", [_candidate()])[0]
    candidate_b = store.save_candidates("task-b", "query", [_candidate()])[0]
    evidence_a = _evidence(
        task_id="task-a",
        candidate_id=candidate_a.candidate_id,
        evidence_id=shared_id,
    )
    evidence_b = _evidence(
        task_id="task-b",
        candidate_id=candidate_b.candidate_id,
        evidence_id=shared_id,
    )
    store.save_evidence(evidence_a)
    store.save_evidence(evidence_b)

    updated = store.update_verification(
        "task-a",
        shared_id,
        local_verification="verified",
        local_evidence=[Evidence("tests/test_fix.py:12", "回归测试通过")],
        linked_test_tool_call_ids=["call-1"],
        verification_explanation="本地测试验证了官方结论。",
    )

    assert updated.local_verification == "verified"
    assert updated.linked_test_tool_call_ids == ["call-1"]
    assert store.get_evidence("task-b", shared_id).local_verification == "unverified"


def test_update_verification_rejects_cross_task_evidence(tmp_path):
    store = ResearchEvidenceStore(tmp_path / "deepfix.sqlite3")
    candidate = store.save_candidates("task-a", "query", [_candidate()])[0]
    external = _evidence(task_id="task-a", candidate_id=candidate.candidate_id)
    store.save_evidence(external)

    with pytest.raises(KeyError):
        store.update_verification(
            "task-b",
            external.evidence_id,
            local_verification="contradicted",
            local_evidence=[Evidence("src/model.py:3", "源码行为相反")],
            linked_test_tool_call_ids=[],
            verification_explanation="本地源码与外部资料冲突。",
        )


def test_update_verification_revalidates_status_before_persisting(tmp_path):
    store = ResearchEvidenceStore(tmp_path / "deepfix.sqlite3")
    candidate = store.save_candidates("task-a", "query", [_candidate()])[0]
    external = _evidence(task_id="task-a", candidate_id=candidate.candidate_id)
    store.save_evidence(external)

    with pytest.raises(ValidationError):
        store.update_verification(
            "task-a",
            external.evidence_id,
            local_verification="trusted",  # type: ignore[arg-type]
            local_evidence=[],
            linked_test_tool_call_ids=[],
            verification_explanation=None,
        )

    assert store.get_evidence("task-a", external.evidence_id) == external
