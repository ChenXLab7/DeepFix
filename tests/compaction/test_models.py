import pytest
from pydantic import ValidationError

from deepfix.compaction.models import (
    CompactionDelta,
    CompactionSnapshot,
    DeterministicEvidenceBlock,
)


def _delta_payload():
    return {
        "user_constraint_candidates": [],
        "confirmed_fact_candidates": [],
        "hypothesis_transitions": [],
        "experiments": [],
        "conflict_candidates": [],
        "unresolved_questions": [],
        "next_steps": [],
    }


def _snapshot_payload():
    return {
        "task_id": "task-a",
        "version": 1,
        "previous_version": None,
        "lifecycle": "prepared",
        "created_at": "2026-08-22T00:00:00+00:00",
        "activated_at": None,
        "abandoned_at": None,
        "abandon_reason": None,
        "source_work_unit_ids": [],
        "task_goal": "修复符号错误",
        "user_constraints": [],
        "confirmed_facts": [],
        "deterministic_evidence": DeterministicEvidenceBlock(),
        "active_hypotheses": [],
        "rejected_hypotheses": [],
        "confirmed_hypotheses": [],
        "changed_files": [],
        "experiments": [],
        "test_results": [],
        "conflicts": [],
        "unresolved_questions": [],
        "next_steps": [],
        "artifact_references": [],
        "content_hash": "a" * 64,
    }


def test_delta_rejects_model_supplied_claim_id():
    payload = _delta_payload()
    payload["confirmed_fact_candidates"] = [
        {
            "text": "pytest exits 1",
            "sources": [{"kind": "work_unit", "ref_id": "wu-1"}],
            "claim_id": "model-chosen",
        }
    ]

    with pytest.raises(ValidationError, match="claim_id"):
        CompactionDelta.model_validate(payload)


def test_snapshot_lifecycle_requires_matching_timestamp():
    prepared = CompactionSnapshot.model_validate(_snapshot_payload())
    assert prepared.lifecycle == "prepared"

    with pytest.raises(ValidationError, match="activated_at"):
        CompactionSnapshot.model_validate(
            {**_snapshot_payload(), "lifecycle": "active"}
        )

    with pytest.raises(ValidationError, match="abandoned_at"):
        CompactionSnapshot.model_validate(
            {**_snapshot_payload(), "lifecycle": "abandoned"}
        )


def test_reopened_hypothesis_requires_distinct_identity():
    payload = _snapshot_payload()
    payload["active_hypotheses"] = [
        {
            "hypothesis_id": "hyp-old",
            "text": "缓存过期",
            "state": "active",
            "reason": "出现新证据",
            "reopens_hypothesis_id": "hyp-old",
            "sources": [{"kind": "system_evidence", "ref_id": "ev-1"}],
            "updated_in_version": 1,
        }
    ]

    with pytest.raises(ValidationError, match="reopens_hypothesis_id"):
        CompactionSnapshot.model_validate(payload)

