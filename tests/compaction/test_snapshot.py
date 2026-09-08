from dataclasses import replace

import pytest

from deepfix.compaction.errors import SnapshotBuildError
from deepfix.compaction.identity import stable_claim_id
from deepfix.compaction.models import (
    ArtifactReference,
    CompactionDelta,
    ConflictCandidate,
    DeterministicEvidenceBlock,
    FactCandidate,
    HypothesisTransition,
    ProvenancedText,
    ProvenanceRef,
    SystemTestEvidence,
    TaskAnchor,
    UserConstraintCandidate,
    WorkUnit,
)
from deepfix.compaction.snapshot import (
    BuildSnapshotInput,
    CompactionDeltaGenerator,
    CompactionSnapshotBuilder,
)


def test_delta_input_contains_actual_command_and_result():
    from langchain_core.messages import AIMessage, ToolMessage

    from deepfix.compaction.snapshot import _delta_messages
    from deepfix.compaction.work_units import partition_work_units

    messages = [AIMessage(id="a", content="", tool_calls=[{
        "id": "c", "name": "execute", "args": {"command": "pytest UNIQUE_CASE"},
        "type": "tool_call"}]), ToolMessage(id="t", tool_call_id="c", content="UNIQUE_RESULT")]
    units = partition_work_units(messages, set()).units
    payload = _delta_messages(units, messages)[1].content
    assert "UNIQUE_CASE" in payload
    assert "UNIQUE_RESULT" in payload


def test_no_summary_model_call_when_nothing_can_be_compacted():
    assert CompactionDeltaGenerator().generate(object(), []).confirmed_fact_candidates == []


def _delta(**updates):
    values = {
        "user_constraint_candidates": [],
        "confirmed_fact_candidates": [],
        "hypothesis_transitions": [],
        "experiments": [],
        "conflict_candidates": [],
        "unresolved_questions": [],
        "next_steps": [],
    }
    values.update(updates)
    return CompactionDelta(**values)


def _test_evidence(exit_code=1):
    return SystemTestEvidence(
        evidence_id="test-e1",
        command="pytest -q",
        exit_code=exit_code,
        summary="1 failed" if exit_code else "1 passed",
        tool_call_id="pytest-1",
        source_message_id="tool-result-1",
    )


def _unit():
    return WorkUnit(
        unit_id="wu-1",
        purpose="investigate",
        message_ids=["user-latest", "assistant-1", "tool-result-1"],
        tool_call_ids=["pytest-1"],
        state="complete",
        categories={"verify_fail"},
        start_index=0,
        end_index=2,
    )


def _build_input(**updates):
    unit = _unit()
    values = {
        "task_id": "task-a",
        "previous_snapshot": None,
        "compressed_units": (unit,),
        "current_facts": (),
        "current_hypotheses": (),
        "current_unresolved_questions": (),
        "task_anchor": TaskAnchor(
            task_id="task-a",
            task_goal="修复符号错误",
            user_constraints=[],
            latest_user_message_id="user-latest",
            project_root="C:/project",
            project_python="C:/python.exe",
            task_status="investigating",
        ),
        "deterministic_evidence": DeterministicEvidenceBlock(
            tests=[_test_evidence()]
        ),
        "delta": _delta(),
        "artifact_reference": ArtifactReference(
            path="/.deepfix-artifacts/conversation_history/task-a.md",
            kind="conversation_history",
            content_hash="a" * 64,
            work_unit_ids=[unit.unit_id],
        ),
        "input_hash": "b" * 64,
    }
    values.update(updates)
    return BuildSnapshotInput(**values)


def _activate(snapshot):
    return snapshot.model_copy(
        update={
            "lifecycle": "active",
            "activated_at": "2026-08-22T00:00:00+00:00",
        }
    )


def test_builder_ignores_model_attempt_to_replace_system_test():
    build_input = _build_input(
        delta=_delta(
            confirmed_fact_candidates=[
                FactCandidate(
                    text="pytest passed",
                    sources=[ProvenanceRef(kind="work_unit", ref_id="wu-1")],
                )
            ]
        )
    )

    snapshot = CompactionSnapshotBuilder().build(build_input)

    assert snapshot.test_results[0].exit_code == 1
    assert any(item.information_type == "test" for item in snapshot.conflicts)


def test_user_constraint_requires_real_user_message_source():
    build_input = _build_input(
        delta=_delta(
            user_constraint_candidates=[
                UserConstraintCandidate(
                    text="never edit tests",
                    source_user_message_id="missing-message",
                    requested_state="active",
                )
            ]
        )
    )

    with pytest.raises(SnapshotBuildError, match="user_message"):
        CompactionSnapshotBuilder().build(build_input)


def test_fact_identity_survives_source_growth():
    claim_id = stable_claim_id("task-a", "parser branch is wrong")
    first = CompactionSnapshotBuilder().build(
        _build_input(
            delta=_delta(
                confirmed_fact_candidates=[
                    FactCandidate(
                        text="parser branch is wrong",
                        sources=[ProvenanceRef(kind="work_unit", ref_id="wu-1")],
                    )
                ]
            )
        )
    )
    previous = _activate(first)

    second = CompactionSnapshotBuilder().build(
        replace(
            _build_input(),
            previous_snapshot=previous,
            delta=_delta(
                confirmed_fact_candidates=[
                    FactCandidate(
                        text="parser branch is wrong",
                        sources=[
                            ProvenanceRef(
                                kind="user_message", ref_id="user-latest"
                            )
                        ],
                    )
                ]
            ),
        )
    )

    claim = next(item for item in second.confirmed_facts if item.claim_id == claim_id)
    assert claim.claim_id == claim_id
    assert {(source.kind, source.ref_id) for source in claim.sources} == {
        ("work_unit", "wu-1"),
        ("user_message", "user-latest"),
    }


def test_reopen_keeps_rejected_identity_and_creates_linked_new_hypothesis():
    first = CompactionSnapshotBuilder().build(
        replace(
            _build_input(),
            delta=_delta(
                hypothesis_transitions=[
                    HypothesisTransition(
                        text="cache is stale",
                        target_state="active",
                        sources=[
                            ProvenanceRef(kind="work_unit", ref_id="wu-1")
                        ],
                    )
                ]
            ),
        )
    )
    active = first.active_hypotheses[0]
    rejected_snapshot = CompactionSnapshotBuilder().build(
        replace(
            _build_input(),
            previous_snapshot=_activate(first),
            delta=_delta(
                hypothesis_transitions=[
                    HypothesisTransition(
                        hypothesis_id=active.hypothesis_id,
                        text=active.text,
                        target_state="rejected",
                        reason="cache was disabled and failure remained",
                        sources=[
                            ProvenanceRef(kind="work_unit", ref_id="wu-1")
                        ],
                    )
                ]
            ),
        )
    )
    rejected = rejected_snapshot.rejected_hypotheses[0]
    transition = HypothesisTransition(
        text="cache is stale under a new code path",
        target_state="active",
        reason="new trace enters another cache",
        reopens_hypothesis_id=rejected.hypothesis_id,
        sources=[ProvenanceRef(kind="work_unit", ref_id="wu-1")],
    )

    snapshot = CompactionSnapshotBuilder().build(
        replace(
            _build_input(),
            previous_snapshot=_activate(rejected_snapshot),
            delta=_delta(hypothesis_transitions=[transition]),
        )
    )

    assert snapshot.rejected_hypotheses[0] == rejected
    reopened = snapshot.active_hypotheses[0]
    assert reopened.hypothesis_id != rejected.hypothesis_id
    assert reopened.reopens_hypothesis_id == rejected.hypothesis_id


def test_semantic_conflict_remains_unresolved():
    conflict = ConflictCandidate(
        information_type="semantic",
        subject="root cause",
        alternatives=[
            ProvenancedText(
                text="parser branch",
                sources=[ProvenanceRef(kind="work_unit", ref_id="wu-1")],
            ),
            ProvenancedText(
                text="cache branch",
                sources=[ProvenanceRef(kind="work_unit", ref_id="wu-1")],
            ),
        ],
    )

    snapshot = CompactionSnapshotBuilder().build(
        replace(
            _build_input(),
            delta=_delta(conflict_candidates=[conflict]),
        )
    )

    assert snapshot.conflicts[0].resolution == "unresolved_semantic"
    assert snapshot.conflicts[0].source_of_truth is None


def test_builder_rejects_tampered_previous_snapshot():
    first = CompactionSnapshotBuilder().build(_build_input())
    tampered = _activate(first.model_copy(update={"task_goal": "different goal"}))

    with pytest.raises(SnapshotBuildError, match="content_hash"):
        CompactionSnapshotBuilder().build(
            replace(_build_input(), previous_snapshot=tampered)
        )


class _StructuredModel:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.payload

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.payload


class _Model:
    def __init__(self, payload):
        self.structured = _StructuredModel(payload)
        self.schemas = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self.structured


def test_delta_generator_uses_supplied_model_instance_and_validates_output():
    model = _Model(_delta())

    result = CompactionDeltaGenerator().generate(model, (_unit(),))

    assert result == _delta()
    assert model.schemas == [CompactionDelta]
    assert "wu-1" in model.structured.calls[0][1].content
