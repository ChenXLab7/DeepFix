from dataclasses import replace

from deepfix.compaction.models import (
    CompactionDelta,
    FactCandidate,
    HypothesisTransition,
    ProvenanceRef,
    UserConstraint,
)
from deepfix.compaction.snapshot import CompactionSnapshotBuilder

from .test_snapshot import _activate, _build_input


def _paraphrase_delta(index):
    return CompactionDelta(
        user_constraint_candidates=[],
        confirmed_fact_candidates=[
            FactCandidate(
                text=f"第 {index + 1} 次自然语言摘要换了一种说法",
                sources=[ProvenanceRef(kind="work_unit", ref_id="wu-1")],
            )
        ],
        hypothesis_transitions=[],
        experiments=[],
        conflict_candidates=[],
        unresolved_questions=[],
        next_steps=[],
    )


def test_three_compactions_preserve_authoritative_and_rejected_records_byte_for_byte():
    constraint = UserConstraint(
        constraint_id="constraint-original",
        text="不要修改测试文件",
        source_user_message_id="user-latest",
    )
    build_input = _build_input(
        task_anchor=_build_input().task_anchor.model_copy(
            update={"user_constraints": [constraint]}
        )
    )
    first = CompactionSnapshotBuilder().build(
        replace(
            build_input,
            delta=CompactionDelta(
                user_constraint_candidates=[],
                confirmed_fact_candidates=[],
                hypothesis_transitions=[
                    HypothesisTransition(
                        text="依赖缓存损坏",
                        target_state="active",
                        sources=[
                            ProvenanceRef(kind="work_unit", ref_id="wu-1")
                        ],
                    )
                ],
                experiments=[],
                conflict_candidates=[],
                unresolved_questions=[],
                next_steps=[],
            ),
        )
    )
    active = first.active_hypotheses[0]
    current = CompactionSnapshotBuilder().build(
        replace(
            build_input,
            previous_snapshot=_activate(first),
            delta=CompactionDelta(
                user_constraint_candidates=[],
                confirmed_fact_candidates=[],
                hypothesis_transitions=[
                    HypothesisTransition(
                        hypothesis_id=active.hypothesis_id,
                        text=active.text,
                        target_state="rejected",
                        reason="禁用缓存后同一测试仍失败",
                        sources=active.sources,
                    )
                ],
                experiments=[],
                conflict_candidates=[],
                unresolved_questions=[],
                next_steps=[],
            ),
        )
    )
    rejected = current.rejected_hypotheses[0]

    for index in range(3):
        current = CompactionSnapshotBuilder().build(
            replace(
                build_input,
                previous_snapshot=_activate(current),
                delta=_paraphrase_delta(index),
                input_hash=str(index + 1) * 64,
            )
        )

    assert current.user_constraints[0].model_dump_json() == constraint.model_dump_json()
    assert current.rejected_hypotheses[0].model_dump_json() == rejected.model_dump_json()
    assert current.test_results[0].command == "pytest -q"
    assert current.test_results[0].exit_code == 1
    assert len(current.confirmed_facts) == 3
