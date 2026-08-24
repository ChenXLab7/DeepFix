import pytest

from deepfix.investigation.errors import InvestigationStagnationError
from deepfix.investigation.models import (
    InvestigationState,
    NewInvestigationEvent,
    ToolObservation,
)
from deepfix.investigation.stagnation import StagnationDetector, matching_cycle_size
from investigation.helpers import continue_input, stagnated_coordinator


def event(event_type: str, progress: str | None = None) -> ToolObservation:
    return ToolObservation(event_type=event_type, progress_kind=progress)


def no_progress(signature: str, scope: str = "direct") -> ToolObservation:
    return ToolObservation(
        event_type="tool_completed",
        signature=signature,
        result_fingerprint=f"result-{signature}",
        scope=scope,
    )


def state_with_no_progress_count(count: int) -> InvestigationState:
    return InvestigationState.new("task-a").model_copy(
        update={
            "no_progress_count": count,
            "recent_tool_signatures": [f"old-{index}" for index in range(count)],
        }
    )


def test_exact_repeat_triggers_on_third_result():
    detector = StagnationDetector()
    state = InvestigationState.new("task-a")

    for _ in range(2):
        state = detector.after_tool(state, no_progress("same-signature"))
        assert state.stagnation_level == 0
    state = detector.after_tool(state, no_progress("same-signature"))

    assert state.stagnation_level == 1


def test_same_tool_arguments_with_different_results_are_not_exact_repeats():
    detector = StagnationDetector()
    state = InvestigationState.new("task-a")

    for index in range(3):
        state = detector.after_tool(
            state,
            ToolObservation(
                event_type="tool_completed",
                signature="same-tool-args",
                result_fingerprint=f"different-result-{index}",
            ),
        )

    assert state.stagnation_level == 0


@pytest.mark.parametrize("size", [2, 8])
def test_short_cycle_repeated_twice_triggers(size):
    signatures = [f"s-{index}" for index in range(size)] * 2

    assert matching_cycle_size(signatures) == size


def test_six_no_progress_results_trigger_reevaluation():
    detector = StagnationDetector()
    state = InvestigationState.new("task-a")

    for index in range(6):
        state = detector.after_tool(state, no_progress(f"unique-{index}"))

    assert state.stagnation_level == 1
    assert state.reevaluation_required


def test_four_exploratory_results_trigger_reevaluation():
    detector = StagnationDetector()
    state = InvestigationState.new("task-a")

    for index in range(4):
        state = detector.after_tool(
            state,
            no_progress(f"explore-{index}", scope="exploratory"),
        )

    assert state.stagnation_level == 1


def test_phase_change_and_new_file_do_not_reset_stagnation():
    detector = StagnationDetector()
    state = state_with_no_progress_count(5)

    state = detector.after_event(state, event("phase_changed"))
    state = detector.after_tool(
        state,
        no_progress("read-new-file", scope="exploratory"),
    )

    assert state.stagnation_level == 1


def test_artifact_retrieval_counts_as_ordinary_no_progress_activity():
    detector = StagnationDetector()
    state = InvestigationState.new("task-a")

    for index in range(6):
        state = detector.after_tool(
            state,
            ToolObservation(
                event_type="artifact_searched",
                signature=f"artifact-query-{index}",
                result_fingerprint=f"artifact-result-{index}",
            ),
        )

    assert state.no_progress_count == 6
    assert state.progress_generation == 0
    assert state.stagnation_level == 1


def test_strong_progress_resets_generation_and_counters():
    detector = StagnationDetector()
    state = state_with_no_progress_count(5)

    updated = detector.after_event(
        state,
        event("hypothesis_rejected", progress="hypothesis_transition"),
    )

    assert updated.progress_generation == state.progress_generation + 1
    assert updated.no_progress_count == 0
    assert updated.stagnation_level == 0


def test_permit_allows_only_bound_tool_once_and_then_pauses(tmp_path):
    coordinator = stagnated_coordinator(tmp_path)

    permit = coordinator.grant_investigation_permit("task-a", continue_input())

    authorization = coordinator.authorize_tool(
        "task-a",
        "grep",
        {"pattern": "flip", "path": "src/sign.py"},
    )
    assert authorization.allowed
    assert authorization.permit_id == permit.permit_id
    coordinator.record_observation("task-a", no_progress("permitted-grep"))
    with pytest.raises(InvestigationStagnationError):
        coordinator.authorize_tool(
            "task-a",
            "read_file",
            {"file_path": "src/other.py"},
        )


def test_permit_does_not_authorize_a_different_target(tmp_path):
    coordinator = stagnated_coordinator(tmp_path)
    coordinator.grant_investigation_permit("task-a", continue_input())

    with pytest.raises(InvestigationStagnationError):
        coordinator.authorize_tool(
            "task-a",
            "grep",
            {"pattern": "other", "path": "src/sign.py"},
        )

    assert coordinator.state("task-a").permit is not None
    assert not coordinator.state("task-a").permit.consumed


def test_candidate_can_request_one_targeted_tool_from_decision_checkpoint(tmp_path):
    coordinator = stagnated_coordinator(tmp_path)
    state = coordinator.state("task-a")
    state = state.model_copy(
        update={
            "stagnation_level": 0,
            "reevaluation_required": False,
            "diagnostic_decision_required": True,
            "permit": None,
        }
    )
    coordinator.store.commit(
        coordinator.state("task-a").version,
        [
            NewInvestigationEvent(
                event_id="event-decision-checkpoint",
                task_id="task-a",
                event_type="reevaluation_required",
                phase_before=state.agent_phase,
                phase_after=state.agent_phase,
            )
        ],
        state,
    )

    permit = coordinator.grant_investigation_permit("task-a", continue_input())

    assert permit.consumed is False
    authorization = coordinator.authorize_tool(
        "task-a",
        "grep",
        {"pattern": "flip", "path": "src/sign.py"},
    )
    assert authorization.allowed is True
    assert authorization.permit_id == permit.permit_id
