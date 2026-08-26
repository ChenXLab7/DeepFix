import pytest

from deepfix.investigation.models import AgentPhase, ToolObservation
from deepfix.investigation.phase import PhaseResolver


def event(event_type: str, **updates) -> ToolObservation:
    return ToolObservation(event_type=event_type).model_copy(update=updates)


def test_pytest_failure_moves_investigating_to_diagnosing():
    resolved = PhaseResolver().resolve(
        AgentPhase.INVESTIGATING,
        event("test_observed", exit_code=1),
    )

    assert resolved is AgentPhase.DIAGNOSING


def test_phase_changed_never_carries_progress_kind():
    result = PhaseResolver().transition(
        "task-a",
        AgentPhase.DIAGNOSING,
        event("hypothesis_supported"),
    )

    assert result.phase is AgentPhase.PLANNING
    assert result.phase_event is not None
    assert result.phase_event.progress_kind is None


def test_matching_edit_results_from_distinct_tool_calls_get_distinct_phase_ids():
    resolver = PhaseResolver()
    first = resolver.transition(
        "task-a",
        AgentPhase.PLANNING,
        event(
            "file_changed",
            tool_call_id="edit-call-1",
            result_fingerprint="same-artifactless-edit-result",
        ),
    )
    second = resolver.transition(
        "task-a",
        AgentPhase.PLANNING,
        event(
            "file_changed",
            tool_call_id="edit-call-2",
            result_fingerprint="same-artifactless-edit-result",
        ),
    )

    assert first.phase_event is not None
    assert second.phase_event is not None
    assert first.phase_event.event_id != second.phase_event.event_id


@pytest.mark.parametrize(
    ("phase", "event_type", "exit_code", "expected"),
    [
        (AgentPhase.PLANNING, "file_changed", None, AgentPhase.EDITING),
        (
            AgentPhase.EDITING,
            "verification_execution_observed",
            None,
            AgentPhase.TESTING,
        ),
        (AgentPhase.TESTING, "post_edit_test_observed", 0, AgentPhase.REVIEWING),
        (AgentPhase.TESTING, "post_edit_test_observed", 1, AgentPhase.DIAGNOSING),
    ],
)
def test_phase_transitions_require_their_deterministic_trigger(
    phase,
    event_type,
    exit_code,
    expected,
):
    assert PhaseResolver().resolve(
        phase,
        event(event_type, exit_code=exit_code),
    ) is expected


def test_editing_does_not_enter_testing_on_generic_tool_completion():
    assert PhaseResolver().resolve(
        AgentPhase.EDITING,
        event("tool_completed"),
    ) is AgentPhase.EDITING
