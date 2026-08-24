from deepfix.investigation.models import (
    InvestigationState,
    ProgressKind,
    ToolObservation,
)
from deepfix.investigation.progress import ProgressEvaluator


def observation(event_type: str, **updates) -> ToolObservation:
    base = ToolObservation(event_type=event_type)
    return base.model_copy(update=updates)


def test_new_file_and_phase_change_are_not_strong_progress():
    evaluator = ProgressEvaluator()

    assert evaluator.evaluate(observation("file_checked", path="src/new.py")) is None
    assert evaluator.evaluate(observation("phase_changed")) is None


def test_content_fingerprint_change_is_not_strong_progress():
    evaluator = ProgressEvaluator()
    item = observation(
        "file_checked",
        path="src/sign.py",
        result_fingerprint="new-content-hash",
    )

    assert evaluator.evaluate(item) is None


def test_new_test_evidence_and_supported_hypothesis_are_strong_progress():
    evaluator = ProgressEvaluator()

    assert (
        evaluator.evaluate(observation("test_observed", evidence_id="e1"))
        is ProgressKind.TEST_EVIDENCE
    )
    assert (
        evaluator.evaluate(observation("hypothesis_supported", hypothesis_id="h1"))
        is ProgressKind.HYPOTHESIS_TRANSITION
    )


def test_same_test_command_and_result_in_same_generation_is_not_new_progress():
    evaluator = ProgressEvaluator()
    state = InvestigationState.new("task-a")
    first = observation(
        "test_observed",
        signature="pytest-q",
        result_fingerprint="same-result",
        evidence_id="e1",
    )
    second = observation(
        "test_observed",
        signature="pytest-q",
        result_fingerprint="same-result",
        evidence_id="e2",
    )

    state, progress = evaluator.apply(state, first)
    assert progress is ProgressKind.TEST_EVIDENCE
    state, progress = evaluator.apply(state, second)
    assert progress is None


def test_same_result_from_different_test_command_is_new_progress():
    evaluator = ProgressEvaluator()
    state = InvestigationState.new("task-a")
    first = observation(
        "test_observed",
        signature="pytest-unit",
        result_fingerprint="same-result",
    )
    second = observation(
        "test_observed",
        signature="pytest-integration",
        result_fingerprint="same-result",
    )

    state, _ = evaluator.apply(state, first)
    _, progress = evaluator.apply(state, second)

    assert progress is ProgressKind.TEST_EVIDENCE


def test_failed_file_change_is_not_strong_progress():
    evaluator = ProgressEvaluator()

    assert evaluator.evaluate(observation("file_change_failed")) is None


def test_diagnostic_artifact_retrieval_is_not_strong_progress():
    evaluator = ProgressEvaluator()

    assert evaluator.evaluate(observation("artifact_searched")) is None
    assert evaluator.evaluate(observation("artifact_read")) is None
