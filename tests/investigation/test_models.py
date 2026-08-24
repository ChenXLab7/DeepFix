import pytest
from pydantic import ValidationError

from deepfix.investigation.models import (
    AgentPhase,
    CheckedLocation,
    ContinueInvestigationInput,
    InvestigationState,
    ProposedChange,
    RecordHypothesisInput,
)


def test_new_state_has_no_false_progress_or_permit():
    state = InvestigationState.new("task-a")

    assert state.agent_phase is AgentPhase.INVESTIGATING
    assert state.progress_generation == 0
    assert state.stagnation_level == 0
    assert state.permit is None


def test_supported_hypothesis_requires_operational_fields():
    with pytest.raises(ValidationError):
        RecordHypothesisInput(
            statement="sign is inverted twice",
            evidence_ids=["evidence-1"],
            checked_locations=[],
            target_state="supported",
            reason="failure points here",
        )


def test_supported_hypothesis_accepts_complete_operational_fields():
    command = RecordHypothesisInput(
        statement="sign is inverted twice",
        evidence_ids=["evidence-1"],
        checked_locations=[CheckedLocation(path="src/sign.py", start_line=1, end_line=8)],
        proposed_change=ProposedChange(
            path="src/sign.py",
            description="remove the duplicate negation",
        ),
        expected_effect="negative sign is applied exactly once",
        target_state="supported",
        reason="the failing branch contains two negations",
    )

    assert command.target_state == "supported"


def test_rejected_hypothesis_requires_existing_identity():
    with pytest.raises(ValidationError):
        RecordHypothesisInput(
            statement="the parser changes the sign",
            evidence_ids=["evidence-1"],
            checked_locations=[],
            target_state="rejected",
            reason="the parser preserves the input",
        )


def test_continue_intent_requires_a_tool_and_target():
    with pytest.raises(ValidationError):
        ContinueInvestigationInput(
            hypothesis_ids=["hyp-1"],
            unresolved_question="which branch flips sign?",
            expected_evidence="branch condition",
            tool_name="",
            target="src/sign.py",
            reason="inspect the branch",
        )


def test_checked_location_rejects_reversed_line_range():
    with pytest.raises(ValidationError):
        CheckedLocation(path="src/sign.py", start_line=8, end_line=1)

