from __future__ import annotations

import pytest

from deepfix.compaction.models import TaskAnchor
from deepfix.investigation.blackboard import CaseBlackboardView, LoopBudgetView
from deepfix.investigation.evaluation import SystemExperimentRuntimeRecord
from deepfix.investigation.experiments import (
    DeterministicSuccessCriterion,
    ExecutorNarrativeResult,
    ExperimentIntent,
    ExperimentResultStatus,
    ExperimentSpec,
    NoScopeViolationCheck,
    StrategyAlternative,
    StrategyDecisionCandidate,
    StrategyReflection,
)
from deepfix.investigation.loop import (
    ExperimentExecutionContext,
    LocalAgentExperimentExecutor,
    PlannerContractError,
    StrategyPlanner,
)
from deepfix.investigation.models import InvestigationCapability


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="experiment-1",
        task_id="task-1",
        intents={
            ExperimentIntent.INVESTIGATE,
            ExperimentIntent.EDIT,
            ExperimentIntent.VERIFY,
        },
        goal="Find, repair, and verify the defect",
        evidence_gap_ids=[],
        success_criteria=[
            DeterministicSuccessCriterion(
                criterion_id="criterion-1",
                description="No scope violation",
                check=NoScopeViolationCheck(),
            )
        ],
        allowed_capabilities={
            InvestigationCapability.READ,
            InvestigationCapability.MODIFY,
            InvestigationCapability.EXECUTE,
        },
        step_budget=8,
        model_call_budget=4,
        time_budget_seconds=60,
        fallback="Return the strongest evidence and remaining gap",
    )


def _blackboard(*, reevaluation_required: bool = False) -> CaseBlackboardView:
    return CaseBlackboardView(
        task_id="task-1",
        fingerprint="blackboard-1",
        task_anchor=TaskAnchor(
            task_id="task-1",
            task_goal="Repair target",
            latest_user_message_id="message-1",
            project_root="C:/workspace",
            project_python="C:/Python/python.exe",
            task_status="running",
        ),
        reproduction_state="reproduced",
        confirmed_claims=[],
        hypotheses=[],
        evidence_gaps=[],
        recent_experiments=[],
        deterministic_evidence=[],
        research_evidence=[],
        unresolved_conflicts=[],
        artifact_references=[],
        incomplete_operation_ids=[],
        budget=LoopBudgetView(remaining_experiments=3),
        stagnation_level=1 if reevaluation_required else 0,
        reevaluation_required=reevaluation_required,
    )


class _StructuredModel:
    def __init__(self, response: StrategyDecisionCandidate) -> None:
        self.response = response
        self.messages = None

    def with_structured_output(self, schema):
        assert schema is StrategyDecisionCandidate
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.response


def _candidate(*, reflection: StrategyReflection | None = None):
    return StrategyDecisionCandidate(
        decision_type="run_experiment",
        current_assessment="The failure is reproduced",
        evidence_gap_ids=[],
        experiment_spec=_spec(),
        reflection=reflection,
        uncertainty=0.2,
        rationale_refs=[],
    )


def test_normal_planner_returns_one_strategy_without_conversation_history() -> None:
    model = _StructuredModel(_candidate())

    decision = StrategyPlanner(model).plan(_blackboard())

    assert decision.decision_type == "run_experiment"
    assert decision.reflection is None
    rendered = "\n".join(str(message.content) for message in model.messages)
    assert "blackboard-1" in rendered
    assert "conversation history" not in rendered.lower()


def test_reflect_requires_two_alternatives() -> None:
    model = _StructuredModel(_candidate())

    with pytest.raises(PlannerContractError, match="two alternatives"):
        StrategyPlanner(model).plan(_blackboard(reevaluation_required=True))

    reflection = StrategyReflection(
        alternatives=[
            StrategyAlternative(description="Trace callers", rationale="New evidence"),
            StrategyAlternative(description="Minimize input", rationale="Isolate case"),
        ],
        prior_strategy_weakness="Repeated the same inspection",
        selected_alternative_index=0,
    )
    decision = StrategyPlanner(_StructuredModel(_candidate(reflection=reflection))).plan(
        _blackboard(reevaluation_required=True)
    )
    assert len(decision.reflection.alternatives) == 2


class _FakeExperimentAgent:
    def __init__(self) -> None:
        self.config = None
        self.prompt = ""

    def invoke(self, inputs, config):
        self.config = config
        self.prompt = inputs["messages"][0].content
        return {
            "structured_response": ExecutorNarrativeResult(
                claimed_completed_criterion_ids=["criterion-1"],
                evidence_candidates=[],
                hypothesis_updates=[],
                remaining_questions=[],
                executor_recommendation="Evaluate deterministic evidence",
            ),
            "runtime_record": SystemExperimentRuntimeRecord(
                status=ExperimentResultStatus.COMPLETED,
                tool_receipt_ids=["receipt-read", "receipt-edit", "receipt-test"],
                changed_file_evidence_ids=["file-evidence-1"],
                test_evidence_ids=["test-evidence-1"],
            ),
        }


def test_simple_experiment_can_read_edit_and_verify_with_system_evidence_ids() -> None:
    agent = _FakeExperimentAgent()
    executor = LocalAgentExperimentExecutor(
        agent,
        runtime_record_loader=lambda _spec, response: response["runtime_record"],
    )

    result = executor.execute(
        _spec(),
        ExperimentExecutionContext(task_anchor=_blackboard().task_anchor),
    )

    assert result.status is ExperimentResultStatus.COMPLETED
    assert result.changed_file_evidence_ids == ["file-evidence-1"]
    assert result.test_evidence_ids == ["test-evidence-1"]
    assert agent.config["configurable"]["experiment_id"] == "experiment-1"
    assert set(agent.config["configurable"]["allowed_capabilities"]) == {
        "read",
        "modify",
        "execute",
    }
    assert "Find, repair, and verify" in agent.prompt
