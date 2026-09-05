from __future__ import annotations

from dataclasses import dataclass

from deepfix.compaction.models import FileChangeEvidence, TaskAnchor
from deepfix.evaluation.experiment import ExperimentLoopRunner
from deepfix.evaluation.models import EvaluationBudget, EvaluationCase, RunUsage
from deepfix.investigation.blackboard import CaseBlackboardView, LoopBudgetView
from deepfix.investigation.evaluation import (
    ExperimentResultBuilder,
    RepairLoopOutcome,
    SystemExperimentRuntimeRecord,
)
from deepfix.investigation.experiments import (
    DeterministicSuccessCriterion,
    ExecutorNarrativeResult,
    ExperimentAssessment,
    ExperimentAssessmentOutcome,
    ExperimentIntent,
    ExperimentProgressKind,
    ExperimentSpec,
    NoScopeViolationCheck,
    StrategyAlternative,
    StrategyDecisionCandidate,
    StrategyReflection,
    build_strategy_decision,
)
from deepfix.investigation.loop import (
    AdjudicationEvidence,
    DeepFixRepairLoop,
)
from deepfix.investigation.models import InvestigationCapability
from deepfix.verification import OracleEvaluation


def _blackboard(*, reproduction_state="reproduced") -> CaseBlackboardView:
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
        reproduction_state=reproduction_state,
        confirmed_claims=[],
        hypotheses=[],
        evidence_gaps=[],
        recent_experiments=[],
        deterministic_evidence=[],
        research_evidence=[],
        unresolved_conflicts=[],
        artifact_references=[],
        incomplete_operation_ids=[],
        budget=LoopBudgetView(remaining_experiments=2),
        investigation_state_version=7,
    )


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="experiment-1",
        task_id="task-1",
        intents={
            ExperimentIntent.INVESTIGATE,
            ExperimentIntent.EDIT,
            ExperimentIntent.VERIFY,
        },
        goal="Repair and verify",
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
        fallback="Return evidence",
    )


def _result():
    return ExperimentResultBuilder().build(
        _spec(),
        ExecutorNarrativeResult(
            claimed_completed_criterion_ids=["criterion-1"],
            evidence_candidates=[],
            hypothesis_updates=[],
            remaining_questions=[],
            executor_recommendation="Evaluate evidence",
        ),
        SystemExperimentRuntimeRecord(
            status="completed",
            changed_file_evidence_ids=["file-1"],
            test_evidence_ids=["test-1"],
        ),
    )


def _assessment() -> ExperimentAssessment:
    return ExperimentAssessment(
        experiment_id="experiment-1",
        outcome=ExperimentAssessmentOutcome.SUCCEEDED,
        criterion_assessments=[],
        deterministic_evidence_ids=["file-1", "test-1"],
        accepted_claims=[],
        rejected_claims=[],
        closed_evidence_gap_ids=[],
        opened_evidence_gaps=[],
        supported_hypothesis_ids=[],
        rejected_hypothesis_ids=[],
        conflict_ids=[],
        progress_kind=ExperimentProgressKind.STRONG,
    )


class _Builder:
    def __init__(self, blackboard):
        self.blackboard = blackboard

    def build(self, _task_id):
        return self.blackboard


class _Planner:
    def __init__(self, decision):
        self.decision = decision

    def plan(self, _blackboard):
        return self.decision


class _Executor:
    def execute(self, _spec, _context):
        return _result()


class _Evaluator:
    def evaluate(self, _spec, _result):
        return _assessment()


@dataclass
class _Committer:
    calls: int = 0
    expected_versions: list[int] | None = None

    def __post_init__(self):
        self.expected_versions = []

    def commit(self, expected_version, _result, _assessment):
        self.calls += 1
        self.expected_versions.append(expected_version)


def _decision(decision_type="run_experiment"):
    candidate = StrategyDecisionCandidate(
        decision_type=decision_type,
        current_assessment="Ready",
        evidence_gap_ids=[],
        experiment_spec=_spec() if decision_type == "run_experiment" else None,
        uncertainty=0.1,
        rationale_refs=[],
        candidate_outcome=(
            "not_reproduced" if decision_type == "conclude" else None
        ),
        question_for_user=(
            "Which Python version reproduces this?"
            if decision_type == "ask_user"
            else None
        ),
    )
    return build_strategy_decision("task-1", "blackboard-1", candidate)


def _fixed_evidence() -> AdjudicationEvidence:
    return AdjudicationEvidence(
        oracle_evaluation=OracleEvaluation(
            passed_required_oracle_ids=["oracle-1"],
            all_required_satisfied=True,
            fixed_allowed=True,
        ),
        changed_files=[
            FileChangeEvidence(
                evidence_id="file-1",
                path="target.py",
                operation="edit",
                status="succeeded",
            )
        ],
        reproduction_state="reproduced",
    )


def _loop(blackboard, decision, evidence, *, event_sink=None, executor=None):
    committer = _Committer()
    return DeepFixRepairLoop(
        blackboard_builder=_Builder(blackboard),
        planner=_Planner(decision),
        executor=executor or _Executor(),
        evaluator_factory=lambda _blackboard, _result: _Evaluator(),
        reducer_factory=lambda _task_id: committer,
        adjudication_loader=lambda *_args: evidence,
        event_sink=event_sink,
        max_experiments=2,
    )


def test_vertical_loop_repairs_and_verifies() -> None:
    result = _loop(_blackboard(), _decision(), _fixed_evidence()).run("task-1")

    assert result.outcome is RepairLoopOutcome.FIXED
    assert result.experiments[0].spec.intents == {
        ExperimentIntent.INVESTIGATE,
        ExperimentIntent.EDIT,
        ExperimentIntent.VERIFY,
    }
    assert result.oracle_evaluation.all_required_satisfied is True


def test_vertical_loop_commits_through_reducer_at_blackboard_version() -> None:
    reducer = _Committer()
    loop = DeepFixRepairLoop(
        blackboard_builder=_Builder(_blackboard()),
        planner=_Planner(_decision()),
        executor=_Executor(),
        evaluator_factory=lambda _blackboard, _result: _Evaluator(),
        reducer_factory=lambda _task_id: reducer,
        adjudication_loader=lambda *_args: _fixed_evidence(),
        max_experiments=1,
    )

    loop.run("task-1")

    assert reducer.expected_versions == [7]


def test_vertical_loop_reports_not_reproduced_without_change() -> None:
    evidence = AdjudicationEvidence(
        oracle_evaluation=OracleEvaluation(
            all_required_satisfied=False,
            fixed_allowed=False,
        ),
        reproduction_state="not_reproduced",
    )
    result = _loop(
        _blackboard(reproduction_state="not_reproduced"),
        _decision("conclude"),
        evidence,
    ).run("task-1")

    assert result.outcome is RepairLoopOutcome.NOT_REPRODUCED
    assert result.changed_file_evidence_ids == []


def test_ask_user_pauses_even_when_stale_evidence_would_allow_fixed() -> None:
    result = _loop(
        _blackboard(),
        _decision("ask_user"),
        _fixed_evidence(),
    ).run("task-1")

    assert result.outcome is RepairLoopOutcome.BLOCKED
    assert result.progress_events[-1].event_type == "task_paused"


def test_vertical_loop_emits_high_level_events() -> None:
    result = _loop(_blackboard(), _decision(), _fixed_evidence()).run("task-1")

    assert [event.event_type for event in result.progress_events] == [
        "strategy_planned",
        "experiment_started",
        "experiment_completed",
        "experiment_assessed",
        "outcome_adjudicated",
    ]


def test_timeout_is_explicitly_emitted_and_persistable() -> None:
    class TimedOutExecutor:
        def execute(self, spec, _context):
            runtime = SystemExperimentRuntimeRecord(status="timed_out")
            return ExperimentResultBuilder().build(
                spec,
                ExecutorNarrativeResult(
                    claimed_completed_criterion_ids=[],
                    evidence_candidates=[],
                    hypothesis_updates=[],
                    remaining_questions=["Try a smaller reproducer"],
                    executor_recommendation="Reflect before retrying",
                ),
                runtime,
            )

    persisted = []
    evidence = AdjudicationEvidence(
        oracle_evaluation=OracleEvaluation(
            all_required_satisfied=False,
            fixed_allowed=False,
        ),
        unresolved_operation_ids=["timed-out-operation"],
        reproduction_state="reproduced",
    )

    result = _loop(
        _blackboard(),
        _decision(),
        evidence,
        event_sink=persisted.append,
        executor=TimedOutExecutor(),
    ).run("task-1")

    assert "experiment_timed_out" in {
        event.event_type for event in result.progress_events
    }
    assert persisted == result.progress_events


def test_required_oracle_conflict_routes_vertical_loop_to_review() -> None:
    evidence = _fixed_evidence().model_copy(
        update={
            "oracle_evaluation": OracleEvaluation(
                passed_required_oracle_ids=["oracle-1"],
                conflicting_evidence_ids=["full-suite-failure"],
                all_required_satisfied=True,
                fixed_allowed=False,
            )
        }
    )

    result = _loop(_blackboard(), _decision(), evidence).run("task-1")

    assert result.outcome is RepairLoopOutcome.REVIEW


def test_policy_blocked_experiment_cannot_use_stale_fixed_evidence() -> None:
    class PolicyBlockedExecutor:
        def execute(self, spec, _context):
            return ExperimentResultBuilder().build(
                spec,
                ExecutorNarrativeResult(
                    claimed_completed_criterion_ids=[],
                    evidence_candidates=[],
                    hypothesis_updates=[],
                    remaining_questions=[],
                    executor_recommendation="Wait for policy resolution",
                ),
                SystemExperimentRuntimeRecord(status="policy_blocked"),
            )

    result = _loop(
        _blackboard(),
        _decision(),
        _fixed_evidence(),
        executor=PolicyBlockedExecutor(),
    ).run("task-1")

    assert result.outcome is RepairLoopOutcome.BLOCKED
    assert result.progress_events[-1].event_type == "task_paused"


def test_timeout_feedback_is_given_to_the_next_experiment() -> None:
    class SequentialExecutor:
        def __init__(self):
            self.contexts = []

        def execute(self, spec, context):
            self.contexts.append(context)
            if len(self.contexts) == 1:
                return ExperimentResultBuilder().build(
                    spec,
                    ExecutorNarrativeResult(
                        claimed_completed_criterion_ids=[],
                        evidence_candidates=[],
                        hypothesis_updates=[],
                        remaining_questions=["Use a smaller reproducer"],
                        executor_recommendation="Change strategy",
                    ),
                    SystemExperimentRuntimeRecord(status="timed_out"),
                )
            return _result()

    executor = SequentialExecutor()
    evidence_sequence = iter(
        [
            AdjudicationEvidence(
                oracle_evaluation=OracleEvaluation(
                    all_required_satisfied=False,
                    fixed_allowed=False,
                ),
                reproduction_state="reproduced",
            ),
            _fixed_evidence(),
        ]
    )
    loop = DeepFixRepairLoop(
        blackboard_builder=_Builder(_blackboard()),
        planner=_Planner(_decision()),
        executor=executor,
        evaluator_factory=lambda _blackboard, _result: _Evaluator(),
        reducer_factory=lambda _task_id: _Committer(),
        adjudication_loader=lambda *_args: next(evidence_sequence),
        max_experiments=2,
    )

    result = loop.run("task-1")

    assert result.outcome is RepairLoopOutcome.FIXED
    assert "Use a smaller reproducer" in executor.contexts[1].runtime_feedback


def test_reflect_decision_emits_strategy_changed() -> None:
    decision = _decision().model_copy(
        update={
            "reflection": StrategyReflection(
                alternatives=[
                    StrategyAlternative(
                        description="Trace the caller",
                        rationale="The first hypothesis failed",
                    ),
                    StrategyAlternative(
                        description="Minimize the input",
                        rationale="Isolate a different mechanism",
                    ),
                ],
                prior_strategy_weakness="Initial hypothesis was rejected",
                selected_alternative_index=1,
            )
        }
    )

    result = _loop(_blackboard(), decision, _fixed_evidence()).run("task-1")

    assert "strategy_changed" in {
        event.event_type for event in result.progress_events
    }


def test_evaluation_runner_adapts_injected_loop_without_legacy_fallback(tmp_path) -> None:
    loop_result = _loop(_blackboard(), _decision(), _fixed_evidence()).run("task-1")
    runner = ExperimentLoopRunner(
        project_python=tmp_path / "python.exe",
        case_executor=lambda *_args: (
            loop_result,
            RunUsage(
                input_tokens=10,
                output_tokens=5,
                model_calls=2,
                tool_calls=3,
                wall_seconds=0,
            ),
        ),
        oracle_runner=lambda *_args: 0,
        clock=lambda: 1.0,
    )
    case = EvaluationCase(
        case_id="case-1",
        problem="repair",
        allowed_paths=["target.py"],
        required_command="python -m pytest -q",
        expected_outcome="fixed",
    )

    run = runner.run(
        case,
        tmp_path,
        tmp_path / "001",
        EvaluationBudget(
            max_input_tokens=100,
            max_output_tokens=100,
            max_wall_seconds=10,
            max_tool_calls=10,
            max_side_effects=2,
        ),
    )

    assert run.loop == "experiment"
    assert run.conclusion == "fixed"
    assert run.task_id == "task-1"
