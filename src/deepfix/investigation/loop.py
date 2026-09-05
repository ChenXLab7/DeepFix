from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from deepfix.compaction.models import (
    ArtifactReference,
    FileChangeEvidence,
    StrictModel,
    TaskAnchor,
)
from deepfix.investigation.blackboard import CaseBlackboardView
from deepfix.investigation.evaluation import (
    ExperimentResultBuilder,
    RepairLoopOutcome,
    SystemExperimentRuntimeRecord,
    adjudicate_outcome,
)
from deepfix.investigation.experiments import (
    ExecutorNarrativeResult,
    ExperimentAssessment,
    ExperimentResult,
    ExperimentResultStatus,
    ExperimentSpec,
    StrategyDecision,
    StrategyDecisionCandidate,
    build_strategy_decision,
)
from deepfix.investigation.identity import stable_investigation_id
from deepfix.verification import OracleEvaluation

PLANNER_SYSTEM_PROMPT = """You are DeepFix's strategy planner.
Choose exactly one best next decision from the supplied Case Blackboard.
Do not invent tool results or rely on earlier messages. When reflection is required,
compare at least two materially different alternatives before selecting one.
"""


EXPERIMENT_EXECUTOR_SYSTEM_PROMPT = """You are DeepFix's bounded experiment executor.
Complete only the supplied ExperimentSpec using its allowed capabilities and budgets.
You may describe candidate conclusions, but file changes, tests, approvals, and tool
execution are authoritative only when represented by system Tool Receipts/evidence.
Return ExecutorNarrativeResult; never manufacture evidence IDs.
"""


class PlannerContractError(ValueError):
    """The model returned a strategy that violates the planner contract."""


class ExperimentExecutionContext(StrictModel):
    task_anchor: TaskAnchor
    relevant_evidence_ids: list[str] = Field(default_factory=list)
    artifact_references: list[ArtifactReference] = Field(default_factory=list)
    runtime_feedback: list[str] = Field(default_factory=list)


class AdjudicationEvidence(StrictModel):
    oracle_evaluation: OracleEvaluation
    changed_files: list[FileChangeEvidence] = Field(default_factory=list)
    unresolved_operation_ids: list[str] = Field(default_factory=list)
    scope_violation_evidence_ids: list[str] = Field(default_factory=list)
    reproduction_state: Literal["unknown", "reproduced", "not_reproduced"]


class RepairLoopProgressEvent(StrictModel):
    event_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    experiment_id: str | None = None
    detail: str | None = None


class CompletedExperiment(StrictModel):
    spec: ExperimentSpec
    result: ExperimentResult
    assessment: ExperimentAssessment


class RepairLoopResult(StrictModel):
    task_id: str = Field(min_length=1)
    outcome: RepairLoopOutcome
    experiments: list[CompletedExperiment]
    oracle_evaluation: OracleEvaluation
    changed_file_evidence_ids: list[str]
    progress_events: list[RepairLoopProgressEvent]


class StrategyPlanner:
    def __init__(self, model: Any) -> None:
        self._model = model

    def plan(self, blackboard: CaseBlackboardView) -> StrategyDecision:
        structured_model = self._model.with_structured_output(
            StrategyDecisionCandidate
        )
        candidate = structured_model.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=_planner_payload(blackboard)),
            ]
        )
        if not isinstance(candidate, StrategyDecisionCandidate):
            candidate = StrategyDecisionCandidate.model_validate(candidate)
        if blackboard.reevaluation_required:
            if candidate.reflection is None:
                raise PlannerContractError(
                    "REFLECT requires at least two alternatives"
                )
            if len(candidate.reflection.alternatives) < 2:
                raise PlannerContractError(
                    "REFLECT requires at least two alternatives"
                )
        elif candidate.reflection is not None:
            raise PlannerContractError(
                "normal planning must return one strategy without reflection"
            )
        if (
            candidate.experiment_spec is not None
            and candidate.experiment_spec.task_id != blackboard.task_id
        ):
            raise PlannerContractError("experiment task does not match blackboard")
        return build_strategy_decision(
            blackboard.task_id,
            blackboard.fingerprint,
            candidate,
        )


RuntimeRecordLoader = Callable[
    [ExperimentSpec, dict[str, Any]], SystemExperimentRuntimeRecord
]


class LocalAgentExperimentExecutor:
    def __init__(
        self,
        agent: Any,
        *,
        runtime_record_loader: RuntimeRecordLoader,
        result_builder: ExperimentResultBuilder | None = None,
    ) -> None:
        self._agent = agent
        self._runtime_record_loader = runtime_record_loader
        self._result_builder = result_builder or ExperimentResultBuilder()

    def execute(
        self,
        spec: ExperimentSpec,
        context: ExperimentExecutionContext,
    ) -> ExperimentResult:
        if spec.task_id != context.task_anchor.task_id:
            raise ValueError("experiment context task mismatch")
        response = self._agent.invoke(
            {"messages": [HumanMessage(content=_executor_payload(spec, context))]},
            config={
                "configurable": {
                    "task_id": spec.task_id,
                    "thread_id": spec.task_id,
                    "experiment_id": spec.experiment_id,
                    "allowed_capabilities": sorted(
                        capability.value for capability in spec.allowed_capabilities
                    ),
                    "experiment_model_call_budget": spec.model_call_budget,
                    "experiment_token_budget": spec.token_budget,
                    "experiment_time_budget_seconds": spec.time_budget_seconds,
                },
                "recursion_limit": max(2, spec.step_budget * 2 + 1),
            },
        )
        narrative = response.get("structured_response")
        if not isinstance(narrative, ExecutorNarrativeResult):
            narrative = ExecutorNarrativeResult.model_validate(narrative)
        runtime_record = self._runtime_record_loader(spec, response)
        return self._result_builder.build(spec, narrative, runtime_record)


class DeepFixRepairLoop:
    """Evaluation-only outer loop; all authoritative writes remain delegated."""

    def __init__(
        self,
        *,
        blackboard_builder: Any,
        planner: Any,
        executor: Any,
        evaluator_factory: Callable[[CaseBlackboardView, ExperimentResult], Any],
        reducer_factory: Callable[[str], Any],
        adjudication_loader: Callable[..., AdjudicationEvidence],
        event_sink: Callable[[RepairLoopProgressEvent], None] | None = None,
        max_experiments: int = 8,
    ) -> None:
        if max_experiments <= 0:
            raise ValueError("max_experiments must be positive")
        self._blackboard_builder = blackboard_builder
        self._planner = planner
        self._executor = executor
        self._evaluator_factory = evaluator_factory
        self._reducer_factory = reducer_factory
        self._adjudication_loader = adjudication_loader
        self._event_sink = event_sink
        self._max_experiments = max_experiments

    def run(self, task_id: str) -> RepairLoopResult:
        completed: list[CompletedExperiment] = []
        events: list[RepairLoopProgressEvent] = []
        last_evidence: AdjudicationEvidence | None = None
        runtime_feedback: list[str] = []
        for _ in range(self._max_experiments):
            blackboard = self._blackboard_builder.build(task_id)
            decision = self._planner.plan(blackboard)
            self._emit(events, task_id, "strategy_planned")
            if decision.reflection is not None:
                self._emit(
                    events,
                    task_id,
                    "strategy_changed",
                    detail=decision.reflection.prior_strategy_weakness,
                )
            if decision.decision_type != "run_experiment":
                last_evidence = self._adjudication_loader(
                    task_id, blackboard, None, None
                )
                if decision.decision_type == "ask_user":
                    self._emit(
                        events,
                        task_id,
                        "outcome_adjudicated",
                        detail=RepairLoopOutcome.BLOCKED,
                    )
                    self._emit(
                        events,
                        task_id,
                        "task_paused",
                        detail=decision.question_for_user,
                    )
                    return _loop_result(
                        task_id,
                        RepairLoopOutcome.BLOCKED,
                        completed,
                        last_evidence,
                        events,
                    )
                outcome = _adjudicate(last_evidence)
                self._emit(events, task_id, "outcome_adjudicated", detail=outcome)
                return _loop_result(task_id, outcome, completed, last_evidence, events)

            spec = decision.experiment_spec
            if spec is None:
                raise PlannerContractError("run_experiment requires an ExperimentSpec")
            self._emit(
                events,
                task_id,
                "experiment_started",
                experiment_id=spec.experiment_id,
            )
            context = ExperimentExecutionContext(
                task_anchor=blackboard.task_anchor,
                relevant_evidence_ids=[
                    item.evidence_id for item in blackboard.deterministic_evidence
                ],
                artifact_references=blackboard.artifact_references,
                runtime_feedback=[
                    *(item.text for item in blackboard.evidence_gaps),
                    *runtime_feedback,
                ],
            )
            result = self._executor.execute(spec, context)
            for receipt_id in result.tool_receipt_ids:
                self._emit(
                    events,
                    task_id,
                    "experiment_tool_called",
                    experiment_id=spec.experiment_id,
                    detail=receipt_id,
                )
            if result.status is ExperimentResultStatus.TIMED_OUT:
                self._emit(
                    events,
                    task_id,
                    "experiment_timed_out",
                    experiment_id=spec.experiment_id,
                    detail="The next strategy must react to the timeout",
                )
            runtime_feedback = [
                f"Previous experiment {spec.experiment_id} ended as {result.status}",
                *result.executor_narrative.remaining_questions,
            ]
            self._emit(
                events,
                task_id,
                "experiment_completed",
                experiment_id=spec.experiment_id,
                detail=result.status,
            )
            assessment = self._evaluator_factory(blackboard, result).evaluate(
                spec, result
            )
            self._reducer_factory(task_id).commit(
                blackboard.investigation_state_version,
                result,
                assessment,
            )
            completed.append(
                CompletedExperiment(
                    spec=spec,
                    result=result,
                    assessment=assessment,
                )
            )
            self._emit(
                events,
                task_id,
                "experiment_assessed",
                experiment_id=spec.experiment_id,
                detail=assessment.outcome,
            )
            for hypothesis_id in [
                *assessment.supported_hypothesis_ids,
                *assessment.rejected_hypothesis_ids,
            ]:
                self._emit(
                    events,
                    task_id,
                    "hypothesis_updated",
                    experiment_id=spec.experiment_id,
                    detail=hypothesis_id,
                )
            for gap_id in assessment.closed_evidence_gap_ids:
                self._emit(
                    events,
                    task_id,
                    "evidence_gap_closed",
                    experiment_id=spec.experiment_id,
                    detail=gap_id,
                )
            last_evidence = self._adjudication_loader(
                task_id, blackboard, result, assessment
            )
            forced_pause = result.status in {
                ExperimentResultStatus.WAITING_APPROVAL,
                ExperimentResultStatus.POLICY_BLOCKED,
                ExperimentResultStatus.BUDGET_EXHAUSTED,
            }
            outcome = (
                RepairLoopOutcome.BLOCKED
                if forced_pause
                else _adjudicate(last_evidence)
            )
            self._emit(
                events,
                task_id,
                "outcome_adjudicated",
                experiment_id=spec.experiment_id,
                detail=outcome,
            )
            if outcome is not RepairLoopOutcome.CONTINUE:
                if forced_pause:
                    self._emit(
                        events,
                        task_id,
                        "task_paused",
                        experiment_id=spec.experiment_id,
                        detail=result.status,
                    )
                return _loop_result(task_id, outcome, completed, last_evidence, events)

        if last_evidence is None:
            raise RuntimeError("loop exhausted without adjudication evidence")
        self._emit(events, task_id, "task_paused", detail="experiment_budget_exhausted")
        return _loop_result(
            task_id,
            RepairLoopOutcome.BLOCKED,
            completed,
            last_evidence,
            events,
        )

    def _emit(
        self,
        events: list[RepairLoopProgressEvent],
        task_id: str,
        event_type: str,
        *,
        experiment_id: str | None = None,
        detail: object | None = None,
    ) -> None:
        event = RepairLoopProgressEvent(
            event_id=stable_investigation_id(
                "repair-loop-event",
                task_id,
                str(len(events)),
                event_type,
                experiment_id or "none",
            ),
            task_id=task_id,
            event_type=event_type,
            experiment_id=experiment_id,
            detail=str(detail) if detail is not None else None,
        )
        events.append(event)
        if self._event_sink is not None:
            self._event_sink(event)


def _adjudicate(evidence: AdjudicationEvidence) -> RepairLoopOutcome:
    return adjudicate_outcome(
        evidence.oracle_evaluation,
        changed_files=evidence.changed_files,
        unresolved_operation_ids=evidence.unresolved_operation_ids,
        scope_violation_evidence_ids=evidence.scope_violation_evidence_ids,
        reproduction_state=evidence.reproduction_state,
    )


def _loop_result(
    task_id: str,
    outcome: RepairLoopOutcome,
    completed: list[CompletedExperiment],
    evidence: AdjudicationEvidence,
    events: list[RepairLoopProgressEvent],
) -> RepairLoopResult:
    changed_ids = list(
        dict.fromkeys(
            evidence_id
            for item in completed
            for evidence_id in item.result.changed_file_evidence_ids
        )
    )
    return RepairLoopResult(
        task_id=task_id,
        outcome=outcome,
        experiments=completed,
        oracle_evaluation=evidence.oracle_evaluation,
        changed_file_evidence_ids=changed_ids,
        progress_events=events,
    )


def _planner_payload(blackboard: CaseBlackboardView) -> str:
    payload = blackboard.model_dump(mode="json")
    return "Case Blackboard and remaining budget:\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )


def _executor_payload(
    spec: ExperimentSpec,
    context: ExperimentExecutionContext,
) -> str:
    payload = {
        "task_anchor": context.task_anchor.model_dump(mode="json"),
        "experiment_spec": spec.model_dump(mode="json"),
        "relevant_evidence_ids": context.relevant_evidence_ids,
        "artifact_references": [
            item.model_dump(mode="json") for item in context.artifact_references
        ],
        "runtime_feedback": context.runtime_feedback,
        "allowed_capabilities": sorted(
            capability.value for capability in spec.allowed_capabilities
        ),
    }
    return "Execute this bounded experiment:\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )
