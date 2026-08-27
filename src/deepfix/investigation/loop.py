from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from deepfix.compaction.models import ArtifactReference, StrictModel, TaskAnchor
from deepfix.investigation.blackboard import CaseBlackboardView
from deepfix.investigation.evaluation import (
    ExperimentResultBuilder,
    SystemExperimentRuntimeRecord,
)
from deepfix.investigation.experiments import (
    ExecutorNarrativeResult,
    ExperimentResult,
    ExperimentSpec,
    StrategyDecision,
    StrategyDecisionCandidate,
    build_strategy_decision,
)

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
