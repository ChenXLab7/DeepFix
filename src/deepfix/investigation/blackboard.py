from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import Field

from deepfix.compaction.models import (
    ApprovalEvidence,
    ArtifactReference,
    ConflictRecord,
    ExperimentRecord,
    FileChangeEvidence,
    HypothesisRecord,
    ProvenancedClaim,
    ProvenancedText,
    ResearchStatusEvidence,
    StrictModel,
    SystemTestEvidence,
    TaskAnchor,
)
from deepfix.investigation.identity import stable_investigation_id
from deepfix.operations import OperationJournalEntry
from deepfix.protected_context import ProtectedContext
from deepfix.research.models import ExternalEvidence
from deepfix.verification import VerificationPolicy

DeterministicEvidence = Annotated[
    SystemTestEvidence
    | FileChangeEvidence
    | ApprovalEvidence
    | ResearchStatusEvidence,
    Field(union_mode="left_to_right"),
]


class BlackboardHypothesis(StrictModel):
    hypothesis_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    state: Literal["active", "rejected", "confirmed"]
    reason: str | None = None
    provenance_root_ids: list[str] = Field(default_factory=list)


class ResearchEvidenceRef(StrictModel):
    evidence_id: str = Field(min_length=1)
    verification: Literal["unverified", "verified", "contradicted"]
    artifact_path: str | None = None


class LoopBudgetView(StrictModel):
    remaining_experiments: int | None = Field(default=None, ge=0)
    remaining_model_calls: int | None = Field(default=None, ge=0)
    remaining_input_tokens: int | None = Field(default=None, ge=0)
    remaining_output_tokens: int | None = Field(default=None, ge=0)
    remaining_time_seconds: float | None = Field(default=None, ge=0)


class CaseBlackboardView(StrictModel):
    task_id: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    task_anchor: TaskAnchor
    reproduction_state: Literal["unknown", "reproduced", "not_reproduced"]
    working_memory_version: int | None = None
    confirmed_claims: list[ProvenancedClaim]
    hypotheses: list[BlackboardHypothesis]
    evidence_gaps: list[ProvenancedText]
    recent_experiments: list[ExperimentRecord]
    deterministic_evidence: list[DeterministicEvidence]
    research_evidence: list[ResearchEvidenceRef]
    unresolved_conflicts: list[ConflictRecord]
    artifact_references: list[ArtifactReference]
    verification_policy: VerificationPolicy | None = None
    incomplete_operation_ids: list[str]
    budget: LoopBudgetView
    stagnation_level: int = Field(default=0, ge=0)
    reevaluation_required: bool = False

    @property
    def test_results(self) -> list[SystemTestEvidence]:
        return [
            item
            for item in self.deterministic_evidence
            if isinstance(item, SystemTestEvidence)
        ]

    @property
    def changed_files(self) -> list[FileChangeEvidence]:
        return [
            item
            for item in self.deterministic_evidence
            if isinstance(item, FileChangeEvidence)
        ]


class CaseBlackboardBuilder:
    def __init__(
        self,
        context_loader: Callable[[str], ProtectedContext],
        *,
        policy_loader: Callable[[str], VerificationPolicy | None] | None = None,
        operation_loader: Callable[[str], list[OperationJournalEntry]] | None = None,
        research_loader: Callable[[str], list[ExternalEvidence]] | None = None,
        budget_loader: Callable[[str], LoopBudgetView] | None = None,
    ) -> None:
        self._context_loader = context_loader
        self._policy_loader = policy_loader
        self._operation_loader = operation_loader
        self._research_loader = research_loader
        self._budget_loader = budget_loader

    def build(self, task_id: str) -> CaseBlackboardView:
        context = self._context_loader(task_id)
        if context.task_anchor.task_id != task_id:
            raise ValueError("protected context task mismatch")
        snapshot = context.active_snapshot
        current_evidence = [
            *context.deterministic_evidence.tests,
            *context.deterministic_evidence.files,
            *context.deterministic_evidence.approvals,
            *context.deterministic_evidence.research,
        ]
        historical_evidence = []
        if snapshot is not None:
            historical_evidence = [
                *snapshot.deterministic_evidence.tests,
                *snapshot.deterministic_evidence.files,
                *snapshot.deterministic_evidence.approvals,
                *snapshot.deterministic_evidence.research,
                *snapshot.test_results,
                *snapshot.changed_files,
            ]
        deterministic_evidence = _unique_by_id(
            [*current_evidence, *historical_evidence],
            "evidence_id",
        )
        memory = context.working_memory
        confirmed_claims = _unique_by_id(
            [
                *(memory.snapshot.facts if memory else []),
                *(snapshot.confirmed_facts if snapshot else []),
            ],
            "claim_id",
        )
        hypotheses = _project_hypotheses(context)
        constraints = _unique_by_id(
            [
                *context.task_anchor.user_constraints,
                *(snapshot.user_constraints if snapshot else []),
            ],
            "constraint_id",
        )
        anchor = context.task_anchor.model_copy(
            update={"user_constraints": constraints}
        )
        external_research = (
            self._research_loader(task_id) if self._research_loader else []
        )
        research_evidence = _unique_by_id(
            [
                *(
                    ResearchEvidenceRef(
                        evidence_id=item.evidence_id,
                        verification=item.local_verification,
                        artifact_path=item.artifact_path,
                    )
                    for item in external_research
                ),
                *(
                    ResearchEvidenceRef(
                        evidence_id=item.evidence_id,
                        verification=item.verification,
                        artifact_path=item.artifact_path,
                    )
                    for item in context.deterministic_evidence.research
                ),
            ],
            "evidence_id",
        )
        unresolved_questions = [
            *(memory.snapshot.unresolved_questions if memory else []),
            *(snapshot.unresolved_questions if snapshot else []),
        ]
        evidence_gaps = _unique_text(unresolved_questions)
        operations = (
            self._operation_loader(task_id) if self._operation_loader else []
        )
        payload = {
            "task_id": task_id,
            "task_anchor": anchor,
            "reproduction_state": _reproduction_state(deterministic_evidence),
            "working_memory_version": memory.version if memory else None,
            "confirmed_claims": confirmed_claims,
            "hypotheses": hypotheses,
            "evidence_gaps": evidence_gaps,
            "recent_experiments": (
                snapshot.experiments[-5:] if snapshot else []
            ),
            "deterministic_evidence": deterministic_evidence,
            "research_evidence": research_evidence,
            "unresolved_conflicts": snapshot.conflicts if snapshot else [],
            "artifact_references": snapshot.artifact_references if snapshot else [],
            "verification_policy": (
                self._policy_loader(task_id) if self._policy_loader else None
            ),
            "incomplete_operation_ids": [item.operation_id for item in operations],
            "budget": (
                self._budget_loader(task_id)
                if self._budget_loader
                else LoopBudgetView()
            ),
            "stagnation_level": (
                context.investigation_state.stagnation_level
                if context.investigation_state is not None
                else 0
            ),
            "reevaluation_required": (
                context.investigation_state.reevaluation_required
                if context.investigation_state is not None
                else False
            ),
        }
        fingerprint = stable_investigation_id(
            "blackboard",
            task_id,
            json.dumps(
                _jsonable(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        return CaseBlackboardView(fingerprint=fingerprint, **payload)


def _project_hypotheses(context: ProtectedContext) -> list[BlackboardHypothesis]:
    projected: list[BlackboardHypothesis] = []
    investigation = context.investigation_state
    if investigation is not None:
        projected.extend(
            BlackboardHypothesis(
                hypothesis_id=item.hypothesis_id,
                text=item.statement,
                state={
                    "candidate": "active",
                    "rejected": "rejected",
                    "supported": "confirmed",
                }[item.state],
                reason=item.reason,
                provenance_root_ids=list(item.evidence_ids),
            )
            for item in investigation.hypotheses
        )
    memory = context.working_memory
    records: list[HypothesisRecord] = []
    if memory is not None:
        records.extend(memory.snapshot.all_hypotheses())
    snapshot = context.active_snapshot
    if snapshot is not None:
        records.extend(
            [
                *snapshot.active_hypotheses,
                *snapshot.rejected_hypotheses,
                *snapshot.confirmed_hypotheses,
            ]
        )
    projected.extend(
        BlackboardHypothesis(
            hypothesis_id=item.hypothesis_id,
            text=item.text,
            state=item.state,
            reason=item.reason,
            provenance_root_ids=sorted({source.ref_id for source in item.sources}),
        )
        for item in records
    )
    return _unique_by_id(projected, "hypothesis_id")


def _reproduction_state(
    evidence: list[DeterministicEvidence],
) -> Literal["unknown", "reproduced", "not_reproduced"]:
    baseline_tests = [
        item
        for item in evidence
        if isinstance(item, SystemTestEvidence) and item.timing == "baseline"
    ]
    if any(item.exit_code != 0 for item in baseline_tests):
        return "reproduced"
    if baseline_tests and all(item.exit_code == 0 for item in baseline_tests):
        return "not_reproduced"
    return "unknown"


def _unique_by_id(items: list, field_name: str) -> list:
    unique = {}
    for item in items:
        unique.setdefault(getattr(item, field_name), item)
    return list(unique.values())


def _unique_text(items: list[ProvenancedText]) -> list[ProvenancedText]:
    unique: dict[str, ProvenancedText] = {}
    for item in items:
        unique.setdefault(item.text.strip(), item)
    return list(unique.values())


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
