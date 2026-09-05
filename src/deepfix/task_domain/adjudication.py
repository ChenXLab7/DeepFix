from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field

from deepfix.compaction.models import StrictModel, SystemTestEvidence
from deepfix.domain_repositories.evidence import VerificationEvidenceView
from deepfix.domain_repositories.execution import ExecutionIntegrity
from deepfix.task_domain.models import AdjudicationDecision, TaskDefinition, TaskLifecycle
from deepfix.verification import VerificationOracle, VerificationPolicy, evaluate_required_oracles


class OutcomeAdjudicationInput(StrictModel):
    definition: TaskDefinition
    lifecycle: TaskLifecycle
    verification_policy: VerificationPolicy | None
    verification: VerificationEvidenceView
    execution_integrity: ExecutionIntegrity
    successful_change_evidence_ids: list[str] = Field(default_factory=list)
    scope_violation_evidence_ids: list[str] = Field(default_factory=list)
    reproduction_state: Literal["unknown", "reproduced", "not_reproduced"]


class OutcomeAssessment(StrictModel):
    outcome: Literal["fixed", "not_reproduced", "continue", "review", "paused"]
    decision: AdjudicationDecision | None = None
    reason: str = Field(min_length=1)
    blocking_evidence_ids: list[str] = Field(default_factory=list)


class OutcomeAdjudicator:
    """Apply deterministic final-outcome rules without persistence or model calls."""

    def decide(self, value: OutcomeAdjudicationInput) -> OutcomeAssessment:
        self._validate_task_identity(value)
        incomplete = list(dict.fromkeys(value.execution_integrity.incomplete_operation_ids))
        if incomplete:
            return OutcomeAssessment(
                outcome="paused",
                decision=self._decision(value, "paused", [], incomplete),
                reason="execution integrity contains incomplete operations",
            )

        scope_violations = list(dict.fromkeys(value.scope_violation_evidence_ids))
        if scope_violations:
            return OutcomeAssessment(
                outcome="review",
                reason="workspace scope violation requires review",
                blocking_evidence_ids=scope_violations,
            )

        real_change_ids = self._successful_change_ids(value)
        if value.verification_policy is not None:
            oracle = evaluate_required_oracles(
                value.verification_policy,
                value.verification.test_evidence,
            )
            if oracle.conflicting_evidence_ids:
                return OutcomeAssessment(
                    outcome="review",
                    reason="higher-authority verification conflicts with completion",
                    blocking_evidence_ids=oracle.conflicting_evidence_ids,
                )
            if oracle.fixed_allowed and real_change_ids:
                required_test_ids = _required_test_evidence_ids(
                    value.verification_policy,
                    value.verification.test_evidence,
                )
                evidence_ids = list(dict.fromkeys([*real_change_ids, *required_test_ids]))
                return OutcomeAssessment(
                    outcome="fixed",
                    decision=self._decision(value, "fixed", evidence_ids, []),
                    reason="all required verification passed after a successful change",
                )

        if value.reproduction_state == "not_reproduced" and not real_change_ids:
            baseline_ids = _passing_user_baseline_ids(
                value.verification_policy,
                value.verification.test_evidence,
            )
            if baseline_ids:
                return OutcomeAssessment(
                    outcome="not_reproduced",
                    decision=self._decision(
                        value,
                        "not_reproduced",
                        baseline_ids,
                        [],
                    ),
                    reason="user-specified baseline verification passed without a change",
                )

        return OutcomeAssessment(
            outcome="continue",
            reason="required verification is incomplete",
        )

    @staticmethod
    def _validate_task_identity(value: OutcomeAdjudicationInput) -> None:
        task_ids = {
            value.definition.task_id,
            value.lifecycle.task_id,
            *(
                [value.verification_policy.task_id]
                if value.verification_policy is not None
                else []
            ),
        }
        if len(task_ids) != 1:
            raise ValueError("adjudication inputs belong to different tasks")

    @staticmethod
    def _successful_change_ids(value: OutcomeAdjudicationInput) -> list[str]:
        requested = set(value.successful_change_evidence_ids)
        return [
            item.evidence_id
            for item in value.verification.file_change_evidence
            if item.evidence_id in requested and item.status == "succeeded"
        ]

    @staticmethod
    def _decision(
        value: OutcomeAdjudicationInput,
        outcome: Literal["fixed", "not_reproduced", "paused"],
        evidence_ids: list[str],
        operation_ids: list[str],
    ) -> AdjudicationDecision:
        semantic = json.dumps(
            {
                "task_id": value.definition.task_id,
                "lifecycle_version": value.lifecycle.version,
                "outcome": outcome,
                "evidence_ids": evidence_ids,
                "operation_ids": operation_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return AdjudicationDecision(
            decision_id="adjudication_"
            + hashlib.sha256(semantic.encode("utf-8")).hexdigest()[:32],
            task_id=value.definition.task_id,
            outcome=outcome,
            evidence_ids=evidence_ids,
            operation_ids=operation_ids,
            decided_at=value.lifecycle.updated_at,
        )


def _required_test_evidence_ids(
    policy: VerificationPolicy,
    evidence: list[SystemTestEvidence],
) -> list[str]:
    selected: list[str] = []
    for oracle in policy.required_oracles:
        matches = [
            item
            for item in evidence
            if _matches_oracle(item, oracle)
            and _timing_satisfies(item.timing, oracle.required_timing)
        ]
        if matches and matches[-1].exit_code == oracle.expected_exit_code:
            selected.append(matches[-1].evidence_id)
    return selected


def _passing_user_baseline_ids(
    policy: VerificationPolicy | None,
    evidence: list[SystemTestEvidence],
) -> list[str]:
    if policy is None:
        return []
    selected: list[str] = []
    user_oracles = [
        item for item in policy.required_oracles if item.origin == "user_specified"
    ]
    for oracle in user_oracles:
        matches = [
            item
            for item in evidence
            if _matches_oracle(item, oracle) and item.timing == "baseline"
        ]
        if not matches or matches[-1].exit_code != oracle.expected_exit_code:
            return []
        selected.append(matches[-1].evidence_id)
    return selected if user_oracles else []


def _matches_oracle(item: SystemTestEvidence, oracle: VerificationOracle) -> bool:
    return item.origin == oracle.origin and _normalize_command(
        item.command
    ) == _normalize_command(oracle.command)


def _timing_satisfies(actual: str, required: str) -> bool:
    if required == "baseline":
        return actual == "baseline"
    return actual in {"post_change", "post_recovery"}


def _normalize_command(command: str) -> str:
    return " ".join(command.strip().replace("\\", "/").split()).lower()
