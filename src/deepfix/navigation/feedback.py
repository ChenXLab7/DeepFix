"""Read-only projections of trusted state for Todo navigation reminders."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from deepfix.compaction.models import FileChangeEvidence, SystemTestEvidence
from deepfix.compaction.store import CompactionStore
from deepfix.investigation.store import InvestigationStore
from deepfix.verification import (
    VerificationPolicy,
    VerificationPolicyStore,
    evaluate_required_oracles,
)


class NavigationFeedback(BaseModel):
    """Transient deterministic navigation advice derived from trusted state."""

    model_config = ConfigDict(frozen=True)

    milestone_ids: tuple[str, ...] = ()
    lines: tuple[str, ...] = ()
    fingerprint: str


class NavigationFeedbackSource(Protocol):
    """Source of transient navigation feedback for a task."""

    def build(self, task_id: str) -> NavigationFeedback:
        raise NotImplementedError


class LegacyNavigationFeedbackSource:
    """Project milestones from legacy stores without changing their state."""

    def __init__(
        self,
        investigation_store: InvestigationStore,
        evidence_store: CompactionStore,
        verification_store: VerificationPolicyStore,
    ) -> None:
        self._investigation_store = investigation_store
        self._evidence_store = evidence_store
        self._verification_store = verification_store

    def build(self, task_id: str) -> NavigationFeedback:
        """Read current stores once and return stable, non-persisted advice."""

        state = self._investigation_store.load(task_id)
        evidence = self._evidence_store.list_evidence(task_id)
        policy = self._verification_store.load(task_id)

        milestones: dict[str, str] = {}
        if state is not None:
            for hypothesis_id in sorted(set(state.supported_hypothesis_ids)):
                milestone_id = f"supported-hypothesis:{hypothesis_id}"
                milestones[milestone_id] = f"Supported hypothesis: {hypothesis_id}."
            for gap_id in sorted(set(state.closed_evidence_gap_ids)):
                milestone_id = f"closed-evidence-gap:{gap_id}"
                milestones[milestone_id] = f"Closed evidence gap: {gap_id}."

        successful_paths = sorted(
            {
                item.path
                for item in evidence
                if isinstance(item, FileChangeEvidence) and item.status == "succeeded"
            }
        )
        tests = [item for item in evidence if isinstance(item, SystemTestEvidence)]
        evaluation = evaluate_required_oracles(policy, tests) if policy is not None else None
        latest_successful_change = max(
            (
                index
                for index, item in enumerate(evidence)
                if isinstance(item, FileChangeEvidence) and item.status == "succeeded"
            ),
            default=None,
        )
        current_verification = evaluation is not None and evaluation.fixed_allowed and (
            latest_successful_change is None
            or _required_oracles_verified_after(
                policy,
                evidence,
                latest_successful_change,
            )
        )

        if successful_paths and not current_verification:
            milestones["verification-pending"] = (
                "Changed files: "
                + ", ".join(successful_paths)
                + "; required verification is pending."
            )
        if current_verification:
            passed_ids = ", ".join(sorted(evaluation.passed_required_oracle_ids))
            milestones["required-oracles-satisfied"] = (
                f"Required oracles satisfied: {passed_ids}."
            )
        if (
            not successful_paths
            and policy is not None
            and _has_matching_user_baseline_pass(policy, tests)
        ):
            milestones["baseline-not-reproduced"] = (
                "User-specified baseline verification passed before any code change."
            )

        milestone_ids = tuple(sorted(milestones))
        return NavigationFeedback(
            milestone_ids=milestone_ids,
            lines=tuple(milestones[milestone_id] for milestone_id in milestone_ids),
            fingerprint=_fingerprint(milestone_ids),
        )


def _has_matching_user_baseline_pass(
    policy: VerificationPolicy,
    tests: list[SystemTestEvidence],
) -> bool:
    """Whether a user-required command passed at baseline with no matching failure."""

    for oracle in policy.required_oracles:
        if oracle.origin != "user_specified":
            continue
        matching = [
            item
            for item in tests
            if item.origin == "user_specified"
            and item.timing == "baseline"
            and _normalize_command(item.command) == _normalize_command(oracle.command)
        ]
        if matching and all(item.exit_code == oracle.expected_exit_code for item in matching):
            return True
    return False


def _required_oracles_verified_after(
    policy: VerificationPolicy,
    evidence: list[object],
    latest_successful_change: int,
) -> bool:
    """Require every oracle's expected result after the latest successful change."""

    for oracle in policy.required_oracles:
        if not any(
            isinstance(item, SystemTestEvidence)
            and index > latest_successful_change
            and item.exit_code == oracle.expected_exit_code
            and _normalize_command(item.command) == _normalize_command(oracle.command)
            and _timing_satisfies(item.timing, oracle.required_timing)
            for index, item in enumerate(evidence)
        ):
            return False
    return bool(policy.required_oracles)


def _timing_satisfies(actual: str, required: str) -> bool:
    return actual == "baseline" if required == "baseline" else actual in {
        "post_change",
        "post_recovery",
    }


def _normalize_command(command: str) -> str:
    """Compare commands without relying on verification's private helper."""

    return " ".join(command.strip().replace("\\", "/").split()).lower()


def _fingerprint(milestone_ids: tuple[str, ...]) -> str:
    serialized = json.dumps(
        milestone_ids,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
