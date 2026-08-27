from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from deepfix.investigation.experiments import StrategyDecision
from deepfix.investigation.models import (
    InvestigationEventType,
    InvestigationState,
    ScopeKind,
    ToolObservation,
)


def matching_cycle_size(signatures: Sequence[str]) -> int | None:
    for size in range(2, min(8, len(signatures) // 2) + 1):
        if list(signatures[-size:]) == list(signatures[-2 * size : -size]):
            return size
    return None


def progress_fingerprint(state: InvestigationState) -> str:
    evidence_backed_hypotheses = [
        {
            "hypothesis_id": item.hypothesis_id,
            "state": item.state,
            "evidence_ids": sorted(set(item.evidence_ids)),
        }
        for item in state.hypotheses
        if item.state in {"supported", "rejected"} and item.evidence_ids
    ]
    payload = {
        "closed_evidence_gap_ids": sorted(set(state.closed_evidence_gap_ids)),
        "independent_evidence_root_ids": sorted(
            set(state.independent_evidence_root_ids)
        ),
        "hypotheses": sorted(
            evidence_backed_hypotheses,
            key=lambda item: item["hypothesis_id"],
        ),
        "code_state_hash": state.code_state_hash,
        "test_evidence_ids": sorted(set(state.test_evidence_ids)),
        "user_information_ids": sorted(set(state.user_information_ids)),
    }
    return _stable_fingerprint("progress", payload)


def strategy_signature(decision: StrategyDecision) -> str:
    experiment = decision.experiment_spec
    experiment_payload = None
    if experiment is not None:
        experiment_payload = experiment.model_dump(
            mode="json",
            exclude={"experiment_id", "task_id"},
        )
    payload = {
        "decision_type": decision.decision_type,
        "selected_hypothesis_id": decision.selected_hypothesis_id,
        "evidence_gap_ids": sorted(set(decision.evidence_gap_ids)),
        "experiment": experiment_payload,
        "question_for_user": decision.question_for_user,
        "candidate_outcome": decision.candidate_outcome,
    }
    return _stable_fingerprint("strategy", payload)


class ExperimentStagnationDetector:
    def after_experiment(
        self,
        state: InvestigationState,
        decision: StrategyDecision,
        before_fingerprint: str,
    ) -> InvestigationState:
        current_fingerprint = progress_fingerprint(state)
        signature = strategy_signature(decision)
        if current_fingerprint != before_fingerprint:
            return state.model_copy(
                update={
                    "experiment_progress_fingerprint": current_fingerprint,
                    "last_strategy_signature": signature,
                    "no_progress_count": 0,
                    "reevaluation_required": False,
                    "stagnation_level": 0,
                }
            )
        repeated_strategy = signature == state.last_strategy_signature
        no_progress_count = state.no_progress_count + 1
        stalled = repeated_strategy and no_progress_count >= 2
        return state.model_copy(
            update={
                "experiment_progress_fingerprint": current_fingerprint,
                "last_strategy_signature": signature,
                "no_progress_count": no_progress_count,
                "reevaluation_required": stalled or state.reevaluation_required,
                "stagnation_level": max(state.stagnation_level, int(stalled)),
            }
        )


class StagnationDetector:
    def after_event(
        self,
        state: InvestigationState,
        observation: ToolObservation,
    ) -> InvestigationState:
        if observation.event_type is InvestigationEventType.PHASE_CHANGED:
            return state
        if observation.progress_kind is None:
            return state
        return state.model_copy(
            update={
                "progress_generation": state.progress_generation + 1,
                "no_progress_count": 0,
                "exploratory_without_progress": 0,
                "recent_tool_signatures": [],
                "seen_progress_fingerprints": state.seen_progress_fingerprints[-1:],
                "reevaluation_required": False,
                "stagnation_level": 0,
                "permit": None,
                "post_permit_review_pending": False,
                "duplicate_hypothesis_correction_ids": [],
            }
        )

    def after_tool(
        self,
        state: InvestigationState,
        observation: ToolObservation,
    ) -> InvestigationState:
        if observation.progress_kind is not None:
            return self.after_event(state, observation)
        activity_signature = _activity_signature(observation)
        signatures = list(state.recent_tool_signatures)
        if activity_signature:
            signatures = [*signatures, activity_signature][-32:]
        no_progress = state.no_progress_count + 1
        exploratory = state.exploratory_without_progress + int(
            observation.scope is ScopeKind.EXPLORATORY
        )
        repeated = bool(activity_signature) and signatures.count(activity_signature) >= 3
        cycled = matching_cycle_size(signatures) is not None
        stalled = repeated or cycled or no_progress >= 6 or exploratory >= 4
        level = state.stagnation_level
        if state.post_permit_review_pending:
            level = 2
            stalled = True
        elif stalled:
            level = max(level, 1)
        return state.model_copy(
            update={
                "recent_tool_signatures": signatures,
                "no_progress_count": no_progress,
                "exploratory_without_progress": exploratory,
                "reevaluation_required": stalled or state.reevaluation_required,
                "stagnation_level": level,
            }
        )


def _activity_signature(observation: ToolObservation) -> str:
    if not observation.signature:
        return ""
    result = observation.result_fingerprint or "unclassified-result"
    return f"{observation.signature}|{result}"


def _stable_fingerprint(prefix: str, payload: dict) -> str:
    normalized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"
