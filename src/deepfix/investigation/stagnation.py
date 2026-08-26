from __future__ import annotations

from collections.abc import Sequence

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
