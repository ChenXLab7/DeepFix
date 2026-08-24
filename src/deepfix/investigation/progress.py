from __future__ import annotations

import hashlib
from typing import ClassVar

from deepfix.investigation.models import (
    InvestigationEventType,
    InvestigationState,
    ProgressKind,
    ToolObservation,
)


class ProgressEvaluator:
    _STRONG: ClassVar[dict[InvestigationEventType, ProgressKind]] = {
        InvestigationEventType.TEST_OBSERVED: ProgressKind.TEST_EVIDENCE,
        InvestigationEventType.DECISION_EVIDENCE_OBSERVED: ProgressKind.DECISION_EVIDENCE,
        InvestigationEventType.HYPOTHESIS_SUPPORTED: ProgressKind.HYPOTHESIS_TRANSITION,
        InvestigationEventType.HYPOTHESIS_REJECTED: ProgressKind.HYPOTHESIS_TRANSITION,
        InvestigationEventType.FILE_CHANGED: ProgressKind.FILE_CHANGE,
        InvestigationEventType.POST_EDIT_TEST_OBSERVED: ProgressKind.POST_EDIT_TEST,
        InvestigationEventType.USER_INFORMATION_RECEIVED: ProgressKind.USER_INFORMATION,
    }
    _TEST_PROGRESS: ClassVar[frozenset[ProgressKind]] = frozenset({
        ProgressKind.TEST_EVIDENCE,
        ProgressKind.POST_EDIT_TEST,
    })

    def evaluate(self, observation: ToolObservation) -> ProgressKind | None:
        return self._STRONG.get(observation.event_type)

    def apply(
        self,
        state: InvestigationState,
        observation: ToolObservation,
    ) -> tuple[InvestigationState, ProgressKind | None]:
        progress = self.evaluate(observation)
        fingerprint = _progress_fingerprint(observation)
        if (
            progress in self._TEST_PROGRESS
            and fingerprint
            and fingerprint in state.seen_progress_fingerprints
        ):
            return state, None
        if progress is not None and fingerprint:
            fingerprints = [*state.seen_progress_fingerprints, fingerprint][-64:]
            state = state.model_copy(
                update={"seen_progress_fingerprints": fingerprints}
            )
        return state, progress


def _progress_fingerprint(observation: ToolObservation) -> str:
    if not observation.result_fingerprint:
        return ""
    material = f"{observation.signature}|{observation.result_fingerprint}"
    digest = hashlib.sha256(material.encode()).hexdigest()[:32]
    return f"progress_{digest}"
