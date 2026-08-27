from __future__ import annotations

import json
from collections.abc import Callable

from deepfix.investigation.experiments import ExperimentAssessment, ExperimentResult
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import (
    ExperimentClaimRecord,
    InvestigationState,
)
from deepfix.investigation.store import InvestigationStore


class InvestigationStateReducer:
    def __init__(
        self,
        task_id: str,
        store: InvestigationStore,
        *,
        provenance_root_resolver: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.task_id = task_id.strip()
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        self.store = store
        self._resolve_roots = provenance_root_resolver or (lambda value: [value])

    def commit(
        self,
        expected_version: int,
        result: ExperimentResult,
        assessment: ExperimentAssessment,
    ) -> InvestigationState:
        if result.experiment_id != assessment.experiment_id:
            raise ValueError("experiment result/assessment mismatch")
        current = self.store.load(self.task_id) or InvestigationState.new(self.task_id)
        next_state = self._reduce(current, result, assessment)
        event_payload = _canonical_payload(result, assessment)
        event_id = stable_investigation_id(
            "experiment-event",
            self.task_id,
            result.experiment_id,
            event_payload,
        )
        return self.store.commit_experiment(
            expected_version,
            event_id=event_id,
            event_payload=event_payload,
            result=result,
            assessment=assessment,
            next_state=next_state,
        )

    def _reduce(
        self,
        state: InvestigationState,
        result: ExperimentResult,
        assessment: ExperimentAssessment,
    ) -> InvestigationState:
        claims = {item.claim_id: item for item in state.experiment_claims}
        for claim in assessment.accepted_claims:
            root_ids = sorted(
                {
                    root_id
                    for source in claim.sources
                    for root_id in self._resolve_roots(source.ref_id)
                }
            )
            claims[claim.claim_id] = ExperimentClaimRecord(
                claim_id=claim.claim_id,
                text=claim.text,
                state=claim.state,
                sources=claim.sources,
                provenance_root_ids=root_ids,
            )
        hypothesis_updates = []
        supported = set(assessment.supported_hypothesis_ids)
        rejected = set(assessment.rejected_hypothesis_ids)
        for hypothesis in state.hypotheses:
            if hypothesis.hypothesis_id in supported:
                hypothesis_updates.append(
                    hypothesis.model_copy(
                        update={
                            "state": "supported",
                            "reason": "supported by Experiment assessment",
                        }
                    )
                )
            elif hypothesis.hypothesis_id in rejected:
                hypothesis_updates.append(
                    hypothesis.model_copy(
                        update={
                            "state": "rejected",
                            "reason": "rejected by Experiment assessment",
                        }
                    )
                )
            else:
                hypothesis_updates.append(hypothesis)
        evidence_ids = [
            *assessment.deterministic_evidence_ids,
            *(
                evidence_id
                for criterion in assessment.criterion_assessments
                for evidence_id in criterion.evidence_ids
            ),
        ]
        root_ids = sorted(
            {
                root_id
                for evidence_id in evidence_ids
                for root_id in self._resolve_roots(evidence_id)
            }
        )
        return state.model_copy(
            update={
                "active_experiment_id": None,
                "completed_experiment_ids": _unique(
                    [*state.completed_experiment_ids, result.experiment_id]
                ),
                "closed_evidence_gap_ids": _unique(
                    [
                        *state.closed_evidence_gap_ids,
                        *assessment.closed_evidence_gap_ids,
                    ]
                ),
                "independent_evidence_root_ids": _unique(
                    [*state.independent_evidence_root_ids, *root_ids]
                ),
                "experiment_claims": list(claims.values()),
                "hypotheses": hypothesis_updates,
                "supported_hypothesis_ids": _unique(
                    [*state.supported_hypothesis_ids, *supported]
                ),
                "test_evidence_ids": _unique(
                    [*state.test_evidence_ids, *result.test_evidence_ids]
                ),
            }
        )


def _canonical_payload(
    result: ExperimentResult,
    assessment: ExperimentAssessment,
) -> str:
    return json.dumps(
        {
            "result": result.model_dump(mode="json"),
            "assessment": assessment.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _unique(values) -> list[str]:
    return list(dict.fromkeys(values))
