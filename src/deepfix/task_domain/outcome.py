from __future__ import annotations

from typing import Literal

from pydantic import Field

from deepfix.compaction.models import StrictModel


class NarrativeEvidenceCandidate(StrictModel):
    source: str = Field(min_length=1)
    observation: str = Field(min_length=1)


class RepairOutcomeCandidate(StrictModel):
    """Model-authored narrative proposal; never an evidence authority."""

    status: Literal["needs_input", "completed", "blocked"]
    resolution: Literal["fixed", "not_reproduced"] | None = None
    question: str | None = None
    summary: str = Field(min_length=1)
    review_summary: str | None = None
    hypothesis_candidates: list[str] = Field(default_factory=list)
    plan_candidates: list[str] = Field(default_factory=list)
    evidence_candidates: list[NarrativeEvidenceCandidate] = Field(default_factory=list)
    residual_risks: list[str] = Field(default_factory=list)
    unverified_items: list[str] = Field(default_factory=list)
