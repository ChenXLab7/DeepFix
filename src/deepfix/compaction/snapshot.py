from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from deepfix.compaction.errors import SnapshotBuildError
from deepfix.compaction.identity import (
    stable_claim_id,
    stable_hypothesis_id,
    stable_reopened_hypothesis_id,
)
from deepfix.compaction.models import (
    ArtifactReference,
    CompactionDelta,
    CompactionFailureRecord,
    CompactionSnapshot,
    ConflictCandidate,
    ConflictRecord,
    DeterministicEvidenceBlock,
    ExperimentRecord,
    FactCandidate,
    HypothesisRecord,
    HypothesisTransition,
    ProvenancedClaim,
    ProvenancedText,
    ProvenanceRef,
    SnapshotCoverage,
    TaskAnchor,
    UserConstraint,
    UserConstraintCandidate,
    WorkUnit,
)
from deepfix.memory import WorkingMemoryVersion

DELTA_EXTRACTION_PROMPT = """You extract evidence-backed candidates from complete work units.
Return only the CompactionDelta schema. Every semantic item must cite a supplied work_unit
or user_message ID. Outputs are candidates: never invent task IDs, canonical claim or
hypothesis IDs, deterministic test/file/approval/research evidence, paths, versions, or hashes.
Do not repeat facts that are absent from the supplied units.
"""


@dataclass(frozen=True)
class BuildSnapshotInput:
    task_id: str
    previous_snapshot: CompactionSnapshot | None
    compressed_units: tuple[WorkUnit, ...]
    latest_memory: WorkingMemoryVersion | None
    task_anchor: TaskAnchor
    deterministic_evidence: DeterministicEvidenceBlock
    delta: CompactionDelta
    artifact_reference: ArtifactReference
    input_hash: str


class CompactionDeltaGenerator:
    def generate(
        self,
        model: Any,
        work_units: Sequence[WorkUnit],
    ) -> CompactionDelta:
        structured_model = model.with_structured_output(CompactionDelta)
        result = structured_model.invoke(_delta_messages(work_units))
        return CompactionDelta.model_validate(result)

    async def agenerate(
        self,
        model: Any,
        work_units: Sequence[WorkUnit],
    ) -> CompactionDelta:
        structured_model = model.with_structured_output(CompactionDelta)
        result = await structured_model.ainvoke(_delta_messages(work_units))
        return CompactionDelta.model_validate(result)


class CompactionSnapshotBuilder:
    def build(self, value: BuildSnapshotInput) -> CompactionSnapshot:
        self._validate_input(value)
        version = (value.previous_snapshot.version if value.previous_snapshot else 0) + 1
        facts = self._merge_facts(value)
        hypotheses = self._merge_hypotheses(value, version)
        constraints = self._merge_constraints(value)
        experiments = self._merge_experiments(value)
        conflicts = self._merge_conflicts(value)
        conflicts.extend(self._system_fact_conflicts(value, facts))
        unresolved = self._merge_provenanced_text(
            value,
            "unresolved_questions",
        )
        next_steps = self._merge_provenanced_text(value, "next_steps")
        artifacts = _merge_artifacts(
            value.previous_snapshot.artifact_references
            if value.previous_snapshot
            else [],
            [value.artifact_reference],
        )
        payload: dict[str, Any] = {
            "task_id": value.task_id,
            "version": version,
            "previous_version": (
                value.previous_snapshot.version if value.previous_snapshot else None
            ),
            "lifecycle": "prepared",
            "created_at": _utc_now(),
            "activated_at": None,
            "abandoned_at": None,
            "abandon_reason": None,
            "source_work_unit_ids": _ordered_unique(
                [
                    *(
                        value.previous_snapshot.source_work_unit_ids
                        if value.previous_snapshot
                        else []
                    ),
                    *(unit.unit_id for unit in value.compressed_units),
                ]
            ),
            "coverage": SnapshotCoverage(
                last_user_message_id=value.task_anchor.latest_user_message_id,
                covered_message_ids=_ordered_unique(
                    [
                        message_id
                        for unit in value.compressed_units
                        for message_id in unit.message_ids
                    ]
                ),
                covered_work_unit_ids=_ordered_unique(
                    [unit.unit_id for unit in value.compressed_units]
                ),
            ),
            "task_goal": value.task_anchor.task_goal,
            "user_constraints": constraints,
            "confirmed_facts": facts,
            "deterministic_evidence": value.deterministic_evidence,
            "active_hypotheses": [
                item for item in hypotheses.values() if item.state == "active"
            ],
            "rejected_hypotheses": [
                item for item in hypotheses.values() if item.state == "rejected"
            ],
            "confirmed_hypotheses": [
                item for item in hypotheses.values() if item.state == "confirmed"
            ],
            "changed_files": value.deterministic_evidence.files,
            "experiments": experiments,
            "test_results": value.deterministic_evidence.tests,
            "conflicts": _deduplicate_by_id(conflicts, "conflict_id"),
            "unresolved_questions": unresolved,
            "next_steps": next_steps,
            "artifact_references": artifacts,
        }
        payload["content_hash"] = _semantic_hash(payload)
        return CompactionSnapshot.model_validate(payload)

    def _validate_input(self, value: BuildSnapshotInput) -> None:
        if value.task_id != value.task_anchor.task_id:
            raise _build_error(value, "task_id_mismatch")
        if not _is_hash(value.input_hash):
            raise _build_error(value, "invalid_input_hash")
        if not _is_hash(value.artifact_reference.content_hash):
            raise _build_error(value, "invalid_artifact_hash")
        unit_ids = {unit.unit_id for unit in value.compressed_units}
        if any(unit.state != "complete" for unit in value.compressed_units):
            raise _build_error(value, "incomplete_work_unit")
        if not unit_ids <= set(value.artifact_reference.work_unit_ids):
            raise _build_error(value, "artifact_missing_work_unit")
        previous = value.previous_snapshot
        if previous is not None:
            if previous.task_id != value.task_id or previous.lifecycle != "active":
                raise _build_error(value, "invalid_previous_snapshot")
            if snapshot_content_hash(previous) != previous.content_hash:
                raise _build_error(value, "previous_content_hash_mismatch")
        self._validate_delta_sources(value)

    def _validate_delta_sources(self, value: BuildSnapshotInput) -> None:
        work_unit_ids = {unit.unit_id for unit in value.compressed_units}
        message_ids = {
            message_id
            for unit in value.compressed_units
            for message_id in unit.message_ids
        }
        candidates: list[Iterable[ProvenanceRef]] = [
            item.sources for item in value.delta.confirmed_fact_candidates
        ]
        candidates.extend(item.sources for item in value.delta.hypothesis_transitions)
        candidates.extend(item.sources for item in value.delta.experiments)
        candidates.extend(
            alternative.sources
            for conflict in value.delta.conflict_candidates
            for alternative in conflict.alternatives
        )
        candidates.extend(item.sources for item in value.delta.unresolved_questions)
        candidates.extend(item.sources for item in value.delta.next_steps)
        for sources in candidates:
            sources = list(sources)
            if not sources:
                raise _build_error(value, "missing_candidate_provenance")
            for source in sources:
                if source.kind == "work_unit" and source.ref_id in work_unit_ids:
                    continue
                if source.kind == "user_message" and source.ref_id in message_ids:
                    continue
                raise _build_error(value, "invalid_candidate_provenance")
        for candidate in value.delta.user_constraint_candidates:
            if candidate.source_user_message_id != value.task_anchor.latest_user_message_id:
                raise _build_error(value, "invalid_user_message_source")
            if candidate.source_user_message_id not in message_ids:
                raise _build_error(value, "invalid_user_message_source")

    def _merge_constraints(
        self,
        value: BuildSnapshotInput,
    ) -> list[UserConstraint]:
        constraints = {
            item.constraint_id: item for item in value.task_anchor.user_constraints
        }
        for candidate in value.delta.user_constraint_candidates:
            try:
                canonical = _constraint_from_candidate(
                    value.task_id, candidate, constraints
                )
            except ValueError as exc:
                raise _build_error(value, "invalid_constraint_transition") from exc
            constraints[canonical.constraint_id] = canonical
        return list(constraints.values())

    def _merge_facts(self, value: BuildSnapshotInput) -> list[ProvenancedClaim]:
        claims: dict[str, ProvenancedClaim] = {}
        if value.previous_snapshot:
            claims.update(
                {item.claim_id: item for item in value.previous_snapshot.confirmed_facts}
            )
        for candidate in value.delta.confirmed_fact_candidates:
            _upsert_claim(value.task_id, claims, candidate)
        if value.latest_memory:
            for memory_claim in value.latest_memory.snapshot.facts:
                _upsert_claim(value.task_id, claims, memory_claim)
        return list(claims.values())

    def _merge_hypotheses(
        self,
        value: BuildSnapshotInput,
        version: int,
    ) -> dict[str, HypothesisRecord]:
        hypotheses: dict[str, HypothesisRecord] = {}
        if value.previous_snapshot:
            hypotheses.update(
                {
                    item.hypothesis_id: item
                    for item in [
                        *value.previous_snapshot.active_hypotheses,
                        *value.previous_snapshot.rejected_hypotheses,
                        *value.previous_snapshot.confirmed_hypotheses,
                    ]
                }
            )
        for transition in value.delta.hypothesis_transitions:
            _apply_hypothesis_transition(
                value,
                hypotheses,
                transition,
                version,
            )
        if value.latest_memory:
            for item in value.latest_memory.snapshot.all_hypotheses():
                hypotheses[item.hypothesis_id] = item
        return hypotheses

    def _merge_experiments(
        self,
        value: BuildSnapshotInput,
    ) -> list[ExperimentRecord]:
        experiments = {
            item.experiment_id: item
            for item in (
                value.previous_snapshot.experiments if value.previous_snapshot else []
            )
        }
        for item in value.delta.experiments:
            canonical_id = _stable_id(
                "experiment",
                value.task_id,
                item.purpose,
                item.action,
                item.result,
            )
            experiments[canonical_id] = item.model_copy(
                update={"experiment_id": canonical_id}
            )
        if value.latest_memory:
            source = ProvenanceRef(
                kind="working_memory",
                ref_id=f"working-memory-{value.latest_memory.version}",
            )
            for text in value.latest_memory.snapshot.experiments:
                canonical_id = _stable_id("experiment", value.task_id, text)
                experiments[canonical_id] = ExperimentRecord(
                    experiment_id=canonical_id,
                    purpose=text,
                    action=text,
                    result=text,
                    sources=[source],
                )
        return list(experiments.values())

    def _merge_conflicts(self, value: BuildSnapshotInput) -> list[ConflictRecord]:
        conflicts = list(
            value.previous_snapshot.conflicts if value.previous_snapshot else []
        )
        conflicts.extend(_conflict_from_candidate(value.task_id, item) for item in value.delta.conflict_candidates)
        return conflicts

    def _system_fact_conflicts(
        self,
        value: BuildSnapshotInput,
        facts: list[ProvenancedClaim],
    ) -> list[ConflictRecord]:
        conflicts: list[ConflictRecord] = []
        for index, fact in enumerate(facts):
            for test in value.deterministic_evidence.tests:
                if not _contradicts_test(fact.text, test.exit_code):
                    continue
                fact = fact.model_copy(update={"state": "conflict"})
                facts[index] = fact
                evidence_source = ProvenanceRef(
                    kind="system_evidence",
                    ref_id=test.evidence_id,
                )
                conflicts.append(
                    ConflictRecord(
                        conflict_id=_stable_id(
                            "conflict", value.task_id, fact.claim_id, test.evidence_id
                        ),
                        information_type="test",
                        subject=test.command,
                        alternatives=[
                            ProvenancedText(text=fact.text, sources=fact.sources),
                            ProvenancedText(
                                text=f"exit_code={test.exit_code}: {test.summary}",
                                sources=[evidence_source],
                            ),
                        ],
                        source_of_truth=evidence_source,
                        resolution="source_of_truth_applied",
                    )
                )
        return conflicts

    def _merge_provenanced_text(
        self,
        value: BuildSnapshotInput,
        field_name: str,
    ) -> list[ProvenancedText]:
        prior = (
            list(getattr(value.previous_snapshot, field_name))
            if value.previous_snapshot
            else []
        )
        incoming = list(getattr(value.delta, field_name))
        if value.latest_memory:
            source = ProvenanceRef(
                kind="working_memory",
                ref_id=f"working-memory-{value.latest_memory.version}",
            )
            incoming.extend(
                ProvenancedText(text=text, sources=[source])
                for text in getattr(value.latest_memory.snapshot, field_name)
            )
        merged: dict[str, ProvenancedText] = {item.text: item for item in prior}
        for item in incoming:
            existing = merged.get(item.text)
            merged[item.text] = (
                item
                if existing is None
                else existing.model_copy(
                    update={"sources": _merge_sources(existing.sources, item.sources)}
                )
            )
        return list(merged.values())


def snapshot_content_hash(snapshot: CompactionSnapshot) -> str:
    return _semantic_hash(snapshot.model_dump(mode="json"))


def _delta_messages(work_units: Sequence[WorkUnit]) -> list[Any]:
    body = json.dumps(
        [unit.model_dump(mode="json") for unit in work_units],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        SystemMessage(content=DELTA_EXTRACTION_PROMPT),
        HumanMessage(content=body),
    ]


def _constraint_from_candidate(
    task_id: str,
    candidate: UserConstraintCandidate,
    existing: dict[str, UserConstraint],
) -> UserConstraint:
    if candidate.requested_state == "revoked":
        superseded = existing.get(candidate.supersedes_constraint_id or "")
        if superseded is None or superseded.state != "active":
            raise ValueError("revoked constraint 必须引用 active constraint")
    constraint_id = _stable_id(
        "constraint",
        task_id,
        candidate.source_user_message_id,
        candidate.text,
        candidate.requested_state,
    )
    return UserConstraint(
        constraint_id=constraint_id,
        text=candidate.text,
        state=candidate.requested_state,
        source_user_message_id=candidate.source_user_message_id,
        supersedes_constraint_id=candidate.supersedes_constraint_id,
    )


def _upsert_claim(
    task_id: str,
    claims: dict[str, ProvenancedClaim],
    candidate: FactCandidate | ProvenancedClaim,
) -> None:
    claim_id = stable_claim_id(task_id, candidate.text)
    existing = claims.get(claim_id)
    if existing is None:
        claims[claim_id] = ProvenancedClaim(
            claim_id=claim_id,
            text=candidate.text,
            sources=candidate.sources,
            state=getattr(candidate, "state", "confirmed"),
        )
    else:
        claims[claim_id] = existing.model_copy(
            update={"sources": _merge_sources(existing.sources, candidate.sources)}
        )


def _apply_hypothesis_transition(
    value: BuildSnapshotInput,
    hypotheses: dict[str, HypothesisRecord],
    transition: HypothesisTransition,
    version: int,
) -> None:
    if transition.hypothesis_id and transition.reopens_hypothesis_id:
        raise _build_error(value, "ambiguous_hypothesis_identity")
    if transition.reopens_hypothesis_id:
        previous = hypotheses.get(transition.reopens_hypothesis_id)
        if (
            previous is None
            or previous.state != "rejected"
            or transition.target_state != "active"
            or not transition.reason
            or not transition.sources
        ):
            raise _build_error(value, "invalid_hypothesis_reopen")
        new_id = stable_reopened_hypothesis_id(
            value.task_id,
            previous.hypothesis_id,
            transition.sources[0].ref_id,
            transition.text,
        )
        hypotheses[new_id] = HypothesisRecord(
            hypothesis_id=new_id,
            text=transition.text,
            state="active",
            reason=transition.reason,
            reopens_hypothesis_id=previous.hypothesis_id,
            sources=transition.sources,
            updated_in_version=version,
        )
        return
    if transition.hypothesis_id:
        previous = hypotheses.get(transition.hypothesis_id)
        if previous is None or previous.state != "active":
            raise _build_error(value, "invalid_hypothesis_transition")
        if transition.target_state in {"rejected", "confirmed"} and not transition.reason:
            raise _build_error(value, "missing_hypothesis_reason")
        hypotheses[previous.hypothesis_id] = previous.model_copy(
            update={
                "text": transition.text,
                "state": transition.target_state,
                "reason": transition.reason,
                "sources": _merge_sources(previous.sources, transition.sources),
                "updated_in_version": version,
            }
        )
        return
    if transition.target_state != "active" or not transition.sources:
        raise _build_error(value, "invalid_new_hypothesis")
    hypothesis_id = stable_hypothesis_id(
        value.task_id,
        transition.sources[0].ref_id,
        transition.text,
    )
    hypotheses[hypothesis_id] = HypothesisRecord(
        hypothesis_id=hypothesis_id,
        text=transition.text,
        state="active",
        reason=transition.reason,
        sources=transition.sources,
        updated_in_version=version,
    )


def _conflict_from_candidate(
    task_id: str,
    candidate: ConflictCandidate,
) -> ConflictRecord:
    return ConflictRecord(
        conflict_id=_stable_id(
            "conflict",
            task_id,
            candidate.information_type,
            candidate.subject,
            *(item.text for item in candidate.alternatives),
        ),
        information_type=candidate.information_type,
        subject=candidate.subject,
        alternatives=candidate.alternatives,
        source_of_truth=None,
        resolution="unresolved_semantic",
    )


def _contradicts_test(text: str, exit_code: int) -> bool:
    normalized = re.sub(r"\s+", " ", text.casefold())
    says_pass = any(
        marker in normalized
        for marker in ("pytest passed", "tests passed", "test passed", "测试通过", "exit_code=0")
    )
    says_fail = any(
        marker in normalized
        for marker in ("pytest failed", "tests failed", "test failed", "测试失败")
    )
    return (exit_code != 0 and says_pass) or (exit_code == 0 and says_fail)


def _merge_sources(
    previous: Sequence[ProvenanceRef],
    current: Sequence[ProvenanceRef],
) -> list[ProvenanceRef]:
    merged = {(item.kind, item.ref_id): item for item in [*previous, *current]}
    return list(merged.values())


def _merge_artifacts(
    previous: Sequence[ArtifactReference],
    current: Sequence[ArtifactReference],
) -> list[ArtifactReference]:
    merged = {(item.path, item.content_hash): item for item in [*previous, *current]}
    return list(merged.values())


def _deduplicate_by_id(items: Sequence[Any], field_name: str) -> list[Any]:
    merged = {str(getattr(item, field_name)): item for item in items}
    return list(merged.values())


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    for field in (
        "lifecycle",
        "created_at",
        "activated_at",
        "abandoned_at",
        "abandon_reason",
        "content_hash",
    ):
        semantic.pop(field, None)
    encoded = json.dumps(
        semantic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def _is_hash(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _stable_id(prefix: str, *parts: str) -> str:
    material = "|".join((f"{prefix}:v1", *parts))
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _build_error(value: BuildSnapshotInput, error_code: str) -> SnapshotBuildError:
    failure = CompactionFailureRecord(
        attempt_id=_stable_id("attempt", value.task_id, value.input_hash),
        task_id=value.task_id,
        entrypoint="automatic",
        budget_zone="normal_compaction",
        stage="snapshot_validate",
        error_code=error_code,
        input_hash=value.input_hash,
        original_messages_preserved=True,
        artifact_reference=value.artifact_reference.path,
        prepared_snapshot_version=None,
        recorded_at=_utc_now(),
    )
    return SnapshotBuildError(failure)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
