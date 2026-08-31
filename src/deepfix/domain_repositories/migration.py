from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from uuid import uuid4

from pydantic import Field

from deepfix.compaction.models import (
    ApprovalEvidence,
    ArtifactReference,
    CompactionFailureRecord,
    CompactionSnapshot,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    ExperimentRecord,
    FileChangeEvidence,
    HypothesisRecord,
    ProvenancedClaim,
    ProvenancedText,
    ProvenanceRef,
    ResearchStatusEvidence,
    SnapshotCoverage,
    StrictModel,
    SystemTestEvidence,
)
from deepfix.compaction.snapshot import snapshot_content_hash
from deepfix.database import SQLiteDatabase
from deepfix.domain_repositories.evidence import (
    EvidenceKind,
    EvidenceRepository,
    deterministic_provenance_roots,
    restore_deterministic_evidence,
    restore_external_evidence,
)
from deepfix.domain_repositories.execution import (
    ExecutionApproval,
    ExecutionRepository,
    create_execution_approval,
)
from deepfix.domain_repositories.history import (
    ContextTelemetry,
    HistoryRepository,
    HistorySnapshotRecord,
    history_record_from_snapshot,
)
from deepfix.domain_repositories.investigation import (
    InvestigationRepository,
    stable_question_id,
)
from deepfix.investigation.models import (
    InvestigationHypothesis,
    InvestigationState,
    UnresolvedQuestion,
)
from deepfix.investigation.receipts import (
    ToolExecutionReceipt,
    disable_legacy_receipt_writes,
    legacy_receipt_task_guard,
    receipt_task_segment,
)
from deepfix.operations import OperationJournalEntry
from deepfix.research.models import ExternalEvidence, ResearchQuery, SearchCandidate


class DomainMigrationReport(StrictModel):
    domain: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    source_count: int = Field(ge=0)
    target_count: int = Field(ge=0)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_mismatches: list[str] = Field(default_factory=list)
    hash_mismatches: list[str] = Field(default_factory=list)
    missing_references: list[str] = Field(default_factory=list)
    ready_to_switch: bool


class DomainMigrationInProgress(RuntimeError):
    pass


def _fenced_migration(domain: str):
    def decorate(method):
        @wraps(method)
        def wrapped(self, task_id: str):
            with self._migration_fence(domain, task_id):
                result = method(self, task_id)
                if (
                    domain == "execution"
                    and result.ready_to_switch
                    and self.artifact_root is not None
                ):
                    disable_legacy_receipt_writes(
                        self.artifact_root / "investigation_receipts",
                        task_id,
                    )
                return result

        return wrapped

    return decorate


class DomainMigrator:
    def __init__(
        self,
        database: SQLiteDatabase | str | Path,
        *,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.database = (
            database if isinstance(database, SQLiteDatabase) else SQLiteDatabase(database)
        )
        self.artifact_root = (
            Path(artifact_root).expanduser().resolve() if artifact_root is not None else None
        )
        self.evidence = EvidenceRepository(
            self.database,
            artifact_verifier=(self._verify_artifact if self.artifact_root is not None else None),
        )
        self.investigation = InvestigationRepository(self.database)
        self.execution = ExecutionRepository(self.database)
        self.history = HistoryRepository(self.database)
        self._initialize_schema()

    @_fenced_migration("working_memory")
    def migrate_working_memory(self, task_id: str) -> DomainMigrationReport:
        """One-way migration from the retired catch-all memory table."""
        task_id = _required(task_id, "task_id")
        existing = self._load_report("working_memory", task_id)
        if existing is not None:
            return existing
        rows = self._legacy_working_memory_rows(task_id)
        expected: list[tuple[str, str, str]] = []
        missing_references: list[str] = []
        existing_versions = [item.version for item in self.history.list_for_task(task_id)]
        next_version = max(existing_versions, default=0) + 1
        for ordinal, (legacy_version, payload, created_at) in enumerate(rows):
            value = json.loads(payload)
            provenance = ProvenanceRef(
                kind="snapshot_record",
                ref_id=f"legacy-memory:{legacy_version}",
            )
            facts = [_legacy_claim(item, legacy_version) for item in value.get("facts", [])]
            hypotheses = [
                _legacy_hypothesis_record(item, state, legacy_version)
                for field_name, state in (
                    ("active_hypotheses", "active"),
                    ("rejected_hypotheses", "rejected"),
                    ("confirmed_hypotheses", "confirmed"),
                )
                for item in value.get(field_name, [])
            ]
            questions = [
                ProvenancedText(text=str(item), sources=[provenance])
                for item in value.get("unresolved_questions", [])
                if str(item).strip()
            ]
            experiments = [
                ExperimentRecord(
                    experiment_id=_stable_legacy_id(
                        "experiment", task_id, str(legacy_version), str(item)
                    ),
                    purpose=str(item),
                    action=str(item),
                    result=str(item),
                    sources=[provenance],
                )
                for item in value.get("experiments", [])
                if str(item).strip()
            ]
            for claim in facts:
                self.evidence.record_semantic_candidate(
                    task_id,
                    claim,
                    provenance_root_ids=[provenance.ref_id],
                )
            for item in hypotheses:
                imported = InvestigationHypothesis(
                    hypothesis_id=item.hypothesis_id,
                    statement=item.text,
                    state={
                        "active": "candidate",
                        "rejected": "rejected",
                        "confirmed": "supported",
                    }[item.state],
                    evidence_ids=[
                        source.ref_id for source in item.sources if source.kind == "system_evidence"
                    ],
                    checked_locations=[],
                    reason=item.reason or "legacy working memory",
                    reopens_hypothesis_id=item.reopens_hypothesis_id,
                )
                missing = self.investigation.missing_current_evidence_ids(
                    task_id, imported.evidence_ids
                )
                if missing:
                    missing_references.extend(
                        f"hypothesis:{imported.hypothesis_id}:evidence:{evidence_id}"
                        for evidence_id in missing
                    )
                else:
                    self.investigation.backfill_current_hypothesis(task_id, imported)
            for item in questions:
                self.investigation.open_question(
                    UnresolvedQuestion(
                        question_id=stable_question_id(task_id, item.text),
                        task_id=task_id,
                        text=item.text,
                        status="open",
                        source_ids=[provenance.ref_id],
                        created_at=created_at,
                    )
                )
            snapshot = CompactionSnapshot(
                task_id=task_id,
                version=next_version + ordinal,
                previous_version=(next_version + ordinal - 1 if ordinal else None),
                lifecycle="prepared",
                created_at=created_at,
                source_work_unit_ids=[],
                coverage=SnapshotCoverage(),
                task_goal=str(value.get("summary", "legacy memory history")),
                user_constraints=[],
                confirmed_facts=facts,
                deterministic_evidence=DeterministicEvidenceBlock(),
                active_hypotheses=[item for item in hypotheses if item.state == "active"],
                rejected_hypotheses=[item for item in hypotheses if item.state == "rejected"],
                confirmed_hypotheses=[item for item in hypotheses if item.state == "confirmed"],
                changed_files=[],
                experiments=experiments,
                test_results=[],
                conflicts=[],
                unresolved_questions=questions,
                next_steps=[],
                artifact_references=[],
                content_hash="pending",
            )
            snapshot = snapshot.model_copy(update={"content_hash": snapshot_content_hash(snapshot)})
            record = history_record_from_snapshot(
                snapshot,
                input_hash=_canonical_hash(value),
            )
            self.history.backfill(record)
            expected.extend(
                (item.kind, item.item_id, _canonical_payload(item.payload))
                for item in record.semantic_items
                if item.kind != "next_step"
            )
        target = [
            (item.kind, item.item_id, _canonical_payload(item.payload))
            for record in self.history.list_for_task(task_id)
            if record.version >= next_version
            for item in record.semantic_items
            if item.kind != "next_step"
        ]
        source = sorted(expected)
        target = sorted(target)
        report = DomainMigrationReport(
            domain="working_memory",
            task_id=task_id,
            source_count=len(source),
            target_count=len(target),
            source_hash=_canonical_hash(source),
            target_hash=_canonical_hash(target),
            missing_references=sorted(set(missing_references)),
            ready_to_switch=source == target and not missing_references,
        )
        if report.ready_to_switch:
            self._save_report(report)
        return report

    @_fenced_migration("context_telemetry")
    def migrate_context_telemetry(self, task_id: str) -> DomainMigrationReport:
        task_id = _required(task_id, "task_id")
        existing = self._load_report("context_telemetry", task_id)
        if existing is not None:
            return existing
        legacy_payload = self._legacy_context_telemetry(task_id)
        source = [] if legacy_payload is None else [(task_id, legacy_payload)]
        if legacy_payload is not None:
            telemetry = ContextTelemetry(
                task_id=task_id,
                context_peak_tokens=int(legacy_payload.get("context_peak_tokens", 0)),
                context_overflow_count=int(legacy_payload.get("context_overflow_count", 0)),
                active_compaction_count=int(legacy_payload.get("active_compaction_count", 0)),
                latest_usage_ratio=float(legacy_payload.get("latest_usage_ratio", 0.0)),
                latest_budget_zone=legacy_payload.get("latest_budget_zone"),
                normal_compaction_count=int(legacy_payload.get("normal_compaction_count", 0)),
                emergency_compaction_count=int(legacy_payload.get("emergency_compaction_count", 0)),
                compaction_failure_count=int(legacy_payload.get("compaction_failure_count", 0)),
                normal_zone_passthrough_count=int(
                    legacy_payload.get("normal_zone_passthrough_count", 0)
                ),
                manual_compaction_error_count=int(
                    legacy_payload.get("manual_compaction_error_count", 0)
                ),
                overflow_retry_count=int(legacy_payload.get("overflow_retry_count", 0)),
                active_snapshot_version=legacy_payload.get("active_compaction_snapshot_version"),
                last_compaction_artifact=legacy_payload.get("last_compaction_artifact"),
                last_compaction_error=legacy_payload.get("last_compaction_error"),
                last_compaction_at=legacy_payload.get("last_compaction_at"),
            )
            self.history.import_context_telemetry(
                telemetry,
                event_id=f"legacy-context-telemetry:{task_id}",
            )
            normalized_source = telemetry.model_dump(mode="json")
            target = [(task_id, self.history.context_telemetry(task_id).model_dump(mode="json"))]
            source = [(task_id, normalized_source)]
        else:
            target = []
        report = DomainMigrationReport(
            domain="context_telemetry",
            task_id=task_id,
            source_count=len(source),
            target_count=len(target),
            source_hash=_canonical_hash(source),
            target_hash=_canonical_hash(target),
            identity_mismatches=[],
            hash_mismatches=[] if source == target else [task_id],
            missing_references=[],
            ready_to_switch=source == target,
        )
        if report.ready_to_switch:
            self._save_report(report)
        return report

    @_fenced_migration("deterministic_evidence")
    def migrate_deterministic_evidence(self, task_id: str) -> DomainMigrationReport:
        task_id = _required(task_id, "task_id")
        existing = self._load_report("deterministic_evidence", task_id)
        if existing is not None:
            return existing
        source = self._legacy_rows(task_id)
        for _evidence_id, kind, payload in source:
            evidence = _restore_legacy(kind, payload)
            self.evidence.record_deterministic(
                task_id,
                evidence,
                provenance_root_ids=deterministic_provenance_roots(evidence),
            )
        target = [
            item
            for item in self.evidence.list_for_task(task_id)
            if item.kind in _DETERMINISTIC_KINDS
        ]
        source_by_id = {
            evidence_id: _canonical_payload(json.loads(payload))
            for evidence_id, _kind, payload in source
        }
        target_by_id = {
            item.evidence_id: _canonical_payload(
                restore_deterministic_evidence(item).model_dump(mode="json")
            )
            for item in target
        }
        source_ids = set(source_by_id)
        target_ids = set(target_by_id)
        identity_mismatches = sorted(source_ids ^ target_ids)
        hash_mismatches = sorted(
            evidence_id
            for evidence_id in source_ids & target_ids
            if source_by_id[evidence_id] != target_by_id[evidence_id]
        )
        source_projection = [
            (evidence_id, source_by_id[evidence_id]) for evidence_id in sorted(source_by_id)
        ]
        target_projection = [
            (evidence_id, target_by_id[evidence_id]) for evidence_id in sorted(target_by_id)
        ]
        report = DomainMigrationReport(
            domain="deterministic_evidence",
            task_id=task_id,
            source_count=len(source),
            target_count=len(target),
            source_hash=_canonical_hash(source_projection),
            target_hash=_canonical_hash(target_projection),
            identity_mismatches=identity_mismatches,
            hash_mismatches=hash_mismatches,
            missing_references=[],
            ready_to_switch=(
                len(source) == len(target) and not identity_mismatches and not hash_mismatches
            ),
        )
        if report.ready_to_switch:
            self._save_report(report)
        return report

    @_fenced_migration("research")
    def migrate_research(self, task_id: str) -> DomainMigrationReport:
        task_id = _required(task_id, "task_id")
        existing = self._load_report("research", task_id)
        if existing is not None:
            return existing
        queries, candidates, external = self._legacy_research_rows(task_id)
        query_ids_by_text: dict[str, list[str]] = {}
        for query in queries:
            self.evidence.record_research_attempt(
                task_id=query.task_id,
                query_id=query.query_id,
                sanitized_query=query.sanitized_query,
                providers=query.providers,
                provider_errors=query.provider_errors,
                created_at=query.created_at,
            )
            query_ids_by_text.setdefault(query.sanitized_query, []).append(query.query_id)
        candidates_by_query: dict[str, list[SearchCandidate]] = {}
        missing_references: list[str] = []
        for candidate in candidates:
            matching = query_ids_by_text.get(candidate.query, [])
            if not matching:
                missing_references.append(
                    f"candidate:{candidate.candidate_id}:query:{candidate.query}"
                )
                continue
            candidates_by_query.setdefault(matching[-1], []).append(candidate)
        for query_id, items in candidates_by_query.items():
            self.evidence.record_research_candidates(query_id, items)
        known_candidate_ids = {item.candidate_id for item in candidates}
        for item in external:
            if item.candidate_id not in known_candidate_ids:
                missing_references.append(
                    f"evidence:{item.evidence_id}:candidate:{item.candidate_id}"
                )
                continue
            reference = self._artifact_reference(item.artifact_path)
            if reference is None:
                missing_references.append(item.artifact_path)
                continue
            self.evidence.accept_external(
                item,
                provenance_root_ids=[f"url:{item.url}"],
                artifact_references=[reference],
            )

        target_queries = [
            ResearchQuery(
                query_id=item.query_id,
                task_id=item.task_id,
                sanitized_query=item.sanitized_query,
                providers=item.providers,
                provider_errors=item.provider_errors,
                created_at=item.created_at,
            )
            for item in self.evidence.list_research_attempts(task_id)
        ]
        target_candidates = self.evidence.list_research_candidates(task_id)
        target_external = [
            restore_external_evidence(item)
            for item in self.evidence.list_for_task(task_id)
            if item.kind is EvidenceKind.EXTERNAL_RESEARCH
        ]
        source_projection = _research_projection(queries, candidates, external)
        target_projection = _research_projection(
            target_queries,
            target_candidates,
            target_external,
        )
        source_ids = {f"{kind}:{identity}" for kind, identity, _ in source_projection}
        target_ids = {f"{kind}:{identity}" for kind, identity, _ in target_projection}
        source_payloads = {
            f"{kind}:{identity}": payload for kind, identity, payload in source_projection
        }
        target_payloads = {
            f"{kind}:{identity}": payload for kind, identity, payload in target_projection
        }
        identity_mismatches = sorted(source_ids ^ target_ids)
        hash_mismatches = sorted(
            identity
            for identity in source_ids & target_ids
            if source_payloads[identity] != target_payloads[identity]
        )
        report = DomainMigrationReport(
            domain="research",
            task_id=task_id,
            source_count=len(source_projection),
            target_count=len(target_projection),
            source_hash=_canonical_hash(source_projection),
            target_hash=_canonical_hash(target_projection),
            identity_mismatches=identity_mismatches,
            hash_mismatches=hash_mismatches,
            missing_references=sorted(set(missing_references)),
            ready_to_switch=(
                len(source_projection) == len(target_projection)
                and not identity_mismatches
                and not hash_mismatches
                and not missing_references
            ),
        )
        if report.ready_to_switch:
            self._save_report(report)
        return report

    @_fenced_migration("investigation")
    def migrate_investigation(self, task_id: str) -> DomainMigrationReport:
        task_id = _required(task_id, "task_id")
        existing = self._load_report("investigation", task_id)
        if existing is not None:
            return existing

        state = self._legacy_investigation_state(task_id)
        current_hypotheses = [] if state is None else state.hypotheses
        missing_references: list[str] = []
        for hypothesis in current_hypotheses:
            missing = self.investigation.missing_current_evidence_ids(
                task_id,
                hypothesis.evidence_ids,
            )
            if missing:
                missing_references.extend(
                    f"hypothesis:{hypothesis.hypothesis_id}:evidence:{evidence_id}"
                    for evidence_id in missing
                )
                continue
            self.investigation.backfill_current_hypothesis(task_id, hypothesis)

        historical_hypotheses, historical_questions = self._historical_investigation(task_id)
        source_hypotheses = list(current_hypotheses)
        current_by_id = {item.hypothesis_id: item for item in current_hypotheses}
        for historical, source_id in historical_hypotheses:
            current = current_by_id.get(historical.hypothesis_id)
            if current is not None and current.statement != historical.text:
                self.investigation.record_historical_conflict(
                    task_id,
                    current.hypothesis_id,
                    current_statement=current.statement,
                    historical_statement=historical.text,
                    source_id=source_id,
                )
                continue
            if current is not None:
                continue
            evidence_ids = [
                source.ref_id for source in historical.sources if source.kind == "system_evidence"
            ]
            missing = self.investigation.missing_current_evidence_ids(
                task_id,
                evidence_ids,
            )
            if missing:
                missing_references.extend(
                    f"hypothesis:{historical.hypothesis_id}:evidence:{item}" for item in missing
                )
                continue
            imported = InvestigationHypothesis(
                hypothesis_id=historical.hypothesis_id,
                statement=historical.text,
                state={
                    "active": "candidate",
                    "rejected": "rejected",
                    "confirmed": "supported",
                }[historical.state],
                evidence_ids=evidence_ids,
                checked_locations=[],
                reason=historical.reason or f"legacy semantic record: {source_id}",
                reopens_hypothesis_id=historical.reopens_hypothesis_id,
            )
            self.investigation.backfill_current_hypothesis(task_id, imported)
            current_by_id[imported.hypothesis_id] = imported
            source_hypotheses.append(imported)

        source_questions = [
            UnresolvedQuestion(
                question_id=stable_question_id(task_id, text),
                task_id=task_id,
                text=text,
                status="open",
                source_ids=source_ids,
                resolution_evidence_ids=[],
                created_at=created_at,
            )
            for text, source_ids, created_at in historical_questions
        ]
        for question in source_questions:
            self.investigation.open_question(question)

        source_projection = _investigation_projection(
            source_hypotheses,
            source_questions,
        )
        target_projection = _investigation_projection(
            self.investigation.list_hypotheses(task_id),
            self.investigation.list_questions(task_id),
        )
        source_ids = {f"{kind}:{identity}" for kind, identity, _ in source_projection}
        target_ids = {f"{kind}:{identity}" for kind, identity, _ in target_projection}
        source_payloads = {
            f"{kind}:{identity}": payload for kind, identity, payload in source_projection
        }
        target_payloads = {
            f"{kind}:{identity}": payload for kind, identity, payload in target_projection
        }
        identity_mismatches = sorted(source_ids ^ target_ids)
        hash_mismatches = sorted(
            identity
            for identity in source_ids & target_ids
            if source_payloads[identity] != target_payloads[identity]
        )
        report = DomainMigrationReport(
            domain="investigation",
            task_id=task_id,
            source_count=len(source_projection),
            target_count=len(target_projection),
            source_hash=_canonical_hash(source_projection),
            target_hash=_canonical_hash(target_projection),
            identity_mismatches=identity_mismatches,
            hash_mismatches=hash_mismatches,
            missing_references=sorted(set(missing_references)),
            ready_to_switch=(
                len(source_projection) == len(target_projection)
                and not identity_mismatches
                and not hash_mismatches
                and not missing_references
            ),
        )
        if report.ready_to_switch:
            self._save_report(report)
        return report

    @_fenced_migration("execution")
    def migrate_execution(self, task_id: str) -> DomainMigrationReport:
        task_id = _required(task_id, "task_id")
        existing = self._load_report("execution", task_id)
        if existing is not None:
            return existing

        operations = self._legacy_operations(task_id)
        receipts = self._legacy_receipts(task_id)
        approvals = self._legacy_approvals(task_id)
        missing_references: list[str] = []
        for operation in operations:
            references: list[ArtifactReference] = []
            for path in operation.artifact_references:
                reference = self._operation_artifact_reference(path)
                if reference is None:
                    missing_references.append(path)
                else:
                    references.append(reference)
            if len(references) != len(operation.artifact_references):
                continue
            self.execution.backfill_operation(
                operation,
                artifact_references=references,
            )
        for receipt in receipts:
            self.execution.record_receipt(receipt)
        for approval in approvals:
            self.execution.record_approval(approval)

        source = _execution_projection(operations, receipts, approvals)
        target = _execution_projection(
            self.execution.list_operations(task_id),
            [
                item
                for item in (
                    self.execution.load_receipt(task_id, receipt.tool_call_id)
                    for receipt in receipts
                )
                if item is not None
            ],
            self.execution.list_approvals(task_id),
        )
        source_by_id = {f"{kind}:{identity}": payload for kind, identity, payload in source}
        target_by_id = {f"{kind}:{identity}": payload for kind, identity, payload in target}
        source_ids = set(source_by_id)
        target_ids = set(target_by_id)
        identity_mismatches = sorted(source_ids ^ target_ids)
        hash_mismatches = sorted(
            identity
            for identity in source_ids & target_ids
            if source_by_id[identity] != target_by_id[identity]
        )
        report = DomainMigrationReport(
            domain="execution",
            task_id=task_id,
            source_count=len(source),
            target_count=len(target),
            source_hash=_canonical_hash(source),
            target_hash=_canonical_hash(target),
            identity_mismatches=identity_mismatches,
            hash_mismatches=hash_mismatches,
            missing_references=sorted(set(missing_references)),
            ready_to_switch=(
                len(source) == len(target)
                and not identity_mismatches
                and not hash_mismatches
                and not missing_references
            ),
        )
        if report.ready_to_switch:
            self._save_report(report)
        return report

    @_fenced_migration("history")
    def migrate_history(self, task_id: str) -> DomainMigrationReport:
        """Backfill immutable compaction history without mutating legacy rows."""
        task_id = _required(task_id, "task_id")
        existing = self._load_report("history", task_id)
        if existing is not None:
            return existing

        snapshots, failures, migration_event = self._legacy_history_rows(task_id)
        source_records = [
            history_record_from_snapshot(snapshot, input_hash=input_hash)
            for input_hash, snapshot in snapshots
        ]
        missing_references: list[str] = []
        for record in source_records:
            invalid_references = [
                reference.path
                for reference in record.artifact_references
                if self.artifact_root is not None and not self._verify_artifact(reference)
            ]
            if invalid_references:
                missing_references.extend(invalid_references)
                continue
            self.history.backfill(record)
        for failure in failures:
            self.history.record_failure(failure)
        if migration_event is not None:
            self.history.record_migration(
                task_id,
                migration_event.active_snapshot_version,
                migration_event,
            )

        target_records = self.history.list_for_task(task_id)
        target_failures = self.history.list_failures(task_id)
        target_event = self.history.migrated_event(task_id)
        source = _history_projection(source_records, failures, migration_event)
        target = _history_projection(target_records, target_failures, target_event)
        source_by_id = {f"{kind}:{identity}": payload for kind, identity, payload in source}
        target_by_id = {f"{kind}:{identity}": payload for kind, identity, payload in target}
        source_ids = set(source_by_id)
        target_ids = set(target_by_id)
        identity_mismatches = sorted(source_ids ^ target_ids)
        hash_mismatches = sorted(
            identity
            for identity in source_ids & target_ids
            if source_by_id[identity] != target_by_id[identity]
        )
        report = DomainMigrationReport(
            domain="history",
            task_id=task_id,
            source_count=len(source),
            target_count=len(target),
            source_hash=_canonical_hash(source),
            target_hash=_canonical_hash(target),
            identity_mismatches=identity_mismatches,
            hash_mismatches=hash_mismatches,
            missing_references=sorted(set(missing_references)),
            ready_to_switch=(
                len(source) == len(target)
                and not identity_mismatches
                and not hash_mismatches
                and not missing_references
            ),
        )
        if report.ready_to_switch:
            self._save_report(report)
        return report

    def _legacy_rows(self, task_id: str) -> list[tuple[str, str, str]]:
        with self.database.connection() as connection:
            table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'deterministic_evidence'
                """
            ).fetchone()
            if table is None:
                return []
            rows = connection.execute(
                """
                SELECT evidence_id, kind, payload
                FROM deterministic_evidence
                WHERE task_id = ? ORDER BY evidence_id
                """,
                (task_id,),
            ).fetchall()
        return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]

    def _legacy_working_memory_rows(
        self,
        task_id: str,
    ) -> list[tuple[int, str, str]]:
        """Migration-only SQL reader; runtime code never imports memory models."""
        with self.database.connection() as connection:
            table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'working_memory'
                """
            ).fetchone()
            rows = (
                []
                if table is None
                else connection.execute(
                    """
                    SELECT version, payload, created_at FROM working_memory
                    WHERE task_id = ? ORDER BY version
                    """,
                    (task_id,),
                ).fetchall()
            )
        return [(int(row[0]), str(row[1]), str(row[2])) for row in rows]

    def _legacy_context_telemetry(self, task_id: str) -> dict[str, object] | None:
        with self.database.connection() as connection:
            table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'context_metrics'
                """
            ).fetchone()
            row = (
                None
                if table is None
                else connection.execute(
                    "SELECT payload FROM context_metrics WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
            )
        return None if row is None else dict(json.loads(str(row[0])))

    def _legacy_history_rows(
        self,
        task_id: str,
    ) -> tuple[
        list[tuple[str, CompactionSnapshot]],
        list[CompactionFailureRecord],
        DeepFixCompactionEvent | None,
    ]:
        with self.database.connection() as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            snapshot_rows = (
                connection.execute(
                    """
                    SELECT input_hash, payload FROM compaction_snapshots
                    WHERE task_id = ? ORDER BY version
                    """,
                    (task_id,),
                ).fetchall()
                if "compaction_snapshots" in tables
                else []
            )
            failure_rows = (
                connection.execute(
                    """
                    SELECT payload FROM compaction_failures
                    WHERE task_id = ? ORDER BY recorded_at, attempt_id, stage
                    """,
                    (task_id,),
                ).fetchall()
                if "compaction_failures" in tables
                else []
            )
            migration_row = (
                connection.execute(
                    """
                    SELECT event_payload FROM context_migrations
                    WHERE task_id = ?
                    """,
                    (task_id,),
                ).fetchone()
                if "context_migrations" in tables
                else None
            )
        return (
            [
                (str(row[0]), CompactionSnapshot.model_validate_json(str(row[1])))
                for row in snapshot_rows
            ],
            [CompactionFailureRecord.model_validate_json(str(row[0])) for row in failure_rows],
            (
                None
                if migration_row is None
                else DeepFixCompactionEvent.model_validate_json(str(migration_row[0]))
            ),
        )

    def _legacy_research_rows(
        self,
        task_id: str,
    ) -> tuple[list[ResearchQuery], list[SearchCandidate], list[ExternalEvidence]]:
        with self.database.connection() as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            query_rows = (
                connection.execute(
                    "SELECT payload FROM research_queries WHERE task_id = ? ORDER BY created_at, rowid",
                    (task_id,),
                ).fetchall()
                if "research_queries" in tables
                else []
            )
            candidate_rows = (
                connection.execute(
                    "SELECT payload FROM search_candidates WHERE task_id = ? ORDER BY created_at, rowid",
                    (task_id,),
                ).fetchall()
                if "search_candidates" in tables
                else []
            )
            evidence_rows = (
                connection.execute(
                    "SELECT payload FROM external_evidence WHERE task_id = ? ORDER BY updated_at, rowid",
                    (task_id,),
                ).fetchall()
                if "external_evidence" in tables
                else []
            )
        return (
            [ResearchQuery.model_validate_json(str(row[0])) for row in query_rows],
            [SearchCandidate.model_validate_json(str(row[0])) for row in candidate_rows],
            [ExternalEvidence.model_validate_json(str(row[0])) for row in evidence_rows],
        )

    def _legacy_investigation_state(self, task_id: str) -> InvestigationState | None:
        with self.database.connection() as connection:
            table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'investigation_state'
                """
            ).fetchone()
            row = (
                None
                if table is None
                else connection.execute(
                    "SELECT payload FROM investigation_state WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
            )
        return None if row is None else InvestigationState.model_validate_json(str(row[0]))

    def _legacy_operations(self, task_id: str) -> list[OperationJournalEntry]:
        with self.database.connection() as connection:
            table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'operation_journal'
                """
            ).fetchone()
            rows = (
                []
                if table is None
                else connection.execute(
                    """
                    SELECT payload FROM operation_journal
                    WHERE task_id = ? ORDER BY created_at, operation_id
                    """,
                    (task_id,),
                ).fetchall()
            )
        return [OperationJournalEntry.model_validate_json(str(row[0])) for row in rows]

    def _legacy_receipts(self, task_id: str) -> list[ToolExecutionReceipt]:
        if self.artifact_root is None:
            return []
        directory = self.artifact_root / "investigation_receipts" / receipt_task_segment(task_id)
        if not directory.is_dir():
            return []
        return [
            ToolExecutionReceipt.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        ]

    def _legacy_approvals(self, task_id: str) -> list[ExecutionApproval]:
        with self.database.connection() as connection:
            table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'legacy_task_projection'
                """
            ).fetchone()
            row = (
                None
                if table is None
                else connection.execute(
                    "SELECT payload FROM legacy_task_projection WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
            )
        payload = {} if row is None else json.loads(str(row[0]))
        records = payload.get("approvals", [])
        approvals: list[ExecutionApproval] = []
        for ordinal, record in enumerate(records):
            operation = str(record.get("operation", "")).strip()
            decision = str(record.get("decision", "")).strip()
            risk = str(record.get("risk", "")).strip()
            approvals.append(
                create_execution_approval(
                    task_id=task_id,
                    operation=operation,
                    decision=decision,
                    risk=risk,
                    legacy_ordinal=ordinal,
                    created_at=f"legacy:{ordinal:08d}",
                )
            )
        return approvals

    def _operation_artifact_reference(
        self,
        relative_path: str,
    ) -> ArtifactReference | None:
        if self.artifact_root is None:
            return None
        candidate = (self.artifact_root / relative_path).resolve()
        try:
            candidate.relative_to(self.artifact_root)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        return ArtifactReference(
            path=relative_path,
            kind="operation_result",
            content_hash=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        )

    def _historical_investigation(
        self,
        task_id: str,
    ) -> tuple[
        list[tuple[HypothesisRecord, str]],
        list[tuple[str, list[str], str]],
    ]:
        hypotheses: list[tuple[HypothesisRecord, str]] = []
        questions: dict[str, tuple[str, list[str], str]] = {}
        with self.database.connection() as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            snapshot_rows = (
                connection.execute(
                    """
                    SELECT version, payload, created_at FROM compaction_snapshots
                    WHERE task_id = ? ORDER BY version
                    """,
                    (task_id,),
                ).fetchall()
                if "compaction_snapshots" in tables
                else []
            )
            memory_rows = (
                connection.execute(
                    """
                    SELECT version, payload, created_at FROM working_memory
                    WHERE task_id = ? ORDER BY version
                    """,
                    (task_id,),
                ).fetchall()
                if "working_memory" in tables
                else []
            )

        for version, payload, created_at in snapshot_rows:
            snapshot = json.loads(str(payload))
            for raw in [
                *snapshot.get("active_hypotheses", []),
                *snapshot.get("rejected_hypotheses", []),
                *snapshot.get("confirmed_hypotheses", []),
            ]:
                item = HypothesisRecord.model_validate(raw)
                source_id = _historical_hypothesis_source(
                    item,
                    default=f"snapshot:{version}:{item.hypothesis_id}",
                )
                hypotheses.append((item, source_id))
            for raw in snapshot.get("unresolved_questions", []):
                item = ProvenancedText.model_validate(raw)
                source_ids = [f"{source.kind}:{source.ref_id}" for source in item.sources] or [
                    f"snapshot:{version}"
                ]
                _merge_historical_question(
                    questions,
                    item.text,
                    source_ids,
                    str(created_at),
                )

        for record in self.history.list_for_task(task_id):
            for semantic in record.semantic_items:
                if semantic.kind == "hypothesis":
                    item = HypothesisRecord.model_validate(semantic.payload)
                    source_id = (
                        semantic.provenance_root_ids[0]
                        if semantic.provenance_root_ids
                        else f"history:{record.version}:{item.hypothesis_id}"
                    )
                    hypotheses.append((item, source_id))
                elif semantic.kind == "unresolved_question":
                    item = ProvenancedText.model_validate(semantic.payload)
                    _merge_historical_question(
                        questions,
                        item.text,
                        semantic.provenance_root_ids or [f"history:{record.version}"],
                        record.created_at,
                    )

        for version, payload, created_at in memory_rows:
            memory = json.loads(str(payload))
            for field_name, state in (
                ("active_hypotheses", "active"),
                ("rejected_hypotheses", "rejected"),
                ("confirmed_hypotheses", "confirmed"),
            ):
                for raw in memory.get(field_name, []):
                    item = _legacy_hypothesis_record(raw, state, int(version))
                    hypotheses.append((item, f"working_memory:{version}"))
            for item in memory.get("unresolved_questions", []):
                text = str(item).strip()
                if not text:
                    continue
                _merge_historical_question(
                    questions,
                    text,
                    [f"working_memory:{version}"],
                    str(created_at),
                )
        unique_hypotheses = {
            (item.hypothesis_id, source_id): (item, source_id) for item, source_id in hypotheses
        }
        return list(unique_hypotheses.values()), list(questions.values())

    def _artifact_reference(self, virtual_path: str) -> ArtifactReference | None:
        if self.artifact_root is None:
            return None
        prefix = "/.deepfix-artifacts/"
        if not virtual_path.startswith(prefix):
            return None
        candidate = (self.artifact_root / Path(virtual_path.removeprefix(prefix))).resolve()
        try:
            candidate.relative_to(self.artifact_root)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        return ArtifactReference(
            path=virtual_path,
            kind="research",
            content_hash=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            work_unit_ids=[],
        )

    def _verify_artifact(self, reference: ArtifactReference) -> bool:
        resolved = self._artifact_reference(reference.path)
        return resolved is not None and resolved.content_hash == reference.content_hash

    def _load_report(self, domain: str, task_id: str) -> DomainMigrationReport | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT report_json FROM domain_migrations
                WHERE domain = ? AND task_id = ?
                """,
                (domain, task_id),
            ).fetchone()
        return None if row is None else DomainMigrationReport.model_validate_json(str(row[0]))

    def _save_report(self, report: DomainMigrationReport) -> None:
        with self.database.unit_of_work(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO domain_migrations(domain, task_id, report_json, switched_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(domain, task_id) DO NOTHING
                """,
                (report.domain, report.task_id, report.model_dump_json()),
            )

    @contextmanager
    def _migration_fence(self, domain: str, task_id: str):
        normalized_task_id = _required(task_id, "task_id")
        owner_id = uuid4().hex
        receipt_root = (
            None
            if self.artifact_root is None or domain != "execution"
            else self.artifact_root / "investigation_receipts"
        )
        with ExitStack() as stack:
            if receipt_root is not None:
                stack.enter_context(legacy_receipt_task_guard(receipt_root, normalized_task_id))
            try:
                with self.database.unit_of_work(immediate=True) as connection:
                    existing = connection.execute(
                        """
                        SELECT owner_id, started_at
                        FROM domain_migration_fences
                        WHERE domain = ? AND task_id = ?
                        """,
                        (domain, normalized_task_id),
                    ).fetchone()
                    if existing is not None:
                        marker = connection.execute(
                            """
                            SELECT 1 FROM domain_migrations
                            WHERE domain = ? AND task_id = ?
                            """,
                            (domain, normalized_task_id),
                        ).fetchone()
                        stale = connection.execute(
                            "SELECT ? < datetime('now', '-15 minutes')",
                            (str(existing[1]),),
                        ).fetchone()[0]
                        if marker is None and not bool(stale):
                            raise DomainMigrationInProgress(
                                "domain migration already in progress: "
                                f"{domain}:{normalized_task_id}"
                            )
                        connection.execute(
                            """
                            DELETE FROM domain_migration_fences
                            WHERE domain = ? AND task_id = ? AND owner_id = ?
                            """,
                            (domain, normalized_task_id, str(existing[0])),
                        )
                    connection.execute(
                        """
                        INSERT INTO domain_migration_fences(
                            domain, task_id, owner_id, started_at
                        )
                        VALUES (?, ?, ?, datetime('now'))
                        """,
                        (domain, normalized_task_id, owner_id),
                    )
            except sqlite3.IntegrityError as error:
                raise DomainMigrationInProgress(
                    f"domain migration already in progress: {domain}:{normalized_task_id}"
                ) from error
            try:
                yield
            finally:
                with self.database.unit_of_work(immediate=True) as connection:
                    connection.execute(
                        """
                        DELETE FROM domain_migration_fences
                        WHERE domain = ? AND task_id = ? AND owner_id = ?
                        """,
                        (domain, normalized_task_id, owner_id),
                    )

    def _initialize_schema(self) -> None:
        with self.database.unit_of_work() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS domain_migrations (
                    domain TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    switched_at TEXT NOT NULL,
                    PRIMARY KEY(domain, task_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS domain_migration_fences (
                    domain TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    PRIMARY KEY(domain, task_id)
                )
                """
            )


_LEGACY_TYPES = {
    "test": SystemTestEvidence,
    "file": FileChangeEvidence,
    "approval": ApprovalEvidence,
    "research": ResearchStatusEvidence,
}

_DETERMINISTIC_KINDS = {
    EvidenceKind.TEST,
    EvidenceKind.FILE_CHANGE,
    EvidenceKind.APPROVAL,
    EvidenceKind.RESEARCH_STATUS,
}


def domain_is_switched(
    database: SQLiteDatabase,
    domain: str,
    task_id: str,
) -> bool:
    with database.connection() as connection:
        return domain in switched_domains_for_task(connection, task_id, (domain,))


def switched_domains_for_task(
    connection,
    task_id: str,
    domains: tuple[str, ...],
) -> frozenset[str]:
    """Read authority-switch markers on the caller's transaction snapshot."""

    table = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'domain_migrations'
        """
    ).fetchone()
    if table is None:
        return frozenset()
    stored = {
        str(row[0])
        for row in connection.execute(
            "SELECT domain FROM domain_migrations WHERE task_id = ?",
            (task_id,),
        ).fetchall()
    }
    switched = set(domains) & stored
    if stored & {"evidence", "deterministic_evidence"}:
        switched.update(
            domain for domain in domains if domain in {"evidence", "deterministic_evidence"}
        )
    return frozenset(switched)


def ensure_no_domain_migration_fence(
    connection,
    task_id: str,
    domains: tuple[str, ...],
) -> None:
    table = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'domain_migration_fences'
        """
    ).fetchone()
    if table is None:
        return
    fenced = {
        str(row[0])
        for row in connection.execute(
            "SELECT domain FROM domain_migration_fences WHERE task_id = ?",
            (task_id,),
        ).fetchall()
    }
    requested = set(domains)
    if "evidence" in requested:
        requested.add("deterministic_evidence")
    conflict = sorted(fenced & requested)
    if conflict:
        raise DomainMigrationInProgress(
            f"legacy projection is fenced by domain migration: {','.join(conflict)}"
        )


def _restore_legacy(kind: str, payload: str):
    model = _LEGACY_TYPES.get(kind)
    if model is None:
        raise ValueError(f"Unsupported legacy deterministic Evidence kind: {kind}")
    return model.model_validate_json(payload)


def _canonical_payload(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_payload(value).encode("utf-8")).hexdigest()


def _research_projection(
    queries: list[ResearchQuery],
    candidates: list[SearchCandidate],
    evidence: list[ExternalEvidence],
) -> list[tuple[str, str, str]]:
    records = [
        ("query", item.query_id, _canonical_payload(item.model_dump(mode="json")))
        for item in queries
    ]
    records.extend(
        (
            "candidate",
            item.candidate_id,
            _canonical_payload(item.model_dump(mode="json")),
        )
        for item in candidates
    )
    records.extend(
        (
            "evidence",
            item.evidence_id,
            _canonical_payload(item.model_dump(mode="json")),
        )
        for item in evidence
    )
    return sorted(records, key=lambda item: (item[0], item[1]))


def _execution_projection(
    operations: list[OperationJournalEntry],
    receipts: list[ToolExecutionReceipt],
    approvals: list[ExecutionApproval],
) -> list[tuple[str, str, str]]:
    records = [
        (
            "operation",
            item.operation_id,
            _canonical_payload(item.model_dump(mode="json")),
        )
        for item in operations
    ]
    records.extend(
        (
            "receipt",
            item.tool_call_id,
            _canonical_payload(item.model_dump(mode="json")),
        )
        for item in receipts
    )
    records.extend(
        (
            "approval",
            item.approval_id,
            _canonical_payload(item.model_dump(mode="json")),
        )
        for item in approvals
    )
    return sorted(records, key=lambda item: (item[0], item[1]))


def _history_projection(
    snapshots: list[HistorySnapshotRecord],
    failures: list[CompactionFailureRecord],
    migration_event: DeepFixCompactionEvent | None,
) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    for snapshot in snapshots:
        payload = snapshot.model_dump(mode="json")
        payload["semantic_items"] = [
            item.model_dump(mode="json") for item in snapshot.semantic_items
        ]
        records.append(
            (
                "snapshot",
                str(snapshot.version),
                _canonical_payload(payload),
            )
        )
    records.extend(
        (
            "failure",
            f"{item.attempt_id}:{item.stage}",
            _canonical_payload(item.model_dump(mode="json")),
        )
        for item in failures
    )
    if migration_event is not None:
        records.append(
            (
                "migration",
                str(migration_event.active_snapshot_version),
                _canonical_payload(migration_event.model_dump(mode="json")),
            )
        )
    return sorted(records, key=lambda item: (item[0], item[1]))


def _investigation_projection(
    hypotheses: list[InvestigationHypothesis],
    questions: list[UnresolvedQuestion],
) -> list[tuple[str, str, str]]:
    records = [
        (
            "hypothesis",
            item.hypothesis_id,
            _canonical_payload(item.model_dump(mode="json")),
        )
        for item in hypotheses
    ]
    records.extend(
        (
            "question",
            item.question_id,
            _canonical_payload(item.model_dump(mode="json")),
        )
        for item in questions
    )
    return sorted(records, key=lambda item: (item[0], item[1]))


def _historical_hypothesis_source(
    item: HypothesisRecord,
    *,
    default: str,
) -> str:
    if not item.sources:
        return default
    source = item.sources[0]
    return f"{source.kind}:{source.ref_id}"


def _legacy_hypothesis_record(
    raw: object,
    state: str,
    version: int,
) -> HypothesisRecord:
    """Migration-only parser for retired Working Memory hypothesis payloads."""
    if isinstance(raw, str):
        text = raw.strip()
        identity = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        return HypothesisRecord(
            hypothesis_id=f"legacy_hyp_{identity}",
            text=text,
            state=state,
            reason=("legacy working memory" if state != "active" else None),
            sources=[
                ProvenanceRef(
                    kind="snapshot_record",
                    ref_id=f"legacy-memory:{version}",
                )
            ],
            updated_in_version=max(version, 1),
        )
    payload = dict(raw) if isinstance(raw, dict) else {}
    payload.setdefault("state", state)
    payload.setdefault("updated_in_version", max(version, 1))
    payload["sources"] = [
        {
            "kind": (
                "snapshot_record" if item.get("kind") == "working_memory" else item.get("kind")
            ),
            "ref_id": item.get("ref_id"),
        }
        for item in payload.get("sources", [])
        if isinstance(item, dict) and item.get("ref_id")
    ] or [
        {
            "kind": "snapshot_record",
            "ref_id": f"legacy-memory:{version}",
        }
    ]
    return HypothesisRecord.model_validate(payload)


def _legacy_claim(raw: object, version: int) -> ProvenancedClaim:
    """Migration-only parser for retired Working Memory fact payloads."""
    if isinstance(raw, str):
        text = raw.strip()
        return ProvenancedClaim(
            claim_id=_stable_legacy_id("claim", text),
            text=text,
            sources=[
                ProvenanceRef(
                    kind="snapshot_record",
                    ref_id=f"legacy-memory:{version}",
                )
            ],
        )
    payload = dict(raw) if isinstance(raw, dict) else {}
    payload["sources"] = [
        {
            "kind": (
                "snapshot_record" if item.get("kind") == "working_memory" else item.get("kind")
            ),
            "ref_id": item.get("ref_id"),
        }
        for item in payload.get("sources", [])
        if isinstance(item, dict) and item.get("ref_id")
    ] or [
        {
            "kind": "snapshot_record",
            "ref_id": f"legacy-memory:{version}",
        }
    ]
    return ProvenancedClaim.model_validate(payload)


def _stable_legacy_id(prefix: str, *parts: str) -> str:
    material = "\0".join((prefix, *parts))
    return f"legacy_{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _merge_historical_question(
    questions: dict[str, tuple[str, list[str], str]],
    text: str,
    source_ids: list[str],
    created_at: str,
) -> None:
    key = _normalize_text(text)
    existing = questions.get(key)
    if existing is None:
        questions[key] = (text, list(dict.fromkeys(source_ids)), created_at)
        return
    existing_text, existing_sources, existing_created_at = existing
    questions[key] = (
        existing_text,
        list(dict.fromkeys([*existing_sources, *source_ids])),
        min(existing_created_at, created_at),
    )


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized
