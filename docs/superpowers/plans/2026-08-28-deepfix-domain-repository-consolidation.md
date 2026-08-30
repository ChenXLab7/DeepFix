# DeepFix Domain Repository Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate DeepFix's existing Evidence, Research, Investigation, Receipt, Operation, Approval, and Compaction persistence behind four bounded domain repositories without changing the DeepAgents-owned Agent Loop or prematurely deleting Plan 4 compatibility state.

**Architecture:** Reuse the shared `SQLiteDatabase` and UnitOfWork completed in Plan 2. Introduce one immutable Evidence envelope, then migrate each domain independently in the order Evidence → Research → Investigation → Execution → History; every authority switch is protected by stable-ID/hash/count validation and a read-only rollback source. Existing Store class names become thin compatibility facades only after their authoritative writer and reader have moved.

**Tech Stack:** Python 3.11+, DeepAgents 0.7.x, LangChain 1.3.x, LangGraph 1.2.x, Pydantic 2, SQLite/WAL, pytest 8+, Ruff 0.12+

**Spec:** `docs/superpowers/specs/2026-08-27-deepfix-architecture-audit.md`

**Program:** `docs/superpowers/plans/2026-08-28-deepfix-state-authority-migration-program.md` Plan 3

**Status:** COMPLETE — implementation, migration cutover hardening, offline gates,
and completion evidence recorded on 2026-08-30. Awaiting user review before
Plan 4 begins.

## Global Constraints

- DeepAgents/LangGraph continue to own the Agent Loop, Messages, native Todo, Checkpoint, interrupt, and resume. This plan adds no Loop, Planner, Task Graph, Checkpointer, Manager, or model-routing abstraction.
- Continue to reuse DeepAgents Backend, Artifact handling, history serialization, large ToolMessage/media offload, Tool facade, and configured model instances. DeepFix owns only typed domain persistence, CompactionDelta validation/merge, Snapshot activation, and trusted execution recovery.
- Todo remains navigation authority only. No Repository in this plan uses Todo status to authorize tools, declare facts, or adjudicate completion.
- Reuse Plan 2's `SQLiteDatabase`; do not add another database, generic CRUD repository, global Store registry, or second UnitOfWork abstraction.
- One SQLite file does not imply one Repository. Evidence, Investigation, Execution, and History expose separate public contracts and tables.
- Domain Repositories own current facts; Snapshot owns history, not truth. A current record with the same stable ID always overrides a historical Snapshot projection.
- Evidence authority is assigned by trusted DeepFix code. Model output may propose semantic claims but cannot choose or elevate `authority`, `origin`, `verification_state`, provenance roots, or Artifact integrity fields.
- Research acquisition remains in the existing Research Service/tools. `EvidenceRepository` stores accepted external Evidence and research-attempt audit metadata; it does not perform network access.
- Receipt, Operation, and Approval remain distinct domain models inside `ExecutionRepository`.
- Operation lifecycle remains `prepared`, `started`, `observed`, `committed`, or `unknown`. SQLite transactions never span a model call, network call, Shell command, or filesystem mutation.
- Tool-result, research, and conversation bodies remain Artifacts. Artifact content must be atomically written, read back, and hash-verified before a Repository stores its reference.
- An external side effect is never repeated merely because Receipt, Evidence, or state projection failed. Recovery continues from `observed` or `unknown` using workspace/Artifact inspection.
- `InvestigationRepository` stores hypotheses and unresolved questions but never decides tool visibility, task lifecycle, phase, or final outcome.
- `HistoryRepository` stores Snapshot lifecycle, input hash, coverage, historical semantic items, provenance, and Artifact references only. It does not become a second Evidence or Investigation authority.
- Existing `AgentPhase`, Working Memory, giant `TaskState` compatibility fields, public `save_progress`, report reconstruction, and old class-name removals remain Plan 4 work.
- Migration is domain-by-domain: add schema and adapters, backfill with stable IDs, validate count/hash/references, switch readers, then stop only the migrated legacy writes. Do not maintain indefinite dual writes.
- Historical tables and Receipt files remain read-only rollback sources until Plan 4. New production writes must not update them after the corresponding authority switch.
- Preserve Workspace confinement, approval semantics, Receipt idempotency, Operation recovery, required-oracle checks, compaction failure atomicity, overflow retry limits, and false-FIXED protection.
- This plan ends at implementation completion plus focused/core offline tests. It does not run paid QuixBugs or formal A/B acceptance.
- Every production change follows red-green-refactor and ends at a reviewable commit.

---

## File Structure

### New files

- `src/deepfix/domain_repositories/__init__.py` — exports only the four bounded repositories and their public view models.
- `src/deepfix/domain_repositories/evidence.py` — immutable Evidence envelope, trusted constructors, Evidence queries, and research-attempt audit persistence.
- `src/deepfix/domain_repositories/investigation.py` — normalized Hypothesis and UnresolvedQuestion persistence plus compatibility projections.
- `src/deepfix/domain_repositories/execution.py` — Operation, Receipt, Approval persistence, execution-integrity queries, and atomic observation commit.
- `src/deepfix/domain_repositories/history.py` — history-only Snapshot records, lifecycle transitions, coverage, failures, and current-domain-over-history projection.
- `src/deepfix/domain_repositories/migration.py` — one-time domain backfills, stable hashes, validation reports, and authority-switch markers.
- `tests/domain_repositories/test_evidence_repository.py` — immutable envelope, authority, provenance, Artifact, and deterministic Evidence tests.
- `tests/domain_repositories/test_research_migration.py` — external Evidence and research-attempt migration tests.
- `tests/domain_repositories/test_investigation_repository.py` — Hypothesis/UnresolvedQuestion identity and compatibility tests.
- `tests/domain_repositories/test_execution_repository.py` — Receipt concurrency, Operation lifecycle, Approval, recovery, and Artifact-reference tests.
- `tests/domain_repositories/test_history_repository.py` — Snapshot lifecycle, coverage, projection precedence, and failure atomicity tests.
- `tests/domain_repositories/test_migration_gate.py` — cross-domain counts, hashes, stable IDs, rollback sources, and legacy-write-disable tests.
- `tests/domain_repositories/test_service_integration.py` — production construction, restore, reporting input, and recovery wiring tests.

### Existing files modified

- `src/deepfix/compaction/models.py` — retain existing payload models; add no competing authority fields.
- `src/deepfix/compaction/store.py` — become a compatibility facade over `EvidenceRepository` and `HistoryRepository`.
- `src/deepfix/compaction/coordinator.py` — persist prepared/active/abandoned Snapshots through History and preserve Artifact-first replacement order.
- `src/deepfix/compaction/evidence.py` — write trusted deterministic Evidence through `EvidenceRepository`.
- `src/deepfix/compaction/snapshot.py` — merge current domain views over historical Snapshot semantics by stable ID.
- `src/deepfix/research/store.py` — become a compatibility facade; network/retrieval behavior remains outside the Repository.
- `src/deepfix/research/tools.py` — save research attempts and accepted Evidence through the new authority.
- `src/deepfix/investigation/store.py` — become a compatibility facade while legacy phase-shaped materialization remains available through Plan 4.
- `src/deepfix/investigation/models.py` — add `UnresolvedQuestion`; preserve existing event/experiment contracts.
- `src/deepfix/investigation/receipts.py` — retain Artifact serialization helpers; delegate Receipt metadata authority to `ExecutionRepository`.
- `src/deepfix/operations.py` — retain Operation domain models and reconciler; delegate persistence to `ExecutionRepository`.
- `src/deepfix/agent.py` — construct/inject four repositories while retaining DeepAgents middleware and graph ownership.
- `src/deepfix/cli.py` — construct repositories from the same `SQLiteDatabase` and Artifact root.
- `src/deepfix/context.py` — read current facts from domain repositories and historical detail from History.
- `src/deepfix/protected_context.py` — deduplicate current and historical projections by stable ID.
- `src/deepfix/service.py` — use repository-backed restore/recovery inputs without deleting legacy TaskState fields.
- `src/deepfix/reporting.py` — consume repository views where available while retaining its Plan 4 compatibility projection.
- Existing focused tests under `tests/compaction`, `tests/research`, `tests/investigation`, `tests/test_operations.py`, `tests/test_service.py`, `tests/test_context.py`, and `tests/test_reporting.py` — preserve public behavior while asserting the new authority boundary.

## Stable Interfaces

The implementation must keep these public contracts consistent across tasks:

```python
class EvidenceKind(StrEnum):
    TEST = "test"
    FILE_CHANGE = "file_change"
    APPROVAL = "approval"
    RESEARCH_STATUS = "research_status"
    EXTERNAL_RESEARCH = "external_research"
    SEMANTIC_CLAIM = "semantic_claim"


class EvidenceAuthority(StrEnum):
    SYSTEM = "system"
    USER = "user"
    RESEARCH = "research"
    MODEL_SEMANTIC = "model_semantic"


class EvidenceVerification(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    NOT_APPLICABLE = "not_applicable"


class EvidenceEnvelope(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    task_id: str
    kind: EvidenceKind
    origin: str
    authority: EvidenceAuthority
    verification_state: EvidenceVerification
    payload_type: str
    payload: dict[str, JsonValue]
    provenance_root_ids: list[str]
    artifact_references: list[ArtifactReference]
    content_hash: str
    created_at: str


class EvidenceRepository:
    def __init__(self, database: SQLiteDatabase | str | Path) -> None: ...
    def record_deterministic(
        self,
        task_id: str,
        evidence: SystemTestEvidence | FileChangeEvidence | ApprovalEvidence | ResearchStatusEvidence,
        *,
        provenance_root_ids: list[str],
        artifact_references: list[ArtifactReference] = [],
    ) -> EvidenceEnvelope: ...
    def accept_external(
        self,
        evidence: ExternalEvidence,
        *,
        provenance_root_ids: list[str],
        artifact_references: list[ArtifactReference],
    ) -> EvidenceEnvelope: ...
    def record_semantic_candidate(
        self,
        task_id: str,
        claim: ProvenancedClaim,
        *,
        provenance_root_ids: list[str],
    ) -> EvidenceEnvelope: ...
    def get(self, task_id: str, evidence_id: str) -> EvidenceEnvelope: ...
    def list_for_task(self, task_id: str) -> list[EvidenceEnvelope]: ...
    def verification_view(self, task_id: str) -> VerificationEvidenceView: ...


class UnresolvedQuestion(StrictModel):
    question_id: str
    task_id: str
    text: str
    status: Literal["open", "resolved"]
    source_ids: list[str]
    resolution_evidence_ids: list[str]
    created_at: str
    resolved_at: str | None = None


class ExecutionApproval(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: str
    task_id: str
    operation: str
    decision: str
    risk: str
    source_tool_call_id: str | None = None
    created_at: str


class InvestigationRepository:
    def __init__(self, database: SQLiteDatabase | str | Path) -> None: ...
    def record_hypothesis(
        self, task_id: str, hypothesis: InvestigationHypothesis
    ) -> InvestigationHypothesis: ...
    def get_hypothesis(self, task_id: str, hypothesis_id: str) -> InvestigationHypothesis: ...
    def list_hypotheses(self, task_id: str) -> list[InvestigationHypothesis]: ...
    def open_question(self, question: UnresolvedQuestion) -> UnresolvedQuestion: ...
    def resolve_question(
        self, task_id: str, question_id: str, *, evidence_ids: list[str]
    ) -> UnresolvedQuestion: ...
    def list_questions(self, task_id: str, *, status: str | None = None) -> list[UnresolvedQuestion]: ...


class ExecutionRepository:
    def __init__(self, database: SQLiteDatabase | str | Path) -> None: ...
    def prepare(self, entry: NewOperationEntry) -> OperationJournalEntry: ...
    def mark_started(self, operation_id: str) -> OperationJournalEntry: ...
    def record_receipt(self, receipt: ToolExecutionReceipt) -> ToolExecutionReceipt: ...
    def observe_with_receipt(
        self,
        operation_id: str,
        *,
        post_state: OperationStateSnapshot,
        receipt: ToolExecutionReceipt,
        artifact_references: list[ArtifactReference],
    ) -> OperationJournalEntry: ...
    def commit(self, operation_id: str) -> OperationJournalEntry: ...
    def mark_unknown(self, operation_id: str, reason: str) -> OperationJournalEntry: ...
    def load_operation(self, operation_id: str) -> OperationJournalEntry | None: ...
    def load_receipt(self, task_id: str, tool_call_id: str) -> ToolExecutionReceipt | None: ...
    def record_approval(self, approval: ExecutionApproval) -> ExecutionApproval: ...
    def integrity_view(self, task_id: str) -> ExecutionIntegrity: ...


class HistoryRepository:
    def __init__(self, database: SQLiteDatabase | str | Path) -> None: ...
    def save_prepared(self, record: HistorySnapshotRecord) -> HistorySnapshotRecord: ...
    def activate(self, task_id: str, version: int, event: DeepFixCompactionEvent) -> HistorySnapshotRecord: ...
    def abandon(self, task_id: str, version: int, reason: str) -> HistorySnapshotRecord: ...
    def get(self, task_id: str, version: int) -> HistorySnapshotRecord: ...
    def active_from_event(
        self, task_id: str, event: DeepFixCompactionEvent | None
    ) -> HistorySnapshotRecord | None: ...
    def record_failure(self, failure: CompactionFailureRecord) -> None: ...
    def project_snapshot(
        self,
        task_id: str,
        version: int,
        *,
        evidence: EvidenceRepository,
        investigation: InvestigationRepository,
    ) -> CompactionSnapshot: ...
```

Mutable default arguments shown above are documentation shorthand only; implementation uses `None` and copies caller collections.

---

### Task 1: Immutable Evidence Envelope and Trusted Constructors

**Files:**
- Create: `src/deepfix/domain_repositories/__init__.py`
- Create: `src/deepfix/domain_repositories/evidence.py`
- Create: `tests/domain_repositories/test_evidence_repository.py`

**Interfaces:**
- Consumes: `SQLiteDatabase`, existing deterministic Evidence models, `ExternalEvidence`, `ProvenancedClaim`, and `ArtifactReference`.
- Produces: `EvidenceEnvelope`, authority enums, `EvidenceRepository`, and `VerificationEvidenceView`.

- [x] **Step 1: Write failing authority, identity, and immutability tests**

```python
def test_repository_assigns_system_authority_to_test_evidence(tmp_path):
    repository = EvidenceRepository(tmp_path / "deepfix.db")
    item = repository.record_deterministic(
        "task-1",
        system_test_evidence("test-1", exit_code=0),
        provenance_root_ids=["tool-call-1"],
    )
    assert item.authority is EvidenceAuthority.SYSTEM
    assert item.verification_state is EvidenceVerification.VERIFIED
    assert item.provenance_root_ids == ["tool-call-1"]


def test_same_id_same_content_is_idempotent_and_changed_content_conflicts(tmp_path):
    repository = EvidenceRepository(tmp_path / "deepfix.db")
    first = repository.record_deterministic(
        "task-1", system_test_evidence("test-1", exit_code=0),
        provenance_root_ids=["tool-call-1"],
    )
    assert repository.record_deterministic(
        "task-1", system_test_evidence("test-1", exit_code=0),
        provenance_root_ids=["tool-call-1"],
    ) == first
    with pytest.raises(EvidenceIdentityConflict):
        repository.record_deterministic(
            "task-1", system_test_evidence("test-1", exit_code=1),
            provenance_root_ids=["tool-call-1"],
        )


def test_model_semantic_candidate_cannot_request_system_authority(tmp_path):
    signature = inspect.signature(EvidenceRepository.record_semantic_candidate)
    assert "authority" not in signature.parameters
    item = EvidenceRepository(tmp_path / "deepfix.db").record_semantic_candidate(
        "task-1", claim("claim-1"), provenance_root_ids=["receipt-1"]
    )
    assert item.authority is EvidenceAuthority.MODEL_SEMANTIC
```

- [x] **Step 2: Run the focused tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_evidence_repository.py -q`

Expected: FAIL because `deepfix.domain_repositories.evidence` does not exist.

- [x] **Step 3: Implement canonical hashing and trusted constructors**

Create one `evidence_records` table keyed by `(task_id, evidence_id)` with indexed `kind`, `authority`, and `verification_state`. Store canonical JSON and a SHA-256 `content_hash`; reread and compare the inserted envelope before commit.

Use repository-controlled mappings:

```python
_DETERMINISTIC_POLICY = {
    SystemTestEvidence: (EvidenceKind.TEST, EvidenceAuthority.SYSTEM),
    FileChangeEvidence: (EvidenceKind.FILE_CHANGE, EvidenceAuthority.SYSTEM),
    ApprovalEvidence: (EvidenceKind.APPROVAL, EvidenceAuthority.SYSTEM),
    ResearchStatusEvidence: (EvidenceKind.RESEARCH_STATUS, EvidenceAuthority.RESEARCH),
}
```

`record_deterministic()` derives verification from typed payload fields; `record_semantic_candidate()` always sets `MODEL_SEMANTIC`; `accept_external()` always sets `RESEARCH`. Reject empty provenance roots except for explicitly migrated legacy records, which use a stable `legacy:<table>:<id>` root.

- [x] **Step 4: Add Artifact-reference verification tests**

```python
def test_repository_rejects_unverified_artifact_reference(tmp_path):
    repository = EvidenceRepository(tmp_path / "deepfix.db")
    with pytest.raises(ArtifactIntegrityError):
        repository.record_deterministic(
            "task-1",
            research_status_evidence("research-1", "missing.json"),
            provenance_root_ids=["research-attempt-1"],
            artifact_references=[artifact_ref("missing.json", "deadbeef")],
        )
```

Inject an Artifact verifier callback into `EvidenceRepository`; production wiring uses the existing backend/Artifact root, while tests use a deterministic fake. The Repository never writes Artifact bodies.

- [x] **Step 5: Run focused tests and static checks**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_evidence_repository.py tests/compaction/test_models.py -q`

Run: `.venv\Scripts\ruff check src/deepfix/domain_repositories tests/domain_repositories/test_evidence_repository.py`

Expected: PASS.

- [x] **Step 6: Commit Task 1**

```powershell
git add src/deepfix/domain_repositories tests/domain_repositories/test_evidence_repository.py
git commit -m "feat: add immutable evidence repository"
```

---

### Task 2: Deterministic Evidence Backfill and Authority Switch

**Files:**
- Modify: `src/deepfix/domain_repositories/migration.py`
- Modify: `src/deepfix/compaction/store.py`
- Modify: `src/deepfix/compaction/evidence.py`
- Modify: `tests/domain_repositories/test_evidence_repository.py`
- Modify: `tests/compaction/test_store.py`
- Modify: `tests/compaction/test_evidence.py`

**Interfaces:**
- Consumes: Task 1 `EvidenceRepository`; legacy `deterministic_evidence` rows and TaskState evidence/test/change/approval projections.
- Produces: `migrate_deterministic_evidence(task_id) -> DomainMigrationReport` and a `CompactionStore` Evidence compatibility facade.

- [x] **Step 1: Write failing stable migration tests**

```python
def test_deterministic_backfill_preserves_ids_payload_hashes_and_counts(legacy_database):
    report = DomainMigrator(legacy_database).migrate_deterministic_evidence("task-1")
    assert report.source_count == report.target_count == 4
    assert report.identity_mismatches == []
    assert report.hash_mismatches == []
    assert report.missing_artifacts == []
    assert report.ready_to_switch


def test_backfill_is_idempotent_and_does_not_update_legacy_rows(legacy_database):
    before = read_legacy_rows(legacy_database, "deterministic_evidence")
    migrator = DomainMigrator(legacy_database)
    assert migrator.migrate_deterministic_evidence("task-1").ready_to_switch
    assert migrator.migrate_deterministic_evidence("task-1").ready_to_switch
    assert read_legacy_rows(legacy_database, "deterministic_evidence") == before
```

- [x] **Step 2: Run migration tests and verify RED**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_evidence_repository.py -k backfill -q`

Expected: FAIL because the migrator/report do not exist.

- [x] **Step 3: Implement deterministic backfill and validation report**

Add `domain_migrations(domain, task_id, source_hash, target_hash, source_count, target_count, switched_at)` and implement `DomainMigrationReport` with exact mismatch lists. Legacy payloads keep their original `evidence_id`; the canonical envelope hash is computed from the normalized content, while `source_hash` records the canonical ordered legacy rows.

The switch marker may be written only when counts match, all IDs resolve, every Artifact reference verifies, and no payload hash conflict exists.

- [x] **Step 4: Turn `CompactionStore` Evidence methods into a facade**

Retain these legacy methods for Plan 3 callers:

```python
def save_evidence(self, task_id, evidence):
    return self.evidence.record_deterministic(
        task_id,
        evidence,
        provenance_root_ids=legacy_or_typed_roots(evidence),
    )

def list_evidence(self, task_id):
    return [typed_payload(item) for item in self.evidence.list_for_task(task_id)
            if item.kind in DETERMINISTIC_KINDS]
```

Once the switch marker exists, these methods must not write `deterministic_evidence`. The table remains read-only for rollback.

- [x] **Step 5: Run deterministic Evidence regressions**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_evidence_repository.py tests/compaction/test_store.py tests/compaction/test_evidence.py tests/test_protected_context.py -q`

Expected: PASS; public payload behavior remains unchanged and new writes exist only in `evidence_records`.

- [x] **Step 6: Commit Task 2**

```powershell
git add src/deepfix/domain_repositories/migration.py src/deepfix/compaction/store.py src/deepfix/compaction/evidence.py tests/domain_repositories/test_evidence_repository.py tests/compaction/test_store.py tests/compaction/test_evidence.py
git commit -m "refactor: move deterministic evidence authority"
```

### Review Checkpoint A

Pause and report the migrated task count, source/target Evidence counts, hash validation, and proof that legacy deterministic rows are no longer written. Do not begin Research migration until reviewed.

---

### Task 3: External Research Evidence Consolidation

**Files:**
- Create: `tests/domain_repositories/test_research_migration.py`
- Modify: `src/deepfix/domain_repositories/evidence.py`
- Modify: `src/deepfix/domain_repositories/migration.py`
- Modify: `src/deepfix/research/store.py`
- Modify: `src/deepfix/research/tools.py`
- Modify: `tests/research/test_store.py`
- Modify: `tests/research/test_workflow.py`

**Interfaces:**
- Consumes: existing `ResearchQuery`, `SearchCandidate`, `ExternalEvidence`, research Artifacts, and Task 1 Evidence authority.
- Produces: accepted external Evidence in `evidence_records`, `research_attempts` audit records, and a `ResearchEvidenceStore` compatibility facade.

- [x] **Step 1: Write failing acquisition/persistence-boundary tests**

```python
def test_repository_records_attempt_metadata_but_performs_no_network_io(tmp_path):
    repository = EvidenceRepository(tmp_path / "deepfix.db")
    attempt = repository.record_research_attempt(
        task_id="task-1",
        query_id="query-1",
        sanitized_query="pytest timeout",
        providers=["github"],
        provider_errors=[],
    )
    assert attempt.query_id == "query-1"
    assert not hasattr(repository, "search")
    assert not hasattr(repository, "fetch")


def test_external_evidence_keeps_independent_provenance_roots(tmp_path, verified_artifact):
    repository = EvidenceRepository(tmp_path / "deepfix.db", artifact_verifier=verified_artifact)
    item = repository.accept_external(
        external_evidence("evidence-1"),
        provenance_root_ids=["url:https://example.test/release"],
        artifact_references=[verified_artifact.reference],
    )
    assert item.authority is EvidenceAuthority.RESEARCH
    assert item.provenance_root_ids == ["url:https://example.test/release"]
```

- [x] **Step 2: Verify RED**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_research_migration.py -q`

Expected: FAIL because research-attempt persistence and migration are absent.

- [x] **Step 3: Add research-attempt audit and external Evidence migration**

Persist sanitized query, provider names, cleaned provider errors, candidate IDs, and Artifact refs in `research_attempts`; do not persist secrets or raw credentials. Migrate `research_queries`, `search_candidates`, and `external_evidence` with stable IDs and retain those tables read-only.

`ExternalEvidence.local_evidence` remains a typed payload relationship. Its linked IDs must resolve to current `evidence_records`; derived Observation/Claim objects sharing one underlying source retain the same provenance root and count as one independent source.

- [x] **Step 4: Convert the old Store into a facade and switch tool writers**

Keep `ResearchEvidenceStore.save_query/save_candidates/save_evidence/get_evidence/list_evidence/update_verification/query_summary` signatures. Delegate accepted Evidence and attempts to `EvidenceRepository`; candidate retrieval may remain a compatibility audit query until Plan 4 removes the class name. No repository method performs HTTP.

- [x] **Step 5: Run research and verification regressions**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_research_migration.py tests/research tests/test_verification.py tests/test_context.py -q`

Expected: PASS; online-marked tests remain skipped by default.

- [x] **Step 6: Commit Task 3**

```powershell
git add src/deepfix/domain_repositories/evidence.py src/deepfix/domain_repositories/migration.py src/deepfix/research/store.py src/deepfix/research/tools.py tests/domain_repositories/test_research_migration.py tests/research/test_store.py tests/research/test_workflow.py
git commit -m "refactor: consolidate external research evidence"
```

---

### Task 4: Hypothesis and UnresolvedQuestion Authority

**Files:**
- Create: `src/deepfix/domain_repositories/investigation.py`
- Create: `tests/domain_repositories/test_investigation_repository.py`
- Modify: `src/deepfix/domain_repositories/migration.py`
- Modify: `src/deepfix/investigation/models.py`
- Modify: `src/deepfix/investigation/store.py`
- Modify: `src/deepfix/investigation/reducer.py`
- Modify: `tests/investigation/test_store.py`
- Modify: `tests/investigation/test_migration.py`

**Interfaces:**
- Consumes: current `InvestigationHypothesis`, legacy `InvestigationState`, Working Memory/Snapshot unresolved questions as migration inputs only, and current Evidence IDs.
- Produces: normalized `hypotheses`, `unresolved_questions`, and investigation event/experiment persistence behind `InvestigationRepository`.

- [x] **Step 1: Write failing identity and semantic-boundary tests**

```python
def test_hypothesis_transition_requires_stable_id_and_evidence(tmp_path):
    repository = InvestigationRepository(tmp_path / "deepfix.db")
    candidate = hypothesis("h-1", state="candidate", evidence_ids=[])
    repository.record_hypothesis("task-1", candidate)
    supported = candidate.model_copy(update={"state": "supported", "evidence_ids": ["e-1"]})
    assert repository.record_hypothesis("task-1", supported).state == "supported"
    with pytest.raises(HypothesisIdentityConflict):
        repository.record_hypothesis(
            "task-1", hypothesis("h-1", statement="different")
        )


def test_question_is_not_a_todo_or_task_lifecycle_state(tmp_path):
    repository = InvestigationRepository(tmp_path / "deepfix.db")
    question = repository.open_question(unresolved_question("q-1"))
    assert question.status == "open"
    assert not hasattr(question, "todo_status")
    assert not hasattr(repository, "transition_task")
    assert not hasattr(repository, "can_execute")
```

- [x] **Step 2: Verify RED**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_investigation_repository.py -q`

Expected: FAIL because the normalized repository and `UnresolvedQuestion` do not exist.

- [x] **Step 3: Implement normalized tables and transition validation**

Create `hypotheses`, `unresolved_questions`, `investigation_events`, `experiment_results`, `experiment_events`, and `strategy_decisions` under the Repository. A hypothesis statement is immutable for a stable ID; state may move `candidate → supported/rejected`, and a rejected hypothesis can reopen only under a new ID referencing the rejected ID. Resolving a question requires at least one current Evidence ID.

The existing phase, permit, stagnation, and materialized `InvestigationState` fields remain compatibility data through Plan 4 and are not copied into normalized Hypothesis/Question records.

- [x] **Step 4: Backfill and convert `InvestigationStore` into a facade**

Backfill hypotheses from current Investigation state first; use Snapshot/Working Memory only to recover missing unresolved questions and tag their source as historical. If the same stable ID exists in both, the current Investigation record wins and any conflicting historical text is retained only as history/conflict metadata.

Keep `InvestigationStore.load/ensure_started/commit/list_events/commit_experiment/save_strategy_decision` working. Its facade delegates domain records to `InvestigationRepository` while preserving the phase-shaped compatibility projection until Plan 4.

- [x] **Step 5: Run investigation regressions**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_investigation_repository.py tests/investigation/test_store.py tests/investigation/test_migration.py tests/investigation/test_reducer.py tests/navigation/test_feedback.py -q`

Expected: PASS; Todo/navigation behavior is unchanged and Investigation cannot control business lifecycle.

- [x] **Step 6: Commit Task 4**

```powershell
git add src/deepfix/domain_repositories/investigation.py src/deepfix/domain_repositories/migration.py src/deepfix/investigation/models.py src/deepfix/investigation/store.py src/deepfix/investigation/reducer.py tests/domain_repositories/test_investigation_repository.py tests/investigation/test_store.py tests/investigation/test_migration.py
git commit -m "refactor: consolidate investigation authority"
```

### Review Checkpoint B

Pause and report Hypothesis/Question counts, conflict handling, proof that Snapshot history cannot overwrite current Investigation records, and proof that no tool permission or lifecycle method entered `InvestigationRepository`.

---

### Task 5: ExecutionRepository for Operation, Receipt, and Approval

**Files:**
- Create: `src/deepfix/domain_repositories/execution.py`
- Create: `tests/domain_repositories/test_execution_repository.py`
- Modify: `src/deepfix/domain_repositories/migration.py`
- Modify: `src/deepfix/investigation/receipts.py`
- Modify: `src/deepfix/operations.py`
- Modify: `src/deepfix/investigation/middleware.py`
- Modify: `tests/investigation/test_receipts.py`
- Modify: `tests/test_operations.py`
- Modify: `tests/evaluation/test_fault_injection.py`

**Interfaces:**
- Consumes: existing Operation/Receipt models, ApprovalRecord, Artifact helpers, Workspace state hashes, and shared SQLite.
- Produces: atomic `observe_with_receipt()`, `ExecutionIntegrity`, compatibility Receipt/Journal facades, and recovery-safe authority.

- [x] **Step 1: Write failing parallel Receipt and lifecycle tests**

```python
def test_parallel_receipts_are_atomic_and_idempotent(tmp_path):
    repository = ExecutionRepository(tmp_path / "deepfix.db")
    receipts = [receipt("call-1"), receipt("call-2")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        saved = list(pool.map(repository.record_receipt, receipts))
    assert {item.tool_call_id for item in saved} == {"call-1", "call-2"}
    assert repository.record_receipt(receipts[0]) == receipts[0]


def test_observation_commits_receipt_and_operation_state_together(tmp_path):
    repository = ExecutionRepository(tmp_path / "deepfix.db")
    prepared = repository.prepare(new_operation("op-1", "call-1"))
    repository.mark_started(prepared.operation_id)
    observed = repository.observe_with_receipt(
        prepared.operation_id,
        post_state=post_state(),
        receipt=receipt("call-1"),
        artifact_references=[],
    )
    assert observed.status is OperationStatus.OBSERVED
    assert repository.load_receipt("task-1", "call-1") is not None
```

- [x] **Step 2: Verify RED**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_execution_repository.py -q`

Expected: FAIL because `ExecutionRepository` does not exist.

- [x] **Step 3: Implement Execution tables and atomic observation**

Use three semantic tables: `operations`, `receipts`, and `approvals`. Keep Receipt payload small in SQLite; large/raw output remains in `operation_results` Artifact. `observe_with_receipt()` uses one `BEGIN IMMEDIATE` transaction to verify Operation is `started`, insert/replay-check the Receipt, attach verified Artifact refs, and move Operation to `observed`.

The trusted sequence remains:

```text
PREPARED commit
STARTED commit
external side effect
Artifact write + readback + hash verification
Receipt + OBSERVED atomic commit
derived Evidence commit
COMMITTED commit
```

If derived Evidence fails, leave the Operation `observed`; resume projects Evidence from the stored Receipt/workspace without rerunning the side effect.

- [x] **Step 4: Add fault-injection tests**

```python
def test_evidence_commit_failure_never_repeats_external_side_effect(harness):
    harness.fail_once_after_observed_before_evidence()
    first = harness.run_edit()
    assert first.status is OperationStatus.OBSERVED
    resumed = harness.resume()
    assert harness.external_execution_count == 1
    assert resumed.status is OperationStatus.COMMITTED


def test_unverifiable_started_operation_becomes_unknown_and_blocks_resume(harness):
    harness.seed_started_operation_without_receipt_or_matching_workspace()
    result = harness.reconcile()
    assert result.blocks_agent_invocation
    assert harness.repository.integrity_view("task-1").has_unknown_operations
```

- [x] **Step 5: Convert Receipt and Journal Stores into facades**

`ToolExecutionReceiptStore` retains Artifact read/write helpers and delegates Receipt metadata to `ExecutionRepository`. `OperationJournalStore` delegates lifecycle operations. The old Receipt JSON files are imported once, then never written after the execution migration marker. `OperationReconciler` consumes `ExecutionRepository` but keeps its current recovery algorithm.

Approval decisions are persisted as immutable `ExecutionApproval` records and separately projected as deterministic Approval Evidence; neither TaskRepository nor TaskState becomes approval authority. New approvals receive a deterministic `approval_id` from `task_id`, source Tool Call/interrupt ID, decision, risk, and canonical operation. Historical approvals without source IDs use their original task-local ordinal in the migration ID, so repeated backfill is stable.

`operations.artifact_references` stores path plus verified content hash. The compatibility `OperationJournalEntry.artifact_references: list[str]` projection returns only paths to avoid changing the recovery API during Plan 3; the normalized rows remain the authority for integrity checks.

- [x] **Step 6: Run execution/recovery regressions**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_execution_repository.py tests/investigation/test_receipts.py tests/test_operations.py tests/evaluation/test_fault_injection.py tests/investigation/test_middleware.py -q`

Expected: PASS; parallel Receipt tests are stable and fault injection observes exactly one side effect.

- [x] **Step 7: Commit Task 5**

```powershell
git add src/deepfix/domain_repositories/execution.py src/deepfix/domain_repositories/migration.py src/deepfix/investigation/receipts.py src/deepfix/operations.py src/deepfix/investigation/middleware.py tests/domain_repositories/test_execution_repository.py tests/investigation/test_receipts.py tests/test_operations.py tests/evaluation/test_fault_injection.py
git commit -m "refactor: consolidate trusted execution persistence"
```

### Review Checkpoint C

Pause and report parallel Receipt results, Operation lifecycle counts, Artifact verification, approval migration, and the fault-injection proof that resume never duplicates a side effect.

---

### Task 6: History-Only Snapshot Repository

**Files:**
- Create: `src/deepfix/domain_repositories/history.py`
- Create: `tests/domain_repositories/test_history_repository.py`
- Modify: `src/deepfix/domain_repositories/migration.py`
- Modify: `src/deepfix/compaction/models.py`
- Modify: `src/deepfix/compaction/store.py`
- Modify: `src/deepfix/compaction/snapshot.py`
- Modify: `src/deepfix/compaction/coordinator.py`
- Modify: `tests/compaction/test_store.py`
- Modify: `tests/compaction/test_snapshot_drift.py`
- Modify: `tests/compaction/test_coordinator.py`

**Interfaces:**
- Consumes: `CompactionSnapshot`, coverage/work-unit/message IDs, provenance, Artifact references, active compaction events, current Evidence, and current Investigation records.
- Produces: `HistorySnapshotRecord`, `HistoricalSemanticItem`, `HistoryRepository`, and current-domain-over-history projection.

- [x] **Step 1: Write failing history authority tests**

```python
def test_history_record_contains_coverage_and_sources_but_not_current_domain_tables(tmp_path):
    repository = HistoryRepository(tmp_path / "deepfix.db")
    record = repository.save_prepared(history_record(version=1))
    assert record.coverage.covered_message_ids == ["m-1", "m-2"]
    assert record.artifact_references[0].content_hash == "artifact-hash"
    assert not hasattr(record, "deterministic_evidence")
    assert not hasattr(record, "active_hypotheses")


def test_current_domain_record_overrides_stale_snapshot_item(repositories):
    repositories.history.save_prepared(stale_history_with_hypothesis("h-1", "old"))
    repositories.investigation.record_hypothesis(
        "task-1", hypothesis("h-1", statement="current")
    )
    projected = repositories.history.project_snapshot(
        "task-1", 1,
        evidence=repositories.evidence,
        investigation=repositories.investigation,
    )
    assert hypothesis_text(projected, "h-1") == "current"
    assert only_one_hypothesis(projected, "h-1")
```

- [x] **Step 2: Verify RED**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_history_repository.py -q`

Expected: FAIL because the history-only models and repository are absent.

- [x] **Step 3: Implement history schema and lifecycle**

Create `history_snapshots`, `history_semantic_items`, `history_artifact_refs`, `history_coverage`, `compaction_failures`, and `context_migrations`. `HistorySnapshotRecord` holds lifecycle/input hash/coverage/source work units/artifact refs/content hash; historical facts/hypotheses/questions are separate `HistoricalSemanticItem` rows with stable item IDs and provenance roots.

Do not copy current Evidence or Investigation version counters. Prepared replay is idempotent by `(task_id, input_hash)`; only the compaction event's version can activate a prepared record; active records cannot be abandoned.

- [x] **Step 4: Preserve Artifact-first message replacement atomicity**

Keep the coordinator order exact:

```python
artifact = write_and_verify_conversation_history(messages)
prepared = history.save_prepared(build_history_record(artifact, messages))
validate_history_coverage(prepared, messages)
event = build_compaction_event(prepared, artifact)
active = history.activate(task_id, prepared.version, event)
return replace_messages_only_after_activation(messages, active, event)
```

Any Artifact, Snapshot, or activation failure returns/raises through the existing graded policy with original messages untouched. Do not make Middleware set task lifecycle directly.

- [x] **Step 5: Backfill legacy Compaction rows and make `CompactionStore` a facade**

Migrate Snapshots, failures, and context-migration events. Current Evidence/Hypothesis data embedded in legacy Snapshot JSON becomes historical semantic items; current records are read from Evidence/Investigation repositories during projection. Keep old compaction tables read-only.

- [x] **Step 6: Run history/compaction regressions**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_history_repository.py tests/compaction/test_store.py tests/compaction/test_snapshot.py tests/compaction/test_snapshot_drift.py tests/compaction/test_coordinator.py tests/compaction/test_overflow.py -q`

Expected: PASS; stale history never overrides current facts and original messages survive every injected preparation failure.

- [x] **Step 7: Commit Task 6**

```powershell
git add src/deepfix/domain_repositories/history.py src/deepfix/domain_repositories/migration.py src/deepfix/compaction/models.py src/deepfix/compaction/store.py src/deepfix/compaction/snapshot.py src/deepfix/compaction/coordinator.py tests/domain_repositories/test_history_repository.py tests/compaction/test_store.py tests/compaction/test_snapshot_drift.py tests/compaction/test_coordinator.py
git commit -m "refactor: make compaction history only"
```

### Review Checkpoint D

Pause and report Snapshot lifecycle coverage, stale-history precedence tests, Artifact verification results, and failure-injection proof that original Messages are not removed early.

---

### Task 7: Production Wiring and Domain-by-Domain Legacy Write Disable

**Files:**
- Modify: `src/deepfix/domain_repositories/__init__.py`
- Modify: `src/deepfix/agent.py`
- Modify: `src/deepfix/cli.py`
- Modify: `src/deepfix/context.py`
- Modify: `src/deepfix/protected_context.py`
- Modify: `src/deepfix/service.py`
- Modify: `src/deepfix/reporting.py`
- Create: `tests/domain_repositories/test_service_integration.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_context.py`
- Modify: `tests/test_protected_context.py`
- Modify: `tests/test_reporting.py`
- Modify: `tests/test_service.py`

**Interfaces:**
- Consumes: Tasks 1–6 repositories and Plan 2 `SQLiteDatabase`/`TaskRepository`.
- Produces: one production composition root, current-fact reads from domain authorities, history-only context projection, and field-level legacy write disable.

- [x] **Step 1: Write failing construction and authority tests**

```python
def test_cli_injects_all_repositories_from_one_database(cli_factory):
    app = cli_factory()
    paths = {
        app.tasks.database.path,
        app.evidence.database.path,
        app.investigation.database.path,
        app.execution.database.path,
        app.history.database.path,
    }
    assert len(paths) == 1


def test_context_deduplicates_current_and_historical_ids(app):
    app.seed_current_and_history_with_same_ids(
        constraint_id="c-1", evidence_id="e-1", hypothesis_id="h-1"
    )
    rendered = app.render_protected_context()
    assert rendered.count("c-1") == 1
    assert rendered.count("e-1") == 1
    assert rendered.count("h-1") == 1
```

- [x] **Step 2: Verify RED**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_service_integration.py -q`

Expected: FAIL because production still constructs independent legacy Stores.

- [x] **Step 3: Wire one database and four bounded repositories**

Construct `SQLiteDatabase` once in CLI/service composition. Inject repositories into existing middleware/tools/coordinators; do not modify DeepAgents graph ownership, native Todo middleware, Messages, Checkpointer, interrupt, or resume.

Context assembly order is:

```text
Task Definition / user-message constraints
+ current Investigation
+ current Evidence
+ current Execution integrity/approvals
+ historical detail not shadowed by current IDs
```

Historical projections retain source/provenance and never gain authority by appearing in a Snapshot.

- [x] **Step 4: Disable migrated legacy writes field-by-field**

Add explicit migration markers for `evidence`, `research`, `investigation`, `execution`, and `history`. `TaskRepository.save_legacy_projection()` must omit only fields whose domain marker is switched; unmigrated/Plan 4 compatibility fields continue to round-trip. Tests assert no new writes reach:

```text
deterministic_evidence
research_queries/search_candidates/external_evidence
legacy Receipt JSON files
operation_journal
compaction_snapshots/compaction_failures/context_migrations
```

Do not remove these sources or the legacy `TaskState` fields in this task.

- [x] **Step 5: Run integration regressions**

Run: `.venv\Scripts\python -m pytest tests/domain_repositories/test_service_integration.py tests/test_cli.py tests/test_context.py tests/test_protected_context.py tests/test_reporting.py tests/test_service.py -q`

Expected: PASS; restore/reporting data remains available after every migrated legacy writer is disabled.

- [x] **Step 6: Commit Task 7**

```powershell
git add src/deepfix/domain_repositories/__init__.py src/deepfix/agent.py src/deepfix/cli.py src/deepfix/context.py src/deepfix/protected_context.py src/deepfix/service.py src/deepfix/reporting.py tests/domain_repositories/test_service_integration.py tests/test_cli.py tests/test_context.py tests/test_protected_context.py tests/test_reporting.py tests/test_service.py
git commit -m "refactor: wire bounded domain repositories"
```

---

### Task 8: Plan 3 Migration Gate and Completion Record

**Files:**
- Create: `tests/domain_repositories/test_migration_gate.py`
- Modify: `docs/superpowers/plans/2026-08-28-deepfix-domain-repository-consolidation.md`
- Modify: `docs/superpowers/plans/2026-08-28-deepfix-state-authority-migration-program.md`

**Interfaces:**
- Consumes: all Plan 3 outputs.
- Produces: reproducible offline completion evidence and the exact Plan 4 handoff boundary.

- [x] **Step 1: Add the cross-domain migration invariant test**

```python
@pytest.mark.parametrize("domain", ["evidence", "research", "investigation", "execution", "history"])
def test_migration_report_has_no_identity_hash_or_reference_loss(migrated_fixture, domain):
    report = migrated_fixture.report(domain)
    assert report.source_count == report.target_count
    assert report.identity_mismatches == []
    assert report.hash_mismatches == []
    assert report.missing_references == []
    assert report.ready_to_switch


def test_rollback_sources_are_read_only_after_switch(migrated_fixture):
    before = migrated_fixture.legacy_source_hashes()
    migrated_fixture.run_normal_agent_persistence_cycle()
    assert migrated_fixture.legacy_source_hashes() == before
    assert migrated_fixture.current_repository_counts_increased()
```

- [x] **Step 2: Add restore/reporting and no-duplicate-side-effect gate**

Create a legacy task containing deterministic/external Evidence, hypotheses, an unresolved question, two parallel Receipts, one observed Operation, approvals, and an active Snapshot. Migrate, disable legacy writers, restore, reconcile, render context, and build a report. Assert all stable IDs and Artifact hashes survive, the observed side effect count remains one, and current domain records shadow stale Snapshot items.

- [x] **Step 3: Run the Plan 3 focused gate**

Run:

```powershell
.venv\Scripts\python -m pytest `
  tests/domain_repositories `
  tests/compaction `
  tests/research `
  tests/investigation/test_store.py `
  tests/investigation/test_migration.py `
  tests/investigation/test_receipts.py `
  tests/test_operations.py `
  tests/evaluation/test_fault_injection.py -q
```

Expected: PASS, excluding only already documented environment-specific Windows failures if they reproduce unchanged.

- [x] **Step 4: Run trusted integration regressions**

Run:

```powershell
.venv\Scripts\python -m pytest `
  tests/navigation `
  tests/task_domain `
  tests/test_approval.py `
  tests/test_workspace.py `
  tests/test_backend.py `
  tests/test_execution.py `
  tests/test_context.py `
  tests/test_protected_context.py `
  tests/test_verification.py `
  tests/test_reporting.py `
  tests/test_service.py `
  tests/test_cli.py -q
```

Expected: PASS. DeepAgents Todo/Checkpoint behavior and Workspace/approval/verification boundaries are unchanged.

- [x] **Step 5: Run core offline and static checks**

Run: `.venv\Scripts\python -m pytest --import-mode=importlib -q`

Run: `.venv\Scripts\ruff check src tests`

Run: `git diff --check`

Expected: all offline tests pass except any explicitly documented pre-existing environment failures; Ruff and diff checks exit 0. Do not run paid online evaluation.

- [x] **Step 6: Record exact completion evidence**

Update this document with task commit IDs, per-domain source/target counts, stable-ID/hash/reference validation results, parallel Receipt count, fault-injection execution count, focused/core test totals, and the exact read-only rollback tables/files retained for Plan 4.

- [x] **Step 7: Mark Plan 3 complete in the parent program and commit**

```powershell
git add docs/superpowers/plans/2026-08-28-deepfix-domain-repository-consolidation.md docs/superpowers/plans/2026-08-28-deepfix-state-authority-migration-program.md tests/domain_repositories/test_migration_gate.py
git commit -m "docs: record domain repository consolidation"
```

## Plan 3 Completion Evidence

### Task commits

- Task 1: `360164b` — immutable Evidence repository.
- Task 2: `e2f61b5` — deterministic Evidence authority switch.
- Task 3: `c6e1be2` — external research Evidence consolidation.
- Task 4: `6027945` — Investigation authority consolidation.
- Task 5: `60ba669` — trusted Execution persistence consolidation.
- Task 6: `6862064` — history-only compaction authority.
- Task 7: `98abd8c` — production repository wiring and field-level legacy write disable.
- Task 8 implementation/gate: `5521219` — cross-domain migration gate,
  transactional cutover fence, rollback-field freezing, Receipt cutover guard,
  and crash-recoverable migration leases.
- Completion record: the `docs: record domain repository consolidation` commit
  containing this section.

### Repository ownership and tables

- `EvidenceRepository`: `evidence_records`, `evidence_revisions`,
  `research_attempts`, and `research_candidates`. It owns current deterministic,
  accepted external, and semantic-candidate Evidence plus research-attempt audit.
- `InvestigationRepository`: `hypotheses`, `unresolved_questions`,
  `investigation_conflicts`, `investigation_events`, `strategy_decisions`,
  `experiment_results`, and `experiment_events`. It owns current investigation
  beliefs/questions, never navigation or completion.
- `ExecutionRepository`: `operations`, `receipts`, and `approvals`. Operation,
  Receipt, and Approval remain distinct models under one recovery/idempotency boundary.
- `HistoryRepository`: `history_snapshots`, `history_semantic_items`,
  `history_artifact_refs`, `history_coverage`, `compaction_history_failures`, and
  `context_history_migrations`. It owns historical projection and lifecycle, not
  current facts.
- `domain_migrations` and `domain_migration_fences` are migration coordination
  metadata, not a fifth domain authority. Fences are per task/domain, owner-scoped,
  expire after 15 minutes, and are recoverable both before and after marker commit.

### Cross-domain migration invariant

The reproducible legacy fixture migrates the following exact source/target counts:

| Domain | Source | Target |
|---|---:|---:|
| deterministic Evidence | 2 | 2 |
| Research | 3 | 3 |
| Investigation | 2 | 2 |
| Execution | 4 | 4 |
| History | 2 | 2 |

For every domain, source and target canonical hashes are equal; identity mismatch,
content-hash mismatch, and missing-reference lists are empty. Stable test,
hypothesis, question, Receipt, Operation, research Evidence, and Snapshot identities
survive. The Research and conversation-history `ArtifactReference` values retain
their paths and SHA-256 hashes and are verified against the actual files.

Two different legacy Receipts created concurrently migrate exactly once each.
Four concurrent writes of the same migrated Receipt remain idempotent and keep the
authoritative Receipt count at two. One observed Operation remains one record after
two reconciliation passes. The four real fault-injection scenarios each execute the
side-effect handler exactly once (`duplicate_side_effects == 0`). A writer arriving
during execution migration waits on the task fence and then fails closed after the
authority marker activates; it cannot modify the legacy Receipt rollback files.

Current Evidence and Investigation records with the same stable IDs override stale
Snapshot test/hypothesis projections in restore, Protected Context, reporting, and
history projection. The migrated unresolved question and exact Artifact references
remain available. All ten fields in `MIGRATED_LEGACY_FIELDS` are hashed and proven
unchanged by a normal post-switch persistence cycle, while current repository counts
continue to increase.

### Read-only rollback sources retained for Plan 4

- SQLite: `deterministic_evidence`, `research_queries`, `search_candidates`,
  `external_evidence`, `investigation_state`, `working_memory`,
  `operation_journal`, `compaction_snapshots`, `compaction_failures`,
  `context_migrations`, the historical Plan 2 `tasks` table, and migrated fields
  inside `legacy_task_projection`.
- Files: `artifacts/investigation_receipts/<task>/...` legacy Receipt JSON files.
- New production writes target the bounded repositories. Migrated legacy projection
  fields are frozen; late legacy Receipt writes fail closed. Unmigrated Plan 4
  compatibility fields continue to round-trip.

### Verification results

- Task 8 migration gate: **12 passed**.
- Task 8 migration/service/Receipt focused subset: **25 passed**.
- Plan 3 focused gate: **355 passed, 1 skipped, 2 deselected**; one additional
  long-context test reproduces the pre-existing Windows sandbox `_overlapped` /
  `WinError 10106` failure.
- Trusted integration gate: **290 passed**; its only additional failure reuses the
  same environment-limited long-context fixture.
- Core offline suite (`PYTHONPATH=src;tests`, `--import-mode=importlib`):
  **1046 passed, 2 skipped, 2 deselected**; the only two failures are the same
  long-context fixture and the navigation test that reuses it.
- `ruff check src tests`: PASS.
- `git diff --check`: PASS.
- Independent Task 8 code review: PASS, with no remaining Critical or Important
  findings in the declared single-process Agent boundary.
- No paid online QuixBugs or formal A/B acceptance run was executed.

### Plan 4 handoff boundary

Plan 4 still intentionally owns retirement of `AgentPhase`, PhaseResolver and phase
prompts/tool gating, `WorkingMemoryStore`, giant `TaskState` compatibility fact
fields, report compatibility projection, public `save_progress`, duplicate context
aggregation, and legacy Store class names. None is removed by Plan 3.

## Plan 3 Review Checkpoint

Stop after Task 8 and report:

- the four Repository table lists and ownership boundaries;
- migrated task/record counts and stable-ID/hash/Artifact validation;
- proof that parallel Receipts remain idempotent;
- proof that fault injection never duplicates a side effect;
- proof that current domain records override stale Snapshot projections;
- legacy tables/files retained read-only and writers disabled;
- focused/core test totals and any unchanged environment-only failures;
- Plan 4 dependencies still intentionally retained: Phase, Working Memory, giant TaskState compatibility fields, report facade, public `save_progress`, and legacy class names.

Do not begin Plan 4 until the user reviews and approves this checkpoint.
