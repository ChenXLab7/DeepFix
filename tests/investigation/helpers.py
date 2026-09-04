from __future__ import annotations

from pathlib import Path

from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.models import SystemTestEvidence
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.evidence import EvidenceRepository
from deepfix.domain_repositories.investigation import InvestigationRepository
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import (
    CheckedFile,
    CheckedLocation,
    InvestigationHypothesis,
    NewInvestigationEvent,
    ProposedChange,
    RecordHypothesisInput,
    ScopeKind,
)
from deepfix.task_domain.models import TaskDefinition


def coordinator_fixture(tmp_path: Path) -> InvestigationCoordinator:
    database = tmp_path / "deepfix.sqlite3"
    repositories = DomainRepositories.create(database)
    repositories.tasks.create_definition(
        TaskDefinition(
            task_id="task-a",
            original_message_id="message-user",
            original_problem="sign bug",
            approval_mode="manual",
            source_project_root=str(tmp_path),
            workspace_root=str(tmp_path),
            workspace_baseline_id="baseline-task-a",
            project_python="python",
            confinement_level="workspace",
            created_at="2026-08-31T00:00:00+00:00",
        )
    )
    return InvestigationCoordinator(
        store=repositories.investigation,
        tasks=repositories.tasks,
        evidence_repository=repositories.evidence,
        evidence_collector=EvidenceCollector(repositories.evidence),
    )


def seed_evidence(store: EvidenceRepository, task_id: str, evidence_id: str) -> None:
    evidence = SystemTestEvidence(
        evidence_id=evidence_id,
        command="python -m pytest -q",
        exit_code=1,
        summary="1 failed",
        tool_call_id=f"call-{evidence_id}",
        source_message_id=f"msg-{evidence_id}",
    )
    store.record_deterministic(
        task_id,
        evidence,
        provenance_root_ids=[evidence.evidence_id],
    )


def seed_checked_location(
    store: InvestigationRepository,
    task_id: str,
    path: str,
    start_line: int,
    end_line: int,
) -> None:
    state = store.ensure_started(task_id)
    event_id = stable_investigation_id("event", task_id, "checked", path)
    checked = CheckedFile(
        path=path,
        content_fingerprint="f" * 64,
        ranges=[
            CheckedLocation(
                path=path,
                start_line=start_line,
                end_line=end_line,
            )
        ],
        scope=ScopeKind.DIRECT,
        first_event_id=event_id,
        latest_event_id=event_id,
    )
    store.commit(
        state.version,
        [
            NewInvestigationEvent(
                event_id=event_id,
                task_id=task_id,
                event_type="file_checked",
            )
        ],
        state.model_copy(update={"checked_files": [checked]}),
    )


def supported_input(evidence_ids: list[str]) -> RecordHypothesisInput:
    return RecordHypothesisInput(
        statement="sign is inverted twice",
        evidence_ids=evidence_ids,
        checked_locations=[
            CheckedLocation(path="src/sign.py", start_line=1, end_line=20)
        ],
        proposed_change=ProposedChange(
            path="src/sign.py",
            description="remove duplicate inversion",
        ),
        expected_effect="sign=-1 applies exactly one inversion",
        target_state="supported",
        reason="the failing branch and assertion agree",
    )


def stagnated_coordinator(tmp_path: Path) -> InvestigationCoordinator:
    coordinator = coordinator_fixture(tmp_path)
    state = coordinator.store.ensure_started("task-a")
    hypothesis = InvestigationHypothesis(
        hypothesis_id="hyp-1",
        statement="sign flips in helper",
        state="candidate",
        evidence_ids=[],
        checked_locations=[],
        reason="candidate for one bounded follow-up",
    )
    event_id = stable_investigation_id("event", "task-a", "stagnated-fixture")
    coordinator.store.commit(
        state.version,
        [
            NewInvestigationEvent(
                event_id=event_id,
                task_id="task-a",
                event_type="reevaluation_required",
            )
        ],
        state.model_copy(
            update={
                "hypotheses": [hypothesis],
                "reevaluation_required": True,
                "stagnation_level": 1,
            }
        ),
    )
    return coordinator
