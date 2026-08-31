from __future__ import annotations

import sys
from pathlib import Path

from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.models import SystemTestEvidence
from deepfix.compaction.store import CompactionStore
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
from deepfix.investigation.store import InvestigationStore
from deepfix.models import TaskState, TaskStatus
from deepfix.persistence import TaskRepository
from deepfix.research.store import ResearchEvidenceStore


def coordinator_fixture(tmp_path: Path) -> InvestigationCoordinator:
    database = tmp_path / "deepfix.sqlite3"
    tasks = TaskRepository(database)
    tasks.save(
        TaskState(
            task_id="task-a",
            project_root=str(tmp_path),
            project_python=sys.executable,
            user_problem="sign bug",
            approval_mode="manual",
            status=TaskStatus.INVESTIGATING,
            workspace_baseline_id="baseline-task-a",
        )
    )
    compaction = CompactionStore(database)
    return InvestigationCoordinator(
        store=InvestigationStore(database),
        tasks=tasks,
        compaction_store=compaction,
        evidence_collector=EvidenceCollector(
            compaction,
            ResearchEvidenceStore(database),
        ),
    )


def seed_evidence(store: CompactionStore, task_id: str, evidence_id: str) -> None:
    store.save_evidence(
        task_id,
        SystemTestEvidence(
            evidence_id=evidence_id,
            command="python -m pytest -q",
            exit_code=1,
            summary="1 failed",
            tool_call_id=f"call-{evidence_id}",
            source_message_id=f"msg-{evidence_id}",
        ),
    )


def seed_checked_location(
    store: InvestigationStore,
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
