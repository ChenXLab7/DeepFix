from __future__ import annotations

from types import SimpleNamespace

from deepfix.compaction.models import (
    ArtifactReference,
    CompactionSnapshot,
    DeterministicEvidenceBlock,
)


class SnapshotStoreStub:
    def __init__(self, snapshots=()):
        self.snapshots = list(snapshots)
        self.error: Exception | None = None

    def list_snapshots(self, task_id: str):
        if self.error is not None:
            raise self.error
        return list(self.snapshots)


class MemoryDownloadBackend:
    def __init__(self, files: dict[str, bytes] | None = None):
        self.files = dict(files or {})
        self.raise_on_download: Exception | None = None

    def download_files(self, paths: list[str]):
        if self.raise_on_download is not None:
            raise self.raise_on_download
        return [
            SimpleNamespace(
                path=path,
                content=self.files.get(path),
                error=None if path in self.files else "file_not_found",
            )
            for path in paths
        ]


def artifact_reference(
    path: str,
    kind: str = "conversation_history",
) -> ArtifactReference:
    return ArtifactReference(
        path=path,
        kind=kind,
        content_hash="a" * 64,
        work_unit_ids=[],
    )


def snapshot(
    *,
    task_id: str = "task-a",
    version: int = 1,
    lifecycle: str = "active",
    references: list[ArtifactReference] | None = None,
) -> CompactionSnapshot:
    return CompactionSnapshot(
        task_id=task_id,
        version=version,
        previous_version=version - 1 or None,
        lifecycle=lifecycle,
        created_at=f"2026-08-24T00:00:{version:02d}+00:00",
        activated_at=(
            "2026-08-24T00:01:00+00:00" if lifecycle == "active" else None
        ),
        abandoned_at=(
            "2026-08-24T00:02:00+00:00" if lifecycle == "abandoned" else None
        ),
        abandon_reason="test" if lifecycle == "abandoned" else None,
        source_work_unit_ids=[],
        task_goal="repair bug",
        user_constraints=[],
        confirmed_facts=[],
        deterministic_evidence=DeterministicEvidenceBlock(),
        active_hypotheses=[],
        rejected_hypotheses=[],
        confirmed_hypotheses=[],
        changed_files=[],
        experiments=[],
        test_results=[],
        conflicts=[],
        unresolved_questions=[],
        next_steps=[],
        artifact_references=list(references or []),
        content_hash="b" * 64,
    )
