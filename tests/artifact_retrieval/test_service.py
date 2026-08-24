from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from artifact_retrieval.helpers import MemoryDownloadBackend
from deepfix.artifact_retrieval.errors import (
    DiagnosticArtifactSystemError,
    DiagnosticArtifactToolError,
)
from deepfix.artifact_retrieval.models import (
    DiagnosticArtifactCatalog,
    DiagnosticArtifactDescriptor,
    DiagnosticArtifactKind,
    stable_diagnostic_artifact_id,
)
from deepfix.artifact_retrieval.service import DiagnosticArtifactService
from deepfix.models import TaskState, TaskStatus
from deepfix.persistence import TaskRepository


class CatalogCollectorStub:
    def __init__(self, artifacts):
        self.catalog = DiagnosticArtifactCatalog(artifacts=list(artifacts))
        self.calls = []

    def collect(self, task_id, messages, *, expand_history):
        self.calls.append((task_id, list(messages), expand_history))
        return self.catalog


def descriptor(
    path: str,
    *,
    kind: DiagnosticArtifactKind = DiagnosticArtifactKind.LARGE_TOOL_RESULT,
    task_id: str = "task-a",
) -> DiagnosticArtifactDescriptor:
    return DiagnosticArtifactDescriptor(
        artifact_id=stable_diagnostic_artifact_id(task_id, kind, path),
        task_id=task_id,
        kind=kind,
        backend_path=path,
    )


def service_fixture(tmp_path, files: dict[str, bytes], artifacts=None):
    repository = TaskRepository(tmp_path / "deepfix.sqlite3")
    repository.save(
        TaskState(
            task_id="task-a",
            project_root=str(tmp_path),
            project_python=sys.executable,
            user_problem="sign bug",
            approval_mode="manual",
            status=TaskStatus.INVESTIGATING,
        )
    )
    selected = list(
        [descriptor(path) for path in files]
        if artifacts is None
        else artifacts
    )
    collector = CatalogCollectorStub(selected)
    backend = MemoryDownloadBackend(files)
    return (
        DiagnosticArtifactService(repository, collector, backend),
        collector,
        backend,
    )


def messages():
    return [HumanMessage(id="user-current", content="continue")]


def test_search_uses_literal_casefolded_and_with_five_line_window(tmp_path):
    path = "/.deepfix-artifacts/large_tool_results/call_1"
    content = "VALUE[0]\ncontext\nmiddle\ncontext\nFAILED\ntail"
    service, collector, _ = service_fixture(
        tmp_path,
        {path: content.encode()},
    )

    result = service.search(
        "task-a",
        messages(),
        "value[0] failed",
        None,
        10,
    )

    assert result.query_terms == ["value[0]", "failed"]
    assert [(item.start_line, item.end_line) for item in result.matches] == [
        (1, 5)
    ]
    assert "1: VALUE[0]" in result.matches[0].excerpt
    assert "5: FAILED" in result.matches[0].excerpt
    assert result.matches[0].content_hash == hashlib.sha256(
        content.encode()
    ).hexdigest()
    assert collector.calls[0][2] is True


def test_overlapping_windows_merge_and_artifacts_keep_catalog_order(tmp_path):
    first_path = "/.deepfix-artifacts/large_tool_results/a"
    second_path = "/.deepfix-artifacts/large_tool_results/b"
    first = "failure\na\nfailure\nb\nfailure\nc\nd\ne"
    second = "a\nb\nc\nfailure\nd\ne\nf\ng"
    artifacts = [descriptor(second_path), descriptor(first_path)]
    service, _, _ = service_fixture(
        tmp_path,
        {first_path: first.encode(), second_path: second.encode()},
        artifacts,
    )

    result = service.search("task-a", messages(), "failure", None, 10)

    assert [item.artifact_id for item in result.matches] == [
        artifacts[0].artifact_id,
        artifacts[1].artifact_id,
    ]
    assert [(item.start_line, item.end_line) for item in result.matches] == [
        (2, 6),
        (1, 7),
    ]


@pytest.mark.parametrize(
    ("query", "kinds", "max_matches", "error_code"),
    [
        ("", None, 10, "artifact_query_invalid"),
        ("x" * 201, None, 10, "artifact_query_invalid"),
        ("1 2 3 4 5 6 7 8 9", None, 10, "artifact_query_invalid"),
        ("failure", [], 10, "artifact_kind_invalid"),
        ("failure", ["research"], 10, "artifact_kind_invalid"),
        (
            "failure",
            ["large_tool_result", "large_tool_result"],
            10,
            "artifact_kind_invalid",
        ),
        ("failure", None, 0, "artifact_match_limit_invalid"),
        ("failure", None, 21, "artifact_match_limit_invalid"),
    ],
)
def test_search_rejects_invalid_public_inputs(
    tmp_path,
    query,
    kinds,
    max_matches,
    error_code,
):
    path = "/.deepfix-artifacts/large_tool_results/a"
    service, _, _ = service_fixture(tmp_path, {path: b"failure"})

    with pytest.raises(DiagnosticArtifactToolError) as caught:
        service.search(
            "task-a",
            messages(),
            query,
            kinds,
            max_matches,
        )

    assert caught.value.error_code == error_code


def test_search_processes_only_first_32_selected_artifacts(tmp_path):
    paths = [f"/.deepfix-artifacts/large_tool_results/call_{i:02d}" for i in range(33)]
    files = {path: f"failure {index}".encode() for index, path in enumerate(paths)}
    service, _, _ = service_fixture(
        tmp_path,
        files,
        [descriptor(path) for path in paths],
    )

    result = service.search("task-a", messages(), "failure", None, 20)

    assert result.searched_artifact_count == 32
    assert result.omitted_artifact_count == 1
    assert len(result.matches) == 20
    assert result.truncated is True


def test_search_truncates_first_long_excerpt_at_character_budget(tmp_path):
    path = "/.deepfix-artifacts/large_tool_results/long"
    service, _, _ = service_fixture(
        tmp_path,
        {path: ("failure" + "界" * 13_000).encode()},
    )

    result = service.search("task-a", messages(), "failure", None, 10)

    assert len(result.matches) == 1
    assert len(result.matches[0].excerpt) == 12_000
    assert result.truncated is True


def test_search_kind_filter_preserves_catalog_order(tmp_path):
    history_path = "/.deepfix-artifacts/conversation_history/task-a.md"
    large_path = "/.deepfix-artifacts/large_tool_results/call_1"
    history = descriptor(
        history_path,
        kind=DiagnosticArtifactKind.CONVERSATION_HISTORY,
    )
    large = descriptor(large_path)
    service, _, _ = service_fixture(
        tmp_path,
        {history_path: b"failure history", large_path: b"failure large"},
        [large, history],
    )

    result = service.search(
        "task-a",
        messages(),
        "failure",
        ["conversation_history"],
        10,
    )

    assert [item.artifact_id for item in result.matches] == [history.artifact_id]
    assert result.searched_artifact_count == 1


def test_search_empty_catalog_and_no_match_are_tool_errors(tmp_path):
    empty_service, _, _ = service_fixture(tmp_path, {}, [])
    path = "/.deepfix-artifacts/large_tool_results/call_1"
    no_match_service, _, _ = service_fixture(tmp_path, {path: b"all good"})

    with pytest.raises(DiagnosticArtifactToolError) as empty:
        empty_service.search("task-a", messages(), "failure", None, 10)
    with pytest.raises(DiagnosticArtifactToolError) as no_match:
        no_match_service.search("task-a", messages(), "failure", None, 10)

    assert empty.value.error_code == "artifact_catalog_empty"
    assert no_match.value.error_code == "artifact_no_matches"


def test_read_returns_numbered_range_hash_and_truncation(tmp_path):
    path = "/.deepfix-artifacts/large_tool_results/call_1"
    raw = b"zero\none\ntwo\nthree\n"
    item = descriptor(path)
    service, _, _ = service_fixture(tmp_path, {path: raw}, [item])

    result = service.read(
        "task-a",
        messages(),
        item.artifact_id,
        start_line=2,
        line_count=2,
    )

    assert result.start_line == 2
    assert result.end_line == 3
    assert result.total_lines == 4
    assert result.content == "2: one\n3: two"
    assert result.content_hash == hashlib.sha256(raw).hexdigest()
    assert result.truncated is True


def test_read_empty_artifact_has_stable_empty_range(tmp_path):
    path = "/.deepfix-artifacts/large_tool_results/empty"
    item = descriptor(path)
    service, _, _ = service_fixture(tmp_path, {path: b""}, [item])

    result = service.read(
        "task-a",
        messages(),
        item.artifact_id,
        start_line=1,
        line_count=100,
    )

    assert result.start_line == 1
    assert result.end_line == 0
    assert result.total_lines == 0
    assert result.content == ""
    assert result.truncated is False


def test_read_truncates_content_at_character_budget(tmp_path):
    path = "/.deepfix-artifacts/large_tool_results/long-read"
    item = descriptor(path)
    service, _, _ = service_fixture(
        tmp_path,
        {path: ("界" * 13_000).encode()},
        [item],
    )

    result = service.read(
        "task-a",
        messages(),
        item.artifact_id,
        start_line=1,
        line_count=100,
    )

    assert len(result.content) == 12_000
    assert result.truncated is True


@pytest.mark.parametrize(
    ("artifact_id", "start_line", "line_count", "error_code"),
    [
        ("artifact_" + "f" * 32, 1, 100, "artifact_not_authorized"),
        (None, 0, 100, "artifact_line_range_invalid"),
        (None, 1, 0, "artifact_line_range_invalid"),
        (None, 1, 201, "artifact_line_range_invalid"),
        (None, 3, 1, "artifact_line_range_invalid"),
    ],
)
def test_read_rejects_unknown_id_and_invalid_ranges(
    tmp_path,
    artifact_id,
    start_line,
    line_count,
    error_code,
):
    path = "/.deepfix-artifacts/large_tool_results/call_1"
    item = descriptor(path)
    service, _, _ = service_fixture(tmp_path, {path: b"one\ntwo"}, [item])

    with pytest.raises(DiagnosticArtifactToolError) as caught:
        service.read(
            "task-a",
            messages(),
            artifact_id or item.artifact_id,
            start_line,
            line_count,
        )

    assert caught.value.error_code == error_code


@pytest.mark.parametrize(
    ("content_kind", "error_code"),
    [
        ("missing", "artifact_file_not_found"),
        ("non_utf8", "artifact_not_utf8"),
        ("oversized", "artifact_too_large"),
    ],
)
def test_download_content_errors_are_recoverable_tool_errors(
    tmp_path,
    content_kind,
    error_code,
):
    path = "/.deepfix-artifacts/large_tool_results/call_1"
    item = descriptor(path)
    contents = {
        "non_utf8": b"\xff",
        "oversized": b"x" * (10 * 1024 * 1024 + 1),
    }
    files = {} if content_kind == "missing" else {path: contents[content_kind]}
    service, _, _ = service_fixture(tmp_path, files, [item])

    with pytest.raises(DiagnosticArtifactToolError) as caught:
        service.read("task-a", messages(), item.artifact_id, 1, 100)

    assert caught.value.error_code == error_code


def test_backend_exception_and_protocol_mismatch_are_system_errors(tmp_path):
    path = "/.deepfix-artifacts/large_tool_results/call_1"
    item = descriptor(path)
    service, _, backend = service_fixture(tmp_path, {path: b"failure"}, [item])
    backend.raise_on_download = OSError("offline")

    with pytest.raises(DiagnosticArtifactSystemError) as raised:
        service.read("task-a", messages(), item.artifact_id, 1, 100)
    assert raised.value.error_code == "diagnostic_artifact_backend_read_failed"

    service.backend = SimpleNamespace(download_files=lambda paths: [])
    with pytest.raises(DiagnosticArtifactSystemError) as malformed:
        service.read("task-a", messages(), item.artifact_id, 1, 100)
    assert malformed.value.stage == "backend_read"

    service.backend = SimpleNamespace(
        download_files=lambda paths: [
            SimpleNamespace(path="wrong", content=b"failure", error=None)
        ]
    )
    with pytest.raises(DiagnosticArtifactSystemError):
        service.read("task-a", messages(), item.artifact_id, 1, 100)

    service.backend = SimpleNamespace(
        download_files=lambda paths: [SimpleNamespace(path=paths[0])]
    )
    with pytest.raises(DiagnosticArtifactSystemError):
        service.read("task-a", messages(), item.artifact_id, 1, 100)


def test_task_repository_failure_is_a_reference_system_error(tmp_path):
    path = "/.deepfix-artifacts/large_tool_results/call_1"
    service, _, _ = service_fixture(tmp_path, {path: b"failure"})

    with pytest.raises(DiagnosticArtifactSystemError) as caught:
        service.search("unknown-task", messages(), "failure", None, 10)

    assert caught.value.error_code == "diagnostic_artifact_reference_load_failed"
    assert caught.value.stage == "task_read"
