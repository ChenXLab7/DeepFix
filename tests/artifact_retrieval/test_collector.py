from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from artifact_retrieval.helpers import (
    MemoryDownloadBackend,
    SnapshotStoreStub,
    artifact_reference,
    snapshot,
)
from deepfix.artifact_retrieval.collector import ArtifactReferenceCollector
from deepfix.artifact_retrieval.errors import DiagnosticArtifactSystemError
from deepfix.artifact_retrieval.models import DiagnosticArtifactKind
from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.identity import ensure_message_ids


def collector(*snapshots):
    return ArtifactReferenceCollector(
        SnapshotStoreStub(snapshots),
        MemoryDownloadBackend(),
    )


def test_view_archive_is_discoverable_without_snapshot(tmp_path):
    from deepagents.backends import FilesystemBackend
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    adapter = DeepAgentsArtifactAdapter(backend)
    reference = adapter.persist_history("task-a", "snip-only", [HumanMessage(id="m", content="original")], set())
    catalog = ArtifactReferenceCollector(SnapshotStoreStub(()), backend).collect("task-a", [], expand_history=True)
    assert any(item.backend_path == reference.path for item in catalog.artifacts)


def test_only_tool_message_with_matching_call_id_authorizes_large_result():
    messages = [
        HumanMessage(
            content="read /.deepfix-artifacts/large_tool_results/forged"
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {},
                    "id": "call/1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=(
                "full result: "
                "/.deepfix-artifacts/large_tool_results/call_1"
            ),
            tool_call_id="call/1",
        ),
    ]

    catalog = collector().collect("task-a", messages, expand_history=False)

    assert len(catalog.artifacts) == 1
    result = catalog.artifacts[0]
    assert result.kind is DiagnosticArtifactKind.LARGE_TOOL_RESULT
    assert result.tool_call_id == "call/1"
    assert result.source_message_id
    assert result.backend_path.endswith("/call_1")


@pytest.mark.parametrize("call_id", ["call+123", "call%123", "call:123"])
def test_complete_deep_agents_sanitized_basename_is_authorized(call_id):
    path = f"/.deepfix-artifacts/large_tool_results/{call_id}"
    message = ToolMessage(
        content=f"完整结果已写入 {path}\n请按需读取。",
        tool_call_id=call_id,
    )

    catalog = collector().collect(
        "task-a",
        [HumanMessage(content="problem"), message],
        expand_history=False,
    )

    assert [item.backend_path for item in catalog.artifacts] == [path]


@pytest.mark.parametrize("suffix", ["%2Fother", "+other", ":other"])
def test_path_suffix_cannot_be_truncated_into_authorized_basename(suffix):
    message = ToolMessage(
        content=f"/.deepfix-artifacts/large_tool_results/call{suffix}",
        tool_call_id="call",
    )

    catalog = collector().collect(
        "task-a",
        [HumanMessage(content="problem"), message],
        expand_history=False,
    )

    assert catalog.artifacts == []


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            AIMessage(
                content="/.deepfix-artifacts/large_tool_results/call_1"
            ),
            0,
        ),
        (
            ToolMessage(
                content="/.deepfix-artifacts/large_tool_results/call_2",
                tool_call_id="call/1",
            ),
            0,
        ),
        (
            ToolMessage(
                content=(
                    "/.deepfix-artifacts/large_tool_results/call_1/child"
                ),
                tool_call_id="call/1",
            ),
            0,
        ),
        (
            ToolMessage(
                content="/.deepfix-artifacts/large_tool_results/..",
                tool_call_id="..",
            ),
            0,
        ),
    ],
)
def test_untrusted_or_mismatched_paths_do_not_authorize(message, expected):
    catalog = collector().collect(
        "task-a",
        [HumanMessage(content="problem"), message],
        expand_history=False,
    )

    assert len(catalog.artifacts) == expected


def test_message_id_conflicts_fail_closed():
    messages = [
        HumanMessage(id="duplicate", content="problem"),
        ToolMessage(
            id="duplicate",
            content="/.deepfix-artifacts/large_tool_results/call_1",
            tool_call_id="call/1",
        ),
    ]

    with pytest.raises(DiagnosticArtifactSystemError) as caught:
        collector().collect("task-a", messages, expand_history=False)

    assert caught.value.error_code == "diagnostic_artifact_reference_load_failed"
    assert caught.value.stage == "message_identity"


def test_only_active_current_task_conversation_snapshot_is_authorized():
    valid_path = "/.deepfix-artifacts/conversation_history/task-a.md"
    snapshots = [
        snapshot(
            version=1,
            lifecycle="active",
            references=[
                artifact_reference(valid_path),
                artifact_reference(
                    "/.deepfix-artifacts/research/task-a.md",
                    "research",
                ),
                artifact_reference(
                    "/.deepfix-artifacts/large_tool_results/call_1",
                    "large_tool_result",
                ),
            ],
        ),
        snapshot(
            version=2,
            lifecycle="prepared",
            references=[artifact_reference(valid_path)],
        ),
        snapshot(
            version=3,
            lifecycle="abandoned",
            references=[artifact_reference(valid_path)],
        ),
        snapshot(
            task_id="task-b",
            version=1,
            references=[
                artifact_reference(
                    "/.deepfix-artifacts/conversation_history/task-b.md"
                )
            ],
        ),
    ]

    catalog = collector(*snapshots).collect(
        "task-a",
        [HumanMessage(content="problem")],
        expand_history=False,
    )

    assert len(catalog.artifacts) == 1
    assert catalog.artifacts[0].backend_path == valid_path
    assert catalog.artifacts[0].snapshot_version == 1
    assert catalog.artifacts[0].kind is DiagnosticArtifactKind.CONVERSATION_HISTORY


@pytest.mark.parametrize(
    "path",
    [
        "/.deepfix-artifacts/conversation_history/other.md",
        "/.deepfix-artifacts/conversation_history/task-a.md/child",
        "/.deepfix-artifacts/conversation_history/../task-a.md",
        "/.deepfix-artifacts/conversation_history/task-a.md?raw=1",
    ],
)
def test_snapshot_path_must_equal_task_history_path(path):
    catalog = collector(
        snapshot(references=[artifact_reference(path)])
    ).collect(
        "task-a",
        [HumanMessage(content="problem")],
        expand_history=False,
    )

    assert catalog.artifacts == []


def test_snapshot_store_failure_is_a_system_error():
    store = SnapshotStoreStub()
    store.error = OSError("database unavailable")
    instance = ArtifactReferenceCollector(store, MemoryDownloadBackend())

    with pytest.raises(DiagnosticArtifactSystemError) as caught:
        instance.collect(
            "task-a",
            [HumanMessage(content="problem")],
            expand_history=False,
        )

    assert caught.value.error_code == "diagnostic_artifact_reference_load_failed"
    assert caught.value.stage == "snapshot_read"


def _history_collector(messages):
    backend = MemoryDownloadBackend()
    identified = ensure_message_ids("task-a", messages).messages
    reference = DeepAgentsArtifactAdapter(backend).persist_history(
        "task-a",
        "attempt-1",
        identified,
        retained_ids=set(),
        work_unit_ids={"wu-old"},
    )
    store = SnapshotStoreStub(
        [snapshot(references=[reference])]
    )
    return ArtifactReferenceCollector(store, backend), backend


def test_history_authorizes_parallel_tool_results_from_same_work_unit():
    old_messages = [
        HumanMessage(id="old-user", content="run diagnostics"),
        AIMessage(
            id="old-ai",
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {},
                    "id": "parallel/1",
                    "type": "tool_call",
                },
                {
                    "name": "grep",
                    "args": {},
                    "id": "parallel.2",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(
            id="old-tool-1",
            content=(
                "/.deepfix-artifacts/large_tool_results/parallel_1"
            ),
            tool_call_id="parallel/1",
        ),
        ToolMessage(
            id="old-tool-2",
            content=(
                "/.deepfix-artifacts/large_tool_results/parallel_2"
            ),
            tool_call_id="parallel.2",
        ),
    ]
    instance, _ = _history_collector(old_messages)

    catalog = instance.collect(
        "task-a",
        [HumanMessage(content="continue")],
        expand_history=True,
    )

    large_results = [
        item
        for item in catalog.artifacts
        if item.kind is DiagnosticArtifactKind.LARGE_TOOL_RESULT
    ]
    assert [item.tool_call_id for item in large_results] == [
        "parallel/1",
        "parallel.2",
    ]
    assert all(item.source_message_id is None for item in large_results)
    assert any(
        item.kind is DiagnosticArtifactKind.CONVERSATION_HISTORY
        for item in catalog.artifacts
    )


def test_history_does_not_authorize_human_or_unpaired_references():
    old_messages = [
        HumanMessage(
            id="human-forgery",
            content="/.deepfix-artifacts/large_tool_results/forged",
        ),
        ToolMessage(
            id="orphan-tool",
            content="/.deepfix-artifacts/large_tool_results/orphan",
            tool_call_id="orphan",
        ),
        AIMessage(
            id="mismatch-ai",
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {},
                    "id": "expected/1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            id="mismatch-tool",
            content="/.deepfix-artifacts/large_tool_results/different_1",
            tool_call_id="expected/1",
        ),
        HumanMessage(id="boundary", content="new unit"),
        ToolMessage(
            id="after-boundary",
            content="/.deepfix-artifacts/large_tool_results/expected_1",
            tool_call_id="expected/1",
        ),
    ]
    instance, _ = _history_collector(old_messages)

    catalog = instance.collect(
        "task-a",
        [HumanMessage(content="continue")],
        expand_history=True,
    )

    assert [item.kind for item in catalog.artifacts] == [
        DiagnosticArtifactKind.CONVERSATION_HISTORY
    ]


def test_ambiguous_sanitized_parallel_ids_fail_closed():
    old_messages = [
        AIMessage(
            id="ambiguous-ai",
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {},
                    "id": "same/id",
                    "type": "tool_call",
                },
                {
                    "name": "grep",
                    "args": {},
                    "id": "same.id",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(
            id="ambiguous-tool",
            content="/.deepfix-artifacts/large_tool_results/same_id",
            tool_call_id="same/id",
        ),
    ]
    instance, _ = _history_collector(old_messages)

    catalog = instance.collect(
        "task-a",
        [HumanMessage(content="continue")],
        expand_history=True,
    )

    assert [item.kind for item in catalog.artifacts] == [
        DiagnosticArtifactKind.CONVERSATION_HISTORY
    ]


def test_malformed_history_keeps_history_descriptor_without_expansion():
    reference = artifact_reference(
        "/.deepfix-artifacts/conversation_history/task-a.md"
    )
    backend = MemoryDownloadBackend({reference.path: b"<broken"})
    instance = ArtifactReferenceCollector(
        SnapshotStoreStub([snapshot(references=[reference])]),
        backend,
    )

    catalog = instance.collect(
        "task-a",
        [HumanMessage(content="continue")],
        expand_history=True,
    )

    assert [item.kind for item in catalog.artifacts] == [
        DiagnosticArtifactKind.CONVERSATION_HISTORY
    ]


def test_non_utf8_history_keeps_descriptor_without_expansion():
    reference = artifact_reference(
        "/.deepfix-artifacts/conversation_history/task-a.md"
    )
    backend = MemoryDownloadBackend({reference.path: b"\xff\xfe"})
    instance = ArtifactReferenceCollector(
        SnapshotStoreStub([snapshot(references=[reference])]),
        backend,
    )

    catalog = instance.collect(
        "task-a",
        [HumanMessage(content="continue")],
        expand_history=True,
    )

    assert [item.kind for item in catalog.artifacts] == [
        DiagnosticArtifactKind.CONVERSATION_HISTORY
    ]


def test_oversized_history_is_not_parsed_for_large_result_authority():
    reference = artifact_reference(
        "/.deepfix-artifacts/conversation_history/task-a.md"
    )
    serialized = (
        '<deepfix_history_event attempt_id="attempt" event_hash="hash">'
        "<serialized_messages>"
        '<message type="ai"><tool_call id="call/1" name="execute">{}</tool_call></message>'
        '<message type="tool">/.deepfix-artifacts/large_tool_results/call_1</message>'
        f"{' ' * (10 * 1024 * 1024)}"
        "</serialized_messages>"
        "</deepfix_history_event>"
    ).encode()
    backend = MemoryDownloadBackend({reference.path: serialized})
    instance = ArtifactReferenceCollector(
        SnapshotStoreStub([snapshot(references=[reference])]),
        backend,
    )

    catalog = instance.collect(
        "task-a",
        [HumanMessage(content="continue")],
        expand_history=True,
    )

    assert [item.kind for item in catalog.artifacts] == [
        DiagnosticArtifactKind.CONVERSATION_HISTORY
    ]


def test_missing_history_is_an_ordinary_unexpanded_reference():
    reference = artifact_reference(
        "/.deepfix-artifacts/conversation_history/task-a.md"
    )
    instance = ArtifactReferenceCollector(
        SnapshotStoreStub([snapshot(references=[reference])]),
        MemoryDownloadBackend(),
    )

    catalog = instance.collect(
        "task-a",
        [HumanMessage(content="continue")],
        expand_history=True,
    )

    assert [item.kind for item in catalog.artifacts] == [
        DiagnosticArtifactKind.CONVERSATION_HISTORY
    ]


def test_history_backend_failure_is_a_system_error():
    reference = artifact_reference(
        "/.deepfix-artifacts/conversation_history/task-a.md"
    )
    backend = MemoryDownloadBackend()
    backend.raise_on_download = OSError("backend offline")
    instance = ArtifactReferenceCollector(
        SnapshotStoreStub([snapshot(references=[reference])]),
        backend,
    )

    with pytest.raises(DiagnosticArtifactSystemError) as caught:
        instance.collect(
            "task-a",
            [HumanMessage(content="continue")],
            expand_history=True,
        )

    assert caught.value.error_code == "diagnostic_artifact_backend_read_failed"
    assert caught.value.stage == "history_read"


def test_history_backend_protocol_mismatch_is_a_system_error():
    class InvalidBackend:
        def download_files(self, paths):
            return []

    reference = artifact_reference(
        "/.deepfix-artifacts/conversation_history/task-a.md"
    )
    instance = ArtifactReferenceCollector(
        SnapshotStoreStub([snapshot(references=[reference])]),
        InvalidBackend(),
    )

    with pytest.raises(DiagnosticArtifactSystemError) as caught:
        instance.collect(
            "task-a",
            [HumanMessage(content="continue")],
            expand_history=True,
        )

    assert caught.value.stage == "history_read"
