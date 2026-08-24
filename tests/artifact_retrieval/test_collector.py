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


def collector(*snapshots):
    return ArtifactReferenceCollector(
        SnapshotStoreStub(snapshots),
        MemoryDownloadBackend(),
    )


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
