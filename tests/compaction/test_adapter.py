import asyncio
from copy import deepcopy

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import FileDownloadResponse, WriteResult
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepfix.compaction.adapter import DeepAgentsArtifactAdapter
from deepfix.compaction.errors import ArtifactPersistenceError


def _messages():
    return [
        HumanMessage(id="m1", content="fix parser"),
        AIMessage(
            id="m2",
            content="inspect",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "src/parser.py"},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(id="m3", content="source", tool_call_id="c1"),
        AIMessage(id="m4", content="the branch is wrong"),
    ]


def test_history_contains_every_message_and_manifests(tmp_path):
    adapter = DeepAgentsArtifactAdapter(
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    )
    messages = _messages()

    ref = adapter.persist_history(
        "task-a",
        "attempt-1",
        messages,
        {"m4"},
        work_unit_ids={"wu-1"},
    )
    body = adapter.read_verified(ref.path)

    assert all(message.id in body for message in messages)
    assert 'attempt_id="attempt-1"' in body
    assert '<retained_message id="m4"' in body
    assert '<compressed_message id="m1"' in body
    assert ref.work_unit_ids == ["wu-1"]


def test_same_attempt_does_not_append_twice(tmp_path):
    adapter = DeepAgentsArtifactAdapter(
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    )
    messages = _messages()

    first = adapter.persist_history("task-a", "attempt-1", messages, set())
    second = adapter.persist_history("task-a", "attempt-1", messages, set())
    body = adapter.read_verified(first.path)

    assert first.content_hash == second.content_hash
    assert body.count('attempt_id="attempt-1"') == 1


def test_async_history_path_uses_async_backend_methods(tmp_path):
    adapter = DeepAgentsArtifactAdapter(
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    )

    ref = asyncio.run(
        adapter.apersist_history(
            "task-a",
            "attempt-async",
            _messages(),
            {"m4"},
        )
    )

    assert 'attempt_id="attempt-async"' in adapter.read_verified(ref.path)


class _ControlledBackend:
    def __init__(self, mode):
        self.mode = mode
        self.content = None
        self.download_count = 0

    def download_files(self, paths):
        self.download_count += 1
        if self.content is None or (
            self.mode == "missing_after_write" and self.download_count > 1
        ):
            return [FileDownloadResponse(path=paths[0], error="file_not_found")]
        content = self.content
        if self.mode == "corrupt_after_write" and self.download_count > 1:
            content = content.replace(b"fix parser", b"fix parseX")
        return [FileDownloadResponse(path=paths[0], content=content)]

    def write(self, path, content):
        if self.mode == "write_error":
            return WriteResult(error="controlled write failure")
        self.content = content.encode()
        return WriteResult(path=path)

    def edit(self, path, old, new, replace_all=False):
        raise RuntimeError("controlled edit failure")


@pytest.mark.parametrize(
    ("mode", "error_code"),
    [
        ("write_error", "artifact_write_failed"),
        ("missing_after_write", "artifact_verify_failed"),
        ("corrupt_after_write", "artifact_hash_mismatch"),
    ],
)
def test_backend_failure_preserves_input_messages(mode, error_code):
    messages = _messages()
    original = deepcopy(messages)
    adapter = DeepAgentsArtifactAdapter(_ControlledBackend(mode))

    with pytest.raises(ArtifactPersistenceError, match=error_code):
        adapter.persist_history("task-a", "attempt-1", messages, set())

    assert messages == original


def test_edit_exception_is_typed_and_preserves_messages():
    backend = _ControlledBackend("edit_error")
    backend.content = b"existing"
    messages = _messages()
    original = deepcopy(messages)

    with pytest.raises(ArtifactPersistenceError, match="artifact_write_failed"):
        DeepAgentsArtifactAdapter(backend).persist_history(
            "task-a", "attempt-1", messages, set()
        )

    assert messages == original


def test_adapter_does_not_call_private_summarization_helpers(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("private summarization helper called")

    for name in (
        "_create_summary",
        "_acreate_summary",
        "_determine_cutoff_index",
        "_offload_to_backend",
    ):
        monkeypatch.setattr(SummarizationMiddleware, name, forbidden)

    adapter = DeepAgentsArtifactAdapter(
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    )

    assert adapter.persist_history(
        "task-a", "attempt-1", _messages(), set()
    ).content_hash
