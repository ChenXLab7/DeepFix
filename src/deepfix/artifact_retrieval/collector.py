from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any
from xml.etree import ElementTree

from langchain_core.messages import AnyMessage, ToolMessage

from deepfix.artifact_retrieval.errors import DiagnosticArtifactSystemError
from deepfix.artifact_retrieval.models import (
    DiagnosticArtifactCatalog,
    DiagnosticArtifactDescriptor,
    DiagnosticArtifactKind,
    stable_diagnostic_artifact_id,
)
from deepfix.compaction.identity import ensure_message_ids
from deepfix.compaction.store import CompactionStore

_LARGE_ROOT = "/.deepfix-artifacts/large_tool_results/"
_HISTORY_ROOT = "/.deepfix-artifacts/conversation_history/"
_MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
_LARGE_PATH = re.compile(
    re.escape(_LARGE_ROOT) + r"(?P<basename>[A-Za-z0-9_-]+)(?![A-Za-z0-9_./\\?#*])"
)


class ArtifactReferenceCollector:
    def __init__(self, compaction_store: CompactionStore, backend: Any) -> None:
        self.compaction_store = compaction_store
        self.backend = backend

    def collect(
        self,
        task_id: str,
        messages: Sequence[AnyMessage],
        *,
        expand_history: bool,
    ) -> DiagnosticArtifactCatalog:
        normalized_task_id = _required(task_id, "task_id")
        try:
            identities = ensure_message_ids(normalized_task_id, messages)
        except Exception as exc:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_reference_load_failed",
                "message_identity",
            ) from exc
        if identities.conflicted_message_ids:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_reference_load_failed",
                "message_identity",
            )

        descriptors = self._from_current_messages(
            normalized_task_id,
            identities.messages,
        )
        descriptors.extend(self._from_active_snapshots(normalized_task_id))
        if expand_history:
            descriptors.extend(
                self._from_history(normalized_task_id, descriptors)
            )
        return DiagnosticArtifactCatalog(
            artifacts=_deduplicate_and_sort(descriptors)
        )

    @staticmethod
    def _from_current_messages(
        task_id: str,
        messages: Sequence[AnyMessage],
    ) -> list[DiagnosticArtifactDescriptor]:
        descriptors: list[DiagnosticArtifactDescriptor] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            tool_call_id = str(message.tool_call_id or "").strip()
            source_message_id = str(message.id or "").strip()
            if not tool_call_id or not source_message_id:
                continue
            expected_basename = _sanitize_tool_call_id(tool_call_id)
            matching_paths = [
                path
                for path in _strict_large_result_paths(_message_text(message))
                if path.rsplit("/", 1)[-1] == expected_basename
            ]
            if len(matching_paths) != 1:
                continue
            path = matching_paths[0]
            descriptors.append(
                _descriptor(
                    task_id,
                    DiagnosticArtifactKind.LARGE_TOOL_RESULT,
                    path,
                    source_message_id=source_message_id,
                    tool_call_id=tool_call_id,
                )
            )
        return descriptors

    def _from_active_snapshots(
        self,
        task_id: str,
    ) -> list[DiagnosticArtifactDescriptor]:
        try:
            snapshots = self.compaction_store.list_snapshots(task_id)
        except Exception as exc:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_reference_load_failed",
                "snapshot_read",
            ) from exc

        expected_path = f"{_HISTORY_ROOT}{task_id}.md"
        descriptors: list[DiagnosticArtifactDescriptor] = []
        for snapshot in snapshots:
            if snapshot.task_id != task_id or snapshot.lifecycle != "active":
                continue
            for reference in snapshot.artifact_references:
                if (
                    reference.kind != "conversation_history"
                    or reference.path != expected_path
                ):
                    continue
                descriptors.append(
                    _descriptor(
                        task_id,
                        DiagnosticArtifactKind.CONVERSATION_HISTORY,
                        expected_path,
                        snapshot_version=snapshot.version,
                    )
                )
        return descriptors

    def _from_history(
        self,
        task_id: str,
        descriptors: Sequence[DiagnosticArtifactDescriptor],
    ) -> list[DiagnosticArtifactDescriptor]:
        history_paths = sorted(
            {
                item.backend_path
                for item in descriptors
                if item.kind is DiagnosticArtifactKind.CONVERSATION_HISTORY
            }
        )
        expanded: list[DiagnosticArtifactDescriptor] = []
        for history_path in history_paths:
            content = self._download_history(history_path)
            if content is None:
                continue
            for tool_call_id, backend_path in _history_large_result_pairs(content):
                expanded.append(
                    _descriptor(
                        task_id,
                        DiagnosticArtifactKind.LARGE_TOOL_RESULT,
                        backend_path,
                        tool_call_id=tool_call_id,
                    )
                )
        return expanded

    def _download_history(self, path: str) -> str | None:
        try:
            responses = self.backend.download_files([path])
        except Exception as exc:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_backend_read_failed",
                "history_read",
            ) from exc
        if not isinstance(responses, list) or len(responses) != 1:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_backend_read_failed",
                "history_read",
            )
        response = responses[0]
        if not all(hasattr(response, name) for name in ("path", "content", "error")):
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_backend_read_failed",
                "history_read",
            )
        if str(response.path) != path:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_backend_read_failed",
                "history_read",
            )
        if response.error is not None:
            if str(response.error) == "file_not_found":
                return None
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_backend_read_failed",
                "history_read",
            )
        if not isinstance(response.content, bytes):
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_backend_read_failed",
                "history_read",
            )
        if len(response.content) > _MAX_ARTIFACT_BYTES:
            return None
        try:
            return response.content.decode("utf-8")
        except UnicodeDecodeError:
            return None


def _descriptor(
    task_id: str,
    kind: DiagnosticArtifactKind,
    backend_path: str,
    *,
    source_message_id: str | None = None,
    tool_call_id: str | None = None,
    snapshot_version: int | None = None,
) -> DiagnosticArtifactDescriptor:
    return DiagnosticArtifactDescriptor(
        artifact_id=stable_diagnostic_artifact_id(
            task_id,
            kind,
            backend_path,
        ),
        task_id=task_id,
        kind=kind,
        backend_path=backend_path,
        source_message_id=source_message_id,
        tool_call_id=tool_call_id,
        snapshot_version=snapshot_version,
    )


def _deduplicate_and_sort(
    descriptors: Sequence[DiagnosticArtifactDescriptor],
) -> list[DiagnosticArtifactDescriptor]:
    by_id: dict[str, DiagnosticArtifactDescriptor] = {}
    for descriptor in descriptors:
        existing = by_id.get(descriptor.artifact_id)
        if existing is not None:
            if (
                existing.kind != descriptor.kind
                or existing.backend_path != descriptor.backend_path
                or existing.task_id != descriptor.task_id
            ):
                raise DiagnosticArtifactSystemError(
                    "diagnostic_artifact_reference_load_failed",
                    "artifact_id_collision",
                )
            continue
        by_id[descriptor.artifact_id] = descriptor
    return sorted(
        by_id.values(),
        key=lambda item: (item.kind.value, item.backend_path),
    )


def _sanitize_tool_call_id(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace("\\", "_")


def _strict_large_result_paths(text: str) -> list[str]:
    return [match.group(0) for match in _LARGE_PATH.finditer(text)]


def _history_large_result_pairs(content: str) -> list[tuple[str, str]]:
    try:
        root = ElementTree.fromstring(f"<deepfix_history>{content}</deepfix_history>")
    except ElementTree.ParseError:
        return []

    pairs: list[tuple[str, str]] = []
    for event in root.findall("deepfix_history_event"):
        serialized = event.find("serialized_messages")
        if serialized is None:
            continue
        pairs.extend(_serialized_message_pairs(serialized))
    return pairs


def _serialized_message_pairs(
    serialized_messages: ElementTree.Element,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    active_call_ids: set[str] = set()
    for message in serialized_messages.findall("message"):
        message_type = str(message.get("type") or "")
        if message_type in {"ai", "human"}:
            active_call_ids = (
                {
                    call_id
                    for call in message.findall("tool_call")
                    if (call_id := str(call.get("id") or "").strip())
                }
                if message_type == "ai"
                else set()
            )
            continue
        if message_type != "tool" or not active_call_ids:
            continue
        text = "".join(message.itertext())
        for backend_path in _strict_large_result_paths(text):
            basename = backend_path.rsplit("/", 1)[-1]
            matching = [
                call_id
                for call_id in active_call_ids
                if _sanitize_tool_call_id(call_id) == basename
            ]
            if len(matching) == 1:
                pairs.append((matching[0], backend_path))
    return pairs


def _message_text(message: ToolMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return str(message.content)


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} 不能为空")
    return normalized
