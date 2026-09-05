from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from langchain.tools import ToolRuntime
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from deepfix.artifact_retrieval.errors import (
    DiagnosticArtifactSystemError,
    DiagnosticArtifactToolError,
)
from deepfix.artifact_retrieval.models import (
    DiagnosticReadResult,
    DiagnosticSearchResult,
)
from deepfix.artifact_retrieval.service import DiagnosticArtifactService
from deepfix.compaction.identity import stable_generated_message_id
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.errors import InvestigationStateError

_MAX_CONTENT_CHARACTERS = 12_000
_MAX_ERROR_CHARACTERS = 300


def build_search_diagnostic_artifacts_tool(
    service: DiagnosticArtifactService,
    coordinator: InvestigationCoordinator,
) -> BaseTool:
    def search_diagnostic_artifacts(
        query: str,
        runtime: ToolRuntime,
        artifact_kinds: list[str] | None = None,
        max_matches: int = 10,
    ) -> ToolMessage:
        authority = _runtime_authority(runtime)
        if isinstance(authority, ToolMessage):
            return authority
        task_id, call_id, messages = authority
        try:
            result = service.search(
                task_id,
                messages,
                query,
                artifact_kinds,
                max_matches,
            )
        except DiagnosticArtifactToolError as exc:
            return _error_message(
                task_id,
                call_id,
                exc,
                "search_diagnostic_artifacts",
            )
        except DiagnosticArtifactSystemError as exc:
            raise _recovery_error(coordinator, task_id, call_id, exc) from exc
        return _search_message(task_id, call_id, result)

    return StructuredTool.from_function(
        func=search_diagnostic_artifacts,
        name="search_diagnostic_artifacts",
        description=(
            "搜索当前任务已卸载或已压缩的诊断文本；"
            "任务、消息和 Artifact 路径由运行时安全提供。"
        ),
    )


def build_read_diagnostic_artifact_tool(
    service: DiagnosticArtifactService,
    coordinator: InvestigationCoordinator,
) -> BaseTool:
    def read_diagnostic_artifact(
        artifact_id: str,
        runtime: ToolRuntime,
        start_line: int = 1,
        line_count: int = 100,
    ) -> ToolMessage:
        authority = _runtime_authority(runtime)
        if isinstance(authority, ToolMessage):
            return authority
        task_id, call_id, messages = authority
        try:
            result = service.read(
                task_id,
                messages,
                artifact_id,
                start_line,
                line_count,
            )
        except DiagnosticArtifactToolError as exc:
            return _error_message(
                task_id,
                call_id,
                exc,
                "read_diagnostic_artifact",
            )
        except DiagnosticArtifactSystemError as exc:
            raise _recovery_error(coordinator, task_id, call_id, exc) from exc
        return _read_message(task_id, call_id, result)

    return StructuredTool.from_function(
        func=read_diagnostic_artifact,
        name="read_diagnostic_artifact",
        description=(
            "按行读取当前任务已授权的诊断 Artifact；"
            "只能使用搜索结果返回的稳定 artifact_id。"
        ),
    )


def _runtime_authority(
    runtime: ToolRuntime,
) -> tuple[str, str, list[AnyMessage]] | ToolMessage:
    configurable = runtime.config.get("configurable", {})
    task_id = (
        str(configurable.get("thread_id", "")).strip()
        if isinstance(configurable, Mapping)
        else ""
    )
    call_id = str(runtime.tool_call_id or "").strip()
    if not task_id or not call_id:
        error = DiagnosticArtifactToolError(
            "artifact_runtime_invalid",
            "诊断 Artifact 工具缺少任务或调用身份",
        )
        return _error_message(
            task_id or "unknown",
            call_id or "missing-call-id",
            error,
            "diagnostic_artifact_runtime",
        )
    state = runtime.state
    raw_messages = state.get("messages", []) if isinstance(state, Mapping) else []
    messages = (
        list(raw_messages)
        if isinstance(raw_messages, Sequence)
        and not isinstance(raw_messages, (str, bytes))
        else []
    )
    return task_id, call_id, messages


def _search_message(
    task_id: str,
    call_id: str,
    result: DiagnosticSearchResult,
) -> ToolMessage:
    chunks = [
        "\n".join(
            [
                (
                    f"[{match.artifact_id} {match.kind.value} "
                    f"lines {match.start_line}-{match.end_line}]"
                ),
                match.excerpt,
            ]
        )
        for match in result.matches
    ]
    full_content = "\n\n".join(chunks)
    content = full_content[:_MAX_CONTENT_CHARACTERS]
    truncated = result.truncated or len(content) < len(full_content)
    artifact_ids = list(
        dict.fromkeys(match.artifact_id for match in result.matches)
    )
    content_hashes = list(
        dict.fromkeys(match.content_hash for match in result.matches)
    )
    terms_payload = json.dumps(
        result.query_terms,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return ToolMessage(
        id=stable_generated_message_id(
            task_id,
            call_id,
            "diagnostic_artifact_search",
        ),
        content=content,
        tool_call_id=call_id,
        name="search_diagnostic_artifacts",
        status="success",
        artifact={
            "result_type": "diagnostic_artifact_search",
            "artifact_ids": artifact_ids,
            "match_count": len(result.matches),
            "searched_artifact_count": result.searched_artifact_count,
            "omitted_artifact_count": result.omitted_artifact_count,
            "truncated": truncated,
            "content_hashes": content_hashes,
            "query_terms_hash": hashlib.sha256(
                terms_payload.encode("utf-8")
            ).hexdigest(),
        },
    )


def _read_message(
    task_id: str,
    call_id: str,
    result: DiagnosticReadResult,
) -> ToolMessage:
    content = result.content[:_MAX_CONTENT_CHARACTERS]
    truncated = result.truncated or len(content) < len(result.content)
    return ToolMessage(
        id=stable_generated_message_id(
            task_id,
            call_id,
            "diagnostic_artifact_read",
        ),
        content=content,
        tool_call_id=call_id,
        name="read_diagnostic_artifact",
        status="success",
        artifact={
            "result_type": "diagnostic_artifact_read",
            "artifact_id": result.artifact_id,
            "kind": result.kind.value,
            "start_line": result.start_line,
            "end_line": result.end_line,
            "total_lines": result.total_lines,
            "content_hash": result.content_hash,
            "truncated": truncated,
        },
    )


def _error_message(
    task_id: str,
    call_id: str,
    error: DiagnosticArtifactToolError,
    operation: str,
) -> ToolMessage:
    content = f"{operation} 失败：{error.safe_message}"[:_MAX_ERROR_CHARACTERS]
    return ToolMessage(
        id=stable_generated_message_id(task_id, call_id, error.error_code),
        content=content,
        tool_call_id=call_id,
        name=operation,
        status="error",
        artifact={
            "result_type": "diagnostic_artifact_error",
            "error_code": error.error_code,
            "operation": operation,
        },
    )


def _recovery_error(
    coordinator: InvestigationCoordinator,
    task_id: str,
    call_id: str,
    error: DiagnosticArtifactSystemError,
) -> InvestigationStateError:
    return InvestigationStateError(
        coordinator.recovery(
            task_id,
            error.error_code,
            tool_call_id=call_id,
            checkpoint_available=True,
            recovery_action="pause_and_retry_diagnostic_artifact_read",
        )
    )
