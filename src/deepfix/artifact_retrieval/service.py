from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.messages import AnyMessage

from deepfix.artifact_retrieval.collector import ArtifactReferenceCollector
from deepfix.artifact_retrieval.errors import (
    DiagnosticArtifactSystemError,
    DiagnosticArtifactToolError,
)
from deepfix.artifact_retrieval.models import (
    DiagnosticArtifactCatalog,
    DiagnosticArtifactDescriptor,
    DiagnosticArtifactKind,
    DiagnosticMatch,
    DiagnosticReadResult,
    DiagnosticSearchResult,
)
from deepfix.persistence import TaskRepository

_MAX_ARTIFACTS = 32
_MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
_MAX_MATCHES = 20
_MAX_RESULT_CHARACTERS = 12_000
_MAX_READ_LINES = 200


class DiagnosticArtifactService:
    def __init__(
        self,
        tasks: TaskRepository,
        collector: ArtifactReferenceCollector,
        backend: Any,
    ) -> None:
        self.tasks = tasks
        self.collector = collector
        self.backend = backend

    def search(
        self,
        task_id: str,
        messages: Sequence[AnyMessage],
        query: str,
        artifact_kinds: Sequence[DiagnosticArtifactKind | str] | None,
        max_matches: int,
    ) -> DiagnosticSearchResult:
        self._require_task(task_id)
        terms = _query_terms(query)
        limit = _match_limit(max_matches)
        catalog = self.collector.collect(
            task_id,
            messages,
            expand_history=True,
        )
        selected_by_kind = _select_kinds(catalog, artifact_kinds)
        if not selected_by_kind:
            raise DiagnosticArtifactToolError(
                "artifact_catalog_empty",
                "当前任务没有允许检索的诊断 Artifact",
            )
        selected = selected_by_kind[:_MAX_ARTIFACTS]
        omitted = len(selected_by_kind) - len(selected)

        matches: list[DiagnosticMatch] = []
        searched_count = 0
        remaining_characters = _MAX_RESULT_CHARACTERS
        truncated = omitted > 0
        for descriptor_index, descriptor in enumerate(selected):
            searched_count += 1
            text, content_hash = self._download_text(descriptor)
            lines = text.splitlines()
            ranges = _matching_ranges(lines, terms)
            for start, end in ranges:
                excerpt = _numbered_lines(lines, start, end)
                bounded_excerpt = excerpt[:remaining_characters]
                matches.append(
                    DiagnosticMatch(
                        artifact_id=descriptor.artifact_id,
                        kind=descriptor.kind,
                        start_line=start + 1,
                        end_line=end,
                        excerpt=bounded_excerpt,
                        content_hash=content_hash,
                    )
                )
                remaining_characters -= len(bounded_excerpt)
                if len(bounded_excerpt) < len(excerpt):
                    truncated = True
                    break
                if len(matches) == limit or remaining_characters == 0:
                    truncated = (
                        truncated
                        or next(ranges, None) is not None
                        or descriptor_index < len(selected) - 1
                    )
                    break
            if len(matches) == limit or remaining_characters == 0:
                break

        if not matches:
            raise DiagnosticArtifactToolError(
                "artifact_no_matches",
                "当前任务的诊断 Artifact 中没有匹配内容",
            )
        return DiagnosticSearchResult(
            query_terms=terms,
            matches=matches,
            searched_artifact_count=searched_count,
            omitted_artifact_count=omitted,
            truncated=truncated,
        )

    def read(
        self,
        task_id: str,
        messages: Sequence[AnyMessage],
        artifact_id: str,
        start_line: int,
        line_count: int,
    ) -> DiagnosticReadResult:
        self._require_task(task_id)
        _read_range(start_line, line_count)
        catalog = self.collector.collect(
            task_id,
            messages,
            expand_history=True,
        )
        descriptor = catalog.by_id(str(artifact_id or ""))
        if descriptor is None:
            raise DiagnosticArtifactToolError(
                "artifact_not_authorized",
                "Artifact 不属于当前任务或当前已授权目录",
            )
        text, content_hash = self._download_text(descriptor)
        lines = text.splitlines()
        if not lines:
            if start_line != 1:
                raise DiagnosticArtifactToolError(
                    "artifact_line_range_invalid",
                    "空 Artifact 只能从第 1 行读取",
                )
            return DiagnosticReadResult(
                artifact_id=descriptor.artifact_id,
                kind=descriptor.kind,
                start_line=1,
                end_line=0,
                total_lines=0,
                content="",
                content_hash=content_hash,
                truncated=False,
            )
        if start_line > len(lines):
            raise DiagnosticArtifactToolError(
                "artifact_line_range_invalid",
                "start_line 超出 Artifact 行数",
            )
        start_index = start_line - 1
        end_index = min(len(lines), start_index + line_count)
        content = _numbered_lines(lines, start_index, end_index)
        bounded_content = content[:_MAX_RESULT_CHARACTERS]
        return DiagnosticReadResult(
            artifact_id=descriptor.artifact_id,
            kind=descriptor.kind,
            start_line=start_line,
            end_line=end_index,
            total_lines=len(lines),
            content=bounded_content,
            content_hash=content_hash,
            truncated=(
                start_line > 1
                or end_index < len(lines)
                or len(content) > len(bounded_content)
            ),
        )

    def _require_task(self, task_id: str) -> None:
        try:
            task = self.tasks.get(task_id)
        except Exception as exc:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_reference_load_failed",
                "task_read",
            ) from exc
        if task.task_id != task_id:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_reference_load_failed",
                "task_read",
            )

    def _download_text(
        self,
        descriptor: DiagnosticArtifactDescriptor,
    ) -> tuple[str, str]:
        path = descriptor.backend_path
        try:
            responses = self.backend.download_files([path])
        except Exception as exc:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_backend_read_failed",
                "backend_read",
            ) from exc
        if not isinstance(responses, list) or len(responses) != 1:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_backend_read_failed",
                "backend_read",
            )
        response = responses[0]
        if not all(hasattr(response, name) for name in ("path", "content", "error")):
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_backend_read_failed",
                "backend_read",
            )
        if str(response.path) != path:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_backend_read_failed",
                "backend_read",
            )
        if response.error is not None:
            code = (
                "artifact_file_not_found"
                if str(response.error) == "file_not_found"
                else "artifact_file_unreadable"
            )
            raise DiagnosticArtifactToolError(
                code,
                "诊断 Artifact 文件不存在或不可读取",
            )
        if not isinstance(response.content, bytes):
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_backend_read_failed",
                "backend_read",
            )
        if len(response.content) > _MAX_ARTIFACT_BYTES:
            raise DiagnosticArtifactToolError(
                "artifact_too_large",
                "诊断 Artifact 超过 10 MiB 上限",
            )
        content_hash = hashlib.sha256(response.content).hexdigest()
        try:
            return response.content.decode("utf-8"), content_hash
        except UnicodeDecodeError as exc:
            raise DiagnosticArtifactToolError(
                "artifact_not_utf8",
                "诊断 Artifact 不是 UTF-8 文本",
            ) from exc


def _query_terms(query: str) -> list[str]:
    normalized = str(query).strip()
    if not 1 <= len(normalized) <= 200:
        raise DiagnosticArtifactToolError(
            "artifact_query_invalid",
            "查询长度必须为 1 到 200 字符",
        )
    terms = [item.casefold() for item in normalized.split()]
    if not 1 <= len(terms) <= 8:
        raise DiagnosticArtifactToolError(
            "artifact_query_invalid",
            "查询必须包含 1 到 8 个关键词",
        )
    return terms


def _match_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
        raise DiagnosticArtifactToolError(
            "artifact_match_limit_invalid",
            "max_matches 必须为 1 到 20 的整数",
        )
    return value


def _read_range(start_line: int, line_count: int) -> None:
    if (
        isinstance(start_line, bool)
        or not isinstance(start_line, int)
        or start_line < 1
        or isinstance(line_count, bool)
        or not isinstance(line_count, int)
        or not 1 <= line_count <= _MAX_READ_LINES
    ):
        raise DiagnosticArtifactToolError(
            "artifact_line_range_invalid",
            "start_line 必须大于等于 1，line_count 必须为 1 到 200",
        )


def _select_kinds(
    catalog: DiagnosticArtifactCatalog,
    artifact_kinds: Sequence[DiagnosticArtifactKind | str] | None,
) -> list[DiagnosticArtifactDescriptor]:
    if artifact_kinds is None:
        return list(catalog.artifacts)
    try:
        kinds = [DiagnosticArtifactKind(item) for item in artifact_kinds]
    except (TypeError, ValueError) as exc:
        raise DiagnosticArtifactToolError(
            "artifact_kind_invalid",
            "artifact_kinds 只能包含允许的诊断 Artifact 类型",
        ) from exc
    if not kinds or len(set(kinds)) != len(kinds):
        raise DiagnosticArtifactToolError(
            "artifact_kind_invalid",
            "artifact_kinds 不能为空或重复",
        )
    allowed = set(kinds)
    return [item for item in catalog.artifacts if item.kind in allowed]


def _matching_ranges(
    lines: Sequence[str],
    terms: Sequence[str],
) -> Iterator[tuple[int, int]]:
    if not lines:
        return
    folded = [line.casefold() for line in lines]
    if len(terms) == 1:
        term = terms[0]
        candidates = (
            (max(0, index - 2), min(len(lines), index + 3))
            for index, line in enumerate(folded)
            if term in line
        )
    else:
        candidates = (
            (start, min(len(lines), start + 5))
            for start in range(len(lines))
            if all(
                term in "\n".join(folded[start : min(len(lines), start + 5)])
                for term in terms
            )
        )
    yield from _merge_ranges(candidates)


def _merge_ranges(
    ranges: Iterator[tuple[int, int]],
) -> Iterator[tuple[int, int]]:
    pending: tuple[int, int] | None = None
    for start, end in ranges:
        if pending is not None and start < pending[1]:
            pending = (pending[0], max(pending[1], end))
        else:
            if pending is not None:
                yield pending
            pending = (start, end)
    if pending is not None:
        yield pending


def _numbered_lines(lines: Sequence[str], start: int, end: int) -> str:
    return "\n".join(
        f"{index + 1}: {lines[index]}" for index in range(start, end)
    )
