from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import Field, model_validator

from deepfix.compaction.models import StrictModel

_ARTIFACT_ID_PATTERN = r"^artifact_[0-9a-f]{32}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class DiagnosticArtifactKind(StrEnum):
    LARGE_TOOL_RESULT = "large_tool_result"
    CONVERSATION_HISTORY = "conversation_history"


class DiagnosticArtifactDescriptor(StrictModel):
    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    task_id: str = Field(min_length=1)
    kind: DiagnosticArtifactKind
    backend_path: str = Field(min_length=1)
    source_message_id: str | None = None
    tool_call_id: str | None = None
    snapshot_version: int | None = Field(default=None, ge=1)


class DiagnosticArtifactCatalog(StrictModel):
    artifacts: list[DiagnosticArtifactDescriptor] = Field(default_factory=list)

    def by_id(self, artifact_id: str) -> DiagnosticArtifactDescriptor | None:
        return next(
            (item for item in self.artifacts if item.artifact_id == artifact_id),
            None,
        )


class DiagnosticMatch(StrictModel):
    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    kind: DiagnosticArtifactKind
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    excerpt: str
    content_hash: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_line_range(self) -> DiagnosticMatch:
        if self.end_line < self.start_line:
            raise ValueError("end_line 不能小于 start_line")
        return self


class DiagnosticSearchResult(StrictModel):
    query_terms: list[str] = Field(min_length=1, max_length=8)
    matches: list[DiagnosticMatch] = Field(default_factory=list, max_length=20)
    searched_artifact_count: int = Field(ge=0, le=32)
    omitted_artifact_count: int = Field(ge=0)
    truncated: bool


class DiagnosticReadResult(StrictModel):
    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    kind: DiagnosticArtifactKind
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=0)
    total_lines: int = Field(ge=0)
    content: str
    content_hash: str = Field(pattern=_SHA256_PATTERN)
    truncated: bool

    @model_validator(mode="after")
    def validate_line_range(self) -> DiagnosticReadResult:
        if self.total_lines == 0:
            if self.end_line != 0:
                raise ValueError("空 Artifact 的 end_line 必须为 0")
            return self
        if self.end_line < self.start_line:
            raise ValueError("end_line 不能小于 start_line")
        if self.end_line > self.total_lines:
            raise ValueError("end_line 不能大于 total_lines")
        return self


def stable_diagnostic_artifact_id(
    task_id: str,
    kind: DiagnosticArtifactKind,
    backend_path: str,
) -> str:
    normalized_task_id = _required(task_id, "task_id")
    normalized_path = _required(backend_path, "backend_path")
    payload = (
        "diagnostic-artifact:v1|"
        f"{normalized_task_id}|{kind.value}|{normalized_path}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"artifact_{digest[:32]}"


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} 不能为空")
    return normalized
