import pytest
from pydantic import ValidationError

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
    stable_diagnostic_artifact_id,
)


def descriptor(**updates) -> DiagnosticArtifactDescriptor:
    values = {
        "artifact_id": "artifact_" + "a" * 32,
        "task_id": "task-a",
        "kind": DiagnosticArtifactKind.LARGE_TOOL_RESULT,
        "backend_path": "/.deepfix-artifacts/large_tool_results/call_1",
    }
    values.update(updates)
    return DiagnosticArtifactDescriptor(**values)


def test_artifact_id_is_stable_and_task_scoped():
    path = "/.deepfix-artifacts/large_tool_results/call_1"

    first = stable_diagnostic_artifact_id(
        "task-a",
        DiagnosticArtifactKind.LARGE_TOOL_RESULT,
        path,
    )

    assert first == stable_diagnostic_artifact_id(
        "task-a",
        DiagnosticArtifactKind.LARGE_TOOL_RESULT,
        path,
    )
    assert first != stable_diagnostic_artifact_id(
        "task-b",
        DiagnosticArtifactKind.LARGE_TOOL_RESULT,
        path,
    )
    assert first.startswith("artifact_") and len(first) == 41


def test_artifact_id_rejects_blank_authority_fields():
    with pytest.raises(ValueError, match="task_id"):
        stable_diagnostic_artifact_id(
            " ",
            DiagnosticArtifactKind.LARGE_TOOL_RESULT,
            "/.deepfix-artifacts/large_tool_results/call_1",
        )
    with pytest.raises(ValueError, match="backend_path"):
        stable_diagnostic_artifact_id(
            "task-a",
            DiagnosticArtifactKind.LARGE_TOOL_RESULT,
            " ",
        )


def test_descriptor_rejects_unknown_fields_and_invalid_id():
    with pytest.raises(ValidationError):
        descriptor(unexpected=True)
    with pytest.raises(ValidationError):
        descriptor(artifact_id="artifact-not-a-hash")


def test_catalog_resolves_only_exact_stable_id():
    item = descriptor()
    catalog = DiagnosticArtifactCatalog(artifacts=[item])

    assert catalog.by_id(item.artifact_id) == item
    assert catalog.by_id("artifact_" + "b" * 32) is None


def test_search_and_read_models_enforce_line_and_hash_bounds():
    match = DiagnosticMatch(
        artifact_id="artifact_" + "a" * 32,
        kind="large_tool_result",
        start_line=1,
        end_line=3,
        excerpt="1: failure",
        content_hash="b" * 64,
    )
    search = DiagnosticSearchResult(
        query_terms=["failure"],
        matches=[match],
        searched_artifact_count=1,
        omitted_artifact_count=0,
        truncated=False,
    )
    read = DiagnosticReadResult(
        artifact_id=match.artifact_id,
        kind=match.kind,
        start_line=1,
        end_line=3,
        total_lines=3,
        content="1: failure",
        content_hash=match.content_hash,
        truncated=False,
    )

    assert search.matches == [match]
    assert read.total_lines == 3
    with pytest.raises(ValidationError):
        match.model_copy(update={"start_line": 0}, deep=True).model_validate(
            {**match.model_dump(), "start_line": 0}
        )
    with pytest.raises(ValidationError):
        DiagnosticReadResult(
            **{**read.model_dump(), "content_hash": "not-sha256"}
        )


def test_result_models_reject_reversed_line_ranges():
    with pytest.raises(ValidationError, match="end_line"):
        DiagnosticMatch(
            artifact_id="artifact_" + "a" * 32,
            kind="large_tool_result",
            start_line=4,
            end_line=3,
            excerpt="failure",
            content_hash="b" * 64,
        )


def test_retrieval_errors_keep_only_stable_safe_metadata():
    tool_error = DiagnosticArtifactToolError(
        "artifact_query_invalid",
        "x" * 400,
    )
    system_error = DiagnosticArtifactSystemError(
        "diagnostic_artifact_backend_read_failed",
        "backend_read",
    )

    assert str(tool_error) == "artifact_query_invalid"
    assert len(tool_error.safe_message) == 300
    assert str(system_error) == "diagnostic_artifact_backend_read_failed"
    assert system_error.stage == "backend_read"
    assert not hasattr(system_error, "recovery")
