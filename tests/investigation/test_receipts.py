from langchain_core.messages import ToolMessage

from deepfix.investigation.receipts import ToolResultArtifact, ToolResultArtifactStorage


def test_command_result_artifact_is_bounded_durable_and_idempotent(tmp_path):
    root = tmp_path / "deepfix-artifacts" / "investigation_receipts"
    storage = ToolResultArtifactStorage(root)
    result = ToolMessage(
        content="x" * 200,
        name="execute",
        tool_call_id="execute-1",
        artifact={"exit_code": 7},
    )

    first = storage.save_result_artifact(
        "task-a", "execute-1", "execute", result, max_output_bytes=32
    )
    second = storage.save_result_artifact(
        "task-a", "execute-1", "execute", result, max_output_bytes=32
    )

    assert first == second
    path = root.parent / first
    artifact = ToolResultArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    assert artifact.exit_code == 7
    assert len(artifact.output.encode("utf-8")) <= 32
    assert not list(root.parent.rglob("*.tmp"))


def test_command_result_artifact_returns_verified_reference(tmp_path):
    root = tmp_path / "deepfix-artifacts" / "investigation_receipts"
    storage = ToolResultArtifactStorage(root)
    result = ToolMessage(
        content="passed",
        name="execute",
        tool_call_id="execute-1",
        artifact={"exit_code": 0},
    )

    reference = storage.save_result_artifact_reference(
        "task-a", "execute-1", "execute", result
    )

    assert reference.kind == "operation_result"
    assert len(reference.content_hash) == 64
    assert storage.verify_artifact_reference(reference)
