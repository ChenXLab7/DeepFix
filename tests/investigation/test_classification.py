import pytest
from langchain_core.messages import ToolMessage

from deepfix.investigation.classification import (
    is_pytest_verification,
    relation_from_model_reason,
    relation_from_tool_result,
    result_fingerprint,
    tool_signature,
)
from deepfix.investigation.models import ScopeKind


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "python -m pytest tests/test_sign.py",
        "python.exe -m pytest",
        '"C:/Python/python.exe" -m pytest -q',
    ],
)
def test_pytest_classifier_accepts_real_pytest_commands(command):
    assert is_pytest_verification(command, "C:/Python/python.exe")


@pytest.mark.parametrize(
    "command",
    [
        "echo pytest",
        "python --version",
        "pip install pytest",
        "ruff check .",
        "python -c \"print('pytest')\"",
        "python -m pytest -q 2>&1 | head -50",
        "python -m pytest -q > pytest.log",
        "",
    ],
)
def test_pytest_classifier_rejects_non_verification_execute(command):
    assert not is_pytest_verification(command, "C:/Python/python.exe")


def test_tool_signature_is_order_independent_and_does_not_store_arguments():
    first = tool_signature("read_file", {"path": "src/sign.py", "start": 1})
    second = tool_signature("read_file", {"start": 1, "path": "src/sign.py"})

    assert first == second
    assert "src/sign.py" not in first


def test_result_fingerprint_ignores_message_and_tool_call_identity():
    first = ToolMessage(
        id="message-1",
        content="1 failed",
        tool_call_id="call-1",
        artifact={"exit_code": 1},
    )
    second = ToolMessage(
        id="message-2",
        content="1 failed",
        tool_call_id="call-2",
        artifact={"exit_code": 1},
    )

    assert result_fingerprint(first) == result_fingerprint(second)


def test_result_fingerprint_changes_when_deterministic_result_changes():
    failed = ToolMessage(
        content="1 failed",
        tool_call_id="call-1",
        artifact={"exit_code": 1},
    )
    passed = ToolMessage(
        content="1 passed",
        tool_call_id="call-2",
        artifact={"exit_code": 0},
    )

    assert result_fingerprint(failed) != result_fingerprint(passed)


def test_model_reason_cannot_create_dependency_relation():
    assert relation_from_model_reason("src/random.py", "because it may matter") is None


def test_relation_requires_verifiable_source_message():
    edge = relation_from_tool_result(
        task_id="task-a",
        relation="import",
        source="src/api.py",
        target="src/sign.py",
        source_message_id="msg-1",
    )

    assert edge.scope is ScopeKind.DEPENDENCY
    assert edge.source_message_id == "msg-1"


def test_relation_rejects_unverified_relation_type():
    with pytest.raises(ValueError, match="不受支持"):
        relation_from_tool_result(
            task_id="task-a",
            relation="model_reason",
            source="src/api.py",
            target="src/sign.py",
            source_message_id="msg-1",
        )
