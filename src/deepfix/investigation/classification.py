from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage

from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import RelationEdge

_PYTEST_ENTRYPOINTS = {"pytest", "pytest.exe"}
_PYTHON_ENTRYPOINTS = {"python", "python.exe", "py", "py.exe"}
_VERIFIED_RELATIONS = {
    "import",
    "call",
    "traceback",
    "grep_reference",
    "symbol_reference",
    "test_collection",
}


def is_pytest_verification(command: str, project_python: str) -> bool:
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        return False
    if not tokens:
        return False
    if any(_is_shell_control_token(token) for token in tokens):
        return False

    raw_executable = tokens[0].strip('"')
    executable = Path(raw_executable).name.lower()
    if executable in _PYTEST_ENTRYPOINTS:
        return True

    configured = project_python.strip().strip('"')
    configured_name = Path(configured).name.lower() if configured else ""
    is_python = executable in _PYTHON_ENTRYPOINTS or (
        configured_name and executable == configured_name
    )
    return is_python and len(tokens) >= 3 and [item.lower() for item in tokens[1:3]] == [
        "-m",
        "pytest",
    ]


def _is_shell_control_token(token: str) -> bool:
    value = token.strip()
    return (
        value in {"|", "||", "&&", ";", "<", ">"}
        or value.startswith((">", "<"))
        or ">&" in value
        or "<&" in value
    )


def tool_signature(tool_name: str, arguments: Mapping[str, object]) -> str:
    normalized_name = tool_name.strip().lower()
    if not normalized_name:
        raise ValueError("tool_name 不能为空")
    digest = hashlib.sha256(_canonical_json(arguments).encode()).hexdigest()[:32]
    return f"tool_{normalized_name}_{digest}"


def result_fingerprint(message: ToolMessage) -> str:
    artifact = message.artifact if isinstance(message.artifact, Mapping) else {}
    bounded = {
        "status": message.status,
        "exit_code": artifact.get("exit_code"),
        "operation": artifact.get("operation"),
        "path": artifact.get("path"),
        "content_hash": hashlib.sha256(_message_text(message).encode()).hexdigest(),
    }
    digest = hashlib.sha256(_canonical_json(bounded).encode()).hexdigest()[:32]
    return f"result_{digest}"


def relation_from_model_reason(target: str, reason: str) -> None:
    del target, reason


def relation_from_tool_result(
    *,
    task_id: str,
    relation: str,
    source: str,
    target: str,
    source_message_id: str,
) -> RelationEdge:
    normalized_relation = relation.strip().lower()
    if normalized_relation not in _VERIFIED_RELATIONS:
        raise ValueError(f"不受支持的可验证关系: {relation}")
    values = {
        "task_id": task_id.strip(),
        "source": source.strip(),
        "target": target.strip(),
        "source_message_id": source_message_id.strip(),
    }
    if any(not value for value in values.values()):
        raise ValueError("关系边的任务、来源、目标和消息 ID 不能为空")
    return RelationEdge(
        relation_id=stable_investigation_id(
            "relation",
            values["task_id"],
            normalized_relation,
            values["source"],
            values["target"],
            values["source_message_id"],
        ),
        relation=normalized_relation,
        source=values["source"],
        target=values["target"],
        source_message_id=values["source_message_id"],
    )


def _message_text(message: ToolMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return _canonical_json(message.content)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
