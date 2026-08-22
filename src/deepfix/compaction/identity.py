from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

_ORDINAL_KEY = "_deepfix_original_ordinal"


@dataclass(frozen=True)
class MessageIdentityResult:
    messages: list[AnyMessage]
    assigned_message_ids: tuple[str, ...]
    conflicted_message_ids: frozenset[str]


def ensure_message_ids(
    task_id: str,
    messages: Sequence[AnyMessage],
) -> MessageIdentityResult:
    normalized_task_id = _required(task_id, "task_id")
    normalized: list[AnyMessage] = []
    assigned: list[str] = []
    for current_index, message in enumerate(messages):
        kwargs = dict(message.additional_kwargs)
        ordinal = _original_ordinal(kwargs, current_index)
        kwargs[_ORDINAL_KEY] = ordinal
        message_id = str(message.id or "").strip()
        if not message_id:
            message_id = _stable_message_id(normalized_task_id, ordinal, message)
            assigned.append(message_id)
        normalized.append(
            message.model_copy(
                update={"id": message_id, "additional_kwargs": kwargs},
                deep=True,
            )
        )

    counts = Counter(str(message.id) for message in normalized)
    conflicts = frozenset(message_id for message_id, count in counts.items() if count > 1)
    return MessageIdentityResult(normalized, tuple(assigned), conflicts)


def stable_claim_id(task_id: str, text: str) -> str:
    return _prefixed_hash("claim", task_id, "claim:v1", _semantic_text(text))


def stable_hypothesis_id(task_id: str, first_source_id: str, text: str) -> str:
    return _prefixed_hash(
        "hyp",
        task_id,
        "hypothesis:v1",
        _required(first_source_id, "first_source_id"),
        _semantic_text(text),
    )


def stable_reopened_hypothesis_id(
    task_id: str,
    old_hypothesis_id: str,
    new_evidence_id: str,
    text: str,
) -> str:
    return _prefixed_hash(
        "hyp",
        task_id,
        "hypothesis-reopen:v1",
        _required(old_hypothesis_id, "old_hypothesis_id"),
        _required(new_evidence_id, "new_evidence_id"),
        _semantic_text(text),
    )


def stable_work_unit_id(task_id: str, message_ids: Sequence[str]) -> str:
    if not message_ids:
        raise ValueError("message_ids 不能为空")
    return _prefixed_hash("wu", task_id, "work-unit:v1", *message_ids)


def stable_generated_message_id(task_id: str, scope_id: str, result_type: str) -> str:
    return _prefixed_hash(
        "msg",
        task_id,
        "generated-message:v1",
        _required(scope_id, "scope_id"),
        _required(result_type, "result_type"),
    )


def _stable_message_id(task_id: str, ordinal: int, message: AnyMessage) -> str:
    content_hash = _sha256(_canonical_content(message.content))
    tool_ids = ",".join(sorted(_tool_call_ids(message)))
    return _prefixed_hash(
        "msg",
        task_id,
        str(ordinal),
        message.type,
        tool_ids,
        content_hash,
    )


def _tool_call_ids(message: AnyMessage) -> list[str]:
    if isinstance(message, AIMessage):
        return [str(call.get("id", "")) for call in message.tool_calls]
    if isinstance(message, ToolMessage):
        return [str(message.tool_call_id)]
    return []


def _canonical_content(content: Any) -> str:
    return json.dumps(
        _normalize_value(content),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    return value


def _semantic_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", _required(value, "text"))
    return re.sub(r"\s+", " ", normalized).strip()


def _original_ordinal(kwargs: dict[str, Any], fallback: int) -> int:
    value = kwargs.get(_ORDINAL_KEY, fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return fallback
    return value


def _prefixed_hash(prefix: str, *parts: str) -> str:
    material = "|".join(str(part) for part in parts)
    return f"{prefix}_{_sha256(material)[:32]}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} 不能为空")
    return normalized
