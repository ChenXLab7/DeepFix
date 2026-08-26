from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from pydantic import Field, JsonValue

from deepfix.compaction.models import StrictModel
from deepfix.investigation.classification import result_fingerprint
from deepfix.investigation.identity import stable_investigation_id


class ToolExecutionReceipt(StrictModel):
    task_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    call_hash: str = Field(min_length=1)
    tool_message_data: dict[str, JsonValue]
    result_fingerprint: str = Field(min_length=1)

    @property
    def tool_message(self) -> ToolMessage:
        restored = messages_from_dict([self.tool_message_data])[0]
        if not isinstance(restored, ToolMessage):
            raise TypeError("tool execution receipt 不是 ToolMessage")
        return restored


class ToolExecutionReceiptStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self._locks: dict[Path, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def save(self, receipt: ToolExecutionReceipt) -> None:
        path = self._path(receipt.task_id, receipt.tool_call_id)
        payload = receipt.model_dump_json(indent=2)
        with self._lock_for(path):
            existing = self._load_path(path)
            if existing is not None:
                if existing == receipt:
                    return
                raise RuntimeError("tool execution receipt 回执冲突")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open(
                    "x",
                    encoding="utf-8",
                    newline="",
                ) as file:
                    file.write(payload)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            content = path.read_text(encoding="utf-8")
            if content != payload or self._load_path(path) != receipt:
                raise RuntimeError("tool execution receipt 校验失败")

    def load(
        self,
        task_id: str,
        tool_call_id: str,
    ) -> ToolExecutionReceipt | None:
        path = self._path(task_id, tool_call_id)
        return self._load_path(path)

    def _path(self, task_id: str, tool_call_id: str) -> Path:
        task_segment = receipt_task_segment(task_id)
        call_segment = hashlib.sha256(tool_call_id.strip().encode()).hexdigest()[:32]
        if not task_id.strip() or not tool_call_id.strip():
            raise ValueError("receipt task_id 和 tool_call_id 不能为空")
        return self.root_dir / task_segment / f"{call_segment}.json"

    def _load_path(self, path: Path) -> ToolExecutionReceipt | None:
        if not path.exists():
            return None
        if not path.is_file():
            raise RuntimeError("tool execution receipt 缺少内容")
        return ToolExecutionReceipt.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def _lock_for(self, path: Path) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(path, threading.Lock())


def receipt_task_segment(task_id: str) -> str:
    normalized = task_id.strip()
    if not normalized:
        raise ValueError("receipt task_id 不能为空")
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def receipt_from_result(
    task_id: str,
    tool_call: Mapping[str, Any],
    result: ToolMessage,
) -> ToolExecutionReceipt:
    call_id = str(tool_call.get("id", "")).strip()
    tool_name = str(tool_call.get("name", "")).strip()
    return ToolExecutionReceipt(
        task_id=task_id,
        tool_call_id=call_id,
        tool_name=tool_name,
        call_hash=tool_call_hash(task_id, tool_call),
        tool_message_data=message_to_dict(result),
        result_fingerprint=result_fingerprint(result),
    )


def tool_call_hash(task_id: str, tool_call: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        tool_call,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return stable_investigation_id("call", task_id, canonical)
