from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from deepagents.backends import BackendProtocol
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
    def __init__(self, backend: BackendProtocol) -> None:
        self.backend = backend

    def save(self, receipt: ToolExecutionReceipt) -> None:
        path = self._path(receipt.task_id, receipt.tool_call_id)
        payload = receipt.model_dump_json(indent=2)
        written = self.backend.write(path, payload)
        if written.error:
            raise RuntimeError("tool execution receipt 写入失败")
        read = self.backend.read(path)
        content = None if read.file_data is None else read.file_data["content"]
        if read.error or content != payload:
            raise RuntimeError("tool execution receipt 校验失败")

    def load(
        self,
        task_id: str,
        tool_call_id: str,
    ) -> ToolExecutionReceipt | None:
        path = self._path(task_id, tool_call_id)
        result = self.backend.read(path)
        if result.error:
            return None
        if result.file_data is None:
            raise RuntimeError("tool execution receipt 缺少内容")
        return ToolExecutionReceipt.model_validate_json(
            result.file_data["content"]
        )

    @staticmethod
    def _path(task_id: str, tool_call_id: str) -> str:
        task_segment = hashlib.sha256(task_id.strip().encode()).hexdigest()[:32]
        call_segment = hashlib.sha256(tool_call_id.strip().encode()).hexdigest()[:32]
        if not task_id.strip() or not tool_call_id.strip():
            raise ValueError("receipt task_id 和 tool_call_id 不能为空")
        return f"/investigation_receipts/{task_segment}/{call_segment}.json"


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
