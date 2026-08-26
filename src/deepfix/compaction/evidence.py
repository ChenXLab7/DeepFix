from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from deepfix.compaction.identity import ensure_message_ids
from deepfix.compaction.models import (
    ApprovalEvidence,
    DeterministicEvidenceBlock,
    FileChangeEvidence,
    ResearchStatusEvidence,
    SystemTestEvidence,
)
from deepfix.compaction.store import CompactionStore, DeterministicEvidence
from deepfix.models import TaskState
from deepfix.research.store import ResearchEvidenceStore

_FILE_TO_OPERATION = {
    "write_file": "write",
    "edit_file": "edit",
    "delete": "delete",
}


class EvidenceCollector:
    def __init__(
        self,
        store: CompactionStore,
        research_store: ResearchEvidenceStore,
    ) -> None:
        self.store = store
        self.research_store = research_store

    def collect(
        self,
        task_id: str,
        messages: Sequence[AnyMessage],
        task_state: TaskState,
    ) -> DeterministicEvidenceBlock:
        task_id = task_id.strip()
        if not task_id or task_state.task_id != task_id:
            raise ValueError("EvidenceCollector task_id 与 TaskState 不匹配")
        identified = ensure_message_ids(task_id, messages).messages
        calls = _unique_pairs(identified)

        collected: list[DeterministicEvidence] = []
        actual_paths: set[str] = set()
        for call_id, (call, result) in calls.items():
            name = str(call.get("name", ""))
            args = call.get("args", {})
            args = args if isinstance(args, Mapping) else {}
            if name == "execute":
                test = _test_evidence(task_id, call_id, args, result)
                if test is not None:
                    collected.append(test)
                continue
            file_change = _file_evidence(task_id, call_id, name, args, result)
            if file_change is not None:
                collected.append(file_change)
                actual_paths.add(file_change.path)

        for path in dict.fromkeys(task_state.changed_files):
            if path in actual_paths:
                continue
            collected.append(
                FileChangeEvidence(
                    evidence_id=_evidence_id(
                        task_id, "approved_target", path
                    ),
                    path=path,
                    operation="approved_target",
                    status="approved_target",
                )
            )

        for index, approval in enumerate(task_state.approvals):
            collected.append(
                ApprovalEvidence(
                    evidence_id=_evidence_id(
                        task_id,
                        "approval",
                        str(index),
                        approval.operation,
                        approval.decision,
                        approval.risk,
                    ),
                    operation=approval.operation,
                    decision=approval.decision,
                    risk=approval.risk,
                )
            )

        for item in self.research_store.list_evidence(task_id):
            collected.append(
                ResearchStatusEvidence(
                    evidence_id=_evidence_id(
                        task_id,
                        "research",
                        item.evidence_id,
                        item.local_verification,
                        *item.linked_test_tool_call_ids,
                    ),
                    verification=item.local_verification,
                    artifact_path=item.artifact_path,
                )
            )

        for evidence in collected:
            self.store.save_evidence(task_id, evidence)
        return _as_block(self.store.list_evidence(task_id))

    def collect_pair(
        self,
        task_id: str,
        call: Mapping[str, object],
        result: ToolMessage,
        task_state: TaskState,
    ) -> DeterministicEvidence | None:
        block = self.collect(
            task_id,
            [AIMessage(content="", tool_calls=[dict(call)]), result],
            task_state,
        )
        call_id = str(call.get("id", ""))
        return next(
            (
                item
                for item in [*block.tests, *block.files]
                if item.tool_call_id == call_id
            ),
            None,
        )


def _unique_pairs(
    messages: Sequence[AnyMessage],
) -> dict[str, tuple[dict[str, Any], ToolMessage]]:
    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    results: dict[str, list[ToolMessage]] = defaultdict(list)
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls[str(call.get("id", ""))].append(call)
        elif isinstance(message, ToolMessage):
            results[str(message.tool_call_id)].append(message)
    unique = {
        call_id: (items[0], results[call_id][0])
        for call_id, items in calls.items()
        if call_id and len(items) == 1 and len(results[call_id]) == 1
    }
    return unique


def _test_evidence(
    task_id: str,
    call_id: str,
    args: Mapping[str, object],
    result: ToolMessage,
) -> SystemTestEvidence | None:
    command = str(args.get("command", "")).strip()
    artifact = result.artifact
    if "pytest" not in command.lower() or not isinstance(artifact, Mapping):
        return None
    exit_code = artifact.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return None
    return SystemTestEvidence(
        evidence_id=_evidence_id(
            task_id,
            "test",
            call_id,
            str(result.id),
            command,
            str(exit_code),
        ),
        command=command,
        exit_code=exit_code,
        summary=_message_text(result),
        tool_call_id=call_id,
        source_message_id=str(result.id),
    )


def _file_evidence(
    task_id: str,
    call_id: str,
    tool_name: str,
    args: Mapping[str, object],
    result: ToolMessage,
) -> FileChangeEvidence | None:
    expected_operation = _FILE_TO_OPERATION.get(tool_name)
    artifact = result.artifact
    if expected_operation is None:
        return None
    if isinstance(artifact, Mapping):
        operation = artifact.get("operation")
        status = artifact.get("status")
        path = artifact.get("path")
    elif _is_deepagents_edit_success(tool_name, result):
        operation = "edit"
        status = "succeeded"
        path = args.get("file_path")
    else:
        return None
    if (
        operation != expected_operation
        or status not in {"succeeded", "failed"}
        or not isinstance(path, str)
        or not path.strip()
        or (status == "succeeded" and result.status == "error")
    ):
        return None
    return FileChangeEvidence(
        evidence_id=_evidence_id(
            task_id,
            "file",
            call_id,
            str(result.id),
            operation,
            status,
            path,
        ),
        path=path,
        operation=operation,
        status=status,
        tool_call_id=call_id,
        source_message_id=str(result.id),
    )


def _is_deepagents_edit_success(tool_name: str, result: ToolMessage) -> bool:
    return (
        tool_name == "edit_file"
        and result.status != "error"
        and _message_text(result).startswith("Successfully replaced ")
    )


def _as_block(
    evidence: Sequence[DeterministicEvidence],
) -> DeterministicEvidenceBlock:
    return DeterministicEvidenceBlock(
        tests=[item for item in evidence if isinstance(item, SystemTestEvidence)],
        files=[item for item in evidence if isinstance(item, FileChangeEvidence)],
        approvals=[item for item in evidence if isinstance(item, ApprovalEvidence)],
        research=[item for item in evidence if isinstance(item, ResearchStatusEvidence)],
    )


def _message_text(message: ToolMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return json.dumps(message.content, ensure_ascii=False, sort_keys=True)


def _evidence_id(task_id: str, *parts: str) -> str:
    material = "|".join((task_id, "evidence:v1", *parts))
    return f"evidence_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"
