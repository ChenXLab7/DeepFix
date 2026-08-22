from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from deepfix.compaction.errors import ProtectedContextLoadError
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.identity import stable_conversation_message_id
from deepfix.compaction.models import (
    CompactionSnapshot,
    ContextRecoveryMetadata,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    ProvenanceRef,
    TaskAnchor,
)
from deepfix.compaction.store import CompactionStore
from deepfix.context import render_working_memory
from deepfix.memory import ProgressSnapshot, WorkingMemoryStore, WorkingMemoryVersion
from deepfix.persistence import TaskRepository
from deepfix.prompting import model_request_task_id
from deepfix.research.store import ResearchEvidenceStore


@dataclass(frozen=True)
class ProtectedContext:
    task_anchor: TaskAnchor
    working_memory: WorkingMemoryVersion | None
    deterministic_evidence: DeterministicEvidenceBlock
    active_snapshot: CompactionSnapshot | None


@dataclass(frozen=True)
class ProjectedContext:
    task_anchor_xml: str
    working_memory_xml: str
    deterministic_evidence_xml: str
    visible_snapshot_xml: str | None


class ProtectedContextBuilder:
    def __init__(
        self,
        tasks: TaskRepository,
        memory: WorkingMemoryStore,
        compaction: CompactionStore,
        research: ResearchEvidenceStore,
        collector: EvidenceCollector,
    ) -> None:
        self.tasks = tasks
        self.memory = memory
        self.compaction = compaction
        self.research = research
        self.collector = collector

    def build(
        self,
        task_id: str,
        request_messages: Sequence[AnyMessage],
        event: DeepFixCompactionEvent | dict[str, Any] | None,
    ) -> ProtectedContext:
        memory_version: WorkingMemoryVersion | None = None
        active_snapshot: CompactionSnapshot | None = None
        try:
            task = self.tasks.get(task_id)
            memory_version = self.memory.latest(task_id)
            normalized_event = (
                DeepFixCompactionEvent.model_validate(event)
                if isinstance(event, dict)
                else event
            )
            active_snapshot = self.compaction.active_snapshot_from_event(
                task_id,
                normalized_event,
            )
            evidence = self.collector.collect(task_id, request_messages, task)
        except Exception as exc:
            raise ProtectedContextLoadError(
                ContextRecoveryMetadata(
                    task_id=task_id,
                    stage="protected_context",
                    error_code="protected_context_read_failed",
                    working_memory_version=(
                        memory_version.version if memory_version else None
                    ),
                    active_snapshot_version=(
                        active_snapshot.version if active_snapshot else None
                    ),
                    original_messages_preserved=True,
                )
            ) from exc

        latest_user_message_id = _latest_user_message_id(task.conversation)
        if latest_user_message_id is None:
            latest_user_message_id = next(
                (
                    str(message.id)
                    for message in reversed(request_messages)
                    if isinstance(message, HumanMessage) and message.id
                ),
                stable_conversation_message_id(
                    task_id,
                    0,
                    "user",
                    task.user_problem,
                ),
            )
        anchor = TaskAnchor(
            task_id=task_id,
            task_goal=task.user_problem,
            user_constraints=(
                active_snapshot.user_constraints if active_snapshot else []
            ),
            latest_user_message_id=latest_user_message_id,
            project_root=task.project_root,
            project_python=task.project_python,
            task_status=task.status.value,
        )
        return ProtectedContext(anchor, memory_version, evidence, active_snapshot)


class ProtectedContextProjector:
    def project(
        self,
        context: ProtectedContext,
        snapshot: CompactionSnapshot | None = None,
    ) -> ProjectedContext:
        snapshot = snapshot if snapshot is not None else context.active_snapshot
        memory = _memory_with_snapshot_provenance(context.working_memory, snapshot)
        anchor_xml = _render_anchor(context.task_anchor)
        memory_xml = _render_memory(memory)
        evidence_xml = _render_deterministic_evidence(
            context.deterministic_evidence
        )
        visible_snapshot = _render_visible_snapshot(
            snapshot,
            context.task_anchor,
            memory,
            context.deterministic_evidence,
        )
        if visible_snapshot:
            memory_xml = memory_xml.replace(
                "</deepfix_working_memory>",
                f"{visible_snapshot}\n</deepfix_working_memory>",
            )
        return ProjectedContext(
            task_anchor_xml=anchor_xml,
            working_memory_xml=memory_xml,
            deterministic_evidence_xml=evidence_xml,
            visible_snapshot_xml=visible_snapshot,
        )


class ProtectedContextMiddleware(AgentMiddleware):
    def __init__(
        self,
        builder: ProtectedContextBuilder,
        projector: ProtectedContextProjector | None = None,
    ) -> None:
        self.builder = builder
        self.projector = projector or ProtectedContextProjector()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        task_id = model_request_task_id(request)
        if not task_id:
            return handler(request)
        raw_event = request.state.get("_deepfix_compaction_event")
        context = self.builder.build(task_id, request.messages, raw_event)
        protected = render_protected_context(context, projector=self.projector)
        original = request.system_message.text if request.system_message else ""
        content = f"{original}\n\n{protected}" if original else protected
        return handler(
            request.override(system_message=SystemMessage(content=content))
        )


def render_protected_context(
    context: ProtectedContext,
    *,
    projector: ProtectedContextProjector | None = None,
) -> str:
    projected = (projector or ProtectedContextProjector()).project(context)
    return (
        f"{projected.task_anchor_xml}\n\n"
        f"{projected.working_memory_xml}\n\n"
        f"{projected.deterministic_evidence_xml}"
    )


def _render_anchor(anchor: TaskAnchor) -> str:
    lines = [
        "<deepfix_task_anchor>",
        f"<task_id>{_xml(anchor.task_id, 100)}</task_id>",
        f"<task_goal>{_xml(anchor.task_goal, 2000)}</task_goal>",
        f"<latest_user_message_id>{_xml(anchor.latest_user_message_id, 120)}</latest_user_message_id>",
        f"<project_root>{_xml(anchor.project_root, 500)}</project_root>",
        f"<project_python>{_xml(anchor.project_python, 500)}</project_python>",
        f"<task_status>{_xml(anchor.task_status, 80)}</task_status>",
        "<user_constraints>",
    ]
    lines.extend(
        (
            f'<constraint constraint_id="{_xml(item.constraint_id, 120)}" '
            f'state="{item.state}" source_user_message_id="'
            f'{_xml(item.source_user_message_id, 120)}">'
            f"{_xml(item.text, 800)}</constraint>"
        )
        for item in anchor.user_constraints[:30]
    )
    lines.extend(("</user_constraints>", "</deepfix_task_anchor>"))
    return "\n".join(lines)


def _render_memory(memory: WorkingMemoryVersion | None) -> str:
    if memory is not None:
        return render_working_memory(memory)
    return (
        '<deepfix_working_memory version="none">\n'
        "<facts></facts>\n"
        "<active_hypotheses></active_hypotheses>\n"
        "<rejected_hypotheses></rejected_hypotheses>\n"
        "<confirmed_hypotheses></confirmed_hypotheses>\n"
        "<evidence></evidence>\n"
        "<checked_files></checked_files>\n"
        "<experiments></experiments>\n"
        "<next_steps></next_steps>\n"
        "<unresolved_questions></unresolved_questions>\n"
        "<coverage></coverage>\n"
        "</deepfix_working_memory>"
    )


def _render_deterministic_evidence(block: DeterministicEvidenceBlock) -> str:
    lines = ["<deepfix_deterministic_evidence>", "<tests>"]
    lines.extend(
        (
            f'<test evidence_id="{_xml(item.evidence_id, 120)}" '
            f'tool_call_id="{_xml(item.tool_call_id, 120)}" '
            f'source_message_id="{_xml(item.source_message_id, 120)}">'
            f"<command>{_xml(item.command, 500)}</command>"
            f"<exit_code>{item.exit_code}</exit_code>"
            f"<summary>{_xml(item.summary, 800)}</summary></test>"
        )
        for item in block.tests[:30]
    )
    lines.extend(("</tests>", "<files>"))
    lines.extend(
        (
            f'<file evidence_id="{_xml(item.evidence_id, 120)}" '
            f'operation="{item.operation}" status="{item.status}">'
            f"{_xml(item.path, 500)}</file>"
        )
        for item in block.files[:50]
    )
    lines.extend(("</files>", "<approvals>"))
    lines.extend(
        (
            f'<approval evidence_id="{_xml(item.evidence_id, 120)}" '
            f'decision="{_xml(item.decision, 80)}" risk="{_xml(item.risk, 80)}">'
            f"{_xml(item.operation, 200)}</approval>"
        )
        for item in block.approvals[:50]
    )
    lines.extend(("</approvals>", "<research>"))
    lines.extend(
        (
            f'<research_status evidence_id="{_xml(item.evidence_id, 120)}" '
            f'verification="{item.verification}">'
            f"{_xml(item.artifact_path or '', 500)}</research_status>"
        )
        for item in block.research[:30]
    )
    lines.extend(("</research>", "</deepfix_deterministic_evidence>"))
    return "\n".join(lines)


def _render_visible_snapshot(
    snapshot: CompactionSnapshot | None,
    anchor: TaskAnchor,
    memory: WorkingMemoryVersion | None,
    evidence: DeterministicEvidenceBlock,
) -> str | None:
    if snapshot is None:
        return None
    claim_ids = {
        item.claim_id for item in memory.snapshot.facts
    } if memory else set()
    hypothesis_ids = {
        item.hypothesis_id for item in memory.snapshot.all_hypotheses()
    } if memory else set()
    constraint_ids = {item.constraint_id for item in anchor.user_constraints}
    evidence_ids = {
        item.evidence_id
        for item in [
            *evidence.tests,
            *evidence.files,
            *evidence.approvals,
            *evidence.research,
        ]
    }
    lines = [f'<compaction_history_projection version="{snapshot.version}">']
    lines.extend(
        f'<historical_constraint constraint_id="{_xml(item.constraint_id, 120)}" />'
        for item in snapshot.user_constraints
        if item.constraint_id not in constraint_ids
    )
    lines.extend(
        (
            f'<historical_fact claim_id="{_xml(item.claim_id, 120)}">'
            f"{_xml(item.text, 500)}</historical_fact>"
        )
        for item in snapshot.confirmed_facts
        if item.claim_id not in claim_ids
    )
    lines.extend(
        (
            f'<historical_hypothesis hypothesis_id="{_xml(item.hypothesis_id, 120)}" '
            f'state="{item.state}">{_xml(item.text, 500)}</historical_hypothesis>'
        )
        for item in [
            *snapshot.active_hypotheses,
            *snapshot.rejected_hypotheses,
            *snapshot.confirmed_hypotheses,
        ]
        if item.hypothesis_id not in hypothesis_ids
    )
    lines.extend(
        f'<historical_evidence evidence_id="{_xml(item.evidence_id, 120)}" />'
        for item in [
            *snapshot.deterministic_evidence.tests,
            *snapshot.deterministic_evidence.files,
            *snapshot.deterministic_evidence.approvals,
            *snapshot.deterministic_evidence.research,
        ]
        if item.evidence_id not in evidence_ids
    )
    lines.extend(
        (
            f'<artifact kind="{item.kind}" content_hash="{_xml(item.content_hash, 80)}">'
            f"{_xml(item.path, 600)}</artifact>"
        )
        for item in snapshot.artifact_references[:30]
    )
    lines.append("</compaction_history_projection>")
    return "\n".join(lines)


def _memory_with_snapshot_provenance(
    memory: WorkingMemoryVersion | None,
    snapshot: CompactionSnapshot | None,
) -> WorkingMemoryVersion | None:
    if memory is None or snapshot is None:
        return memory
    snapshot_claims = {item.claim_id: item for item in snapshot.confirmed_facts}
    snapshot_hypotheses = {
        item.hypothesis_id: item
        for item in [
            *snapshot.active_hypotheses,
            *snapshot.rejected_hypotheses,
            *snapshot.confirmed_hypotheses,
        ]
    }
    facts = [
        item.model_copy(
            update={
                "sources": _merge_sources(
                    item.sources,
                    snapshot_claims.get(item.claim_id, item).sources,
                )
            }
        )
        for item in memory.snapshot.facts
    ]
    hypotheses = [
        item.model_copy(
            update={
                "sources": _merge_sources(
                    item.sources,
                    snapshot_hypotheses.get(item.hypothesis_id, item).sources,
                )
            }
        )
        for item in memory.snapshot.all_hypotheses()
    ]
    merged_snapshot: ProgressSnapshot = memory.snapshot.model_copy(
        update={
            "facts": facts,
            "active_hypotheses": [item for item in hypotheses if item.state == "active"],
            "rejected_hypotheses": [item for item in hypotheses if item.state == "rejected"],
            "confirmed_hypotheses": [item for item in hypotheses if item.state == "confirmed"],
        }
    )
    return WorkingMemoryVersion(
        task_id=memory.task_id,
        version=memory.version,
        snapshot=merged_snapshot,
        created_at=memory.created_at,
    )


def _merge_sources(
    first: Sequence[ProvenanceRef],
    second: Sequence[ProvenanceRef],
) -> list[ProvenanceRef]:
    return list(
        {(item.kind, item.ref_id): item for item in [*first, *second]}.values()
    )


def _latest_user_message_id(conversation: list[dict[str, str]]) -> str | None:
    return next(
        (
            str(entry["id"])
            for entry in reversed(conversation)
            if entry.get("role") == "user" and entry.get("id")
        ),
        None,
    )


def _xml(value: str, limit: int) -> str:
    return escape(value[:limit])
