from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from deepfix.compaction.errors import ProtectedContextLoadError
from deepfix.compaction.identity import ensure_message_ids
from deepfix.compaction.models import (
    ApprovalEvidence,
    CompactionSnapshot,
    ContextRecoveryMetadata,
    DeepFixCompactionEvent,
    DeterministicEvidenceBlock,
    FileChangeEvidence,
    HypothesisRecord,
    ProvenancedClaim,
    ProvenancedText,
    ProvenanceRef,
    ResearchStatusEvidence,
    SystemTestEvidence,
    TaskAnchor,
    UserConstraint,
)
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.evidence import (
    EvidenceKind,
    restore_deterministic_evidence,
    restore_external_evidence,
)
from deepfix.domain_repositories.execution import ExecutionIntegrity
from deepfix.investigation.models import InvestigationHypothesis
from deepfix.prompting import model_request_task_id
from deepfix.research.models import ExternalEvidence


@dataclass(frozen=True)
class ProtectedContext:
    """Request-local projection of current authorities plus compacted history."""

    task_anchor: TaskAnchor
    deterministic_evidence: DeterministicEvidenceBlock
    confirmed_facts: tuple[ProvenancedClaim, ...]
    hypotheses: tuple[HypothesisRecord, ...]
    unresolved_questions: tuple[ProvenancedText, ...]
    execution_integrity: ExecutionIntegrity
    external_evidence: tuple[ExternalEvidence, ...]
    active_snapshot: CompactionSnapshot | None


@dataclass(frozen=True)
class ProjectedContext:
    task_anchor_xml: str
    current_state_xml: str
    deterministic_evidence_xml: str
    execution_integrity_xml: str
    visible_snapshot_xml: str | None


class ProtectedContextBuilder:
    """Build protected context directly from bounded domain repositories."""

    def __init__(self, repositories: DomainRepositories) -> None:
        self.repositories = repositories

    def build(
        self,
        task_id: str,
        request_messages: Sequence[AnyMessage],
        event: DeepFixCompactionEvent | dict[str, Any] | None,
    ) -> ProtectedContext:
        active_snapshot: CompactionSnapshot | None = None
        try:
            definition = self.repositories.tasks.get_definition(task_id)
            lifecycle = self.repositories.tasks.get_lifecycle(task_id)
            normalized_event = (
                DeepFixCompactionEvent.model_validate(event)
                if isinstance(event, dict)
                else event
            )
            active_record = self.repositories.history.active_from_event(
                task_id, normalized_event
            )
            if active_record is not None:
                active_snapshot = self.repositories.history.project_snapshot(
                    task_id,
                    active_record.version,
                    evidence=self.repositories.evidence,
                    investigation=self.repositories.investigation,
                )
            envelopes = self.repositories.evidence.list_for_task(task_id)
            hypotheses = tuple(
                _hypothesis_record(item, active_record.version if active_record else 0)
                for item in self.repositories.investigation.list_hypotheses(task_id)
            )
            questions = tuple(
                ProvenancedText(
                    text=item.text,
                    sources=[
                        ProvenanceRef(kind="snapshot_record", ref_id=source_id)
                        for source_id in item.source_ids
                    ],
                )
                for item in self.repositories.investigation.list_questions(task_id)
                if item.status == "open"
            )
            integrity = self.repositories.execution.integrity_view(task_id)
            execution_approvals = self.repositories.execution.list_approvals(task_id)
        except Exception as exc:
            raise ProtectedContextLoadError(
                ContextRecoveryMetadata(
                    task_id=task_id,
                    stage="protected_context",
                    error_code="protected_context_read_failed",
                    active_snapshot_version=(
                        active_snapshot.version if active_snapshot else None
                    ),
                    original_messages_preserved=True,
                )
            ) from exc

        identities = ensure_message_ids(task_id, request_messages).messages
        user_messages = [item for item in identities if isinstance(item, HumanMessage)]
        latest_user_message_id = (
            str(user_messages[-1].id)
            if user_messages
            else definition.original_message_id
        )
        constraints = _current_constraints(
            task_id,
            definition.original_message_id,
            definition.original_problem,
            user_messages,
            active_snapshot,
        )
        deterministic = [
            restore_deterministic_evidence(item)
            for item in envelopes
            if item.kind
            in {
                EvidenceKind.TEST,
                EvidenceKind.FILE_CHANGE,
                EvidenceKind.APPROVAL,
                EvidenceKind.RESEARCH_STATUS,
            }
        ]
        approval_by_id = {
            item.evidence_id: item
            for item in deterministic
            if isinstance(item, ApprovalEvidence)
        }
        for item in execution_approvals:
            approval_by_id.setdefault(
                item.approval_id,
                ApprovalEvidence(
                    evidence_id=item.approval_id,
                    operation=item.operation,
                    decision=item.decision,
                    risk=item.risk,
                ),
            )
        block = DeterministicEvidenceBlock(
            tests=[item for item in deterministic if isinstance(item, SystemTestEvidence)],
            files=[item for item in deterministic if isinstance(item, FileChangeEvidence)],
            approvals=list(approval_by_id.values()),
            research=[item for item in deterministic if isinstance(item, ResearchStatusEvidence)],
        )
        current_facts = tuple(
            ProvenancedClaim.model_validate(item.payload)
            for item in envelopes
            if item.kind is EvidenceKind.SEMANTIC_CLAIM
        )
        external = tuple(
            restore_external_evidence(item)
            for item in envelopes
            if item.kind is EvidenceKind.EXTERNAL_RESEARCH
        )
        return ProtectedContext(
            task_anchor=TaskAnchor(
                task_id=task_id,
                task_goal=definition.original_problem,
                user_constraints=constraints,
                latest_user_message_id=latest_user_message_id,
                project_root=definition.workspace_root,
                project_python=definition.project_python,
                task_status=lifecycle.status.value,
            ),
            deterministic_evidence=block,
            confirmed_facts=current_facts,
            hypotheses=hypotheses,
            unresolved_questions=questions,
            execution_integrity=integrity,
            external_evidence=external,
            active_snapshot=active_snapshot,
        )


class ProtectedContextProjector:
    def project(
        self,
        context: ProtectedContext,
        snapshot: CompactionSnapshot | None = None,
    ) -> ProjectedContext:
        visible = snapshot if snapshot is not None else context.active_snapshot
        return ProjectedContext(
            task_anchor_xml=_render_anchor(context.task_anchor),
            current_state_xml=_render_current_state(context),
            deterministic_evidence_xml=_render_deterministic_evidence(
                context.deterministic_evidence
            ),
            execution_integrity_xml=_render_execution_integrity(
                context.execution_integrity
            ),
            visible_snapshot_xml=_render_visible_snapshot(visible, context),
        )


class ProtectedContextMiddleware(AgentMiddleware):
    """Compatibility middleware; production uses compaction as the sole injector."""

    def __init__(self, builder: ProtectedContextBuilder) -> None:
        self.builder = builder

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        task_id = model_request_task_id(request)
        if not task_id:
            return handler(request)
        context = self.builder.build(
            task_id, request.messages, request.state.get("_deepfix_compaction_event")
        )
        original = request.system_message.text if request.system_message else ""
        protected = render_protected_context(context)
        content = f"{original}\n\n{protected}" if original else protected
        return handler(request.override(system_message=SystemMessage(content=content)))


def render_protected_context(
    context: ProtectedContext,
    *,
    projector: ProtectedContextProjector | None = None,
) -> str:
    projected = (projector or ProtectedContextProjector()).project(context)
    blocks = [
        projected.task_anchor_xml,
        projected.current_state_xml,
        projected.deterministic_evidence_xml,
        projected.execution_integrity_xml,
    ]
    if projected.visible_snapshot_xml:
        blocks.append(projected.visible_snapshot_xml)
    return "<deepfix_protected_context>\n" + "\n\n".join(blocks) + "\n</deepfix_protected_context>"


def _current_constraints(
    task_id: str,
    original_message_id: str,
    original_problem: str,
    messages: Sequence[HumanMessage],
    snapshot: CompactionSnapshot | None,
) -> list[UserConstraint]:
    constraints = {
        item.constraint_id: item for item in (snapshot.user_constraints if snapshot else [])
    }
    all_messages = [(original_message_id, original_problem)]
    all_messages.extend(
        (str(item.id), str(item.content))
        for item in messages
        if str(item.id) != original_message_id
    )
    for message_id, text in all_messages:
        identity = hashlib.sha256(
            f"{task_id}\0{message_id}\0{' '.join(text.split())}".encode()
        ).hexdigest()[:32]
        constraint = UserConstraint(
            constraint_id=f"constraint_{identity}",
            text=text,
            source_user_message_id=message_id,
        )
        constraints[constraint.constraint_id] = constraint
    return list(constraints.values())


def _hypothesis_record(item: InvestigationHypothesis, version: int) -> HypothesisRecord:
    return HypothesisRecord(
        hypothesis_id=item.hypothesis_id,
        text=item.statement,
        state={"candidate": "active", "supported": "confirmed", "rejected": "rejected"}[item.state],
        reason=item.reason,
        reopens_hypothesis_id=item.reopens_hypothesis_id,
        sources=[
            ProvenanceRef(kind="system_evidence", ref_id=evidence_id)
            for evidence_id in item.evidence_ids
        ],
        updated_in_version=max(version, 1),
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
        f'<constraint constraint_id="{_xml(item.constraint_id, 120)}" state="{item.state}" source_user_message_id="{_xml(item.source_user_message_id, 120)}">{_xml(item.text, 800)}</constraint>'
        for item in anchor.user_constraints[:30]
    )
    return "\n".join([*lines, "</user_constraints>", "</deepfix_task_anchor>"])


def _render_current_state(context: ProtectedContext) -> str:
    lines = ["<deepfix_current_state>", "<confirmed_facts>"]
    lines.extend(
        f'<fact claim_id="{_xml(item.claim_id, 120)}">{_xml(item.text, 500)}</fact>'
        for item in context.confirmed_facts
    )
    lines.extend(("</confirmed_facts>", "<hypotheses>"))
    lines.extend(
        f'<hypothesis hypothesis_id="{_xml(item.hypothesis_id, 120)}" state="{item.state}"><text>{_xml(item.text, 500)}</text><reason>{_xml(item.reason or "", 500)}</reason></hypothesis>'
        for item in context.hypotheses
    )
    lines.extend(("</hypotheses>", "<unresolved_questions>"))
    lines.extend(f"<question>{_xml(item.text, 500)}</question>" for item in context.unresolved_questions)
    lines.extend(("</unresolved_questions>", "<external_research>"))
    lines.extend(
        f'<external evidence_id="{_xml(item.evidence_id, 120)}" verification="{_xml(str(item.local_verification), 80)}"><title>{_xml(item.title, 300)}</title><excerpt>{_xml(item.relevant_excerpt, 600)}</excerpt></external>'
        for item in context.external_evidence
    )
    return "\n".join([*lines, "</external_research>", "</deepfix_current_state>"])


def _render_deterministic_evidence(block: DeterministicEvidenceBlock) -> str:
    lines = ["<deepfix_deterministic_evidence>", "<tests>"]
    lines.extend(
        f'<test evidence_id="{_xml(item.evidence_id, 120)}" tool_call_id="{_xml(item.tool_call_id, 120)}" source_message_id="{_xml(item.source_message_id, 120)}"><command>{_xml(item.command, 500)}</command><exit_code>{item.exit_code}</exit_code><summary>{_xml(item.summary, 800)}</summary></test>'
        for item in block.tests[:30]
    )
    lines.extend(("</tests>", "<files>"))
    lines.extend(
        f'<file evidence_id="{_xml(item.evidence_id, 120)}" operation="{item.operation}" status="{item.status}">{_xml(item.path, 500)}</file>'
        for item in block.files[:50]
    )
    lines.extend(("</files>", "<approvals>"))
    lines.extend(
        f'<approval evidence_id="{_xml(item.evidence_id, 120)}" decision="{_xml(item.decision, 80)}" risk="{_xml(item.risk, 80)}">{_xml(item.operation, 200)}</approval>'
        for item in block.approvals[:50]
    )
    lines.extend(("</approvals>", "<research>"))
    lines.extend(
        f'<research_status evidence_id="{_xml(item.evidence_id, 120)}" verification="{item.verification}">{_xml(item.artifact_path or "", 500)}</research_status>'
        for item in block.research[:30]
    )
    return "\n".join([*lines, "</research>", "</deepfix_deterministic_evidence>"])


def _render_execution_integrity(value: ExecutionIntegrity) -> str:
    return (
        "<deepfix_execution_integrity>"
        f"<receipt_count>{value.receipt_count}</receipt_count>"
        f"<approval_count>{value.approval_count}</approval_count>"
        f"<incomplete_operation_ids>{_xml(','.join(value.incomplete_operation_ids), 1000)}</incomplete_operation_ids>"
        f"<unknown_operation_ids>{_xml(','.join(value.unknown_operation_ids), 1000)}</unknown_operation_ids>"
        "</deepfix_execution_integrity>"
    )


def _render_visible_snapshot(
    snapshot: CompactionSnapshot | None,
    context: ProtectedContext,
) -> str | None:
    if snapshot is None:
        return None
    constraint_ids = {item.constraint_id for item in context.task_anchor.user_constraints}
    claim_ids = {item.claim_id for item in context.confirmed_facts}
    hypothesis_ids = {item.hypothesis_id for item in context.hypotheses}
    evidence_ids = {
        item.evidence_id
        for item in [
            *context.deterministic_evidence.tests,
            *context.deterministic_evidence.files,
            *context.deterministic_evidence.approvals,
            *context.deterministic_evidence.research,
            *context.external_evidence,
        ]
    }
    lines = [f'<compaction_history_projection version="{snapshot.version}">']
    lines.extend(
        f'<historical_constraint constraint_id="{_xml(item.constraint_id, 120)}" />'
        for item in snapshot.user_constraints if item.constraint_id not in constraint_ids
    )
    lines.extend(
        f'<historical_fact claim_id="{_xml(item.claim_id, 120)}">{_xml(item.text, 500)}</historical_fact>'
        for item in snapshot.confirmed_facts if item.claim_id not in claim_ids
    )
    lines.extend(
        f'<historical_hypothesis hypothesis_id="{_xml(item.hypothesis_id, 120)}" state="{item.state}"><text>{_xml(item.text, 500)}</text><reason>{_xml(item.reason or "", 500)}</reason></historical_hypothesis>'
        for item in [*snapshot.active_hypotheses, *snapshot.rejected_hypotheses, *snapshot.confirmed_hypotheses]
        if item.hypothesis_id not in hypothesis_ids
    )
    lines.extend(
        f'<historical_evidence evidence_id="{_xml(item.evidence_id, 120)}" />'
        for item in [*snapshot.deterministic_evidence.tests, *snapshot.deterministic_evidence.files, *snapshot.deterministic_evidence.approvals, *snapshot.deterministic_evidence.research]
        if item.evidence_id not in evidence_ids
    )
    lines.extend(
        f'<artifact kind="{item.kind}" content_hash="{_xml(item.content_hash, 80)}">{_xml(item.path, 600)}</artifact>'
        for item in snapshot.artifact_references[:30]
    )
    return "\n".join([*lines, "</compaction_history_projection>"])


def _xml(value: str, limit: int) -> str:
    return escape(value[:limit])
