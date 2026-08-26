from __future__ import annotations

from collections.abc import Callable
from html import escape
from typing import Literal

from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationToolMiddleware,
)
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_config
from pydantic import ValidationError

from deepfix.compaction.identity import ensure_message_ids, stable_generated_message_id
from deepfix.compaction.models import (
    FactCandidate,
    HypothesisProgressInput,
    HypothesisRecord,
    ProvenancedClaim,
    ProvenanceRef,
    SnapshotCoverage,
)
from deepfix.compaction.store import CompactionStore
from deepfix.compaction.work_units import partition_work_units
from deepfix.investigation.store import InvestigationStore
from deepfix.memory import WorkingMemoryStore, WorkingMemoryVersion
from deepfix.models import Evidence
from deepfix.prompting import PromptPolicyMiddleware
from deepfix.research.middleware import ResearchEvidenceMiddleware
from deepfix.research.store import ResearchEvidenceStore

_TRUNCATION_MARKER = "…[truncated]"


def build_save_progress_tool(
    store: WorkingMemoryStore,
    *,
    compaction_store: CompactionStore | None = None,
    investigation_store: InvestigationStore | None = None,
) -> BaseTool:
    evidence_store = compaction_store or CompactionStore(store.database_path)
    investigation = investigation_store or InvestigationStore(store.database_path)

    def save_progress(
        phase: Literal[
            "clarifying",
            "investigating",
            "planning",
            "editing",
            "testing",
            "reviewing",
        ],
        summary: str,
        facts: list[FactCandidate],
        evidence: list[Evidence],
        hypotheses: list[HypothesisProgressInput],
        checked_files: list[str],
        experiments: list[str],
        next_steps: list[str],
        unresolved_questions: list[str],
        runtime: ToolRuntime,
    ) -> ToolMessage:
        thread_id = str(
            runtime.config.get("configurable", {}).get("thread_id", "")
        ).strip()
        if not thread_id:
            return ToolMessage(
                content="保存工作记忆失败：运行配置缺少 thread_id",
                name="save_progress",
                tool_call_id=runtime.tool_call_id or "",
                status="error",
                id=stable_generated_message_id(
                    "unknown", runtime.tool_call_id or "save-progress", "missing_task"
                ),
            )

        try:
            state_messages = list(runtime.state.get("messages", []))
            identities = ensure_message_ids(thread_id, state_messages)
            work_unit_ids = {
                unit.unit_id
                for unit in partition_work_units(identities.messages, set()).units
            }
            user_message_ids = {
                str(message.id)
                for message in identities.messages
                if isinstance(message, HumanMessage) and message.id
            }
            deterministic_evidence_ids = {
                item.evidence_id for item in evidence_store.list_evidence(thread_id)
            }
            latest_memory = store.latest(thread_id)
            memory_ids = set()
            if latest_memory is not None:
                memory_ids.update(
                    item.claim_id for item in latest_memory.snapshot.facts
                )
                memory_ids.update(
                    item.hypothesis_id
                    for item in latest_memory.snapshot.all_hypotheses()
                )
            investigation_state = investigation.load(thread_id)
            seed_hypotheses = []
            if investigation_state is not None:
                for item in investigation_state.hypotheses:
                    sources = [
                        ProvenanceRef(kind="system_evidence", ref_id=evidence_id)
                        for evidence_id in item.evidence_ids
                    ]
                    seed_hypotheses.append(
                        HypothesisRecord(
                            hypothesis_id=item.hypothesis_id,
                            text=item.statement,
                            state=(
                                "rejected"
                                if item.state == "rejected"
                                else "active"
                            ),
                            reason=item.reason,
                            sources=sources,
                            updated_in_version=1,
                        )
                    )
                    memory_ids.add(item.hypothesis_id)
            last_user_message_id = next(
                (
                    str(message.id)
                    for message in reversed(identities.messages)
                    if isinstance(message, HumanMessage) and message.id
                ),
                None,
            )
            saved = store.save_progress(
                thread_id,
                phase=phase,
                summary=summary,
                facts=facts,
                evidence=evidence,
                hypotheses=hypotheses,
                checked_files=checked_files,
                experiments=experiments,
                next_steps=next_steps,
                unresolved_questions=unresolved_questions,
                coverage=SnapshotCoverage(
                    last_user_message_id=last_user_message_id,
                    covered_message_ids=[
                        str(message.id)
                        for message in identities.messages
                        if message.id
                    ],
                ),
                valid_sources={
                    "user_message": user_message_ids,
                    "work_unit": work_unit_ids,
                    "working_memory": memory_ids,
                    "system_evidence": deterministic_evidence_ids,
                    "artifact": set(),
                    "snapshot_record": set(),
                },
                seed_hypotheses=seed_hypotheses,
            )
        except (ValidationError, ValueError) as exc:
            return ToolMessage(
                content=f"保存工作记忆失败：{exc}",
                name="save_progress",
                tool_call_id=runtime.tool_call_id or "",
                status="error",
                id=stable_generated_message_id(
                    thread_id,
                    runtime.tool_call_id or "save-progress",
                    "validation_error",
                ),
            )

        return ToolMessage(
            content=f"工作记忆已保存为版本 {saved.version}",
            name="save_progress",
            tool_call_id=runtime.tool_call_id or "",
            status="success",
            artifact={
                "version": saved.version,
                "claim_ids": [item.claim_id for item in saved.snapshot.facts],
                "hypothesis_ids": [
                    item.hypothesis_id for item in saved.snapshot.all_hypotheses()
                ],
            },
            id=stable_generated_message_id(
                thread_id,
                runtime.tool_call_id or f"save-progress-{saved.version}",
                "success",
            ),
        )

    return StructuredTool.from_function(
        func=save_progress,
        name="save_progress",
        description=(
            "保存当前修复任务的完整进度快照，使关键事实、证据、假设和下一步"
            "在对话压缩后仍可恢复。任务 ID 由运行时自动提供。"
        ),
    )


def build_context_middleware(
    model,
    backend,
    store: WorkingMemoryStore,
    investigation_store: InvestigationStore,
    research_store: ResearchEvidenceStore | None = None,
):
    summarization = SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=("fraction", 0.70),
        keep=("fraction", 0.15),
        truncate_args_settings={
            "trigger": ("fraction", 0.70),
            "keep": ("fraction", 0.15),
            "max_length": 2000,
            "truncation_text": "...(argument truncated)",
        },
    )
    evidence_store = research_store or ResearchEvidenceStore(store.database_path)
    return [
        summarization,
        SummarizationToolMiddleware(
            summarization,
            system_prompt=(
                "长任务中完成独立阶段后，先用 save_progress 保存关键事实，"
                "再在上下文足够长时调用 compact_conversation。"
            ),
        ),
        PromptPolicyMiddleware(investigation_store),
        ContextMemoryMiddleware(store),
        ResearchEvidenceMiddleware(evidence_store),
    ]


def render_working_memory(version: WorkingMemoryVersion) -> str:
    snapshot = version.snapshot
    sections = [
        f'<deepfix_working_memory version="{version.version}">',
        f"<phase>{_bounded(snapshot.phase, 40)}</phase>",
        f"<summary>{_bounded(snapshot.summary, 1200)}</summary>",
        _render_claims(snapshot.facts),
        _render_evidence(snapshot.evidence),
        _render_hypotheses(
            "active_hypotheses",
            snapshot.active_hypotheses,
            5,
            200,
        ),
        _render_hypotheses(
            "rejected_hypotheses",
            snapshot.rejected_hypotheses,
            10,
            240,
        ),
        _render_hypotheses(
            "confirmed_hypotheses",
            snapshot.confirmed_hypotheses,
            10,
            240,
        ),
        _render_text_items(
            "checked_files", "file", snapshot.checked_files, 20, 240
        ),
        _render_text_items(
            "experiments", "experiment", snapshot.experiments, 15, 300
        ),
        _render_text_items("next_steps", "step", snapshot.next_steps, 5, 200),
        _render_text_items(
            "unresolved_questions",
            "question",
            snapshot.unresolved_questions,
            5,
            200,
        ),
        _render_coverage(snapshot.coverage),
        "</deepfix_working_memory>",
    ]
    return "\n".join(sections)


class ContextMemoryMiddleware(AgentMiddleware):
    def __init__(self, store: WorkingMemoryStore) -> None:
        self.store = store

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        thread_id = self._thread_id(request)
        if not thread_id:
            return handler(request)
        latest = self.store.latest(thread_id)
        if latest is None:
            return handler(request)

        memory_block = render_working_memory(latest)
        original = request.system_message.text if request.system_message else ""
        content = f"{original}\n\n{memory_block}" if original else memory_block
        system_message = SystemMessage(content=content)
        estimate = count_tokens_approximately(
            [system_message, *request.messages],
            tools=request.tools or [],
        )
        self.store.record_peak_tokens(thread_id, estimate)
        return handler(request.override(system_message=system_message))

    @staticmethod
    def _thread_id(request: ModelRequest) -> str:
        execution_info = (
            request.runtime.execution_info if request.runtime is not None else None
        )
        if execution_info is not None and execution_info.thread_id:
            return execution_info.thread_id.strip()
        try:
            config = get_config()
        except RuntimeError:
            return ""
        return str(config.get("configurable", {}).get("thread_id", "")).strip()


def _bounded(value: str, limit: int) -> str:
    escaped = escape(value)
    if len(escaped) <= limit:
        return escaped
    return escaped[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _render_text_items(
    section_name: str,
    item_name: str,
    values: list[str],
    item_limit: int,
    character_limit: int,
) -> str:
    lines = [f"<{section_name}>"]
    lines.extend(
        f"<{item_name}>{_bounded(value, character_limit)}</{item_name}>"
        for value in values[:item_limit]
    )
    if len(values) > item_limit:
        lines.append(f"<truncated>{_TRUNCATION_MARKER}</truncated>")
    lines.append(f"</{section_name}>")
    return "\n".join(lines)


def _render_evidence(values: list[Evidence]) -> str:
    lines = ["<evidence>"]
    lines.extend(
        (
            "<item>"
            f"<source>{_bounded(value.source, 120)}</source>"
            f"<observation>{_bounded(value.observation, 240)}</observation>"
            "</item>"
        )
        for value in values[:10]
    )
    if len(values) > 10:
        lines.append(f"<truncated>{_TRUNCATION_MARKER}</truncated>")
    lines.append("</evidence>")
    return "\n".join(lines)


def _render_claims(values: list[ProvenancedClaim] | list[str]) -> str:
    lines = ["<facts>"]
    for value in values[:8]:
        text = value if isinstance(value, str) else value.text
        claim_id = "legacy" if isinstance(value, str) else value.claim_id
        provenance = "" if isinstance(value, str) else _render_provenance(value.sources)
        lines.append(
            f'<fact claim_id="{_bounded(claim_id, 100)}">'
            f"{_bounded(text, 200)}{provenance}</fact>"
        )
    if len(values) > 8:
        lines.append(f'<omitted_count value="{len(values) - 8}" />')
    lines.append("</facts>")
    return "\n".join(lines)


def _render_hypotheses(
    section_name: str,
    values: list[HypothesisRecord] | list[str],
    item_limit: int,
    character_limit: int,
) -> str:
    lines = [f"<{section_name}>"]
    for value in values[:item_limit]:
        if isinstance(value, str):
            hypothesis_id, text, reason = "legacy", value, None
        else:
            hypothesis_id, text, reason = value.hypothesis_id, value.text, value.reason
        provenance = "" if isinstance(value, str) else _render_provenance(value.sources)
        lines.append(
            f'<hypothesis hypothesis_id="{_bounded(hypothesis_id, 100)}">'
            f"<text>{_bounded(text, character_limit)}</text>"
            f"<reason>{_bounded(reason or '', character_limit)}</reason>"
            f"{provenance}"
            "</hypothesis>"
        )
    if len(values) > item_limit:
        lines.append(f'<omitted_count value="{len(values) - item_limit}" />')
    lines.append(f"</{section_name}>")
    return "\n".join(lines)


def _render_provenance(values: list[ProvenanceRef]) -> str:
    return "<sources>" + "".join(
        f'<source kind="{source.kind}" ref_id="{_bounded(source.ref_id, 120)}" />'
        for source in values[:10]
    ) + "</sources>"


def _render_coverage(value: SnapshotCoverage) -> str:
    lines = ["<coverage>"]
    if value.last_user_message_id:
        lines.append(
            f"<last_user_message_id>{_bounded(value.last_user_message_id, 100)}</last_user_message_id>"
        )
    lines.append(f"<covered_message_count>{len(value.covered_message_ids)}</covered_message_count>")
    lines.append(
        f"<covered_work_unit_count>{len(value.covered_work_unit_ids)}</covered_work_unit_count>"
    )
    lines.append("</coverage>")
    return "\n".join(lines)
