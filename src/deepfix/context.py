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
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_config
from pydantic import ValidationError

from deepfix.memory import ProgressSnapshot, WorkingMemoryStore, WorkingMemoryVersion
from deepfix.models import Evidence

_TRUNCATION_MARKER = "…[truncated]"


def build_save_progress_tool(store: WorkingMemoryStore) -> BaseTool:
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
        facts: list[str],
        evidence: list[Evidence],
        active_hypotheses: list[str],
        rejected_hypotheses: list[str],
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
            )

        try:
            snapshot = ProgressSnapshot(
                phase=phase,
                summary=summary,
                facts=facts,
                evidence=evidence,
                active_hypotheses=active_hypotheses,
                rejected_hypotheses=rejected_hypotheses,
                checked_files=checked_files,
                experiments=experiments,
                next_steps=next_steps,
                unresolved_questions=unresolved_questions,
            )
        except ValidationError as exc:
            return ToolMessage(
                content=f"保存工作记忆失败：{exc}",
                name="save_progress",
                tool_call_id=runtime.tool_call_id or "",
                status="error",
            )

        saved = store.save(thread_id, snapshot)
        return ToolMessage(
            content=f"工作记忆已保存为版本 {saved.version}",
            name="save_progress",
            tool_call_id=runtime.tool_call_id or "",
            status="success",
            artifact={"version": saved.version},
        )

    return StructuredTool.from_function(
        func=save_progress,
        name="save_progress",
        description=(
            "保存当前修复任务的完整进度快照，使关键事实、证据、假设和下一步"
            "在对话压缩后仍可恢复。任务 ID 由运行时自动提供。"
        ),
    )


def build_context_middleware(model, backend, store: WorkingMemoryStore):
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
    return [
        summarization,
        SummarizationToolMiddleware(
            summarization,
            system_prompt=(
                "长任务中完成独立阶段后，先用 save_progress 保存关键事实，"
                "再在上下文足够长时调用 compact_conversation。"
            ),
        ),
        ContextMemoryMiddleware(store),
    ]


def render_working_memory(version: WorkingMemoryVersion) -> str:
    snapshot = version.snapshot
    sections = [
        f'<deepfix_working_memory version="{version.version}">',
        f"<phase>{_bounded(snapshot.phase, 40)}</phase>",
        f"<summary>{_bounded(snapshot.summary, 1200)}</summary>",
        _render_text_items("facts", "fact", snapshot.facts, 8, 200),
        _render_evidence(snapshot.evidence),
        _render_text_items(
            "active_hypotheses",
            "hypothesis",
            snapshot.active_hypotheses,
            5,
            200,
        ),
        _render_text_items("next_steps", "step", snapshot.next_steps, 5, 200),
        _render_text_items(
            "unresolved_questions",
            "question",
            snapshot.unresolved_questions,
            5,
            200,
        ),
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
