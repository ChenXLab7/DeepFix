from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command

from deepfix.compaction.coordinator import CompactionCoordinator


def build_compact_conversation_tool(
    coordinator: CompactionCoordinator,
) -> BaseTool:
    def compact_conversation(runtime: ToolRuntime) -> Command:
        return coordinator.compact_manually(runtime)

    async def acompact_conversation(runtime: ToolRuntime) -> Command:
        return await coordinator.acompact_manually(runtime)

    return StructuredTool.from_function(
        func=compact_conversation,
        coroutine=acompact_conversation,
        name="compact_conversation",
        description=(
            "在上下文达到压缩条件时，先保存可恢复的完整历史，再生成结构化 Snapshot。"
            "任务、消息、预算和 artifact 路径均由运行时确定。"
        ),
    )
