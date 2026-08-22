from __future__ import annotations

from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langgraph.config import get_config

from deepfix.memory import WorkingMemoryStore
from deepfix.prompts import CORE_REPAIR_PROMPT, PHASE_PROMPTS, RESEARCH_POLICY_PROMPT


class PromptPolicyMiddleware(AgentMiddleware):
    def __init__(self, store: WorkingMemoryStore) -> None:
        self.store = store

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        task_id = model_request_task_id(request)
        latest = self.store.latest(task_id) if task_id else None
        phase = latest.snapshot.phase if latest is not None else "investigating"
        original = request.system_message.text if request.system_message else ""
        parts = [original] if original else []
        if CORE_REPAIR_PROMPT not in original:
            parts.append(CORE_REPAIR_PROMPT)
        parts.extend((PHASE_PROMPTS[phase], RESEARCH_POLICY_PROMPT))
        return handler(
            request.override(system_message=SystemMessage(content="\n\n".join(parts)))
        )


def model_request_task_id(request: ModelRequest) -> str:
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
