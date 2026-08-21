from __future__ import annotations

from deepagents import create_deep_agent
from deepagents.profiles.harness import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from langchain_deepseek import ChatDeepSeek

from deepfix.backend import build_backend
from deepfix.config import AppConfig
from deepfix.models import RepairOutcome
from deepfix.prompts import REPAIR_SYSTEM_PROMPT


def build_model(config: AppConfig) -> ChatDeepSeek:
    return ChatDeepSeek(model=config.model_name, temperature=0)


def build_agent(config: AppConfig, checkpointer):
    register_harness_profile(
        f"deepseek:{config.model_name}",
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
        ),
    )
    return create_deep_agent(
        model=build_model(config),
        system_prompt=REPAIR_SYSTEM_PROMPT,
        backend=build_backend(config),
        subagents=[],
        response_format=RepairOutcome,
        interrupt_on={
            "write_file": True,
            "edit_file": True,
            "delete": True,
            "execute": True,
        },
        checkpointer=checkpointer,
        name="deepfix_repair_agent",
    )
