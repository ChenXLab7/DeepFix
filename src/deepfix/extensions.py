from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool

from deepfix.approval import PolicyAction, RiskLevel

_PROTECTED_MIDDLEWARE_NAMES = frozenset(
    {
        "FilesystemMiddleware",
        "SummarizationMiddleware",
        "SummarizationToolMiddleware",
        "ContextMemoryMiddleware",
        "HumanInTheLoopMiddleware",
    }
)
_RESEARCH_TOOL_POLICY = {
    "inspect_dependency": (RiskLevel.L0, False),
    "search_technical_sources": (RiskLevel.L1, True),
    "fetch_external_evidence": (RiskLevel.L1, True),
    "link_external_evidence": (RiskLevel.L0, False),
}


@dataclass(frozen=True)
class ToolRegistration:
    tool: BaseTool
    risk: RiskLevel
    policy_action: PolicyAction
    network_access: bool = False


@dataclass(frozen=True)
class AgentExtensions:
    tools: tuple[ToolRegistration, ...] = ()
    middleware: tuple[AgentMiddleware, ...] = ()
    skill_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedAgentExtensions:
    tools: tuple[ToolRegistration, ...]
    middleware: tuple[AgentMiddleware, ...]
    skill_sources: tuple[str, ...]


def merge_extensions(
    extensions: AgentExtensions,
    *,
    existing_tool_names: set[str] | frozenset[str] = frozenset(),
    allowed_skill_roots: tuple[str | Path, ...] = (),
) -> ResolvedAgentExtensions:
    seen = set(existing_tool_names)
    for registration in extensions.tools:
        _validate_registration(registration)
        name = registration.tool.name.strip()
        if name in seen:
            raise ValueError(f"重复 Tool 名称: {name}")
        seen.add(name)

    for middleware in extensions.middleware:
        protected_base = next(
            (
                base.__name__
                for base in type(middleware).__mro__
                if base.__name__ in _PROTECTED_MIDDLEWARE_NAMES
            ),
            None,
        )
        if protected_base is not None:
            raise ValueError(f"不能替换受保护 Middleware: {protected_base}")

    skill_sources = _resolve_skill_sources(
        extensions.skill_sources,
        allowed_skill_roots,
    )
    return ResolvedAgentExtensions(
        tools=extensions.tools,
        middleware=extensions.middleware,
        skill_sources=skill_sources,
    )


def build_research_extensions(
    *,
    inspect_dependency: BaseTool,
    search_technical_sources: BaseTool,
    fetch_external_evidence: BaseTool,
    link_external_evidence: BaseTool,
) -> AgentExtensions:
    supplied = (
        inspect_dependency,
        search_technical_sources,
        fetch_external_evidence,
        link_external_evidence,
    )
    registrations: list[ToolRegistration] = []
    for tool in supplied:
        metadata = _RESEARCH_TOOL_POLICY.get(tool.name)
        if metadata is None:
            raise ValueError(f"未知或错误的 research Tool 名称: {tool.name}")
        risk, network_access = metadata
        registrations.append(
            ToolRegistration(
                tool=tool,
                risk=risk,
                policy_action=PolicyAction.ALLOW,
                network_access=network_access,
            )
        )
    if {item.tool.name for item in registrations} != set(_RESEARCH_TOOL_POLICY):
        raise ValueError("research Tool 名称不完整或重复")
    return AgentExtensions(tools=tuple(registrations))


def _validate_registration(registration: ToolRegistration) -> None:
    if not isinstance(registration.tool, BaseTool):
        raise TypeError("ToolRegistration.tool 必须是 BaseTool")
    if not isinstance(registration.risk, RiskLevel):
        raise TypeError("ToolRegistration.risk 必须是 RiskLevel")
    if not isinstance(registration.policy_action, PolicyAction):
        raise TypeError("ToolRegistration.policy_action 必须是 PolicyAction")
    if type(registration.network_access) is not bool:
        raise TypeError("ToolRegistration.network_access 必须是 bool")
    if not registration.tool.name.strip():
        raise ValueError("Tool 名称不能为空")


def _resolve_skill_sources(
    sources: tuple[str, ...],
    allowed_roots: tuple[str | Path, ...],
) -> tuple[str, ...]:
    if not sources:
        return ()
    roots = tuple(Path(root).expanduser().resolve() for root in allowed_roots)
    resolved: list[str] = []
    for raw_source in sources:
        source = Path(raw_source).expanduser().resolve()
        if not source.is_dir() or not any(
            source == root or source.is_relative_to(root)
            for root in roots
        ):
            raise ValueError(f"Skill 目录未被允许: {source}")
        resolved.append(str(source))
    return tuple(dict.fromkeys(resolved))
