from __future__ import annotations

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import StructuredTool

from deepfix.approval import PolicyAction, RiskLevel
from deepfix.extensions import (
    AgentExtensions,
    ToolRegistration,
    build_research_extensions,
    merge_extensions,
)


def _tool(name: str):
    def run() -> str:
        return "ok"

    return StructuredTool.from_function(run, name=name, description=f"{name} test tool")


def _registration(
    name: str,
    *,
    risk: RiskLevel = RiskLevel.L0,
    action: PolicyAction = PolicyAction.ALLOW,
    network_access: bool = False,
) -> ToolRegistration:
    return ToolRegistration(
        tool=_tool(name),
        risk=risk,
        policy_action=action,
        network_access=network_access,
    )


def test_duplicate_extension_tool_names_are_rejected():
    extensions = AgentExtensions(
        tools=(_registration("inspect"), _registration("inspect"))
    )

    with pytest.raises(ValueError, match="重复 Tool 名称"):
        merge_extensions(extensions)


def test_extension_cannot_replace_an_existing_tool():
    extensions = AgentExtensions(tools=(_registration("save_progress"),))

    with pytest.raises(ValueError, match="重复 Tool 名称"):
        merge_extensions(extensions, existing_tool_names={"save_progress"})


def test_invalid_tool_policy_metadata_is_rejected():
    registration = ToolRegistration(
        tool=_tool("unsafe_metadata"),
        risk="L0",  # type: ignore[arg-type]
        policy_action=PolicyAction.ALLOW,
        network_access=False,
    )

    with pytest.raises(TypeError, match="RiskLevel"):
        merge_extensions(AgentExtensions(tools=(registration,)))


def test_network_access_must_be_an_explicit_boolean():
    registration = ToolRegistration(
        tool=_tool("network_tool"),
        risk=RiskLevel.L1,
        policy_action=PolicyAction.ALLOW,
        network_access="yes",  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="network_access"):
        merge_extensions(AgentExtensions(tools=(registration,)))


def test_policy_action_must_use_the_declared_enum():
    registration = ToolRegistration(
        tool=_tool("invalid_action"),
        risk=RiskLevel.L1,
        policy_action="allow",  # type: ignore[arg-type]
        network_access=False,
    )

    with pytest.raises(TypeError, match="PolicyAction"):
        merge_extensions(AgentExtensions(tools=(registration,)))


@pytest.mark.parametrize(
    "protected_name",
    [
        "FilesystemMiddleware",
        "SummarizationMiddleware",
        "SummarizationToolMiddleware",
        "ContextMemoryMiddleware",
        "MessageIdentityMiddleware",
        "LegacyContextMigrationMiddleware",
        "ProtectedContextMiddleware",
        "DeepFixCompactionMiddleware",
        "HumanInTheLoopMiddleware",
    ],
)
def test_protected_middleware_cannot_be_replaced(protected_name):
    protected_type = type(protected_name, (AgentMiddleware,), {})
    extensions = AgentExtensions(middleware=(protected_type(),))

    with pytest.raises(ValueError, match="受保护 Middleware"):
        merge_extensions(extensions)


def test_skills_default_to_none():
    resolved = merge_extensions(AgentExtensions())

    assert resolved.skill_sources == ()


def test_skill_source_requires_an_explicit_allowlist(tmp_path):
    skill_dir = tmp_path / "skills" / "python-debugging"
    skill_dir.mkdir(parents=True)
    extensions = AgentExtensions(skill_sources=(str(skill_dir),))

    with pytest.raises(ValueError, match="Skill 目录未被允许"):
        merge_extensions(extensions)


def test_skill_source_must_stay_beneath_allowed_directory(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    extensions = AgentExtensions(skill_sources=(str(outside),))

    with pytest.raises(ValueError, match="Skill 目录未被允许"):
        merge_extensions(extensions, allowed_skill_roots=(allowed,))


def test_allowed_local_skill_directory_is_normalized(tmp_path):
    allowed = tmp_path / "allowed"
    skill_dir = allowed / "python-debugging"
    skill_dir.mkdir(parents=True)

    resolved = merge_extensions(
        AgentExtensions(skill_sources=(str(skill_dir),)),
        allowed_skill_roots=(allowed,),
    )

    assert resolved.skill_sources == (str(skill_dir.resolve()),)


def test_research_extensions_have_approved_metadata():
    extensions = build_research_extensions(
        inspect_dependency=_tool("inspect_dependency"),
        search_technical_sources=_tool("search_technical_sources"),
        fetch_external_evidence=_tool("fetch_external_evidence"),
        link_external_evidence=_tool("link_external_evidence"),
    )

    metadata = {
        item.tool.name: (item.risk, item.policy_action, item.network_access)
        for item in extensions.tools
    }
    assert metadata == {
        "inspect_dependency": (RiskLevel.L0, PolicyAction.ALLOW, False),
        "search_technical_sources": (RiskLevel.L1, PolicyAction.ALLOW, True),
        "fetch_external_evidence": (RiskLevel.L1, PolicyAction.ALLOW, True),
        "link_external_evidence": (RiskLevel.L0, PolicyAction.ALLOW, False),
    }


def test_research_factory_rejects_a_misnamed_tool():
    with pytest.raises(ValueError, match="research Tool 名称"):
        build_research_extensions(
            inspect_dependency=_tool("wrong_name"),
            search_technical_sources=_tool("search_technical_sources"),
            fetch_external_evidence=_tool("fetch_external_evidence"),
            link_external_evidence=_tool("link_external_evidence"),
        )
