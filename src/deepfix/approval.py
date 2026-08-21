from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from deepfix.config import ApprovalMode


class RiskLevel(StrEnum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class PolicyAction(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    risk: RiskLevel
    action: PolicyAction
    reason: str


_READ_ONLY_TOOLS = frozenset({"ls", "read_file", "glob", "grep"})
_WRITE_TOOLS = frozenset({"write_file", "edit_file"})

_DESTRUCTIVE_COMMANDS = (
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\s+-[a-z]*f[a-z]*\b", re.IGNORECASE),
    re.compile(r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\b", re.IGNORECASE),
    re.compile(r"\brm\s+-[a-z]*f[a-z]*r[a-z]*\b", re.IGNORECASE),
    re.compile(r"\bremove-item\b.*\s-(?:recurse|r)\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/s\b", re.IGNORECASE),
    re.compile(r"\b(?:shutdown|reboot|restart-computer|stop-computer)\b", re.IGNORECASE),
)

_SENSITIVE_COMMANDS = (
    re.compile(r"(?:^|\s)(?:pip|python\s+-m\s+pip|uv)\s+install\b", re.IGNORECASE),
    re.compile(r"\bgit\s+(?:commit|push)\b", re.IGNORECASE),
    re.compile(r"(?:^|\s)(?:export\s+\w+=|set\s+\w+=|setx\s+\w+|\$env:)", re.IGNORECASE),
    re.compile(r"(?:^|\s)(?:[a-z]:[\\/]|/)", re.IGNORECASE),
    re.compile(r"&&|\|\||[|><;\r\n]"),
)

_ROUTINE_COMMAND = re.compile(
    r"^(?:pytest(?:\s|$)|python\s+-m\s+pytest(?:\s|$)|ruff\s+(?:check|format)(?:\s|$))",
    re.IGNORECASE,
)


class ApprovalPolicy:
    def __init__(self, mode: ApprovalMode) -> None:
        self.mode = mode

    def evaluate(self, tool_name: str, args: Mapping[str, object]) -> PolicyDecision:
        if tool_name in _READ_ONLY_TOOLS:
            return PolicyDecision(RiskLevel.L0, PolicyAction.ALLOW, "只读操作，允许执行")

        if tool_name in _WRITE_TOOLS:
            return self._routine_decision("项目内常规文件修改")

        if tool_name != "execute":
            return PolicyDecision(
                RiskLevel.L2,
                PolicyAction.ASK,
                "未知工具可能产生副作用，需要人工审批",
            )

        command = str(args.get("command", "")).strip()
        if not command:
            return PolicyDecision(
                RiskLevel.L2,
                PolicyAction.ASK,
                "Shell 命令为空或缺失，需要人工审批",
            )

        if any(pattern.search(command) for pattern in _DESTRUCTIVE_COMMANDS):
            return PolicyDecision(
                RiskLevel.L3,
                PolicyAction.DENY,
                "检测到破坏性命令，拒绝执行",
            )

        if ".." in command or any(pattern.search(command) for pattern in _SENSITIVE_COMMANDS):
            return PolicyDecision(
                RiskLevel.L2,
                PolicyAction.ASK,
                "命令可能改变环境、越过项目边界或组合执行，需要人工审批",
            )

        if _ROUTINE_COMMAND.match(command):
            return self._routine_decision("项目内常规测试或静态检查")

        return PolicyDecision(
            RiskLevel.L2,
            PolicyAction.ASK,
            "命令不在常规操作列表中，需要人工审批",
        )

    def _routine_decision(self, reason: str) -> PolicyDecision:
        action = PolicyAction.ASK if self.mode is ApprovalMode.MANUAL else PolicyAction.ALLOW
        return PolicyDecision(RiskLevel.L1, action, reason)
