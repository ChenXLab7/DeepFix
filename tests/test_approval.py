import pytest

from deepfix.approval import ApprovalPolicy, PolicyAction, RiskLevel
from deepfix.config import ApprovalMode


@pytest.mark.parametrize(
    ("mode", "tool", "args", "expected_risk", "expected_action"),
    [
        (ApprovalMode.MANUAL, "read_file", {"path": "/src/a.py"}, RiskLevel.L0, PolicyAction.ALLOW),
        (ApprovalMode.MANUAL, "write_file", {"path": "/src/a.py"}, RiskLevel.L1, PolicyAction.ASK),
        (ApprovalMode.GUARDED, "edit_file", {"path": "/src/a.py"}, RiskLevel.L1, PolicyAction.ALLOW),
        (ApprovalMode.MANUAL, "execute", {"command": "pytest -q"}, RiskLevel.L1, PolicyAction.ASK),
        (ApprovalMode.GUARDED, "execute", {"command": "python -m pytest -q"}, RiskLevel.L1, PolicyAction.ALLOW),
        (ApprovalMode.GUARDED, "execute", {"command": "ruff check src"}, RiskLevel.L1, PolicyAction.ALLOW),
        (ApprovalMode.GUARDED, "execute", {"command": "pip install rich"}, RiskLevel.L2, PolicyAction.ASK),
        (ApprovalMode.GUARDED, "execute", {"command": "pytest -q && echo done"}, RiskLevel.L2, PolicyAction.ASK),
        (ApprovalMode.GUARDED, "execute", {"command": "pytest ../tests"}, RiskLevel.L2, PolicyAction.ASK),
        (ApprovalMode.GUARDED, "execute", {"command": "pytest C:\\other\\tests"}, RiskLevel.L2, PolicyAction.ASK),
        (ApprovalMode.GUARDED, "execute", {"command": "git reset --hard"}, RiskLevel.L3, PolicyAction.DENY),
        (
            ApprovalMode.GUARDED,
            "execute",
            {"command": "Remove-Item build -Recurse"},
            RiskLevel.L3,
            PolicyAction.DENY,
        ),
        (
            ApprovalMode.GUARDED,
            "execute",
            {"command": "git reset --hard && pytest -q"},
            RiskLevel.L3,
            PolicyAction.DENY,
        ),
        (ApprovalMode.GUARDED, "execute", {}, RiskLevel.L2, PolicyAction.ASK),
        (ApprovalMode.GUARDED, "unknown_tool", {}, RiskLevel.L2, PolicyAction.ASK),
    ],
)
def test_policy_classifies_operation(mode, tool, args, expected_risk, expected_action):
    decision = ApprovalPolicy(mode).evaluate(tool, args)

    assert decision.risk is expected_risk
    assert decision.action is expected_action


def test_denied_operation_explains_rejection():
    decision = ApprovalPolicy(ApprovalMode.GUARDED).evaluate(
        "execute",
        {"command": "git clean -fd"},
    )

    assert decision.risk is RiskLevel.L3
    assert decision.action is PolicyAction.DENY
    assert "拒绝" in decision.reason


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "del /s build",
        "shutdown /s",
    ],
)
def test_destructive_shell_commands_are_denied(command):
    decision = ApprovalPolicy(ApprovalMode.GUARDED).evaluate("execute", {"command": command})

    assert decision.risk is RiskLevel.L3
    assert decision.action is PolicyAction.DENY


@pytest.mark.parametrize(
    "command",
    [
        "git commit -m fix",
        "export TOKEN=secret",
        "pytest -q | tee result.txt",
        "pytest -q > result.txt",
    ],
)
def test_sensitive_shell_syntax_uses_sensitive_rule(command):
    decision = ApprovalPolicy(ApprovalMode.GUARDED).evaluate("execute", {"command": command})

    assert decision.risk is RiskLevel.L2
    assert decision.action is PolicyAction.ASK
    assert "越过项目边界" in decision.reason
