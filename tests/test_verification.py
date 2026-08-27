from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage

from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.models import SystemTestEvidence
from deepfix.compaction.store import CompactionStore
from deepfix.config import ApprovalMode
from deepfix.models import TaskState
from deepfix.research.store import ResearchEvidenceStore
from deepfix.verification import (
    VerificationPolicyBuilder,
    VerificationPolicyConflict,
    VerificationPolicyStore,
    evaluate_required_oracles,
)
from deepfix.workspace import WorkspaceFactory


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    test = root / "tests" / "test_value.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_value(): assert True\n", encoding="utf-8")
    (root / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


@pytest.fixture
def task_workspace(tmp_path, project):
    return WorkspaceFactory(tmp_path / "workspaces").create("task-a", project)


@pytest.fixture
def task(task_workspace):
    value = TaskState.create(
        task_workspace.root,
        "只允许修改 value.py，并运行 python -m pytest tests/test_value.py -q 验证",
        ApprovalMode.MANUAL,
        source_project_root=task_workspace.baseline.source_root,
        workspace_baseline_id=task_workspace.baseline.baseline_id,
    )
    value.task_id = "task-a"
    return value


def test_existing_user_test_is_required_targeted_oracle(task_workspace, task):
    policy = VerificationPolicyBuilder().build(task, task_workspace)

    oracle = policy.required_oracles[0]
    assert oracle.origin == "user_specified"
    assert oracle.scope == "targeted"
    assert oracle.command == "python -m pytest tests/test_value.py -q"


def test_agent_created_test_cannot_be_required(task_workspace, task):
    generated = task_workspace.root / "tests" / "test_generated.py"
    generated.write_text("def test_x(): assert True\n", encoding="utf-8")

    policy = VerificationPolicyBuilder().build(task, task_workspace)

    assert all("test_generated.py" not in item.command for item in policy.required_oracles)
    assert all(item.origin != "agent_generated" for item in policy.required_oracles)


def test_policy_cannot_downgrade_required_oracle(tmp_path, task_workspace, task):
    store = VerificationPolicyStore(tmp_path / "state.db")
    policy = VerificationPolicyBuilder().build(task, task_workspace)
    store.save(policy)
    changed = policy.model_copy(
        update={
            "version": policy.version + 1,
            "required_oracles": [],
            "supplemental_oracles": policy.required_oracles,
        }
    )

    with pytest.raises(VerificationPolicyConflict):
        store.save(changed)


def test_repository_suite_is_required_when_user_did_not_name_a_test(
    task_workspace,
    task,
):
    task.user_problem = "修复 value.py 中的问题并验证"

    policy = VerificationPolicyBuilder().build(task, task_workspace)

    assert [item.command for item in policy.required_oracles] == [
        "python -m pytest -q"
    ]


def test_required_unavailable_and_related_suite_failure_block_fixed(
    task_workspace,
    task,
):
    policy = VerificationPolicyBuilder().build(task, task_workspace)
    unavailable = evaluate_required_oracles(policy, [])
    targeted = policy.required_oracles[0]
    targeted_pass = _test_evidence(
        targeted.command,
        origin="user_specified",
        scope="targeted",
        exit_code=0,
        evidence_id="targeted-pass",
    )
    suite_failure = _test_evidence(
        "python -m pytest -q",
        origin="repository_existing",
        scope="full_suite",
        exit_code=1,
        evidence_id="suite-failure",
    )
    conflict = evaluate_required_oracles(
        policy,
        [targeted_pass, suite_failure],
    )

    assert unavailable.fixed_allowed is False
    assert unavailable.unavailable_required_oracle_ids == [targeted.oracle_id]
    assert conflict.all_required_satisfied is True
    assert conflict.fixed_allowed is False
    assert conflict.conflicting_evidence_ids == ["suite-failure"]


def test_modified_or_new_test_is_downgraded_to_agent_generated(
    tmp_path,
    task_workspace,
    task,
):
    generated = task_workspace.root / "tests" / "test_generated.py"
    generated.write_text("def test_x(): assert True\n", encoding="utf-8")
    database = tmp_path / "evidence.db"
    collector = EvidenceCollector(
        CompactionStore(database),
        ResearchEvidenceStore(database),
    )
    call = {
        "name": "execute",
        "args": {"command": "python -m pytest tests/test_generated.py -q"},
        "id": "generated-test",
        "type": "tool_call",
    }
    result = ToolMessage(
        id="generated-result",
        content="1 passed",
        name="execute",
        tool_call_id="generated-test",
        artifact={"exit_code": 0},
    )

    evidence = collector.collect_pair(task.task_id, call, result, task)

    assert evidence.origin == "agent_generated"
    assert evidence.timing == "post_change"


def _test_evidence(
    command: str,
    *,
    origin: str,
    scope: str,
    exit_code: int,
    evidence_id: str,
) -> SystemTestEvidence:
    return SystemTestEvidence(
        evidence_id=evidence_id,
        command=command,
        exit_code=exit_code,
        summary="test result",
        tool_call_id=f"call-{evidence_id}",
        source_message_id=f"message-{evidence_id}",
        origin=origin,
        scope=scope,
        timing="post_change",
        workspace_baseline_id="baseline-a",
        code_state_hash="code-a",
    )
