from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage

from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.models import SystemTestEvidence
from deepfix.domain_repositories.evidence import EvidenceRepository
from deepfix.task_domain.models import TaskDefinition
from deepfix.task_domain.repository import TaskRepository
from deepfix.verification import (
    VerificationPolicy,
    VerificationPolicyBuilder,
    VerificationPolicyConflict,
    evaluate_required_oracles,
    pytest_target_paths,
    unavailable_required_oracle_paths,
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
    return TaskDefinition(
        task_id="task-a",
        original_message_id="message-task-a",
        original_problem="只允许修改 value.py，并运行 python -m pytest tests/test_value.py -q 验证",
        approval_mode="manual",
        source_project_root=task_workspace.baseline.source_root,
        workspace_root=str(task_workspace.root),
        workspace_baseline_id=task_workspace.baseline.baseline_id,
        project_python="python",
        confinement_level="guarded_local",
        created_at="2026-08-31T00:00:00+00:00",
    )


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
    store = TaskRepository(tmp_path / "state.db")
    policy = VerificationPolicyBuilder().build(task, task_workspace)
    store.save_verification_policy(policy)
    changed = policy.model_copy(
        update={
            "version": policy.version + 1,
            "required_oracles": [],
            "supplemental_oracles": policy.required_oracles,
        }
    )

    with pytest.raises(VerificationPolicyConflict):
        store.save_verification_policy(changed)


def test_policy_is_persisted_by_task_repository(
    tmp_path,
    task_workspace,
    task,
):
    tasks = TaskRepository(tmp_path / "state.db")
    policy = VerificationPolicyBuilder().build(task, task_workspace)

    tasks.save_verification_policy(policy)

    assert tasks.load_verification_policy(task.task_id) == policy


def test_verification_policy_has_no_workspace_allowed_paths(
    task_workspace,
    task,
):
    policy = VerificationPolicyBuilder().build(task, task_workspace)

    assert "allowed_paths" not in VerificationPolicy.model_fields
    assert "allowed_paths" not in policy.model_dump()


def test_relevant_paths_do_not_authorize_workspace_mutation(
    task_workspace,
    task,
):
    policy = VerificationPolicyBuilder().build(task, task_workspace)
    oracle = policy.required_oracles[0].model_copy(
        update={"relevant_paths": ["tests/test_value.py"]}
    )

    assert oracle.relevant_paths == ["tests/test_value.py"]
    assert not hasattr(oracle, "can_modify")


def test_repository_suite_is_required_when_user_did_not_name_a_test(
    task_workspace,
    task,
):
    task = task.model_copy(update={"original_problem": "修复 value.py 中的问题并验证"})

    policy = VerificationPolicyBuilder().build(task, task_workspace)

    assert [item.command for item in policy.required_oracles] == ["python -m pytest -q"]


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


@pytest.mark.parametrize("exit_code", [2, 3, 4, 5, -1])
def test_pytest_usage_error_keeps_required_oracle_unavailable(
    task_workspace,
    task,
    exit_code,
):
    policy = VerificationPolicyBuilder().build(task, task_workspace)
    oracle = policy.required_oracles[0]
    usage_error = _test_evidence(
        oracle.command,
        origin="user_specified",
        scope="targeted",
        exit_code=exit_code,
        evidence_id="pytest-usage-error",
    ).model_copy(update={"summary": "pytest: error: unrecognized arguments: --timeout=5"})

    evaluation = evaluate_required_oracles(policy, [usage_error])

    assert evaluation.unavailable_required_oracle_ids == [oracle.oracle_id]
    assert evaluation.failed_required_oracle_ids == []
    assert evaluation.fixed_allowed is False


@pytest.mark.parametrize("exit_code", [2, 3, 4, 5, -1])
def test_pytest_usage_error_is_not_a_repository_failure_conflict(
    task_workspace,
    task,
    exit_code,
):
    policy = VerificationPolicyBuilder().build(task, task_workspace)
    oracle = policy.required_oracles[0]
    targeted_pass = _test_evidence(
        oracle.command,
        origin="user_specified",
        scope="targeted",
        exit_code=0,
        evidence_id="targeted-pass",
    )
    suite_usage_error = _test_evidence(
        "python -m pytest -q",
        origin="repository_existing",
        scope="full_suite",
        exit_code=exit_code,
        evidence_id="suite-usage-error",
    )

    evaluation = evaluate_required_oracles(
        policy,
        [targeted_pass, suite_usage_error],
    )

    assert evaluation.conflicting_evidence_ids == []
    assert evaluation.fixed_allowed is True


@pytest.mark.parametrize(
    ("current_hash", "baseline", "allowed"),
    [
        ("code-a", "baseline-a", True),
        ("code-b", "baseline-a", False),
        ("code-a", "baseline-b", False),
    ],
)
def test_verification_is_bound_to_live_workspace(
    task_workspace, task, current_hash, baseline, allowed
):
    policy = VerificationPolicyBuilder().build(task, task_workspace)
    passing = _test_evidence(
        policy.required_oracles[0].command,
        origin="user_specified",
        scope="targeted",
        exit_code=0,
        evidence_id="pass",
    )
    result = evaluate_required_oracles(
        policy, [passing], current_code_state_hash=current_hash, workspace_baseline_id=baseline
    )
    assert result.fixed_allowed is allowed


def test_old_repository_failure_does_not_block_current_pass(task_workspace, task):
    policy = VerificationPolicyBuilder().build(task, task_workspace)
    failure = _test_evidence(
        "python -m pytest -q",
        origin="repository_existing",
        scope="full_suite",
        exit_code=1,
        evidence_id="old-fail",
    )
    passing = _test_evidence(
        policy.required_oracles[0].command,
        origin="user_specified",
        scope="targeted",
        exit_code=0,
        evidence_id="current-pass",
    ).model_copy(update={"code_state_hash": "code-b"})
    result = evaluate_required_oracles(policy, [failure, passing], current_code_state_hash="code-b")
    assert result.fixed_allowed
    assert result.accepted_required_evidence_ids == ["current-pass"]


def test_same_version_repository_failure_still_blocks_current_pass(task_workspace, task):
    policy = VerificationPolicyBuilder().build(task, task_workspace)
    failure = _test_evidence(
        "python -m pytest -q",
        origin="repository_existing",
        scope="full_suite",
        exit_code=1,
        evidence_id="fail",
    )
    passing = _test_evidence(
        policy.required_oracles[0].command,
        origin="user_specified",
        scope="targeted",
        exit_code=0,
        evidence_id="pass",
    )
    result = evaluate_required_oracles(policy, [failure, passing], current_code_state_hash="code-a")
    assert not result.fixed_allowed
    assert result.conflicting_evidence_ids == ["fail"]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("python -m pytest -k test_regression -q", []),
        ("python -m pytest -m slow -q", []),
        ("python -m pytest --ignore tests/test_optional.py tests -q", ["tests"]),
        (
            "python -m pytest --log-file reports/test.log tests/test_value.py -q",
            ["tests/test_value.py"],
        ),
        (
            "python -m pytest --cov-report xml:reports/coverage.xml tests -q",
            ["tests"],
        ),
        (
            "python -m pytest --setup-show tests/test_missing.py -q",
            ["tests/test_missing.py"],
        ),
        (
            "python -m pytest --plugin-flag tests/test_missing.py -q",
            ["tests/test_missing.py"],
        ),
        ("python -m pytest integration/ -q", ["integration/"]),
        ("python -m pytest test_suite -q", ["test_suite"]),
        ("python -m pytest ./tests/test_value.py -q", ["tests/test_value.py"]),
        ("python -m pytest ../outside/test_value.py -q", ["../outside/test_value.py"]),
    ],
)
def test_pytest_target_paths_only_returns_positional_targets(command, expected):
    assert pytest_target_paths(command) == expected


def test_required_oracle_path_preflight_rejects_workspace_escape(
    task_workspace,
    task,
):
    policy = VerificationPolicyBuilder().build(task, task_workspace)
    escaped = policy.required_oracles[0].model_copy(
        update={"relevant_paths": ["../outside/test_value.py"]}
    )
    policy = policy.model_copy(update={"required_oracles": [escaped]})

    assert unavailable_required_oracle_paths(policy, task_workspace.root) == [
        "../outside/test_value.py"
    ]


def test_modified_or_new_test_is_downgraded_to_agent_generated(
    tmp_path,
    task_workspace,
    task,
):
    generated = task_workspace.root / "tests" / "test_generated.py"
    generated.write_text("def test_x(): assert True\n", encoding="utf-8")
    database = tmp_path / "evidence.db"
    collector = EvidenceCollector(EvidenceRepository(database))
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
