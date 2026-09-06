from __future__ import annotations

import hashlib
from pathlib import Path

from deepfix.compaction.models import FileChangeEvidence, SystemTestEvidence
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.execution import create_execution_approval
from deepfix.investigation.models import InvestigationHypothesis
from deepfix.reporting import _repository_test_lines, build_task_report_view, render_report
from deepfix.task_domain.models import (
    AdjudicationDecision,
    TaskDefinition,
    TaskInput,
    TaskLifecycleStatus,
    TaskRun,
)
from deepfix.verification import VerificationOracle, VerificationPolicy

TASK_ID = "task-report-view"
COMMAND = "python -m pytest tests/test_value.py -q"


def test_report_is_reconstructed_from_authoritative_repositories(
    tmp_path: Path,
) -> None:
    repositories = DomainRepositories.create(tmp_path / "deepfix.db")
    _seed_task(repositories)
    _insert_stale_legacy_projection(repositories)

    view = build_task_report_view(repositories, TASK_ID)
    report = render_report(view)

    assert view.definition.original_problem == "修复 value.py 中的边界错误"
    assert view.decision is not None
    assert view.decision.evidence_ids == ["change-1", "test-pass"]
    assert view.hypotheses == []
    assert "value.py" in report
    assert "[通过] `python -m pytest tests/test_value.py -q`" in report
    assert "[L1] edit_file：approve" in report
    assert "处理结果：已修复（fixed）" in report
    assert "上下文峰值估算：4200 tokens" in report
    assert "生效 Snapshot 版本：无" in report
    assert "旧模型声称 tests/ 被修改" not in report
    assert "旧模型声称根因未知" not in report


def test_report_view_is_ephemeral_and_not_persisted(tmp_path: Path) -> None:
    repositories = DomainRepositories.create(tmp_path / "deepfix.db")
    _seed_task(repositories)

    first = build_task_report_view(repositories, TASK_ID)
    second = build_task_report_view(repositories, TASK_ID)
    with repositories.database.connection() as connection:
        table_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert second == first
    assert "task_reports" not in table_names
    assert "task_report_views" not in table_names


def test_report_labels_pytest_usage_error_as_infrastructure_error() -> None:
    usage_error = SystemTestEvidence(
        evidence_id="pytest-usage-error",
        command="python -m pytest -q --bad-option",
        exit_code=4,
        summary="pytest: error: unrecognized arguments: --bad-option",
        tool_call_id="usage-call",
        source_message_id="usage-result",
        origin="repository_existing",
        scope="full_suite",
        timing="post_change",
        workspace_baseline_id="baseline-1",
        code_state_hash="code-state-1",
    )

    lines = _repository_test_lines([usage_error])

    assert lines[0].startswith("- [基础设施错误]")


def test_report_shows_waiting_input_as_handoff_and_continuous_run_history(tmp_path: Path) -> None:
    repositories = DomainRepositories.create(tmp_path / "deepfix.db")
    _seed_task(repositories, completed=False)
    repositories.tasks.record_input(TaskInput(
        input_id="input-1", task_id=TASK_ID, text="Python 3.11 才失败", kind="constraint",
        created_at="2026-08-31T01:01:00+00:00",
    ))
    repositories.tasks.start_run(TaskRun(
        run_id="run-2", task_id=TASK_ID, input_id="input-1", invocations=0,
        created_at="2026-08-31T01:01:00+00:00",
    ))
    lifecycle = repositories.tasks.get_lifecycle(TASK_ID)
    repositories.tasks.transition_lifecycle(
        TASK_ID, TaskLifecycleStatus.WAITING_INPUT, reason="请提供复现命令", expected_version=lifecycle.version,
    )

    view = build_task_report_view(repositories, TASK_ID)
    assert view.lifecycle.reason == "请提供复现命令"
    report = render_report(view)

    assert "交接状态：等待用户补充（handoff）" in report
    assert "等待输入问题：请提供复现命令" in report
    assert "最新运行：run-2（调用 0 次）" in report
    assert "累计 Agent 调用：0" in report
    assert "[constraint] Python 3.11 才失败" in report


def _seed_task(repositories: DomainRepositories, *, completed: bool = True) -> None:
    repositories.tasks.create_definition(
        TaskDefinition(
            task_id=TASK_ID,
            original_message_id="message-user-1",
            original_problem="修复 value.py 中的边界错误",
            approval_mode="manual",
            source_project_root="C:/source",
            workspace_root="C:/workspace",
            workspace_baseline_id="baseline-1",
            project_python="C:/Python/python.exe",
            confinement_level="guarded_local",
            created_at="2026-08-31T00:00:00+00:00",
        )
    )
    repositories.tasks.transition_lifecycle(
        TASK_ID,
        TaskLifecycleStatus.RUNNING,
        expected_version=1,
    )
    repositories.tasks.save_verification_policy(
        VerificationPolicy(
            policy_id="policy-1",
            task_id=TASK_ID,
            version=1,
            required_oracles=[
                VerificationOracle(
                    oracle_id="oracle-target",
                    origin="user_specified",
                    command=COMMAND,
                    scope="targeted",
                    role="required",
                    required_timing="post_change",
                    relevant_paths=["tests/test_value.py"],
                )
            ],
            supplemental_oracles=[],
            conflict_rules=[],
        )
    )
    change = FileChangeEvidence(
        evidence_id="change-1",
        path="value.py",
        operation="edit",
        status="succeeded",
        tool_call_id="call-edit",
        source_message_id="message-edit",
    )
    test = SystemTestEvidence(
        evidence_id="test-pass",
        command=COMMAND,
        exit_code=0,
        summary="1 passed",
        tool_call_id="call-test",
        source_message_id="message-test",
        origin="user_specified",
        scope="targeted",
        timing="post_change",
        workspace_baseline_id="baseline-1",
        code_state_hash="code-state-2",
    )
    repositories.evidence.record_deterministic(
        TASK_ID,
        change,
        provenance_root_ids=["call-edit"],
    )
    repositories.evidence.record_deterministic(
        TASK_ID,
        test,
        provenance_root_ids=["call-test"],
    )
    repositories.investigation.record_hypothesis(
        TASK_ID,
        InvestigationHypothesis(
            hypothesis_id="hypothesis-root-cause",
            statement="当前根因：边界条件使用了错误比较符",
            state="supported",
            evidence_ids=["test-pass"],
            checked_locations=[],
            reason="目标测试与代码检查一致",
        ),
    )
    repositories.execution.record_approval(
        create_execution_approval(
            task_id=TASK_ID,
            operation="edit_file",
            decision="approve",
            risk="L1",
            source_tool_call_id="call-edit",
            created_at="2026-08-31T00:30:00+00:00",
        )
    )
    repositories.history.record_budget_observation(
        TASK_ID,
        event_id="budget-1",
        estimated_tokens=4_200,
        usage_ratio=0.42,
        zone="normal",
    )
    if completed:
        repositories.tasks.transition_lifecycle(
            TASK_ID,
            TaskLifecycleStatus.COMPLETED,
            expected_version=2,
        )
        repositories.tasks.record_adjudication(
            AdjudicationDecision(
                decision_id="adjudication-1",
                task_id=TASK_ID,
                outcome="fixed",
                evidence_ids=["change-1", "test-pass"],
                operation_ids=[],
                decided_at="2026-08-31T01:00:00+00:00",
            )
        )


def _insert_stale_legacy_projection(repositories: DomainRepositories) -> None:
    payload = (
        '{"diagnosis":"旧模型声称根因未知",'
        '"changed_files":["tests/test_value.py"],'
        '"final_summary":"旧模型声称 tests/ 被修改"}'
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with repositories.database.unit_of_work(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO legacy_task_projection(
                task_id, legacy_phase_status, legacy_paused_from,
                payload, payload_hash, updated_at
            ) VALUES (?, ?, NULL, ?, ?, ?)
            """,
            (
                TASK_ID,
                "completed",
                payload,
                digest,
                "2026-08-31T02:00:00+00:00",
            ),
        )
