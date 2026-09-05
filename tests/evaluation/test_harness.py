import json
import os
import stat
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from pydantic import SecretStr

from deepfix.config import AppConfig, ApprovalMode, ModelRoleConfig
from deepfix.domain_repositories import DomainRepositories
from deepfix.evaluation.harness import EvaluationHarness
from deepfix.evaluation.legacy import LegacyLoopRunner, _run_oracle
from deepfix.evaluation.models import (
    EvaluationBudget,
    EvaluationCase,
    EvaluationRun,
    RunUsage,
)
from deepfix.investigation.receipts import receipt_from_result
from deepfix.investigation.token_budget import (
    TokenBudgetCallbackHandler,
    TokenBudgetMiddleware,
)
from deepfix.task_domain.models import AdjudicationDecision, TaskDefinition, TaskLifecycleStatus
from deepfix.task_domain.runtime import TaskRuntime

BUDGET = EvaluationBudget(
    max_input_tokens=100_000,
    max_output_tokens=20_000,
    max_wall_seconds=600,
    max_tool_calls=40,
    max_side_effects=5,
)


def case(**updates) -> EvaluationCase:
    values = {
        "case_id": "sample-buggy",
        "problem": "repair the sample",
        "allowed_paths": ["value.py"],
        "required_command": "python -m pytest -q",
        "expected_outcome": "fixed",
    }
    values.update(updates)
    return EvaluationCase.model_validate(values)


class RecordingRunner:
    def __init__(self, *, changes: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[EvaluationCase, Path, Path, EvaluationBudget]] = []
        self.changes = changes or {}

    def run(
        self,
        evaluation_case: EvaluationCase,
        workspace: Path,
        run_dir: Path,
        budget: EvaluationBudget,
    ) -> EvaluationRun:
        self.calls.append((evaluation_case, workspace, run_dir, budget))
        for relative_path, content in self.changes.items():
            target = workspace / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return EvaluationRun(
            run_id=run_dir.name,
            case_id=evaluation_case.case_id,
            loop="legacy",
            task_id=f"task-{run_dir.name}",
            conclusion="fixed",
            oracle_exit_code=0,
            scope_violations=[],
            usage=RunUsage(
                input_tokens=10,
                output_tokens=2,
                model_calls=1,
                tool_calls=1,
                wall_seconds=0.5,
            ),
        )


def test_each_run_gets_a_fresh_copy(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    runner = RecordingRunner()
    harness = EvaluationHarness(tmp_path / "runs", runner)

    first = harness.run_case(case(), source, run_index=1, budget=BUDGET)
    (first.workspace / "value.py").write_text("VALUE = 2\n", encoding="utf-8")
    second = harness.run_case(case(), source, run_index=2, budget=BUDGET)

    assert (second.workspace / "value.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert runner.calls[1][3] == BUDGET


def test_existing_run_directory_is_never_reused(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    harness = EvaluationHarness(tmp_path / "runs", RecordingRunner())

    harness.run_case(case(), source, run_index=1, budget=BUDGET)

    with pytest.raises(FileExistsError):
        harness.run_case(case(), source, run_index=1, budget=BUDGET)


def test_git_metadata_is_not_copied_even_when_it_contains_read_only_files(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    git_pack = source / ".git" / "objects" / "pack" / "readonly.idx"
    git_pack.parent.mkdir(parents=True)
    git_pack.write_bytes(b"git metadata")
    git_pack.chmod(stat.S_IREAD)
    harness = EvaluationHarness(tmp_path / "runs", RecordingRunner())

    try:
        execution = harness.run_case(case(), source, run_index=1, budget=BUDGET)
    finally:
        git_pack.chmod(stat.S_IREAD | stat.S_IWRITE)
        copied_pack = (
            tmp_path
            / "runs"
            / "sample-buggy"
            / "001"
            / "workspace"
            / ".git"
            / "objects"
            / "pack"
            / "readonly.idx"
        )
        if copied_pack.exists():
            copied_pack.chmod(stat.S_IREAD | stat.S_IWRITE)

    assert not (execution.workspace / ".git").exists()


def test_correct_control_is_copied_before_gold_material_is_removed(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "python_programs").mkdir(parents=True)
    (source / "correct_python_programs").mkdir()
    (source / "correct_java_programs").mkdir()
    (source / "nested" / "answers").mkdir(parents=True)
    (source / "python_programs" / "sample.py").write_text(
        "VALUE = 'buggy'\n",
        encoding="utf-8",
    )
    (source / "correct_python_programs" / "sample.py").write_text(
        "VALUE = 'correct'\n",
        encoding="utf-8",
    )
    (source / "correct_java_programs" / "SAMPLE.java").write_text(
        "class SAMPLE {}\n",
        encoding="utf-8",
    )
    (source / "answer.patch").write_text("secret patch", encoding="utf-8")
    (source / "nested" / "answers" / "solution.py").write_text(
        "SECRET = True\n",
        encoding="utf-8",
    )
    harness = EvaluationHarness(tmp_path / "runs", RecordingRunner())
    control = case(
        case_id="sample-correct-control",
        allowed_paths=["python_programs/sample.py"],
        expected_outcome="not_reproduced",
        source_variant="correct_control",
    )

    execution = harness.run_case(control, source, run_index=1, budget=BUDGET)

    assert (execution.workspace / "python_programs" / "sample.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 'correct'\n"
    assert not (execution.workspace / "correct_python_programs").exists()
    assert not (execution.workspace / "correct_java_programs").exists()
    assert not (execution.workspace / "nested" / "answers").exists()
    assert not (execution.workspace / "answer.patch").exists()
    assert (source / "correct_python_programs" / "sample.py").is_file()


def test_preparation_hashes_are_outside_workspace_and_scope_changes_are_detected(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "other.py").write_text("OTHER = 1\n", encoding="utf-8")
    runner = RecordingRunner(
        changes={
            "value.py": "VALUE = 2\n",
            "other.py": "OTHER = 2\n",
        }
    )
    harness = EvaluationHarness(tmp_path / "runs", runner)

    execution = harness.run_case(case(), source, run_index=1, budget=BUDGET)

    assert execution.run.scope_violations == ["other.py"]
    assert (execution.run_dir / "preparation.json").is_file()
    assert not (execution.workspace / "preparation.json").exists()


def test_source_subdirectory_cannot_escape_source_root(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    unsafe = case(source_subdir="../outside")
    harness = EvaluationHarness(tmp_path / "runs", RecordingRunner())

    with pytest.raises(ValueError, match="source_subdir"):
        harness.run_case(unsafe, source, run_index=1, budget=BUDGET)


def test_source_symlink_is_rejected_before_workspace_copy(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    link = source / "outside-link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    harness = EvaluationHarness(tmp_path / "runs", RecordingRunner())

    with pytest.raises(ValueError, match="link or junction"):
        harness.run_case(case(), source, run_index=1, budget=BUDGET)

    assert not (tmp_path / "runs" / "sample-buggy" / "001").exists()


def test_source_root_directory_link_is_rejected_before_resolve(tmp_path) -> None:
    actual_source = tmp_path / "actual-source"
    actual_source.mkdir()
    (actual_source / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    source_link = tmp_path / "source-link"
    _make_directory_link(source_link, actual_source)
    harness = EvaluationHarness(tmp_path / "runs", RecordingRunner())

    try:
        with pytest.raises(ValueError, match="link or junction"):
            harness.run_case(case(), source_link, run_index=1, budget=BUDGET)
    finally:
        _remove_directory_link(source_link)

    assert not (tmp_path / "runs" / "sample-buggy" / "001").exists()


def test_source_subdirectory_link_is_rejected_before_resolve(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    actual_subdirectory = source / "actual"
    actual_subdirectory.mkdir()
    (actual_subdirectory / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    linked_subdirectory = source / "linked"
    _make_directory_link(linked_subdirectory, actual_subdirectory)
    harness = EvaluationHarness(tmp_path / "runs", RecordingRunner())

    try:
        with pytest.raises(ValueError, match="link or junction"):
            harness.run_case(
                case(source_subdir="linked"),
                source,
                run_index=1,
                budget=BUDGET,
            )
    finally:
        _remove_directory_link(linked_subdirectory)

    assert not (tmp_path / "runs" / "sample-buggy" / "001").exists()


def _make_directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            pytest.skip("directory symlink creation is unavailable")
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("directory junction creation is unavailable")


def _remove_directory_link(link: Path) -> None:
    if link.exists() or link.is_symlink():
        link.rmdir()


def _app_config(workspace: Path, run_dir: Path) -> AppConfig:
    role = ModelRoleConfig(
        model_name="test-model",
        api_key=SecretStr("test-key"),
        base_url="https://example.invalid",
    )
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        project_root=workspace,
        database_path=run_dir / "deepfix.sqlite3",
        artifacts_path=artifacts,
        main_model=role,
        compaction_model=role,
        approval_mode=ApprovalMode.GUARDED,
        project_python=Path(sys.executable),
    )


class FakeService:
    def __init__(self, task: TaskRuntime) -> None:
        self.task = task
        self.problems: list[str] = []
        workspace = Path(task._workspace)
        self.repositories = DomainRepositories.create(workspace / "evaluation-test.db")
        self.repository = self.repositories.tasks
        self.repository.create_definition(
            TaskDefinition(
                task_id=task.task_id,
                original_message_id="message-evaluation",
                original_problem="fixture",
                approval_mode="guarded",
                source_project_root=str(workspace),
                workspace_root=str(workspace),
                workspace_baseline_id=None,
                project_python=sys.executable,
                confinement_level="legacy_local",
                created_at="2026-08-31T00:00:00+00:00",
            )
        )
        resolution = getattr(task, "_resolution", None)
        if resolution is not None:
            self.repository.record_adjudication(
                AdjudicationDecision(
                    decision_id="decision-evaluation",
                    task_id=task.task_id,
                    outcome=resolution,
                    decided_at="2026-08-31T00:00:00+00:00",
                )
            )
        for index in range(int(getattr(task, "_tool_calls", 0))):
            call_id = f"tool-{index + 1}"
            self.repositories.execution.record_receipt(
                receipt_from_result(
                    task.task_id,
                    {
                        "name": "read_file",
                        "id": call_id,
                        "args": {"file_path": "sample.py"},
                        "type": "tool_call",
                    },
                    ToolMessage(
                        content="ok", name="read_file", tool_call_id=call_id
                    ),
                )
            )

    def start(self, problem: str) -> TaskRuntime:
        self.problems.append(problem)
        return self.task


def _task(
    workspace: Path,
    *,
    status: TaskLifecycleStatus,
    resolution: str | None = None,
) -> TaskRuntime:
    task = TaskRuntime(
        task_id="task-evaluation",
        lifecycle=status,
    )
    object.__setattr__(task, "_workspace", str(workspace))
    object.__setattr__(task, "_resolution", resolution)
    object.__setattr__(task, "_tool_calls", 2)
    return task


def test_legacy_runner_invokes_service_oracle_and_trace_summary(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _app_config(workspace, run_dir)
    trace = config.artifacts_path / "debug" / "llm_calls.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "event": "response",
                "task_id": "task-evaluation",
                "input_tokens": 30,
                "output_tokens": 5,
                "duration_seconds": 1.2,
            }
        ),
        encoding="utf-8",
    )
    service = FakeService(
        _task(workspace, status=TaskLifecycleStatus.COMPLETED, resolution="fixed")
    )
    seen_oracles: list[tuple[str, Path, int, Path]] = []

    @contextmanager
    def service_factory(received_config):
        assert received_config is config
        yield service

    def oracle_runner(
        command: str,
        root: Path,
        timeout: int,
        project_python: Path,
    ) -> int:
        seen_oracles.append((command, root, timeout, project_python))
        return 0

    times = iter([10.0, 12.5])
    runner = LegacyLoopRunner(
        project_python=Path(sys.executable),
        config_factory=lambda *_: config,
        service_factory=service_factory,
        oracle_runner=oracle_runner,
        clock=lambda: next(times),
    )

    run = runner.run(case(), workspace, run_dir, BUDGET)

    assert service.problems == ["repair the sample"]
    assert seen_oracles == [
        ("python -m pytest -q", workspace, 600, Path(sys.executable).resolve())
    ]
    assert run.conclusion == "fixed"
    assert run.oracle_exit_code == 0
    assert run.usage.input_tokens == 30
    assert run.usage.output_tokens == 5
    assert run.usage.model_calls == 1
    assert run.usage.tool_calls == 2
    assert run.usage.wall_seconds == 2.5


def test_default_legacy_runner_receives_runtime_token_budget(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _app_config(workspace, run_dir)
    received: list[TokenBudgetMiddleware] = []
    received_callbacks: list[tuple[TokenBudgetCallbackHandler, ...]] = []

    @contextmanager
    def default_service_factory(
        received_config,
        *,
        evaluation_middleware=None,
        compaction_callbacks=(),
    ):
        assert received_config is config
        received.append(evaluation_middleware)
        received_callbacks.append(compaction_callbacks)
        evaluation_middleware.store.initialize(
            "task-evaluation",
            input_cap=BUDGET.max_input_tokens,
            output_cap=BUDGET.max_output_tokens,
        )
        main = evaluation_middleware.store.reserve(
            "task-evaluation",
            "main-test-call",
            input_tokens=20,
            output_tokens=10,
        )
        evaluation_middleware.store.settle(
            main.reservation_id,
            input_tokens=12,
            output_tokens=4,
        )
        compact = evaluation_middleware.store.reserve(
            "task-evaluation",
            "compaction-test-call",
            input_tokens=8,
            output_tokens=3,
        )
        evaluation_middleware.store.charge_unknown(compact.reservation_id)
        yield FakeService(
            _task(workspace, status=TaskLifecycleStatus.COMPLETED, resolution="fixed")
        )

    monkeypatch.setattr(
        "deepfix.evaluation.legacy._default_service_factory",
        default_service_factory,
    )
    times = iter([1.0, 2.0])
    runner = LegacyLoopRunner(
        project_python=Path(sys.executable),
        config_factory=lambda *_: config,
        oracle_runner=lambda *_: 0,
        clock=lambda: next(times),
    )

    run = runner.run(case(), workspace, run_dir, BUDGET)

    assert len(received) == 1
    assert received[0].input_cap == BUDGET.max_input_tokens
    assert received[0].output_cap == BUDGET.max_output_tokens
    assert len(received_callbacks[0]) == 1
    assert received_callbacks[0][0].role == "compaction"
    assert run.usage.input_tokens == 20
    assert run.usage.output_tokens == 7
    assert run.usage.model_calls == 2
    assert run.usage.usage_estimated is True


def test_default_oracle_binds_python_command_to_selected_interpreter(
    tmp_path,
    monkeypatch,
) -> None:
    selected_python = tmp_path / "Selected Python" / "python.exe"
    selected_python.parent.mkdir()
    selected_python.touch()
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("deepfix.evaluation.legacy.subprocess.run", fake_run)

    exit_code = _run_oracle(
        "python -m pytest tests/test_sample.py -q",
        tmp_path,
        30,
        selected_python,
    )

    assert exit_code == 0
    assert str(captured["command"]).startswith(f'"{selected_python}" -m pytest')


def test_default_oracle_rejects_commands_that_cannot_bind_selected_python(
    tmp_path,
    monkeypatch,
) -> None:
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("deepfix.evaluation.legacy.subprocess.run", fake_run)

    exit_code = _run_oracle(
        "pytest tests/test_sample.py -q",
        tmp_path,
        30,
        Path(sys.executable),
    )

    assert exit_code == 127
    assert called is False


def test_legacy_runner_does_not_approve_manual_actions(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _app_config(workspace, run_dir)
    task = _task(workspace, status=TaskLifecycleStatus.WAITING_APPROVAL)
    task = task.model_copy(update={"pending_actions": [
        {
            "name": "execute",
            "policy_action": "ask",
            "risk": "L2",
            "args": {"command": "custom-command"},
        }
    ]})
    object.__setattr__(task, "_workspace", str(workspace))
    object.__setattr__(task, "_resolution", None)
    object.__setattr__(task, "_tool_calls", 2)

    @contextmanager
    def service_factory(_config):
        yield FakeService(task)

    times = iter([1.0, 2.0])
    runner = LegacyLoopRunner(
        project_python=Path(sys.executable),
        config_factory=lambda *_: config,
        service_factory=service_factory,
        oracle_runner=lambda *_: 1,
        clock=lambda: next(times),
    )

    run = runner.run(case(), workspace, run_dir, BUDGET)

    assert run.conclusion == "blocked"
    assert run.sanitized_error_code == "manual_approval_required"
    assert task.pending_actions[0]["policy_action"] == "ask"


def test_default_legacy_config_is_isolated_below_run_directory(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPFIX_MAIN_API_KEY", "test-main-key")
    monkeypatch.setenv("DEEPFIX_COMPACTION_API_KEY", "test-compaction-key")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    captured: list[AppConfig] = []

    @contextmanager
    def service_factory(config: AppConfig):
        captured.append(config)
        yield FakeService(
            _task(workspace, status=TaskLifecycleStatus.COMPLETED, resolution="fixed")
        )

    times = iter([1.0, 2.0])
    runner = LegacyLoopRunner(
        project_python=Path(sys.executable),
        service_factory=service_factory,
        oracle_runner=lambda *_: 0,
        clock=lambda: next(times),
    )

    runner.run(case(), workspace, run_dir, BUDGET)

    assert captured[0].project_root == workspace.resolve()
    assert captured[0].database_path == (run_dir / "deepfix.sqlite3").resolve()
    assert captured[0].artifacts_path == (run_dir / "artifacts").resolve()
    assert captured[0].approval_mode is ApprovalMode.GUARDED
