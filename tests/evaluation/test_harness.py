import json
import stat
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import SecretStr

from deepfix.config import AppConfig, ApprovalMode, ModelRoleConfig
from deepfix.evaluation.harness import EvaluationHarness
from deepfix.evaluation.legacy import LegacyLoopRunner
from deepfix.evaluation.models import (
    EvaluationBudget,
    EvaluationCase,
    EvaluationRun,
    RunUsage,
)
from deepfix.models import TaskState, TaskStatus

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
    def __init__(self, task: TaskState) -> None:
        self.task = task
        self.problems: list[str] = []

    def start(self, problem: str) -> TaskState:
        self.problems.append(problem)
        return self.task


def _task(
    workspace: Path,
    *,
    status: TaskStatus,
    resolution: str | None = None,
) -> TaskState:
    task = TaskState.create(
        workspace,
        "fixture",
        ApprovalMode.GUARDED,
        sys.executable,
    )
    task.task_id = "task-evaluation"
    task.status = status
    task.resolution = resolution
    task.processed_tool_call_ids = ["tool-1", "tool-2"]
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
        _task(workspace, status=TaskStatus.COMPLETED, resolution="fixed")
    )
    seen_oracles: list[tuple[str, Path, int]] = []

    @contextmanager
    def service_factory(received_config):
        assert received_config is config
        yield service

    def oracle_runner(command: str, root: Path, timeout: int) -> int:
        seen_oracles.append((command, root, timeout))
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
    assert seen_oracles == [("python -m pytest -q", workspace, 600)]
    assert run.conclusion == "fixed"
    assert run.oracle_exit_code == 0
    assert run.usage.input_tokens == 30
    assert run.usage.output_tokens == 5
    assert run.usage.model_calls == 1
    assert run.usage.tool_calls == 2
    assert run.usage.wall_seconds == 2.5


def test_legacy_runner_does_not_approve_manual_actions(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _app_config(workspace, run_dir)
    task = _task(workspace, status=TaskStatus.WAITING_APPROVAL)
    task.pending_actions = [
        {
            "name": "execute",
            "policy_action": "ask",
            "risk": "L2",
            "args": {"command": "custom-command"},
        }
    ]

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
            _task(workspace, status=TaskStatus.COMPLETED, resolution="fixed")
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
