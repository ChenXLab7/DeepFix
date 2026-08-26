from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from deepfix.evaluation.models import (
    EvaluationBudget,
    EvaluationCase,
    EvaluationRun,
)

_IGNORED_RUNTIME_DIRECTORIES = frozenset(
    {
        ".deepfix",
        ".deepfix-artifacts",
        ".git",
        ".pytest_cache",
        "__pycache__",
    }
)
_IGNORED_RUNTIME_FILES = frozenset({".coverage"})
_IGNORED_RUNTIME_SUFFIXES = frozenset({".pyc", ".pyo"})
_GOLD_DIRECTORIES = frozenset(
    {"correct_python_programs", "gold", "answers", "solutions"}
)
_GOLD_FILE_SUFFIXES = frozenset({".diff", ".patch"})


class CaseRunner(Protocol):
    def run(
        self,
        case: EvaluationCase,
        workspace: Path,
        run_dir: Path,
        budget: EvaluationBudget,
    ) -> EvaluationRun: ...


@dataclass(frozen=True)
class EvaluationExecution:
    workspace: Path
    run_dir: Path
    run: EvaluationRun


class EvaluationHarness:
    def __init__(self, runs_root: Path, runner: CaseRunner) -> None:
        self.runs_root = runs_root.expanduser().resolve()
        self.runner = runner

    def run_case(
        self,
        case: EvaluationCase,
        source: Path,
        *,
        run_index: int,
        budget: EvaluationBudget,
    ) -> EvaluationExecution:
        if run_index <= 0:
            raise ValueError("run_index must be positive")

        source_root = source.expanduser().resolve()
        source_directory = (source_root / case.source_subdir).resolve()
        try:
            source_directory.relative_to(source_root)
        except ValueError as error:
            raise ValueError("source_subdir escapes source root") from error
        if not source_directory.is_dir():
            raise ValueError(f"source directory does not exist: {source_directory}")

        allowed_paths = {_normalized_relative(path) for path in case.allowed_paths}
        run_dir = self.runs_root / case.case_id / f"{run_index:03d}"
        if run_dir.exists():
            raise FileExistsError(run_dir)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir()
        workspace = run_dir / "workspace"
        shutil.copytree(
            source_directory,
            workspace,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git"),
        )

        if case.source_variant == "correct_control":
            _prepare_correct_control(workspace, allowed_paths)
        _remove_gold_material(workspace)

        preparation_hashes = _workspace_hashes(workspace)
        (run_dir / "preparation.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "case_id": case.case_id,
                    "workspace_hashes": preparation_hashes,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        run = self.runner.run(case, workspace, run_dir, budget)
        if run.case_id != case.case_id:
            raise ValueError("runner returned a mismatched case_id")
        scope_violations = sorted(
            set(run.scope_violations)
            | _scope_violations(
                preparation_hashes,
                _workspace_hashes(workspace),
                allowed_paths,
            )
        )
        return EvaluationExecution(
            workspace=workspace,
            run_dir=run_dir,
            run=run.model_copy(update={"scope_violations": scope_violations}),
        )


def _normalized_relative(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise ValueError(f"unsafe relative path: {value}")
    return path.as_posix()


def _prepare_correct_control(workspace: Path, allowed_paths: set[str]) -> None:
    if len(allowed_paths) != 1:
        raise ValueError("correct_control requires exactly one allowed path")
    target_relative = next(iter(allowed_paths))
    prefix = "python_programs/"
    if not target_relative.startswith(prefix):
        raise ValueError("correct_control path must be below python_programs")

    answer_relative = target_relative.removeprefix(prefix)
    answer_root = (workspace / "correct_python_programs").resolve()
    answer = (answer_root / answer_relative).resolve()
    target = workspace / target_relative
    try:
        answer.relative_to(answer_root)
        target.resolve().relative_to(workspace.resolve())
    except ValueError as error:
        raise ValueError("correct_control answer escapes workspace") from error
    if not answer.is_file():
        raise ValueError(f"correct_control answer does not exist: {answer_relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(answer, target)


def _remove_gold_material(workspace: Path) -> None:
    for name in _GOLD_DIRECTORIES | {".git"}:
        _remove_path(workspace / name)
    for path in list(workspace.rglob("*")):
        if path.is_file() and path.suffix.lower() in _GOLD_FILE_SUFFIXES:
            path.unlink()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _workspace_hashes(workspace: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if _is_runtime_path(relative) or path.is_dir():
            continue
        normalized = relative.as_posix()
        if path.is_symlink():
            hashes[normalized] = f"symlink:{path.readlink()}"
        elif path.is_file():
            hashes[normalized] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def _is_runtime_path(path: Path) -> bool:
    return (
        any(part in _IGNORED_RUNTIME_DIRECTORIES for part in path.parts)
        or path.name in _IGNORED_RUNTIME_FILES
        or path.suffix.lower() in _IGNORED_RUNTIME_SUFFIXES
    )


def _scope_violations(
    before: dict[str, str],
    after: dict[str, str],
    allowed_paths: set[str],
) -> set[str]:
    changed = {
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    }
    return changed - allowed_paths
