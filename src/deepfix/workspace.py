from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from deepfix.compaction.models import StrictModel

_BASELINE_FILE = ".deepfix-baseline.json"
_IGNORED_DIRECTORIES = frozenset(
    {
        ".deepfix",
        ".deepfix-artifacts",
        ".deepfix-runtime",
        ".git",
        ".pytest_cache",
        ".venv",
        "__pycache__",
    }
)
_IGNORED_FILES = frozenset({_BASELINE_FILE, ".coverage"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class WorkspaceBaselineError(RuntimeError):
    pass


class WorkspaceScopeError(ValueError):
    pass


class WorkspaceBaseline(StrictModel):
    baseline_id: str
    task_id: str
    source_root: str
    workspace_root: str
    git_head: str | None
    source_dirty_fingerprint: str | None
    code_state_hash: str
    managed_file_hashes: dict[str, str]
    python_fingerprint: str


class TaskWorkspace(StrictModel):
    task_id: str
    root: Path
    baseline: WorkspaceBaseline


class WorkspacePathPolicy:
    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    def resolve_allowed(self, path: str | Path) -> Path:
        raw = Path(path).expanduser()
        candidate = raw if raw.is_absolute() else self.root / raw
        canonical = candidate.resolve(strict=False)
        try:
            common = Path(os.path.commonpath((self.root, canonical)))
        except ValueError as error:
            raise WorkspaceScopeError(f"path escapes workspace: {path}") from error
        if os.path.normcase(str(common)) != os.path.normcase(str(self.root)):
            raise WorkspaceScopeError(f"path escapes workspace: {path}")
        relative = canonical.relative_to(self.root)
        if (
            relative.name == _BASELINE_FILE
            or (relative.parts and relative.parts[0] == ".deepfix-runtime")
        ):
            raise WorkspaceScopeError(f"path targets internal workspace state: {path}")
        return canonical


class WorkspaceFactory:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def create(self, task_id: str, source_root: str | Path) -> TaskWorkspace:
        source = Path(source_root).expanduser().resolve(strict=True)
        if not source.is_dir():
            raise NotADirectoryError(source)
        target = self.root / _task_segment(task_id)
        if target.exists():
            raise FileExistsError(target)
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, symlinks=True, ignore=_copy_ignore)
        hashes = _managed_file_hashes(target)
        baseline = WorkspaceBaseline(
            baseline_id=_baseline_id(task_id, source, hashes),
            task_id=task_id,
            source_root=str(source),
            workspace_root=str(target),
            git_head=_git_head(source),
            source_dirty_fingerprint=_git_dirty_fingerprint(source),
            code_state_hash=_hash_manifest(hashes),
            managed_file_hashes=hashes,
            python_fingerprint=_python_fingerprint(),
        )
        _atomic_write_json(
            target / _BASELINE_FILE,
            baseline.model_dump_json(indent=2),
        )
        return TaskWorkspace(task_id=task_id, root=target, baseline=baseline)

    def load(self, task_id: str) -> TaskWorkspace:
        target = (self.root / _task_segment(task_id)).resolve(strict=True)
        try:
            target.relative_to(self.root)
        except ValueError as error:
            raise WorkspaceBaselineError("workspace baseline identity mismatch") from error
        path = target / _BASELINE_FILE
        baseline = WorkspaceBaseline.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        expected_baseline_id = _baseline_id(
            task_id,
            Path(baseline.source_root),
            baseline.managed_file_hashes,
        )
        if (
            baseline.task_id != task_id
            or Path(baseline.workspace_root).resolve() != target
            or baseline.code_state_hash
            != _hash_manifest(baseline.managed_file_hashes)
            or baseline.baseline_id != expected_baseline_id
        ):
            raise WorkspaceBaselineError("workspace baseline identity mismatch")
        return TaskWorkspace(task_id=task_id, root=target, baseline=baseline)


def compute_code_state_hash(workspace: str | Path) -> str:
    return _hash_manifest(_managed_file_hashes(Path(workspace).resolve(strict=True)))


def _task_segment(task_id: str) -> str:
    if _SAFE_TASK_ID.fullmatch(task_id) is None:
        raise ValueError("task_id is not a safe workspace segment")
    return task_id


def _managed_file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if _is_ignored(relative) or path.is_dir():
            continue
        normalized = relative.as_posix()
        if path.is_symlink():
            hashes[normalized] = f"symlink:{path.readlink()}"
        elif path.is_file():
            hashes[normalized] = _file_sha256(path)
    return hashes


def _baseline_id(
    task_id: str,
    source_root: Path,
    managed_file_hashes: dict[str, str],
) -> str:
    payload = {
        "task_id": task_id,
        "source_root": str(source_root.expanduser().resolve()),
        "managed_file_hashes": dict(sorted(managed_file_hashes.items())),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hash_manifest(managed_file_hashes: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(sorted(managed_file_hashes.items())),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _git_head(source_root: Path) -> str | None:
    return _git_value(source_root, "rev-parse", "HEAD")


def _git_dirty_fingerprint(source_root: Path) -> str | None:
    status = _git_value(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status is None:
        return None
    return hashlib.sha256(status.encode("utf-8")).hexdigest()


def _python_fingerprint() -> str:
    payload = f"{Path(sys.executable).resolve()}\0{sys.version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in _IGNORED_DIRECTORIES
        or name in _IGNORED_FILES
        or Path(name).suffix.lower() in _IGNORED_SUFFIXES
    }


def _atomic_write_json(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _git_value(source_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_ignored(relative: Path) -> bool:
    return (
        any(part in _IGNORED_DIRECTORIES for part in relative.parts)
        or relative.name in _IGNORED_FILES
        or relative.suffix.lower() in _IGNORED_SUFFIXES
    )
