from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse, GrepResult

from deepfix.config import AppConfig
from deepfix.investigation.classification import is_pytest_verification
from deepfix.task_domain.models import TaskDefinition
from deepfix.workspace import TaskWorkspace, WorkspaceBaseline

_SAFE_ENVIRONMENT_VARIABLES = (
    "COMSPEC",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "WINDIR",
)
_INTERNAL_GREP_PARTS = frozenset(
    {".deepfix-artifacts", ".deepfix-runtime", ".pytest_cache"}
)


class GuardedLocalShellBackend(LocalShellBackend):
    """Local shell backend with command-class hard timeout ceilings."""

    def __init__(
        self,
        *,
        diagnostic_timeout_seconds: int,
        verification_timeout_seconds: int,
        project_python: str,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
        command_validator: Callable[[str], tuple[bool, str]] | None = None,
        command_transform: Callable[[str], str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            timeout=verification_timeout_seconds,
            max_output_bytes=max_output_bytes,
            env=env,
            **kwargs,
        )
        self._diagnostic_timeout_seconds = diagnostic_timeout_seconds
        self._verification_timeout_seconds = verification_timeout_seconds
        self._project_python = project_python
        self._deepfix_max_output_bytes = max_output_bytes
        self._deepfix_env = env
        self._command_validator = command_validator
        self._command_transform = command_transform

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if self._command_validator is not None:
            allowed, reason = self._command_validator(command)
            if not allowed:
                return ExecuteResponse(
                    output=f"Denied: {reason}",
                    exit_code=126,
                    truncated=False,
                )
        if self._command_transform is not None:
            command = self._command_transform(command)

        is_verification = is_pytest_verification(command, self._project_python)
        command_kind = "verification" if is_verification else "diagnostic"
        hard_limit = (
            self._verification_timeout_seconds
            if is_verification
            else self._diagnostic_timeout_seconds
        )
        effective_timeout = hard_limit if timeout is None else min(timeout, hard_limit)
        process: subprocess.Popen[str] | None = None
        try:
            from deepfix.execution import _command_tokens

            tokens = _command_tokens(command)
            direct_expression = len(tokens) == 3 and tokens[1] == "-c"
            if direct_expression:
                # Do not execute workspace/site startup code before a pure diagnostic.
                tokens[1:1] = ["-I", "-S"]
            process = subprocess.Popen(
                tokens if direct_expression else command,
                shell=not direct_expression,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                env=self._deepfix_env,
                cwd=str(self.cwd),
                **_process_group_options(),
            )
            stdout, stderr = process.communicate(timeout=effective_timeout)
            return _execute_response(
                stdout,
                stderr,
                process.returncode,
                self._deepfix_max_output_bytes,
            )
        except subprocess.TimeoutExpired:
            if process is not None:
                _terminate_process_tree(process)
                _finish_terminated_process(process)
            return ExecuteResponse(
                output=(
                    "[DeepFix execution timeout]\n"
                    "timed_out: true\n"
                    f"command_kind: {command_kind}\n"
                    f"timeout_seconds: {effective_timeout}\n"
                    "exit_code: 124\n"
                    "命令已被强制终止，可能存在死循环、阻塞或等待外部输入。"
                    "请把本次超时作为诊断证据，检查循环边界或缩小实验；不要原样重试。"
                ),
                exit_code=124,
                truncated=False,
            )
        except Exception as error:  # noqa: BLE001
            if process is not None and process.poll() is None:
                _terminate_process_tree(process)
            return ExecuteResponse(
                output=f"Error executing command ({type(error).__name__}): {error}",
                exit_code=1,
                truncated=False,
            )


def _process_group_options() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        _terminate_windows_process_tree(process.pid)
        if process.poll() is None:
            process.kill()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return


def _terminate_windows_process_tree(root_pid: int) -> None:
    import ctypes
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    create_snapshot.restype = wintypes.HANDLE
    process_first = kernel32.Process32FirstW
    process_first.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    process_first.restype = wintypes.BOOL
    process_next = kernel32.Process32NextW
    process_next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    process_next.restype = wintypes.BOOL
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    terminate_process = kernel32.TerminateProcess
    terminate_process.argtypes = [wintypes.HANDLE, wintypes.UINT]
    terminate_process.restype = wintypes.BOOL
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    wait_for_single_object.restype = wintypes.DWORD
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    snapshot = create_snapshot(0x00000002, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        return
    children: dict[int, list[int]] = defaultdict(list)
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(ProcessEntry)
        has_entry = bool(process_first(snapshot, ctypes.byref(entry)))
        while has_entry:
            children[int(entry.th32ParentProcessID)].append(
                int(entry.th32ProcessID)
            )
            has_entry = bool(process_next(snapshot, ctypes.byref(entry)))
    finally:
        close_handle(snapshot)

    descendants: list[int] = []
    pending = list(children.get(root_pid, ()))
    while pending:
        pid = pending.pop()
        descendants.append(pid)
        pending.extend(children.get(pid, ()))

    process_terminate = 0x0001
    synchronize = 0x00100000
    for pid in [*reversed(descendants), root_pid]:
        handle = open_process(process_terminate | synchronize, False, pid)
        if not handle:
            continue
        try:
            terminate_process(handle, 1)
            wait_for_single_object(handle, 1000)
        finally:
            close_handle(handle)


def _finish_terminated_process(process: subprocess.Popen[str]) -> None:
    try:
        process.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            process.kill()
        # communicate() owns reader threads for PIPE streams on Windows.
        # Closing those streams here can block forever on the reader's lock.
        # The daemon readers receive EOF after the terminated process tree
        # releases its inherited handles.


def _execute_response(
    stdout: str,
    stderr: str,
    exit_code: int,
    max_output_bytes: int,
) -> ExecuteResponse:
    output_parts: list[str] = []
    if stdout:
        output_parts.append(stdout)
    if stderr:
        output_parts.extend(
            f"[stderr] {line}" for line in stderr.strip().split("\n")
        )
    output = "\n".join(output_parts) if output_parts else "<no output>"
    truncated = len(output) > max_output_bytes
    if truncated:
        output = (
            output[:max_output_bytes]
            + f"\n\n... Output truncated at {max_output_bytes} bytes."
        )
    if exit_code != 0:
        output = f"{output.rstrip()}\n\nExit code: {exit_code}"
    return ExecuteResponse(
        output=output,
        exit_code=exit_code,
        truncated=truncated,
    )


def _build_project_backend(
    config: AppConfig,
    workspace: TaskWorkspace | None = None,
) -> LocalShellBackend:
    shell_environment = {
        name: value
        for name in _SAFE_ENVIRONMENT_VARIABLES
        if (value := os.environ.get(name)) is not None
    }
    executable_directory = str(Path(sys.executable).parent)
    inherited_path = os.environ.get("PATH", "")
    shell_environment["PATH"] = os.pathsep.join(
        item for item in (executable_directory, inherited_path) if item
    )

    command_validator = None
    command_transform = None
    root = config.project_root
    if workspace is not None:
        from deepfix.execution import (
            WorkspaceCommandPolicy,
            _bind_project_python,
            task_scoped_environment,
        )

        policy = WorkspaceCommandPolicy(workspace, project_python=config.project_python)
        shell_environment = task_scoped_environment(
            workspace,
            config.project_python,
        )
        root = workspace.root
        command_validator = lambda command: (
            (decision := policy.evaluate(command)).allowed,
            decision.reason,
        )
        command_transform = lambda command: _bind_project_python(
            command,
            config.project_python,
        )

    return GuardedLocalShellBackend(
        root_dir=root,
        virtual_mode=True,
        diagnostic_timeout_seconds=config.diagnostic_timeout_seconds,
        verification_timeout_seconds=config.verification_timeout_seconds,
        project_python=str(config.project_python),
        max_output_bytes=100_000,
        env=shell_environment,
        inherit_env=False,
        command_validator=command_validator,
        command_transform=command_transform,
    )


class DeepFixBackend(CompositeBackend):
    def __init__(
        self,
        config: AppConfig,
        workspace: TaskWorkspace | None = None,
    ) -> None:
        self.config = config
        self.active_workspace = workspace
        super().__init__(
            default=_build_project_backend(config, workspace),
            routes={
                "/.deepfix-artifacts/": FilesystemBackend(
                    root_dir=config.artifacts_path,
                    virtual_mode=True,
                )
            },
            artifacts_root="/.deepfix-artifacts",
        )

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        if path is None or path == "/":
            return _filter_project_grep_result(
                self.default.grep(
                    pattern,
                    path="/",
                    glob=glob,
                    max_count=max_count,
                )
            )
        return super().grep(pattern, path=path, glob=glob, max_count=max_count)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        if path is None or path == "/":
            return _filter_project_grep_result(
                await self.default.agrep(
                    pattern,
                    path="/",
                    glob=glob,
                    max_count=max_count,
                )
            )
        return await super().agrep(
            pattern,
            path=path,
            glob=glob,
            max_count=max_count,
        )

    def activate_workspace(self, task: TaskDefinition | TaskWorkspace) -> None:
        workspace = task if isinstance(task, TaskWorkspace) else _task_workspace(task)
        if self.active_workspace is not None and (
            self.active_workspace.task_id != workspace.task_id
        ):
            raise RuntimeError("backend is already bound to another task workspace")
        self.active_workspace = workspace
        self.default = _build_project_backend(self.config, workspace)


def _task_workspace(task: TaskDefinition) -> TaskWorkspace:
    root = Path(task.workspace_root).resolve(strict=True)
    baseline_path = root / ".deepfix-baseline.json"
    baseline = WorkspaceBaseline.model_validate_json(
        baseline_path.read_text(encoding="utf-8")
    )
    if baseline.task_id != task.task_id or baseline.baseline_id != task.workspace_baseline_id:
        raise RuntimeError("task workspace baseline identity mismatch")
    return TaskWorkspace(task_id=task.task_id, root=root, baseline=baseline)


def _filter_project_grep_result(result: GrepResult) -> GrepResult:
    matches = [
        match
        for match in result.matches or []
        if not _INTERNAL_GREP_PARTS.intersection(
            PurePosixPath(str(match["path"]).replace("\\", "/")).parts
        )
    ]
    return GrepResult(
        error=result.error,
        matches=matches,
        truncated=result.truncated,
    )


def build_backend(
    config: AppConfig,
    workspace: TaskWorkspace | None = None,
) -> DeepFixBackend:
    """Build isolated project and internal-artifact filesystem routes."""
    return DeepFixBackend(config, workspace)
