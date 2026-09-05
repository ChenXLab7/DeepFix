from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Literal

from pydantic import Field

from deepfix.compaction.models import StrictModel, SystemTestEvidence
from deepfix.workspace import TaskWorkspace

_PYTEST_COMMAND = re.compile(
    r"(?i)(?:python(?:\.exe)?\s+-m\s+pytest|pytest)(?:\s+[^\r\n，。；;]+)?"
)
_PYTEST_OPTIONS_WITH_VALUE = frozenset(
    {
        "-W",
        "-c",
        "-k",
        "-m",
        "-o",
        "-r",
        "--basetemp",
        "--capture",
        "--confcutdir",
        "--cov",
        "--cov-config",
        "--cov-context",
        "--cov-fail-under",
        "--cov-report",
        "--deselect",
        "--doctest-glob",
        "--doctest-report",
        "--durations",
        "--ignore",
        "--ignore-glob",
        "--import-mode",
        "--junit-prefix",
        "--junitxml",
        "--log-cli-date-format",
        "--log-cli-format",
        "--log-cli-level",
        "--log-file",
        "--log-file-date-format",
        "--log-file-format",
        "--log-file-level",
        "--maxfail",
        "--override-ini",
        "--pdbcls",
        "--rootdir",
        "--tb",
        "--timeout",
        "--timeout-method",
        "--verbosity",
    }
)


class VerificationPolicyConflict(RuntimeError):
    pass


class VerificationOracle(StrictModel):
    oracle_id: str = Field(min_length=1)
    origin: Literal["user_specified", "repository_existing"]
    command: str = Field(min_length=1)
    scope: Literal["targeted", "module", "full_suite"]
    role: Literal["required", "supplemental"]
    expected_exit_code: int = 0
    required_timing: Literal["baseline", "post_change"] = "post_change"
    relevant_paths: list[str] = Field(default_factory=list)


class OracleConflictRule(StrictModel):
    rule_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    blocking_scopes: list[Literal["module", "full_suite"]]


class VerificationPolicy(StrictModel):
    policy_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    required_oracles: list[VerificationOracle]
    supplemental_oracles: list[VerificationOracle]
    conflict_rules: list[OracleConflictRule]


class OracleEvaluation(StrictModel):
    passed_required_oracle_ids: list[str] = Field(default_factory=list)
    failed_required_oracle_ids: list[str] = Field(default_factory=list)
    unavailable_required_oracle_ids: list[str] = Field(default_factory=list)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    all_required_satisfied: bool
    fixed_allowed: bool


def classify_pytest_result(
    exit_code: int,
) -> Literal["passed", "test_failure", "infrastructure_error"]:
    if exit_code == 0:
        return "passed"
    if exit_code == 4:
        return "infrastructure_error"
    return "test_failure"


def unavailable_required_oracle_paths(
    policy: VerificationPolicy,
    workspace_root: str | Path,
) -> list[str]:
    root = Path(workspace_root).resolve()
    unavailable: list[str] = []
    for oracle in policy.required_oracles:
        for relative in oracle.relevant_paths:
            candidate = (root / relative).resolve(strict=False)
            try:
                candidate.relative_to(root)
            except ValueError:
                unavailable.append(relative)
                continue
            if not candidate.exists():
                unavailable.append(relative)
    return list(dict.fromkeys(unavailable))


class VerificationPolicyBuilder:
    def build(
        self,
        task,
        workspace: TaskWorkspace,
    ) -> VerificationPolicy:
        if workspace.task_id != task.task_id:
            raise ValueError("VerificationPolicy task/workspace 不匹配")
        required = [
            _oracle(command, "user_specified", "required")
            for command in extract_user_pytest_commands(
                getattr(task, "original_problem", "")
                or getattr(task, "user_problem", "")
            )
        ]
        supplemental: list[VerificationOracle] = []
        if _has_repository_tests(workspace.baseline.managed_file_hashes):
            repository_role = "supplemental" if required else "required"
            repository = _oracle(
                "python -m pytest -q",
                "repository_existing",
                repository_role,
            )
            if all(
                _normalize_command(item.command) != _normalize_command(repository.command)
                for item in required
            ):
                if repository_role == "required":
                    required.append(repository)
                else:
                    supplemental.append(repository)
        policy_id = stable_verification_id(task.task_id, required, supplemental)
        return VerificationPolicy(
            policy_id=policy_id,
            task_id=task.task_id,
            version=1,
            required_oracles=required,
            supplemental_oracles=supplemental,
            conflict_rules=[
                OracleConflictRule(
                    rule_id="related_repository_failure_blocks_fixed",
                    description=(
                        "同一代码状态下相关 module/full-suite 失败阻止 FIXED"
                    ),
                    blocking_scopes=["module", "full_suite"],
                )
            ],
        )


def evaluate_required_oracles(
    policy: VerificationPolicy,
    evidence: list[SystemTestEvidence],
) -> OracleEvaluation:
    passed: list[str] = []
    failed: list[str] = []
    unavailable: list[str] = []
    accepted_evidence: list[SystemTestEvidence] = []
    for oracle in policy.required_oracles:
        matches = [
            item
            for item in evidence
            if _normalize_command(item.command) == _normalize_command(oracle.command)
            and item.origin == oracle.origin
            and _timing_satisfies(item.timing, oracle.required_timing)
        ]
        if not matches or (
            classify_pytest_result(matches[-1].exit_code) == "infrastructure_error"
        ):
            unavailable.append(oracle.oracle_id)
        elif matches[-1].exit_code == oracle.expected_exit_code:
            passed.append(oracle.oracle_id)
            accepted_evidence.append(matches[-1])
        else:
            failed.append(oracle.oracle_id)
    blocking_scopes = {
        scope for rule in policy.conflict_rules for scope in rule.blocking_scopes
    }
    accepted_hashes = {item.code_state_hash for item in accepted_evidence}
    conflicts = [
        item.evidence_id
        for item in evidence
        if item.timing == "post_change"
        and classify_pytest_result(item.exit_code) == "test_failure"
        and item.scope in blocking_scopes
        and item.origin == "repository_existing"
        and (not accepted_hashes or item.code_state_hash in accepted_hashes)
    ]
    if len(accepted_hashes) > 1:
        conflicts.extend(item.evidence_id for item in accepted_evidence)
    conflicts = list(dict.fromkeys(conflicts))
    satisfied = bool(policy.required_oracles) and not failed and not unavailable
    return OracleEvaluation(
        passed_required_oracle_ids=passed,
        failed_required_oracle_ids=failed,
        unavailable_required_oracle_ids=unavailable,
        conflicting_evidence_ids=conflicts,
        all_required_satisfied=satisfied,
        fixed_allowed=satisfied and not conflicts,
    )


def extract_user_pytest_commands(problem: str) -> list[str]:
    commands: list[str] = []
    for match in _PYTEST_COMMAND.finditer(problem):
        command = _clean_pytest_command(match.group(0))
        if command and command not in commands:
            commands.append(command)
    return commands


def _clean_pytest_command(candidate: str) -> str:
    try:
        tokens = shlex.split(candidate.replace("\\", "/"), posix=True)
    except ValueError:
        return candidate.strip().rstrip(".。")
    retained: list[str] = []
    for token in tokens:
        if any("\u4e00" <= character <= "\u9fff" for character in token):
            break
        retained.append(token)
    return " ".join(retained).strip().rstrip(".。")


def classify_pytest_scope(command: str) -> Literal["targeted", "module", "full_suite"]:
    targets = pytest_target_paths(command)
    if not targets:
        return "full_suite"
    if any("::" in token for token in _tokens_after_pytest(command)):
        return "targeted"
    return "targeted" if len(targets) == 1 else "module"


def pytest_target_paths(command: str) -> list[str]:
    return [
        _normalize_pytest_target(token)
        for token in _pytest_positional_tokens(command)
    ]


def stable_verification_id(
    task_id: str,
    required: list[VerificationOracle],
    supplemental: list[VerificationOracle],
) -> str:
    payload = json.dumps(
        {
            "task_id": task_id,
            "required": [item.model_dump(mode="json") for item in required],
            "supplemental": [item.model_dump(mode="json") for item in supplemental],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "policy_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _oracle(
    command: str,
    origin: Literal["user_specified", "repository_existing"],
    role: Literal["required", "supplemental"],
) -> VerificationOracle:
    normalized = _normalize_command(command)
    return VerificationOracle(
        oracle_id="oracle_"
        + hashlib.sha256(f"{origin}|{normalized}".encode()).hexdigest()[:32],
        origin=origin,
        command=command,
        scope=classify_pytest_scope(command),
        role=role,
        relevant_paths=pytest_target_paths(command),
    )


def _tokens_after_pytest(command: str) -> list[str]:
    try:
        tokens = shlex.split(command.replace("\\", "/"), posix=True)
    except ValueError:
        return []
    lowered = [item.lower() for item in tokens]
    try:
        index = lowered.index("pytest")
    except ValueError:
        return []
    return tokens[index + 1 :]


def _pytest_positional_tokens(command: str) -> list[str]:
    positional: list[str] = []
    skip_value = False
    options_ended = False
    for token in _tokens_after_pytest(command):
        if skip_value:
            skip_value = False
            continue
        if token == "--":
            options_ended = True
            continue
        if not options_ended and token.startswith("-"):
            option = token.split("=", 1)[0]
            # Parsing remains intentionally independent of pytest internals and
            # project plugins. Unknown options are treated as boolean flags;
            # plugin options with path-like values must use --option=value.
            skip_value = "=" not in token and option in _PYTEST_OPTIONS_WITH_VALUE
            continue
        positional.append(token)
    return positional


def _normalize_pytest_target(token: str) -> str:
    value = token.split("::", 1)[0].replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _normalize_command(command: str) -> str:
    return " ".join(command.strip().replace("\\", "/").split()).lower()


def _timing_satisfies(
    actual: Literal["baseline", "post_change", "post_recovery"],
    required: Literal["baseline", "post_change"],
) -> bool:
    if required == "post_change":
        return actual in {"post_change", "post_recovery"}
    return actual == "baseline"


def _has_repository_tests(hashes: dict[str, str]) -> bool:
    return any(
        path.startswith(("tests/", "test/"))
        or Path(path).name.startswith("test_")
        for path in hashes
    )
