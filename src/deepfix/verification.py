from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from deepfix.compaction.models import StrictModel, SystemTestEvidence
from deepfix.workspace import TaskWorkspace

if TYPE_CHECKING:
    from deepfix.task_domain.repository import TaskRepository

_PYTEST_COMMAND = re.compile(
    r"(?i)(?:python(?:\.exe)?\s+-m\s+pytest|pytest)(?:\s+[^\r\n，。；;]+)?"
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


class VerificationPolicyStore:
    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        tasks: TaskRepository | None = None,
    ) -> None:
        from deepfix.task_domain.repository import TaskRepository

        if tasks is None and database_path is None:
            raise ValueError("database_path or tasks is required")
        self.tasks = tasks or TaskRepository(database_path)

    def save(self, policy: VerificationPolicy) -> None:
        self.tasks.save_verification_policy(policy)

    def load(self, task_id: str, version: int | None = None) -> VerificationPolicy | None:
        return self.tasks.load_verification_policy(task_id, version)


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
        if not matches:
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
        and item.exit_code != 0
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
        token.split("::", 1)[0].replace("\\", "/").lstrip("./")
        for token in _tokens_after_pytest(command)
        if not token.startswith("-") and _looks_like_test_target(token)
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


def _looks_like_test_target(token: str) -> bool:
    value = token.split("::", 1)[0]
    return "/" in value or value.endswith(".py") or value.startswith("test")


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
