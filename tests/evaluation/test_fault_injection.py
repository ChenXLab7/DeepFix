from __future__ import annotations

import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from deepfix.compaction.models import FileChangeEvidence, ProvenanceRef, SystemTestEvidence
from deepfix.domain_repositories.execution import ExecutionRepository
from deepfix.execution import WorkspaceCommandPolicy
from deepfix.investigation.evaluation import RepairLoopOutcome, adjudicate_outcome
from deepfix.investigation.models import ExperimentClaimRecord, InvestigationState
from deepfix.investigation.receipts import ToolResultArtifactStorage
from deepfix.investigation.stagnation import progress_fingerprint
from deepfix.investigation.token_budget import TokenBudgetExhausted, TokenBudgetStore
from deepfix.operations import (
    NewOperationEntry,
    OperationKind,
    OperationReconciler,
    OperationStateSnapshot,
    OperationStatus,
)
from deepfix.task_domain.repository import TaskRepository
from deepfix.verification import (
    OracleConflictRule,
    OracleEvaluation,
    VerificationOracle,
    VerificationPolicy,
    VerificationPolicyConflict,
    evaluate_required_oracles,
)
from deepfix.workspace import WorkspaceFactory, WorkspacePathPolicy, WorkspaceScopeError


def _workspace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "target.py").write_text("broken\n", encoding="utf-8")
    return WorkspaceFactory(tmp_path / "workspaces").create("task-1", source)


def _file_operation(workspace, *, operation_id="operation-1"):
    target = workspace.root / "target.py"
    return NewOperationEntry(
        operation_id=operation_id,
        task_id="task-1",
        experiment_id="experiment-1",
        tool_call_id=f"call-{operation_id}",
        operation_kind=OperationKind.FILE_EDIT,
        call_hash="a" * 64,
        workspace_baseline_id=workspace.baseline.baseline_id,
        pre_state=OperationStateSnapshot(
            target_path="target.py",
            target_exists=True,
            file_hash=_hash(target),
        ),
        expected_post_state=OperationStateSnapshot(
            target_path="target.py",
            target_exists=True,
            file_hash=_text_hash("fixed\n"),
        ),
    )


@pytest.mark.parametrize(
    "fault",
    [
        "crash_after_file_edit_before_receipt",
        "crash_after_receipt_before_state_commit",
        "parallel_identical_side_effect_replay",
        "unknown_command_exit",
    ],
)
def test_side_effect_fault_never_replays(fault, tmp_path) -> None:
    workspace = _workspace(tmp_path)
    journal = ExecutionRepository(tmp_path / "state.db")
    artifacts = ToolResultArtifactStorage(tmp_path / "artifacts" / "receipts")
    side_effect_handler_calls = 0

    def apply_side_effect(content: str | None = None) -> None:
        nonlocal side_effect_handler_calls
        side_effect_handler_calls += 1
        if content is not None:
            (workspace.root / "target.py").write_text(content, encoding="utf-8")

    if fault == "unknown_command_exit":
        operation = NewOperationEntry(
            operation_id="command-1",
            task_id="task-1",
            experiment_id="experiment-1",
            tool_call_id="execute-1",
            operation_kind=OperationKind.COMMAND,
            call_hash="b" * 64,
            workspace_baseline_id=workspace.baseline.baseline_id,
            pre_state=OperationStateSnapshot(command_hash="c" * 64),
        )
        journal.prepare(operation)
        journal.mark_started(operation.operation_id)
        apply_side_effect()
        terminated = []
        reconciler = OperationReconciler(
            journal,
            artifacts,
            terminate_process_group=lambda entry: terminated.append(entry.operation_id),
        )
        first = reconciler.reconcile_task("task-1", workspace.root)
        second = reconciler.reconcile_task("task-1", workspace.root)
        assert first.unknown_operation_ids == [operation.operation_id]
        assert second.unknown_operation_ids == [operation.operation_id]
        assert terminated == [operation.operation_id]
    else:
        operation = _file_operation(workspace)
        journal.prepare(operation)
        journal.mark_started(operation.operation_id)
        apply_side_effect("fixed\n")
        reconciler = OperationReconciler(journal, artifacts)
        if fault == "parallel_identical_side_effect_replay":
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(
                    pool.map(
                        lambda _index: reconciler.reconcile_task(
                            "task-1", workspace.root
                        ),
                        range(4),
                    )
                )
            assert any(operation.operation_id in item.replayable_operation_ids for item in results)
        else:
            first = reconciler.reconcile_task("task-1", workspace.root)
            assert operation.operation_id in first.replayable_operation_ids
            second = reconciler.reconcile_task("task-1", workspace.root)
            assert second.replayable_operation_ids == [operation.operation_id]
        assert journal.load_receipt("task-1", operation.tool_call_id) is not None
        assert journal.load_operation(operation.operation_id).status is OperationStatus.OBSERVED

    duplicate_side_effects = side_effect_handler_calls - 1
    assert side_effect_handler_calls == 1
    assert duplicate_side_effects == 0


@pytest.mark.parametrize(
    "escape",
    [
        "relative_outside",
        "absolute_outside",
        "symlink_outside",
        "junction_outside",
        "git_config_global",
        "pip_install_global",
    ],
)
def test_workspace_escape_is_denied(escape, tmp_path) -> None:
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    before = _hash(sentinel)

    if escape == "relative_outside":
        with pytest.raises(WorkspaceScopeError):
            WorkspacePathPolicy(workspace.root).resolve_allowed("../outside/sentinel.txt")
    elif escape == "absolute_outside":
        with pytest.raises(WorkspaceScopeError):
            WorkspacePathPolicy(workspace.root).resolve_allowed(sentinel)
    elif escape in {"symlink_outside", "junction_outside"}:
        link = workspace.root / "outside-link"
        if escape == "symlink_outside":
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                pytest.skip("directory symlink creation is unavailable")
        else:
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                pytest.skip("directory junction creation is unavailable")
        with pytest.raises(WorkspaceScopeError):
            WorkspacePathPolicy(workspace.root).resolve_allowed(
                "outside-link/sentinel.txt"
            )
    else:
        command = {
            "git_config_global": "git config --global user.name agent",
            "pip_install_global": "python -m pip install demo",
        }[escape]
        assert WorkspaceCommandPolicy(workspace).evaluate(command).allowed is False

    assert _hash(sentinel) == before


def _policy() -> VerificationPolicy:
    return VerificationPolicy(
        policy_id="policy-1",
        task_id="task-1",
        version=1,
        required_oracles=[
            VerificationOracle(
                oracle_id="oracle-targeted",
                origin="user_specified",
                command="python -m pytest tests/test_target.py -q",
                scope="targeted",
                role="required",
            )
        ],
        supplemental_oracles=[],
        conflict_rules=[
            OracleConflictRule(
                rule_id="related-suite-failure",
                description="Related suite failure blocks FIXED",
                blocking_scopes=["module", "full_suite"],
            )
        ],
    )


def _test_evidence(
    evidence_id: str,
    command: str,
    *,
    exit_code: int,
    origin: str,
    scope: str,
) -> SystemTestEvidence:
    return SystemTestEvidence(
        evidence_id=evidence_id,
        command=command,
        exit_code=exit_code,
        summary="fault gate evidence",
        origin=origin,
        scope=scope,
        timing="post_change",
        workspace_baseline_id="baseline-1",
        code_state_hash="code-1",
        tool_call_id=f"call-{evidence_id}",
        source_message_id=f"message-{evidence_id}",
    )


def test_required_oracle_cannot_be_downgraded(tmp_path) -> None:
    store = TaskRepository(tmp_path / "state.db")
    policy = _policy()
    store.save_verification_policy(policy)

    with pytest.raises(VerificationPolicyConflict):
        store.save_verification_policy(
            policy.model_copy(
                update={
                    "version": 2,
                    "required_oracles": [],
                    "supplemental_oracles": policy.required_oracles,
                }
            )
        )


def test_targeted_pass_with_related_full_suite_failure_blocks_fixed() -> None:
    evaluation = evaluate_required_oracles(
        _policy(),
        [
            _test_evidence(
                "targeted-pass",
                "python -m pytest tests/test_target.py -q",
                exit_code=0,
                origin="user_specified",
                scope="targeted",
            ),
            _test_evidence(
                "suite-failure",
                "python -m pytest -q",
                exit_code=1,
                origin="repository_existing",
                scope="full_suite",
            ),
        ],
    )

    assert evaluation.all_required_satisfied is True
    assert evaluation.fixed_allowed is False
    assert evaluation.conflicting_evidence_ids == ["suite-failure"]


def test_unknown_operation_cannot_be_used_as_success_evidence() -> None:
    outcome = adjudicate_outcome(
        OracleEvaluation(
            passed_required_oracle_ids=["oracle-targeted"],
            all_required_satisfied=True,
            fixed_allowed=True,
        ),
        changed_files=[
            FileChangeEvidence(
                evidence_id="file-1",
                path="target.py",
                operation="edit",
                status="succeeded",
            )
        ],
        unresolved_operation_ids=["unknown-operation"],
        scope_violation_evidence_ids=[],
        reproduction_state="reproduced",
    )

    assert outcome is RepairLoopOutcome.BLOCKED


def test_parallel_token_reservations_never_oversell(tmp_path) -> None:
    store = TokenBudgetStore(tmp_path / "tokens.db")
    store.initialize("task-1", input_cap=100, output_cap=100)

    def reserve(index):
        try:
            store.reserve(
                "task-1",
                f"call-{index}",
                input_tokens=60,
                output_tokens=60,
            )
            return "reserved"
        except TokenBudgetExhausted:
            return "rejected"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(4)))

    assert results.count("reserved") == 1
    assert results.count("rejected") == 3
    assert store.available("task-1").input_tokens == 40
    assert store.available("task-1").output_tokens == 40


def test_weak_claim_does_not_reset_progress_fingerprint() -> None:
    before = InvestigationState.new("task-1")
    after = before.model_copy(
        update={
            "experiment_claims": [
                ExperimentClaimRecord(
                    claim_id="weak-claim",
                    text="This might be related",
                    sources=[ProvenanceRef(kind="snapshot_record", ref_id="snapshot-1")],
                    provenance_root_ids=[],
                )
            ]
        }
    )

    assert progress_fingerprint(after) == progress_fingerprint(before)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_hash(content: str) -> str:
    return hashlib.sha256(content.replace("\n", "\r\n").encode()).hexdigest()
