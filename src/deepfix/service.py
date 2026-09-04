from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from deepfix.approval import ApprovalPolicy, PolicyAction
from deepfix.compaction.errors import ContextCoordinationError
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.identity import (
    ensure_message_ids,
    stable_conversation_message_id,
)
from deepfix.config import AppConfig, redact_config_secrets
from deepfix.debug import append_debug_record
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.execution import create_execution_approval
from deepfix.domain_repositories.migration import (
    DomainAuthorityMigrationError,
    DomainMigrator,
)
from deepfix.investigation.classification import is_pytest_verification
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.errors import InvestigationCoordinationError
from deepfix.investigation.models import InvestigationRecoveryMetadata
from deepfix.operations import OperationReconciler
from deepfix.persistence import TaskRepository
from deepfix.task_domain.adjudication import OutcomeAdjudicationInput, OutcomeAdjudicator
from deepfix.task_domain.models import TaskDefinition, TaskLifecycleStatus
from deepfix.task_domain.outcome import RepairOutcomeCandidate
from deepfix.task_domain.runtime import TaskRuntime
from deepfix.verification import (
    VerificationPolicyBuilder,
    classify_pytest_result,
    unavailable_required_oracle_paths,
)
from deepfix.workspace import WorkspaceFactory


class BugfixService:
    def __init__(
        self,
        agent,
        repository: TaskRepository,
        policy: ApprovalPolicy,
        config: AppConfig,
        investigation: InvestigationCoordinator | None = None,
        operation_reconciler: OperationReconciler | None = None,
        workspace_factory: WorkspaceFactory | None = None,
        execution_backend: object | None = None,
        repositories: DomainRepositories | None = None,
        domain_migrator: DomainMigrator | None = None,
    ) -> None:
        self.agent = agent
        self.repository = repository
        self.policy = policy
        self.config = config
        self.repositories = repositories or DomainRepositories.create(repository.database)
        self.investigation = investigation
        self.operation_reconciler = operation_reconciler
        self.workspace_factory = workspace_factory
        self.execution_backend = execution_backend
        self.domain_migrator = domain_migrator or DomainMigrator(
            self.repositories.database,
            artifact_root=config.artifacts_path,
        )
        self._agent_invocations: dict[str, int] = {}

    def start(self, problem: str) -> TaskRuntime:
        normalized_problem = problem.strip()
        if not normalized_problem:
            raise ValueError("问题描述不能为空")
        task_id = uuid4().hex
        source_root = Path(self.config.project_root).resolve()
        workspace_root = source_root
        workspace_baseline_id = None
        confinement_level = "legacy_local"
        workspace = None
        if self.workspace_factory is not None:
            workspace = self.workspace_factory.create(task_id, source_root)
            workspace_root = workspace.root
            workspace_baseline_id = workspace.baseline.baseline_id
            confinement_level = "guarded_local"
        message_id = stable_conversation_message_id(
            task_id,
            0,
            "user",
            normalized_problem,
        )
        definition = TaskDefinition(
            task_id=task_id,
            original_message_id=message_id,
            original_problem=normalized_problem,
            approval_mode=self.config.approval_mode.value,
            source_project_root=str(source_root),
            workspace_root=str(workspace_root),
            workspace_baseline_id=workspace_baseline_id,
            project_python=str(self.config.project_python),
            confinement_level=confinement_level,
            created_at=datetime.now(UTC).isoformat(),
        )
        verification = None
        if workspace is not None:
            verification = VerificationPolicyBuilder().build(definition, workspace)
        self.repository.create_definition(definition)
        if verification is not None:
            self.repository.save_verification_policy(verification)
        self.repository.transition_lifecycle(
            task_id,
            TaskLifecycleStatus.RUNNING,
            expected_version=1,
        )
        migration_failure = self._guard_task_authority_migration(task_id)
        if migration_failure is not None:
            return migration_failure
        return self._invoke_authorities(
            task_id,
            {"messages": [HumanMessage(id=message_id, content=normalized_problem)]},
        )

    def _runtime_from_authorities(
        self,
        task_id: str,
        *,
        pending_actions: list[dict[str, object]] | None = None,
        fallback_reason: str | None = None,
    ) -> TaskRuntime:
        lifecycle = self.repository.get_lifecycle(task_id)
        decision = self.repository.latest_adjudication(task_id)
        return TaskRuntime(
            task_id=task_id,
            lifecycle=lifecycle.status,
            pending_actions=deepcopy(pending_actions or []),
            latest_decision_id=(decision.decision_id if decision is not None else None),
            pause_reason=lifecycle.reason or fallback_reason,
        )

    def get_runtime(self, task_id: str) -> TaskRuntime:
        migration_failure = self._guard_task_authority_migration(task_id)
        if migration_failure is not None:
            return migration_failure
        lifecycle = self.repository.get_lifecycle(task_id)
        actions = (
            self.pending_actions(task_id)
            if lifecycle.status
            in {TaskLifecycleStatus.WAITING_APPROVAL, TaskLifecycleStatus.PAUSED}
            else []
        )
        return self._runtime_from_authorities(task_id, pending_actions=actions)

    def continue_task(
        self,
        task_id: str,
        user_message: str | None = None,
    ) -> TaskRuntime:
        migration_failure = self._guard_task_authority_migration(task_id)
        if migration_failure is not None:
            return migration_failure
        lifecycle = self.repository.get_lifecycle(task_id)
        if lifecycle.status in {
            TaskLifecycleStatus.COMPLETED,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
        }:
            raise ValueError("终态任务不能继续")
        checkpoint_actions = self.pending_actions(task_id)
        if (
            lifecycle.status is TaskLifecycleStatus.PAUSED
            and lifecycle.paused_from is TaskLifecycleStatus.WAITING_APPROVAL
            and checkpoint_actions
            and not user_message
        ):
            self.repository.transition_lifecycle(
                task_id,
                TaskLifecycleStatus.WAITING_APPROVAL,
                expected_version=lifecycle.version,
            )
            return self._runtime_from_authorities(
                task_id,
                pending_actions=checkpoint_actions,
            )
        if lifecycle.status is TaskLifecycleStatus.WAITING_APPROVAL:
            return self._runtime_from_authorities(
                task_id,
                pending_actions=checkpoint_actions,
            )
        if not user_message or not user_message.strip():
            raise ValueError("继续任务需要用户消息")
        resumed_from_pause = lifecycle.status is TaskLifecycleStatus.PAUSED
        if resumed_from_pause:
            lifecycle = self.repository.transition_lifecycle(
                task_id,
                TaskLifecycleStatus.RUNNING,
                expected_version=lifecycle.version,
            )
        content = user_message.strip()
        message_id = stable_conversation_message_id(
            task_id,
            self._graph_message_count(task_id),
            "user",
            content,
        )
        if resumed_from_pause and not self._record_lifecycle_authority(
            task_id,
            lambda: self.investigation.record_resumed(task_id, message_id),
        ):
            return self.get_runtime(task_id)
        if not self._record_lifecycle_authority(
            task_id,
            lambda: self.investigation.record_user_information(task_id, message_id),
        ):
            return self.get_runtime(task_id)
        return self._invoke_authorities(
            task_id,
            {"messages": [HumanMessage(id=message_id, content=content)]},
        )

    def pending_actions(self, task_id: str) -> list[dict[str, object]]:
        get_state = getattr(self.agent, "get_state", None)
        if not callable(get_state):
            return []
        snapshot = get_state({"configurable": {"thread_id": task_id}})
        interrupts = [
            interrupt
            for graph_task in getattr(snapshot, "tasks", ())
            for interrupt in getattr(graph_task, "interrupts", ())
        ]
        return self._normalize_actions(interrupts)

    def pause_task(self, task_id: str, reason: str = "用户暂停任务") -> TaskRuntime:
        migration_failure = self._guard_task_authority_migration(task_id)
        if migration_failure is not None:
            return migration_failure
        lifecycle = self.repository.get_lifecycle(task_id)
        if lifecycle.status in {
            TaskLifecycleStatus.COMPLETED,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
        }:
            raise ValueError("终态任务不能暂停")
        if lifecycle.status is not TaskLifecycleStatus.PAUSED:
            self.repository.transition_lifecycle(
                task_id,
                TaskLifecycleStatus.PAUSED,
                reason=reason,
                expected_version=lifecycle.version,
            )
        return self.get_runtime(task_id)

    def decide(self, task_id: str, decisions: list[str]) -> TaskRuntime:
        migration_failure = self._guard_task_authority_migration(task_id)
        if migration_failure is not None:
            return migration_failure
        lifecycle = self.repository.get_lifecycle(task_id)
        if lifecycle.status is not TaskLifecycleStatus.WAITING_APPROVAL:
            raise ValueError("任务当前不在等待审批状态")
        return self._resume_actions_authority(task_id, decisions)

    def _invoke_authorities(self, task_id: str, value: object) -> TaskRuntime:
        definition = self.repository.get_definition(task_id)
        activate = getattr(self.execution_backend, "activate_workspace", None)
        if callable(activate):
            try:
                activate(definition)
            except Exception as exc:  # noqa: BLE001 - trusted execution boundary
                safe_error = redact_config_secrets(str(exc), self.config)
                return self._pause_authority(
                    task_id,
                    f"Task Workspace 无法激活：{type(exc).__name__}: {safe_error}",
                )
        if self.operation_reconciler is not None:
            try:
                reconciliation = self.operation_reconciler.reconcile_task(
                    task_id,
                    definition.workspace_root,
                )
            except Exception as exc:  # noqa: BLE001 - recovery boundary
                safe_error = redact_config_secrets(str(exc), self.config)
                return self._pause_authority(
                    task_id,
                    f"副作用恢复检查失败：{type(exc).__name__}: {safe_error}",
                )
            if reconciliation.blocks_agent_invocation:
                unresolved = [
                    *reconciliation.conflict_operation_ids,
                    *reconciliation.unknown_operation_ids,
                ]
                return self._pause_authority(
                    task_id,
                    "副作用操作需要人工恢复：" + ", ".join(dict.fromkeys(unresolved)),
                )
        verification = self.repository.load_verification_policy(task_id)
        if verification is not None:
            unavailable_paths = unavailable_required_oracle_paths(
                verification,
                definition.workspace_root,
            )
            if unavailable_paths:
                return self._pause_authority(
                    task_id,
                    "Required Oracle 路径不可用：" + ", ".join(unavailable_paths),
                )
        invocation_count = self._agent_invocations.get(task_id, 0)
        if invocation_count >= self.config.max_agent_invocations:
            return self._pause_authority(task_id, "已达到 Agent 最大调用次数")
        self._agent_invocations[task_id] = invocation_count + 1
        graph_config = {
            "configurable": {
                "thread_id": task_id,
                "workspace_root": definition.workspace_root,
                "workspace_baseline_id": definition.workspace_baseline_id,
            },
            "recursion_limit": self.config.max_graph_steps,
        }
        try:
            result = self.agent.invoke(value, graph_config)
        except InvestigationCoordinationError as exc:
            if exc.recovery.task_id != task_id:
                return self._fail_authority(task_id, "Agent 返回了其他任务的调查恢复信息")
            recovery = self._sanitize_investigation_recovery(exc.recovery)
            self._record_investigation_recovery(recovery)
            return self._pause_authority(
                task_id,
                f"调查协调需要恢复：{recovery.error_code}",
            )
        except ContextCoordinationError as exc:
            if exc.recovery.task_id != task_id:
                return self._fail_authority(task_id, "Agent 返回了其他任务的上下文恢复信息")
            self._record_context_recovery(exc.recovery)
            return self._pause_authority(
                task_id,
                f"上下文协调需要恢复：{exc.recovery.error_code}",
            )
        except GraphRecursionError:
            return self._pause_authority(
                task_id,
                "已达到单次 Agent 执行步骤上限，可能存在重复工具调用循环",
            )
        except Exception as exc:  # noqa: BLE001 - persist Agent boundary failure
            safe_error = redact_config_secrets(str(exc), self.config)
            if _is_model_provider_error(exc):
                return self._pause_authority(
                    task_id,
                    f"模型服务调用需要恢复：{type(exc).__name__}: {safe_error}",
                )
            return self._fail_authority(task_id, f"Agent 执行失败: {safe_error}")

        messages = ensure_message_ids(task_id, result.get("messages", [])).messages
        EvidenceCollector(self.repositories.evidence).collect(task_id, messages, definition)
        if self._consecutive_post_change_failures(task_id) >= (
            self.config.max_consecutive_test_failures
        ):
            return self._pause_authority(task_id, "已达到最大连续测试失败次数")
        interrupts = result.get("__interrupt__", ())
        if interrupts:
            return self._handle_interrupts_authority(task_id, interrupts)
        response = result.get("structured_response")
        if response is None:
            return self.get_runtime(task_id)
        outcome = (
            response
            if isinstance(response, RepairOutcomeCandidate)
            else RepairOutcomeCandidate.model_validate(response)
        )
        return self._apply_outcome_authority(task_id, outcome)

    def _handle_interrupts_authority(
        self,
        task_id: str,
        interrupts: object,
    ) -> TaskRuntime:
        actions = self._normalize_actions(interrupts)
        prior_shell_calls = sum(
            item.operation == "execute"
            for item in self.repositories.execution.list_approvals(task_id)
        )
        if prior_shell_calls + sum(item["name"] == "execute" for item in actions) > (
            self.config.max_shell_calls
        ):
            return self._pause_authority(task_id, "已达到 Shell 最大执行次数")
        lifecycle = self.repository.get_lifecycle(task_id)
        self.repository.transition_lifecycle(
            task_id,
            TaskLifecycleStatus.WAITING_APPROVAL,
            expected_version=lifecycle.version,
        )
        if actions and all(
            action["policy_action"] == PolicyAction.ALLOW.value for action in actions
        ):
            return self._resume_actions_authority(
                task_id,
                ["approve"] * len(actions),
            )
        return self._runtime_from_authorities(task_id, pending_actions=actions)

    def _resume_actions_authority(
        self,
        task_id: str,
        choices: list[str],
    ) -> TaskRuntime:
        actions = self.pending_actions(task_id)
        if len(choices) != len(actions):
            raise ValueError("审批决定数量与待审批操作数量不一致")
        graph_decisions: list[dict[str, str]] = []
        approved_paths: list[str] = []
        for index, (action, choice) in enumerate(zip(actions, choices, strict=True)):
            name = str(action["name"])
            args = action["args"]
            assert isinstance(args, Mapping)
            policy = self.policy.evaluate(name, args)
            if policy.action is PolicyAction.DENY:
                final_choice = "reject"
                reason = policy.reason
            elif policy.action is PolicyAction.ALLOW:
                final_choice = "approve"
                reason = ""
            else:
                if choice not in {"approve", "reject"}:
                    raise ValueError("审批决定必须是 approve 或 reject")
                final_choice = choice
                reason = "用户拒绝该操作"
            graph_decisions.append(
                {"type": "approve"}
                if final_choice == "approve"
                else {"type": "reject", "message": reason}
            )
            if final_choice == "approve" and name in {
                "write_file",
                "edit_file",
                "delete",
            }:
                path = str(args.get("file_path", args.get("path", "")))
                if path:
                    approved_paths.append(path)
            interrupt_id = str(action.get("interrupt_id", "")).strip()
            action_index = int(action.get("interrupt_action_index", index))
            approval_source = (
                f"interrupt:{interrupt_id}:{action_index}"
                if interrupt_id
                else f"legacy-action:{index}:{name}:{dict(args)}"
            )
            source_id = stable_conversation_message_id(
                task_id,
                index,
                "approval",
                approval_source,
            )
            self.repositories.execution.record_approval(
                create_execution_approval(
                    task_id=task_id,
                    operation=name,
                    decision=final_choice,
                    risk=policy.risk.value,
                    source_tool_call_id=source_id,
                )
            )
        existing_paths = {
            item.path
            for item in self.repositories.evidence.verification_view(task_id).file_change_evidence
            if item.status == "succeeded"
        }
        if len(existing_paths | set(approved_paths)) > self.config.max_changed_files:
            return self._pause_authority(task_id, "批准后将超过最大修改文件数")
        lifecycle = self.repository.get_lifecycle(task_id)
        self.repository.transition_lifecycle(
            task_id,
            TaskLifecycleStatus.RUNNING,
            expected_version=lifecycle.version,
        )
        return self._invoke_authorities(
            task_id,
            Command(resume={"decisions": graph_decisions}),
        )

    def _apply_outcome_authority(
        self,
        task_id: str,
        outcome: RepairOutcomeCandidate,
    ) -> TaskRuntime:
        if outcome.status == "needs_input":
            runtime = self._pause_authority(
                task_id,
                outcome.question or outcome.summary,
            )
            self._record_lifecycle_authority(
                task_id,
                lambda: self.investigation.record_needs_input(
                    task_id,
                    f"agent-invocation-{self._agent_invocations.get(task_id, 0)}",
                ),
            )
            return runtime
        if outcome.status == "blocked":
            return self._pause_authority(task_id, outcome.summary)
        verification = self.repositories.evidence.verification_view(task_id)
        successful_changes = [
            item.evidence_id
            for item in verification.file_change_evidence
            if item.status == "succeeded"
        ]
        if successful_changes or any(
            classify_pytest_result(item.exit_code) == "test_failure"
            for item in verification.test_evidence
        ):
            reproduction_state = "reproduced"
        elif any(item.exit_code == 0 for item in verification.test_evidence):
            reproduction_state = "not_reproduced"
        else:
            reproduction_state = "unknown"
        assessment = OutcomeAdjudicator().decide(
            OutcomeAdjudicationInput(
                definition=self.repository.get_definition(task_id),
                lifecycle=self.repository.get_lifecycle(task_id),
                verification_policy=self.repository.load_verification_policy(task_id),
                verification=verification,
                execution_integrity=self.repositories.execution.integrity_view(task_id),
                successful_change_evidence_ids=successful_changes,
                scope_violation_evidence_ids=[],
                reproduction_state=reproduction_state,
            )
        )
        if assessment.decision is None or assessment.outcome not in {
            "fixed",
            "not_reproduced",
        }:
            reason = (
                "缺少通过的测试证据，不能标记为完成"
                if assessment.reason == "required verification is incomplete"
                else assessment.reason
            )
            return self._pause_authority(task_id, reason)
        lifecycle = self.repository.get_lifecycle(task_id)
        self.repository.transition_lifecycle(
            task_id,
            TaskLifecycleStatus.COMPLETED,
            expected_version=lifecycle.version,
        )
        self.repository.record_adjudication(assessment.decision)
        return self.get_runtime(task_id)

    def _pause_authority(self, task_id: str, reason: str) -> TaskRuntime:
        lifecycle = self.repository.get_lifecycle(task_id)
        if lifecycle.status is not TaskLifecycleStatus.PAUSED:
            self.repository.transition_lifecycle(
                task_id,
                TaskLifecycleStatus.PAUSED,
                reason=reason,
                expected_version=lifecycle.version,
            )
        if self.investigation is not None:
            try:
                self.investigation.record_paused(task_id, reason)
            except InvestigationCoordinationError as exc:
                if exc.recovery.task_id == task_id:
                    self._record_investigation_recovery(
                        self._sanitize_investigation_recovery(exc.recovery)
                    )
        return self.get_runtime(task_id)

    def _fail_authority(self, task_id: str, reason: str) -> TaskRuntime:
        lifecycle = self.repository.get_lifecycle(task_id)
        self.repository.transition_lifecycle(
            task_id,
            TaskLifecycleStatus.FAILED,
            reason=reason,
            expected_version=lifecycle.version,
        )
        return self.get_runtime(task_id)

    def _record_lifecycle_authority(self, task_id: str, operation) -> bool:
        if self.investigation is None:
            return True
        try:
            operation()
            return True
        except InvestigationCoordinationError as exc:
            if exc.recovery.task_id != task_id:
                self._fail_authority(task_id, "调查生命周期返回了其他任务的恢复信息")
                return False
            recovery = self._sanitize_investigation_recovery(exc.recovery)
            self._record_investigation_recovery(recovery)
            self._pause_authority(
                task_id,
                f"调查生命周期需要恢复：{recovery.error_code}",
            )
            return False

    def _graph_message_count(self, task_id: str) -> int:
        get_state = getattr(self.agent, "get_state", None)
        if not callable(get_state):
            return 0
        snapshot = get_state({"configurable": {"thread_id": task_id}})
        values = getattr(snapshot, "values", {})
        if isinstance(values, Mapping):
            messages = values.get("messages", [])
            if isinstance(messages, list):
                return len(messages)
        return 0

    def _consecutive_post_change_failures(self, task_id: str) -> int:
        count = 0
        for item in reversed(self.repositories.evidence.verification_view(task_id).test_evidence):
            if item.timing not in {"post_change", "post_recovery"}:
                continue
            classification = classify_pytest_result(item.exit_code)
            if classification == "passed":
                break
            if classification == "test_failure":
                count += 1
        return count

    def _normalize_actions(self, interrupts: object) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for interrupt in interrupts:
            interrupt_id = str(getattr(interrupt, "id", "")).strip()
            value = getattr(interrupt, "value", interrupt)
            if not isinstance(value, Mapping):
                continue
            for action_index, request in enumerate(value.get("action_requests", [])):
                if not isinstance(request, Mapping):
                    continue
                name = str(request.get("name", ""))
                raw_args = request.get("args", {})
                args = dict(raw_args) if isinstance(raw_args, Mapping) else {}
                decision = self.policy.evaluate(name, args)
                normalized.append(
                    {
                        "name": name,
                        "args": args,
                        "description": str(request.get("description", "")),
                        "risk": decision.risk.value,
                        "policy_action": decision.action.value,
                        "reason": decision.reason,
                        "interrupt_id": interrupt_id,
                        "interrupt_action_index": action_index,
                    }
                )
        return normalized

    def _migrate_task_authorities(self, task_id: str) -> None:
        migrations = (
            ("evidence", self.domain_migrator.migrate_deterministic_evidence),
            ("research", self.domain_migrator.migrate_research),
            ("investigation", self.domain_migrator.migrate_investigation),
            ("execution", self.domain_migrator.migrate_execution),
            ("history", self.domain_migrator.migrate_history),
            ("working_memory", self.domain_migrator.migrate_working_memory),
        )
        for domain, migrate in migrations:
            try:
                report = migrate(task_id)
                if not report.ready_to_switch:
                    error = DomainAuthorityMigrationError(
                        task_id=task_id,
                        domain=domain,
                        error_code="domain_authority_migration_not_ready",
                    )
                    self._record_domain_migration_failure(task_id, domain, error)
                    raise error
            except DomainAuthorityMigrationError:
                raise
            except Exception as exc:
                self._record_domain_migration_failure(task_id, domain, exc)
                raise DomainAuthorityMigrationError(
                    task_id=task_id,
                    domain=domain,
                    error_code="domain_authority_migration_failed",
                    cause=exc,
                ) from exc

    def _guard_task_authority_migration(self, task_id: str) -> TaskRuntime | None:
        try:
            self._migrate_task_authorities(task_id)
        except DomainAuthorityMigrationError as error:
            lifecycle = self.repository.get_lifecycle(task_id)
            reason = f"领域权威迁移需要恢复：{error.error_code}:{error.domain}"
            if lifecycle.status not in {
                TaskLifecycleStatus.PAUSED,
                TaskLifecycleStatus.COMPLETED,
                TaskLifecycleStatus.FAILED,
                TaskLifecycleStatus.CANCELLED,
            }:
                lifecycle = self.repository.transition_lifecycle(
                    task_id,
                    TaskLifecycleStatus.PAUSED,
                    reason=reason,
                    expected_version=lifecycle.version,
                )
            return TaskRuntime(
                task_id=task_id,
                lifecycle=lifecycle.status,
                pending_actions=[],
                latest_decision_id=None,
                pause_reason=reason,
            )
        return None

    def _record_domain_migration_failure(
        self,
        task_id: str,
        domain: str,
        error: Exception,
    ) -> None:
        try:
            append_debug_record(
                self.config.artifacts_path / "debug" / "llm_calls.jsonl",
                {
                    "event": "domain_migration_failed",
                    "task_id": task_id,
                    "domain": domain,
                    "error_type": type(error).__name__,
                    "error_detail": redact_config_secrets(str(error), self.config),
                },
            )
        except OSError:
            return

    def _sanitize_investigation_recovery(
        self,
        recovery: InvestigationRecoveryMetadata,
    ) -> InvestigationRecoveryMetadata:
        detail = recovery.error_detail
        if detail is not None:
            detail = redact_config_secrets(detail, self.config)
        return recovery.model_copy(update={"error_detail": detail})

    def _record_investigation_recovery(
        self,
        recovery: InvestigationRecoveryMetadata,
    ) -> None:
        record = {
            "event": "investigation_recovery",
            "task_id": recovery.task_id,
            "error_code": recovery.error_code,
            "error_type": recovery.error_type,
            "error_detail": recovery.error_detail,
            "error_fingerprint": recovery.error_fingerprint,
            "recovery_action": recovery.recovery_action,
        }
        try:
            append_debug_record(
                self.config.artifacts_path / "debug" / "llm_calls.jsonl",
                record,
            )
        except OSError:
            return

    def _record_context_recovery(self, recovery) -> None:
        try:
            append_debug_record(
                self.config.artifacts_path / "debug" / "llm_calls.jsonl",
                {
                    "event": "context_recovery",
                    **recovery.model_dump(mode="json"),
                },
            )
        except OSError:
            return

    @staticmethod
    def _is_pytest(command: str, project_python: str) -> bool:
        return is_pytest_verification(command, project_python)


def _is_model_provider_error(error: Exception) -> bool:
    recoverable_bases = {
        ("httpx", "TransportError"),
        ("openai", "APIError"),
    }
    return any(
        (base.__module__, base.__name__) in recoverable_bases
        for base in type(error).__mro__
    )
