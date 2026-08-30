from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from deepfix.approval import ApprovalPolicy, PolicyAction
from deepfix.compaction.errors import ContextCoordinationError
from deepfix.compaction.evidence import EvidenceCollector
from deepfix.compaction.identity import (
    ensure_message_ids,
    stable_conversation_message_id,
)
from deepfix.compaction.models import FileChangeEvidence, SystemTestEvidence
from deepfix.compaction.store import CompactionStore
from deepfix.config import AppConfig, redact_config_secrets
from deepfix.debug import append_debug_record
from deepfix.domain_repositories import DomainRepositories
from deepfix.domain_repositories.execution import create_execution_approval
from deepfix.domain_repositories.migration import DomainMigrator, domain_is_switched
from deepfix.investigation.classification import is_pytest_verification
from deepfix.investigation.coordinator import InvestigationCoordinator
from deepfix.investigation.errors import InvestigationCoordinationError
from deepfix.investigation.identity import stable_investigation_id
from deepfix.investigation.models import InvestigationRecoveryMetadata
from deepfix.investigation.receipts import receipt_task_segment
from deepfix.memory import WorkingMemoryStore
from deepfix.models import ApprovalRecord, RepairOutcome, TaskState, TaskStatus, TestResult
from deepfix.operations import OperationReconciler
from deepfix.persistence import TaskRepository
from deepfix.research.store import ResearchEvidenceStore
from deepfix.task_domain.migration import task_definition_from_legacy
from deepfix.task_domain.models import AdjudicationDecision, TaskLifecycleStatus
from deepfix.verification import (
    VerificationPolicyBuilder,
    VerificationPolicyStore,
    evaluate_required_oracles,
)
from deepfix.workspace import WorkspaceFactory


class BugfixService:
    def __init__(
        self,
        agent,
        repository: TaskRepository,
        policy: ApprovalPolicy,
        config: AppConfig,
        working_memory_store: WorkingMemoryStore,
        research_evidence_store: ResearchEvidenceStore,
        compaction_store: CompactionStore | None = None,
        investigation: InvestigationCoordinator | None = None,
        operation_reconciler: OperationReconciler | None = None,
        workspace_factory: WorkspaceFactory | None = None,
        verification_policy_store: VerificationPolicyStore | None = None,
        execution_backend: object | None = None,
        repositories: DomainRepositories | None = None,
        domain_migrator: DomainMigrator | None = None,
    ) -> None:
        self.agent = agent
        self.repository = repository
        self.policy = policy
        self.config = config
        self.working_memory_store = working_memory_store
        self.research_evidence_store = research_evidence_store
        self.repositories = repositories or DomainRepositories.create(repository.database)
        self.compaction_store = compaction_store or CompactionStore(
            config.database_path,
            repositories=self.repositories,
        )
        self.investigation = investigation
        self.operation_reconciler = operation_reconciler
        self.workspace_factory = workspace_factory
        self.verification_policy_store = verification_policy_store
        self.execution_backend = execution_backend
        self.domain_migrator = domain_migrator or DomainMigrator(
            self.repositories.database,
            artifact_root=config.artifacts_path,
        )

    def start(self, problem: str) -> TaskState:
        task = TaskState.create(
            self.config.project_root,
            problem,
            self.config.approval_mode,
            self.config.project_python,
        )
        verification = None
        if self.workspace_factory is not None:
            workspace = self.workspace_factory.create(
                task.task_id,
                self.config.project_root,
            )
            task.source_project_root = workspace.baseline.source_root
            task.workspace_root = str(workspace.root)
            task.project_root = str(workspace.root)
            task.workspace_baseline_id = workspace.baseline.baseline_id
            task.confinement_level = "guarded_local"
            if self.verification_policy_store is not None:
                verification = VerificationPolicyBuilder().build(task, workspace)
                task.verification_policy_id = verification.policy_id
                task.verification_policy_version = verification.version
                task.required_oracle_count = len(verification.required_oracles)
        message_id = stable_conversation_message_id(
            task.task_id,
            len(task.conversation),
            "user",
            task.user_problem,
        )
        message = {"id": message_id, "role": "user", "content": task.user_problem}
        task.conversation.append(message)
        task.transition_to(TaskStatus.INVESTIGATING)
        self.repository.create_definition(task_definition_from_legacy(task))
        if verification is not None:
            self.repository.save_verification_policy(verification)
        self.repository.transition_lifecycle(
            task.task_id,
            TaskLifecycleStatus.RUNNING,
            expected_version=1,
        )
        self._migrate_task_authorities(task.task_id)
        self._sync_context(task)
        self.repository.save_legacy_projection(task)
        return self._invoke(
            task,
            {"messages": [HumanMessage(id=message_id, content=task.user_problem)]},
        )

    def continue_task(
        self,
        task_id: str,
        user_message: str | None = None,
    ) -> TaskState:
        task = self._load_current_task(task_id)
        resumed_from_pause = task.status is TaskStatus.PAUSED
        if resumed_from_pause:
            task.resume()
        if task.status is TaskStatus.WAITING_APPROVAL and task.pending_actions:
            self._save(task)
            return task
        if task.status is TaskStatus.CLARIFYING:
            if not user_message or not user_message.strip():
                raise ValueError("恢复澄清任务需要用户补充信息")
            task.pending_question = None
            task.transition_to(TaskStatus.INVESTIGATING)
        if not user_message or not user_message.strip():
            raise ValueError("继续任务需要用户消息")

        content = user_message.strip()
        message_id = stable_conversation_message_id(
            task.task_id,
            len(task.conversation),
            "user",
            content,
        )
        message = {"id": message_id, "role": "user", "content": content}
        task.conversation.append(message)
        self._save(task)
        if resumed_from_pause and not self._record_lifecycle(
            task,
            lambda: self.investigation.record_resumed(task.task_id, message_id),
        ):
            return task
        if not self._record_lifecycle(
            task,
            lambda: self.investigation.record_user_information(task.task_id, message_id),
        ):
            return task
        return self._invoke(
            task,
            {"messages": [HumanMessage(id=message_id, content=content)]},
        )

    def pending_actions(self, task_id: str) -> list[dict[str, object]]:
        return deepcopy(self._load_current_task(task_id).pending_actions)

    def pause_task(self, task_id: str, reason: str = "用户暂停任务") -> TaskState:
        task = self._load_current_task(task_id)
        if task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            raise ValueError("终态任务不能暂停")
        return self._pause(task, reason)

    def decide(self, task_id: str, decisions: list[str]) -> TaskState:
        task = self._load_current_task(task_id)
        if task.status is not TaskStatus.WAITING_APPROVAL:
            raise ValueError("任务当前不在等待审批状态")
        return self._resume_actions(task, decisions)

    def _invoke(self, task: TaskState, value: object) -> TaskState:
        activate = getattr(self.execution_backend, "activate_workspace", None)
        if callable(activate):
            try:
                activate(task)
            except Exception as exc:  # noqa: BLE001 - trusted execution boundary
                safe_error = redact_config_secrets(str(exc), self.config)
                return self._pause(
                    task,
                    f"Task Workspace 无法激活：{type(exc).__name__}: {safe_error}",
                )
        if self.operation_reconciler is not None:
            try:
                reconciliation = self.operation_reconciler.reconcile_task(
                    task.task_id,
                    task.workspace_root or task.project_root,
                )
            except Exception as exc:  # noqa: BLE001 - recovery boundary
                safe_error = redact_config_secrets(str(exc), self.config)
                return self._pause(
                    task,
                    f"副作用恢复检查失败：{type(exc).__name__}: {safe_error}",
                )
            if reconciliation.blocks_agent_invocation:
                unresolved = [
                    *reconciliation.conflict_operation_ids,
                    *reconciliation.unknown_operation_ids,
                ]
                task.unresolved_operation_ids = list(dict.fromkeys(unresolved))
                return self._pause(
                    task,
                    "副作用操作需要人工恢复：" + ", ".join(dict.fromkeys(unresolved)),
                )
            task.unresolved_operation_ids = list(
                dict.fromkeys(
                    [
                        *reconciliation.prepared_operation_ids,
                        *reconciliation.replayable_operation_ids,
                    ]
                )
            )
        if task.agent_invocations >= self.config.max_agent_invocations:
            return self._pause(task, "已达到 Agent 最大调用次数")

        task.agent_invocations += 1
        self._save(task)
        graph_config = {
            "configurable": {
                "thread_id": task.task_id,
                "workspace_root": task.workspace_root,
                "workspace_baseline_id": task.workspace_baseline_id,
            },
            "recursion_limit": self.config.max_graph_steps,
        }
        try:
            result = self.agent.invoke(value, graph_config)
        except InvestigationCoordinationError as exc:
            if exc.recovery.task_id != task.task_id:
                task.final_summary = "Agent 返回了其他任务的调查恢复信息"
                task.transition_to(TaskStatus.FAILED)
                self._save(task)
                return task
            recovery = self._sanitize_investigation_recovery(exc.recovery)
            task.investigation_recovery = recovery
            self._record_investigation_recovery(recovery)
            return self._pause(
                task,
                f"调查协调需要恢复：{recovery.error_code}",
            )
        except ContextCoordinationError as exc:
            if exc.recovery.task_id != task.task_id:
                task.final_summary = "Agent 返回了其他任务的上下文恢复信息"
                task.transition_to(TaskStatus.FAILED)
                self._save(task)
                return task
            task.context_recovery = exc.recovery
            return self._pause(
                task,
                f"上下文协调需要恢复：{exc.recovery.error_code}",
            )
        except GraphRecursionError:
            return self._pause(
                task,
                "已达到单次 Agent 执行步骤上限，可能存在重复工具调用循环",
            )
        except Exception as exc:  # noqa: BLE001 - persist every Agent boundary failure
            safe_error = redact_config_secrets(str(exc), self.config)
            task.final_summary = f"Agent 执行失败: {safe_error}"
            task.transition_to(TaskStatus.FAILED)
            self._save(task)
            return task

        task.context_recovery = None
        task.investigation_recovery = None
        self._record_tool_results(task, result)
        if self.operation_reconciler is not None:
            task.unresolved_operation_ids = [
                item.operation_id
                for item in self.operation_reconciler.journal.list_incomplete(task.task_id)
            ]
        self._refresh_verification(task)
        if task.consecutive_test_failures >= self.config.max_consecutive_test_failures:
            return self._pause(task, "已达到最大连续测试失败次数")

        interrupts = result.get("__interrupt__", ())
        if interrupts:
            return self._handle_interrupts(task, interrupts)

        response = result.get("structured_response")
        if response is None:
            self._save(task)
            return task
        outcome = (
            response
            if isinstance(response, RepairOutcome)
            else RepairOutcome.model_validate(response)
        )
        return self._apply_outcome(task, outcome)

    def _refresh_verification(self, task: TaskState) -> None:
        if not task.verification_policy_id:
            return
        policy = self.repository.load_verification_policy(
            task.task_id,
            task.verification_policy_version,
        )
        if policy is None or policy.policy_id != task.verification_policy_id:
            return
        tests = [
            item
            for item in self.compaction_store.list_evidence(task.task_id)
            if isinstance(item, SystemTestEvidence)
        ]
        evaluation = evaluate_required_oracles(policy, tests)
        task.required_oracle_count = len(policy.required_oracles)
        task.passed_required_oracle_count = len(evaluation.passed_required_oracle_ids)
        task.supplemental_failure_count = len(evaluation.conflicting_evidence_ids)

    def _handle_interrupts(self, task: TaskState, interrupts: object) -> TaskState:
        task.pending_actions = self._normalize_actions(interrupts)
        task.shell_calls += sum(action["name"] == "execute" for action in task.pending_actions)
        if task.shell_calls > self.config.max_shell_calls:
            return self._pause(task, "已达到 Shell 最大执行次数")

        task.transition_to(TaskStatus.WAITING_APPROVAL)
        self._save(task)
        if all(
            action["policy_action"] == PolicyAction.ALLOW.value for action in task.pending_actions
        ):
            return self._resume_actions(
                task,
                ["approve"] * len(task.pending_actions),
            )
        return task

    def _normalize_actions(self, interrupts: object) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for interrupt in interrupts:
            value = getattr(interrupt, "value", interrupt)
            if not isinstance(value, Mapping):
                continue
            for request in value.get("action_requests", []):
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
                    }
                )
        return normalized

    def _resume_actions(self, task: TaskState, choices: list[str]) -> TaskState:
        actions = task.pending_actions
        if len(choices) != len(actions):
            raise ValueError("审批决定数量与待审批操作数量不一致")

        graph_decisions: list[dict[str, str]] = []
        approved_paths: list[str] = []
        approved_write = False
        approved_test = False
        records: list[ApprovalRecord] = []

        for action, choice in zip(actions, choices, strict=True):
            name = str(action["name"])
            args = action["args"]
            assert isinstance(args, Mapping)
            policy_decision = self.policy.evaluate(name, args)
            if policy_decision.action is PolicyAction.DENY:
                final_choice = "reject"
                rejection_reason = policy_decision.reason
            elif policy_decision.action is PolicyAction.ALLOW:
                final_choice = "approve"
                rejection_reason = ""
            else:
                if choice not in {"approve", "reject"}:
                    raise ValueError("审批决定必须是 approve 或 reject")
                final_choice = choice
                rejection_reason = "用户拒绝该操作"

            if final_choice == "approve":
                graph_decisions.append({"type": "approve"})
                if name in {"write_file", "edit_file", "delete"}:
                    approved_write = True
                    path = str(args.get("file_path", args.get("path", "")))
                    if path:
                        approved_paths.append(path)
                if name == "execute" and self._is_pytest(
                    str(args.get("command", "")), task.project_python
                ):
                    approved_test = True
            else:
                graph_decisions.append({"type": "reject", "message": rejection_reason})
            records.append(
                ApprovalRecord(
                    operation=name,
                    decision=final_choice,
                    risk=policy_decision.risk.value,
                )
            )

        new_paths = set(task.changed_files)
        new_paths.update(approved_paths)
        if len(new_paths) > self.config.max_changed_files:
            return self._pause(task, "批准后将超过最大修改文件数")

        task.approvals.extend(records)
        task.changed_files = list(dict.fromkeys([*task.changed_files, *approved_paths]))
        task.pending_actions = []
        if approved_write:
            task.transition_to(TaskStatus.EDITING)
        if approved_test:
            task.transition_to(TaskStatus.TESTING)
        if not approved_write and not approved_test:
            task.transition_to(TaskStatus.INVESTIGATING)
        self._save(task)
        return self._invoke(task, Command(resume={"decisions": graph_decisions}))

    def _record_tool_results(self, task: TaskState, result: Mapping[str, Any]) -> None:
        raw_messages = result.get("messages", [])
        messages = ensure_message_ids(task.task_id, raw_messages).messages
        EvidenceCollector(
            self.compaction_store,
            self.research_evidence_store,
        ).collect(task.task_id, messages, task)
        self._sync_context(task)
        calls: dict[str, tuple[str, Mapping[str, object]]] = {}
        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    calls[str(call["id"])] = (call["name"], call["args"])

        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            call_id = str(message.tool_call_id)
            if call_id in task.processed_tool_call_ids:
                continue
            task.processed_tool_call_ids.append(call_id)
            name, args = calls.get(call_id, (str(message.name or ""), {}))
            command = str(args.get("command", ""))
            if name != "execute" or not self._is_pytest(command, task.project_python):
                continue
            artifact = message.artifact
            if not isinstance(artifact, Mapping) or "exit_code" not in artifact:
                continue
            exit_code = int(artifact["exit_code"])
            if not any(item.tool_call_id == call_id for item in task.test_results):
                task.test_results.append(
                    TestResult(
                        command=command,
                        exit_code=exit_code,
                        summary=str(message.content),
                        tool_call_id=call_id,
                        source_message_id=str(message.id),
                    )
                )
            if task.changed_files:
                if exit_code == 0:
                    task.consecutive_test_failures = 0
                else:
                    task.consecutive_test_failures += 1

    def _apply_outcome(self, task: TaskState, outcome: RepairOutcome) -> TaskState:
        self._sync_context(task)
        task.resolution = None
        task.hypotheses = list(outcome.hypotheses)
        task.diagnosis = outcome.diagnosis
        task.repair_plan = list(outcome.repair_plan)
        task.evidence = list(outcome.evidence)
        task.review = outcome.review_summary
        task.final_summary = outcome.summary
        task.residual_risks = list(outcome.residual_risks)
        task.unverified_items = list(outcome.unverified_items)

        if outcome.status == "needs_input":
            self._return_to_investigating(task)
            task.pending_question = outcome.question
            task.transition_to(TaskStatus.CLARIFYING)
            self._save(task)
            self._record_lifecycle(
                task,
                lambda: self.investigation.record_needs_input(
                    task.task_id,
                    f"agent-invocation-{task.agent_invocations}",
                ),
            )
            return task
        elif outcome.status == "blocked":
            return self._pause(task, outcome.summary)
        elif not any(result.exit_code == 0 for result in task.test_results):
            return self._pause(task, "缺少通过的测试证据，不能标记为完成")
        elif outcome.resolution == "not_reproduced" and any(
            result.exit_code != 0 for result in task.test_results
        ):
            return self._pause(
                task,
                "当前任务存在失败测试证据，不能标记为未复现",
            )
        elif outcome.resolution == "not_reproduced" and task.successful_changed_files:
            return self._pause(
                task,
                "当前任务已经成功修改文件，不能标记为未复现",
            )
        else:
            if task.successful_changed_files:
                if task.verification_policy_id:
                    policy = self.repository.load_verification_policy(
                        task.task_id,
                        task.verification_policy_version,
                    )
                    tests = [
                        item
                        for item in self.compaction_store.list_evidence(task.task_id)
                        if isinstance(item, SystemTestEvidence)
                    ]
                    if policy is None or not evaluate_required_oracles(policy, tests).fixed_allowed:
                        return self._pause(
                            task,
                            "required verification oracle 未全部满足，不能标记为已修复",
                        )
                if not any(result.exit_code != 0 for result in task.test_results):
                    return self._pause(
                        task,
                        "缺少修复前的失败测试证据，不能标记为已修复",
                    )
                task.resolution = "fixed"
            elif any(result.exit_code != 0 for result in task.test_results):
                return self._pause(
                    task,
                    "当前任务存在失败测试证据且没有成功修改，不能标记为完成",
                )
            else:
                task.resolution = "not_reproduced"
                task.final_summary = (
                    "未复现用户描述的问题：当前环境中所运行的 pytest 测试通过，且未修改代码。"
                )
            self._return_to_investigating(task)
            task.transition_to(TaskStatus.REVIEWING)
            task.transition_to(TaskStatus.COMPLETED)
        self._save(task)
        if task.status is TaskStatus.COMPLETED:
            self._record_adjudication(task)
        return task

    def _return_to_investigating(self, task: TaskState) -> None:
        if task.status is not TaskStatus.INVESTIGATING:
            task.transition_to(TaskStatus.INVESTIGATING)

    def _pause(self, task: TaskState, reason: str) -> TaskState:
        task.pause_reason = reason
        task.final_summary = reason
        if task.status is not TaskStatus.PAUSED:
            task.transition_to(TaskStatus.PAUSED)
        self._save(task)
        if self.investigation is not None:
            try:
                self.investigation.record_paused(task.task_id, reason)
            except InvestigationCoordinationError as exc:
                if exc.recovery.task_id == task.task_id:
                    task.investigation_recovery = exc.recovery
                task.final_summary = f"{reason}；调查生命周期记录失败：{exc.recovery.error_code}"
                self._save(task)
        return task

    def _record_lifecycle(self, task: TaskState, operation) -> bool:
        if self.investigation is None:
            return True
        try:
            operation()
            return True
        except InvestigationCoordinationError as exc:
            if exc.recovery.task_id != task.task_id:
                task.final_summary = "调查生命周期返回了其他任务的恢复信息"
                task.transition_to(TaskStatus.FAILED)
                self._save(task)
                return False
            task.investigation_recovery = exc.recovery
            task.final_summary = f"调查生命周期需要恢复：{exc.recovery.error_code}"
            if task.status is not TaskStatus.PAUSED:
                task.transition_to(TaskStatus.PAUSED)
            self._save(task)
            return False

    def _save(self, task: TaskState) -> None:
        self._project_execution_approvals(task)
        self._sync_context(task)
        self.repository.save_legacy_projection(task)

    def _load_current_task(self, task_id: str) -> TaskState:
        self._migrate_task_authorities(task_id)
        task = self.repository.get(task_id)
        self._sync_context(task)
        return task

    def _migrate_task_authorities(self, task_id: str) -> None:
        migrations = (
            ("evidence", self.domain_migrator.migrate_deterministic_evidence),
            ("research", self.domain_migrator.migrate_research),
            ("investigation", self.domain_migrator.migrate_investigation),
            ("execution", self.domain_migrator.migrate_execution),
            ("history", self.domain_migrator.migrate_history),
        )
        for domain, migrate in migrations:
            try:
                migrate(task_id)
            except Exception as exc:  # noqa: BLE001 - legacy remains authoritative
                self._record_domain_migration_failure(task_id, domain, exc)

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

    def _project_execution_approvals(self, task: TaskState) -> None:
        for ordinal, record in enumerate(task.approvals):
            self.repositories.execution.record_approval(
                create_execution_approval(
                    task_id=task.task_id,
                    operation=record.operation,
                    decision=record.decision,
                    risk=record.risk,
                    legacy_ordinal=ordinal,
                    created_at=f"legacy:{ordinal:08d}",
                )
            )

    def _record_adjudication(self, task: TaskState) -> None:
        if task.resolution not in {"fixed", "not_reproduced"}:
            return
        lifecycle = self.repository.get_lifecycle(task.task_id)
        evidence_ids = sorted(
            {item.evidence_id for item in self.compaction_store.list_evidence(task.task_id)}
        )
        decision = AdjudicationDecision(
            decision_id=stable_investigation_id(
                "adjudication",
                task.task_id,
                str(lifecycle.version),
                task.resolution,
            ),
            task_id=task.task_id,
            outcome=task.resolution,
            evidence_ids=evidence_ids,
            operation_ids=[],
            decided_at=lifecycle.updated_at,
        )
        self.repository.record_adjudication(decision)

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

    def _sync_context(self, task: TaskState) -> None:
        latest = self.working_memory_store.latest(task.task_id)
        if latest is not None:
            task.working_memory_version = latest.version
            evidence_keys = {(item.source, item.observation) for item in task.evidence}
            for item in latest.snapshot.evidence:
                key = (item.source, item.observation)
                if key not in evidence_keys:
                    task.evidence.append(item)
                    evidence_keys.add(key)
            task.hypotheses = list(
                dict.fromkeys(
                    [
                        *task.hypotheses,
                        *(item.text for item in latest.snapshot.active_hypotheses),
                    ]
                )
            )

        deterministic_evidence = self.compaction_store.list_evidence(task.task_id)
        successful_changed_files: list[str] = []
        changed_files: list[str] = []
        latest_change_verification = "not_applicable"
        current_tests: list[TestResult] = []
        for item in deterministic_evidence:
            if isinstance(item, FileChangeEvidence):
                changed_files = list(dict.fromkeys([*changed_files, item.path]))
                if item.status == "succeeded":
                    successful_changed_files = list(
                        dict.fromkeys([*successful_changed_files, item.path])
                    )
                    latest_change_verification = "pending"
            elif (
                isinstance(item, SystemTestEvidence)
                and latest_change_verification != "not_applicable"
            ):
                latest_change_verification = "passed" if item.exit_code == 0 else "failed"
            if isinstance(item, SystemTestEvidence):
                current_tests.append(
                    TestResult(
                        command=item.command,
                        exit_code=item.exit_code,
                        summary=item.summary,
                        tool_call_id=item.tool_call_id,
                        source_message_id=item.source_message_id,
                    )
                )
        if domain_is_switched(self.repositories.database, "evidence", task.task_id):
            task.changed_files = changed_files
            task.test_results = current_tests
        else:
            task.changed_files = list(dict.fromkeys([*task.changed_files, *changed_files]))
        task.successful_changed_files = successful_changed_files
        task.latest_change_verification = latest_change_verification

        if domain_is_switched(
            self.repositories.database,
            "investigation",
            task.task_id,
        ):
            task.hypotheses = [
                item.statement
                for item in self.repositories.investigation.list_hypotheses(task.task_id)
                if item.state != "rejected"
            ]

        if domain_is_switched(
            self.repositories.database,
            "execution",
            task.task_id,
        ):
            task.approvals = [
                ApprovalRecord(item.operation, item.decision, item.risk)
                for item in self.repositories.execution.list_approvals(task.task_id)
            ]
            task.unresolved_operation_ids = [
                item.operation_id
                for item in self.repositories.execution.list_incomplete(task.task_id)
            ]

        task.context_metrics = self.working_memory_store.metrics(task.task_id)
        external_evidence = self.research_evidence_store.list_evidence(task.task_id)
        task.external_evidence_ids = [item.evidence_id for item in external_evidence]
        query_count, provider_errors = self.research_evidence_store.query_summary(task.task_id)
        task.research_query_count = query_count
        task.research_provider_errors = [
            _bounded_provider_error(error) for error in provider_errors[:10]
        ]
        if self.config.artifacts_path.exists():
            task.offloaded_artifacts = sorted(
                relative_path
                for path in self.config.artifacts_path.rglob("*")
                if path.is_file()
                and _artifact_belongs_to_task(
                    relative_path := path.relative_to(self.config.artifacts_path).as_posix(),
                    task,
                )
            )

    @staticmethod
    def _is_pytest(command: str, project_python: str) -> bool:
        return is_pytest_verification(command, project_python)


def _bounded_provider_error(value: str, limit: int = 300) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _artifact_belongs_to_task(relative_path: str, task: TaskState) -> bool:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    last_compaction = (task.context_metrics.last_compaction_artifact or "").replace("\\", "/")
    artifact_prefix = ".deepfix-artifacts/"
    last_compaction = last_compaction.removeprefix(artifact_prefix)
    exact_paths = {
        f"conversation_history/{task.task_id}.md",
        last_compaction.lstrip("/"),
    }
    task_prefixes = (
        f"research/{task.task_id}/",
        f"investigation_receipts/{receipt_task_segment(task.task_id)}/",
    )
    return normalized in exact_paths or normalized.startswith(task_prefixes)
