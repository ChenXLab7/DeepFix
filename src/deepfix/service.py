from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from deepfix.approval import ApprovalPolicy, PolicyAction
from deepfix.compaction.identity import (
    ensure_message_ids,
    stable_conversation_message_id,
)
from deepfix.config import AppConfig
from deepfix.memory import WorkingMemoryStore
from deepfix.models import ApprovalRecord, RepairOutcome, TaskState, TaskStatus, TestResult
from deepfix.persistence import TaskRepository
from deepfix.research.store import ResearchEvidenceStore


class BugfixService:
    def __init__(
        self,
        agent,
        repository: TaskRepository,
        policy: ApprovalPolicy,
        config: AppConfig,
        working_memory_store: WorkingMemoryStore,
        research_evidence_store: ResearchEvidenceStore,
    ) -> None:
        self.agent = agent
        self.repository = repository
        self.policy = policy
        self.config = config
        self.working_memory_store = working_memory_store
        self.research_evidence_store = research_evidence_store

    def start(self, problem: str) -> TaskState:
        task = TaskState.create(
            self.config.project_root,
            problem,
            self.config.approval_mode,
            self.config.project_python,
        )
        message_id = stable_conversation_message_id(
            task.task_id,
            len(task.conversation),
            "user",
            task.user_problem,
        )
        message = {"id": message_id, "role": "user", "content": task.user_problem}
        task.conversation.append(message)
        task.transition_to(TaskStatus.INVESTIGATING)
        self._save(task)
        return self._invoke(
            task,
            {"messages": [HumanMessage(id=message_id, content=task.user_problem)]},
        )

    def continue_task(
        self,
        task_id: str,
        user_message: str | None = None,
    ) -> TaskState:
        task = self.repository.get(task_id)
        if task.status is TaskStatus.PAUSED:
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
        return self._invoke(
            task,
            {"messages": [HumanMessage(id=message_id, content=content)]},
        )

    def pending_actions(self, task_id: str) -> list[dict[str, object]]:
        return deepcopy(self.repository.get(task_id).pending_actions)

    def pause_task(self, task_id: str, reason: str = "用户暂停任务") -> TaskState:
        task = self.repository.get(task_id)
        if task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            raise ValueError("终态任务不能暂停")
        return self._pause(task, reason)

    def decide(self, task_id: str, decisions: list[str]) -> TaskState:
        task = self.repository.get(task_id)
        if task.status is not TaskStatus.WAITING_APPROVAL:
            raise ValueError("任务当前不在等待审批状态")
        return self._resume_actions(task, decisions)

    def _invoke(self, task: TaskState, value: object) -> TaskState:
        if task.agent_invocations >= self.config.max_agent_invocations:
            return self._pause(task, "已达到 Agent 最大调用次数")

        task.agent_invocations += 1
        self._save(task)
        graph_config = {"configurable": {"thread_id": task.task_id}}
        try:
            result = self.agent.invoke(value, graph_config)
        except ContextOverflowError as exc:
            self.working_memory_store.record_overflow(task.task_id)
            return self._pause(
                task,
                f"上下文压缩后仍超出模型限制，任务已暂停: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - persist every Agent boundary failure
            task.final_summary = f"Agent 执行失败: {exc}"
            task.transition_to(TaskStatus.FAILED)
            self._save(task)
            return task

        self._record_tool_results(task, result)
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

    def _handle_interrupts(self, task: TaskState, interrupts: object) -> TaskState:
        task.pending_actions = self._normalize_actions(interrupts)
        task.shell_calls += sum(
            action["name"] == "execute" for action in task.pending_actions
        )
        if task.shell_calls > self.config.max_shell_calls:
            return self._pause(task, "已达到 Shell 最大执行次数")

        task.transition_to(TaskStatus.WAITING_APPROVAL)
        self._save(task)
        if all(
            action["policy_action"] == PolicyAction.ALLOW.value
            for action in task.pending_actions
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
                if name == "execute" and self._is_pytest(str(args.get("command", ""))):
                    approved_test = True
            else:
                graph_decisions.append(
                    {"type": "reject", "message": rejection_reason}
                )
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
            if name == "compact_conversation":
                if message.text.startswith("Conversation compacted."):
                    self.working_memory_store.record_compaction(task.task_id)
                continue
            command = str(args.get("command", ""))
            if name != "execute" or not self._is_pytest(command):
                continue
            artifact = message.artifact
            if not isinstance(artifact, Mapping) or "exit_code" not in artifact:
                continue
            exit_code = int(artifact["exit_code"])
            task.test_results.append(
                TestResult(
                    command=command,
                    exit_code=exit_code,
                    summary=str(message.content),
                    tool_call_id=call_id,
                    source_message_id=str(message.id),
                )
            )
            if exit_code == 0:
                task.consecutive_test_failures = 0
            else:
                task.consecutive_test_failures += 1

    def _apply_outcome(self, task: TaskState, outcome: RepairOutcome) -> TaskState:
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
        elif outcome.status == "blocked":
            return self._pause(task, outcome.summary)
        elif not any(result.exit_code == 0 for result in task.test_results):
            return self._pause(task, "缺少通过的测试证据，不能标记为完成")
        else:
            self._return_to_investigating(task)
            task.transition_to(TaskStatus.REVIEWING)
            task.transition_to(TaskStatus.COMPLETED)
        self._save(task)
        return task

    def _return_to_investigating(self, task: TaskState) -> None:
        if task.status is not TaskStatus.INVESTIGATING:
            task.transition_to(TaskStatus.INVESTIGATING)

    def _pause(self, task: TaskState, reason: str) -> TaskState:
        task.final_summary = reason
        if task.status is not TaskStatus.PAUSED:
            task.transition_to(TaskStatus.PAUSED)
        self._save(task)
        return task

    def _save(self, task: TaskState) -> None:
        self._sync_context(task)
        self.repository.save(task)

    def _sync_context(self, task: TaskState) -> None:
        latest = self.working_memory_store.latest(task.task_id)
        if latest is not None:
            task.working_memory_version = latest.version
            evidence_keys = {
                (item.source, item.observation)
                for item in task.evidence
            }
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

        task.context_metrics = self.working_memory_store.metrics(task.task_id)
        external_evidence = self.research_evidence_store.list_evidence(task.task_id)
        task.external_evidence_ids = [item.evidence_id for item in external_evidence]
        query_count, provider_errors = self.research_evidence_store.query_summary(
            task.task_id
        )
        task.research_query_count = query_count
        task.research_provider_errors = [
            _bounded_provider_error(error)
            for error in provider_errors[:10]
        ]
        if self.config.artifacts_path.exists():
            task.offloaded_artifacts = sorted(
                path.relative_to(self.config.artifacts_path).as_posix()
                for path in self.config.artifacts_path.rglob("*")
                if path.is_file()
            )

    @staticmethod
    def _is_pytest(command: str) -> bool:
        normalized = command.strip().lower()
        return normalized == "pytest" or normalized.startswith(
            ("pytest ", "python -m pytest")
        )


def _bounded_provider_error(value: str, limit: int = 300) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"
