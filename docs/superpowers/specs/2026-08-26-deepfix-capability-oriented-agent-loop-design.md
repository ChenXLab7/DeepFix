# DeepFix 能力导向 Agent Loop 设计

日期：2026-08-26  
状态：根据首轮审阅修订，待复审
适用范围：DeepFix Python Bug 修复 Agent 的调查、修改、验证与结论生成流程

## 1. 背景

DeepFix 已经实现调查阶段、Tool Receipt、确定性证据、Working Memory、上下文压缩、Artifact Retrieval、
停滞检测、审批和类型化恢复。它能够阻止一部分重复读取、重复执行和无限增长的 Agent Loop，但当前核心
仍然是一个以单次 Tool Call 为推进单位的 Deep Agents 循环：模型自由选择下一个工具，Middleware 和
Coordinator 在工具前后进行门禁、分类、阶段迁移和停滞判断。

该结构优先解决“模型不要失控”，但没有充分解决“系统如何帮助模型完成复杂 Bug 调查”。复杂问题可能
合理地需要连续读取多个相关文件、构造最小复现、比较调用链并多次验证。若系统在每个 Tool Call 后用
固定阶段和局部进展规则干预，模型的合理调查路径也可能被提前截断。另一方面，仅加强提示词或增加模型
自由度又会重新引入重复工具调用、错误结论和无边界上下文增长。

本设计采用 **能力导向双层循环**：外层以策略和证据缺口为中心，内层以有目标、有预算、有成功标准的
完整 Experiment 为执行单位。Experiment 不是新的阶段状态机，而是可自适应粒度的执行边界：简单 Bug
可以在一个 Experiment 内完成读取、修改和验证，复杂 Bug 才拆成多个实验。第一阶段仍使用同一个主模型
完成 StrategyPlanner、ExperimentExecutor 和必要的语义评估；Multi-Agent 和复杂模型路由延后。

本设计不是在现有 Reliability Core 旁边增加第二套状态机。它保留已实现的安全与证据基础设施，并取代
`2026-08-24-deepfix-investigation-reliability-core-design.md` 中以单个 Tool Call 为核心的阶段门禁、进展
判定和停滞策略。旧设计中的 Tool Receipt、稳定 ID、确定性证据、审批、类型化恢复、Artifact
Retrieval 和 Progress Events 继续复用。

## 2. 目标

- 将 Agent 的主要循环单位从单个 Tool Call 提升为完整 Experiment。
- 让模型围绕假设、Evidence Gap、预期结果和替代策略进行长程推理。
- 系统只硬性约束安全性、真实性、用户范围和资源边界，不过度干预实验内部的合理工具选择。
- 以权威证据裁决 `FIXED`、`NOT_REPRODUCED` 等结论，避免模型结论与真实执行结果不一致。
- 将停滞定义为策略或实验重复失败，而不是简单的工具次数增长。
- 保留单一调查状态真相来源，避免 Blackboard、Working Memory 和 Snapshot 重复持久化同一事实。
- 继续复用 Deep Agents 的模型、Backend、Tool、Middleware、Artifact 和内部工具循环能力。
- 保留可替换的 ExperimentExecutor 边界，但先用实验验证单 Agent 双层循环，不实现 Subagent。
- 对副作用操作建立独立工作区、Baseline 和 Operation Journal，使恢复依据真实 Workspace 状态。
- 为测试证据记录来源、范围和修改时序，使最终裁决能够区分证据强度。
- 建立可重复的 QuixBugs 评测基线，证明能力提升而不只依赖单次成功演示。

## 3. 非目标

- 第一阶段不实现任何 Subagent 或固定的 Planner、Investigator、Editor、Tester 多 Agent 团队。
- 第一阶段不新增独立 Planner/Evaluator 模型配置、复杂 Model Router 或精细 Token 调度器。
- 不把每个逻辑组件都强制实现为独立 Controller、Manager、Store 或模型调用。
- 不为 QuixBugs 的具体算法硬编码答案、文件路径或修复模板。
- 不让模型生成的 Working Memory、Claim 或最终文本覆盖系统测试、文件修改和审批记录。
- 不重写 Deep Agents Backend、Tool 外观、大型结果卸载或 Conversation Artifact。
- 不同时长期维护旧 Tool-Call Loop 和新 Experiment Loop。
- 不以降低 Token 成本为首要目标；第一阶段允许合理增加推理成本，以换取更高修复成功率。
- 不在本设计中实现通用软件项目管理、需求开发或非 Bug 修复 Agent。

## 4. 方案比较

### 4.1 方案 A：继续增强当前 Tool-Call 状态机

继续增加阶段规则、重复签名、Prompt 检查点和特殊 ToolMessage。改动较小，但 Coordinator 会继续吸收
越来越多互相影响的规则，模型仍然以“下一次调用哪个工具”而不是“下一项实验解决什么问题”为中心。
该方案适合作为临时止血，不适合作为长期能力架构。

### 4.2 方案 B：能力导向双层循环（采用）

外层 StrategyPlanner 根据当前案件状态选择 Experiment；内层 ExperimentExecutor 在明确能力和预算内
自主完成多个工具调用；EvidenceEvaluator 评价完整实验；Reducer 原子提交状态；OutcomeAdjudicator
根据权威证据裁决最终结果。

该方案保留现有安全、证据和上下文基础设施，同时把 Loop Engineering 从“约束工具行为”提升为“组织
策略、实验和反馈”。复杂度适中，也为未来 Subagent 留出自然接口。

### 4.3 方案 C：立即引入完整 Multi-Agent 团队

为规划、调查、编辑和测试分别建立 Agent。该方案可能增加并行能力，但会立即引入上下文同步、事实冲突、
重复工具调用、成本和任务所有权问题。在单 Agent 的实验契约、证据提交和最终裁决尚未稳定前，多 Agent
只会放大现有 Loop 缺陷，因此不采用。

## 5. 总体架构

```text
BugfixService
      │
      ▼
DeepFixRepairLoop
├── CaseBlackboardBuilder
├── StrategyPlanner
├── ExperimentExecutor
├── EvidenceEvaluator
├── InvestigationStateReducer
└── OutcomeAdjudicator
      │
      ▼
InvestigationStore / Deterministic Evidence / Research Store
```

`BugfixService` 负责业务生命周期、人工审批、类型化恢复和报告交付；`DeepFixRepairLoop` 负责外层策略循环；
`LocalAgentExperimentExecutor` 包装 Deep Agents Graph，负责一个有边界的内层实验。

一次外层循环固定为：

1. `CaseBlackboardBuilder` 从权威 Store 构造当前只读视图；
2. `StrategyPlanner` 选择运行 Experiment、询问用户或提出结束；
3. 系统校验用户约束、安全策略和剩余预算；
4. `ExperimentExecutor` 在实验边界内自主调用多个工具；
5. `EvidenceEvaluator` 结合 Receipt 和模型语义判断评价结果；
6. `InvestigationStateReducer` 原子提交实验、事实候选、假设和 Evidence Gap 变化；
7. `OutcomeAdjudicator` 检查是否具备结束条件，否则重新进入 Planner。

## 6. 状态权威与 Blackboard

### 6.1 权威来源

| 信息类型 | 权威来源 |
|---|---|
| 用户目标和硬约束 | Task State 与稳定 User Message |
| 调查状态、假设、Evidence Gap、Experiment | Investigation Store |
| 文件读取、修改、测试、命令和审批 | Deterministic Evidence / Tool Receipt |
| 外部研究 | Research Store |
| 语义阶段记忆 | Working Memory |
| 压缩历史与旧来源引用 | CompactionSnapshot / Artifact |

模型只能生成带来源的语义候选、假设和策略，不能覆盖对应信息类型的权威记录。

### 6.2 `CaseBlackboardView`

Blackboard 是动态只读投影，不新建 Blackboard Store。它至少包含：

```python
class CaseBlackboardView(BaseModel):
    task_anchor: TaskAnchor
    reproduction_status: ReproductionStatus
    confirmed_claims: list[ProvenancedClaim]
    active_hypotheses: list[InvestigationHypothesis]
    rejected_hypotheses: list[InvestigationHypothesis]
    evidence_gaps: list[EvidenceGap]
    recent_experiments: list[ExperimentSummary]
    changed_files: list[FileChangeEvidence]
    test_results: list[SystemTestEvidence]
    research_evidence: list[ResearchEvidenceRef]
    unresolved_conflicts: list[EvidenceConflict]
    artifact_references: list[ArtifactReference]
    budget: LoopBudgetView
```

相同 `constraint_id`、`evidence_id`、`claim_id`、`hypothesis_id`、`experiment_id` 只展示一次。Protected
Context 展示当前权威状态，CompactionSnapshot 主要提供压缩历史和来源引用；Blackboard Builder 应复用
已有去重投影规则，不在 Prompt 中再次复制同一正文。

Working Memory 继续用于压缩恢复和语义连续性，但不是实时调查决策的权威来源。

## 7. StrategyPlanner

Planner 每轮只负责决定下一项策略，不直接执行文件操作。输出为：

```python
class StrategyDecision(BaseModel):
    decision_id: str
    task_id: str
    decision_type: Literal["run_experiment", "ask_user", "conclude"]
    current_assessment: str
    selected_hypothesis_id: str | None
    evidence_gap_ids: list[str]
    experiment_spec: ExperimentSpec | None
    reflection: StrategyReflection | None
    uncertainty: float
    rationale_refs: list[ProvenanceRef]
```

模型返回决策候选；Builder 根据任务、Blackboard 版本和规范内容确定性生成 `decision_id`。引用必须属于
当前任务。`run_experiment` 必须包含 Experiment；`ask_user` 必须指出无法由本地或外部证据解决的信息缺口；
`conclude` 必须提供候选 Outcome，但不能直接完成任务。

Planner 的输入是去重后的 Blackboard、剩余预算和最近实验评估，不是完整 Conversation Messages。需要旧
细节时，通过辅助性的 Artifact Retrieval 按需补充上下文。

正常情况下 Planner 只生成当前最佳策略，不为每轮强制列举替代方案。当连续实验无进展或证据冲突时，
Planner 才使用 `REFLECT` 策略；此时 `reflection` 必须比较至少两条替代路线、指出当前假设可能错误的
原因，并选择与此前实质不同的实验。`REFLECT` 是决策策略，不新增 AgentPhase。

Artifact Retrieval 是 Planner 构造决策上下文或 Executor 补充实验上下文的辅助操作。普通 Artifact
搜索和局部读取不单独创建正式 Experiment，也不增加 Experiment 数量或参与停滞统计；它们仍产生 Tool
Receipt、Artifact 访问记录和成本统计。只有“恢复并分析一组历史诊断材料”本身就是当前策略目标时，
Planner 才把它纳入一个正式 `INVESTIGATE` Experiment。

## 8. Experiment 与内层执行

### 8.1 Experiment 意图与自适应粒度

Experiment 是执行前定义的目标和预算。以下名称是可组合的意图标签，不是必须依次迁移的阶段：

- `REPRODUCE`：复现或反驳用户报告；
- `INVESTIGATE`：定位根因、验证假设或调用关系；
- `EDIT`：执行一组有明确因果目的的代码修改；
- `VERIFY`：执行修改后验证或回归测试；
- `RESEARCH`：查询外部权威资料；
- `REVIEW`：检查修改范围、风险、冲突和证据充分性。

```python
class ExperimentSpec(BaseModel):
    experiment_id: str
    task_id: str
    intents: set[ExperimentIntent]
    goal: str
    target_hypothesis_id: str | None
    evidence_gap_ids: list[str]
    success_criteria: list[SuccessCriterion]
    allowed_capabilities: set[InvestigationCapability]
    suggested_actions: list[SuggestedAction]
    step_budget: int
    model_call_budget: int
    token_budget: int | None
    time_budget_seconds: int
    fallback: str
```

`experiment_id` 由系统根据 task、StrategyDecision 和规范化实验定义生成。Planner 可以建议工具动作，但
Executor 可以在目标、能力和预算内改变顺序或选择等价工具。

粒度由问题和证据需求决定，不由意图标签强制拆分。简单 Bug 可以使用
`{INVESTIGATE, EDIT, VERIFY}` 在一个 Experiment 内完成读取、最小修改和指定验证；复杂 Bug 可以把复现、
竞争假设调查、修改和回归验证拆成多个 Experiment。拆分依据是上下文、风险、审批或预算边界，而不是为了
让 Experiment 看起来符合固定流程。

Experiment 只保留恢复所需的最小持久化生命周期：`prepared`、`running`、`waiting_approval`、
`completed`、`abandoned`。这些状态不规定 REPRODUCE、EDIT、VERIFY 的先后顺序，也不决定工具能力；它们
只用于审批续跑、幂等和故障恢复。人工审批会暂停并恢复同一个 `experiment_id`，不会创建新实验或计为
停滞。

### 8.2 与 WorkUnit 的边界

`Experiment` 是面向未来的策略执行单位；`WorkUnit` 是执行后根据 AIMessage Tool Call、配对
ToolMessage 和解释划分的历史单元，服务于上下文压缩。一个 Experiment 可以产生多个 WorkUnit，二者
使用稳定 ID 关联，但 WorkUnit 不驱动策略状态。

### 8.3 `ExperimentExecutor`

```python
class ExperimentExecutor(Protocol):
    def execute(
        self,
        experiment: ExperimentSpec,
        context: ExperimentContext,
    ) -> ExperimentResult: ...
```

第一阶段只实现 `LocalAgentExperimentExecutor`。它使用当前主模型和 Deep Agents Graph，在一个
Experiment Prompt 中执行多个工具，并返回：

```python
class ExperimentResult(BaseModel):
    experiment_id: str
    status: ExperimentResultStatus
    completed_criteria: list[str]
    tool_receipt_ids: list[str]
    observation_ids: list[str]
    evidence_candidates: list[EvidenceCandidate]
    hypothesis_updates: list[HypothesisUpdateCandidate]
    changed_files: list[str]
    test_evidence_ids: list[str]
    remaining_questions: list[str]
    executor_recommendation: str
```

Executor 不得直接修改 Investigation State，也不得宣布任务完成。

### 8.4 自主权和硬边界

系统硬性检查：用户文件范围、危险操作、审批、实验能力、时间/步骤/Token 预算、未变化状态下的完全重复
动作和 Tool Receipt 完整性。

系统不因以下情况自动阻断：不同范围或目的的文件重读、修改后重测、发现新的可验证依赖、实验内部调整
工具顺序、模型否定旧假设。原则是约束执行后果、真实性和资源边界，不微观管理合理推理路径。

## 9. EvidenceEvaluator 与进展

EvidenceEvaluator 采用 **确定性规则优先、LLM 语义判断兜底**。确定性层读取 Tool Receipt 和权威
Store，直接确认命令、exit code、超时、文件哈希、修改范围、测试时序和审批，这些判断不得调用模型。
只有“结果支持或反驳哪个假设”“是否缩小问题范围”“是否填补语义 Evidence Gap”等无法由规则可靠判断
的问题才调用主模型。语义层只能输出带来源的候选；规则已经能决定的字段不允许模型重判。

```python
class ExperimentAssessment(BaseModel):
    experiment_id: str
    outcome: ExperimentAssessmentOutcome
    deterministic_evidence_ids: list[str]
    accepted_claims: list[ProvenancedClaim]
    rejected_claims: list[RejectedClaim]
    closed_evidence_gap_ids: list[str]
    opened_evidence_gaps: list[EvidenceGap]
    supported_hypothesis_ids: list[str]
    rejected_hypothesis_ids: list[str]
    conflict_ids: list[str]
    progress_kind: ExperimentProgressKind
    recommended_strategy_change: str | None
```

Assessment outcome 为 `SUCCEEDED`、`PARTIALLY_SUCCEEDED`、`INCONCLUSIVE`、`FAILED`、`TIMED_OUT`、
`POLICY_BLOCKED` 或 `BUDGET_EXHAUSTED`。命令失败不等于实验没有价值；例如 pytest 超时可以成为死循环
证据。

强进展包括：复现或明确无法复现、关闭关键 Evidence Gap、用真实证据支持或排除假设、定位因果相关代码、
完成有效修改、获得修改后测试、发现用户描述与真实行为冲突。弱进展包括尚未验证的新依赖、合理新假设和
不完整但可用的执行结果。Phase 变化、保存 Working Memory、新文件本身、无来源 reason 和未变化状态下
重复相同命令不算强进展；弱进展不能无限重置停滞。

### 9.1 测试证据可信度

现有 `SystemTestEvidence` 需要扩展测试来源、范围和代码时序，而不是只记录命令与 exit code：

```python
class SystemTestEvidence(StrictModel):
    evidence_id: str
    command: str
    exit_code: int
    summary: str
    tool_call_id: str
    source_message_id: str
    origin: Literal[
        "user_specified",
        "repository_existing",
        "agent_generated",
        "minimal_reproduction",
    ]
    scope: Literal["targeted", "module", "full_suite"]
    timing: Literal["baseline", "post_change", "post_recovery"]
    workspace_baseline_id: str
    code_state_hash: str
    test_target_paths: list[str]
    test_content_hashes: dict[str, str]
```

`origin` 由用户命令来源、Baseline 中测试文件是否已存在及测试文件是否被 Agent 修改确定性推导；模型不能
自行把生成测试标为仓库测试。`scope` 由测试收集目标和命令分类器推导；不能只根据模型描述填写。

证据强度不是一个由模型自由生成的总分，而由 Adjudicator 根据任务要求组合判断：用户指定的验证命令是
当前任务的首要 Oracle；仓库原有 full suite 提供更强回归保证；Agent 生成测试和最小复现可以支持根因，
但不能单独证明 `FIXED`。如果 Agent 修改过测试文件，之后的通过结果必须明确降级，并且用户禁止修改测试
时构成范围违规。

## 10. Reducer、Phase 与停滞

### 10.1 唯一状态写入口

`InvestigationStateReducer` 接收当前版本、ExperimentResult 和 ExperimentAssessment，在一个事务中提交
Experiment Event、假设、Evidence Gap、冲突和物化状态。它负责幂等、乐观锁、状态迁移和稳定 ID；
Planner、Executor、Middleware 与未来 Subagent 都不能直接写调查状态。

### 10.2 Phase 为派生状态

`AgentPhase` 继续用于 CLI、Prompt、Debug 和少量安全策略，但不再是主要工具决策引擎。Phase 根据已提交
事实推导：未复现为 investigating，验证根因为 diagnosing，准备修改为 planning，修改实验进行中为
editing，修改后验证为 testing，证据足以裁决为 reviewing。这样模型发现根因后不会因阶段同步失败而被
阻止合理推进。

### 10.3 实验级停滞

停滞比较 Experiment 的目标假设、Evidence Gap、证据来源、策略族、工具参数和执行前后 Blackboard
版本：

1. 第一次无进展：反馈失败原因，允许 Planner 重新规划；
2. 连续第二次无进展：强制 `REFLECT`，比较至少两条替代策略；
3. 连续第三次无进展：必须改变假设、证据来源或 Experiment 意图；若评测反复出现该瓶颈，再单独评估
   是否需要 Subagent；
4. 仍无进展或任务硬预算耗尽：返回 `PAUSED`、`NEEDS_INPUT` 或 `BLOCKED`。

只有策略实质重复且 Blackboard 没有变化才算循环。代码哈希变化后重新运行同一测试是正常验证，不算
重复。

## 11. OutcomeAdjudicator

Planner 可以建议结束，系统根据任务约束、复现、修改、测试、审批和未解决问题裁决：

- `FIXED`：发生允许范围内的真实修改；修改后的用户指定验证或仓库原有测试通过；修改与根因证据有关；
  无范围违规。Agent 生成测试或最小复现通过不能单独满足该条件。
- `NOT_REPRODUCED`：修改前指定测试通过或用户报告无法复现；没有无依据修改；报告明确指出与用户描述
  不一致。
- `PARTIALLY_VERIFIED`：修改完成但只能执行局部验证，或完整回归因环境受限无法执行。
- `NEEDS_INPUT`：关键行为预期或复现条件只能由用户补充。
- `BLOCKED`：依赖、权限、网络、环境或测试基础设施阻止继续。
- `FAILED`：问题已复现，但在预算内无法得到可靠修复或修改未通过要求验证。

Planner 的 `CONCLUDE` 证据不足时，Adjudicator 创建新的 Evidence Gap 并返回外层循环；存在冲突时创建
`REVIEW` Experiment；只有裁决通过才生成最终报告。模型负责解释，系统负责状态真实性。

## 12. 副作用一致性与恢复

数据库事务和稳定 ID 只能保证 DeepFix 内部记录幂等，不能回滚已经发生在真实文件系统或子进程中的副
作用。新 Loop 必须先建立 Task Workspace 和 Operation Journal，再扩大模型自主执行范围。

### 12.1 独立 Task Workspace

每个 Task 保存 `source_project_root`、`workspace_root` 和 `workspace_baseline_id`。新 Experiment Loop 不
直接在用户原始目录执行副作用：

- 干净 Git 仓库优先创建独立 git worktree；
- 脏 Git 仓库或非 Git 项目创建保留当前文件状态的隔离副本；
- Python 解释器和只读依赖可以复用，但源码修改、生成文件和测试缓存限定在 Task Workspace；
- Baseline 至少记录 Git HEAD（若有）、初始 dirty 状态摘要、受管文件 hash、允许修改范围和 Python 环境
  指纹；
- 任务成功后输出可审阅 patch 和 Workspace 路径；把修改提升回用户原项目是显式交付动作，不属于
  OutcomeAdjudicator 的隐式副作用。

评测 Harness 必须总是创建全新 Workspace。旧任务迁移到新 Loop 时，如果无法证明现有目录的 Baseline，
任务暂停并要求建立恢复快照，不能猜测哪些修改由 Agent 产生。

### 12.2 Operation Journal

所有文件修改和命令执行先在 DeepFix 内部 Artifact/持久化目录写入追加式 Journal：

```python
class OperationJournalEntry(StrictModel):
    operation_id: str
    task_id: str
    experiment_id: str
    tool_call_id: str
    operation_kind: Literal["file_write", "file_edit", "file_delete", "command"]
    call_hash: str
    workspace_baseline_id: str
    pre_state: OperationStateSnapshot
    status: Literal["prepared", "started", "observed", "committed", "unknown"]
    post_state: OperationStateSnapshot | None
    receipt_id: str | None
    artifact_references: list[str]
```

顺序固定为：

1. 校验权限、Workspace 和 Experiment；
2. 写入并 fsync `prepared` Journal，记录参数 hash 和操作前文件 hash；
3. 标记 `started` 后只执行一次副作用；
4. 把 stdout/stderr 或修改差异先写入内部有界 Artifact，并记录真实操作后状态；
5. 保存并回读校验 Tool Receipt，Journal 进入 `observed`；
6. 提交 Deterministic Evidence、Experiment Event 和 Investigation State；
7. 所有提交确认后标记 `committed`。

Journal 不复制完整 ToolMessage 或大型输出，只保存生命周期、hash 和 Artifact/Receipt 引用。Receipt 仍是
工具结果权威对象；Journal 解决副作用跨越 Receipt 和数据库提交边界的恢复问题。

### 12.3 恢复协调

恢复时先冻结对应 Task Workspace，扫描非 committed Journal，再读取真实文件 hash、Git 状态、内部输出
Artifact 和已有 Receipt：

- 文件操作后状态与 Journal 一致：重建 Receipt/Evidence 并幂等补交，不重新修改文件；
- 文件操作状态与 pre/post hash 都不一致：标记冲突并暂停，保留 Workspace 供人工检查；
- 命令有完整输出 Artifact 和退出状态：重建 Receipt 并补交；
- 命令已经启动但没有可验证退出状态：标记 `unknown`，终止仍存活的受管进程组，不自动重跑；
- 只读操作无法确认结果：可以创建新的恢复 Experiment，但必须生成新 operation ID，不能把它伪装成旧
  操作重放。

恢复完成前 Planner 不得继续，OutcomeAdjudicator 也不得使用 `prepared`、`started` 或 `unknown` 操作
作为成功证据。

## 13. 错误传播与预算

业务失败作为 ExperimentResult 反馈给模型，包括测试失败、命令超时、文件不存在、搜索无结果、假设被
反驳和实验预算耗尽。命令超时必须保存清洗后的 stdout/stderr 尾部、超时、命令指纹和进程树终止状态，
Executor 可以基于该证据继续思考；只有未变化状态下完全重复同一超时命令才被拒绝。

基础设施失败抛带恢复元数据的类型化异常，包括 Store 提交失败、Receipt 写入或回读失败、权威证据读取
失败、Blackboard 构造失败、连续结构校验失败和 Graph 状态损坏。传播边界固定为：

```text
内部组件 → InvestigationCoordinator / DeepFixRepairLoop
         → BugfixService
         → TaskStatus.PAUSED
```

Middleware、Executor、Evaluator 和 Reducer 不直接修改业务任务状态。已经执行的工具不得因事件提交失败而
自动重跑；恢复时使用稳定 ID 幂等补交。

预算分为工具级、Experiment 级和 Task 级。工具级限制命令时间和输出；Experiment 级限制工具调用、模型
调用、Token 和总时间；Task 级限制实验数量、连续无进展实验、总 Token/费用和总运行时间。软预算要求
Planner 缩小策略，硬预算才暂停。Graph recursion limit 只作为最后保险。

## 14. 模型、Prompt、工具、记忆与 Skill

第一阶段沿用当前两模型配置，不新增 Model Router 或角色 API Key：主模型承担 Planner、Executor 和必要
的语义评估，压缩模型继续承担已有 Compaction 工作。Working Memory 仍由主 Agent 通过结构化工具保存。
是否拆分独立 Planner、Evaluator 或 Subagent 模型，必须由第一阶段评测证明存在模型能力或成本瓶颈后再
单独设计。

Prompt 按 `Core Policy + Role Prompt + Case Blackboard + Experiment Contract + Runtime Feedback` 动态
组合，不把整个修复生命周期和所有工具细节堆入同一个系统提示词。

工具继续通过能力注册声明 `capability`、风险、审批、并行、超时、Receipt 和结果 Schema。Experiment
申请 READ、SEARCH、EXECUTE、MODIFY、RESEARCH、MEMORY 等能力，而不是把策略绑定到固定工具名。

Skill 只提供领域策略知识和 Experiment 模板，例如 pytest 调试、依赖冲突或异步死锁。Skill 不得修改
Investigation State、宣布测试通过、绕过审批、重置停滞或覆盖系统证据。Skill 不是第二个 Loop
Controller。

## 15. 延后扩展

第一阶段只实现本地 Experiment 执行。`ExperimentExecutor` 的输入输出保持明确，足以作为未来替换执行者
的最小接缝，但不实现 `SubagentExperimentExecutor`、InvestigationPacket、Agent 通信协议或委派策略。
Multi-Agent、独立 Planner/Evaluator 模型、复杂模型路由和精细 Token 调度都延后；只有 A/B 评测证明
单 Agent Experiment Loop 的具体瓶颈无法通过现有主模型和工具解决时，才为该瓶颈单独立项。

## 16. 模块复用与迁移

本设计中的 Blackboard、Planner、Executor、Evaluator、Reducer 和 Adjudicator 是逻辑职责，不要求每个
职责都建立独立 Manager、Controller、Store 或模型调用。第一阶段只按独立不变量和可测试边界提取最少
文件，优先考虑 `blackboard.py`、`experiments.py`、`loop.py`、`evaluation.py`、`reducer.py`、
`workspace.py`；Outcome 规则可以先作为 `evaluation.py` 的纯函数，Operation Journal 可以由 Workspace
组件提供。只有文件职责再次明显膨胀时才继续拆分。

- `coordinator.py` 保留为薄门面，将策略、评估和状态提交委托给专门组件；
- `progress.py` 的有效逻辑迁入 EvidenceEvaluator，迁移完成后删除，避免两套进展标准；
- `stagnation.py` 改为 Experiment/Strategy 级检测；
- `phase.py` 保留为派生状态计算器；
- `middleware.py` 保留审批、安全、超时、Receipt 和运行时适配，移除被 Experiment 策略替代的细粒度
  门禁；
- `context.py` 继续提供 Working Memory 和压缩集成，`save_progress` 不再推动调查流程；
- `service.py` 通过 DeepFixRepairLoop 的标准结果工作，不理解 Experiment 内部状态；
- `agent.py` 继续作为 Composition Root；
- Deep Agents Graph 被 `LocalAgentExperimentExecutor` 复用，不再独自承担完整任务生命周期。

Investigation Store 只新增 StrategyDecision、Experiment、ExperimentEvent 记录，不新增 Blackboard、
Planner 或 Evaluator Store。旧任务没有活动 Experiment 时由 Blackboard 重新规划；旧 Tool Observation
保留为历史证据；旧 Tool 级 stagnation 字段只作为历史指标。迁移必须版本化、幂等且可中断恢复。

交付顺序以实验验证而不是组件数量为中心：

1. 固化旧 Loop 的 QuixBugs 基线、失败轨迹和评测预算；
2. 先实现 Task Workspace、Operation Journal 和增强 Test Evidence，补齐副作用与裁决可信度；
3. 实现最小纵向 Experiment Loop：Blackboard → 单次最佳策略 → 自适应 Experiment → 确定性优先评估
   → Reducer → Outcome；
4. 只在评测 Harness 中临时保留旧/新 Loop 选择，使用同一模型和相近预算进行 A/B；
5. 若能力和结论可信度达到第 19 节门禁，再迁移现有入口并删除被替代的 Tool 级策略；
6. 若没有明显改善，先分析轨迹并删减无效结构，不继续增加 Controller、模型角色或 Multi-Agent。

兼容选择不得成为长期面向用户的双 Loop 产品模式。

## 17. 测试与评测

### 17.1 单元测试

- Blackboard 权威投影、任务隔离、去重和冲突；
- StrategyDecision、ExperimentSpec、Result 和 Assessment 校验；
- Executor 工具、模型、Token 和时间预算；
- Evaluator 确定性证据优先、Claim 来源校验和进展分类；
- Reducer 幂等、并发、假设与 Evidence Gap 迁移；
- Phase 派生和 Experiment 级停滞；
- OutcomeAdjudicator 六种结果；
- 类型化异常和恢复元数据；
- Workspace Baseline、Journal 生命周期和真实文件 hash 对账；
- 测试 origin、scope、timing 的确定性分类及证据降级。

### 17.2 集成与端到端测试

- 一个 Experiment 连续执行多个相关工具；
- 中间工具没有进展但实验最终获得根因证据；
- pytest 超时反馈给模型并切换策略；
- 修改后允许重复运行同一测试；
- 未变化状态下拒绝相同无效实验；
- 证据不足的 CONCLUDE 重新产生 Evidence Gap；
- 程序本来正确时输出 NOT_REPRODUCED；
- 修改范围违规阻止 FIXED；
- 连续实验失败进入 REFLECT 而不是工具循环；
- 文件修改后 Receipt/Store 故障通过 Workspace 和 Journal 重建结果，不重复执行修改；
- 命令执行状态未知时停止受管进程、标记 unknown 并暂停，不自动重跑；
- Agent 生成测试通过不能单独裁决 FIXED；
- 多轮 Experiment、压缩、Artifact Retrieval、审批和恢复联合流程。

### 17.3 QuixBugs Harness

每个 Python Bug 使用独立 Task 和干净项目副本，固定 Buggy 版本、用户问题、允许修改范围、验证命令、
模型和预算。不得让一个 Task 同时修复整个 QuixBugs 项目。评测集覆盖单文件条件、边界、递归、数据结构、
跨函数调用、死循环、正常代码对照、干扰文件、大 Artifact 和首次假设错误。

相同模型、API、Bug、初始仓库和相近预算下对比旧 Tool-Call Loop 与新 Experiment Loop。代表性任务至少
运行三次，记录修复成功率、错误 FIXED 率、NOT_REPRODUCED 准确率、根因准确率、初始假设错误后的恢复
率、无效重复调用率、Experiment/Tool/Model Call 数量、Token、耗时和 Artifact Retrieval 次数。

A/B 结论必须结合轨迹解释改善来自哪里。如果新架构没有明显提升复杂 Bug 调查、错误假设恢复或结论
可信度，或者只是以不可接受的调用与 Token 增长换取偶然成功，则暂停全面迁移，优先删除无效抽象并定位
真实瓶颈，而不是继续扩展架构。

## 18. 可观测性

Progress Events 与 CLI 至少展示：`StrategyPlanned`、`ExperimentStarted`、`ExperimentToolCalled`、
`ExperimentTimedOut`、`ExperimentCompleted`、`ExperimentAssessed`、`HypothesisUpdated`、
`EvidenceGapClosed`、`StrategyChanged`、`OutcomeProposed`、`OutcomeAdjudicated` 和 `TaskPaused`。

Debug 记录稳定 task/decision/experiment/evidence ID、预算变化和清洗后的错误，不记录 API Key、完整环境
变量或无界源码正文。CLI 展示目标、实验、证据、假设和下一步，不把原始 LLM trace 作为默认进度界面。

## 19. 第一阶段验收标准

1. Deep Agents Graph 在一个 Experiment 内可以连续完成多个工具调用，并在实验边界返回结构化结果。
2. Planner、Executor、Evaluator、Reducer 和 Adjudicator 具有独立可测试接口，第一阶段可共享主模型。
3. Blackboard 不建立重复事实 Store，权威来源、去重和冲突规则与 Protected Context 一致。
4. 系统基于 Experiment 而不是单个 Tool Call 判断进展与停滞。
5. 超时、测试失败和假设被反驳会反馈给模型，不直接终止整个任务。
6. 每个新 Loop Task 在独立 Workspace 运行，并可依据 Baseline、Journal 和真实 hash 恢复已发生副作用。
7. 所有 Task 都能在预算内结束或以恢复元数据暂停，不需要人工强制中断。
8. 评测中错误 `FIXED`、越界修改、重复副作用、丢失 Receipt 和无恢复元数据的基础设施失败均为零。
9. 测试证据包含 origin、scope 和 timing；Agent 生成测试或最小复现不能单独证明 FIXED。
10. 正确识别修改前已通过的任务为 `NOT_REPRODUCED`。
11. 至少解决三个当前架构稳定失败的代表性 Bug，总体修复成功率明显高于相同模型的现有基线。
12. 连续失败时可观察到假设、证据来源或 Experiment 意图的实质变化，而不是只改变工具表面参数。
13. Working Memory、Compaction、Artifact Retrieval、Research、审批和 CLI Progress 联合测试无回归。
14. 只有 A/B 门禁通过才替换生产入口并删除旧 Tool 级策略；不长期维护两套 Loop。
15. 第一阶段不实现 Multi-Agent、独立角色模型或复杂 Token 调度。
