# DeepFix Agent 架构自查报告

**日期：** 2026-08-27  
**审查范围：** DeepFix 当前生产 Agent Loop、evaluation-only Experiment Loop，以及本机实际安装的 DeepAgents 0.7.8、LangChain 1.3.16、LangGraph 1.2.11。  
**审查方式：** 只读检查源码、测试、依赖实现和真实控制链；本报告不构成生产迁移授权。

## 1. 结论

DeepFix 在 DeepAgents/LangChain 之上重复构建了一套任务推进控制面，导致 `AgentPhase`、`TaskStatus`、Working Memory、阶段提示、停滞控制和 Experiment Planner 之间存在多套真相源；但 Receipt、Evidence、Operation Journal、Workspace Confinement、VerificationPolicy 和证据保真压缩属于 DeepFix 独有的可信执行面，应当保留。领域概念需要保持清晰，持久化组件则不应与概念一一对应；DeepFix 的结构化状态最终应收敛到同一 SQLite 数据库之上的少量 Repository。

目标架构应把系统拆成两个清晰层次：

- **轻量导航层：** 只回答最终目标、当前步骤和下一步，由 LangGraph Agent State 中的 Todo 负责。
- **可信执行层：** 继续由 Evidence、Receipt、Journal、Workspace 和 VerificationPolicy 判断实际上发生了什么，以及任务是否真的完成。

## 2. 当前真实控制链

```text
CLI
 ├─ 创建 TaskState / TaskWorkspace / VerificationPolicy
 ├─ 创建各类 DeepFix Store
 ├─ 创建 LangGraph SqliteSaver
 └─ build_agent
      ↓
DeepAgents graph
 ├─ FilesystemMiddleware / PatchToolCalls
 ├─ InvestigationMiddleware
 │    ├─ 读取 InvestigationState
 │    ├─ 按 AgentPhase 过滤工具
 │    └─ 注入调查状态
 ├─ PromptPolicyMiddleware
 │    └─ 再次读取 Phase 并注入阶段提示
 ├─ ProtectedContextMiddleware
 │    └─ 汇总 Task / Working Memory / Evidence / Snapshot / Investigation
 ├─ DeepFixCompactionMiddleware
 │    └─ 再次构造 Protected Context，计算预算并按需压缩
 └─ Main LLM
      ↓
Tool Call
 ├─ HITL / ApprovalPolicy
 ├─ Receipt 幂等检查
 ├─ Operation Journal prepare/start
 ├─ Backend / Workspace / Shell 执行
 ├─ Receipt 保存与回读校验
 ├─ Journal observe/commit
 ├─ 确定性 Evidence 收集
 ├─ PhaseResolver
 └─ StagnationDetector
      ↓
LangGraph Checkpoint
      ↓
BugfixService 再次投影 ToolMessage、Working Memory 和 Snapshot
 ├─ 更新 TaskState
 ├─ 运行 VerificationPolicy required-oracle 裁决
 └─ 生成最终报告
```

主要问题不是某一个模块过大，而是同一信息会由多个 Middleware 从多个 Store 重复读取、重新解释、重新注入和重新持久化。

## 3. 与框架高度重叠的模块

| DeepFix 模块或字段 | 当前职责 | 重叠来源 | 建议 |
|---|---|---|---|
| `AgentPhase` | 调查、诊断、编辑、测试导航 | `TaskStatus`、`WorkingMemory.phase`、Todo 当前项 | 迁移后删除 |
| `PHASE_PROMPTS` | 告诉模型当前阶段该做什么 | Todo、稳定 Repair Policy | 删除阶段提示矩阵 |
| `WorkingMemory.phase` | 保存模型认为的阶段 | `AgentPhase`、`TaskStatus` | 删除 |
| `WorkingMemory.next_steps` | 保存模型下一步计划 | Todo、RepairPlan、ExperimentSpec | Todo 接管后删除 |
| `save_progress` | 同时写摘要、事实、假设、阶段、下一步 | Evidence、Investigation、Snapshot、Todo | 拆分并降级，最终删除公共入口 |
| TaskState 活动状态 | INVESTIGATING、EDITING、TESTING 等 | AgentPhase、Todo | 只保留生命周期状态 |
| `TaskState.conversation` | 保存对话 | LangGraph Checkpointer | 迁移后删除 |
| TaskState 事实字段 | 保存 evidence、hypotheses、tests、changed files | 专用 Store | 改为查询投影 |
| Protected Context 构造 | 构造模型上下文 | Compaction 再次构造同一上下文 | 合并为一次构造 |
| Experiment Planner | 决定当前实验 | Todo 导航 | 不得与 Todo 同时成为生产权威 |

LangChain 1.3.16 已提供公开的 `TodoListMiddleware`、`PlanningState.todos` 和 `write_todos`，Todo 可随 LangGraph Checkpointer 持久化。因此不应增加独立 SQLite Todo Store。

框架 Todo 仍缺少 DeepFix 需要的三个约束：最多一个 `in_progress`、每次模型调用前稳定注入当前 Todo、基于主模型决策轮的停滞提醒。这些能力应由一个很薄的 DeepFix Todo Adapter 补充。

## 4. 与框架部分重叠的模块

### 4.1 Compaction

DeepAgents 已提供大型 ToolMessage 卸载、`conversation_history` Artifact、自然语言 Summarization 和基本 Tool Call/ToolMessage 边界保护。DeepFix 的 Task Anchor、结构化 Snapshot、provenance、完整工作单元、权威字段合并和失败原子性高于通用框架能力，因此 DeepFix Compaction 不应整体删除。

应继续复用 DeepAgents 的 Backend、Artifact、历史序列化和大型结果卸载，但由 DeepFix 负责 Snapshot 构建、Pydantic 校验、字段合并和事件提交。

### 4.2 Approval

LangChain HITL 负责 interrupt、Checkpoint 和恢复执行；DeepFix ApprovalPolicy 负责风险分类、业务审批规则、审计证据和 Service 状态映射。DeepFix ApprovalPolicy 应保留为 HITL 的策略适配器。

### 4.3 Working Memory

DeepAgents MemoryMiddleware 主要加载 `AGENTS.md` 等长期项目说明，并不等同于单任务 Working Memory。两者不是直接重复。但 DeepFix Working Memory 内部与 InvestigationStore、CompactionSnapshot、TaskState 存在明显重叠。

WorkingMemoryStore 的长期目标是删除，而不是升级成 WorkingMemory V2。导航迁移到 Todo，当前认知迁移到 Investigation/Evidence，压缩历史迁移到 HistoryRepository；迁移完成后只允许保留兼容读取期，不得继续双写。

### 4.4 Message Identity

LangGraph Checkpointer 可以保存消息，但不能替代 DeepFix 在 Receipt 幂等、WorkUnit coverage、provenance 和恢复重放中需要的确定性消息 ID。稳定 Message ID 应作为薄层保留。

## 5. DeepFix 独有且应保留的能力

- Tool Receipt 及并行调用幂等回执。
- Operation Journal 的 prepare/start/observe/commit 和恢复重建。
- 每任务独立 Workspace、路径 canonicalize、symlink/junction 越界防护。
- Shell confinement、解释器绑定、超时和进程树终止。
- 确定性 Evidence 与外部 Research Evidence；Research 获取能力保留为 Service，不再要求独立 Store。
- 测试来源、范围、修改前后时机和 provenance root。
- VerificationPolicy、required oracle、冲突证据和 false-FIXED 防护。
- Artifact Retrieval。
- 证据保真的 CompactionSnapshot 和类型化恢复异常。
- Service 将无法安全恢复的任务转换为 `PAUSED` 的边界。

这些模块构成 DeepFix 的领域价值，不应为了简化导航层而删除。

## 6. 当前多套真相源

| 信息 | 当前来源 | 推荐唯一权威 |
|---|---|---|
| 原始任务目标 | HumanMessage、TaskState、TaskAnchor、Snapshot | 绑定原始用户消息 ID 的 Task Definition |
| 用户约束 | User Messages、TaskAnchor、Snapshot、模型摘要 | 用户消息及带 source ID 的约束投影 |
| 当前步骤 | AgentPhase、TaskStatus、WM、RepairPlan、ExperimentSpec | Todo State |
| 生命周期 | TaskStatus、AgentPhase | 精简后的 TaskStatus |
| 当前对话 | Graph Messages、TaskState.conversation、Artifact | Checkpointer；旧历史由 Artifact 负责 |
| 假设 | InvestigationState、WM、Snapshot、TaskState、Outcome | InvestigationRepository |
| 工具执行事实 | ToolMessage、Receipt、Journal、模型描述 | ExecutionRepository 中语义独立的 Receipt + Operation |
| 文件修改事实 | ToolMessage、TaskState、Snapshot | EvidenceRepository；来源追溯到 Operation、Receipt 和工作区 hash |
| 测试事实 | ToolMessage、TaskState、Snapshot | EvidenceRepository 中的 TestEvidence；完成条件由 VerificationPolicy 判断 |
| 研究状态 | WM、Snapshot、ResearchStore | EvidenceRepository 中带来源和验证状态的 ExternalEvidence |
| 下一步 | WM.next_steps、RepairPlan、ExperimentSpec | Todo State |

## 7. 调查阶段自锁链

当前阶段推进的关键并不是 `save_progress`，而是 `record_hypothesis`：

```text
失败测试 → TEST_OBSERVED → DIAGNOSING
                        ↓
模型即使在自然语言中发现根因，也不会改变 InvestigationState
                        ↓
必须调用 record_hypothesis(target_state="supported", evidence_ids=...)
                        ↓
HYPOTHESIS_SUPPORTED → PLANNING → 修改工具开放
```

如果模型没有按接口提交受支持假设，PromptPolicy 会持续注入诊断阶段提示，InvestigationMiddleware 会继续隐藏修改工具。进一步触发 stagnation 后，系统可能只允许 `record_hypothesis`、`continue_investigation` 和 `save_progress` 等元工具，形成“已经知道答案，但没有资格继续”的控制循环。

因此现有系统过度依赖模型正确维护结构化控制状态。Todo 应负责导航，而安全工具是否可执行应由直接业务不变量判断，不应继续由语义 Phase 间接控制。

## 8. 删除、合并、适配和保留清单

### 可直接删除

- 生产构建未使用的 `ContextMemoryMiddleware`。
- 生产构建未使用的 `ResearchEvidenceMiddleware`。
- 遗留 `build_context_middleware` 及只验证旧实现存在的测试。

### 迁移后删除

- `WorkingMemory.phase`、`WorkingMemory.next_steps`。
- `AgentPhase`、`PhaseResolver`、`PHASE_PROMPTS`。
- phase-based tool visibility。
- stagnation 的 permit/阶段纠正状态机。
- `continue_investigation`。
- 当前过载的公共 `save_progress`。
- `TaskState.conversation` 以及事实字段的重复存储。

### 合并

- ProtectedContext 和 Compaction 的上下文读取/构造。
- CaseBlackboard 与其他上下文投影中的重复聚合逻辑。
- 确定性 Evidence 与外部 Research Evidence 的持久化访问统一到 `EvidenceRepository`；Research 搜索、抓取和验证仍由独立 Service 负责。
- Receipt 与 Operation 保留为不同领域模型，但持久化、幂等、恢复和查询统一到 `ExecutionRepository`。
- VerificationPolicy、Task Definition、生命周期、资源预算和最终裁决统一由边界明确的 `TaskRepository` 管理。
- Hypothesis 与 UnresolvedQuestion 保留不同语义，但统一由 `InvestigationRepository` 管理。
- 当前 `CompactionStore` 收窄为 `HistoryRepository`，不再拥有当前 Evidence、Hypothesis 或 Task 状态。

### 降为薄适配器

- ApprovalPolicy → LangChain HITL 策略。
- Todo → LangChain Todo State/Checkpointer 的 DeepFix Adapter。
- Stable Message ID → Checkpointer 之上的幂等身份层。
- TaskState → 不可变 Task Definition、生命周期和只读报告投影；报告投影不持久化为第二份事实。
- PromptPolicy → 单一稳定 Repair Policy。

### 保留

- Evidence、Research 获取能力、Receipt、Operation Journal、Workspace、Shell Confinement。
- VerificationPolicy、Test Evidence provenance、Artifact Retrieval。
- 结构化 CompactionSnapshot、类型化恢复和硬预算保护。

## 9. 推荐最终架构

```text
Task Domain
不可变定义 + 用户约束投影 + 生命周期
        ↓
Todo Navigation（LangGraph Agent State）
当前步骤 + 下一步
        ↓
Main Agent
稳定 Repair Policy + Todo + 单次构造的 Context
        ↓
Trusted Execution Plane
Approval → Journal → Backend → Receipt → Evidence
        ↓
InvestigationRepository + EvidenceRepository
        ↓
TaskRepository 中的 VerificationPolicy
        ↓
OutcomeAdjudicator
        ↓
FIXED / NOT_REPRODUCED / PAUSED / FAILED
```

Todo 不写 Evidence，不复制假设、Receipt 或 Journal，也不作为任务完成权威。导航提醒以已完成的 Tool Round 为计数单位：同一个 AIMessage 发起的一个或多个并行 Tool Call，加上全部配对 ToolMessage/Receipt，合计为一个 Tool Round。成功且实质改变 Todo 状态时计数归零，机械改写文本不能重置计数。

不应从 Todo 生成新的权威 Phase。UI 可以派生临时活动标签，但不能写回业务状态或控制工具权限。

### 9.1 Todo 与真实进展的轻量反馈环

Todo 负责导航，真实进展仍由 Investigation、Evidence、Execution 和 Verification 的权威数据决定。两者通过只读、瞬时的 Navigation Feedback 形成反馈环，不增加新的 Store：

```text
连续 3 个已完成 Tool Round 且 Todo 无有效更新
                         │
                         ├── 或出现一次新的确定性里程碑
                         ▼
             Todo Navigation Middleware
                         ▼
          下一次 Model Call 动态注入 Reminder
```

确定性里程碑至少包括：带 Evidence 的 Hypothesis 获得支持、UnresolvedQuestion 被解决、成功代码修改后仍缺验证、required oracle 全部满足，以及用户问题在基线验证中未复现。里程碑只触发一次提醒，不自动完成或推进 Todo。

Reminder 应同时展示当前 Todo 和一小段真实进展，例如支持假设的 Evidence ID、未解决问题数量、是否已有代码修改以及 required oracle 状态，并要求模型检查：

1. 当前 `in_progress` 项是否已满足；
2. 是否已有足够证据停止继续调查；
3. 是否应完成当前项并进入修改、验证或收尾；
4. 下一步是否仍服务于不可变的原始任务目标。

`rounds_since_todo_update`、最近 Todo 进展指纹和最近提醒指纹属于 Graph State 中的非权威运行元数据，可随 Checkpointer 恢复，但不进入 TaskRepository、领域 Store 或最终报告。Todo 的状态变化、当前 `in_progress` 项变化以及项目新增或删除属于有效更新；只改写文案不属于有效更新。Reminder 注入后轮数清零，如果 Todo 仍未更新，则三个 Tool Round 后再次提醒。

Todo 更新只重置导航提醒计数，不能重置 Investigation 的真实停滞判断。Todo 状态变化、Phase 变化、普通文件读取和文案改写都不能被视为决策相关进展；真实进展只能来自独立新 Evidence、Hypothesis 状态迁移、问题解决、已提交代码状态变化或 oracle 结果。Harness 不得根据这些信号自动修改 Todo，也不得使用 Todo 控制工具权限或最终裁决。

## 10. 最小状态与持久化架构

### 10.1 权威状态域

```text
LangGraph State
├── messages
└── todos

DeepFix
├── TaskRepository
├── InvestigationRepository
├── EvidenceRepository
├── ExecutionRepository
└── HistoryRepository
```

Graph State 回答“聊了什么、准备做什么”；Task 回答“用户要求什么、任务能否继续和最终裁决是什么”；Investigation 回答“当前如何解释 Bug、还有什么未知”；Evidence 回答“实际观察到了什么”；Execution 回答“工具与副作用实际执行到哪里”；History 回答“被压缩掉的历史在哪里”。OutcomeAdjudicator 是读取这些权威视图的纯服务，不构成新的状态域。

架构必须遵守三个不变量：

1. **State owns navigation：** Todo 是当前计划的唯一权威；
2. **Domain Repositories own current facts：** Hypothesis、Evidence、Receipt 和 Operation 不能被 Todo、报告或 Snapshot 覆盖；
3. **Snapshot owns history, not truth：** Snapshot 只恢复历史上下文，当前领域记录始终优先。

Task lifecycle 只保留 `CREATED`、`RUNNING`、`WAITING_APPROVAL`、`PAUSED`、`COMPLETED`、`FAILED` 和 `CANCELLED`；`INVESTIGATING`、`DIAGNOSING`、`EDITING`、`TESTING`、`REVIEWING` 不再作为业务状态。

### 10.2 Persistence Boundary

DeepFix 默认使用一个 `deepfix.db` 和一个 Artifact 根目录。LangGraph Checkpointer 与五个 DeepFix Repository 可以共享同一 SQLite 文件，但框架表仍由 LangGraph 管理。五个 Repository 必须支持共享 connection/UnitOfWork，避免“同一数据库文件、不同连接分别提交”造成的伪原子性。SQLite 只保存结构化元数据；大型 Tool 输出、完整对话历史、研究文档和诊断内容继续写入 Artifact，并在写入、回读和 hash 校验成功后再提交数据库引用。

Receipt 和 Operation 的 Repository 合并不能消除外部副作用边界。`prepared`、`started`、`observed`、`committed` 和 `unknown` 生命周期仍然保留，数据库事务不得跨越模型调用、Shell 命令、文件修改或网络请求。

### 10.3 TaskRepository 准入边界

“具有 `task_id`”不是进入 TaskRepository 的充分条件。只有定义任务契约、任务生命周期、全任务资源限制或最终裁决的数据，才能由 TaskRepository 持有：

```text
允许：
task_definitions
task_lifecycle
verification_policies
adjudication_decisions
token_budgets
token_reservations

禁止：
messages / todos
hypotheses / unresolved_questions
evidence / test_results / changed_files
operations / receipts / approvals
snapshots / artifacts
conversation summaries
report fields
```

Task Definition 是绑定 `original_message_id` 的不可变快照；后续用户补充继续以新的 User Message 或带来源消息 ID 的 Constraint 投影存在，不反向修改原始问题。TaskRepository 不得提供可以整体覆盖任务的通用 `save(TaskState)` 接口，只提供创建定义、迁移生命周期、保存 VerificationPolicy、预留/结算预算和记录 AdjudicationDecision 等窄操作。

CLI 和报告通过只读 `TaskReportView` 查询多个 Repository；该 View 不持久化。Token reservation 只是 Task Budget 的内部事务账本，不能成为把模型调用、工具执行或其他运行事件继续塞入 TaskRepository 的入口。

OutcomeAdjudicator 不直接读取 Receipt、Journal 或它们的 `prepared/started/observed/committed` 状态。Execution/Recovery 层先将底层状态投影成 `ExecutionIntegrity`，Evidence 层提供 required oracle、冲突 Evidence 和代码状态视图；Adjudicator 只读取 Task Definition、VerificationPolicy、Evidence view、ExecutionIntegrity 和 code state，输出裁决结果及其依据的 Evidence/Operation ID。TaskRepository 只保存裁决结果和引用，不复制 Evidence 内容。

### 10.4 History 不是当前真相

HistoryRepository 只保存 Snapshot 生命周期、覆盖消息/工作单元 ID、历史语义项、source refs 和 Artifact refs。Snapshot 中每个历史事实和假设必须保留稳定 ID 与 provenance；同一 ID 的当前领域记录始终覆盖 Snapshot 投影。除保证安全激活和幂等所需的 `prepared/active/abandoned`、`input_hash` 和 coverage 信息外，不再为 Snapshot 复制 `task_version`、`research_version`、`evidence_version` 等领域版本。

## 11. 重构优先级

### P0：冻结权威来源和现有行为

- **删除：** 无。
- **保留：** 全部可靠性模块。
- **风险：** 最低。
- **验证：** characterization tests、固定 QuixBugs 案例、恢复和 fault-injection 全部通过。

### P1：接入轻量 Todo 导航

- **删除：** 暂不删除旧 Phase。
- **保留：** 旧 Phase 作为 shadow state。
- **实现：** 复用 LangChain Todo State，增加单一 `in_progress`、稳定注入、三个 Tool Round 兜底提醒和确定性里程碑的一次性提前提醒。
- **风险：** 机械更新 Todo、压缩或恢复后导航丢失。
- **验证：** 多次压缩、Checkpoint 恢复、并行工具和无实质更新测试。

### P2：移除 Working Memory 导航职责

- **删除：** `phase`、`next_steps`。
- **迁移：** checked files、hypotheses、evidence、experiments 到专用 Store。
- **降级：** `save_progress` 变为兼容入口，不再写 Phase、下一步或确定性事实。
- **风险：** 老任务恢复和报告字段缺失。
- **验证：** 数据迁移、连续压缩和报告一致性测试。

### P3：移除 Phase 自锁链

- **删除：** AgentPhase、PhaseResolver、阶段提示、phase-based tool filtering、permit 状态机。
- **保留：** 范围校验、审批、Receipt、Journal、VerificationPolicy 和硬循环保护。
- **风险：** 模型过早修改。
- **验证：** 越界修改与 fault-injection 零违规，false-FIXED 不升高，已知根因但未调用元工具的案例可以继续推进。

### P4：合并上下文与持久化投影

- **删除：** TaskState 对话和事实副本、遗留 Context Middleware。
- **合并：** Protected Context、Compaction、Blackboard 的共同读取和投影。
- **风险：** 历史兼容和报告查询变化。
- **验证：** 每轮只构造一次上下文，旧任务可恢复，Artifact/Snapshot 失败时原消息不被删除。

### P5：决定 Experiment Loop 去留

- **规则：** 当前 evaluation-only Experiment Loop 不进入生产。
- **通过 A/B：** 单独编写生产迁移计划，并明确它取代哪一套旧控制逻辑。
- **失败或不确定：** 保留 Legacy 生产入口，根据 trace 删除或修订无效机制，不继续增加抽象。
- **验证指标：** 修复成功率、false-FIXED、错误假设恢复率、无效重复调用率、Token、调用数、耗时和成功修复/100k Token。

## 12. 与 Plan 4 Task 4 的关系

本报告不改变已冻结 A/B gate，也不提前执行生产迁移。Plan 4 Task 4 应继续对比当前 Legacy Control 与 evaluation-only Experiment Treatment。实验结果只回答 Experiment Loop 是否值得进入后续迁移设计；即使结果为 `pass`，Todo/Phase/Working Memory 的删减仍需单独的、经过审阅的生产迁移计划。
