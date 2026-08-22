# DeepFix 证据保真上下文压缩设计

日期：2026-08-22  
状态：待审阅  
适用范围：当前单 Repair Agent 的 Python Bug 修复流程

## 1. 背景

DeepFix 当前复用 Deep Agents 的 `SummarizationMiddleware` 和
`SummarizationToolMiddleware`：达到模型上下文 70% 时自动摘要，只保留最近 15%；Agent
也能主动调用 `compact_conversation`。Deep Agents 还负责大型 `ToolMessage` 卸载、内联媒体
卸载和 `conversation_history` artifact。DeepFix 另外使用版本化 `WorkingMemoryStore` 保存
调查进度，并在模型调用时动态注入最新 Working Memory。

现有实现存在五个证据保真风险：

1. 固定 70%/15% 只按消息和 token 选择切点，虽然上游会避免直接拆开部分
   `AIMessage`/`ToolMessage` 对，但它不识别“目的—并行调用—全部结果—结果解释”组成的完整
   工作单元。
2. `ProgressSnapshot` 已保存 `rejected_hypotheses`、`checked_files`、`experiments`，当前
   `render_working_memory()` 却没有注入这些字段。
3. 上游摘要是普通自然语言。多次压缩会把“上一版摘要”再次交给模型总结，用户约束、排除原因和
   测试结果可能逐轮漂移。
4. 当前没有独立 Task Anchor、系统确定性证据保护块、结构化压缩快照和工作单元模型。
5. Deep Agents 自动压缩在 history 写入失败时仍继续摘要；手动 `compact_conversation` 当前先
   生成摘要再写 artifact。二者都不满足“先确认可恢复，再替换旧上下文”的失败保护要求。

## 2. 目标

- 采用“证据保真优先”的 DeepFix 压缩协调层，而不是只调整上游百分比。
- 继续复用 Deep Agents 的大型 ToolMessage 卸载、`conversation_history` artifact、媒体处理、
  Backend 与 `compact_conversation` 用户体验；Delta 生成复用 Agent 已有模型实例。
- 每次模型调用动态注入 Task Anchor、完整字段 Working Memory 和系统确定性证据；这些内容不写入
  Conversation Messages，也不参与被摘要消息的选择。
- 以完整工作单元为最小保留/压缩单位，不能拆开 Tool Call、配对 ToolMessage 和结果解释。
- 使用自适应 token 区间和优先级选择，尽量保留有用工作单元，不再固定保留最近 15%。
- 通过版本化、结构化 `CompactionSnapshot` 合并历史，后续压缩不再反复总结自然语言摘要。
- artifact 或 snapshot 任一步失败时保持旧消息有效；ContextOverflow 最多执行一次最小上下文重试。
- 保持当前 HITL、任务隔离、研究证据隔离、本地测试完成门禁和单 Agent 架构不变。

## 3. 非目标

- 不自研大型 ToolMessage、媒体或通用文件卸载系统。
- 不引入向量数据库、跨任务长期记忆或 Investigator 子 Agent。
- 不让模型修改用户约束、真实测试退出码、审批决定、实际文件操作或外部证据验证状态。
- 不保证把无限长度的保护数据原文全部塞进模型窗口；超出可用预算的细节必须可恢复且不能静默丢失。
- 不在本设计中改变 DeepSeek 模型、修复工作流、审批等级或任务完成条件。

## 4. 方案比较

### 4.1 方案 A：只调整 Deep Agents 阈值和提示词

把 70%/15% 改成更保守的参数，并提示 Agent 更频繁调用 `save_progress`。优点是改动最小；缺点是
仍然按消息切分、仍然反复总结自然语言摘要，也无法保证 artifact 失败时不替换旧上下文。该方案不能
解决核心问题，不采用。

### 4.2 方案 B：DeepFix 压缩协调层（采用）

DeepFix 在模型调用和主动压缩入口上增加协调层，负责权威数据、工作单元、预算、结构化 Snapshot
和提交顺序；Deep Agents 继续提供成熟的 Backend、Artifact、history 序列化、大型 ToolMessage/
媒体卸载和 Tool 外观，DeepFix 复用同一个模型实例生成结构化 Delta。
该方案在证据保真、可测试性和复用成本之间最平衡。

协调层会把 Deep Agents 封装在一个窄适配器后。适配器优先使用公开 Backend、Artifact 和 Middleware
接口；当前 `deepagents>=0.7,<0.8` 如果没有公开的 history 序列化入口，只允许为这一项设置受契约
测试保护的兼容边界。DeepFix 不调用 Deep Agents 的私有 cutoff、summary 或 event 方法。

### 4.3 方案 C：完全替换 Deep Agents 上下文能力

DeepFix 自己实现消息卸载、媒体、history、摘要 Tool 和溢出裁剪。它控制最强，但重复已有能力、维护面
大，并容易让本项目偏离“基于 Deep Agents SDK 的 Agent”目标，不采用。

## 5. 总体架构

```text
Model Call / compact_conversation
             │
             ▼
ProtectedContextMiddleware ───────────────┐
  ├── Task Anchor                        │ 动态 System Block
  ├── 完整字段 Working Memory             │ 不写 Conversation Messages
  └── Deterministic Evidence              │
             │                            │
             ├── EvidenceCollector ───────┘
             ├── WorkUnitPartitioner
             ├── ContextBudgetMonitor
             └── CompactionSnapshotBuilder
                         │
                         ▼
              Deep Agents Adapter
              ├── 大型 ToolMessage 卸载
              ├── conversation_history / media
              ├── Backend / Artifact / history 序列化
              └── compact_conversation Tool 外观
                         │
             ┌───────────┴────────────┐
             ▼                        ▼
CompactionSnapshotStore(SQLite)   Artifact Backend
版本化结构化历史                  完整可恢复消息与大型结果
```

`CompactionSnapshotBuilder` 由 DeepFix 实现，并直接复用 Agent 已有模型实例生成
`CompactionDelta`；该调用不经过 Deep Agents 的私有摘要方法。

`ProtectedContextMiddleware` 是协调入口，不把权威数据复制进普通消息。现有
`ContextMemoryMiddleware` 和 `ResearchEvidenceMiddleware` 的渲染逻辑会被复用，但不再各自独立拼接
可能相互重复的 System Block。`PromptPolicyMiddleware` 继续提供稳定规则和阶段策略。

自动压缩与主动 `compact_conversation` 必须调用同一个协调器，不允许存在两套切分、Snapshot 或失败
提交逻辑。对 Agent 而言，Tool 名称和用途保持不变。

## 6. 数据权威与保护区

### 6.1 分类型权威来源

DeepFix 不建立“系统证据高于用户约束”或“用户高于系统”的全局顺序。不同信息类型有各自唯一的
权威来源：

| 信息类型 | 权威来源 | 模型职责 |
|---|---|---|
| 任务目标、用户约束及其撤销 | 带稳定 message ID 的用户原始消息 | 提出带原文位置的候选，不得改写规范值 |
| pytest 结果 | 配对 execute Tool Call/ToolMessage 与确定性证据 ledger | 解释结果对假设的影响，不得改 exit_code |
| 文件操作结果 | write/edit/delete ToolMessage 与工作区观测 | 总结修改目的和影响，不得把批准目标说成已成功修改 |
| 审批决定 | BugfixService 的 ApprovalRecord/ledger | 解释约束，不得改变 approve/reject |
| 外部研究验证状态 | ResearchEvidenceStore | 总结 verified/contradicted/unverified 的含义，不得改状态 |
| 语义事实、实验解释和假设 | 带 provenance 的 Working Memory、WorkUnit 和 Snapshot | 提取、合并并维护 active/rejected/confirmed 状态 |

Working Memory 和 CompactionSnapshot 是语义记忆与历史投影，不是用户约束或确定性记录的第二权威
Store。发生冲突时先按信息类型找到该类型的权威来源；只有纯语义信息没有确定性来源时才保持
`unresolved`。冲突不会静默删除，而是记录为 `ConflictRecord` 并显示来源。

### 6.2 Task Anchor

`TaskAnchor` 每次从当前任务记录动态构造：

```python
class TaskAnchor(BaseModel):
    task_id: str
    original_problem: str
    user_constraints: list[UserConstraint]
    project_root: str
    project_python: str
    approval_mode: str
    task_status: str
    pending_question: str | None
```

```python
class UserConstraint(BaseModel):
    constraint_id: str
    text: str
    source_user_message_id: str
    state: Literal["active", "revoked"]
    superseded_by: str | None
```

约束文本必须能够在对应用户消息中逐字定位；模型只能提出候选，系统验证来源后才能入账。新增、撤销或
替换约束必须引用更晚的用户消息。没有用户来源的 Snapshot 内容不能进入 `user_constraints`。
Task Anchor 默认注入全部 active 约束；revoked 约束保留在版本化 Snapshot 中，只在存在冲突或追溯
需要时显示来源，不再作为当前指令执行。

为支持可靠来源，TaskState 的用户对话账本需要为每条用户输入保存稳定 `message_id`。旧任务没有 ID
时按 `task_id + 原始顺序 + 内容哈希` 确定性迁移，不能生成每次读取都变化的 ID。

### 6.3 完整字段 Working Memory

保护区必须呈现 `ProgressSnapshot` 的全部字段类别：

- `phase`
- `summary`
- `facts`
- `evidence`
- `active_hypotheses`
- `rejected_hypotheses`
- `checked_files`
- `experiments`
- `next_steps`
- `unresolved_questions`

“完整”指不允许像当前实现一样遗漏整个字段类别。由于模型窗口有限，渲染遵循以下规则：

- 在字段和总预算允许时注入所有条目原文。
- 超预算时按证据权威与新近程度保留条目，但每个字段仍必须出现。
- 被移出的条目必须显示 `omitted_count`、内容哈希和可读取的 snapshot/artifact 引用，不能只写
  `truncated` 后丢失恢复路径。
- Working Memory Store 中的原始版本不因 prompt 渲染而修改。

### 6.4 系统确定性证据块

`EvidenceCollector` 只从当前 `task_id` 的系统 Store、Graph ToolMessage 和 Service 记录收集：

```python
class DeterministicEvidenceBlock(BaseModel):
    tests: list[SystemTestEvidence]
    file_changes: list[FileChangeEvidence]
    approvals: list[SystemApprovalEvidence]
    external_evidence: list[ExternalVerificationEvidence]
```

```python
class SystemTestEvidence(BaseModel):
    evidence_id: str
    tool_call_id: str
    command: str
    exit_code: int
    summary: str

class FileChangeEvidence(BaseModel):
    evidence_id: str
    path: str
    operation: Literal["write_file", "edit_file", "delete"]
    state: Literal["approved_target", "tool_succeeded", "observed"]
    tool_call_id: str | None

class SystemApprovalEvidence(BaseModel):
    evidence_id: str
    operation: str
    decision: Literal["approve", "reject"]
    risk: str

class ExternalVerificationEvidence(BaseModel):
    evidence_id: str
    evidence_level: Literal["E1", "E2", "E3"]
    verification: Literal["unverified", "verified", "contradicted"]
    linked_test_tool_call_ids: list[str]
    artifact_path: str
```

当前 `TaskState.changed_files` 在审批时就记录路径，因此旧数据只能标记为 `approved_target`，不能冒充
操作已经成功。未来只有成功的 write/edit/delete ToolMessage 或工作区观测才能提升状态。

当前旧 `TestResult` 没有 `tool_call_id`。迁移时保留真实 command/exit_code，并生成稳定的
`legacy:<task_id>:<index>` 来源 ID；新结果必须持久化真实 Tool Call ID。

Collector 在任何消息可能进入压缩前，把新发现的配对测试结果、文件操作结果和审批记录幂等写入
任务级 `deterministic_evidence` ledger。后续保护区和 Snapshot 从 ledger 重建，不能依赖已经被压缩的
Graph Messages 仍然存在。Research Evidence 继续以 `ResearchEvidenceStore` 为唯一权威源，只在读取时
投影到统一证据块，避免两份可写状态。

### 6.5 保护区注入形式

三个保护块经过 XML 转义后追加到 System Message：

```xml
<deepfix_task_anchor>...</deepfix_task_anchor>
<deepfix_working_memory>...</deepfix_working_memory>
<deepfix_deterministic_evidence>...</deepfix_deterministic_evidence>
```

保护区通过 Runtime `thread_id` 读取并再次核对每条记录的 `task_id`。Task Anchor 或系统证据读取失败
时禁止模型调用并抛出类型化异常；Working Memory 尚不存在是合法状态，显示为 `version=none`。

### 6.6 去重投影规则

Store 可以为恢复和审计保存重叠投影，但一次 ModelRequest 中同一规范实体只能展示一次。
`ProtectedContextProjector` 在渲染前建立三个全局索引：`constraint_id`、`evidence_id`、
`hypothesis_id`。索引采用固定所有者：

| 规范实体 | 当前状态的展示所有者 | Snapshot 在模型请求中的职责 |
|---|---|---|
| constraint_id | Task Anchor | 不重复约束正文或状态，只保留未被当前 Anchor 覆盖的历史来源 |
| evidence_id | Deterministic Evidence | 不重复测试、文件、审批或研究记录，只保留历史 WorkUnit/artifact 来源 |
| hypothesis_id | 最新 Working Memory；若不存在才使用 Snapshot | 不重复当前假设，只补充历史迁移来源和已压缩工作单元 |

投影顺序固定为 Task Anchor → Deterministic Evidence → Working Memory → CompactionSnapshot。前一层登记
的 ID，后一层不得再次渲染实体正文、当前状态或同义副本。后一层携带的新 provenance 合并进唯一的
规范实体；Snapshot 中相应条目从可见历史投影中移除。这样模型只看到一份当前状态，同时仍能通过
Snapshot version、WorkUnit ID 和 artifact 引用追溯历史。

`CompactionSnapshot.deterministic_evidence`、`changed_files`、`test_results` 和
`user_constraints` 仍可作为不可变恢复数据持久化，但生成 ModelRequest 时由上述索引过滤；当前
Protected Context 存在对应 ID 时，Snapshot 不展示这些字段的实体内容。不同 ID 即使文本相同也不
自动合并，除非 Builder 的规范化规则能证明它们是同一实体。去重只影响展示，不删除 Store 数据。

为让假设去重可执行，Working Memory 的 active/rejected 假设在持久化层归一化为带
`hypothesis_id` 的记录。新的 `save_progress` 可以回传保护区中已有的 ID；兼容旧字符串输入时，Store
先按当前任务已有 Snapshot/Working Memory 的规范化文本和 provenance 查找原 ID，找不到才按
`task_id + 首次来源 ID + 规范化文本` 生成稳定 ID。同一假设从 active 迁移到 rejected/confirmed 时
复用该 ID，不能因状态变化创建重复实体。

## 7. 完整工作单元

### 7.1 模型

```python
class WorkUnit(BaseModel):
    unit_id: str
    purpose: str
    message_ids: list[str]
    tool_call_ids: list[str]
    state: Literal["complete", "incomplete", "ambiguous"]
    categories: set[Literal[
        "read", "search", "modify", "verify_pass", "verify_fail", "other"
    ]]
    start_index: int
    end_index: int
```

`message_ids` 按原顺序包含构成该工作单元的真实消息：

1. 操作或验证目的；优先取 Tool Call 所在 AIMessage 的文本。文本为空时只用 Tool 名称和经过现有
   脱敏/截断的参数生成 `purpose` 元数据，不虚构一个 message ID，也不让模型补写。
2. 一个包含一个或多个 Tool Call 的 AIMessage。
3. 与该 AIMessage 全部 Tool Call ID 配对的 ToolMessage；并行 Tool Call 属于同一单元，结果顺序
   不影响归属。
4. Tool 结果之后、下一个用户消息或下一个带 Tool Call 的 AIMessage 之前，连续出现的 Assistant
   解释消息。

### 7.2 边界规则

- 一个 AIMessage 中的全部并行 Tool Call 和其全部结果只能整体保留或整体压缩。
- 缺少任一结果的单元是 `incomplete`，必须保留。
- 存在重复 Tool Call ID、孤立 ToolMessage、结果跨越后续用户轮次或无法唯一配对时标记
  `ambiguous`，必须保留。
- 普通用户/助手对话形成独立 conversational unit；最近用户输入始终保留。
- 已由 Deep Agents 卸载的大型 ToolMessage 指针仍属于原工作单元，artifact 本身不复制进 prompt。
- WorkUnitPartitioner 是纯函数：输入消息列表，输出有序 units、未归属消息和诊断，不修改消息。

## 8. 自适应 Token 预算

### 8.1 预算口径

`ContextBudgetMonitor` 计算最终模型请求，而不是只计算 Conversation Messages：

```text
request_tokens =
    base_system_prompt
  + protected_context
  + effective_messages
  + tool_schemas

usable_input_tokens = model_context_tokens - reserved_output_tokens - safety_margin_tokens
usage_ratio = request_tokens / usable_input_tokens
```

`model_context_tokens` 由 `AppConfig` 的单一能力配置提供：显式环境配置优先，其次使用 DeepFix
按 model name 维护且受测试保护的能力表；两者都不存在时启动失败。该值同步到 Model profile 与
Monitor，且必须是正整数。不能让 Deep Agents 和 DeepFix 使用不同窗口值。`reserved_output_tokens`
和安全余量也必须显式配置并进入测试，不能隐含在不同组件中。

### 8.2 区间与边界

边界采用无重叠定义：

| usage_ratio | 区域 | 行为 |
|---|---|---|
| `<= 0.75` | 正常区 | 正常运行 |
| `> 0.75 and <= 0.82` | 观察区 | 不自动压缩；Working Memory 过旧时注入一次保存提示 |
| `> 0.82 and <= 0.90` | 正常压缩区 | 执行事务式压缩，目标回到 `<= 0.75` |
| `> 0.90` | 紧急压缩区 | 事务式压缩，只保留必保单元并以 `<= 0.65` 为目标 |

Working Memory “过旧”采用覆盖关系而非时间猜测。`save_progress` 由 Runtime 自动记录
`SnapshotCoverage(last_user_message_id, covered_tool_call_ids)`；当最新用户消息或最新完成工作单元的
Tool Call ID 未被 coverage 覆盖时，Working Memory 过旧。旧版本没有 coverage 时视为过旧。

观察区提示对同一 `(working_memory_version, latest_work_unit_id)` 只出现一次，避免每次 Model Call
重复消耗上下文。

### 8.3 保留选择

压缩先放入结构化 Snapshot，再在剩余 token 预算内按以下优先级整体选择 WorkUnit：

1. 最近用户输入。
2. 全部 `incomplete` 或 `ambiguous` 工作单元。
3. 最近修改及其验证单元。
4. 最近失败测试及其 Assistant 分析。
5. 其他完整工作单元，按新到旧填充。

同一优先级按新近程度排序。任何单元只有整体放入或整体转入 Snapshot 两种结果。若单个必保单元已经
超过预算，先复用大型 ToolMessage 卸载把正文变为 artifact 指针；仍超限则进入 Overflow 最小安全
流程，不能切掉半个单元。

## 9. 结构化 CompactionSnapshot

### 9.1 顶层模型

```python
class CompactionSnapshot(BaseModel):
    task_id: str
    version: int
    previous_version: int | None
    created_at: str
    source_work_unit_ids: list[str]
    task_goal: str
    user_constraints: list[UserConstraint]
    confirmed_facts: list[ProvenancedClaim]
    deterministic_evidence: DeterministicEvidenceBlock
    active_hypotheses: list[HypothesisRecord]
    rejected_hypotheses: list[HypothesisRecord]
    confirmed_hypotheses: list[HypothesisRecord]
    changed_files: list[FileChangeEvidence]
    experiments: list[ExperimentRecord]
    test_results: list[SystemTestEvidence]
    conflicts: list[ConflictRecord]
    unresolved_questions: list[ProvenancedText]
    next_steps: list[ProvenancedText]
    artifact_references: list[ArtifactReference]
    content_hash: str
```

必需的嵌套类型定义如下：

```python
class ProvenanceRef(BaseModel):
    kind: Literal[
        "user_message", "work_unit", "working_memory", "system_evidence",
        "artifact", "snapshot_record"
    ]
    ref_id: str

class ProvenancedText(BaseModel):
    text: str
    sources: list[ProvenanceRef]

class ProvenancedClaim(ProvenancedText):
    state: Literal["confirmed", "conflict"]

class HypothesisRecord(BaseModel):
    hypothesis_id: str
    text: str
    state: Literal["active", "rejected", "confirmed"]
    reason: str | None
    sources: list[ProvenanceRef]
    updated_in_version: int

class ExperimentRecord(BaseModel):
    experiment_id: str
    purpose: str
    action: str
    result: str
    sources: list[ProvenanceRef]

class ConflictRecord(BaseModel):
    conflict_id: str
    information_type: Literal[
        "user_constraint", "test", "file_operation", "approval",
        "research_status", "semantic"
    ]
    subject: str
    alternatives: list[ProvenancedText]
    source_of_truth: ProvenanceRef | None
    resolution: Literal["source_of_truth_applied", "unresolved_semantic"]

class ArtifactReference(BaseModel):
    path: str
    kind: Literal["conversation_history", "large_tool_result", "research", "snapshot_detail"]
    content_hash: str
    work_unit_ids: list[str]
```

Delta 模型唯一允许返回的增量类型也被完整约束：

```python
class UserConstraintCandidate(BaseModel):
    text: str
    source_user_message_id: str
    requested_state: Literal["active", "revoked"]
    supersedes_constraint_id: str | None

class HypothesisTransition(BaseModel):
    hypothesis_id: str | None
    text: str
    target_state: Literal["active", "rejected", "confirmed"]
    reason: str | None
    sources: list[ProvenanceRef]

class ConflictCandidate(BaseModel):
    information_type: Literal[
        "user_constraint", "test", "file_operation", "approval",
        "research_status", "semantic"
    ]
    subject: str
    alternatives: list[ProvenancedText]

class CompactionDelta(BaseModel):
    user_constraint_candidates: list[UserConstraintCandidate]
    confirmed_fact_candidates: list[ProvenancedClaim]
    hypothesis_transitions: list[HypothesisTransition]
    experiments: list[ExperimentRecord]
    conflict_candidates: list[ConflictCandidate]
    unresolved_questions: list[ProvenancedText]
    next_steps: list[ProvenancedText]
```

`changed_files` 和 `test_results` 是 `deterministic_evidence` 的规范投影，方便模型消费；Builder 必须从
系统证据生成，不能接受模型提供的另一份值。`content_hash` 对除自身外的规范 JSON 计算，用于写后
校验和重复请求幂等判断。

### 9.2 合并输入与规则

后续压缩只允许使用：

```text
上一版结构化 Snapshot
+ 本次进入压缩的完整 WorkUnit
+ 最新 Working Memory
+ 当前系统确定性证据
+ 当前 Task Anchor
```

模型只生成受限的 `CompactionDelta`，用于提取新工作单元里的候选约束、候选事实、实验、假设变化、
冲突、未解决问题和下一步。`user_constraint_candidates` 只有在文本可逐字定位到指定用户消息、状态变化
符合用户来源规则后，才由 Builder 转成规范 `UserConstraint`。模型不能直接生成或覆盖规范用户约束，
也不能生成 task ID、系统证据、changed_files、test_results、版本号、artifact 路径或哈希。

确定性合并规则：

- `task_goal` 始终来自 TaskState 原始问题。
- 用户约束只能由带来源的用户消息新增、撤销或替换。
- tests、file changes 和 approvals 每次从 deterministic evidence ledger 重建；external verification
  每次从 ResearchEvidenceStore 重建；各自覆盖 Snapshot 中对应类型的旧投影。
- 假设允许 `active -> rejected` 或 `active -> confirmed`。被排除假设重新出现时创建新 ID，并通过
  provenance 指向旧记录，不能删除原排除原因。
- 同一规范事实按稳定 ID 去重；文本相同但来源增加时合并来源。
- 新信息与用户/系统权威来源冲突时添加 `ConflictRecord`，按 `information_type` 查找该类型的
  `source_of_truth`；纯语义冲突没有确定来源时保持 `unresolved_semantic`。
- 旧的详细消息转为 artifact 引用，但 Snapshot 中保留足以判断结论和定位来源的结构化信息。
- Builder 完成 Pydantic 校验、任务 ID 校验、权威字段校验、关键字段指纹校验后才能持久化。

### 9.3 Snapshot 存储与模型可见性

新增 `compaction_snapshots(task_id, version, payload, input_hash, created_at)`，复用现有 SQLite WAL、
busy timeout 和任务复合主键模式。版本按任务单调递增。

模型请求中的旧自然语言摘要被一个有界的结构化 Snapshot 消息替代：

```xml
<deepfix_compaction_snapshot version="7" hash="...">
  ...有界结构化字段与 artifact 引用...
</deepfix_compaction_snapshot>
```

下一次压缩从 Store 读取 version 7，不解析这条消息，也不要求模型重新总结它。保护区仍在 System
Message 动态注入，因此 Snapshot 消息即使为了预算缩短，也不能覆盖 Task Anchor 或系统证据。

## 10. 压缩事务与失败保护

### 10.1 自动压缩和主动压缩的统一顺序

每次压缩生成稳定 `compaction_event_id` 和输入哈希，严格执行：

1. WorkUnitPartitioner 计算待压缩单元和待保留单元，不修改消息。
2. 使用 Deep Agents history/media 能力，把所有待压缩单元的完整消息追加到当前
   `conversation_history/{session_id}.md`，写入 event ID、消息 ID 和内容哈希。
3. 检查 Backend 写结果，并回读事件尾部验证 event ID 和内容哈希。未确认成功则停止。
4. 生成 `CompactionDelta`，由 Builder 与旧 Snapshot、Working Memory、Task Anchor、系统证据合并。
5. 完成 schema、task ID、用户约束、确定性证据、工作单元覆盖和 artifact 引用校验。
6. 在 SQLite 事务中写入新 Snapshot，再回读校验 version 和 `content_hash`。
7. 构造 `snapshot message + 完整保留单元` 的请求视图。
8. 最后提交新的 DeepFix compaction event；只有该 event 被 Graph state 接受后，旧消息才不再进入
   后续有效上下文。

步骤 1—6 不删除或替换任何旧消息。步骤 7 的模型调用失败时不提交 event；已写 artifact/Snapshot
是可复用的安全准备记录，而不是当前生效版本。event 保存 `active_snapshot_version`，系统只能使用
event 指向的 Snapshot，不能把未提交的“最新行”误当成生效版本。

这里“完整 Conversation Artifact”指所有即将退出有效上下文的消息逐条、无摘要地写入 history；
仍被保留的消息继续存在于 Graph checkpoint。history event 另外保存保留单元的 ID/哈希清单，因此
`history + checkpoint` 可以重建压缩前的完整有效会话，而不是只留下模型摘要。

主动 `compact_conversation` 返回成功 ToolMessage 前同样必须完成 1—6；尚未达到压缩条件时仍返回
“不需要压缩”的正常 ToolMessage，事务开始后的失败则抛出类型化协调异常，不更新 compaction event。
自动与主动入口以同一输入哈希实现幂等，重试不能重复追加同一 history 事件。

### 10.2 Artifact 或 Snapshot 失败

- 一旦自动或主动压缩事务已经开始，artifact、Delta、Snapshot 或写后校验失败都停止本次模型调用，
  保留原消息并抛出类型化协调异常；不能在失败后静默改用未压缩请求继续。
- 异常携带失败阶段、usage ratio、原消息是否保持、Working Memory 版本、已确认 artifact 和准备好的
  Snapshot 版本。BugfixService 捕获后把任务转为 `PAUSED` 并持久化恢复元数据。
- `usage_ratio > 0.90` 仍属于同一传播路径，只是异常代码标记为 `emergency_compaction_failed`，便于报告
  说明任务已经接近上限。
- 没有成功 artifact/Snapshot 时不得通过删除旧消息构造最小上下文。

### 10.3 ContextOverflow 最小安全重试

任意区域第一次捕获 `ContextOverflowError` 时：

1. 增加 overflow 指标并检查本次调用的 `overflow_retry_attempted` 标记。
2. 确认本轮完整 conversation artifact 和 Snapshot 已按事务顺序成功。
3. 构造最小安全上下文，目标 `<= 0.50`：三个保护块、当前结构化 Snapshot、最近用户输入、所有未完成
   或 ambiguous 工作单元，以及恢复 artifact 引用。
4. 调用同一个模型 handler 一次，并设置 `overflow_retry_attempted=True`。

第二次 ContextOverflow 不再压缩或重试，以原始 Overflow 为 cause，向 Service 抛出带恢复元数据的
`ContextRecoveryRequired`。Service 按当前任务暂停语义持久化任务，
保留 checkpoint、Snapshot 版本和 artifact 路径。一次 `agent.invoke` 最多发生一次 Overflow 重试。

### 10.4 类型化异常与业务状态边界

```python
class ContextRecoveryMetadata(BaseModel):
    task_id: str
    stage: Literal[
        "protected_context", "artifact_write", "artifact_verify",
        "delta_generation", "snapshot_validate", "snapshot_write",
        "snapshot_verify", "compacted_model_call", "overflow_retry"
    ]
    error_code: str
    usage_ratio: float | None
    working_memory_version: int | None
    active_snapshot_version: int | None
    prepared_snapshot_version: int | None
    conversation_artifact: str | None
    original_messages_preserved: bool

class ContextCoordinationError(RuntimeError):
    recovery: ContextRecoveryMetadata

class ProtectedContextLoadError(ContextCoordinationError): ...
class ArtifactPersistenceError(ContextCoordinationError): ...
class SnapshotBuildError(ContextCoordinationError): ...
class SnapshotPersistenceError(ContextCoordinationError): ...
class ContextRecoveryRequired(ContextCoordinationError): ...
```

Middleware/Coordinator 可以写自己的 artifact、Snapshot、ledger 和技术指标，但不能导入
`TaskStatus`、调用 `TaskRepository.save()`、调用 `BugfixService._pause()`，也不能直接修改 TaskState
业务状态。它只返回正常 ModelResponse/Tool 结果，或抛出上述类型化异常。

`BugfixService._invoke()` 是唯一业务状态边界：捕获 `ContextCoordinationError`，验证异常 task ID 与
当前任务一致，把经过长度限制且不含原始敏感内容的 recovery 元数据复制到 TaskState，然后调用统一
pause 流程。任意协调异常都转为 `PAUSED`，而不是 `FAILED`。非协调层的一般模型/Tool 异常仍沿用现有
错误策略，不被此规则吞掉。

## 11. Deep Agents 复用边界

继续复用：

- Deep Agents 的 Backend/CompositeBackend 和 Artifact 路由。
- `conversation_history` 的消息序列化格式、媒体引用与 history 文件组织。
- FilesystemMiddleware 的大型 ToolMessage、结果文件和媒体卸载。
- `compact_conversation` 的 Tool 名称、参数外观、Agent 主动触发方式和“不需要压缩”反馈。
- Agent 已构造的同一个模型实例；不创建第二个配置不同的摘要模型。

由 DeepFix 独立负责：

- WorkUnit 分区、保留优先级、自适应预算和 cutoff。
- 使用同一模型实例生成 `CompactionDelta`。
- `CompactionDelta` 的 Pydantic 校验、来源验证和安全字段白名单。
- 旧 Snapshot、Delta、Working Memory 与分类型权威来源的确定性合并。
- Snapshot 版本化、写后校验、去重投影和 compaction event 提交。
- Artifact/Snapshot 失败保护、最小安全上下文和单次 Overflow 重试。

DeepFix 正常运行路径不调用 `_create_summary`、`_acreate_summary`、`_determine_cutoff_index`、
`_lc_helper` 等 Deep Agents/LangChain 私有摘要方法，也不读写其私有 event。DeepFix 使用自己的
`_deepfix_compaction_event` 状态模型。

Adapter 首选公开 API。若当前 0.7 版本确实没有公开的 history 序列化或大型结果卸载入口，允许 Adapter
为这些非摘要能力保留最小兼容调用，但必须逐项列入契约测试和升级检查；该例外不扩展到 Delta 生成、
Snapshot 合并、cutoff 或事件提交。唯一额外例外是旧任务迁移器可把 checkpoint 中已经序列化的
`_summarization_event` 当作只读 legacy 数据；它不能调用对应私有方法，也不能在迁移后继续写该 key。
其他 DeepFix 模块不能访问任何 Deep Agents 私有成员。

## 12. Middleware 顺序

逻辑顺序固定为：

1. `PromptPolicyMiddleware` 生成核心规则和当前阶段策略。
2. `ProtectedContextMiddleware` 读取当前任务的 Anchor、Working Memory、系统证据并构造最终 System
   Message。
3. 协调层用“最终 System Message + messages + tools”计算预算，必要时事务式压缩。
4. 模型收到保护区、当前结构化 Snapshot 和完整保留 WorkUnit。
5. HITL 与 Tool 执行行为保持现有顺序和权限。

实现时必须通过调用链测试验证真实 wrapper 顺序，不能只断言 middleware 列表中的类名顺序。

## 13. 持久化、迁移与指标

### 13.1 向后兼容

- 新表和字段采用加法迁移，不删除现有 `working_memory`、`context_metrics` 或任务 payload。新增
  `deterministic_evidence(task_id, evidence_id, kind, payload, created_at)`，使用任务复合主键和幂等
  evidence ID；CompactionSnapshot 只读取该 ledger，不能成为其反向写入者。
- TaskState 增加可空 `context_recovery: ContextRecoveryMetadata | None`。只有 BugfixService 在捕获
  类型化协调异常时写入；Middleware、Coordinator 和 Snapshot Builder 均没有该字段的写权限。
- 旧 Working Memory 没有 `SnapshotCoverage` 时仍可注入，但在观察区视为过旧。
- 旧 TaskState 用户消息生成稳定 ID；旧 changed_files 标记为 `approved_target`；旧测试结果保留原退出码
  并使用稳定 legacy provenance。
- 已有 Deep Agents 自然语言 `_summarization_event` 作为 `legacy_summary` 非权威输入处理：先写入
  history artifact 并建立引用，再由第一次结构化 Snapshot 吸收可验证内容。它不能覆盖用户约束或
  系统证据。
- 恢复旧任务时不要求一次性重写全部历史；首次需要压缩时按任务惰性迁移。

### 13.2 指标

`ContextMetrics` 增加：

- `latest_usage_ratio`
- `latest_budget_zone`
- `normal_compaction_count`
- `emergency_compaction_count`
- `compaction_failure_count`
- `overflow_retry_count`
- `active_compaction_snapshot_version`
- `last_compaction_artifact`
- `last_compaction_error`

指标来自协调器事件，不再通过匹配 `compact_conversation` ToolMessage 的英文文本判断是否压缩成功。
报告显示最近区域、Snapshot 版本、失败/重试次数和恢复引用，但不把指标当作任务完成证据。

## 14. 安全与任务隔离

- 所有 Store 查询必须同时使用 Runtime `thread_id` 和记录 task ID；不接受模型传入 task ID、版本号或
  artifact 路径。
- Task Anchor、Working Memory、系统证据、Snapshot 和 artifacts 在渲染前进行 XML 转义和长度预算。
- Tool 参数中的密钥或大文本继续依赖现有截断/卸载；Snapshot 不复制秘密，只保存安全目的和内部
  artifact 引用。
- Snapshot 模型输出视为不可信输入，必须通过 schema、来源和权威字段白名单校验。
- protected context 只存在于当前 ModelRequest，不写回 Conversation Messages，避免每轮累积。

## 15. 错误处理

- 无法取得 task ID：Middleware 抛出 `ProtectedContextLoadError`；Service 捕获并暂停，不允许把该调用
  标记完成。
- Task Anchor 或确定性证据 Store 读取失败：抛出带恢复元数据的类型化异常，不能在缺少该类型权威
  来源时继续模型推理。
- Working Memory 缺失：合法，注入 `version=none`；读取异常则抛出 `ProtectedContextLoadError`，由
  Service 暂停。
- WorkUnit 边界模糊：整段保留，记录诊断；不能猜测切点。
- token 窗口能力缺失或非法：启动配置失败，不能退回固定消息数伪装百分比预算。
- history 写入或回读校验失败：旧消息保持有效，Coordinator 抛出 `ArtifactPersistenceError`。
- Delta 模型调用、schema 或合并校验失败：不写 Snapshot、不提交 event，抛出
  `SnapshotBuildError`。
- Snapshot SQLite 写入或回读校验失败：不提交 event，抛出 `SnapshotPersistenceError`。
- 模型处理 compacted request 失败：不提交 event；Coordinator 包装为带
  `stage=compacted_model_call` 的 `ContextCoordinationError`，准备好的 artifact/Snapshot 可按输入
  哈希复用。
- 第二次 ContextOverflow：不再重试，抛出 `ContextRecoveryRequired`。
- 以上异常均由 BugfixService 捕获并转成 `PAUSED`；Middleware/Coordinator 不修改 TaskState 状态。

## 16. 测试设计

默认测试不访问网络、不调用真实 DeepSeek。使用 Fake Model、真实 SQLite、受控 Backend 和真实
LangChain Message 对象。

### 16.1 保护区

- task-a 请求只注入 task-a 的 Anchor、Working Memory、测试、审批和外部证据。
- 所有 Working Memory 字段类别都出现，特别覆盖 `rejected_hypotheses`、`checked_files`、
  `experiments`。
- 超预算条目包含 omitted_count、哈希和可读引用，不发生静默丢失。
- 模型记忆声称 pytest 通过、但系统 exit_code 非零时，保护区只把系统结果作为当前事实并显示冲突。
- protected block 不进入 `request.messages` 或 Graph messages。
- 相同 constraint_id、evidence_id、hypothesis_id 在完整 ModelRequest 中各只出现一个规范实体；
  Snapshot 只贡献未重复的历史和来源引用。

### 16.2 WorkUnitPartitioner

- 单 Tool Call、配对 ToolMessage 和 Assistant 解释整体分组。
- 一个 AIMessage 的多个并行 Tool Call、乱序返回结果仍属于同一单元。
- 缺失结果、孤立 ToolMessage、重复 ID 和无法识别边界均标记 incomplete/ambiguous 并保留。
- cutoff 永远落在 WorkUnit 之间；属性测试随机生成消息序列，断言没有配对 ID 分居两侧。

### 16.3 自适应预算

- 精确覆盖 0.75、0.82、0.90 三个边界及其前后值。
- 观察区仅在 coverage 过旧时提示一次保存 Working Memory。
- 正常压缩回到不高于 0.75；紧急压缩只保留必保单元并回到不高于 0.65。
- 选择顺序依次验证最近用户、未完成单元、修改/验证、失败测试/分析、其他最近单元。
- 预算计算包含 System/保护区/messages/tools 和输出预留。

### 16.4 结构化 Snapshot

- 连续三次及以上压缩后，用户约束逐字不变、排除假设及原因不变、真实测试 command/exit_code 不变。
- tests/changed_files/approvals/external verification 无法被伪造 Delta 覆盖。
- active 假设能迁移到 rejected/confirmed；重新提出不会删除旧排除记录。
- 冲突信息生成 ConflictRecord，并按 information_type 使用对应 source_of_truth；纯语义冲突保持
  unresolved_semantic。
- 相同输入哈希重试不重复版本或 history 事件；不同输入产生单调新版本。
- 后续压缩读取 Store Snapshot，不把上一条 Snapshot 消息交给模型再次自然语言总结。

### 16.5 事务失败

- artifact write 返回错误、抛异常、回读缺少 event 或哈希不符时，compaction event 和有效消息不变。
- Delta/Snapshot 校验失败、SQLite 写入失败或回读哈希不符时，旧消息不删除。
- compacted model call 失败时，event 不生效，准备记录可复用。
- 主动 compact Tool 在事务开始后失败时抛出类型化异常，不报告 `Conversation compacted.`。
- 任一区域发生压缩准备失败时，Service 捕获异常并暂停；90% 以上使用 emergency 错误代码。
- Middleware/Coordinator 的测试断言它们从不写 TaskState/TaskRepository；Service 测试断言所有
  ContextCoordinationError 子类都转为 PAUSED 并保存恢复元数据。

### 16.6 Deep Agents 适配边界

- Adapter 契约覆盖 Backend/Artifact 路由、history 序列化、媒体/大型结果卸载和 Tool 外观。
- 把 Deep Agents 私有 summary、cutoff 和 event 方法替换成会立即失败的哨兵，完整压缩工作流仍通过，
  证明正常路径没有调用它们。
- Delta Builder 收到的 model 对象与 Repair Agent 是同一实例；结构化校验和 Snapshot 合并调用都发生
  在 DeepFix 模块。
- legacy `_summarization_event` 只被迁移器读取一次，迁移后的 state 只包含
  `_deepfix_compaction_event`。

### 16.7 Overflow

- 第一次 ContextOverflow 构造不高于 50% 的最小安全上下文并只重试一次。
- 第二次仍 Overflow 时 Service 暂停；一次 agent invocation 的 handler 调用总数为 2。
- artifact/Snapshot 未确认时不得删除消息执行最小重试。
- Overflow 最小上下文仍包含三个保护块、最新用户输入、未完成单元和 artifact 引用。

### 16.8 一次性 Bug 项目端到端测试

构造一个不访问网络的一次性 Python Bug 项目和长消息历史，包含：并行只读调查、失败 pytest、错误
假设及排除原因、文件修改审批、成功修改、通过 pytest、外部证据状态和至少三轮压缩。端到端断言：

- 目标项目的真实测试退出码贯穿 Snapshot、保护区、TaskState 和报告。
- 用户硬约束、排除原因、修改/验证工作单元未被拆分或漂移。
- conversation history 和大型结果只写 `DEEPFIX_HOME/artifacts`，目标项目不出现研究/压缩文件。
- 最终完成仍由 BugfixService 的本地通过测试门禁决定，而不是 Snapshot 或模型文本。

## 17. 建议模块边界

```text
src/deepfix/
├── context.py                 # save_progress 与 middleware 组装
├── protected_context.py       # Middleware、分类型来源读取与全局去重投影
├── compaction/
│   ├── models.py              # WorkUnit、Snapshot、Conflict、Budget 模型
│   ├── errors.py              # 类型化协调异常与恢复元数据
│   ├── budget.py              # ContextBudgetMonitor
│   ├── evidence.py            # EvidenceCollector
│   ├── work_units.py          # WorkUnitPartitioner
│   ├── snapshot.py            # Builder、确定性合并与 Store
│   ├── adapter.py             # Deep Agents Backend/Artifact/history/Tool 兼容边界
│   └── coordinator.py         # 自动/主动压缩事务与 Overflow 重试
├── memory.py                  # ProgressSnapshot coverage 与指标持久化
├── models.py                  # TaskState 确定性证据字段的兼容扩展
└── service.py                 # 捕获协调异常并独占 PAUSED 业务状态转换
```

模块之间使用上文模型和窄接口通信。`WorkUnitPartitioner`、`ContextBudgetMonitor`、Snapshot 合并器
保持纯函数优先；只有 Store、Backend、模型调用和 Service 边界拥有副作用。

## 18. 完成标准

- 不再存在固定 70% 触发、固定 15% 保留的 DeepFix 配置。
- 每次模型调用都有任务隔离的 Task Anchor、完整字段 Working Memory 和系统确定性证据保护块。
- 用户约束、测试/文件/审批和研究状态分别由对应权威 Store 决定，不存在跨类型全局优先级。
- 相同 constraint_id、evidence_id、hypothesis_id 在单次模型请求中只展示一个规范实体；Snapshot
  主要提供压缩历史和来源引用。
- 任何压缩切点都不能拆分单个或并行 Tool 工作单元。
- 三次以上压缩后用户约束、排除原因和真实测试结果保持一致并可追溯。
- artifact、Snapshot 或校验失败时旧消息保持有效；90% 以上安全暂停。
- ContextOverflow 只执行一次最小安全重试，第二次暂停并保留恢复信息。
- Deep Agents Backend/Artifact、history 序列化、大型 ToolMessage/media 卸载、Tool 外观和模型实例
  仍被复用；Delta、校验、合并和 event 提交由 DeepFix 负责。
- 所有协调失败以恢复元数据异常传播到 BugfixService；只有 Service 修改任务为 `PAUSED`。
- 原有单 Agent、HITL、研究证据隔离和本地测试完成门禁全部回归通过。
