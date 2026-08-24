# DeepFix 调查可靠性核心设计

日期：2026-08-24  
状态：待审阅  
适用范围：当前单 Repair Agent 的 Python Bug 调查与阶段推进流程

## 1. 背景

DeepFix 当前把修复 Agent 的阶段提示绑定到最新 Working Memory：如果没有 Working Memory，
`PromptPolicyMiddleware` 默认使用 `investigating`；如果模型没有主动调用 `save_progress`，阶段就不会
改变。Deep Agents 在一次 `agent.invoke()` 内部可以连续完成多轮 Model Call 和只读 Tool Call，
`BugfixService` 只能在 interrupt、最终结构化结果或异常返回时重新获得控制权。因此，Service 侧的
`max_agent_invocations` 和 Shell 调用计数无法发现内部反复 `read_file` 的调查循环。

QuixBugs 调试记录已经出现了这种失效模式：Agent 能获得 pytest 失败和大量源码，但没有保存工作记忆、
没有形成显式根因假设，也没有切换阶段，而是继续扫描文件。一次 BFS 调试轨迹包含 71 次模型请求、
69 次 `read_file`、1 次 `execute`、0 次 `save_progress`、0 次修改；阶段始终是
`investigating`，上下文从约 5 千字符增长到 11 万字符以上。

这不是单纯的提示词问题。系统缺少四项确定性能力：独立的推理阶段状态、内部 Tool 活动观测、调查
进展判定，以及停滞后的受控恢复。本设计新增 **DeepFix Investigation Coordinator Middleware**，
保留 Deep Agents 的内部循环和工具执行机制，但让 DeepFix 可以在每个 Tool Result 后更新调查状态、
控制阶段工具集，并在确定停滞时安全暂停任务。

## 2. 目标

- 将业务生命周期 `TaskStatus`、Agent 推理阶段 `AgentPhase` 和叙事型 Working Memory 明确分离。
- 在 Deep Agents 内部循环中观察每次工具结果，而不是只在 `agent.invoke()` 返回后统计。
- pytest 真实失败后自动进入诊断阶段；形成有证据支持的根因假设后进入规划阶段。
- 自动记录已检查文件、测试证据、工具签名、假设状态和有效进展，不依赖模型频繁调用
  `save_progress`。
- 用确定性停滞规则阻止重复读取、短周期扫描和无边界扩展，同时允许一次有理由的继续调查。
- 在禁止编辑的阶段隐藏修改工具；只有存在受支持假设时才允许进入规划和修改。
- 发生 Store 故障或二级停滞时保存恢复元数据，由 `BugfixService` 把任务转为 `PAUSED`。
- 保持审批、上下文压缩、研究证据、单 Agent 结构和 Deep Agents Backend/Tool 能力不变。

## 3. 非目标

- 本设计不实现 pytest 大输出的诊断摘要、Artifact 搜索或顺序读取控制。
- 本设计不实现 CLI 实时进度渲染，也不正式化 debug trace 的开关、脱敏、轮转和保留策略。
- 本设计不修改 QuixBugs 项目代码或为特定算法硬编码规则。
- 本设计不引入 Investigator 子 Agent、多 Agent 协作或独立长上下文任务。
- 本设计不判断一个假设在语义上绝对正确；系统只验证其来源、证据和可操作性。
- 本设计不以任意 Tool Call 数量代替进展，也不把所有新文件读取都视为有效调查。

上述能力分别属于后续的 “Diagnostic Artifact Retrieval” 和 “Progress Events + CLI Renderer”
子项目。本设计只提供它们可以订阅的结构化调查事件，但两项都是整体 reliability 修复的必交付项，
不是可选预留。只有三个子项目全部实施并通过联合端到端测试后，才能宣称 DeepFix reliability 修复完成。

## 4. 方案比较

### 4.1 方案 A：只加强提示词和 `save_progress`

要求模型在读到失败后主动保存阶段，并增加“不要重复读取”的提示。改动最小，但状态推进仍依赖模型
服从；模型不调用 `save_progress` 时系统仍看不到内部循环，也不能确定性限制工具。该方案不能解决已
观察到的失效模式，不采用。

### 4.2 方案 B：DeepFix Investigation Coordinator Middleware（采用）

在 Deep Agents 的 Model/Tool 循环边界增加 DeepFix 中间件适配器。协调器维护事件和物化状态，决定
推理阶段、工具可见性、进展代次和停滞级别；Deep Agents 继续负责 Agent 循环、Backend、工具执行、
interrupt 和 checkpoint。

该方案能观察内部 Tool Result，又不需要重写通用 Agent 执行器。所有判断逻辑放在纯 DeepFix 组件中，
中间件只负责把 Deep Agents 的请求与结果转换成领域事件，因此可以用离线单元测试覆盖。

### 4.3 方案 C：由 `BugfixService` 接管逐步执行

把每个模型调用和工具调用都提升为 Service 的一个显式 step。控制力最强，但会重写 Deep Agents 的
循环、interrupt 和 checkpoint 语义，影响范围远超当前问题，也降低基于 Deep Agents SDK 的项目价值。
不采用。

## 5. 状态权威与总体流程

### 5.1 三类状态

| 状态 | 权威来源 | 职责 | 不能做什么 |
|---|---|---|---|
| `TaskStatus` | `TaskRepository` / `BugfixService` | 业务生命周期、审批、暂停、完成 | 不驱动细粒度推理提示 |
| `AgentPhase` | `InvestigationStore` | 当前推理阶段和阶段工具门禁 | 不直接改变业务任务状态 |
| Working Memory | `WorkingMemoryStore` | 事实、假设、实验和下一步的语义快照 | 不覆盖 `AgentPhase` 或确定性调查事件 |

`AgentPhase` 定义为：

```python
class AgentPhase(StrEnum):
    CLARIFYING = "clarifying"
    INVESTIGATING = "investigating"
    DIAGNOSING = "diagnosing"
    PLANNING = "planning"
    EDITING = "editing"
    TESTING = "testing"
    REVIEWING = "reviewing"
```

三个状态允许暂时不同。例如，用户批准 pytest 后，`TaskStatus` 可以是 `TESTING`；pytest 返回失败后，
`AgentPhase` 是 `DIAGNOSING`；旧 Working Memory 仍可能写着 `investigating`。此时 Prompt Policy 必须
使用 `AgentPhase=diagnosing`，不能回退到 Working Memory。

`PAUSED` 是业务状态，不加入 `AgentPhase`。暂停时保存 `paused_agent_phase`，恢复后继续该阶段；若恢复时
收到新的用户信息，则把这条信息记为强进展并清除一次重评门禁。

### 5.2 数据流

```text
Model Request
    │
    ├── InvestigationStore 当前状态
    │       ├── AgentPhase
    │       ├── progress_generation
    │       └── stagnation gate
    │
    ├── InvestigationMiddleware
    │       ├── 注入 bounded investigation state
    │       ├── 按阶段过滤工具
    │       └── 执行前检查停滞许可
    │
    ▼
Deep Agents Model / Tool Loop
    │
    ▼
Tool Result
    │
    ├── 标准化为 InvestigationEvent
    ├── ProgressEvaluator 判定进展
    ├── PhaseResolver 推导阶段
    ├── StagnationDetector 更新门禁
    └── InvestigationStore 幂等提交事件与物化状态
```

`PromptPolicyMiddleware` 不再读取 `WorkingMemoryStore.snapshot.phase`，而是读取
`InvestigationStore.state.agent_phase`。Working Memory 中的 `phase` 字段暂时保留用于兼容和历史展示，
但渲染时明确标记为 snapshot metadata。

## 6. 组件边界

### 6.1 `InvestigationCoordinator`

纯领域协调器，不依赖 Model、CLI 或业务状态迁移。它接收当前状态和一个规范化命令/事件，依次调用：

1. `ProgressEvaluator`：判定是否产生强进展以及进展种类；
2. `PhaseResolver`：根据确定性事件和门禁推导下一 `AgentPhase`；
3. `StagnationDetector`：更新重复、周期、无进展和探索范围计数；
4. `InvestigationStore`：以一个事务提交事件和新物化状态。

协调器不得执行工具、修改 `TaskStatus`、生成自然语言诊断或读取任意 Artifact 正文。

### 6.2 `InvestigationMiddleware`

Deep Agents 适配层，职责仅为：

- 从 runtime `thread_id` 解析并校验 `task_id`；
- 在 Model Request 中注入当前调查状态；
- 根据 `AgentPhase` 和停滞门禁过滤可见工具；
- 在工具执行前验证重评许可；
- 在工具完成后生成 `tool_completed` 及其派生事件；
- 将结构化输入错误转换为 error `ToolMessage`；
- 将 Store/协调器恢复异常原样传播到 Service。

中间件不得直接调用 `TaskState.transition_to()`。它也不得吞掉真实 `ToolMessage`、改写 exit code、破坏
Tool Call/ToolMessage 配对或在事件提交失败后擅自重跑已执行工具。

### 6.3 `InvestigationStore`

独立、任务隔离、追加事件加物化状态的 SQLite Store。它自动维护调查进度，不要求模型调用
`save_progress`。事件是审计和幂等依据，物化状态是 Prompt 和门禁的快速当前投影。

Store 与 Working Memory 可以使用同一数据库文件，但拥有独立表和接口。`InvestigationStore` 是
`AgentPhase`、进展代次、停滞状态和工具观测的唯一权威；Working Memory 是模型整理的叙事记忆。

### 6.4 结构化工具

新增两个轻量 Tool：

- `record_hypothesis`：登记或更新一个带来源的根因假设，并请求系统判断它是否达到
  `supported` 门槛；
- `continue_investigation`：在一级停滞后说明尚未解决的问题、预期证据和下一次工具调用，申请一次
  调查许可。

`save_progress` 继续负责较完整的 Working Memory 快照，不再承担阶段切换的必要条件。模型可以在独立
调查阶段后调用它，但不调用也不会阻止系统记录文件检查、测试或阶段迁移。

## 7. 事件与物化状态

### 7.1 `InvestigationEvent`

```python
class InvestigationEvent(BaseModel):
    event_id: str
    task_id: str
    sequence: int
    event_type: InvestigationEventType
    source_message_id: str | None = None
    tool_call_id: str | None = None
    phase_before: AgentPhase
    phase_after: AgentPhase
    progress_kind: ProgressKind | None = None
    payload: dict[str, JsonValue]
    created_at: datetime
```

事件类型至少包括：

- `task_started`、`user_information_received`；
- `tool_completed`、`test_observed`、`file_checked`、`file_changed`；
- `hypothesis_recorded`、`hypothesis_rejected`、`hypothesis_supported`；
- `investigation_intent_recorded`、`phase_changed`；
- `reevaluation_required`、`investigation_permit_granted`、
  `investigation_stagnated`。

`event_id` 对可重放来源按 `task_id + event_type + tool_call_id + result_fingerprint` 确定性生成；没有
Tool Call 的命令使用规范化输入哈希。相同事件重复提交返回已有 sequence，不重复增加计数或推进阶段。

事件 payload 只保存有界事实、ID、路径、范围、exit code、哈希和 Artifact 引用。完整 stdout、源码、
prompt、API key 和环境变量不得写入 Investigation Store。

### 7.2 `InvestigationState`

```python
class InvestigationState(BaseModel):
    task_id: str
    version: int
    migration_version: int
    agent_phase: AgentPhase
    paused_agent_phase: AgentPhase | None
    checked_files: list[CheckedFile]
    recent_tool_signatures: list[ToolSignature]
    test_evidence_ids: list[str]
    hypothesis_ids: list[str]
    supported_hypothesis_ids: list[str]
    progress_generation: int
    last_progress_event_id: str | None
    last_progress_at: datetime | None
    reevaluation_required: bool
    stagnation_level: Literal[0, 1, 2]
    permit: InvestigationPermit | None
```

所有列表必须有固定上限；被移出的旧条目由事件 sequence 或已有 Artifact/证据 ID 追溯。工具签名包含
规范化 tool name、关键参数哈希和结果指纹，不保存大结果正文。

`CheckedFile` 至少记录规范路径、内容指纹、已检查行范围、scope 分类、首次/最近事件 ID。相同内容和
重叠范围幂等合并。文件内容变化只允许重新评估该范围；只有变化内容产生 decision-relevant evidence
时才形成强进展，指纹变化本身仍只是活动。

### 7.3 假设身份与状态

调查假设使用 Working Memory 已采用的稳定 `hypothesis_id` 规则；同一 ID 在 Protected Context 中只
展示一次。`record_hypothesis` 输入为：

```python
class RecordHypothesisInput(BaseModel):
    hypothesis_id: str | None = None
    statement: str
    evidence_ids: list[str]
    checked_locations: list[CheckedLocation]
    proposed_change: ProposedChange | None = None
    expected_effect: str | None = None
    target_state: Literal["candidate", "rejected", "supported"]
    reason: str
```

- 新假设不提供 ID，由系统根据任务、首次有效来源和规范化 statement 生成。
- 更新或拒绝已有假设必须提供已有 ID，不能依赖文本相似度。
- 引用的 evidence、文件、行范围必须属于当前任务且已存在于确定性 Store/Investigation Store。
- `rejected` 必须给出排除原因及支持来源。
- `supported` 至少需要一个有效证据 ID、一个已检查的相关位置、一个具体拟修改目标和预期效果。
- 满足这些条件只表示 **supported hypothesis**，不表示系统证明语义结论为真。

只有 `hypothesis_supported` 事件能打开 `DIAGNOSING → PLANNING`。模型在普通文本、最终输出或
`save_progress` 中声称“已找到根因”都不能绕过门禁。

Investigation Store 的 `candidate/rejected/supported` 是操作门禁状态，Working Memory 现有的
`active/rejected/confirmed` 是语义快照状态，两者不做有损的同名转换：candidate 和 supported 都投影为
Working Memory 的 active 假设，并额外显示 investigation support 状态；rejected 投影为 rejected。
Working Memory 的 confirmed 只表示模型语义总结，不能反向生成 supported 事件或打开编辑门禁。后续若
统一两个枚举，必须作为独立数据迁移设计，不能在本项目中静默把 supported 写成 confirmed。

## 8. 阶段迁移与工具门禁

### 8.1 确定性迁移

| 当前阶段 | 事件/条件 | 下一阶段 |
|---|---|---|
| 无状态 | 新任务开始 | `INVESTIGATING` |
| 任意非澄清阶段 | Agent 明确需要用户输入 | `CLARIFYING` |
| `CLARIFYING` | 收到用户补充 | 恢复进入澄清前阶段；缺失时为 `INVESTIGATING` |
| `INVESTIGATING` | 有效 pytest 结果且 exit code 非 0 | `DIAGNOSING` |
| `INVESTIGATING` | 没有基线测试、但有效证据指向具体位置 | 仍为 `INVESTIGATING` |
| `DIAGNOSING` | `hypothesis_supported` | `PLANNING` |
| `PLANNING` | 成功的 write/edit/delete Tool Result | `EDITING` |
| `EDITING` | 审批后真实开始执行、且被系统分类为 verification/test 的命令 | `TESTING` |
| `TESTING` | 修改后 pytest 失败 | `DIAGNOSING` |
| `TESTING` | 修改后 pytest 通过 | `REVIEWING` |
| `REVIEWING` | 新失败证据或用户要求继续 | `DIAGNOSING` 或 `INVESTIGATING` |

pytest 失败自动进入 `DIAGNOSING`，但绝不自动进入 `EDITING`。测试 pass 只有在存在成功文件修改事件且
测试发生在修改之后时，才能进入 `REVIEWING`；原始基线 pass 不是修复完成证据。

写操作执行失败不进入 `EDITING`，而是回到 `PLANNING` 并保留失败证据。审批只代表允许执行，不代表
工具成功；`TaskStatus` 可因审批暂时成为 `EDITING`/`TESTING`，`AgentPhase` 仍以实际事件推进。

`EDITING → TESTING` 不能由模型生成 execute Tool Call、Service 批准 execute 或任意 Shell 命令触发。
系统必须在审批恢复后观察到工具执行真正开始，并用确定性命令分类器确认它是当前项目的测试或验证命令。
安装依赖、查看版本、打印文件、启动交互 Shell 等普通 execute 保持 `EDITING`。如果执行开始后发生
异常，阶段仍可保留 `TESTING`，并记录 verification execution failure；只有配对 Tool Result 才能决定
随后进入 `DIAGNOSING`、`REVIEWING` 或继续 `TESTING`。

### 8.2 动态工具可见性

| 阶段/门禁 | 可见工具 |
|---|---|
| `INVESTIGATING` | read/search、`execute`、研究工具、`record_hypothesis`、`save_progress`、压缩工具 |
| `DIAGNOSING` | 同上，重点允许 `record_hypothesis`；不显示 write/edit/delete |
| `PLANNING` | 调查工具及 write/edit/delete；写操作仍经过既有审批 |
| `EDITING` | 修改工具、必要读取、`execute`、记忆/压缩工具 |
| `TESTING` | `execute`、必要读取、假设更新、记忆/压缩工具 |
| `REVIEWING` | 只读核对、测试、记忆/压缩工具 |
| 一级停滞 | 只显示 `record_hypothesis`、`continue_investigation`、`save_progress` |

隐藏修改工具是能力门禁，不只是 Prompt 建议。扩展 Tool 必须声明 `investigation_capability`；缺少声明
的扩展在 investigating/diagnosing 和停滞受限状态下不可见，不能因为“未知”就被推定为只读。进入
planning 后仍沿用扩展原有 interrupt/审批策略，调查协调层不放宽外部副作用权限。

### 8.3 Prompt 投影

每次 Model Call 注入有界块：

```xml
<deepfix_investigation_state>
  phase: diagnosing
  progress_generation: 4
  recent_progress: test_observed(evidence_id=...)
  checked_files: ...
  supported_hypotheses: ...
  reevaluation_required: false
</deepfix_investigation_state>
```

该块是 Investigation Store 当前投影，不写入 Conversation Messages。相同 `hypothesis_id` 和
`evidence_id` 已在 Protected Context 的 Working Memory 或 Deterministic Evidence 展示时，这里只显示
ID、状态和门禁关系，不重复正文。完整历史通过 Investigation event sequence 和 Artifact 引用追溯。

## 9. 调查进展模型

### 9.1 Tool Activity 不等于 Investigation Progress

每个 Tool Result 都产生活动事件，但只有新增、可复核且能改变调查决策的信息才增加
`progress_generation`。以下属于强进展：

- 新的有效测试证据，或同一测试在文件修改后产生新结果；
- 新的异常类型、失败断言或 traceback 位置，并且改变了待验证假设、调查范围或下一步决策；
- 从源码或搜索结果提取出的 decision-relevant evidence，例如支持/反驳现有假设、定位实际失败路径，
  或证明某个符号/分支与失败测试存在可验证关系；
- 假设从 candidate 变为 supported/rejected，或出现有新证据的重新开启；
- 成功的文件修改；
- 修改后的测试结果；
- 用户补充信息。

`phase_changed` 是上述原始进展事件的派生结果，始终设置 `progress_kind=None`，自身不能增加
`progress_generation` 或清除停滞门禁。一次原始事件即使同时引起阶段变化，也只能计一次强进展。

以下活动本身不增加进展代次：

- 对相同内容和相同行范围重复 `read_file`；
- 仅仅首次读取一个新文件、首次看到一个新行范围或发现一个尚未证明相关的新代码位置；
- 相同 grep/glob/ls 查询得到相同结果；
- 未修改代码时重复相同 pytest 并得到相同失败；
- 只改变分页 offset，但没有获得新的相关位置或证据；
- 没有关联假设/问题地继续读取不相关新文件；
- 空的或只改写措辞的 `save_progress`。

### 9.2 调查范围

文件/查询目标分为：

- `direct`：用户明确路径、失败测试文件、traceback 位置、被测符号定义、已支持假设的修改目标；
- `dependency`：通过可验证的 import/call/traceback/grep/符号引用等来源边关联到 direct 目标的文件；
- `exploratory`：没有上述可复核关联的其他目标。

分类由系统验证的来源边生成，不由模型自由声明。有效来源边至少包括：解析出的 import、静态或运行时
call 关系、traceback frame、grep/符号引用命中、测试收集关系，以及确定性证据直接指向的定义位置。
每条边必须保存 source、target、relation type 和产生它的 Tool Result/message ID。

`continue_investigation.reason` 只表达调查意图和 permit 理由。即使它引用现有 hypothesis、evidence 或
checked location，也不能单独把 exploratory 目标提升为 dependency。只有 Tool Result 或既有系统记录
提供上述可验证关系边后，Scope Classifier 才能改变分类；验证失败时仍按 exploratory 计数。

## 10. 停滞检测与恢复

### 10.1 一级停滞条件

所有条件都限定在同一 `progress_generation` 内，任一满足即进入 level 1：

1. **精确重复**：相同 tool、规范参数和结果指纹出现 3 次；
2. **短周期**：长度 2～8 的工具签名序列完整重复 2 次，期间没有强进展；
3. **无进展上限**：连续 6 个调查类 Tool Result 没有强进展；
4. **范围扩张**：连续检查 4 个互不相同的 exploratory 文件而没有强进展。

检测使用已完成 Tool Result，不因模型仅提出调用就误判。并行 Tool Call 按同一批次记录，但每个结果有
独立签名；批次顺序规范化后参与周期检测。

进入 level 1 时记录 `reevaluation_required` 事件，下一次 Model Request 只暴露三个元工具。模型必须
选择：登记/更新假设、保存有意义进度，或申请一次继续调查。

### 10.2 一次性继续调查许可

`continue_investigation` 输入为：

```python
class ContinueInvestigationInput(BaseModel):
    hypothesis_ids: list[str]
    unresolved_question: str
    expected_evidence: str
    tool_name: str
    target: str
    reason: str
```

所有 hypothesis ID 必须属于当前任务；tool/target 必须是调查类操作，且 reason 必须把目标与一个现有
假设、失败证据或未解决问题关联。校验成功后生成仅一次、绑定 tool name 和规范 target 的 permit。
其他工具不能消费该 permit。

许可工具返回后 permit 立即消费。系统给模型一个 Model Turn 用来调用 `record_hypothesis` 或形成其他
强进展。此时状态标记 `post_permit_review_pending`，只显示三个元工具；合法的 `record_hypothesis`
状态迁移可产生强进展并恢复正常。`save_progress` 可以在暂停前保存叙事记忆，但本身不清除门禁。若下一
个动作仍是普通调查，系统把状态升级为 level 2，不执行该工具，并抛出类型化
`InvestigationStagnationError`。

### 10.3 重置规则与误报保护

- 强进展增加 `progress_generation`，清空当前代次的重复/周期/无进展计数和 level 1 门禁。
- 文件内容指纹变化后允许重新读取相同范围且不算精确重复，但只有提取出 decision-relevant evidence
  才能成为强进展；指纹未变化则仍是重复。
- 成功编辑后允许重新运行此前失败测试，不计为重复测试。
- pytest 偶发失败只有在结果指纹变化时作为新证据；相同失败不能无限重置。
- 用户补充信息清除一次重评门禁，但不删除既有事件历史。
- `save_progress` 无论内容长短都不直接重置停滞；其中的新事实必须先通过确定性事件或
  `record_hypothesis` 的结构化来源验证才能成为强进展。纯措辞变化和无来源的新假设不能重置。
- 无法可靠规范化工具参数或结果时，不做精确重复判定，但仍计入无进展上限；系统不因解析不确定而
  伪造进展。

## 11. 中间件集成顺序

语义顺序固定为：

1. `MessageIdentityMiddleware`
2. `LegacyContextMigrationMiddleware`
3. `InvestigationMiddleware`
4. `PromptPolicyMiddleware`
5. `ProtectedContextMiddleware`
6. `DeepFixCompactionMiddleware`
7. Research/其他声明式扩展中间件
8. 可选 `LLMTraceMiddleware`（最外层观测最终请求，不参与业务逻辑）

这里的编号描述必须满足的可观察语义，而不是假设框架对列表的调用方向。实现计划阶段必须先用契约测试
验证 Deep Agents/LangChain 当前版本的 before/after/wrap 嵌套顺序，再排列实际 middleware 列表。

Investigation 需要先确定 phase 和工具门禁；Prompt Policy 才能选择阶段提示；Protected Context 和
Compaction 随后投影权威信息及处理预算。Trace 若启用，只记录最终可见请求，并遵循后续 debug 子项目
定义的脱敏策略。

## 12. 错误传播与恢复

### 12.1 类型化异常

```python
class InvestigationCoordinationError(Exception):
    recovery: InvestigationRecoveryMetadata

class InvestigationStateError(InvestigationCoordinationError): ...
class InvestigationStagnationError(InvestigationCoordinationError): ...
```

恢复元数据至少包含 `task_id`、`error_code`、`agent_phase`、`state_version`、最后成功 event sequence、
触发 tool_call_id、permit 状态、checkpoint 可用性和安全的恢复建议。不得包含完整 Tool 输出或 secret。

### 12.2 失败边界

- 读取当前 Investigation State 失败：禁止 Model Call，抛出 `InvestigationStateError`。
- 工具执行前的 event/permit 校验失败：不执行工具；结构化输入错误返回 error `ToolMessage`，Store 故障
  抛类型化异常。
- 工具尚未执行且提交失败：可以在同一 checkpoint 恢复后重试。
- 工具已执行但结果事件提交失败：绝不自动重跑工具；保留原 ToolMessage 和 checkpoint，恢复时按稳定
  event ID 幂等重放提交。
- level 2 再次调查：不执行目标工具，抛 `InvestigationStagnationError`。

`BugfixService` 捕获 `InvestigationCoordinationError`，核对 recovery.task_id 后把恢复信息保存到任务并
转为 `PAUSED`。中间件和 Coordinator 不直接修改 `TaskStatus`。如果异常携带其他 task_id，Service 按
任务隔离破坏处理为失败，而不是保存到当前任务。

Service 在成功修改 TaskStatus 后，通过 Coordinator 的公开 lifecycle command 记录 `task_paused`、
`task_resumed` 或 `user_information_received`；这些命令只更新 Investigation State 中的阶段恢复元数据，
不会由 Coordinator 反向修改业务状态。若 lifecycle event 提交失败，Service 保留已保存的业务状态和
恢复信息，任务维持 PAUSED，恢复时按稳定 event ID 补交。

## 13. 旧任务迁移

首次加载没有 Investigation State 的旧任务时，迁移器按原顺序读取：

1. `TaskState` 业务记录；
2. Graph 中具有稳定 ID 的配对 AIMessage/ToolMessage；
3. 确定性测试、文件和审批证据；
4. Working Memory 仅作为候选语义来源。

阶段推导规则为：

- 无工具证据：`INVESTIGATING`；
- 最近有效 pytest 失败，之后无成功修改：`DIAGNOSING`；
- 失败后存在成功修改，但没有修改后真实 verification/test execution：`EDITING`；
- 修改后存在已开始但尚无配对结果的 verification/test execution：`TESTING`；
- 最近修改后 pytest pass：`REVIEWING`；
- 当前业务状态是 `CLARIFYING`：`CLARIFYING`，并保存推导出的前一阶段。

Working Memory 的 phase 不参与推导。旧假设在 provenance 能映射到当前任务的 message、evidence 或
checked location 时可以导入 candidate；只有同时满足第 7.3 节全部 supported 条件，包括具体修改目标和
预期效果，才导入 supported。无法完整验证的只保留为 candidate 或 Working Memory 历史，不打开编辑
门禁。

迁移保存 `migration_version`，事件 ID 确定性生成。重复启动、checkpoint replay 或迁移中断重试不得
重复事件、重复计数或改变已物化阶段。已有较新 Investigation State 时迁移器直接跳过。

## 14. 测试设计

### 14.1 状态权威

- 没有 Working Memory 时，新任务从 `INVESTIGATING` 开始。
- pytest 失败把 AgentPhase 切到 `DIAGNOSING`，Prompt 使用诊断提示。
- phase_changed 事件自身不增加 progress_generation，也不重置停滞。
- Working Memory phase 过旧或冲突时不能覆盖 AgentPhase。
- TaskStatus 与 AgentPhase 不一致时，各自消费者读取正确权威源。
- 暂停/恢复保留推理阶段；用户补充形成强进展。

### 14.2 假设和编辑门禁

- 无来源、跨任务来源、无 checked location 或无拟修改目标的 supported 请求被拒绝。
- 合法 `record_hypothesis` 生成稳定 ID，并从 diagnosing 进入 planning。
- 普通文本、最终输出或 `save_progress` 不能绕过 supported hypothesis 门禁。
- planning 前 write/edit/delete 不可见；planning 后仍执行原审批策略。
- write ToolMessage 失败不进入 editing；修改后测试 pass 才进入 reviewing。

### 14.3 调查观测与幂等

- read_file/grep/glob/ls/execute 的结果在 Agent 内部循环中被自动记录。
- 相同 tool_call/result replay 只生成一个事件，不重复 progress_generation。
- 并行 Tool Calls 生成独立结果事件和规范化批次签名。
- checked ranges 合并正确；内容改变后的重读不会因指纹变化自动产生进展，只有新决策证据才产生。
- 首次读取新文件/新位置只记录活动；只有提取 decision-relevant evidence 才增加进展代次。
- dependency 只由带来源 ID 的 import/call/traceback/grep 等可验证关系边产生，模型 reason 不能提升。
- Store 不保存大 stdout、源码正文或 secret。

### 14.4 停滞检测

- 精确重复 3 次进入 level 1。
- 长度 2、8 的短周期重复两次触发；长度外或中间有强进展不触发。
- 连续 6 次无进展和 4 个 exploratory 文件分别触发。
- gcd 式重复测试/读文件和 BFS 式全库扫描在达到大上下文前被重评门禁截断。
- 有效 `continue_investigation` 只放行绑定的一次工具；错误目标不能消费许可。
- 许可后有强进展恢复正常；无进展后再次调查抛类型化异常。
- direct/dependency 的合理多文件调查、修改后重测和有新决策证据的内容变化重读不产生误报。
- 反复切换 phase 不能重置重复、周期或无进展计数。

### 14.5 正常端到端流程

离线 Fake Model/Fake Tool 流程覆盖：

```text
用户问题
→ 基线 pytest fail
→ diagnosing
→ 读取直接相关源码与测试
→ record_hypothesis(supported)
→ planning
→ 审批并最小修改
→ 审批验证命令但 phase 仍为 editing
→ verification/test 真实开始执行后进入 testing
→ pytest pass
→ reviewing / completed
```

断言 TaskStatus、AgentPhase、事件序列、工具可见性、审批记录、真实 exit code 和最终报告一致。

### 14.6 故障保护与迁移

- State 读取失败阻止 Model Call，Service 保存恢复信息并暂停。
- 工具执行后 event commit 失败不重跑工具，恢复后幂等补交。
- stagnation recovery 的 task_id 不匹配时任务隔离失败关闭。
- 旧任务的四种阶段推导、重复迁移、迁移中断恢复均有测试。
- 与现有 Protected Context、Compaction、Research 和 Service 测试联合运行，确保消息配对和
  ContextOverflow 恢复不回归。

## 15. 验收标准

1. `PromptPolicyMiddleware` 不再依赖 `save_progress` 的 phase 决定当前推理提示。
2. Investigation Middleware 能观察一次 `agent.invoke()` 内部的只读 Tool 循环。
3. 已检查文件和测试证据由系统自动记录，不要求模型主动保存 Working Memory。
4. 有效 pytest fail 自动进入 `DIAGNOSING`，但不能直接获得编辑能力。
5. 只有带当前任务有效证据的 supported hypothesis 能进入 `PLANNING` 并显示修改工具。
6. 精确重复、短周期、无进展和无边界探索能在既定阈值触发重评。
7. 模型只有一次结构化继续调查许可；持续无进展会以可恢复方式暂停，而不是无限增长上下文。
8. 正常的基线失败、调查、最小修改、修改后验证流程不被误阻断。
9. 事件提交和旧任务迁移可重放、幂等、任务隔离，Store 不持久化大结果或敏感正文。
10. phase_changed、新文件和新代码位置不会自行重置停滞；dependency 只能来自系统可验证关系边。
11. `EDITING → TESTING` 只由真实开始的 verification/test execution 触发，审批和普通 execute 均无效。
12. 新增定向测试、现有完整离线测试和 Ruff 全部通过。

## 16. Reliability 项目交付门禁与后续接口

整体 reliability 修复由三个有顺序依赖的必交付子项目组成：

1. **Investigation Reliability Core**：本设计，提供阶段权威、调查事件、进展判定和停滞门禁；
2. **Diagnostic Artifact Retrieval**：消费测试事件和 Artifact ID，提供有界诊断摘要、按需定位和防止
   大输出分页重新灌满上下文的读取策略；
3. **Progress Events + CLI Renderer**：消费结构化进度事件，显示当前 phase、最近有效进展、正在执行的
   工具、重评/暂停原因，并把 LLM trace 降为可选调试观测。

每个子项目分别执行 spec → plan → implementation → review，不能在本核心实施计划中混合实现。第二项
依赖第一项的 test/artifact 事件，第三项依赖第一项的 phase/progress/stagnation 事件；第一项可以先独立
落地，但不能据此关闭整体 reliability 修复任务。

本核心承诺稳定输出：`InvestigationEvent`、`InvestigationState`、阶段变化、进展变化、停滞事件和
Artifact/证据 ID。后续项目只读订阅这些接口：

- Diagnostic Artifact Retrieval 用 `test_observed` 和 Artifact ID 构造有界 pytest 诊断视图；
- Progress Events + CLI Renderer 把 phase/progress/stagnation 事件渲染为终端进度；
- debug trace 只作为可选观测器，不再承担判断 Agent 是否循环的业务职责。

后续子项目不得反向修改 `AgentPhase`、progress_generation 或停滞门禁；它们需要改变调查状态时，必须
通过 Coordinator 的公开命令生成事件。

整体完成门禁为：三个子项目均已实施；跨子项目端到端测试证明大型 pytest 失败可被 Agent 定位读取、
循环会被核心门禁阻止、终端能显示真实阶段和停滞原因；现有完整测试和 Ruff 通过。在此之前，报告只能
声明某个子项目完成，不能声明 “DeepFix reliability 修复完成”。
