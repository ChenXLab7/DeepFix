# DeepFix 诊断 Artifact 检索设计

日期：2026-08-24

状态：已批准（2026-08-24）
适用范围：单 Repair Agent 对超长 Tool 结果和压缩历史的只读诊断检索

## 1. 背景

DeepFix 已复用 Deep Agents 的大型 ToolMessage 自动卸载能力，并通过 DeepFix Compaction 保存可恢复的
`conversation_history` Artifact。大型 pytest 输出、traceback 和 Shell 结果超过阈值后会写入
`/.deepfix-artifacts/large_tool_results/{sanitized_tool_call_id}`，模型只保留头尾预览和文件引用；压缩前的
完整 Graph Messages 会写入 `/.deepfix-artifacts/conversation_history/{task_id}.md`。

现有通用 `grep` 和 `read_file` 从文本能力上可以读取这些文件，但它们接受路径，不能证明某个路径属于
当前任务，也不能把源码搜索与诊断历史检索区分开。模型如果直接扫描 Artifact 根目录，可能看到其他任务、
研究正文或 Tool 执行回执。当前 Investigation Coordinator 也无法记录模型检索了哪个旧证据范围。

本设计增加一个轻量的 **Diagnostic Artifact Retrieval** 层。它不建立持久化搜索索引，不引入向量数据库，
也不拦截 Deep Agents 的私有卸载方法；每次调用都从当前任务的 Graph Messages 和 CompactionStore 重新
构造允许访问的 Artifact 目录，然后通过现有 Backend 做受限文本搜索或按行读取。

## 2. 目标

- 让 Agent 能重新搜索超长 pytest、traceback、Shell 输出和压缩前历史。
- 复用普通文本匹配能力，但不向模型开放 Artifact 根目录或任意 Backend 路径。
- 只从当前任务的真实 ToolMessage 和生效 Compaction Snapshot 接受 Artifact 归属。
- 使用稳定 `artifact_id` 代替路径作为公开 Tool 输入。
- 返回带行号、范围、内容哈希和截断状态的有界结果。
- 把成功检索记录为 Investigation 事件，但不把重新读取旧信息计算为 strong progress。
- 复用现有 `InvestigationStateError` 和 BugfixService PAUSED 恢复边界，但由 Tool 适配层补齐恢复元数据，
  底层检索服务不修改 Investigation 或任务状态。
- 保持同步离线测试，不访问网络，不修改目标项目。

## 3. 非目标

- 不搜索 `research`、`tool_execution_receipts`、媒体、Snapshot detail 或任意用户文件。
- 不实现正则表达式、模糊搜索、全文搜索引擎、embedding、reranker 或语义摘要模型。
- 不新增 Artifact Registry 数据库表、后台索引任务或 Artifact 生命周期状态机。
- 不修改 Deep Agents 的卸载路径、阈值、预览格式或私有卸载函数。
- 不把 Artifact 片段提升为新的 pytest、文件修改、审批或研究确定性证据。
- 不实现 CLI 实时进度展示；该能力属于后续 Progress Events + CLI Renderer 子项目。
- 不解决 Deep Agents Backend 中理论上的跨任务 `tool_call_id` 路径碰撞；本设计只允许通过当前任务权威
  引用访问，且每次读取都返回当前内容哈希供后续引用。

## 4. 方案比较

### 4.1 直接使用通用 `grep` 和 `read_file`

改动最少，但模型必须提供路径，无法确定任务归属，也无法限制 Artifact 类型。通用工具产生的结果缺少稳定
Artifact 身份和检索范围，不能建立可靠 provenance，不采用。

### 4.2 拦截 Deep Agents 卸载并持久化索引

卸载时立即写 Artifact Registry，搜索速度快，但需要依赖 Deep Agents Middleware 的包装顺序和内部实现，
并引入数据库迁移、索引同步和失败恢复。当前数据规模不需要该复杂度，不采用。

### 4.3 权威引用驱动的惰性目录（采用）

每次 Tool 调用从当前 Graph Messages 和生效 Compaction Snapshot 收集引用。搜索时用 Backend 下载允许的
UTF-8 文本并执行有界普通关键词匹配；读取时重新构造同一目录，用稳定 `artifact_id` 解析路径。该方案没有
额外持久化状态，不扫描根目录，也不依赖 Deep Agents 私有方法。

## 5. 总体架构

```text
ToolRuntime.state.messages       CompactionStore active Snapshot
             \                    /
              ArtifactReferenceCollector
                         |
                DiagnosticArtifactCatalog
                         |
              DiagnosticArtifactService
                  /              \
       bounded text search     bounded line read
                  \              /
             existing Backend.download_files
                         |
          structured ToolMessage + Investigation event
```

新增包：

```text
src/deepfix/artifact_retrieval/
    __init__.py
    models.py
    service.py
    tools.py
```

### 5.1 `ArtifactReferenceCollector`

职责只有三个：规范 Graph Message 身份、收集当前任务允许引用、生成稳定目录。不读取任意目录，不持久化结果。

接口：

```python
class ArtifactReferenceCollector:
    def collect(
        self,
        task_id: str,
        messages: Sequence[AnyMessage],
        *,
        expand_history: bool,
    ) -> DiagnosticArtifactCatalog: ...
```

`expand_history=False` 收集当前 ToolMessage 引用和生效 conversation history；`expand_history=True` 还读取
当前任务 conversation history，从其中的序列化 AI Tool Call/ToolMessage 工作单元发现旧的大型结果引用。
搜索和读取均使用 `True`，因此压缩后仍能解析之前的 `artifact_id`。

### 5.2 `DiagnosticArtifactService`

职责是任务校验、Backend 读取、UTF-8/大小验证、普通文本搜索、按行切片和有界结果构造。它不关心 LangChain
ToolRuntime，也不修改业务或 Investigation 状态。

```python
class DiagnosticArtifactService:
    def search(
        self,
        task_id: str,
        messages: Sequence[AnyMessage],
        query: str,
        artifact_kinds: Sequence[DiagnosticArtifactKind] | None,
        max_matches: int,
    ) -> DiagnosticSearchResult: ...

    def read(
        self,
        task_id: str,
        messages: Sequence[AnyMessage],
        artifact_id: str,
        start_line: int,
        line_count: int,
    ) -> DiagnosticReadResult: ...
```

### 5.3 Tool 外观

公开两个 StructuredTool：

```text
search_diagnostic_artifacts(query, artifact_kinds=None, max_matches=10)
read_diagnostic_artifact(artifact_id, start_line, line_count=100)
```

`task_id`、Graph Messages 和 Backend 路径都不出现在 Tool schema。工具从 `ToolRuntime` 获取当前 thread ID
和 state messages，调用 Service，再返回稳定 ToolMessage。

## 6. Artifact 权威来源与身份

### 6.1 允许类型

```python
class DiagnosticArtifactKind(StrEnum):
    LARGE_TOOL_RESULT = "large_tool_result"
    CONVERSATION_HISTORY = "conversation_history"
```

其他 ArtifactReference kind 即使存在于 Snapshot，也不进入目录。

### 6.2 当前 Graph Messages

Collector 先使用 `ensure_message_ids(task_id, messages)`。只有真实 `ToolMessage` 可以提供大型结果引用，并且：

- `tool_call_id` 非空；
- 内容中出现的路径严格匹配
  `/.deepfix-artifacts/large_tool_results/{sanitized_tool_call_id}`；
- `sanitized_tool_call_id` 使用与 Deep Agents 0.7 路径约定一致的规则：把 `.`, `/`, `\` 替换为 `_`；
  DeepFix 在自己的 Collector 内实现这三个字符的规范化，不导入 Deep Agents 私有 helper；
- 路径后不允许额外 `/`、`..`、查询参数或通配符；
- `source_message_id` 使用该 ToolMessage 的稳定 ID。

用户消息、AI 普通文本或模型生成的任意路径不会建立访问权限。

### 6.3 CompactionStore

Collector 从 `CompactionStore.list_snapshots(task_id)` 只选择 `lifecycle="active"` 的 Snapshot，并从其
`artifact_references` 选择 `kind="conversation_history"`。引用必须符合：

```text
/.deepfix-artifacts/conversation_history/{task_id}.md
```

Prepared、abandoned、其他 task ID、其他 kind 或不符合固定路径的引用全部忽略。Snapshot 的引用哈希表示
历史事件/section 身份，不等同于当前追加式文件的完整内容哈希，因此读取结果单独计算实际字节 SHA-256。

### 6.4 从历史发现旧大型结果

conversation history 属于当前任务后，Collector 可以读取并解析每个 `<deepfix_history_event>` 中的
`<serialized_messages>`。只处理 `<message type="ai">` 中的 `<tool_call id>` 和其后、下一个 AI/Human
消息前的 `<message type="tool">`。ToolMessage 文本中的大型结果路径 basename 必须对应该工作单元内某个
Tool Call ID 的 sanitized 值。

因此 HumanMessage 即使包含一个看似合法的 Artifact 路径，也不能授权该路径。并行 Tool Call 使用同一
工作单元的 Call ID 集合。XML 无法解析、边界含糊或引用无法配对时优先不授权，不做文本猜测。

### 6.5 稳定 ID

```text
artifact_id = artifact_<sha256(
    "diagnostic-artifact:v1|task_id|kind|normalized_backend_path"
)[:32]>
```

同一路径在不同任务下具有不同 ID。公开结果返回 `artifact_id`，不把 Backend path 作为后续 Tool 输入。

## 7. 领域模型

```python
class DiagnosticArtifactDescriptor(StrictModel):
    artifact_id: str
    task_id: str
    kind: DiagnosticArtifactKind
    backend_path: str
    source_message_id: str | None = None
    tool_call_id: str | None = None
    snapshot_version: int | None = None


class DiagnosticMatch(StrictModel):
    artifact_id: str
    kind: DiagnosticArtifactKind
    start_line: int
    end_line: int
    excerpt: str
    content_hash: str


class DiagnosticSearchResult(StrictModel):
    query_terms: list[str]
    matches: list[DiagnosticMatch]
    searched_artifact_count: int
    omitted_artifact_count: int
    truncated: bool


class DiagnosticReadResult(StrictModel):
    artifact_id: str
    kind: DiagnosticArtifactKind
    start_line: int
    end_line: int
    total_lines: int
    content: str
    content_hash: str
    truncated: bool
```

`backend_path` 只存在于 Service 内部模型，不进入公开 Tool 输入。ToolMessage artifact 可以返回结果的安全
字段，但不得返回可供下一次调用使用的任意路径参数。

## 8. 搜索与读取语义

### 8.1 查询规范化

- `query.strip()` 后必须为 1–200 字符；
- 按 Unicode 空白拆成 1–8 个关键词；
- 使用 Unicode `casefold()` 做大小写不敏感的普通子串匹配；
- 不解释引号、转义、glob 或正则元字符；它们只是普通字符；
- 所有关键词必须同时出现在同一个候选上下文窗口中，语义为 AND。

### 8.2 匹配窗口

逐行检查关键词。一个窗口由命中行前后各 2 行构成；关键词可以分布在该 5 行窗口中。重叠窗口合并，
返回一段带原始 1-based 行号的 excerpt。结果按目录顺序、起始行排序，保证离线确定性。

### 8.3 固定上限

```text
最多处理 32 个 Artifact
单个 Artifact 下载后最多 10 MiB
max_matches 参数范围 1–20，默认 10
搜索返回总字符最多 12,000
read start_line >= 1
read line_count 范围 1–200，默认 100
read 返回字符最多 12,000
```

超过 Artifact 数量时按目录确定性顺序处理前 32 个，并设置 `omitted_artifact_count`。超过匹配或字符预算
时在完整 UTF-8 字符边界截断，并设置 `truncated=True`。不允许调用者扩大字符、文件大小或 Artifact 数量
上限。

### 8.4 内容哈希

Backend 返回 bytes 后先检查 10 MiB 上限，再按 UTF-8 解码；`content_hash` 是原始 bytes 的完整 SHA-256。
搜索和读取每次重新计算，不把 Snapshot section hash 当作文件哈希。当前版本不要求模型回传哈希，也不
建立持久化完整性状态；哈希用于 provenance、事件记录和人工核对。

## 9. ToolMessage 协议

成功搜索：

```text
status="success"
content=带行号的有界匹配片段
artifact={
  "result_type": "diagnostic_artifact_search",
  "artifact_ids": [...],
  "match_count": N,
  "searched_artifact_count": N,
  "truncated": bool,
  "content_hashes": [...]
}
```

成功读取：

```text
status="success"
content=带行号的有界内容
artifact={
  "result_type": "diagnostic_artifact_read",
  "artifact_id": "artifact_...",
  "kind": "large_tool_result|conversation_history",
  "start_line": N,
  "end_line": N,
  "total_lines": N,
  "content_hash": "...",
  "truncated": bool
}
```

所有 ToolMessage ID 通过 `stable_generated_message_id(task_id, tool_call_id, result_type)` 生成。没有匹配、
未知 Artifact、缺失文件、非 UTF-8、超大文件和输入校验错误返回 `status="error"`，内容限制为 300 字符，
不回显文件正文、Backend 内部异常或其他任务信息。

## 10. Investigation 集成

两个工具使用 `InvestigationCapability.READ`，风险等级 L0，不加入 `interrupt_on`。Agent 的
`core_tool_names`、能力映射和工具列表显式注册它们，扩展不能使用相同名称覆盖。

新增事件：

```python
InvestigationEventType.ARTIFACT_SEARCHED = "artifact_searched"
InvestigationEventType.ARTIFACT_READ = "artifact_read"
```

InvestigationMiddleware 看到成功 ToolMessage 的 `result_type` 后，由 Coordinator 记录对应事件：

- `source_message_id` 为 ToolMessage 稳定 ID；
- `tool_call_id` 为真实 Tool Call ID；
- payload 只含 artifact ID、范围、匹配数、截断状态和内容哈希；
- `progress_kind=None`；
- Artifact 正文和查询原文不进入 InvestigationStore；查询只保存规范化 terms 的 SHA-256。

`artifact_searched` 和 `artifact_read` 进入普通 no-progress/stagnation 计算。它们不能清空
`no_progress_count`、增加 `progress_generation` 或改变 AgentPhase。只有随后发生 supported/rejected
假设迁移等现有 strong progress，才重置停滞。

正常阶段按 READ 能力可见。Level 1 停滞后，它们与其他读取 Tool 一样必须获得绑定工具和目标的一次性
`continue_investigation` permit。搜索结果已经包含上下文片段，permit 消费后的下一步应形成假设迁移；
不能通过连续搜索延迟 Level 2 暂停。

### 10.1 与假设证据的关系

检索结果是已有诊断内容的可追溯视图，不是新的 DeterministicEvidence。第一版不修改
`RecordHypothesisInput`，也不新增 `retrieval_event_ids` 或类似字段：模型可以在假设 `reason` 中解释它从
哪个 `artifact_id`、内容哈希和行范围获得线索，系统 provenance 由紧邻的 `artifact_searched`/
`artifact_read` 事件保存，但该引用本身不能满足 supported 假设所需的确定性证据校验。

因此，检索可以推动模型提出或排除候选方向；要把假设迁移为 supported，仍必须使用现有机制产生真实
测试、traceback 关联、源码关系或其他 Coordinator 认可的 evidence ID。这样既保留检索来源，又避免把
旧日志片段错误升级为新的 pytest 结果或文件事实。

## 11. 提示词

在调查/诊断提示中加入固定规则：

```text
当 ToolMessage 表明完整结果已卸载时，使用 diagnostic artifact 工具检索真实内容。
不要根据截断预览猜测，也不要使用普通 grep/read_file 扫描 Artifact 根目录。
Artifact 检索只是恢复旧证据；读取后必须更新、支持或排除假设，不能以重复检索代替进展。
```

不增加 Protected Context 区块。现有 ToolMessage 预览和 Snapshot artifact references 只提示内容存在，
实际访问始终经过专用 Tool。

## 12. 错误传播

### 12.1 可预期 Tool 错误

以下错误返回稳定 error ToolMessage，Agent 可以调整输入或改用其他证据：

- 空/过长查询、关键词过多、非法 kind 或数值超限；
- 当前任务没有允许的 Artifact 或没有匹配；
- `artifact_id` 不在重新构造的当前任务目录；
- 文件不存在、不是 UTF-8、超过 10 MiB、行号超界；
- 历史 XML 中某条旧大型结果引用无法安全配对。

单条历史引用无法解析时跳过该引用；conversation history 本身仍可作为文本搜索对象。

### 12.2 系统级恢复错误

以下错误表示权威状态或 Backend 不可靠，不能伪装为“没有匹配”：

- TaskRepository/CompactionStore 读取异常；
- Backend `download_files` 抛出异常、返回数量异常或协议对象不可解释；
- 当前 task ID 与权威 TaskState 不一致；
- 多个不同描述符生成同一 `artifact_id`。

Collector/Service 抛出内部类型化 `DiagnosticArtifactSystemError`，只携带稳定错误码、可安全记录的阶段和
原始异常链，不构造业务恢复信息。错误码至少为
`diagnostic_artifact_reference_load_failed` 或 `diagnostic_artifact_backend_read_failed`。

持有 `InvestigationCoordinator` 的 Tool 适配层捕获该错误，通过 Coordinator 为当前 task 构造
`InvestigationRecoveryMetadata`，再抛出现有 `InvestigationStateError`。BugfixService 继续负责校验 task
ID、持久化恢复信息并把任务转为 PAUSED。Collector、Artifact Service、Tool 和 Middleware 都不直接修改
`TaskStatus`；其中 Tool 只负责异常类型转换，不提交 Investigation 事件或业务状态。

## 13. Deep Agents 复用边界

继续复用：

- CompositeBackend 和 `artifacts_root` 路由；
- Deep Agents 大型 ToolMessage 自动卸载；
- `/.deepfix-artifacts/large_tool_results/{sanitized_tool_call_id}` 公共路径约定；
- LangChain XML history 序列化结果；
- StructuredTool、ToolRuntime 和 ToolMessage 外观。

DeepFix 负责：

- 当前任务引用收集和授权；
- history 工作单元解析；
- 稳定 Artifact ID；
- 下载大小、UTF-8、路径和结果预算校验；
- 文本匹配、按行读取和 provenance；
- 内部检索错误类型；
- Tool 适配层到 Investigation 类型化恢复错误的转换；
- Investigation 事件提交。

不调用或 monkeypatch Deep Agents 私有 offload、summary、cutoff 或 event 方法。兼容测试可以实例化公开
FilesystemMiddleware/CompositeBackend 行为，不能把私有函数当作生产依赖。

## 14. 测试要求

### 14.1 身份与归属

- 同一 task/kind/path 生成相同 ID，不同 task 生成不同 ID；
- 现有稳定 Message ID 被复用，缺失 ID 按现有规则生成；
- Human/AI 文本中的伪造路径不授权；
- 当前 ToolMessage 的真实引用被授权；
- 并行 Tool Call 与 ToolMessage 正确配对；
- 生效 Snapshot 的 conversation history 被授权，prepared/abandoned/跨任务引用被拒绝；
- 历史中的旧大型结果只在 AI Tool Call/ToolMessage 工作单元可验证时被授权；
- `..`、额外子路径、其他 Artifact kind 和任意路径输入均不可访问。

### 14.2 搜索与读取

- 单关键词、多关键词 AND、Unicode casefold、普通正则字符；
- 5 行窗口、重叠合并、确定性排序和 1-based 行号；
- 32 Artifact、20 match、12,000 字符、200 行和 10 MiB 上限；
- UTF-8 解码、空文件、末尾换行和超长单行；
- 未知 ID、无匹配、缺失文件、非 UTF-8 和超界返回稳定 error ToolMessage；
- 结果 metadata 不暴露可作为输入的任意 Backend path。

### 14.3 Investigation 与 Service

- 两个成功事件 `progress_kind=None`，不改变阶段或进展代次；
- 连续检索增加 no-progress 并最终触发现有停滞门禁；
- Level 1 下无 permit 被拒绝，绑定 permit 只允许一次；
- Backend/CompactionStore 系统异常先成为 `DiagnosticArtifactSystemError`，再由 Tool 适配层转换为带当前
  task ID 和恢复元数据的 `InvestigationStateError`；
- Tool 适配层以外的组件不构造恢复元数据，也不修改 TaskStatus；
- BugfixService 把该错误持久化为 PAUSED，跨任务恢复信息 fail closed；
- 检索事件保留 artifact ID、哈希和行范围，但检索结果不能单独满足 supported 假设的 evidence ID 校验；
- 两个 Tool 为 READ/L0，不进入审批列表，扩展不能覆盖。

### 14.4 端到端兼容

- 用公开 Deep Agents FilesystemMiddleware 和 CompositeBackend 产生真实卸载引用，再由搜索 Tool 找到
  pytest/traceback；
- 压缩后只保留 Snapshot/history 引用时，搜索仍能找到旧日志和其中引用的大型结果；
- 检索不访问网络、不修改目标项目、不读取 research/receipt；
- 正常 Bug 修复可以用检索片段提出候选方向或解释排除原因；支持假设仍需 Coordinator 认可的 evidence ID，
  真实 pytest exit_code 仍只来自确定性证据；
- 与 Investigation、Compaction、Research、Approval、Service 全量离线测试共同通过。

## 15. 预计修改范围

新增：

```text
src/deepfix/artifact_retrieval/__init__.py
src/deepfix/artifact_retrieval/models.py
src/deepfix/artifact_retrieval/service.py
src/deepfix/artifact_retrieval/tools.py
tests/artifact_retrieval/
```

修改：

```text
src/deepfix/investigation/models.py
src/deepfix/investigation/coordinator.py
src/deepfix/agent.py
src/deepfix/prompts.py
tests/investigation/test_coordinator.py
tests/investigation/test_workflow.py
tests/test_agent.py
tests/test_approval.py
tests/test_service.py
```

`src/deepfix/agent.py`、`src/deepfix/prompts.py` 与用户现有未提交修改重叠时必须逐 hunk 暂存；不得提交
`src/deepfix/debug.py` 或 `docs/debug/`。

## 16. 完成标准

- Agent 能通过两个专用 Tool 搜索和按行读取当前任务的两类诊断 Artifact；
- 无法通过 Tool schema、伪造消息、跨任务 ID 或路径文本读取其他内容；
- 搜索/读取结果有稳定身份、行号、哈希和严格预算；
- 检索不伪造确定性证据、不计算 strong progress、不绕过停滞 permit；
- 系统读取故障通过现有类型化恢复边界暂停，普通输入/文件错误保持 Tool 级可恢复；
- 不新增 Registry、索引服务、模型调用、网络依赖或目标项目写入；
- 全部新测试和现有离线套件通过；
- 本子项目完成后，整体 reliability 项目仍需完成 Progress Events + CLI Renderer，才能关闭完整可靠性
  改造范围。
