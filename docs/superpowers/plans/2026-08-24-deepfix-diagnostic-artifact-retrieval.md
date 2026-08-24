# DeepFix Diagnostic Artifact Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为单 Repair Agent 增加任务隔离、只读、有界的诊断 Artifact 搜索与按行读取能力，同时保持 Investigation 进展、恢复和审批规则可信。

**Architecture:** 每次调用从当前 Graph Messages 和生效 Compaction Snapshot 惰性重建允许目录，不建立 Registry 或索引。Collector 只确定授权和稳定身份，Service 只负责 Backend 读取及文本预算，StructuredTool 隐藏 task ID 和路径；InvestigationMiddleware 在成功结果后提交无强进展事件，系统故障沿现有类型化恢复边界进入 BugfixService。

**Tech Stack:** Python 3.11、Pydantic 2、LangChain StructuredTool/ToolRuntime/ToolMessage、Deep Agents 0.7 CompositeBackend/FilesystemMiddleware、SQLite CompactionStore/InvestigationStore、pytest、Ruff。

**Spec:** `docs/superpowers/specs/2026-08-24-deepfix-diagnostic-artifact-retrieval-design.md`

## Global Constraints

- 只授权 `large_tool_result` 和 `conversation_history`；不得读取 research、tool receipts、Snapshot detail 或任意路径。
- 不新增 Artifact Registry、后台索引、embedding、模型调用、网络访问或目标项目写入。
- 查询为 Unicode `casefold()` 后的普通子串 AND 匹配，不支持 regex/glob/模糊搜索。
- 单次最多 32 个 Artifact、单个 10 MiB、20 个 match、12,000 返回字符；read 最多 200 行和 12,000 字符。
- `artifact_id` 由 task ID、kind、规范 Backend path 确定性生成；公开 Tool schema 不含 task ID 或 Backend path。
- 检索事件 `progress_kind=None`，不得伪造测试、文件、审批、研究证据，也不得单独满足 supported 假设的 evidence ID 校验。
- 普通输入、无匹配、未知 ID、缺失/非 UTF-8/超大文件返回有界 error ToolMessage；权威 Store 或 Backend 协议故障才抛类型化恢复异常。
- DeepFix 自己实现 `.`, `/`, `\` 到 `_` 的 Tool Call ID 规范化，不导入 Deep Agents 私有 helper。
- 所有实现测试离线运行；生产代码不得依赖 Deep Agents 私有 offload、summary、cutoff 或 event 方法。
- `src/deepfix/agent.py` 和 `src/deepfix/prompts.py` 有用户未提交修改，执行时必须逐 hunk 暂存；不得提交 `src/deepfix/debug.py` 或 `docs/debug/`。

## File Structure

新增：

- `src/deepfix/artifact_retrieval/__init__.py`：公开稳定领域接口。
- `src/deepfix/artifact_retrieval/models.py`：kind、descriptor、catalog、search/read 结果和稳定 ID。
- `src/deepfix/artifact_retrieval/errors.py`：普通 Tool 错误和内部系统错误。
- `src/deepfix/artifact_retrieval/collector.py`：当前消息、active Snapshot 和 history 工作单元授权。
- `src/deepfix/artifact_retrieval/service.py`：Backend 下载、UTF-8/大小校验、关键词搜索和按行读取。
- `src/deepfix/artifact_retrieval/tools.py`：两个 StructuredTool、ToolMessage 协议和恢复异常转换。
- `tests/artifact_retrieval/helpers.py`：任务、Backend、Snapshot 和 ToolRuntime 测试夹具。
- `tests/artifact_retrieval/test_models.py`：模型、身份和输入边界。
- `tests/artifact_retrieval/test_collector.py`：引用授权与 history 配对。
- `tests/artifact_retrieval/test_service.py`：搜索、读取、预算和错误分类。
- `tests/artifact_retrieval/test_tools.py`：公开 schema、ToolMessage 和系统错误转换。
- `tests/artifact_retrieval/test_workflow.py`：Deep Agents 卸载/压缩后的端到端检索。

修改：

- `src/deepfix/investigation/models.py`：增加两个检索事件类型。
- `src/deepfix/investigation/coordinator.py`：把成功 Tool artifact 投影为无进展 observation。
- `src/deepfix/agent.py`：构造 Service/Tools、注册核心名称和 READ 能力。
- `src/deepfix/prompts.py`：加入卸载结果的专用检索规则。
- `tests/investigation/test_coordinator.py`：事件 payload、幂等和非强进展。
- `tests/investigation/test_middleware.py`：阶段可见性、停滞 permit 和重复检索。
- `tests/test_agent.py`：工具装配、不可覆盖和无审批。
- `tests/test_approval.py`：检索工具不进入 interrupt 配置。
- `tests/test_prompting.py`：提示规则覆盖。
- `tests/test_service.py`：恢复异常跨 Service 边界进入 PAUSED。

---

## Batch 1：授权目录和稳定身份

### Task 1: 定义检索领域模型、稳定身份与错误类型

**Files:**
- Create: `src/deepfix/artifact_retrieval/__init__.py`
- Create: `src/deepfix/artifact_retrieval/models.py`
- Create: `src/deepfix/artifact_retrieval/errors.py`
- Create: `tests/artifact_retrieval/__init__.py`
- Create: `tests/artifact_retrieval/test_models.py`

**Interfaces:**
- Consumes: `deepfix.compaction.models.StrictModel`。
- Produces: `DiagnosticArtifactKind`, `DiagnosticArtifactDescriptor`, `DiagnosticArtifactCatalog`, `DiagnosticMatch`, `DiagnosticSearchResult`, `DiagnosticReadResult`, `stable_diagnostic_artifact_id()`, `DiagnosticArtifactToolError`, `DiagnosticArtifactSystemError`。

- [ ] **Step 1: 写稳定身份和严格模型的失败测试**

```python
import pytest
from pydantic import ValidationError

from deepfix.artifact_retrieval.models import (
    DiagnosticArtifactDescriptor,
    DiagnosticArtifactKind,
    stable_diagnostic_artifact_id,
)


def test_artifact_id_is_stable_and_task_scoped():
    path = "/.deepfix-artifacts/large_tool_results/call_1"
    first = stable_diagnostic_artifact_id(
        "task-a", DiagnosticArtifactKind.LARGE_TOOL_RESULT, path
    )
    assert first == stable_diagnostic_artifact_id(
        "task-a", DiagnosticArtifactKind.LARGE_TOOL_RESULT, path
    )
    assert first != stable_diagnostic_artifact_id(
        "task-b", DiagnosticArtifactKind.LARGE_TOOL_RESULT, path
    )
    assert first.startswith("artifact_") and len(first) == 41


def test_descriptor_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        DiagnosticArtifactDescriptor(
            artifact_id="artifact_" + "a" * 32,
            task_id="task-a",
            kind="large_tool_result",
            backend_path="/.deepfix-artifacts/large_tool_results/call_1",
            unexpected=True,
        )
```

- [ ] **Step 2: 运行测试确认因模块不存在而失败**

Run: `pytest tests/artifact_retrieval/test_models.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deepfix.artifact_retrieval'`。

- [ ] **Step 3: 实现严格模型和确定性 ID**

```python
class DiagnosticArtifactKind(StrEnum):
    LARGE_TOOL_RESULT = "large_tool_result"
    CONVERSATION_HISTORY = "conversation_history"


class DiagnosticArtifactDescriptor(StrictModel):
    artifact_id: str = Field(pattern=r"^artifact_[0-9a-f]{32}$")
    task_id: str = Field(min_length=1)
    kind: DiagnosticArtifactKind
    backend_path: str = Field(min_length=1)
    source_message_id: str | None = None
    tool_call_id: str | None = None
    snapshot_version: int | None = Field(default=None, ge=1)


class DiagnosticArtifactCatalog(StrictModel):
    artifacts: list[DiagnosticArtifactDescriptor] = Field(default_factory=list)

    def by_id(self, artifact_id: str) -> DiagnosticArtifactDescriptor | None:
        return next((item for item in self.artifacts if item.artifact_id == artifact_id), None)


def stable_diagnostic_artifact_id(task_id, kind, backend_path):
    payload = f"diagnostic-artifact:v1|{task_id.strip()}|{kind.value}|{backend_path}"
    return f"artifact_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"
```

同时按设计定义 `DiagnosticMatch`、`DiagnosticSearchResult`、`DiagnosticReadResult`。`DiagnosticArtifactToolError` 只含稳定 `error_code` 和安全 message；`DiagnosticArtifactSystemError` 只含 `error_code`、`stage`，并保留异常链，不携带 `TaskStatus` 或恢复元数据。

```python
class DiagnosticArtifactToolError(ValueError):
    def __init__(self, error_code: str, safe_message: str) -> None:
        self.error_code = error_code
        self.safe_message = safe_message[:300]
        super().__init__(error_code)


class DiagnosticArtifactSystemError(RuntimeError):
    def __init__(self, error_code: str, stage: str) -> None:
        self.error_code = error_code
        self.stage = stage
        super().__init__(error_code)
```

- [ ] **Step 4: 补齐上限和错误字段测试并运行**

Run: `pytest tests/artifact_retrieval/test_models.py -v`

Expected: PASS；未知 kind、负行号、非法 hash/ID 和 extra 字段均被拒绝。

- [ ] **Step 5: 运行 Ruff 并提交**

Run: `ruff check src/deepfix/artifact_retrieval tests/artifact_retrieval/test_models.py`

```powershell
git add -- src/deepfix/artifact_retrieval tests/artifact_retrieval/__init__.py tests/artifact_retrieval/test_models.py
git commit -m "feat: define diagnostic artifact retrieval models"
```

### Task 2: 从当前消息和 active Snapshot 构造授权目录

**Files:**
- Create: `src/deepfix/artifact_retrieval/collector.py`
- Create: `tests/artifact_retrieval/helpers.py`
- Create: `tests/artifact_retrieval/test_collector.py`
- Modify: `src/deepfix/artifact_retrieval/__init__.py`

**Interfaces:**
- Consumes: `ensure_message_ids(task_id, messages)`, `CompactionStore.list_snapshots(task_id)`, `DiagnosticArtifactCatalog`。
- Produces: `ArtifactReferenceCollector(compaction_store, backend).collect(task_id, messages, *, expand_history) -> DiagnosticArtifactCatalog`。

- [ ] **Step 1: 写当前 ToolMessage 授权和伪造路径拒绝测试**

```python
def test_only_paired_tool_message_authorizes_large_result(collector):
    messages = [
        HumanMessage(content="read /.deepfix-artifacts/large_tool_results/forged"),
        AIMessage(content="", tool_calls=[{
            "name": "execute", "args": {}, "id": "call/1", "type": "tool_call"
        }]),
        ToolMessage(
            content="full result: /.deepfix-artifacts/large_tool_results/call_1",
            tool_call_id="call/1",
        ),
    ]
    catalog = collector.collect("task-a", messages, expand_history=False)
    assert [item.tool_call_id for item in catalog.artifacts] == ["call/1"]
    assert catalog.artifacts[0].source_message_id


def test_ai_text_and_wrong_basename_do_not_authorize(collector):
    # AI 普通文本、call/1 配 call_2、额外子路径和 .. 均得到空目录。
```

另加重复稳定 Message ID 测试：`ensure_message_ids()` 返回非空 `conflicted_message_ids` 时，Collector 不得猜测
来源，必须抛 `DiagnosticArtifactSystemError("diagnostic_artifact_reference_load_failed", "message_identity")`。

- [ ] **Step 2: 写 active Snapshot conversation history 授权测试**

测试必须保存 prepared、active、abandoned 和另一 task 的 Snapshot，断言只收集：

```text
/.deepfix-artifacts/conversation_history/task-a.md
```

并保留 `snapshot_version`；`research` 和 `snapshot_detail` 被忽略。

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/artifact_retrieval/test_collector.py -v`

Expected: FAIL because `ArtifactReferenceCollector` is not defined。

- [ ] **Step 4: 实现 fail-closed 引用收集**

```python
_LARGE_ROOT = "/.deepfix-artifacts/large_tool_results/"
_HISTORY_ROOT = "/.deepfix-artifacts/conversation_history/"


def _sanitize_tool_call_id(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace("\\", "_")


class ArtifactReferenceCollector:
    def __init__(self, compaction_store: CompactionStore, backend: Any) -> None:
        self.compaction_store = compaction_store
        self.backend = backend

    def collect(self, task_id, messages, *, expand_history):
        identities = ensure_message_ids(task_id, messages)
        if identities.conflicted_message_ids:
            raise DiagnosticArtifactSystemError(
                "diagnostic_artifact_reference_load_failed", "message_identity"
            )
        identified = identities.messages
        descriptors = self._from_current_messages(task_id, identified)
        descriptors.extend(self._from_active_snapshots(task_id))
        if expand_history:
            descriptors.extend(self._from_history(task_id, descriptors))
        return DiagnosticArtifactCatalog(
            artifacts=_deduplicate_and_sort(task_id, descriptors)
        )
```

路径必须由固定 root 和严格 basename 重建，不用 `resolve()`、glob、目录扫描或模型文本作为授权来源。`list_snapshots()` 抛异常时包装为 `DiagnosticArtifactSystemError("diagnostic_artifact_reference_load_failed", "snapshot_read")`。
`_deduplicate_and_sort(task_id, descriptors)` 按 `(kind.value, backend_path)` 排序；相同
`artifact_id` 和相同描述符只展示一次，相同 ID 对应不同 kind/path 时抛
`DiagnosticArtifactSystemError("diagnostic_artifact_reference_load_failed", "artifact_id_collision")`。

- [ ] **Step 5: 运行 Collector 测试和现有身份测试**

Run: `pytest tests/artifact_retrieval/test_collector.py tests/compaction/test_identity.py -v`

Expected: PASS。

- [ ] **Step 6: Ruff 和提交**

Run: `ruff check src/deepfix/artifact_retrieval tests/artifact_retrieval`

```powershell
git add -- src/deepfix/artifact_retrieval tests/artifact_retrieval
git commit -m "feat: authorize current diagnostic artifacts"
```

### Task 3: 从 conversation history 安全发现旧大型结果

**Files:**
- Modify: `src/deepfix/artifact_retrieval/collector.py`
- Modify: `tests/artifact_retrieval/test_collector.py`

**Interfaces:**
- Consumes: Task 2 的 active conversation descriptor 和 Backend `download_files([path])`。
- Produces: `_from_history()`，只授权可验证 AI Tool Call/配对 ToolMessage 工作单元中的大型结果。

- [ ] **Step 1: 写 history 配对失败测试**

用 `DeepAgentsArtifactAdapter.persist_history()` 生成真实 XML，覆盖：

```python
AIMessage(content="", tool_calls=[
    {"name": "execute", "args": {}, "id": "parallel/1", "type": "tool_call"},
    {"name": "grep", "args": {}, "id": "parallel.2", "type": "tool_call"},
])
```

断言 `parallel_1` 和 `parallel_2` 都可授权；HumanMessage 伪造引用、无对应 AI call、跨到下一 AI/Human 消息的引用、basename 不匹配均不可授权。XML 损坏时 conversation history 自身仍留在目录，旧 large result 不授权。

- [ ] **Step 2: 运行目标测试确认失败**

Run: `pytest tests/artifact_retrieval/test_collector.py -k history -v`

Expected: FAIL because history expansion returns no old large result descriptors。

- [ ] **Step 3: 实现有界 XML 工作单元解析**

```python
def _authorized_history_tool_call_ids(serialized_messages: Element) -> Iterator[tuple[str, str]]:
    active_call_ids: set[str] = set()
    for message in serialized_messages.findall("message"):
        message_type = message.get("type")
        if message_type in {"ai", "human"}:
            active_call_ids = (
                {str(call.get("id")) for call in message.findall("tool_call")}
                if message_type == "ai"
                else set()
            )
            continue
        if message_type != "tool" or not active_call_ids:
            continue
        for backend_path in _strict_large_result_paths("".join(message.itertext())):
            basename = backend_path.rsplit("/", 1)[-1]
            matching = [call_id for call_id in active_call_ids if _sanitize_tool_call_id(call_id) == basename]
            if len(matching) == 1:
                yield matching[0], backend_path
```

`_strict_large_result_paths(text)` 使用转义后的固定 root 正则提取无 `/`、`..`、`?`、`#`、`*` 的 basename，
并返回完整 Backend path 字符串；它不接受调用者传入 root。仅解析
`<deepfix_history_event>/<serialized_messages>`；拒绝边界含糊、重复 sanitized ID 或解析失败的引用。下载返回
列表数量不为 1、对象缺少协议字段或 Backend 抛异常，转为 `DiagnosticArtifactSystemError`；
`file_not_found` 作为普通不可用引用，不升级系统错误。

- [ ] **Step 4: 验证并提交 Batch 1**

Run: `pytest tests/artifact_retrieval/test_collector.py -v`

Run: `ruff check src/deepfix/artifact_retrieval tests/artifact_retrieval`

```powershell
git add -- src/deepfix/artifact_retrieval/collector.py tests/artifact_retrieval/test_collector.py
git commit -m "feat: recover authorized artifacts from history"
```

**Batch 1 审查点：** 检查目录是否完全由权威引用构造、是否存在任意路径入口、并行 Tool Call 是否 fail closed；批准后再进入 Batch 2。

---

## Batch 2：有界 Service、Tool 和 Investigation 事件

### Task 4: 实现确定性普通文本搜索和按行读取

**Files:**
- Create: `src/deepfix/artifact_retrieval/service.py`
- Create: `tests/artifact_retrieval/test_service.py`
- Modify: `src/deepfix/artifact_retrieval/__init__.py`

**Interfaces:**
- Consumes: `TaskRepository.get(task_id)`,
  `ArtifactReferenceCollector.collect(task_id, messages, expand_history=True)`，Backend `download_files(paths)`。
- Produces: `DiagnosticArtifactService.search(task_id, messages, query, artifact_kinds, max_matches) -> DiagnosticSearchResult`
  和 `read(task_id, messages, artifact_id, start_line, line_count) -> DiagnosticReadResult`。

- [ ] **Step 1: 写搜索语义失败测试**

```python
def test_search_uses_literal_casefolded_and_with_five_line_window(service):
    result = service.search(
        "task-a", messages(), "VALUE[0] failed", None, 10
    )
    assert result.query_terms == ["value[0]", "failed"]
    assert [(m.start_line, m.end_line) for m in result.matches] == [(1, 5)]
    assert "VALUE[0]" in result.matches[0].excerpt


def test_overlapping_windows_are_merged_and_sorted(service):
    result = service.search("task-a", messages(), "failure", None, 10)
    assert [(item.artifact_id, item.start_line, item.end_line) for item in result.matches] == [
        (ARTIFACT_ID, 1, 8),
        (SECOND_ARTIFACT_ID, 4, 8),
    ]


def test_regex_characters_are_literal(service):
    result = service.search("task-a", messages(), "value[0]", None, 10)
    assert len(result.matches) == 1
    assert "value[0]" in result.matches[0].excerpt.casefold()
```

- [ ] **Step 2: 写读取和固定预算失败测试**

覆盖目录含 33 个 Artifact 时只处理前 32 个、请求 `max_matches=21`、请求 201 行、12,000 字符、10 MiB、
1-based 行号、空文件、末尾换行、超长单行、UTF-8 解码、完整 bytes SHA-256。超出公开参数范围必须返回
稳定 Tool error；内部目录和内容超限则按设计截断并设置计数/标志，调用者不能扩大硬上限。

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/artifact_retrieval/test_service.py -v`

Expected: FAIL because `DiagnosticArtifactService` does not exist。

- [ ] **Step 4: 实现下载分类、搜索和读取**

```python
class DiagnosticArtifactService:
    def __init__(self, tasks: TaskRepository, collector: ArtifactReferenceCollector, backend: Any):
        self.tasks = tasks
        self.collector = collector
        self.backend = backend

    def search(self, task_id, messages, query, artifact_kinds=None, max_matches=10):
        self._require_task(task_id)
        terms = _query_terms(query)
        catalog = self.collector.collect(task_id, messages, expand_history=True)
        selected = _select_kinds(catalog, artifact_kinds)[:32]
        return _bounded_matches(selected, terms, max_matches, self._download_text)

    def read(self, task_id, messages, artifact_id, start_line=1, line_count=100):
        self._require_task(task_id)
        catalog = self.collector.collect(task_id, messages, expand_history=True)
        descriptor = catalog.by_id(artifact_id)
        if descriptor is None:
            raise DiagnosticArtifactToolError("artifact_not_authorized", "Artifact 不属于当前任务")
        text, content_hash = self._download_text(descriptor)
        return _bounded_lines(descriptor, text, content_hash, start_line, line_count)
```

查询和 kind 校验使用以下确定接口；`None` 表示两类都搜索，空列表或重复/未知 kind 为普通输入错误：

```python
def _query_terms(query: str) -> list[str]:
    normalized = query.strip()
    if not 1 <= len(normalized) <= 200:
        raise DiagnosticArtifactToolError("artifact_query_invalid", "查询长度必须为 1 到 200 字符")
    terms = [item.casefold() for item in normalized.split()]
    if not 1 <= len(terms) <= 8:
        raise DiagnosticArtifactToolError("artifact_query_invalid", "查询必须包含 1 到 8 个关键词")
    return terms


def _select_kinds(catalog, artifact_kinds):
    # 返回按 catalog 顺序过滤的新 list；None 选择全部，空/重复/未知值抛 artifact_kind_invalid。
```

实现时把上面的注释展开为：先将每个字符串转换为 `DiagnosticArtifactKind`，捕获 `ValueError` 并抛
`artifact_kind_invalid`；检查转换后列表非空且 `len(set(kinds)) == len(kinds)`；最后按 catalog 原顺序过滤。
`_bounded_matches(descriptors, terms, max_matches, downloader)` 对每行的前后两行建立 5 行窗口，要求所有 term
都出现在窗口 `casefold()` 文本中，合并同一 Artifact 的重叠窗口，再按目录序号/起始行排序；达到 20 条或
12,000 字符即停止并标记 truncated。`_bounded_lines()` 使用 `splitlines()` 保持 1-based 范围，字符预算在
Unicode 字符边界截断；两个 helper 均返回 Task 1 的严格结果模型。

`_require_task()` 调用 `tasks.get(task_id)` 并校验返回对象的 `task_id` 完全一致；Repository 抛异常或身份不一致
时产生 `DiagnosticArtifactSystemError("diagnostic_artifact_reference_load_failed", "task_read")`。
`search()` 在目录为空时抛 `artifact_catalog_empty`，匹配为空时抛 `artifact_no_matches`，两者都是普通
`DiagnosticArtifactToolError`。目录超过 32 个时，`omitted_artifact_count` 保存未处理数量；不能静默伪装为
完整搜索。

`_download_text()` 必须先验证响应数量/协议，再检查 bytes 长度，最后 UTF-8 解码。Backend 抛异常或协议损坏为 `DiagnosticArtifactSystemError("diagnostic_artifact_backend_read_failed", "backend_read")`；响应中的 file missing、非 UTF-8、超大和行号越界为 `DiagnosticArtifactToolError`。

- [ ] **Step 5: 运行 Service 和 Collector 测试**

Run: `pytest tests/artifact_retrieval/test_service.py tests/artifact_retrieval/test_collector.py -v`

Expected: PASS，排序、哈希和截断结果重复运行一致。

- [ ] **Step 6: Ruff 和提交**

Run: `ruff check src/deepfix/artifact_retrieval tests/artifact_retrieval`

```powershell
git add -- src/deepfix/artifact_retrieval tests/artifact_retrieval/test_service.py
git commit -m "feat: search and read diagnostic artifacts safely"
```

### Task 5: 提供两个隐藏任务权限的 StructuredTool

**Files:**
- Create: `src/deepfix/artifact_retrieval/tools.py`
- Create: `tests/artifact_retrieval/test_tools.py`
- Modify: `src/deepfix/artifact_retrieval/__init__.py`

**Interfaces:**
- Consumes: `DiagnosticArtifactService`, `InvestigationCoordinator.recovery()`, `stable_generated_message_id()`。
- Produces: `build_search_diagnostic_artifacts_tool(service, coordinator)` 和 `build_read_diagnostic_artifact_tool(service, coordinator)`。

- [ ] **Step 1: 写 Tool schema 和成功 ToolMessage 失败测试**

```python
def test_search_schema_hides_task_messages_runtime_and_path(tool):
    assert set(tool.args) == {"query", "artifact_kinds", "max_matches"}


def test_read_schema_accepts_only_stable_id_and_range(tool):
    assert set(tool.args) == {"artifact_id", "start_line", "line_count"}


def test_search_returns_stable_message_and_safe_artifact(search_tool):
    result = invoke_tool(search_tool, search_call("call-1"), "task-a", messages())
    assert result.status == "success"
    assert result.id == stable_generated_message_id(
        "task-a", "call-1", "diagnostic_artifact_search"
    )
    assert "backend_path" not in str(result.artifact)
```

- [ ] **Step 2: 写普通错误和系统错误转换测试**

普通 `DiagnosticArtifactToolError` 返回 `status="error"`、最多 300 字符且不回显正文。系统错误必须：

```python
with pytest.raises(InvestigationStateError) as caught:
    invoke_tool(tool, call, "task-a", messages)
assert caught.value.recovery.error_code == "diagnostic_artifact_backend_read_failed"
assert caught.value.recovery.task_id == "task-a"
assert caught.value.recovery.tool_call_id == "call-system"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/artifact_retrieval/test_tools.py -v`

Expected: FAIL because tool builders are missing。

- [ ] **Step 4: 实现同步 Tool 外观和类型化边界**

```python
def build_search_diagnostic_artifacts_tool(service, coordinator):
    def search_diagnostic_artifacts(
        query: str,
        artifact_kinds: list[str] | None = None,
        max_matches: int = 10,
        runtime: ToolRuntime = None,
    ) -> ToolMessage:
        task_id, call_id, messages = _runtime_authority(runtime)
        try:
            result = service.search(task_id, messages, query, artifact_kinds, max_matches)
            return _search_message(task_id, call_id, result)
        except DiagnosticArtifactToolError as exc:
            return _error_message(task_id, call_id, exc)
        except DiagnosticArtifactSystemError as exc:
            raise InvestigationStateError(
                coordinator.recovery(
                    task_id,
                    exc.error_code,
                    tool_call_id=call_id,
                    checkpoint_available=True,
                    recovery_action="pause_and_retry_diagnostic_artifact_read",
                )
            ) from exc

    return StructuredTool.from_function(
        func=search_diagnostic_artifacts,
        name="search_diagnostic_artifacts",
        description="搜索当前任务已卸载或已压缩的诊断文本。",
    )
```

`_search_message()` 只返回 `artifact_ids`、匹配/搜索计数、`truncated`、`content_hashes` 和根据规范化
`query_terms` 计算的 `query_terms_hash`；不返回 query 原文或 Backend path。`_read_message()` 只返回稳定
artifact ID、kind、行范围、总行数、完整内容哈希和截断标志。`artifact_kinds` 在函数内部转换为 enum，
非法值通过 `DiagnosticArtifactToolError` 进入稳定 error ToolMessage。

读取 Tool 使用同一模式。`_runtime_authority()` 仅从 `ToolRuntime.config.configurable.thread_id`、`runtime.state["messages"]` 和 `runtime.tool_call_id` 取值；这些字段不得出现在 args schema。

```python
def _runtime_authority(runtime: ToolRuntime) -> tuple[str, str, list[AnyMessage]]:
    task_id = str(runtime.config.get("configurable", {}).get("thread_id", "")).strip()
    call_id = str(runtime.tool_call_id or "").strip()
    messages = list(runtime.state.get("messages", []))
    if not task_id or not call_id:
        raise DiagnosticArtifactToolError(
            "artifact_runtime_invalid", "诊断 Artifact 工具缺少任务或调用身份"
        )
    return task_id, call_id, messages
```

runtime 身份错误发生在 task ID 尚不可用时，用 scope `unknown`/`missing-call-id` 生成有界 error ToolMessage，
不尝试构造系统恢复元数据。

- [ ] **Step 5: 验证 Tool 协议并提交**

Run: `pytest tests/artifact_retrieval/test_tools.py -v`

Run: `ruff check src/deepfix/artifact_retrieval tests/artifact_retrieval`

```powershell
git add -- src/deepfix/artifact_retrieval tests/artifact_retrieval/test_tools.py
git commit -m "feat: expose diagnostic artifact tools"
```

### Task 6: 把成功检索记录为无强进展 Investigation 事件

**Files:**
- Modify: `src/deepfix/investigation/models.py:48-70`
- Modify: `src/deepfix/investigation/coordinator.py:174-292`
- Modify: `tests/investigation/test_coordinator.py`
- Modify: `tests/investigation/test_progress.py`
- Modify: `tests/investigation/test_stagnation.py`

**Interfaces:**
- Consumes: ToolMessage artifact 的 `result_type`, artifact IDs, line ranges, counts, hashes 和 `query_terms_hash`。
- Produces: `InvestigationEventType.ARTIFACT_SEARCHED`, `ARTIFACT_READ`；成功结果经现有 `record_observation()` 幂等提交。

- [ ] **Step 1: 写事件映射和 payload 白名单失败测试**

```python
def test_successful_artifact_search_records_no_progress_event(coordinator):
    result = ToolMessage(
        id="msg-search",
        tool_call_id="search-1",
        content="bounded excerpt",
        status="success",
        artifact={
            "result_type": "diagnostic_artifact_search",
            "artifact_ids": ["artifact_" + "a" * 32],
            "match_count": 1,
            "searched_artifact_count": 2,
            "truncated": False,
            "content_hashes": ["b" * 64],
            "query_terms_hash": "c" * 64,
        },
    )
    state = coordinator.record_tool_result(
        "task-a", {"name": "search_diagnostic_artifacts", "id": "search-1", "args": {"query": "secret"}}, result
    )
    event = coordinator.store.list_events("task-a")[-1]
    assert event.event_type == "artifact_searched"
    assert event.progress_kind is None
    assert "secret" not in str(event.payload)
    assert state.progress_generation == 0
```

- [ ] **Step 2: 写非强进展和停滞测试**

断言 `ProgressEvaluator._STRONG` 不包含两个新事件；连续检索增加 `no_progress_count`，在现有第六次边界进入 Level 1；`PHASE_CHANGED` 不重置；相同 ToolMessage 重放不重复事件。

- [ ] **Step 3: 运行目标测试确认失败**

Run: `pytest tests/investigation/test_coordinator.py tests/investigation/test_progress.py tests/investigation/test_stagnation.py -k artifact -v`

Expected: FAIL because event enum/mapping is absent。

- [ ] **Step 4: 在 EvidenceCollector 前识别专用成功结果**

```python
artifact = result.artifact if isinstance(result.artifact, Mapping) else {}
result_type = str(artifact.get("result_type", ""))
if result.status == "success" and result_type in {
    "diagnostic_artifact_search",
    "diagnostic_artifact_read",
}:
    event_type = (
        InvestigationEventType.ARTIFACT_SEARCHED
        if result_type.endswith("search")
        else InvestigationEventType.ARTIFACT_READ
    )
    return self.record_observation(
        task_id,
        self._tool_observation(
            event_type,
            call_id,
            result,
            signature,
            fingerprint,
            payload=_diagnostic_artifact_event_payload(result_type, artifact),
        ),
    )
```

`_diagnostic_artifact_event_payload()` 必须逐字段复制并校验 JSON-safe 标量/list，绝不复制 ToolMessage 正文、query 或 Backend path。error ToolMessage 继续走普通 `TOOL_COMPLETED`，不伪装成功检索。

- [ ] **Step 5: 运行 Investigation 测试并提交**

Run: `pytest tests/investigation -v`

Expected: PASS。

Run: `ruff check src/deepfix/investigation tests/investigation`

```powershell
git add -- src/deepfix/investigation/models.py src/deepfix/investigation/coordinator.py tests/investigation
git commit -m "feat: track diagnostic artifact retrieval events"
```

**Batch 2 审查点：** 检查普通错误/系统错误边界、返回预算、事件 payload 和 stagnation；确认检索不能成为 strong progress 后进入 Batch 3。

---

## Batch 3：Agent 装配、恢复和端到端兼容

### Task 7: 注册 Agent 工具、READ 能力、停滞许可和提示规则

**Files:**
- Modify: `src/deepfix/agent.py:85-205`
- Modify: `src/deepfix/prompts.py:20-98`
- Modify: `src/deepfix/investigation/coordinator.py:145-166,881-903`
- Modify: `tests/test_agent.py:20-155`
- Modify: `tests/test_approval.py`
- Modify: `tests/test_prompting.py`
- Modify: `tests/investigation/test_middleware.py`

**Interfaces:**
- Consumes: Task 5 两个 tool builders，现有 `InvestigationMiddleware` capabilities/permit。
- Produces: Agent 中不可被扩展覆盖的 `search_diagnostic_artifacts` 和 `read_diagnostic_artifact`，两者 `InvestigationCapability.READ`、L0、无审批。

- [ ] **Step 1: 写 Agent 组装和审批失败测试**

```python
assert {
    "search_diagnostic_artifacts",
    "read_diagnostic_artifact",
} <= agent.nodes["tools"].bound.tools_by_name.keys()
assert "search_diagnostic_artifacts" not in interrupt_middleware.interrupt_on
assert "read_diagnostic_artifact" not in interrupt_middleware.interrupt_on
```

再注册同名 extension，断言 `merge_extensions()` 拒绝覆盖。修改既有 middleware 捕获测试时，不把用户的 `LLMTraceMiddleware` 块纳入提交。

- [ ] **Step 2: 写阶段可见性和 permit 失败测试**

正常 investigating/diagnosing/testing 阶段按 READ 能力可见；Level 1 默认只显示三个 meta tools。通过
`continue_investigation(tool_name="search_diagnostic_artifacts", target="AssertionError")` 发放绑定 permit 后，
下一次模型调用只额外显示该获批工具且只允许一次搜索；第二次触发现有
`InvestigationStagnationError`。读取许可的 target 是完整 `artifact_id`。

- [ ] **Step 3: 写提示规则失败测试**

断言调查提示明确包含：完整结果卸载时使用专用工具；禁止普通 grep/read_file 扫 Artifact 根；不能根据截断预览猜测；检索后更新/支持/排除假设；真实 exit code 仍来自系统证据。

- [ ] **Step 4: 运行目标测试确认失败**

Run: `pytest tests/test_agent.py tests/test_approval.py tests/test_prompting.py tests/investigation/test_middleware.py -v`

Expected: FAIL on missing tools/capabilities/prompt text。

- [ ] **Step 5: 构造共享 Collector/Service 并注册工具**

在 `build_agent()` 已有 `resolved_backend`, `tasks`, `compaction`, `investigation` 构造完成后加入：

```python
artifact_collector = ArtifactReferenceCollector(compaction, resolved_backend)
artifact_service = DiagnosticArtifactService(tasks, artifact_collector, resolved_backend)
search_artifacts = build_search_diagnostic_artifacts_tool(artifact_service, investigation)
read_artifact = build_read_diagnostic_artifact_tool(artifact_service, investigation)
```

把两个名称加入 `core_tool_names`，两个对象加入 `tools`，能力都映射为 `InvestigationCapability.READ`；不要加入 `core_interrupts`。Prompt 只追加设计批准的固定规则，不改其他修复流程。

在 `allowed_tool_names()` 的 Level 1 分支中，当 permit 存在且尚未消费时，只额外加入
`permit.tool_name`；Level 2 或已消费 permit 不加入。`_target_from_arguments()` 对搜索工具返回折叠空白后的
query，对读取工具返回 `artifact_id`，使 `continue_investigation.target` 与实际调用使用同一稳定语义；
`artifact_kinds`、match/line 预算仍由固定上限约束，不能通过 permit 扩权。

- [ ] **Step 6: 逐 hunk 暂存、验证并提交**

Run: `pytest tests/test_agent.py tests/test_approval.py tests/test_prompting.py tests/investigation/test_middleware.py -v`

Run: `ruff check src/deepfix/artifact_retrieval src/deepfix/investigation src/deepfix/agent.py src/deepfix/prompts.py tests/artifact_retrieval tests/investigation tests/test_agent.py tests/test_approval.py tests/test_prompting.py`

先运行 `git diff -- src/deepfix/agent.py src/deepfix/prompts.py`，只交互式或逐 patch 暂存本任务 hunk，然后：

```powershell
git add -- src/deepfix/investigation/coordinator.py tests/test_agent.py tests/test_approval.py tests/test_prompting.py tests/investigation/test_middleware.py
git diff --cached --check
git commit -m "feat: assemble diagnostic artifact retrieval"
```

提交前 `git diff --cached --name-only` 不得包含 `src/deepfix/debug.py` 或 `docs/debug/`。

### Task 8: 验证系统错误穿过 BugfixService 后进入 PAUSED

**Files:**
- Modify: `tests/test_service.py:680-740`
- Modify: `tests/artifact_retrieval/test_tools.py`

**Interfaces:**
- Consumes: Task 5 Tool 适配层生成的 `InvestigationStateError`；现有 `BugfixService._invoke()` 无需生产改动。
- Produces: 回归证明 Artifact 组件不直接修改 TaskStatus，Service 是唯一 PAUSED 边界。

- [ ] **Step 1: 写 Service 恢复边界测试**

```python
class DiagnosticArtifactErrorAgent:
    def invoke(self, value, config):
        raise InvestigationStateError(
            InvestigationRecoveryMetadata(
                task_id=config["configurable"]["thread_id"],
                error_code="diagnostic_artifact_backend_read_failed",
                agent_phase="investigating",
                state_version=2,
                last_event_sequence=5,
                tool_call_id="search-1",
                checkpoint_available=True,
                recovery_action="pause_and_retry_diagnostic_artifact_read",
            )
        )
```

断言 Service 保存 `investigation_recovery`、状态为 PAUSED、没有 context recovery；跨 task recovery 仍 fail closed 为 FAILED。

- [ ] **Step 2: 写“组件不改状态”测试**

让 Tool 的 Backend 协议失败，在捕获 `InvestigationStateError` 后直接读取 TaskRepository，断言任务仍是 INVESTIGATING；只有把错误交给 `BugfixService._invoke()` 后才变为 PAUSED。

- [ ] **Step 3: 运行测试；仅在暴露现有边界缺陷时最小修复 Service**

Run: `pytest tests/artifact_retrieval/test_tools.py tests/test_service.py -k "diagnostic_artifact or investigation_error" -v`

Expected: PASS without production Service changes。若失败，先使用 `superpowers:systematic-debugging` 查明是测试夹具问题还是 `BugfixService` 未覆盖现有 `InvestigationCoordinationError`，不得引入 Artifact 专用 TaskState 字段。

- [ ] **Step 4: 提交恢复测试**

```powershell
git add -- tests/artifact_retrieval/test_tools.py tests/test_service.py
git commit -m "test: cover diagnostic artifact recovery boundary"
```

### Task 9: 添加真实 Deep Agents 卸载与压缩历史端到端测试

**Files:**
- Create: `tests/artifact_retrieval/test_workflow.py`
- Modify: `tests/test_backend.py`

**Interfaces:**
- Consumes: 公开 Deep Agents `FilesystemMiddleware`/`CompositeBackend` 行为、`DeepAgentsArtifactAdapter.persist_history()`、两个真实 StructuredTool。
- Produces: 离线兼容保证：当前大型结果与压缩后旧结果均可被专用工具找回。

- [ ] **Step 1: 写真实大型 ToolMessage 卸载兼容测试**

使用公开 Deep Agents Middleware 产生超过阈值的 ToolMessage，断言模型侧只留预览和
`/.deepfix-artifacts/large_tool_results/{sanitized_id}` 引用；随后调用 `search_diagnostic_artifacts("AssertionError")` 找到完整正文。测试只允许导入公开 Middleware/Backend 类，不导入 `_offload_tool_result` 等私有符号。

- [ ] **Step 2: 写压缩 history 发现旧结果测试**

```python
adapter.persist_history(
    "task-a",
    "attempt-1",
    ensure_message_ids("task-a", old_messages).messages,
    retained_ids=set(),
    work_unit_ids={"wu-old"},
)
```

保存并激活引用该 conversation history 的 Snapshot；当前 Graph Messages 不再含旧 ToolMessage。搜索应先找到 history，再从其中授权旧 large result；read 使用搜索返回的 `artifact_id` 成功返回指定行。

- [ ] **Step 3: 写跨任务与非目标 Artifact 端到端拒绝测试**

Backend 同时存在 task-b history、research 和 receipt 文件。task-a 两个 Tool 都不能得到这些 ID 或正文，也不能通过构造 `artifact_<hash>` 绕过目录。

- [ ] **Step 4: 运行兼容测试**

Run: `pytest tests/artifact_retrieval/test_workflow.py tests/test_backend.py -v`

Expected: PASS，测试不访问网络、不修改 fixture 项目。

- [ ] **Step 5: 提交端到端测试**

```powershell
git add -- tests/artifact_retrieval/test_workflow.py tests/test_backend.py
git commit -m "test: verify diagnostic artifact retrieval workflow"
```

**Batch 3 审查点：** 审查 Agent 工具可见性、审批、用户改动隔离、PAUSED 边界和公开 Deep Agents 兼容性；批准后执行最终验证。

---

### Task 10: 全量回归、静态检查和提交边界审计

**Files:**
- Verify only; do not modify production code unless a failing test identifies an in-scope defect through systematic debugging.

**Interfaces:**
- Consumes: Tasks 1–9 的全部实现。
- Produces: 可审计的最终测试证据和干净的本子项目提交集合。

- [ ] **Step 1: 运行新子项目和相关子系统测试**

Run: `pytest tests/artifact_retrieval tests/investigation tests/compaction -v`

Expected: all PASS。

- [ ] **Step 2: 运行 Agent、审批、Service 和 Backend 回归**

Run: `pytest tests/test_agent.py tests/test_approval.py tests/test_prompting.py tests/test_service.py tests/test_backend.py -v`

Expected: all PASS。

- [ ] **Step 3: 运行完整离线测试**

Run: `pytest`

Expected: all non-online tests PASS；既有配置中的 online tests 保持 deselected。

- [ ] **Step 4: 运行 Ruff 并区分用户既有 baseline**

Run: `ruff check src tests`

Expected: 新增/修改的本子项目文件无问题。若仍只出现用户既有 `src/deepfix/agent.py` debug import ordering 和 `src/deepfix/debug.py` 的 SIM117/BLE001，记录为 pre-existing，不擅自修复。

- [ ] **Step 5: 审计 staged/unstaged 和提交历史**

Run: `git status --short`

Run: `git diff --check`

Run: `git log --oneline --decorate -12`

确认用户的 `src/deepfix/debug.py`, `docs/debug/` 以及不属于本项目的 agent/prompts hunk 仍未被提交；没有 `.env`、API key、数据库或 Artifact 正文进入 Git。

- [ ] **Step 6: 使用完成前验证和代码审查技能**

调用 `superpowers:verification-before-completion` 核对最新命令输出，再调用 `superpowers:requesting-code-review` 做规格符合性和代码质量审查。审查发现问题时先走 `superpowers:receiving-code-review` 和相应 TDD 修复，不直接宣布完成。

## Execution Checkpoints

- Checkpoint 1：Tasks 1–3，审查授权目录和 history 配对。
- Checkpoint 2：Tasks 4–6，审查预算、错误分类和非强进展语义。
- Checkpoint 3：Tasks 7–9，审查 Agent 装配、Service 恢复和 Deep Agents 兼容。
- Final Checkpoint：Task 10，全量验证与提交边界审计。
