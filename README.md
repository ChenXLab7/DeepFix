# DeepFix Agent

DeepFix 是一个基于 `deepagents` SDK 的 Python 代码缺陷修复 Agent。它从终端接收问题，调查项目、运行测试、提出最小修改，并在具有副作用的操作前执行风险审批。第一版坚持单 Agent：一个 Repair Agent 负责澄清、调查、修改、验证和报告，先把闭环可靠性做好，再考虑拆分多 Agent。

## 安装与首次运行

```powershell
cd 'C:\Users\17823\Documents\AI Agent\deepfix-agent'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
$env:DEEPSEEK_API_KEY = "your-key"
deepfix new --project 'C:\projects\broken-python-app' --mode manual '运行 pytest 时 test_divide 失败'
```

不要把真实 API Key 写进源码、配置样例、测试或 Git 历史。

如果目标项目使用的不是当前终端里的 Python，显式传入它的解释器。DeepFix 会把该路径保存在任务中，恢复任务时继续使用同一个环境：

```powershell
deepfix new --project 'C:\projects\broken-python-app' `
  --python 'C:\projects\broken-python-app\.venv\Scripts\python.exe' `
  --mode manual '运行 pytest 时 test_divide 失败'
```

## 终端命令

```powershell
# 创建任务；manual 模式会询问所有 L1/L2 操作
deepfix new --project 'C:\projects\broken-python-app' --mode manual '除法测试失败'

# guarded 模式自动允许常规项目内修改、pytest 和 Ruff，L2 仍询问，L3 仍拒绝
deepfix new --project 'C:\projects\broken-python-app' --mode guarded '除法测试失败'

# 查看任务 ID 和状态；该命令不需要初始化模型或读取 API Key
deepfix list

# 给澄清中的任务补充信息，或恢复暂停/待审批任务
deepfix resume TASK_ID 'Python 3.12，完整堆栈位于 tests 输出中'
deepfix resume TASK_ID
```

审批提示只接受：`a` 批准、`r` 拒绝、`q` 暂停。选择 `q` 会保留原任务 ID、图检查点和待审批动作，之后可以继续恢复。

## 核心架构

```text
Terminal CLI
    │
    ▼
BugfixService ─────────────── TaskRepository(SQLite)
    │                              业务状态、审批、证据、报告指标
    ▼
Single Repair Agent ───────── LangGraph Checkpointer(SQLite)
    │                              消息历史与图恢复位置
    ├── read/search Tools
    ├── file/shell Tools + HITL
    ├── research Tools ────── ResearchEvidenceStore(SQLite)
    │                              查询、候选、验证状态；按 task_id 隔离
    │          └───────────── DEEPFIX_HOME/artifacts/research
    │                              清洗后的完整外部正文
    ├── save_progress ─────── WorkingMemoryStore(SQLite)
    │                              版本化事实、证据、假设、下一步
    └── compact_conversation
             │
             └────────────── DEEPFIX_HOME/artifacts
                              完整历史与大型 Tool 输出
```

四层数据各自解决不同问题：

- Checkpointer 保证多轮任务和审批中断可以恢复。
- WorkingMemoryStore 保证有损摘要不会删除关键事实和下一步。
- CompactionSnapshot 保存结构化压缩历史、来源和生命周期。
- Conversation Artifact 保存被压缩消息的完整可恢复副本。

### 证据保真的上下文管理

DeepFix 不再采用固定的“70% 时压缩、保留最近 15%”，也不依赖 Deep Agents 的自然语言摘要作为运行时记忆。每次模型调用都会动态注入三个不可压缩保护块：Task Anchor、完整 Working Memory 和系统确定性证据。测试退出码、文件操作、审批和研究验证状态分别来自对应 Store；模型只能生成带来源的语义事实候选和假设，不能覆盖这些权威记录。

预算分为四个区域：

| 使用率 | 行为 |
|---|---|
| 0%～75% | 正常运行 |
| >75%～82% | 观察区；Working Memory 覆盖过旧时提示保存 |
| >82%～90% | 按完整工作单元执行普通压缩 |
| >90% | 紧急压缩；无法安全准备时暂停任务 |

一个工作单元包含操作目的、AI Tool Call、全部配对 ToolMessage，以及 Assistant 对结果的解释；并行 Tool Call 也属于同一单元。压缩时整个单元保留或整个进入 Snapshot，无法确认边界时优先保留。所有 Graph Message 都有稳定 ID：已有 `message.id` 原样复用，缺失 ID 根据任务、原始序号、消息类型、Tool Call ID 和规范化内容确定性生成。WorkUnit、来源、覆盖范围和 history 幂等都使用这些 ID。

Agent 可以主动调用 `compact_conversation`，自动和主动入口共享 DeepFix Compaction Coordinator。协调器严格按“写入并回读完整 history artifact → 构造并校验 Snapshot → 保存 prepared Snapshot → 模型调用成功后提交 event”的顺序运行。Snapshot 具有 `prepared`、`active`、`abandoned` 生命周期；真正生效的版本始终由 `_deepfix_compaction_event.active_snapshot_version` 决定。后续压缩按字段合并旧 Snapshot、新工作单元、最新 Working Memory 和系统证据，不反复总结旧自然语言摘要。

普通压缩区的准备失败会保留原消息、记录失败，并允许原请求直通一次；主动压缩在非紧急区失败会返回 error ToolMessage。紧急区失败或一次最小安全上下文重试后仍然 Overflow 时，`BugfixService` 把任务转为 `PAUSED`，并保存 `context_recovery`（失败阶段、错误码、Snapshot/Artifact 引用及“原消息是否保留”）。恢复时继续使用原 task ID 和 checkpoint；恢复成功后才清除该字段。

完整 conversation history、大型 Tool 结果和媒体仍写在 `DEEPFIX_HOME/artifacts` 下，通过独立 Backend 路由保存，不会写入目标项目。Snapshot 只保留有界结构化状态与 artifact 引用。

## 技术资料证据流

LLM 的训练知识可能过期，因此 Agent 可以查找佐证，但外部网页永远只是线索，不能替代本地源码和测试。单 Agent 内注册了四个边界清晰的工具：

1. `inspect_dependency(package_name)`：只读目标项目的 `pyproject.toml`、`requirements.txt`、`poetry.lock`、`uv.lock`，再通过 `--python` 指定的解释器查询已安装版本；不访问网络。
2. `search_technical_sources(query, package_name)`：先拒绝密钥、用户路径、内部域名、长代码/日志等敏感查询，再检索 PyPI、确认后的官方 GitHub 仓库和可选 Tavily 官方域名结果。查询和候选项写入 SQLite。
3. `fetch_external_evidence(candidate_id)`：只能抓取当前任务已经保存的候选项，不能接收任意 URL。每次请求和重定向都重新执行 HTTPS、DNS、公网地址和域名校验；正文清洗后写入 artifact，SQLite 只保存有界摘要和元数据。
4. `link_external_evidence(...)`：把外部结论关联到真实、成对出现的 `execute` Tool Call/ToolMessage。`verified` 必须引用整数 `exit_code=0` 的 pytest；失败测试只能支持 `contradicted`，伪造 ID 不能改变状态。

证据等级描述“来源本身有多可靠”，本地验证状态描述“它是否适用于当前项目”，二者不能混为一谈：

| 等级 | 含义 | 典型来源 |
|---|---|---|
| E1 | 官方一手资料 | PyPI 元数据、官方文档、官方源码、Release |
| E2 | 有维护者或仓库状态佐证 | 已合并 PR、已关闭 Issue、维护者回复、已回答 Discussion |
| E3 | 未确认的外部线索 | 普通用户尚未确认的 Issue/Discussion |

一条 E1 资料也可能与当前项目版本不一致；一条 E3 线索经过真实本地测试后也可以成为有效支持。报告会明确区分“仅为外部线索”“已通过真实测试关联”和“已被本地证据推翻”，并保留版本差异警告。

### 可选在线能力

只配置 `DEEPSEEK_API_KEY` 时，Agent 仍可完成本地修复，并可使用无密钥的 PyPI 和 GitHub REST 公共查询。额外配置都是可选的：

```powershell
# 提高 GitHub API 配额，并启用官方仓库 Discussions 查询
$env:GITHUB_TOKEN = "your-github-token"

# 仅在两项同时存在时启用 Tavily；结果仍限制在已确认的官方域名
$env:DEEPFIX_SEARCH_PROVIDER = "tavily"
$env:TAVILY_API_KEY = "your-tavily-key"
```

缺少 GitHub/Tavily Token 不会阻止启动，也不会阻止本地调查、修改和验证。Token 只在构造请求时读取，不会进入任务配置、候选记录、错误文本或目标项目 Shell 环境。

## 审批与安全模型

| 等级 | 示例 | manual | guarded |
|---|---|---|---|
| L0 | `read_file`、`grep`、`glob` | 自动允许 | 自动允许 |
| L1 | 项目内编辑、`pytest -q`、`ruff check` | 人工审批 | 自动允许 |
| L2 | 安装依赖、组合命令、未知命令、删除 | 人工审批 | 人工审批 |
| L3 | `git reset --hard`、递归强制删除、关机命令 | 强制拒绝 | 强制拒绝 |

`LocalShellBackend` 始终以目标项目为根目录，并通过显式环境白名单启动 Shell，`DEEPSEEK_API_KEY` 不会传给项目命令。DeepFix 自己的历史文件通过 `CompositeBackend` 写到 `DEEPFIX_HOME/artifacts`，不会污染目标项目 Git 工作区。

研究工具不会削弱原有审批策略：只读依赖检查和经过约束的搜索/抓取按登记风险执行；项目写入、删除、Shell、安装依赖及未知工具仍由 HITL 策略独立判断。外部正文被包裹为“不可信来源”，不能把网页里的提示当成 Agent 指令。

研究数据按 `task_id` 隔离。候选 ID、证据 ID、真实 URL 和 artifact 路径由系统生成或查询，模型不能在公开 Tool 参数中提供 task ID、任意 URL、Token 或保存路径。完整正文位于：

```text
DEEPFIX_HOME/artifacts/research/<task_id>/<evidence_id>.md
```

这不是操作系统级沙箱。真实项目运行前仍应使用一次性副本、容器或受限账户，并审查每一个 L2 操作。

## 确定性完成条件

模型说“已经修复”不能让任务完成。`BugfixService` 只根据真实 `execute` ToolMessage 中的 `exit_code` 接受测试证据；没有至少一次通过测试时，`completed` 会被改为 `PAUSED`。最终报告同样从任务状态、退出码、工作记忆版本和上下文指标渲染，不相信模型措辞。

## 开发验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .

# 默认测试永不访问真实网络；online 用例会被自动排除
.\.venv\Scripts\python.exe -m pytest --collect-only -q

# 显式在线冒烟测试。没有开关时会安全跳过
$env:DEEPFIX_RUN_ONLINE = "1"
.\.venv\Scripts\python.exe -m pytest -m online tests/research/test_online.py -q
```

确定性测试不调用真实 DeepSeek，覆盖版本并发、任务隔离、ToolRuntime 身份、上下文注入、Backend 路由、主动压缩去重、溢出暂停、HITL、CLI 交互，以及从依赖检查、官方检索、E1/E2/E3 抓取、真实 pytest 关联、状态同步到报告渲染的完整离线证据链。

## 面试演示：一次性 Python Bug 项目

不要直接拿重要仓库做现场演示。下面先创建一个可随时删除的一次性项目，其中 `discount` 的边界条件故意写错：

```powershell
$demo = Join-Path $env:TEMP 'deepfix-interview-demo'
New-Item -ItemType Directory -Force $demo | Out-Null
@'
def discount(total: int) -> int:
    return 10 if total > 100 else 0
'@ | Set-Content -Encoding utf8 (Join-Path $demo 'pricing.py')
@'
from pricing import discount

def test_discount_starts_at_100():
    assert discount(100) == 10
'@ | Set-Content -Encoding utf8 (Join-Path $demo 'test_pricing.py')
python -m pytest -q $demo
deepfix new --project $demo --python (Get-Command python).Source --mode manual `
  'pytest 的 100 元折扣边界测试失败；请调查根因、最小修复并验证'
```

演示时按顺序讲五件事：Agent 先用本地证据复现；有版本/API 疑问时才查官方资料；网页只能成为候选线索；写文件和 Shell 受审批策略控制；最终完成状态只认真实 pytest 的 `exit_code`。最后展示任务 ID、审批记录、修改文件、测试结果、外部证据分类及 artifact 路径。这样面试官看到的是一条可审计的工程闭环，而不是“模型说修好了”。

## 多 Agent 演进方向

当前没有 Investigator 子 Agent。只有当真实任务表明“仓库搜索、日志阅读、调用链分析和反复实验”长期占据主 Agent 上下文时，才把只读调查提取为 Investigator。它应返回结构化、带证据的 `InvestigationReport`，主 Agent 仍负责审批、修改、验证和最终结论。拆分依据是上下文隔离收益，而不是为了展示更多 Agent。

## 真实演示验收

真实模型演示必须先把故障样例复制到一次性目录，再运行 `deepfix new`。验收记录需要包含真实任务 ID、审批选择、修改文件、目标测试退出码和最终报告；API Key 与用户主目录必须脱敏。本仓库不伪造尚未执行的在线演示记录。
