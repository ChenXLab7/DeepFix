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
    ├── save_progress ─────── WorkingMemoryStore(SQLite)
    │                              版本化事实、证据、假设、下一步
    └── compact_conversation
             │
             └────────────── DEEPFIX_HOME/artifacts
                              完整历史与大型 Tool 输出
```

三层数据各自解决不同问题：

- Checkpointer 保证多轮任务和审批中断可以恢复。
- Summarization Middleware 保证当前对话能放进模型上下文。
- WorkingMemoryStore 保证有损摘要不会删除关键事实和下一步。

Agent 可以主动调用 `compact_conversation`；如果它忘记，70% 上下文阈值会自动压缩，压缩后保留最近约 15%。主动和自动压缩共享同一个引擎和 `_summarization_event` 状态。压缩前，系统提示要求 Agent 先调用 `save_progress` 保存完整进度快照。

## 审批与安全模型

| 等级 | 示例 | manual | guarded |
|---|---|---|---|
| L0 | `read_file`、`grep`、`glob` | 自动允许 | 自动允许 |
| L1 | 项目内编辑、`pytest -q`、`ruff check` | 人工审批 | 自动允许 |
| L2 | 安装依赖、组合命令、未知命令、删除 | 人工审批 | 人工审批 |
| L3 | `git reset --hard`、递归强制删除、关机命令 | 强制拒绝 | 强制拒绝 |

`LocalShellBackend` 始终以目标项目为根目录，并通过显式环境白名单启动 Shell，`DEEPSEEK_API_KEY` 不会传给项目命令。DeepFix 自己的历史文件通过 `CompositeBackend` 写到 `DEEPFIX_HOME/artifacts`，不会污染目标项目 Git 工作区。

这不是操作系统级沙箱。真实项目运行前仍应使用一次性副本、容器或受限账户，并审查每一个 L2 操作。

## 确定性完成条件

模型说“已经修复”不能让任务完成。`BugfixService` 只根据真实 `execute` ToolMessage 中的 `exit_code` 接受测试证据；没有至少一次通过测试时，`completed` 会被改为 `PAUSED`。最终报告同样从任务状态、退出码、工作记忆版本和上下文指标渲染，不相信模型措辞。

## 开发验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

确定性测试不调用真实 DeepSeek，覆盖版本并发、任务隔离、ToolRuntime 身份、上下文注入、Backend 路由、主动压缩去重、溢出暂停、HITL 和 CLI 交互。

## 多 Agent 演进方向

当前没有 Investigator 子 Agent。只有当真实任务表明“仓库搜索、日志阅读、调用链分析和反复实验”长期占据主 Agent 上下文时，才把只读调查提取为 Investigator。它应返回结构化、带证据的 `InvestigationReport`，主 Agent 仍负责审批、修改、验证和最终结论。拆分依据是上下文隔离收益，而不是为了展示更多 Agent。

## 真实演示验收

真实模型演示必须先把故障样例复制到一次性目录，再运行 `deepfix new`。验收记录需要包含真实任务 ID、审批选择、修改文件、目标测试退出码和最终报告；API Key 与用户主目录必须脱敏。本仓库不伪造尚未执行的在线演示记录。
