# DeepFix 两模型与 `.env` 配置设计

日期：2026-08-22  
状态：待用户审阅

## 1. 目标

DeepFix 将模型调用拆分为两个明确角色：

1. **主修复模型（main model）**：理解用户问题、调查代码、调用工具、提出并执行修复、调用
   `save_progress`、生成最终 `RepairOutcome`。
2. **压缩提取模型（compaction model）**：只读取待压缩的完整 WorkUnit，并生成不可信的结构化
   `CompactionDelta` 候选；它不拥有工具，不修改项目，不决定任务状态，也不能覆盖系统确定性证据。

Working Memory 的机制保持不变：主修复模型在关键阶段主动调用 `save_progress`，观察区在覆盖过旧时
提醒它保存。此次不新增自动 Working Memory 模型调用。

用户可以为两个角色分别配置 DeepSeek 模型与 API Key，也可以只配置一个通用 Key。配置文件默认从
`src/deepfix/.env` 加载，同时保留系统环境变量优先级。

## 2. 非目标

- 不增加第三个 Working Memory 模型。
- 不改变 `save_progress` 的输入、校验、版本化或假设迁移接口。
- 不改变 CompactionSnapshot 的权威来源、合并规则和事务式提交顺序。
- 不支持在运行中的同一任务里动态切换模型。
- 不实现多供应商路由、负载均衡、Key 轮换或按 Token 实时选模。
- 不把 `.env`、API Key、鉴权错误中的敏感内容写入任务、SQLite、Artifact、报告或目标项目环境。

## 3. 配置接口

默认配置文件：

```text
<repository>/src/deepfix/.env
```

示例：

```dotenv
# 通用兜底 Key；只配置这一项即可运行
DEEPSEEK_API_KEY=sk-example

# 主修复模型
DEEPFIX_MAIN_MODEL=deepseek-v4-pro
DEEPFIX_MAIN_API_KEY=

# 压缩提取模型
DEEPFIX_COMPACTION_MODEL=deepseek-v4-flash
DEEPFIX_COMPACTION_API_KEY=

# 两个角色共用 DeepSeek API 地址
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

`.env` 使用 `python-dotenv` 加载。加载发生在 `load_config()` 读取配置之前，使用 `override=False`，因此
调用进程已经设置的环境变量始终优先于文件内容。

### 3.1 Key 回退规则

主模型 Key：

```text
DEEPFIX_MAIN_API_KEY
    -> DEEPSEEK_API_KEY
```

压缩模型 Key：

```text
DEEPFIX_COMPACTION_API_KEY
    -> DEEPFIX_MAIN_API_KEY
    -> DEEPSEEK_API_KEY
```

空字符串和纯空白按“未配置”处理。最终仍找不到 Key 时，`load_config()` 在创建任何模型、Backend 或
数据库连接之前抛出不包含 Key 内容的配置错误。

这个规则允许三种常见用法：

- 一个通用 Key：两个模型共用 `DEEPSEEK_API_KEY`。
- 一个主模型专用 Key：压缩模型回退复用它。
- 两个独立 Key：分别计费、限流或撤销。

### 3.2 模型名与默认值

- `DEEPFIX_MAIN_MODEL` 默认 `deepseek-v4-pro`。
- `DEEPFIX_COMPACTION_MODEL` 默认 `deepseek-v4-flash`。
- 两项都允许显式设置为 DeepSeek 当前可用的其他模型标识；DeepFix 不维护封闭的模型名白名单。
- 模型名为空或只有空白时视为配置错误，不静默退回旧模型。

默认组合优先保证复杂代码修复质量，同时让高频、结构化的压缩提取使用成本更低的模型。需要最低成本
时，用户可以把两个模型都设为 `deepseek-v4-flash`。

旧的 `deepseek-chat` 默认值被移除。已有代码若直接读取 `AppConfig.model_name`，迁移为读取
`AppConfig.main_model.model_name`；不保留含义模糊的可写别名。

### 3.3 Base URL

两个角色共用 `DEEPSEEK_BASE_URL`，默认 `https://api.deepseek.com`。本次不提供角色级 Base URL，
因为目标是配置 DeepSeek 的不同模型与 Key，而不是引入多供应商架构。

Base URL 必须是非空 HTTPS URL。错误消息只描述字段问题，不回显查询参数或用户信息。

## 4. 配置模型与 Secret 边界

新增不可变的角色配置对象，概念接口如下：

```python
@dataclass(frozen=True)
class ModelRoleConfig:
    model_name: str
    api_key: SecretStr
    base_url: str


@dataclass(frozen=True)
class AppConfig:
    ...
    main_model: ModelRoleConfig
    compaction_model: ModelRoleConfig
```

Key 使用 `pydantic.SecretStr`，只有模型工厂创建 `ChatDeepSeek` 时调用 `get_secret_value()`。以下位置禁止
接收或序列化角色 Key：

- `TaskState` 与 `TaskRepository`
- LangGraph configurable/state
- Working Memory、CompactionSnapshot 和 Deterministic Evidence
- Conversation/Research Artifact
- 报告、CLI 输出和异常文本
- `LocalShellBackend.env`

`ModelRoleConfig.__repr__`、dataclass 输出和测试失败信息均不能暴露明文 Key。

## 5. 模型构造与依赖注入

`src/deepfix/agent.py` 将含义模糊的 `build_model()` 拆为：

```python
build_main_model(config) -> ChatDeepSeek
build_compaction_model(config) -> ChatDeepSeek
```

两者都设置确定性所需的 `temperature=0`，但读取各自的 `model_name`、`api_key` 和共同 Base URL。

`build_agent()` 的依赖流变为：

```text
AppConfig.main_model
    -> build_main_model
    -> create_deep_agent(model=main_model)

AppConfig.compaction_model
    -> build_compaction_model
    -> CompactionCoordinator(model=compaction_model)
    -> CompactionDeltaGenerator
```

关键不变量：

- Agent 的正常 Model Call 和结构化 `RepairOutcome` 只使用主模型。
- `CompactionDeltaGenerator.generate/agenerate` 只使用压缩模型。
- `save_progress` 不接收模型实例，仍由主模型生成参数、Store 确定性校验并持久化。
- Protected Context、EvidenceCollector、WorkUnitPartitioner 和 SnapshotBuilder 都不调用模型。
- 自动压缩、主动 `compact_conversation` 和 Overflow 恢复共用同一个压缩模型实例。
- HarnessProfile 使用主模型的 provider/model key 注册，因为它约束的是主 Agent 图；压缩模型不是一个
  Deep Agent，不注册 HarnessProfile。

## 6. 上下文预算兼容

`ContextBudgetMonitor` 优先读取模型实例的 `profile.max_input_tokens`。如果 SDK 没有提供 profile，则显式
预算表加入 `deepseek-v4-flash` 和 `deepseek-v4-pro` 的上下文上限。预算表只用于安全估算，不决定模型
是否可用；未知模型仍要求 SDK profile 或显式配置，不能猜测上下文长度。

主请求的预算由主模型 profile 决定。压缩模型只处理选中的完整 WorkUnit，其上下文溢出属于
Delta 生成失败，继续遵守现有正常区直通、紧急区暂停和原消息保留规则。

## 7. `.env` 加载与路径规则

默认路径通过 `Path(__file__).with_name(".env")` 定位，因此与当前工作目录无关。从仓库根目录、目标项目
目录或安装后的 CLI 启动时，不会误读目标项目的 `.env`。

加载规则：

1. 若默认文件不存在，继续读取已有系统环境变量，不报文件缺失错误。
2. 若文件存在，以 UTF-8 读取并由 `python-dotenv` 解析。
3. `override=False`，系统环境变量优先。
4. `.env` 内容永远不复制到目标项目或 Artifact。
5. 仓库 `.gitignore` 的 `.env` 规则继续作为防误提交保护；README 只提供 `.env.example` 风格片段，
   不生成包含真实 Key 的文件。

本次不向上搜索仓库根目录或当前工作目录的 `.env`，避免 DeepFix 在修复第三方项目时意外加载该项目的
密钥。未来若需要自定义路径，应另行设计显式 `DEEPFIX_ENV_FILE`，本次不增加隐式搜索规则。

## 8. 初始化与错误传播

配置与模型初始化顺序：

```text
加载 src/deepfix/.env
    -> 解析并校验两个角色配置
    -> 创建主模型
    -> 创建压缩模型
    -> 构造 Agent/Coordinator
```

配置错误在 CLI 启动边界直接失败，不创建任务。运行期间：

- 主模型鉴权或调用失败沿现有 Agent 边界处理，任务进入 `FAILED`；错误摘要必须经过敏感信息清理。
- 压缩模型调用失败由 `CompactionCoordinator` 归类为 `delta_generation_failed`，沿已有分级策略处理：
  普通压缩区保留原消息并直通一次，紧急区通过类型化恢复异常令 Service 转为 `PAUSED`。
- 不因压缩模型失败回退调用主模型。模型回退只发生在配置 Key 解析阶段，运行时切换会使成本、审计和
  幂等行为不可预测。

## 9. 测试策略

### 9.1 配置测试

- 只设置 `DEEPSEEK_API_KEY` 时两个角色共用该 Key。
- 只设置 `DEEPFIX_MAIN_API_KEY` 时压缩模型回退使用主 Key。
- 两个角色 Key 均设置时保持隔离。
- 空白角色 Key 被视为缺失；所有回退都缺失时快速失败。
- 系统环境变量覆盖 `src/deepfix/.env`。
- 默认模型为 Pro/Flash，显式模型名分别生效。
- 非 HTTPS 或空 Base URL 被拒绝。
- `repr(AppConfig)`、异常和序列化中不出现测试 Secret。

### 9.2 Agent 装配测试

- `create_deep_agent` 收到主模型实例。
- `CompactionCoordinator.model` 是不同的压缩模型实例。
- 两个模型收到各自的模型名和 Key。
- `save_progress` 不持有压缩模型或新增模型依赖。
- HarnessProfile 按主模型名注册。

### 9.3 压缩与回归测试

- CompactionDelta 的同步和异步生成都调用压缩模型，不调用主模型。
- 主模型与压缩模型使用相同名称时仍是两个职责明确的实例。
- 压缩模型失败继续满足正常区直通、紧急区暂停、旧消息不删除。
- V4 模型预算识别覆盖四个阈值。
- 全量离线测试不发送真实 DeepSeek 请求。
- Backend 环境泄漏测试继续确认所有 DeepSeek Key 都不会进入目标 Shell。

## 10. 文档与使用示例

README 更新以下内容：

- 从 PowerShell 单一环境变量示例改为 `src/deepfix/.env` 示例。
- 解释一个 Key 与两个独立 Key 的配置方式。
- 解释主模型、压缩模型和 Working Memory 的真实职责。
- 给出成本优先与质量优先两套模型组合。
- 明确 `.env` 不应提交，目标项目 Shell 无法读取 Key。
- 删除把 `deepseek-chat` 作为当前默认模型的说明。

## 11. 验收标准

1. 用户只配置一个 Key 即可启动，两个模型角色都可用。
2. 用户配置两个 Key 时，主调用和压缩调用严格使用各自 Key。
3. 用户可独立配置两个 DeepSeek 模型名。
4. Working Memory 仍由主 Agent 调用 `save_progress`，没有隐藏的额外模型费用。
5. CompactionDelta 只由压缩模型生成，主模型不承担压缩提取调用。
6. `.env` 的值不覆盖进程环境变量，也不会泄漏到任务、日志、报告、Artifact 或目标项目 Shell。
7. 现有上下文压缩、失败恢复、审批、研究证据和完成条件测试保持通过。
