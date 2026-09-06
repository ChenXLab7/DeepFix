# 持续 Debug / Bug Fix Harness 实施记录

基线：e60fb7d；实施分支：codex/deepfix-guided-debugging。

## 本次交付

- 同一 Bug Task、Workspace 和 LangGraph thread 持续使用。正常提问进入 WAITING_INPUT；PAUSED 表达运行受阻或用户停止。
- TaskInput 保留原文、类型、来源 ID 和替代关系。约束必须显式标为 constraint；一般补充仍作为用户陈述进入上下文，不认证为事实。原始问题不可变。
- TaskRun 只保存输入关联、身份和调用用量，不另存一套生命周期。输入与运行记录原子写入；审批恢复沿用当前 Run。每轮调用预算跨 Service 重建保留，累计用量从记录求和。
- OS 文件锁阻止同一 Task 并发恢复；进程退出自动释放，避免增加租约状态。锁文件位于数据库旁 task-locks 目录。
- 模型返回 continue、无具体问题的 blocked、或完成候选缺少验证时，Service 在同一 Run 中继续执行，直到明确交接或预算边界；不递归累积执行栈。
- duplicate read/execute/hypothesis 返回策略重评估反馈，不按次数触发暂停。Operation 身份重放和未知副作用恢复保护保持。读取签名包含 Workspace 代码版本，文件变化后可以重新读取。
- 移除旧诊断/修复门禁字段及补丁失败强迫拒绝根因的规则；假设表为当前权威，调查状态持久化时不再复制假设内容。
- 配置 Python/wrapper 路径参与同一个受控命令策略。审批前展示底层硬拒绝；批准不会越过硬边界。解释器调用路径不 resolve 成 venv 的基础解释器。
- 验证和导航共用当前代码/baseline 匹配。NOT_REPRODUCED 同样不能引用过期基线通过。pytest 中断、内部错误、未收集测试不算断言失败。
- 历史 Snapshot 重建只读取历史语义项；当前证据、假设和有效输入来自各自 Repository。
- CLI 支持持续回复、EOF 安全离开、--once、输入类型和更正；报告显示运行用量、用户贡献及交接状态。

## 使用

```powershell
deepfix new --project C:\projects\demo --python C:\projects\demo\.venv\Scripts\python.exe --once '修复解析失败'
deepfix resume TASK_ID '只在空字符串输入时发生'
deepfix resume TASK_ID --kind hypothesis '可能遗漏空输入处理'
deepfix resume TASK_ID --kind constraint '不要修改现有测试'
deepfix resume TASK_ID --kind constraint --supersedes INPUT_ID '允许添加回归测试'
```

默认交互会在需要用户信息时显示具体问题并接收回复；/pause 保留任务后退出。
--once 在正常信息交接时退出；副作用审批仍遵循 manual/guarded 模式。
用户可从任务报告中取得 input ID。输入更正仅允许同任务同类型引用。

## 验证策略

回归覆盖重复搜索后继续读取、文件变化后重读、失败补丁不强迫否定根因、解释器路径与硬拒绝、过期测试、输入/Run 原子性、并发恢复、跨 Service 用量、SQLite Graph 重建和长轮次 CLI。
生产模型能力和 Benchmark 通过率需独立实测；这些确定性测试不调用付费模型。
真实 POSIX venv 集成用例在 Windows 跳过，Windows 上另有路径保留回归。

## 本次没有声称完成的部分

- 没有新增 Multi-Agent、Task Graph 或通用代码任务能力。
- 当前活动状态仍复用现有 task_lifecycle 表；Run 是身份/用量记录，并非第二套状态机。尚未增加任务重新打开接口。
- 验证已绑定代码和 baseline，尚无完整依赖环境 fingerprint；同路径环境被外部替换后，应重新验证。
- 新用户输入不会自动改写 Required Oracle；验证策略的用户显式修订接口另行实现。
- 旧任务的数据库与 Graph 迁移适配仍保留；这两类兼容处理覆盖不同输入，不能简单互删。后续可独立迁出运行热路径。
- Experiment Loop 保持评测专用；未将它变成第二套生产调度。
- 修复保留在 Task Workspace；patch 导出、apply-back 与源仓库冲突处理仍未提供正式产品接口。
- Token 预留机制仍主要供评测使用；本次持久化的是 Run 的 Agent invoke 次数，不等同于全生产 token 预算闭环。

以上缺口不由新增控制层掩盖。后续按真实案例优先级分别处理环境适用性、验证策略修订和修复结果取用。
