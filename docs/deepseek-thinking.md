# DeepSeek thinking

2026-09-07 起，主修复模型发送 `thinking.type="enabled"`；压缩模型保持 `disabled`，不改变现有结构化压缩行为。

当前安装的 SDK 未在请求序列化时回传 `reasoning_content`。`agent.py` 中的小型 ChatDeepSeek 适配保留该字段，并将主模型结构化工具的强制选择转换为 `auto`，避免 thinking 接口拒绝强制 tool_choice。原有结果 schema 和最终 Verification 不变。

已有非 thinking 会话中的 assistant 消息使用空 reasoning_content；不会生成或猜测历史推理内容。是否能成功继续具体旧任务仍需实际恢复验证。

验证：15 项模型装配/Checkpoint 定向测试通过，ruff 通过；使用本机配置实际完成两轮无副作用 API 调用，首轮返回 thinking 并调用探测工具，第二轮收到工具结果后调用 RepairOutcomeCandidate。未重跑 PySnooper，也不据此声称修复能力提高。

容器当时未运行，Docker 旧安装未同步。启动容器后需更新其中代码才能使用此配置。

接口说明：https://api-docs.deepseek.com/guides/thinking_mode/
