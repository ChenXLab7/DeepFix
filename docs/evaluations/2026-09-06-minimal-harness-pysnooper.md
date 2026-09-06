# 最小 Harness 删减与 PySnooper 实测

日期：2026-09-06。目标：删除生产调查控制，检查可信执行是否保持，以及同一 Case 的真实行为是否改善。

## 实现范围

- 原地删减 InvestigationMiddleware；没有新增执行框架、中间件文件、Store 或生命周期。
- 删除 duplicate 授权、progress/stagnation/reevaluation、checked_files 登记与调查状态提示。工具直接使用现有 TaskRepository、ExecutionRepository 和 EvidenceCollector。
- 同 ID 的恢复仍读 Receipt；新 ID 的重复工具调用正常执行。Evidence 提交失败仍保留可恢复的 OBSERVED Operation，不重新执行副作用。
- 删除生产 Todo 催促与自定义 Todo 提示，使用默认 TodoListMiddleware；默认不暴露 record_hypothesis。
- Artifact 检索、上下文、报告及正常 Resume 不再维护调查状态。旧 InvestigationRepository 仅按需供历史/评测代码使用，生产不主动初始化。
- 保留 Workspace、安全审批、Operation/Receipt、必要 Evidence、Checkpoint/压缩、Verification 及原有资源上限。

旧类型名称和不进入生产路径的研究代码未全面清理。没有增加 Run Budget、READY/yielded、旧数据迁移或模型调优。服务层原有 continue/blocked 路由未重写，只删除强制重评估的措辞；不能据此宣称已重做全部 Runtime 行为。

## 验证

最终离线回归：`939 passed, 3 skipped, 2 deselected`；`ruff check src tests` 通过。

测试更新移除了对已退休调查控制的断言，保留副作用持久化失败、Receipt 回放、并行读取、审批、Workspace、验证与同 Task 恢复测试。新增少量用例直接检查新 ID 重复执行、未知副作用不重放，以及上下文/报告不依赖 Investigation。测试数量减少并非测试覆盖增加的证据。

初次整体检查发现 45 个失败，来自旧接口/控制断言和本次修改中的测试接线；这些相关用例已经更新或随被删除的生产控制退休。没有通过恢复 Controller 使旧断言通过，也没有扩展到无关历史修复。

## 实验条件

- 基线 DeepFix：`67831726e8d212c42ac6d3c704ec780588b26edc`。
- 容器：`deepfix-bugsinpy`，DeepAgents `0.7.13`；本地单元测试依赖为 `0.7.8`。前后实测均在同一容器依赖下执行，没有升级依赖。
- 配置模型：主模型与压缩模型均为 `deepseek-v4-pro`；Provider 响应模型名为 `deepseek-v4-pro-ga-260813`。保持原有 `thinking=disabled`、temperature=0 和请求超时。
- 两组 graph step 上限均为 80、内部 Agent invocation 上限均为 30。相同配置不等于相同实际模型调用预算，因为 Graph 结构不同。
- PySnooper 来自 `/home/workspace/bug2/PySnooper` 的两份独立干净副本，metadata 中逐文件 hash 对比完全一致。两组独立 DEEPFIX_HOME、新 Task；不复用旧消息。
- 解释器为既有 `/home/python-venv-wrapper`。两组都先真实复现 `TypeError: __init__() got an unexpected keyword argument 'custom_repr'`。
- 保持 manual 模式；测试脚本只审批目标 pytest 命令及 pysnooper/ 生产文件操作，其他请求会交还。两组均只审批了目标 pytest，没有遇到范围外审批。

相同用户问题：

> PySnooper 的 tests/test_pysnooper.py::test_custom_repr_single 测试失败。请复现并修复这个 Bug。只修改 pysnooper/ 下的生产代码，不修改测试或依赖。使用 /home/python-venv-wrapper -m pytest -q -s tests/test_pysnooper.py::test_custom_repr_single 验证。搜索仅限 pysnooper/ 和 tests/，排除内部 Artifact、运行目录和缓存。

## 实际结果

| 项目 | 基线 | 删减后 |
|---|---|---|
| Task | dc79ea4bd07446ff913ea5bde899f3db | eef89f293f724581b217905d464db522 |
| 模型调用次数 | 6 | 17 |
| 发出的工具调用 | execute 1、grep 7、read_file 2、ls 1、write_todos 1 | execute 1、grep 17 |
| Python 文件修改 | 0 | 0 |
| 最后停止原因 | investigation_lifecycle_commit_failed | 单次 Agent step limit |
| 修复后通过验证 | 无 | 无 |

工具调用次数依据模型响应统计，不等于所有请求都成功执行或都进入下一轮请求；达到资源上限前的最后一次调用需与 Checkpoint/Receipt 区分。17 次模型调用也不是服务报告的 graph.invoke 次数。

### 基线

失败测试结果进入模型。随后模型尝试用真实操作系统绝对路径调用虚拟 read_file，收到 File not found；之后使用正确虚拟路径读取了 tracer.py 的第一页。重复 custom_repr 搜索收到 duplicate_read_correction，系统请求也有 strategy_feedback。

第 6 次模型调用后进入调查状态提交错误，未完成修改。错误记录为 `investigation_lifecycle_commit_failed`，恢复记录未保留底层异常详情。本轮不修复该旧控制器错误，不将其归因于模型能力。

### 删减后

失败测试结果进入模型。首次 custom_repr grep 返回：

```text
/bugsinpy_run_test.sh
/tests/test_pysnooper.py
```

`def __init__` 搜索返回 tracer.py、variables.py 等文件路径。之后继续发出相同 custom_repr 搜索，已进入后续请求的返回均为真实匹配路径，没有 duplicate correction、strategy_feedback 或调查签名投影，也没有截断提示。

本轮没有调用 read_file，故关键生产源码上下文没有通过读取进入模型；不能断言“模型已获得充分源码但无法理解”。同样不能把问题归因于返回丢失：搜索结果已出现在下一轮实际模型请求中。已确认的失败位置是：模型在获得文件位置后没有转入内容读取，而是持续选择同一 pattern-only 搜索。

默认 grep 模式未改变，仍只给出命中文件路径。这次去控制不足以让模型主动跨过“定位文件 → 阅读内容”这一步。

## 结论与限制

删除调查控制的行为已验证，现有可靠性边界的回归通过；但这次 autonomous 运行没有显示修复能力恢复。不能声称当前低修复表现主要由这些控制造成，也不能反过来证明控制完全无影响。

仅各一次运行，而且基线受状态提交错误影响；没有原生 Agent 参考组，不是正式通过率或因果评测。未继续调整模型、grep 默认值、预算或加入策略控制。下一步若继续调查，应围绕真实请求中的工具选择与信息获取开展独立实验，不把它混入本次删减提交。

## 原始记录位置

本地目录：`C:\Users\17823\Documents\AI Agent\minimal-harness-runs`。

- `run_case.py`：此次单 Case 运行脚本，不是 Benchmark 平台。
- `baseline-llm_calls.jsonl` / `candidate-llm_calls.jsonl`：实际模型请求、ToolMessage 与响应。
- `baseline-run.log` / `candidate-run.log`：命令、审批与最终报告。
- `baseline-metadata.json` / `candidate-metadata.json`：项目文件 hash 与配置白名单。
- `comparison.json`：从原始请求/响应提取的结果。
- `final-regression.txt`：最终回归结果。

源码归档 SHA256：

```text
baseline.tar  a425e4ad71db613b4c6bed11acbd7c92eb5a6f3681145880e5d5b6c2826a722e
candidate.tar 2fe16b29922edd60c56aa309bb2ee0443a8df503fd06262210b9329223ebbf9e
```

容器记录保留在 `/home/minimal-harness-runs`，没有覆盖原项目或旧 DeepFix 安装。原始日志和凭据不纳入 Git 提交；配置只记录白名单字段。
