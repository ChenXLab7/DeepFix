我刚对 DeepFix Agent 做了一次真实单 Bug 实验，发现当前 Agent 存在两个需要一起解决的问题：

1. Agent 可能长期停留在 `investigating` 阶段，重复调用 Shell/LLM，但始终不进入修改阶段。
2. CLI 运行过程中缺乏清晰的用户可见进度，我经常无法判断 Agent 当前在调查什么、是否已经定位根因、是否已经修改、测试到了什么程度。

请基于下面的完整实验现象，在当前仓库中定位根因并做系统性修复，不要只针对 gcd 样例打补丁。

## 一、实验场景

测试项目是 QuixBugs，目标 Bug 为 `gcd`。

用户任务：

```text
gcd 功能的相关测试存在失败。请调查根因，在不修改测试和测试数据的前提下进行最小必要修改，并运行 python -m pytest python_testcases/test_gcd.py -q --timeout=5 验证修复结果。
```

修复前独立运行：

```powershell
python -m pytest python_testcases/test_gcd.py -q --timeout=5
```

结果：

```text
5 failed, 1 passed
```

失败均为：

```text
RecursionError: maximum recursion depth exceeded
```

相关实现：

```python
def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(a % b, b)
```

pytest traceback 已经明确显示递归不断停留在：

```python
return gcd(a % b, b)
```

正常的修复 Agent 应该在读取实现、测试并复现后，很快形成“递归参数没有正确推进”的根因假设，必要时进行一个最小验证，然后修改代码并重新运行目标测试。

## 二、实际 DeepFix 行为

本次任务 ID：

```text
bd24d49b3fca4d74a722f79542836aab
```

当前已经存在 `LLMTraceMiddleware`，用于记录每一次主模型调用的：

- call id
- task id
- trigger
- 最终 system prompt
- messages
- Graph State
- tool calls
- response
- token 使用
- context 大小

本次 gcd 任务最终发生了 **27 次主模型调用**。

前 7 次基本正常：

```text
#1  ls
#2  ls
#3  read_file
#4  read_file / ls
#5  read_file
#6  read_file
#7  execute 目标 pytest
```

第 7 次已经执行：

```text
python -m pytest python_testcases/test_gcd.py -q --timeout=5
```

并获得完整的 `RecursionError` 证据。

但是从第 8 次开始，Agent 没有形成 diagnosis，也没有调用 `edit_file`，而是进入：

```text
LLM
→ execute
→ ToolMessage
→ LLM
→ execute
→ ToolMessage
→ ...
```

的循环。

模型连续执行大量类似：

```powershell
python -c "print(13 % 13); ..."
python -c "print(37 % 600, 600 % 37, ...)"
python -c "print(20 % 100, 100 % 20, ...)"
python -c "print(624129 % 2061517, ...)"
python -c "print(3 % 12, 12 % 3)"
```

的手工取模实验。

后半段已经明显开始重复之前做过的计算，例如：

```text
13 % 13
20 % 100, 100 % 20
3 % 12, 12 % 3
37 % 600, 600 % 37, ...
```

被多次重复执行。

最终：

```text
没有 edit_file
没有 diagnosis
没有 repair_plan
没有 Working Memory
没有通过测试
```

任务因为：

```text
已达到 Shell 最大执行次数
```

而终止。

## 三、关键状态问题

这 27 次主模型调用中，System Prompt 的 phase 始终都是：

```xml
<deepfix_phase name="investigating">
...
</deepfix_phase>
```

同时 Working Memory 始终是：

```xml
<deepfix_working_memory version="none">
...
</deepfix_working_memory>
```

也就是说整个任务从第一次到最后一次调用：

```text
phase = investigating
working_memory = none
```

一直没有变化。

当前 `PromptPolicyMiddleware` 的核心逻辑大致为：

```python
latest = self.store.latest(task_id) if task_id else None
phase = latest.snapshot.phase if latest is not None else "investigating"
```

然后每次 Model Call 都根据这个 phase 注入：

```python
PHASE_PROMPTS[phase]
```

因此，只要：

```text
WorkingMemoryStore.latest(task_id) == None
```

phase 就永远回退成：

```text
investigating
```

而 investigating prompt 又包含类似语义：

```text
建立可证伪的根因假设，用最小实验逐个验证。
根因未得到证据支持前不要修改代码。
```

于是形成：

```text
没有 Working Memory
        ↓
phase 默认 investigating
        ↓
每轮 LLM 都继续收到 investigating prompt
        ↓
模型倾向继续调查和实验
        ↓
继续 execute
        ↓
ToolMessage 触发下一次 Model Call
        ↓
Working Memory 仍不存在
        ↓
phase 仍然 investigating
        ↓
继续调查
```

本次 gcd 实际进入了这个循环。

同时还观察到状态不一致：

```text
Task Anchor:
task_status = testing
```

但同一次模型调用中：

```text
deepfix_phase = investigating
```

仍然存在。

说明当前至少存在两套没有可靠同步的阶段概念：

```text
TaskState / TaskStatus
```

和：

```text
WorkingMemory.snapshot.phase
```

而 `PromptPolicyMiddleware` 当前把后者作为 phase prompt 的来源。

## 四、对照实验

同一个系统此前修复 `to_base` Bug 时能够正常完成。

成功链路大致为：

```text
读取项目
→ 读取实现和测试
→ pytest
→ save_progress
→ edit_file
→ pytest
→ save_progress
→ RepairOutcome
```

而 gcd 任务中模型没有主动调用 `save_progress`，于是 phase 一直没有离开 `investigating`。

这说明当前阶段推进在很大程度上取决于：

```text
LLM 是否恰好主动调用 save_progress
```

这个设计不够可靠。

`save_progress` 应主要承担 Working Memory 持久化，例如：

```text
facts
evidence
hypotheses
checked_files
experiments
next_steps
coverage
```

不应该事实上成为 Agent 从 investigating 推进到其他阶段的唯一控制通道。

## 五、需要解决的阶段推进问题

请重点检查：

```text
PromptPolicyMiddleware
WorkingMemoryStore
save_progress
PHASE_PROMPTS
BugfixService
TaskState / TaskStatus
ProtectedContext
Agent tool-loop
```

目标是把：

```text
业务任务状态
Agent 推理阶段
Working Memory 持久化
```

三者职责理清。

### 1. Phase 推进不能完全依赖 save_progress

必须消除：

```text
Working Memory 不存在
→ phase 永远 investigating
```

这种情况。

需要明确：

```text
TaskStatus
Agent reasoning phase
WorkingMemory.snapshot.phase
```

谁是 source of truth。

如果 phase 属于任务执行状态，应由确定性的状态机制维护；Working Memory 中的 phase 可以作为持久化快照，但不应反过来成为整个流程推进的唯一驱动。

### 2. PromptPolicyMiddleware 必须获得可靠 phase

不能简单：

```python
latest.snapshot.phase if latest else "investigating"
```

然后整个任务永久卡在调查阶段。

需要设计明确、可测试的 phase resolution 机制。

例如业务状态已经进入：

```text
editing
testing
reviewing
```

时，模型侧不能继续无条件收到 `investigating` prompt。

同时要考虑 interrupt/resume，不能为了修复这个问题破坏现有状态生命周期。

### 3. 保留 Working Memory 原有职责

Working Memory 仍负责长期保存：

```text
facts
evidence
hypotheses
checked_files
experiments
next_steps
coverage
```

以及服务于长任务、恢复和 Compaction。

但：

```text
Working Memory = none
```

不能等价于：

```text
Agent 必须继续 investigating
```

### 4. 增加无进展调查循环检测

目前只有 `max_shell_calls` 最终上限，这个保护太晚。

需要识别类似：

```text
连续多轮只进行 read/grep/execute
+
没有新增 evidence
+
没有新增 hypothesis
+
没有新增文件发现
+
没有 phase 推进
+
工具参数高度重复
```

的 stagnation 状态。

不要简单禁止相同工具重复执行，因为真实 debugging 允许重新跑 pytest。

重点检测的是：

```text
重复动作 + 没有新增信息 + 没有状态进展
```

检测到后，可以要求模型：

```text
重新评估已有证据
形成当前最可能根因
说明继续实验能够区分什么假设
```

如果无法给出新的可证伪调查目标，则不要继续无限执行 Shell；必要时安全暂停。

### 5. 不要简单强制修改代码

不能变成：

```text
pytest fail → 自动 editing
```

DeepFix 仍应保持：

```text
调查
→ 有证据支持的 hypothesis
→ 最小修改
→ 测试验证
```

本次需要解决的是：

```text
证据已经足够，但 Agent 无法结束调查
```

而不是取消调查阶段。

## 六、新增：CLI 用户可见运行进度

除了修复循环问题，还需要增加一个轻量的 **运行时进度展示机制**。

目前 Agent 长时间运行时，用户只能看到工具审批或最终结果，无法直观知道：

```text
现在处于什么阶段
正在调查什么
是否已经复现
是否已经定位根因
是否进行了修改
当前测试结果是什么
为什么还在继续
```

这会导致即使 Agent 没有真正卡死，用户也很难区分：

```text
正常调查
```

和：

```text
已经进入无效循环
```

### 1. CLI 希望呈现的效果

例如：

```text
[Investigating] 正在定位 gcd 相关实现和测试
[Investigating] 正在运行目标测试
[Investigating] 已复现：5 failed, 1 passed，主要错误为 RecursionError
[Diagnosing] 当前假设：递归参数未正确推进
[Editing] 已确认根因，准备修改 python_programs/gcd.py
[Editing] 已修改 python_programs/gcd.py
[Testing] 正在重新运行 python_testcases/test_gcd.py
[Testing] 6 passed
[Reviewing] 验证通过，正在生成最终报告
```

如果 Agent 进入重复调查，也应该能看到：

```text
[Investigating] 正在验证递归参数变化
[Investigating] 调查连续多轮无新增证据，正在重新评估当前假设
```

这样用户能清楚知道 Agent 到了哪一步。

### 2. 不要让 LLM自由生成进度文本

进度信息应尽量来自**确定性事件和状态**，例如：

```text
TaskStatus transition
Tool Call
Tool Result
pytest exit_code
changed_files
approval state
Working Memory update
stagnation detection
```

不要为了显示进度再额外调用 LLM，也不要让模型随意声称：

```text
已定位根因
已修复
测试通过
```

这些关键状态必须由系统事实支持。

### 3. 建议使用结构化 Progress Event

可以结合现有架构设计类似：

```python
ProgressEvent(
    task_id=...,
    phase="testing",
    event="test_started",
    message="正在运行 python_testcases/test_gcd.py",
)
```

或者等价实现。

重点不是具体类名，而是尽量让：

```text
业务事件
→ Progress Event
→ CLI renderer
```

分层。

不要把大量 `print()` 散落到 Agent、Tool 和 Repository 中。

### 4. 需要展示的最小信息

建议至少包含：

```text
当前阶段
当前动作
刚完成的关键结果
修改文件
测试状态
审批等待
暂停/失败原因
```

例如文件修改：

```text
[Editing] 已修改 python_programs/gcd.py
```

测试失败：

```text
[Testing] 目标测试失败：5 failed, 1 passed
```

测试成功：

```text
[Testing] 目标测试通过：6 passed
```

等待审批：

```text
[Approval] 等待批准 execute: ...
```

安全暂停：

```text
[Paused] 连续调查未产生新证据，任务已暂停
```

### 5. 不要把内部噪声全部展示给用户

不需要显示：

```text
每一次 Model Call
完整 prompt
完整 Working Memory
完整 Graph State
token 细节
内部 message ID
```

这些继续留在 `LLMTraceMiddleware` 调试日志里。

正常 CLI 应该是**高层进度**，Debug Trace 是**底层诊断**，两者不要混在一起。

推荐关系：

```text
LLMTraceMiddleware
→ 开发者调试

Progress Events
→ 普通用户运行时观察
```

## 七、需要增加的回归测试

请新增确定性测试，不调用真实 DeepSeek。

至少覆盖：

### Case A：没有 Working Memory

```text
WorkingMemoryStore.latest(task_id) == None
```

不能导致 phase 永远错误固定在 investigating。

### Case B：Working Memory phase 与任务状态不一致

例如：

```text
WorkingMemory.phase = investigating
TaskStatus = testing
```

必须存在明确、可测试的 phase resolution 规则。

### Case C：重复调查

连续多轮执行相似：

```text
execute/read_file
```

但没有新增 evidence/hypothesis/progress。

应在远早于 `max_shell_calls` 时检测 stagnation。

### Case D：正常调查不能被误杀

例如：

```text
pytest fail
→ read_file
→ 一个针对性实验
→ hypothesis
```

属于合理调查，不能被循环保护误判。

### Case E：正常修复链路

```text
investigate
→ pytest fail
→ hypothesis
→ edit
→ pytest pass
→ completed
```

保持正常。

### Case F：Progress Event

验证：

```text
TaskStatus transition
tool execution
file edit
pytest result
approval
pause
```

能够生成正确的用户可见进度事件。

同时验证：

- progress 输出不额外调用模型；
- 不根据 LLM 自述伪造测试成功；
- 重复保存/恢复不会重复打印错误的历史进度；
- resume 后仍能继续显示正确阶段。

## 八、必须保持的现有边界

修改不能破坏：

- 单 Repair Agent 架构；
- manual / guarded HITL；
- Shell 风险审批；
- `max_shell_calls` 最终保险；
- deterministic test evidence；
- Working Memory 版本化；
- LangGraph Checkpointer resume；
- Compaction；
- Research evidence；
- 没有真实 `exit_code=0` 测试证据不能 `completed`；
- 主模型和压缩模型角色隔离。

## 九、最终验收

重新使用 gcd 单 Bug 场景。

正常轨迹应接近：

```text
读取实现和测试
→ 运行目标 pytest
→ 得到 RecursionError
→ 形成递归不收敛假设
→ 必要的最小验证
→ 确认根因
→ edit_file
→ 重新运行目标 pytest
→ exit_code=0
→ completed
```

不要求固定 Model Call 次数，但不能再次出现：

```text
20+ 次连续 execute
```

或者：

```text
相同取模实验不断重复
```

同时 CLI 应能实时看到类似：

```text
[Investigating] 已复现目标测试失败
[Diagnosing] 正在验证递归不收敛假设
[Editing] 已确认根因，正在修改 gcd.py
[Testing] 正在重新运行目标测试
[Testing] 6 passed
[Reviewing] 正在生成最终报告
```

最后检查 Trace 中：

```text
phase
task_status
working_memory_version
tool_calls
```

应能够看到合理状态推进。

请先阅读现有实现和测试，确认上述分析是否与代码实际行为一致。如果实际根因还有其他因素，以代码和测试为准修正判断。然后做最小但架构上正确的修改，运行相关单测、完整离线测试以及 Ruff。

不要通过修改 QuixBugs 的 gcd 样例本身来让实验通过；修复目标是 DeepFix Agent 的阶段推进、循环控制和运行时可观测性。

## 十、补充实验：mergesort 单 Bug 暴露 Large Tool Result 读取问题

在前面的 `gcd` 实验之后，我又使用 QuixBugs 的 `mergesort` 做了一次独立单 Bug 测试。

用户任务为：

```text
mergesort 功能的相关测试存在失败。请调查根因，在不修改测试和测试数据的前提下进行最小必要修改，并运行 python -m pytest python_testcases/test_mergesort.py -q --timeout=5 验证修复结果。
```

修复前目标测试结果：

```text
13 failed, 1 passed
```

主要异常：

```text
RecursionError: maximum recursion depth exceeded
```

源码核心逻辑为：

```python
def mergesort(arr):
    ...

    if len(arr) == 0:
        return arr
    else:
        middle = len(arr) // 2
        left = mergesort(arr[:middle])
        right = mergesort(arr[middle:])
```

当：

```python
arr = [10]
```

时：

```text
middle = 0

left  = mergesort([])
right = mergesort([10])
```

因此 `right` 又得到原来的 `[10]`，递归永远无法收敛。合理根因应很快定位为：

```text
递归终止条件只覆盖 len(arr) == 0，
没有覆盖 len(arr) == 1。
```

pytest traceback 本身已经明确暴露了 `mergesort.py` 的递归位置以及 `RecursionError`。

### 实际 Agent 行为

本次任务 ID：

```text
b5229ea2ff784435a810c73ac0b8024e
```

前四次 Model Call 基本正常：

```text
#1 ls
#2 ls
#3 read_file：mergesort.py + test_mergesort.py
#4 execute：目标 pytest
```

但是 pytest 输出因为递归 traceback 非常大，被 Large Tool Result 机制卸载到了：

```text
/.deepfix-artifacts/large_tool_results/
call_00_mDb6VoLZWt7p35bPy2Xs6584
```

给模型的 Tool Result 没有直接提供完整错误上下文，而是告诉模型：

```text
Tool result too large...

完整结果已经保存到 Artifact。

可以通过 read_file，
使用 offset / limit 分段读取。
```

同时 deterministic evidence 中只保留了失败测试摘要和 Artifact 路径。

模型随后开始顺序分页读取该 Artifact。

实际模式大致为：

```text
offset=0
offset=60
offset=100
offset=140
offset=180
...
offset=1900
offset=1940
...
```

每次读取约 40 行。

到约 `offset=1940` 时，模型仍然在读取同一个 pytest Artifact：

```text
read_file(
    file_path="/.deepfix-artifacts/large_tool_results/call_00_mDb6VoLZWt7p35bPy2Xs6584",
    offset=1940,
    limit=40,
)
```

因此这次看到的几十次 `read_file` 并不是简单地反复读取 `mergesort.py`。

更准确的行为是：

```text
pytest 产生巨大 traceback
        ↓
Large Tool Result 被 offload
        ↓
模型只获得 Artifact 路径和少量 preview
        ↓
模型想寻找完整 failure / exception context
        ↓
从 Artifact 头部开始顺序分页
        ↓
每读取一段产生一个 ToolMessage
        ↓
再次 Model Call
        ↓
继续读取下一段
```

最终出现了超过 60 次连续 `read_file` Model Call。

运行过程中 Context 也持续增长：

```text
#5   ≈ 13K chars
#20  ≈ 43K chars
#40  ≈ 82K chars
#60  ≈ 121K chars
#68  ≈ 135K chars
```

最后任务并没有自己终止，而是手动 `KeyboardInterrupt`。

------

## 十一、这次实验暴露出的真实问题

这次问题不能简单理解成：

```text
LLM 重复调用 read_file
```

真正的问题至少有三层。

### 1. Large Tool Result 缺少面向调试任务的结构化摘要

Large Tool Result 当前主要做：

```text
完整结果过大
→ 保存 Artifact
→ 返回 preview + Artifact 路径
→ 提示模型自行 read_file(offset, limit)
```

这个机制对保存上下文是合理的。

但是对于：

```text
pytest
stack trace
编译错误
静态检查日志
测试失败
```

这样的调试输出，如果只给模型 Artifact 路径，就可能迫使模型自己寻找：

```text
真正 exception 是什么
出错文件和行号是什么
哪个输入触发错误
最关键的 traceback frame 是什么
总共有多少种独立 failure
```

本次 mergesort 中，LLM 很可能就是为了找到这些信息，从头开始不断分页读取。

也就是说：

```text
Large Tool Result 成功解决了“单次结果过大”
```

但是没有解决：

```text
模型如何高效定位 Artifact 中真正有价值的信息
```

的问题。

### 2. Offload 后的内容被重新逐块灌回 Context

Large Tool Result 的设计目的本来是：

```text
大结果
→ offload
→ Context 中只保留摘要
```

但此次实际发生：

```text
大结果
→ offload

→ read_file 第 1 段
→ ToolMessage 加回 Context

→ read_file 第 2 段
→ ToolMessage 再加回 Context

→ read_file 第 3 段
→ ToolMessage 再加回 Context

...
```

最终等价于：

```text
把刚刚卸载出去的长日志，
又通过几十轮工具调用逐渐重新塞回上下文。
```

这会同时造成：

```text
大量 Model Calls
大量 token 消耗
上下文持续增长
调试延迟增大
```

因此 Artifact offload 目前只有：

```text
storage offload
```

但缺少有效的：

```text
retrieval strategy
```

### 3. 得到足够证据后没有停止读取

即使模型为了寻找首个有效异常而顺序读取 Artifact 尚可解释，当它已经看到：

```text
arr = [1]
RecursionError
mergesort.py
递归调用位置
```

以后，结合此前已经读取过的源码：

```python
if len(arr) == 0:
    return arr
```

就已经有足够证据建立：

```text
长度 1 没有递归终止
```

这一强根因假设。

此时理想行为应该是：

```text
停止继续阅读 traceback
→ 保存 hypothesis / evidence
→ 必要时进行一个最小验证
→ edit_file
```

而不是继续把后面的 traceback 全部读取完。

因此这里仍然和前面发现的 stagnation 问题有关：

```text
已经获得足够新证据
但系统没有判断“调查收益已经明显下降”
```

------

## 十二、需要增加的 Large Tool Result 修复要求

请在之前 phase/stagnation 修复的基础上，再检查：

```text
Large Tool Result middleware
Tool result offload
Artifact 存储
execute / pytest result normalization
DeterministicEvidence
read_file Artifact 访问逻辑
Context management
```

目标不是取消 Artifact offload，而是让：

```text
offload + retrieval
```

形成完整闭环。

### 1. 对调试类大输出生成确定性的 Diagnostic Summary

对于已知结构的输出，尤其是：

```text
pytest
Python traceback
ruff
mypy
编译器错误
```

在 offload 时应尽量确定性抽取有价值的信息。

例如 pytest 大结果可以保留：

```text
command:
python -m pytest python_testcases/test_mergesort.py -q --timeout=5

exit_code:
1

summary:
13 failed, 1 passed

primary_exception:
RecursionError: maximum recursion depth exceeded

primary_location:
python_programs/mergesort.py:21

representative_failure:
test_mergesort[...]

relevant_frame:
left = mergesort(arr[:middle])

representative_input:
arr = [1]

artifact_path:
...

artifact_lines:
2450
```

具体字段可以结合现有 Evidence 数据结构设计。

不要依赖 LLM 再读两千行 traceback 才知道：

```text
RecursionError
```

是什么。

### 2. Diagnostic Summary 必须是确定性生成

不要为了生成摘要额外调用一次 LLM。

优先使用：

```text
pytest 输出结构
正则/解析
traceback pattern
failed summary
exception line
文件:行号
```

等确定性方法。

LLM 仍然可以根据这些事实进行诊断，但：

```text
“测试产生了什么异常”
```

这类信息不应该需要模型通过几十次 Tool Call 自己寻找。

### 3. Artifact 保留完整原始内容

不要为了摘要丢弃完整日志。

仍然保持：

```text
Diagnostic Summary
+
Full Artifact
```

两层结构。

模型只有在摘要不足时才读取 Artifact。

例如：

```text
摘要：
RecursionError
mergesort.py:21
arr=[1]

完整日志：
Artifact
```

这样既保留可追溯性，也降低默认 Context 成本。

### 4. 支持更合理的 Artifact 定位方式

不要让唯一策略是：

```text
offset=0
offset=40
offset=80
...
```

至少考虑提供能够快速定位的能力，例如：

```text
grep Artifact
search Artifact
find exception
find "RecursionError"
find "FAILED"
find "AssertionError"
```

或者提供结构化区段索引。

例如模型如果需要 traceback，可以：

```text
search_artifact(
    artifact_id=...,
    query="RecursionError"
)
```

再只读取命中位置附近几十行。

如果现有 `grep` / `read_file` 已能安全支持 Artifact，则可以复用，不一定新增工具。

关键目标：

```text
让模型能够直接跳到高价值位置
```

而不是线性扫描整个 Artifact。

### 5. 阻止 Artifact 顺序扫描重新填满 Context

需要增加保护，例如：

```text
同一 Artifact
连续多次 read_file
offset 单调增加
但没有产生新的 evidence / hypothesis / phase change
```

应视为潜在的 sequential-scan stagnation。

达到合理阈值后：

```text
停止继续线性读取
→ 提示优先使用搜索
→ 根据已有摘要重新评估
→ 必要时暂停
```

不应该允许：

```text
50+ 次 read_file
```

仅仅为了翻完一个 pytest traceback。

### 6. 已获得高价值异常后应降低继续读取优先级

如果系统已经有：

```text
exception_type
location
representative frame
failing input
```

并且源码已经读取，

则 Agent 应优先：

```text
形成 hypothesis
```

而不是默认继续扫描剩余日志。

这不需要硬编码成：

```text
看到 RecursionError 就必须 edit
```

而应该加入已有 stagnation / progress 机制：

```text
继续读取是否能够区分当前假设？
```

如果不能，则没有继续调查的收益。

------

## 十三、补充回归测试

除前面已有 phase/stagnation 测试外，再增加以下离线测试。

### Case G：大型 pytest 输出 offload

构造一个超过阈值的 pytest traceback。

验证：

```text
完整输出被保存 Artifact
```

同时模型可见的 deterministic evidence 中包含：

```text
exit_code
failed/passed summary
primary exception
location
artifact reference
```

而不是只有：

```text
Tool result too large, go read it yourself
```

### Case H：Diagnostic Summary 不依赖 LLM

验证生成 summary：

```text
不增加 Model Call
```

并能从典型：

```text
AssertionError
RecursionError
ImportError
Timeout
```

中提取至少基础异常信息。

### Case I：Artifact 定位读取

对于一个 2000+ 行 Artifact：

```text
目标异常位于第 1800 行
```

Agent 不应被迫：

```text
offset 0
offset 40
offset 80
...
```

顺序扫描 45 次。

应能够通过搜索/索引快速读取命中区域。

### Case J：防止 Context 回灌

模拟：

```text
大 Tool Result 被 offload
```

然后连续读取多个 Artifact chunk。

验证系统能够识别：

```text
同一 Artifact 的无进展顺序扫描
```

并在远早于几十轮 Model Call 时采取措施。

### Case K：允许必要的 Artifact 深挖

不能因为增加限制就完全禁止读取 Artifact。

例如：

```text
摘要只显示 AssertionError
```

但具体 expected / actual 被截断。

此时模型读取：

```text
命中位置前后 30 行
```

属于合理行为，不应被 stagnation detector 拦截。

------

## 十四、更新后的总体问题模型

经过 gcd 与 mergesort 两次实验，目前 DeepFix 的无效长循环可以归纳为三类：

```text
A. phase deadlock
Working Memory 没有推进
→ phase 长期 investigating

B. investigation stagnation
模型不断 execute/read
→ 没有新增 hypothesis/evidence/progress

C. artifact sequential scan
大 Tool Result offload
→ 模型线性分页读取
→ 内容重新灌回 Context
```

它们可能互相增强：

```text
Large pytest output
        ↓
Artifact
        ↓
连续 read_file
        ↓
Working Memory 没更新
        ↓
phase 仍 investigating
        ↓
Prompt 要求继续调查
        ↓
继续 read Artifact
        ↓
Context 越来越大
```

因此最终修复应该把：

```text
Phase management
Progress/Stagnation detection
Large Tool Result retrieval
CLI Progress visibility
```

看成同一个 Agent reliability 问题的不同层面，而不是分别通过增加几个次数上限解决。

次数上限仍然可以作为最终 safety net，但不能成为主要控制机制。

------

## 十五、更新后的验收要求

重新跑：

```text
gcd
mergesort
to_base
```

三个独立 Bug 场景。

对于 mergesort，期望大致链路：

```text
读取 mergesort.py
读取 test_mergesort.py
        ↓
运行 pytest
        ↓
pytest 输出较大
        ↓
完整日志 offload
+
系统提供结构化失败摘要
        ↓
模型直接知道：
RecursionError
mergesort.py
arr=[1]
        ↓
结合源码形成 hypothesis
        ↓
必要的最小验证
        ↓
edit_file
        ↓
目标 pytest
        ↓
通过
        ↓
completed
```

不要求固定 Model Call 数量。

但是不应该再次出现：

```text
几十次 read_file 同一个 Artifact
```

也不应该因为 Artifact offload 导致：

```text
Context 从十几 KB
一路重新增长到一百多 KB
```

最后通过 LLM Trace 检查：

```text
tool_calls
artifact reads
context size
phase
working_memory_version
task_status
```

确认：

```text
大日志没有被线性重新灌回 Context
Agent 能够在获得足够证据后结束调查
phase 能够正常推进
```

## 十六、补充实验：breadth_first_search 暴露 Strong Evidence 下的 Repository Scan Loop

在 `gcd` 和 `mergesort` 两次实验之后，又使用 QuixBugs 的 `breadth_first_search` 做了一次独立单 Bug 测试。

本次实验主要用于验证：

```text
Strong Evidence → Stop Investigating
```

即：当源码、测试和 traceback 已经提供非常明确的根因线索时，Repair Agent 是否能够及时结束调查，形成 hypothesis，并进入修改与验证阶段。

### 1. 实验场景

用户任务：

```text
breadth_first_search 功能的相关测试存在失败。
请调查根因，在不修改测试和测试数据的前提下进行最小必要修改，
并运行：

python -m pytest python_testcases/test_breadth_first_search.py -q --timeout=5

验证修复结果。
```

修复前独立运行目标测试：

```text
1 failed, 4 passed
```

唯一失败发生在两个互不连通的节点场景。

测试明确要求：

```text
Case 3: Two unconnected nodes in graph
Output: Path not found
```

实际执行却得到：

```text
IndexError: pop from an empty deque

python_programs/breadth_first_search.py:12

node = queue.popleft()
```

目标实现为：

```python
def breadth_first_search(startnode, goalnode):
    queue = Queue()
    queue.append(startnode)

    nodesseen = set()
    nodesseen.add(startnode)

    while True:
        node = queue.popleft()

        if node is goalnode:
            return True
        else:
            queue.extend(
                node for node in node.successors
                if node not in nodesseen
            )
            nodesseen.update(node.successors)

    return False
```

目标测试则明确要求不可达节点返回 `False`。

因此，在源码、测试和真实 pytest failure 都获得后，已经存在非常强的根因证据：

```text
goal 不可达
        ↓
BFS 最终耗尽 queue
        ↓
实现使用 while True
        ↓
queue 已空仍执行 popleft()
        ↓
IndexError
```

一个合理的根因假设应当很快收敛为：

```text
当搜索队列已经耗尽且仍未找到 goalnode 时，
breadth_first_search 应结束并返回 False，
而不是继续从空队列调用 popleft()。
```

Agent 在目标 pytest 执行之前已经读取了：

```text
breadth_first_search.py
test_breadth_first_search.py
node.py
```

pytest 又进一步提供了准确的异常类型、文件和出错行。

------

### 2. 实际 Agent 行为

本次任务 ID：

```text
60032d5dae0343a8b33e2c899668c30b
```

前几次 Model Call 基本正常：

```text
#1  ls

#2  ls

#3  read_file
    breadth_first_search.py
    test_breadth_first_search.py
    node.py

#4  read_file
    python_testcases/node.py

#5  execute
    python -m pytest python_testcases/test_breadth_first_search.py -q --timeout=5
```

第 5 次 Model Call 已经复现：

```text
1 failed, 4 passed
IndexError: pop from an empty deque
breadth_first_search.py:12
```

但是 Agent 没有：

```text
save_progress
形成 hypothesis
edit_file
```

而是继续调用 `read_file`。

最开始读取：

```text
breadth_first_search_test.py
depth_first_search.py
```

这一小段行为还可以理解为查看同仓库中的类似图搜索实现。

例如 `depth_first_search.py` 中已经存在非常明确的搜索失败处理：

```python
if node in nodesvisited:
    return False
elif node is goalnode:
    return True
```

这实际上进一步支持了“不可达时应正常返回 False”这一判断。

但模型没有因此结束调查。

随后调查范围继续扩大到：

```text
shortest_path_length.py
shortest_paths.py
shortest_path_lengths.py
detect_cycle.py
topological_ordering.py
minimum_spanning_tree.py
reverse_linked_list.py
depth_first_search_test.py
detect_cycle_test.py
minimum_spanning_tree_test.py
shortest_path_length_test.py
shortest_paths_test.py
shortest_path_lengths_test.py
topological_ordering_test.py
reverse_linked_list_test.py
...
```

也就是说，调查从：

```text
breadth_first_search 的直接根因
```

逐渐扩展成：

```text
扫描仓库中大量其他图算法、
搜索算法甚至链表算法，
寻找可能的参考实现。
```

------

### 3. 最终形成 Repository Reference Scan Loop

终端 Trace 显示，从第 6 次以后，Agent 几乎持续只调用：

```text
read_file
```

实际出现：

```text
LLM #6
→ read_file

LLM #7
→ read_file

LLM #8
→ read_file

...

LLM #69
→ read_file

LLM #70
→ read_file
```

Context 同时持续增长：

```text
#6   ≈ 15.9K chars
#20  ≈ 34.6K chars
#40  ≈ 66.9K chars
#50  ≈ 82.9K chars
#60  ≈ 98.6K chars
#70  ≈ 116K chars
```

一直没有出现：

```text
edit_file
save_progress
修复后 pytest
RepairOutcome
```

任务最终也不是由 DeepFix 主动发现 stagnation，而是人工 `KeyboardInterrupt`。

进一步检查 tool args 后可以看到，这不是单纯“读取很多不同文件”。

Agent 后半段开始重复读取同一组相关性较弱的文件，形成稳定的周期性仓库扫描。

可以概括为：

```text
A
→ B
→ C
→ D
→ E
→ F
→ G
→ H

然后再次：

A
→ B
→ C
→ D
→ E
→ F
→ G
→ H
```

模型没有获得新的测试证据、没有形成 Working Memory、没有进入修改阶段，却继续执行同类调查操作。

因此本次问题可以定义为：

```text
Repository Reference Scan Loop
```

------

## 十七、本次实验的重要意义

`breadth_first_search` 与 `mergesort` 有一个非常重要的区别。

`mergesort` 的大量 `read_file` 存在：

```text
Large pytest traceback
        ↓
Large Tool Result offload
        ↓
LLM 分页读取 Artifact
```

这一特殊因素。

而本次 `breadth_first_search`：

```text
pytest 输出很小
没有 Large Tool Result
没有 Artifact sequential scan
异常完整直接返回
```

因此这次实验说明：

> Large Tool Result 并不是 repeated read 的必要条件。

即使没有 Artifact 问题，只要 Agent 无法判断“当前证据已经足够”，它仍然可能进入长期调查循环。

因此目前可以确认 DeepFix 存在更加一般的：

```text
Investigation Convergence Failure
```

问题。

------

## 十八、状态问题再次复现

pytest 完成以后，同一次模型调用中的 Task Anchor 已经显示：

```text
task_status = testing
```

但 Agent 获得的 reasoning phase 仍然是：

```xml
<deepfix_phase name="investigating">
先阅读相关源码和测试并复现故障。
建立可证伪的根因假设，用最小实验逐个验证；
根因未得到证据支持前不要修改代码。
</deepfix_phase>
```

同时：

```xml
<deepfix_working_memory version="none">
```

仍然没有变化。

因此再次确认存在：

```text
TaskStatus = testing

但

Agent Phase = investigating
```

的状态分裂。

这意味着即使业务层已经记录：

```text
测试已经执行
```

PromptPolicy 仍然不断向模型强调：

```text
继续调查
根因没有充分证据前不要修改
```

从而进一步增加继续调查的倾向。

这与此前 `gcd` 实验得到的 phase 问题完全一致。

------

## 十九、新发现：read_file 没有形成可靠的 Checked-File State

本次实验还暴露出另一个重要问题。

Agent 实际已经读取了大量文件，例如：

```text
breadth_first_search.py
test_breadth_first_search.py
node.py
depth_first_search.py
shortest_paths.py
topological_ordering.py
...
```

但 Protected Context 中的：

```xml
<deepfix_deterministic_evidence>
    <files>
    </files>
</deepfix_deterministic_evidence>
```

依然没有可靠记录这些已经完成的源码调查。

Working Memory 同样为：

```text
version = none
```

因此当前系统中：

```text
read_file 成功
```

主要只是产生：

```text
ToolMessage
```

并进入 conversation history。

系统并没有独立维护类似：

```text
checked_files
read_ranges
recent_investigation_targets
```

这样的确定性调查状态。

这意味着随着任务越来越长，模型判断：

```text
我已经读过什么？
哪些文件已经检查？
这些文件有没有提供新信息？
```

越来越依赖完整 message history。

这种机制对长期 Agent loop 非常脆弱。

------

## 二十、需要补充的修复要求

在此前：

```text
Phase management
Stagnation detection
Large Tool Result retrieval
CLI Progress visibility
```

基础上，再增加以下三个方面。

### 1. Investigation Progress Tracking

需要建立一种不完全依赖 `save_progress` 的轻量确定性调查状态。

例如成功：

```text
read_file
grep
glob
execute
```

以后，系统至少能够知道：

```text
读取了什么文件
读取了哪个范围
运行了什么测试
得到了什么失败类型
是否产生新的 deterministic evidence
```

可以考虑维护类似：

```text
checked_files
recent_tool_signatures
recent_test_evidence
investigation_progress
```

具体数据结构根据现有架构决定。

Working Memory 继续承担：

```text
facts
hypotheses
semantic evidence
next_steps
```

等模型总结后的高层语义状态。

但是：

```text
“这个文件已经读过”
```

这种确定性事实，不应该必须等待模型主动 `save_progress` 才存在。

------

### 2. Investigation Scope Control

Agent 应优先调查：

```text
失败测试
直接目标实现
直接依赖
与当前 hypothesis 明确相关的代码
```

当模型准备扩大范围时，例如从：

```text
breadth_first_search.py
```

扩展到：

```text
minimum_spanning_tree.py
reverse_linked_list.py
shortest_path_lengths.py
```

系统应该能够识别调查范围正在持续扩张。

不要求完全禁止跨文件调查。

真实复杂 Bug 可能确实需要查看多个模块。

但扩大范围应该满足类似：

```text
这个新文件能够验证什么 hypothesis？
它与当前 failure 的因果关系是什么？
```

如果连续读取多个新文件：

```text
没有新增 evidence
没有 hypothesis 更新
没有 phase 变化
没有新的直接依赖关系
```

则应该认为调查收益正在下降。

------

### 3. Tool Cycle / Repository Scan Detection

当前 `max_shell_calls` 无法保护这种情况，因为模型几乎没有调用 Shell。

因此 stagnation detector 必须覆盖：

```text
read_file
grep
glob
ls
execute
research
```

等调查工具。

特别需要支持周期性工具序列检测。

例如最近工具签名：

```text
read_file(A)
read_file(B)
read_file(C)
read_file(D)
read_file(E)

read_file(A)
read_file(B)
read_file(C)
read_file(D)
read_file(E)
```

如果第二轮序列开始重复，并且期间：

```text
没有新增 hypothesis
没有 Working Memory 更新
没有新 test evidence
没有 edit
没有 phase transition
```

应当高置信度判定：

```text
investigation stagnation
```

而不是继续允许第五、第六轮重复。

检测到以后应优先：

```text
停止继续横向扫描
        ↓
要求模型重新评估当前已有证据
        ↓
形成当前最可能 hypothesis
        ↓
如果仍需继续调查，
必须说明下一步调查能够区分什么假设
```

如果无法形成新的有效调查方向，则安全暂停。

------

## 二十一、不要使用简单的 read_file 次数限制

本次修复不能简单实现成：

```text
read_file > N
→ stop
```

因为真实 repository debugging 可能合理读取多个文件。

真正需要识别的是：

```text
大量 investigation tool calls
+
低信息增益
+
没有状态推进
+
调查范围持续扩散
或
工具序列开始重复
```

因此更合理的判断维度包括：

```text
是否首次读取该文件
是否读取相同范围
是否产生新的 deterministic evidence
是否形成新的 hypothesis
是否改变当前 hypothesis 状态
是否有新的 test result
是否产生 edit
是否发生 phase transition
工具调用序列是否出现周期
```

目标是检测：

```text
No Progress
```

而不是限制：

```text
Tool Count
```

------

## 二十二、补充回归测试

在原有 Case A–K 基础上，再增加：

### Case L：Strong Evidence 应结束调查

构造：

```text
源码中存在 while True + queue.popleft
测试要求 unreachable 返回 False
pytest 返回 IndexError: pop from an empty deque
```

验证 Agent 获得这些事实后，不应继续大量读取无关文件。

应能够优先进入：

```text
hypothesis
→ edit
→ test
```

------

### Case M：Checked-File Tracking

连续：

```text
read_file(A)
read_file(B)
read_file(C)
```

以后，系统应确定性记录：

```text
A/B/C 已经检查
```

而不要求 LLM 必须先调用 `save_progress`。

resume / checkpointer 恢复以后，也应该保持一致。

------

### Case N：Repository Scope Expansion

允许：

```text
target.py
→ test_target.py
→ direct_dependency.py
```

这种正常范围扩展。

但如果进一步出现：

```text
unrelated_a.py
unrelated_b.py
unrelated_c.py
...
```

且没有新的 evidence / hypothesis，则应触发 investigation re-evaluation。

------

### Case O：周期性 read_file Loop

构造：

```text
A → B → C → D
A → B → C → D
```

且中间没有：

```text
evidence update
hypothesis update
phase transition
edit
new test result
```

应在第二轮附近检测 stagnation。

不能允许重复几十轮。

------

### Case P：合理的多文件 Investigation 不应误杀

构造一个真实需要：

```text
controller.py
service.py
repository.py
model.py
```

共同调查的 Bug。

如果每次新文件都提供：

```text
新的直接依赖
新的调用链信息
新的 evidence
```

则即使连续多次 `read_file`，也不应判定 stagnation。

------

## 二十三、三次失败实验的统一模型

目前已经进行了三个非常有代表性的失败实验。

### gcd

表现：

```text
execute
→ execute
→ execute
→ ...
```

模型不断手工进行取模实验。

对应：

```text
Investigation Experiment Loop
```

原始报告已经记录该任务出现 27 次主模型调用，并最终因 Shell 上限停止。

------

### mergesort

表现：

```text
read Artifact offset=0
read Artifact offset=40
read Artifact offset=80
...
```

Large Tool Result 被 offload 后，模型线性扫描 pytest traceback。

对应：

```text
Artifact Sequential Scan Loop
```

并造成 offload 内容重新逐块进入 Context。

------

### breadth_first_search

表现：

```text
read direct code
→ pytest gives strong evidence
→ read related code
→ read less related code
→ repeat repository reference files
→ ...
```

对应：

```text
Repository Reference Scan Loop
```

此次没有 Large Tool Result，因此说明 Agent 即使获得完整、简短、明确的 traceback，仍然可能无法结束 investigating。

------

三种循环表面形式不同：

```text
execute loop

artifact read loop

repository read loop
```

但共同结构高度一致：

```text
获得足够或接近足够的证据
        ↓
没有形成可靠 Progress State
        ↓
Working Memory 没有推进
        ↓
reasoning phase 长期 investigating
        ↓
Prompt 继续强化“继续调查”
        ↓
模型再寻找一点额外证据
        ↓
Tool Call
        ↓
没有系统判断此次 Tool Call
是否真的增加了信息
        ↓
下一轮 Model Call
        ↓
继续调查
```

因此目前 DeepFix 的核心问题已经可以更准确地定义为：

```text
Investigation Convergence Failure
```

而不是单纯：

```text
Shell Loop
```

或者：

```text
Large Tool Result Bug
```

------

## 二十四、更新后的 DeepFix Reliability 修复范围

请把当前修复范围统一看成：

```text
1. Phase Management

2. Investigation Progress Tracking

3. Checked-File / Tool Fact Tracking

4. Strong-Evidence Re-evaluation

5. Stagnation Detection

6. Tool Cycle Detection

7. Investigation Scope Control

8. Large Tool Result Diagnostic Summary

9. Artifact Retrieval Strategy

10. Context Growth Control

11. CLI Progress Visibility
```

这些不是彼此独立的零散功能，而共同服务于一个目标：

```text
让 Repair Agent 能够知道：

我现在知道什么？
我已经检查过什么？
当前最可能的假设是什么？
下一次工具调用能增加什么信息？
什么时候证据已经足够开始修改？
什么时候我正在重复自己？
```

不要通过增加大量固定次数限制来解决。

例如：

```text
max_read_file = 10
max_investigation_calls = 20
```

只能作为最后的 safety net。

真正的主控制机制应当围绕：

```text
Progress
Evidence Gain
Phase
Scope
Stagnation
```

设计。

------

## 二十五、更新后的验收场景

在完成这一轮 Reliability 修复之后，至少重新运行：

```text
to_base
gcd
mergesort
breadth_first_search
```

其中 `breadth_first_search` 的理想轨迹应接近：

```text
读取：
breadth_first_search.py
test_breadth_first_search.py
node.py

        ↓

运行目标 pytest

        ↓

得到：

1 failed, 4 passed
IndexError: pop from an empty deque
breadth_first_search.py:12

        ↓

形成 hypothesis：

unreachable graph 耗尽 queue，
但 while True 仍继续 popleft

        ↓

必要时最多进行少量直接相关调查

        ↓

edit_file

        ↓

重新运行：

python -m pytest python_testcases/test_breadth_first_search.py -q --timeout=5

        ↓

exit_code = 0

        ↓

completed
```

不要求固定 Model Call 数量。

但是不能再次出现：

```text
几十次连续 read_file
```

不能出现：

```text
为了 breadth_first_search
持续扫描 MST / Floyd-Warshall /
linked-list 等无直接因果关系代码
```

也不能出现：

```text
TaskStatus 已 testing
但 reasoning phase 长时间仍 investigating
```

最后使用 LLM Trace 检查：

```text
phase trajectory
task_status
working_memory_version
checked_files
recent tool signatures
read_file targets
stagnation events
context size
first edit Model Call
```

确认 Agent 能够：

```text
获得强证据
→ 收敛调查
→ 形成 hypothesis
→ 修改
→ 验证
```

而不是：

```text
获得强证据
→ 继续搜索更多证据
→ 横向扫描仓库
→ Context 持续增长
→ 人工终止
```

此次 `breadth_first_search` 应作为：

```text
Strong Evidence → Stop Investigating
+
Repository Scan Loop Detection
```

的固定回归案例。