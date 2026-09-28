# 本轮方法与实验说明

本轮唯一核心方法为 Verifier-Gated Failure-Triggered Reflection。
未增加 Episodic Memory、训练、强化学习或新的生物预测模型。

## 现有项目与接入位置

主入口为 cli.py 的 main，由 fixed-enzyme-agent 或 python -m fixed_enzyme_agent.cli 调用。
自由链路通过 planner/planner.py 规划、planner/validator.py 校验、executor.py 执行；
tools/builtin.py 封装 EnzGFM、EpHod、UniStab 和密码子优化，使用 BioArtifact。
固定链路在 workflow.py 中调用同一组 adapters，在 scoring.py 排序，并由
markdown_report.py 和 llm/result_analyzer.py 生成报告。原有模型 stdout 日志由 runner.py 保存。
测试位于 tests，JSON 配置由 config.py 载入。

新 verifier 在工具调用前后、计划检查以及最终候选约束检查处工作。
输出 VerificationResult，包括 passed、failure_type、reason、evidence、recoverable、scope。
recoverable 由当前状态下是否存在未尝试的程序允许修复决定，与剩余重试次数分别处理。
scope 区分 execution_validity 与 scientific_objective_satisfaction。

## 五类失败

- tool_execution_failure：异常、超时、空结果、必需字段缺失、非数字或 NaN/Inf、缺少记录预测。
- invalid_tool_argument：不满足 ToolSpec 的输入、非法超时类型、重复 ID、越界残基、非法突变格式、模型不接受的序列类型或长度。
- missing_required_step：计划没有覆盖原 plan 的 requested_capabilities、配置中的 required_capabilities，或显式数值约束所必需的能力。
- constraint_violation：最终候选违反用户配置的数值阈值，或有明确约束却没有最终候选。
- inconsistent_result：突变描述与序列不一致；预测工具改变或删除输入序列。只检查可逐字比较的冲突。

最适 pH 范围来自 target_ph 与 ph_tolerance 或显式范围；activity_proxy 与 ddg
保留原模型定义，不解释为概率、真实酶活或实验证实的热稳定性。
中间筛选候选允许不满足最终目标，因此不因中间候选偏离目标就反思。
固定 workflow 在最后一轮 selected 上检查；自由 route 没有新增候选排序逻辑，
因此最终检查覆盖其输出的全部蛋白候选。

科学合理性判断、预测精度、湿实验事实、跨模型物理意义的一致性属于 NOT_IMPLEMENTED。
自由自然语言中未转成结构化配置的限制不保证被程序完整识别；应在正式任务中明确给出 required_capabilities 与阈值。

## 反思与执行

验证通过直接继续。验证失败且存在允许修复，并且预算未耗尽时，才向 LLM 发送
原任务、当前计划、失败动作、失败类别与证据、历史执行 trace 和 allowed_repairs。
LLM 返回 failure_analysis、repair_strategy、repair_id；程序依据 ID 得到 revised_action。
不接受 LLM 任意改写文件、阈值、序列、预测结果或运行命令。

允许的局部修复包括切换已注册的同能力工具、补齐缺失能力、恢复任务原 runtime 超时值，
以及在用户明确指定 max_timeout_seconds 的范围内延长超时。
程序比较 action 指纹，拒绝完全相同的失败调用；成功步骤的输出保留，失败调用操作深拷贝状态，
其部分修改不会污染后续修复。每次尝试使用新的模型工作目录，防止旧输出被误读为新结果。

默认 max_retries=2，按整次运行累计。无修复、反思 API 失败、非法 repair_id 或预算耗尽时，
返回失败并保存 trajectory。没有备选模型时，普通异常通常不可恢复；不会把未知异常自动当成可修复。
固定 run 的模型调用只提供超时修复，不重排既有固定流程。

固定链路的前置路径检查、候选库加载、绘图和报告生成仍采用原有异常处理；
这些阶段未增加自动反思。自由工具结果接受统一类型与对应已知预测字段检查；
新插件的专业输出仍需扩展确定性规则，不能假定任意插件都已获得科学验证。

JSONL trajectory 记录 task_id、step_id、planned_action、tool_name、tool_arguments、
tool_result_status、verification_passed、failure_type、failure_reason、reflection_triggered、
repair_action、retry_index、final_status，并保存证据、耗时和 action_hash。
自由链路另保存脱敏后的输入配置和验证通过的结果快照。日志是运行记录，不实施 memory 检索。

## 实验公平性和指标

实验复用生产 WorkflowExecutor，不另建 Agent demo。两组使用相同的初始冻结计划、
工具集合、输入、模型设置、seed 和初始故障安排；唯一方法差异是是否开启 verifier/reflection 闭环。
Baseline 使用原执行路径，旁路观察只计算指标，不把验证结果反馈给 Agent。
真实模式共用配置的 DeepSeek 模型、默认 temperature；本轮没有更改采样参数。
反思产生的额外调用数和耗时是干预结果，需要计入成本。

offline 使用 FixtureTool 和 FixtureReflector，只检验控制流。real 模式使用原模型 adapters
和真实 LLMReflector。两种结果不得混合统计。冻结计划评估执行修复能力，不衡量 planner 生成质量。

支持 tool_failure、invalid_result、argument_failure、missing_step。前两种按每次工具调用抽样，
后两种按任务抽样。固定 seed 下，采样由 task_id、工具和调用次数决定。修复调用也可能失败。
注入仅存在于 experiments，正式 Agent 不导入该代码。

final_success_rate：最终输出经相同外部 oracle 检查合格的任务数除以总任务数。
failure_recovery_rate：后续同一步骤得到 PASS 的可恢复失败事件数除以全部可恢复失败事件数。
verification_pass_rate：工具调用尝试中 PASS 的数量除以工具调用尝试总数；不把额外计划 PASS 加入分母。
repeated_failure_rate：修复后紧接着同一步骤再次产生相同 failure_type 的次数除以实施修复次数。
invalid_tool_call_rate：invalid_tool_argument 的工具调用尝试数除以工具调用尝试总数。
avg_tool_calls：实际进入工具 run 的次数平均值；在调用前拦截的参数错误单独计入 attempted_calls。
avg_reflections、avg_retries 分别为反思触发次数和实施修复次数的平均值。
没有分母时记 null，不填零冒充观测。报告真实墙钟耗时；token usage 和 API cost 记 null。

本轮正式服务器/API 实验尚未执行，原因及离线结果见 verification_results.md。
针对真实生物任务的最终 benchmark 与湿实验验证均属于后续工作。
