# 本轮验证记录

记录日期：2026-09-27。测试主环境为 Windows、Python 3.10.18；
numpy 1.26.4、pandas 2.3.3、matplotlib 3.10.9、pytest 9.1.1。

全部 59 项 pytest 测试通过，其中原有 34 项继续通过。
新增测试覆盖正常结果、工具异常、空返回、字段缺失、NaN、显式约束、非法残基位置、
多位点突变、可判定冲突、PASS 不反思、可恢复失败反思、恢复成功、预算耗尽、
相同动作拦截、非法反思动作拦截、成功状态保留、关闭后原行为、固定模型超时修复、
trajectory、固定 seed 复现和四种故障注入。

执行了 Python compileall 与 git diff --check，均通过。当前 Python 环境未安装 ruff，
因此不声称 ruff 检查已通过；格式检查采用 Git 空白检查，源码语法用 compileall 检查。
仓库检查包含未忽略的新文件，允许本轮要求的中文文档，继续限制英文源码和配置。

执行了独立离线 smoke test，以及四类故障的完整概率矩阵。
每个方法、概率和故障类型使用 20 个测试任务重复，共 640 次离线任务运行。
这些任务复用一个小型序列 fixture，变化来自注入 seed，不代表 20 类独立生物学任务。
模型与反思器都是明确标注的测试替身，未调用 DeepSeek 或实际酶预测模型。

以下成功率为实际运行输出；每对数字依次为 Baseline / VerifierReflection，
概率顺序为 0、0.1、0.2、0.3。

| 故障类型 | 0 | 0.1 | 0.2 | 0.3 |
| --- | --- | --- | --- | --- |
| tool_failure | 1.00 / 1.00 | 0.75 / 0.95 | 0.55 / 0.80 | 0.30 / 0.70 |
| invalid_result | 1.00 / 1.00 | 0.75 / 0.95 | 0.55 / 0.80 | 0.30 / 0.70 |
| argument_failure | 1.00 / 1.00 | 0.90 / 1.00 | 0.85 / 1.00 | 0.85 / 1.00 |
| missing_step | 1.00 / 1.00 | 0.95 / 1.00 | 0.85 / 1.00 | 0.80 / 1.00 |

最终可追溯输出位于 `results/verifier_reflection_release/<failure_type>/`，
每类有 summary.json、summary.csv、experiment.json、raw 和 trajectories。
零故障情况下两组成功率相同，新方法没有触发反思。
工具失败概率 0.3 时仍有修复失败，不能声称闭环保证任务成功。
这些结果只证明当前注入和替身条件下的工程恢复行为，不证明真实 LLM 反思质量、
真实酶模型可靠性、科学发现能力或湿实验效果，也没有检验统计显著性。

真实实验尚在进行准备，以下部分为计划，不填写虚构数值：
真实 DeepSeek API、EnzGFM/EpHod/UniStab GPU 实验、真实任务集最终 benchmark、湿实验比对。
当前环境不存在 DEEPSEEK_API_KEY，且执行测试的 Python 没有 openai SDK；
本次会话也没有已配置可运行的 Linux 模型服务器和权重环境。
因此真实模式仅完成代码接入，尚未验证 API 或 GPU 执行结果。

Dockerfile 和操作指南已经生成，但当前没有 docker 可执行程序，镜像 build/run/save/load 未实际运行。
目标平台依赖锁定、镜像构建验证与 iGEM GitLab 发布仍需完成。
token usage 和 API cost 没有可靠观测值，输出 null。
