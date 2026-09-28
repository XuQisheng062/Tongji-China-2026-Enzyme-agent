# 服务器运行指南

主程序要求 Python 3.10 或更新版本。本轮离线测试使用 Python 3.10.18。
离线实验和 verifier 不需要 CUDA、GPU 或 API key。真实 EnzGFM、EpHod、UniStab
需要分别准备各自环境和权重；GPU/CUDA 版本必须与这些模型的 PyTorch 环境匹配。
UniStab 的 batch_size 保持 1。不要用主程序的环境替代所有外部模型环境。

## 安装与配置

以下命令从项目根目录运行，路径可换成服务器上的实际位置。

```bash
cd /home/xkj2006/agent/FixedEnzymeAgent
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp config/request.example.json config/test.json
```

编辑 `config/test.json` 中的模型 repo、python、checkpoint、model_location，
以及序列、候选库和 selection。示例中的 mutation_fasta 是占位路径，应替换或设为 null。
配置中的相对模型路径按配置文件目录解析。外部模型初始化可参考
`scripts/clone_external_models_sparse.sh` 及三个 external 目录内的环境文件。

```bash
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
fixed-enzyme-agent check-paths config/test.json
fixed-enzyme-agent check-gpu config/test.json
```

API key 只放环境变量或未跟踪的 env 文件，不写入 Git。不要提交真实个人数据或敏感配置。
本轮没有增加运行依赖。发布前须在目标 Linux 环境安装并验证，再保存实际锁定依赖：

```bash
python -m pip freeze > requirements-server.lock.txt
```

检查该文件是否包含本机路径、私有包地址或凭据，再纳入发布版本。当前 pyproject.toml
仍是版本范围声明，不能把它称为已经跨平台验证的完整 lockfile。

## 正式运行

原始固定链路：配置中的 reflection.enabled 和 verification.enabled 均设为 false。

```bash
fixed-enzyme-agent run config/test.json
```

固定链路开启本轮功能：

```bash
fixed-enzyme-agent run config/test.json --reflection
```

自由规划链路开启本轮功能：

```bash
fixed-enzyme-agent route config/test.json data/input.fasta \
  --request '根据配置中明确的活性、最适 pH 和稳定性约束评估候选' \
  --output-dir outputs/verified_route_1 --output-format csv --reflection --execute
```

FASTA 第一条默认为参考序列；建议使用 BioArtifact JSON 显式设置 data.reference_id。
已有候选需要同长度的蛋白质序列。route 当前评估输入候选，不等同于 run 的多轮突变筛选。
正式命令没有故障注入开关；注入代码只在 experiments 目录。

可加入以下配置。max_retries 是整次运行的修复预算，不是每个步骤各两次。
max_timeout_seconds 只有用户明确配置后才允许增加工具超时时间。

```json
{
  "reflection": {"enabled": true, "max_retries": 2, "max_timeout_seconds": 14400},
  "verification": {
    "enabled": true,
    "required_capabilities": ["activity_ranking", "ph_prediction", "stability_prediction"],
    "constraints": {}
  }
}
```

上面是合并到现有配置的片段。数值约束来自 selection 的显式阈值，或
verification.constraints，例如 `{"ph_opt":{"ge":7.0,"le":8.0}}`。
不应把只写在自然语言中的所有约束都当作已可靠解析；运行前确认结构化配置。
没有显式阈值时，verifier 不添加生物学阈值。相同字段同时出现时以
verification.constraints 为准。固定 run 对最终 selected 做检查；route 对导出的全部蛋白候选做检查。

关闭两项配置后恢复原执行路径。只启用 verification.enabled 时验证失败即停止，不调用反思。
固定链路的修复目前仅支持已配置上限内的超时调整；自由链路另外支持同能力工具切换、
补充必需步骤、将错误的调用超时恢复到有效的 runtime.timeout_seconds。
没有可验证修复的错误直接返回失败。

## 测试和实验

```bash
python -m pytest -q
python experiments/run_verifier_reflection_ablation.py --dry-run --max-tasks 2 --seed 42
python experiments/run_verifier_reflection_ablation.py --mode offline \
  --max-tasks 2 --seed 42 --failure-probability 0 0.3 --output-dir results/smoke_1
```

离线完整矩阵（每种故障 2 方法 × 4 概率 × 20 个重复测试任务）：

```bash
for kind in tool_failure invalid_result argument_failure missing_step; do
  python experiments/run_verifier_reflection_ablation.py --mode offline \
    --max-tasks 20 --seed 42 --failure-probability 0 0.1 0.2 0.3 \
    --failure-type "$kind" --output-dir "results/matrix_1/$kind"
done
```

离线模式使用同一生产 executor，但模型和反思器是确定性测试替身，不能据此宣称
真实 DeepSeek 反思有效。20 个任务是同一小型 fixture 的不同故障种子，不是 20 类生物任务。

真实模型实验用冻结计划确保两组的初始路线完全相同。复制并编辑
`config/benchmark_tasks.example.json`；示例仅含三氨基酸序列，必须换成适合模型的真实测试序列。
input 路径相对于任务清单文件。每项都需要 request、input 和 plan；可先用 route 生成
task_route.json，再将其 plan 对象放入清单。两个方法复用同一计划，不重新采样 planner。

```bash
python experiments/run_verifier_reflection_ablation.py --mode real \
  --config config/test.json --tasks config/benchmark_tasks.example.json \
  --max-tasks 10 --seed 42 --failure-probability 0 0.1 0.2 0.3 \
  --failure-type tool_failure --output-dir results/real_1
```

真实模式会执行原有模型并在可恢复失败时调用 DeepSeek；缺少 key、环境、权重时不能完成真实测试。
每种能力只有一个模型时，没有等价备用工具就不会自动恢复普通执行错误；这应计入结果。
注入超时参数错误可以使用 argument_failure。随机故障通过 seed、task_id、tool、attempt
共同决定，两组首次调用使用相同故障安排；额外修复调用也可能失败。

## 后台执行和检查

项目未发现 SLURM 或 LSF 提交脚本，因此这里只给通用 nohup：

```bash
mkdir -p logs
nohup .venv/bin/python experiments/run_verifier_reflection_ablation.py \
  --mode offline --max-tasks 20 --seed 42 --failure-probability 0 0.1 0.2 0.3 \
  --output-dir results/background_1 > logs/verifier_reflection.log 2>&1 &
echo $!
tail -f logs/verifier_reflection.log
ps -ef | grep '[r]un_verifier_reflection_ablation.py'
```

实验目录包含 raw（每任务结果及注入记录）、trajectories（JSONL）、summary.json、summary.csv
和 experiment.json。正式 route 的 execution 目录包含 verification_input.json、
verification_status.json、verified_artifacts、trajectories 和原有执行 trace。
固定 run 仍写 outputs 下 tryN，额外保存 trajectories。

程序不支持 resume。失败后保留已有目录，修改根因，指定新的 output-dir 重新运行。
实验拒绝覆盖已有输出目录；run 会自动产生新的 tryN。没有成功的最终结果时，不能把部分输出写为成功。
token usage 和 API cost 尚不能从当前包装接口可靠取得，结果使用 null，不估算。
