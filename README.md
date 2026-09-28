# Fixed Enzyme Agent

Fixed Enzyme Agent is a typed, plug-in workflow engine for enzyme engineering.
It separates language understanding from scientific computation: DeepSeek may
select registered tools and propose a workflow, while a deterministic validator
checks the plan before any tool runs. The language model never generates or
executes Python code and never substitutes a biological prediction.

## Why it is useful

- Add or remove tools without editing the planner or executor.
- Let DeepSeek read the complete tool catalog and build a validated task route.
- Select providers by capability, declared input/output artifacts, health, and priority.
- Normalize FASTA, GenBank, CSV, JSON, and sequence-focused SBOL into one contract.
- Accept Chinese or English requests and generate a report in the configured language.
- Save an auditable workflow plan and execution trace for every run.
- Fail clearly when no healthy provider can satisfy a requested capability.
- Run codon optimization as an optional tool for one or more expression hosts.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"  # Windows
# .venv/bin/python -m pip install -e ".[dev]"    # Linux or macOS
```

Model weights and virtual environments are intentionally excluded from Git.
Follow the upstream model instructions or use the sparse setup helper:

```bash
bash scripts/clone_external_models_sparse.sh
```

Copy `config/request.example.json`, update model paths, then run:

```bash
fixed-enzyme-agent check-paths config/my-request.json
fixed-enzyme-agent run config/my-request.json --request "Optimize activity near pH 8"
```

The request may be Chinese or English. Set `report.language` to `zh` or `en`.
Provide the API key through `DEEPSEEK_API_KEY` or `--ask-api-key`; secrets are
never written to configuration or output files.

## Unified files and LLM routing

The flexible route uses `bioartifact/v1` as the mandatory input and output
contract for every model tool. File conversion happens at the workflow boundary;
model adapters may still create their own temporary FASTA or CSV files internally.

Convert a biological file without running models:

```bash
fixed-enzyme-agent convert input.gb normalized.json --standard SBOL --standard RFC10
fixed-enzyme-agent convert normalized.json candidates.fasta
```

Ask DeepSeek to read every installed tool and propose a route:

```bash
export DEEPSEEK_API_KEY="your-key"
fixed-enzyme-agent route config/my-request.json data/input.fasta \
  --request "Rank activity and pH, then optimize codons for E. coli" \
  --output-dir outputs/route-1 \
  --output-format json \
  --output-format fasta
```

Add `--execute` to run the validated route. The route file separately lists
installed optional tools and unavailable external tools; neither category runs
without explicit user action. See [docs/BIOFORMATS.md](docs/BIOFORMATS.md) for
the contract, formats, exports, and RFC handling.

## Iterative mutation rounds

The complete `run` command performs two rounds by default. The best three
candidates from a round become the parent sequences for the next round, and the
same model and hook order is repeated. Configure this without changing model
paths:

```json
{
  "workflow": {
    "rounds": 2,
    "branching_factor": 3,
    "plugin_roots": ["../plugins"],
    "hooks": {
      "after_enzgfm": ["my_filter"],
      "after_ephod": [],
      "after_unistab": []
    }
  }
}
```

Use `--rounds N` for a one-run override. Hook tools are discovered through the
existing plug-in registry and run at the named insertion point without changing
the EnzGFM, EpHod, and UniStab model paths or order.

## Custom mutation FASTA library

Set `candidate.mutation_fasta` to a protein FASTA file to replace exhaustive
single-substitution generation in round one. Each record must have the same
length as the configured WT sequence and contain one or more amino-acid
substitutions. Mutation names are derived from the sequence comparison; FASTA
headers are descriptive only. Later rounds continue normal single-substitution
generation from the selected parent sequences.

```json
{
  "candidate": {
    "mutation_fasta": "../data/my_mutants.fasta"
  }
}
```

The one-run CLI equivalent is `--mutation-fasta /path/to/my_mutants.fasta`.

## Plug-in contract

An in-process tool subclasses `AgentTool`, declares a `ToolSpec`, and implements
`run(context)`. Routed model tools declare `artifact_contract="bioartifact/v1"`
and exactly one `artifact` input and output. Register a tool at startup or place
it in a configured plug-in root:

```text
plugins/my_tool/
  tool.json
  tool.py
```

`tool.py` exports `create_tool(manifest)`. Discovery is restricted to explicit
administrator-configured roots because a Python plug-in is trusted executable
code. Removing the directory removes the provider on the next startup.

See [ARCHITECTURE.md](ARCHITECTURE.md) for planning and validation rules.

## Reproducibility and repository size

本轮增加 Verifier-Gated Failure-Triggered Reflection。`run` 和 `route --execute`
可使用 `--reflection` 开启，默认关闭，修复预算默认为整次运行两次。
只有程序验证失败且存在白名单修复时才调用反思；不会加入 memory、模型训练或 RL。
中文说明见 [服务器指南](docs/server_run_guide.md)、[Docker 指南](docs/docker_guide.md)
和 [方法与实验说明](docs/verifier_reflection_design.md)。

```bash
python experiments/run_verifier_reflection_ablation.py --mode offline \
  --max-tasks 2 --seed 42 --output-dir results/smoke
```

离线实验使用测试替身，并非真实酶预测结果或真实 LLM 能力评测。

```bash
python scripts/validate_repository.py
python -m pytest -q
```

仓库检查器检查已跟踪及未忽略的新文件，限制单文件与源代码总量低于 50,000,000 字节。
源代码与配置保持 ASCII，本轮按要求允许中文 Markdown 和 docs 文档。
生成结果、缓存、模型权重和外部环境不进入提交。当前依赖在 pyproject.toml 中限制版本范围；
正式发布仍需提交经过目标平台验证的 lockfile。源代码大小检查不代表已检查远端 Git 历史体积。

## License

The agent source is available under the MIT License. External model directories
retain their upstream licenses and terms.
