# Fixed Enzyme Agent

Fixed Enzyme Agent is a typed, plug-in workflow engine for enzyme engineering.
It separates language understanding from scientific computation: DeepSeek may
select registered tools and propose a workflow, while a deterministic validator
checks the plan before any tool runs. The language model never generates or
executes Python code and never substitutes a biological prediction.

## Why it is useful

- Add or remove tools without editing the planner or executor.
- Select providers by capability, declared input/output artifacts, health, and priority.
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

## Plug-in contract

An in-process tool subclasses `AgentTool`, declares a `ToolSpec`, and implements
`run(context)`. Register it at startup or place it in a configured plug-in root:

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

```bash
python scripts/validate_repository.py
python -m pytest -q
```

The validator rejects non-English repository text, tracked files of 50 MB or
more, and a total source tree above 50 MB. Generated outputs, caches, model
weights, and external environments are ignored. Dependencies are bounded in
`pyproject.toml`; releases should also publish a platform-specific lock file.

## License

The agent source is available under the MIT License. External model directories
retain their upstream licenses and terms.
