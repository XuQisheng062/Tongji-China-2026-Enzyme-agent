# Architecture

## Safety boundary

```text
Biological file -> Format converter -> BioArtifact v1
                                          |
User request + complete ToolRegistry -> WorkflowPlanner
                                          |
                                     TaskRoute JSON
                               /          |           \
                    executable plan  additions  external ideas
                               |
                         Plan validator
                               |
                    Deterministic executor
                               |
                         BioArtifact v1
                               |
                         Format exports
```

DeepSeek converts natural language into goals and may choose among healthy
providers. It receives only the serialized tool catalog. Its JSON response is
parsed into `WorkflowPlan` and rejected unless every tool exists, every input is
available, dependencies are topologically ordered, and all requested
capabilities are satisfied. The richer route planner also reports compatible
installed tools that were not requested and external tools that require user
action. Suggestions are never executed implicitly.

The executor invokes only objects already registered by the application. It
does not evaluate model-produced source code, shell commands, module names, or
paths. Scientific results are produced by tools, never by the language model.

## Tool contract

`ToolSpec` declares a stable name, version, capabilities, input artifact types,
output artifact types, selection priority, optional properties, and an optional
artifact contract. `AgentTool` adds health checking plus input/output validation.
Every model in an LLM-generated route must consume and return `bioartifact/v1`
under the single artifact name `artifact`; route construction rejects any tool
that violates this rule. A provider with missing model
weights or an unsuitable runtime remains visible but unavailable, so planning
fails with an actionable error instead of silently degrading.

## Format boundary

`BioFormatConverter` imports FASTA, GenBank, CSV, BioArtifact JSON, SBOL XML, and
SBOL JSON-LD into the canonical artifact. It exports results to the same formats.
The current SBOL and GenBank handlers focus on sequence interchange; complete
design graphs remain available for future converter plug-ins without changing
model interfaces.

SBOL is represented both as a file standard and a standards tag. iGEM and
BioBricks RFC identifiers are metadata because RFCs define design and assembly
rules rather than one universal serialization syntax.

`ToolRegistry` supports register, unregister, lookup, capability queries, and
discovery from explicit plug-in roots. Multiple providers may implement the same
capability. The deterministic planner selects the lowest priority value and then
the lexicographically first name; the LLM planner may use tool properties and
user constraints, but its selection is subject to the same validator.

## Execution and audit trail

Plans are DAGs represented in topological order. Independent branches can be
made concurrent later without changing the tool interface. Before execution,
the engine writes `00_workflow_plan.json`; after execution it writes
`01_execution_trace.json`. These files record choices and produced artifact
names without exposing API keys.

## Extension rules

Adding a provider must not require changes to the registry, planner, validator,
or executor. A plug-in supplies `tool.json` and `tool.py`. Tool names and artifact
types are API identifiers and must remain English ASCII. Plug-ins are trusted
code and should only be installed from reviewed sources.

## Codon optimization provider

The built-in `codon_optimizer` consumes a canonical artifact containing protein
records and `host_organisms`. It returns the same artifact with DNA records and
an `optimized_cds_set`, while preserving the existing CSV and FASTA result
bundle. Built-in host profiles work offline; other organisms may use the Kazusa
cache and network.
