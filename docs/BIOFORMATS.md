# Biological Format Interoperability

## Canonical contract

The routed workflow uses one in-memory contract, `bioartifact/v1`. Every routed
model tool must declare exactly one `artifact` input and one `artifact` output.
The executor validates the Python type before and after every tool call. A tool
that uses another contract cannot enter an LLM-generated executable route.

A `BioArtifact` contains sequence records, tabular or prediction data, metadata,
and standards tags. This keeps the scientific payload stable while adapters
materialize the model-specific FASTA or CSV files required by external models.

## Supported formats

| Format | Input | Output | Notes |
| --- | --- | --- | --- |
| FASTA | Yes | Yes | DNA, RNA, or protein sequences |
| GenBank | Yes | Yes | Sequence-focused interchange |
| CSV | Yes | Yes | Requires a `sequence` column on input |
| BioArtifact JSON | Yes | Yes | Lossless canonical representation |
| SBOL XML | Yes | Yes | Sequence-focused SBOL interchange |
| SBOL JSON-LD | Yes | Yes | Sequence-focused SBOL interchange |

SBOL is a data standard. iGEM and BioBricks RFCs are specifications for design,
assembly, naming, and interfaces; they are stored as standards metadata such as
`RFC10`. An RFC identifier is not treated as a standalone file syntax.

## Conversion

```bash
fixed-enzyme-agent convert input.gb normalized.json --standard SBOL --standard RFC10
fixed-enzyme-agent convert normalized.json candidates.fasta
fixed-enzyme-agent convert normalized.json design.xml --to-format sbol
```

Formats are detected from file extensions unless `--from-format` or
`--to-format` is supplied. Supported explicit names are `fasta`, `genbank`,
`csv`, `json`, `sbol`, and `sbol-json`.

## LLM task routing

```bash
export DEEPSEEK_API_KEY="your-key"
fixed-enzyme-agent route config/test.json data/input.fasta \
  --request "Rank activity and pH, then optimize codons for E. coli" \
  --output-dir outputs/route-1 \
  --output-format json \
  --output-format fasta \
  --execute
```

The planner receives the complete installed tool catalog, the user request, an
input summary, and requested output formats. It separates its response into:

- a validated executable route containing installed and healthy tools only;
- optional installed tools that may improve the workflow but are not executed;
- external tool suggestions that require explicit user action;
- practical guidance and the input/output conversion plan.

The output directory contains `normalized_input.json`, `task_route.json`, the
execution audit files, and one exported result file per requested format.
