from __future__ import annotations

from pathlib import Path
import csv

from ..adapters import EnzGFMAdapter, EpHodAdapter, UniStabAdapter
from ..bioformats import BioArtifact, BioRecord, SCHEMA_VERSION
from ..codon_optimization import resolve_codon_optimization_request, write_codon_optimization_outputs
from ..runner import build_env
from ..types import Candidate
from .base import AgentTool, ToolContext, ToolHealth, ToolSpec


def _artifact(context: ToolContext) -> BioArtifact:
    value = context.artifacts["artifact"]
    if not isinstance(value, BioArtifact):
        raise TypeError(f"Model tools require {SCHEMA_VERSION}")
    return value


def _reference_and_mutants(artifact: BioArtifact) -> tuple[BioRecord, list[BioRecord]]:
    if not artifact.records:
        raise ValueError("Artifact contains no sequence records")
    reference_id = artifact.data.get("reference_id")
    reference = next((record for record in artifact.records
                      if record.id == reference_id or record.annotations.get("role") == "reference"),
                     artifact.records[0])
    if reference.molecule_type != "protein":
        raise ValueError(
            "Current enzyme prediction models require protein records; "
            "translate or select a CDS before execution"
        )
    mutants = [record for record in artifact.records if record is not reference]
    return reference, mutants


def _mutation_label(reference: str, mutant: str) -> str:
    if len(reference) != len(mutant):
        raise ValueError("Reference and mutant sequences must have equal length")
    parts = [f"{wt}{index}{mt}" for index, (wt, mt) in
             enumerate(zip(reference, mutant), start=1) if wt != mt]
    if not parts:
        raise ValueError("Mutant sequence is identical to the reference")
    return ":".join(parts)


def _runtime(context: ToolContext) -> tuple[dict[str, str], int, bool]:
    runtime = context.config.get("runtime", {})
    force_cpu = bool(context.config.get("force_cpu", False))
    env = build_env(
        force_cpu=force_cpu,
        gpu_id=int(runtime.get("gpu_id", 0)),
        cache_dir=runtime.get("cache_dir", "cache"),
    )
    return env, int(runtime.get("timeout_seconds", 7200)), force_cpu


class EnzGFMTool(AgentTool):
    spec = ToolSpec(
        "enzgfm", "1.0", "Mutation activity proxy provider",
        ("mutation_effect_prediction", "activity_ranking"),
        ("artifact",), ("artifact",),
        properties={"batch": True, "gpu": True, "offline": True,
                    "molecule_types": ["protein"]},
        artifact_contract=SCHEMA_VERSION,
    )

    def __init__(self, model_config: dict):
        self.adapter = EnzGFMAdapter(
            model_config["repo"], model_config["python"], model_config["model_location"]
        )

    def healthcheck(self) -> ToolHealth:
        errors = self.adapter.preflight()
        return ToolHealth(not errors, tuple(errors))

    def run(self, context: ToolContext):
        artifact = _artifact(context)
        reference, mutants = _reference_and_mutants(artifact)
        mutations = [record.annotations.get("mutation") or
                     _mutation_label(reference.sequence, record.sequence) for record in mutants]
        env, timeout, force_cpu = _runtime(context)
        scores = self.adapter.score_mutations(
            reference.sequence, mutations,
            work_dir=context.run_dir / context.step_id, env=env, timeout=timeout,
            force_cpu=force_cpu,
        )
        output = artifact.clone()
        output.data["activity_scores"] = scores
        return {"artifact": output}


class EpHodTool(AgentTool):
    spec = ToolSpec(
        "ephod", "1.0", "Protein optimum pH prediction provider",
        ("ph_prediction",), ("artifact",), ("artifact",),
        properties={"batch": True, "gpu": False, "offline": True,
                    "molecule_types": ["protein"]},
        artifact_contract=SCHEMA_VERSION,
    )

    def __init__(self, model_config: dict):
        self.adapter = EpHodAdapter(model_config["repo"], model_config["python"])

    def healthcheck(self) -> ToolHealth:
        errors = self.adapter.preflight()
        return ToolHealth(not errors, tuple(errors))

    def run(self, context: ToolContext):
        artifact = _artifact(context)
        env, timeout, _ = _runtime(context)
        values = self.adapter.predict(
            {record.id: record.sequence for record in artifact.records},
            work_dir=context.run_dir / context.step_id, env=env, timeout=timeout,
        )
        output = artifact.clone()
        output.data["ph_predictions"] = values
        return {"artifact": output}


class UniStabTool(AgentTool):
    spec = ToolSpec(
        "unistab", "1.0", "Mutation stability prediction provider",
        ("stability_prediction",), ("artifact",), ("artifact",),
        properties={"batch": True, "gpu": True, "offline": True,
                    "molecule_types": ["protein"]},
        artifact_contract=SCHEMA_VERSION,
    )

    def __init__(self, model_config: dict):
        self.adapter = UniStabAdapter(
            model_config["repo"], model_config["python"], model_config["checkpoint"]
        )

    def healthcheck(self) -> ToolHealth:
        errors = self.adapter.preflight()
        return ToolHealth(not errors, tuple(errors))

    def run(self, context: ToolContext):
        artifact = _artifact(context)
        reference, mutants = _reference_and_mutants(artifact)
        env, timeout, force_cpu = _runtime(context)
        values = self.adapter.predict_ddg(
            reference.sequence, {record.id: record.sequence for record in mutants},
            work_dir=context.run_dir / context.step_id, env=env, timeout=timeout,
            force_cpu=force_cpu,
            batch_size=int(context.config.get("models", {}).get("unistab", {}).get("batch_size", 1)),
        )
        output = artifact.clone()
        output.data["stability_predictions"] = values
        return {"artifact": output}


class CodonOptimizationTool(AgentTool):
    spec = ToolSpec(
        "codon_optimizer", "1.0", "Multi-host synonymous codon optimization provider",
        ("codon_optimization",),
        ("artifact",), ("artifact",),
        properties={"batch": True, "gpu": False, "network": "optional",
                    "offline_builtin_hosts": True, "molecule_types": ["protein"]},
        artifact_contract=SCHEMA_VERSION,
    )

    @staticmethod
    def _candidate(value, wt_sequence: str) -> Candidate:
        if isinstance(value, Candidate):
            return value
        if not isinstance(value, dict) or not value.get("mutation"):
            raise ValueError("selected_candidates must contain Candidate objects or mappings")
        candidate = Candidate(mutation=str(value["mutation"]), wt_sequence=wt_sequence)
        candidate.mutant_sequence = value.get("mutant_sequence")
        return candidate

    def run(self, context: ToolContext):
        artifact = _artifact(context)
        reference, mutants = _reference_and_mutants(artifact)
        wt_sequence = reference.sequence
        hosts = artifact.data.get("host_organisms", [])
        if isinstance(hosts, str):
            hosts = [hosts]
        if not isinstance(hosts, (list, tuple)) or not hosts:
            raise ValueError("host_organisms must be a non-empty string array")
        request = resolve_codon_optimization_request(
            "Please perform codon optimization for " + " and ".join(map(str, hosts))
        )
        selected_values = artifact.data.get("selected_candidates")
        if selected_values is None:
            selected_values = [{"mutation": record.annotations.get("mutation") or
                                _mutation_label(wt_sequence, record.sequence),
                                "mutant_sequence": record.sequence} for record in mutants]
        selected = [self._candidate(value, wt_sequence) for value in selected_values]
        result = write_codon_optimization_outputs(
            run_dir=context.run_dir,
            sequence_name=str(context.config.get("sequence_name", "enzyme")),
            wt_sequence=wt_sequence,
            selected=selected,
            resolved_request=request,
            cache_dir=context.config.get("runtime", {}).get("cache_dir"),
            kazusa_timeout=float(context.config.get("runtime", {}).get("kazusa_timeout", 10.0)),
        )
        optimized = []
        if result.get("csv"):
            with Path(result["csv"]).open("r", encoding="utf-8", newline="") as handle:
                optimized = list(csv.DictReader(handle))
        output = artifact.clone()
        output.data["optimized_cds_set"] = optimized
        output.data["codon_optimization_result"] = result
        output.records.extend(
            BioRecord(
                id=str(row.get("name", f"cds_{index}")),
                sequence=str(row["optimized_cds_with_stop"]),
                molecule_type="dna",
                annotations={"host": row.get("organism"), "mutation": row.get("mutation")},
            )
            for index, row in enumerate(optimized, start=1)
        )
        return {"artifact": output}


def register_configured_tools(registry, config: dict) -> list[str]:
    """Register every configured built-in provider without coupling the planner to it."""
    factories = {"enzgfm": EnzGFMTool, "ephod": EpHodTool, "unistab": UniStabTool}
    registered = []
    for name, model_config in config.get("models", {}).items():
        factory = factories.get(name)
        if factory is not None and model_config.get("enabled", True):
            registry.register(factory(model_config))
            registered.append(name)
    if config.get("tools", {}).get("codon_optimizer", {}).get("enabled", True):
        registry.register(CodonOptimizationTool())
        registered.append("codon_optimizer")
    return registered
