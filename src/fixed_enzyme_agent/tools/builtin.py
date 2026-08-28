from __future__ import annotations

from pathlib import Path
import csv

from ..adapters import EnzGFMAdapter, EpHodAdapter, UniStabAdapter
from ..codon_optimization import resolve_codon_optimization_request, write_codon_optimization_outputs
from ..runner import build_env
from ..types import Candidate
from .base import AgentTool, ToolContext, ToolHealth, ToolSpec


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
        ("protein_sequence", "mutation_list"), ("activity_scores",),
        properties={"batch": True, "gpu": True, "offline": True},
    )

    def __init__(self, model_config: dict):
        self.adapter = EnzGFMAdapter(
            model_config["repo"], model_config["python"], model_config["model_location"]
        )

    def healthcheck(self) -> ToolHealth:
        errors = self.adapter.preflight()
        return ToolHealth(not errors, tuple(errors))

    def run(self, context: ToolContext):
        env, timeout, force_cpu = _runtime(context)
        scores = self.adapter.score_mutations(
            context.artifacts["protein_sequence"], context.artifacts["mutation_list"],
            work_dir=context.run_dir / context.step_id, env=env, timeout=timeout,
            force_cpu=force_cpu,
        )
        return {"activity_scores": scores}


class EpHodTool(AgentTool):
    spec = ToolSpec(
        "ephod", "1.0", "Protein optimum pH prediction provider",
        ("ph_prediction",), ("protein_sequences",), ("ph_predictions",),
        properties={"batch": True, "gpu": False, "offline": True},
    )

    def __init__(self, model_config: dict):
        self.adapter = EpHodAdapter(model_config["repo"], model_config["python"])

    def healthcheck(self) -> ToolHealth:
        errors = self.adapter.preflight()
        return ToolHealth(not errors, tuple(errors))

    def run(self, context: ToolContext):
        env, timeout, _ = _runtime(context)
        values = self.adapter.predict(
            context.artifacts["protein_sequences"],
            work_dir=context.run_dir / context.step_id, env=env, timeout=timeout,
        )
        return {"ph_predictions": values}


class UniStabTool(AgentTool):
    spec = ToolSpec(
        "unistab", "1.0", "Mutation stability prediction provider",
        ("stability_prediction",), ("protein_sequence", "mutant_sequences"),
        ("stability_predictions",),
        properties={"batch": True, "gpu": True, "offline": True},
    )

    def __init__(self, model_config: dict):
        self.adapter = UniStabAdapter(
            model_config["repo"], model_config["python"], model_config["checkpoint"]
        )

    def healthcheck(self) -> ToolHealth:
        errors = self.adapter.preflight()
        return ToolHealth(not errors, tuple(errors))

    def run(self, context: ToolContext):
        env, timeout, force_cpu = _runtime(context)
        values = self.adapter.predict_ddg(
            context.artifacts["protein_sequence"], context.artifacts["mutant_sequences"],
            work_dir=context.run_dir / context.step_id, env=env, timeout=timeout,
            force_cpu=force_cpu,
        )
        return {"stability_predictions": values}


class CodonOptimizationTool(AgentTool):
    spec = ToolSpec(
        "codon_optimizer", "1.0", "Multi-host synonymous codon optimization provider",
        ("codon_optimization",),
        ("protein_sequence", "selected_candidates", "host_organisms"),
        ("optimized_cds_set", "codon_optimization_result"),
        properties={"batch": True, "gpu": False, "network": "optional", "offline_builtin_hosts": True},
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
        wt_sequence = str(context.artifacts["protein_sequence"])
        hosts = context.artifacts["host_organisms"]
        if isinstance(hosts, str):
            hosts = [hosts]
        if not isinstance(hosts, (list, tuple)) or not hosts:
            raise ValueError("host_organisms must be a non-empty string array")
        request = resolve_codon_optimization_request(
            "Please perform codon optimization for " + " and ".join(map(str, hosts))
        )
        selected = [self._candidate(value, wt_sequence) for value in context.artifacts["selected_candidates"]]
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
        return {"optimized_cds_set": optimized, "codon_optimization_result": result}


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
