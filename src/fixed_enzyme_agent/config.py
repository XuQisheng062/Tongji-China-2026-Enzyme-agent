from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .mutations import normalize_sequence


class ConfigError(ValueError):
    pass


def _resolve_path(value: str | None, base: Path) -> str | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


def _resolve_python(value: str, base: Path) -> str:
    if "/" not in value and "\\" not in value and not value.startswith("."):
        return value
    return _resolve_path(value, base) or value


def validate_candidate_and_selection(data: dict[str, Any], *, strict_ph_target: bool) -> None:
    candidate = data["candidate"]
    if int(candidate["top_k"]) <= 0:
        raise ConfigError("candidate.top_k \u5fc5\u987b > 0")

    for field in ["allowed_positions", "excluded_positions"]:
        value = candidate.get(field)
        if value is not None:
            if not isinstance(value, list) or not all(isinstance(x, int) and x >= 1 for x in value):
                raise ConfigError(f"candidate.{field} \u5fc5\u987b\u662f 1-based \u6b63\u6574\u6570\u6570\u7ec4\u6216 null")

    selection = data["selection"]
    if int(selection["top_k"]) <= 0:
        raise ConfigError("selection.top_k \u5fc5\u987b > 0")
    if selection["stability_direction"] not in {"lower_is_better", "higher_is_better"}:
        raise ConfigError("selection.stability_direction \u53ea\u80fd\u662f lower_is_better/higher_is_better")
    if selection["stability_operator"] not in {"none", "le", "ge"}:
        raise ConfigError("selection.stability_operator \u53ea\u80fd\u662f none/le/ge")
    if selection["stability_operator"] != "none" and selection["stability_threshold"] is None:
        raise ConfigError("\u8bbe\u7f6e stability_operator \u540e\u5fc5\u987b\u63d0\u4f9b stability_threshold")

    weights = selection["weights"]
    for key in ["ph", "activity", "stability"]:
        weights.setdefault(key, 0.0)
        if float(weights[key]) < 0:
            raise ConfigError(f"selection.weights.{key} \u4e0d\u80fd\u4e3a\u8d1f")
    if sum(float(weights[k]) for k in ["ph", "activity", "stability"]) <= 0:
        raise ConfigError("\u81f3\u5c11\u4e00\u4e2a\u6027\u8d28\u6743\u91cd\u5fc5\u987b > 0")

    if strict_ph_target and float(weights["ph"]) > 0 and selection["target_ph"] is None:
        raise ConfigError("ph \u6743\u91cd > 0 \u65f6\u5fc5\u987b\u8bbe\u7f6e target_ph")
    if selection["target_ph"] is not None and not (0.0 <= float(selection["target_ph"]) <= 14.0):
        raise ConfigError("target_ph \u5fc5\u987b\u4f4d\u4e8e 0~14")
    if selection["ph_tolerance"] is not None and float(selection["ph_tolerance"]) <= 0:
        raise ConfigError("ph_tolerance \u5fc5\u987b > 0")
    minimum_ph_weight = float(selection.get("minimum_ph_weight", 0.4))
    if not 0.0 <= minimum_ph_weight <= 1.0:
        raise ConfigError("selection.minimum_ph_weight must be between zero and one")

    workflow = data.get("workflow", {})
    if int(workflow.get("rounds", 2)) <= 0:
        raise ConfigError("workflow.rounds must be greater than zero")
    if int(workflow.get("branching_factor", 3)) <= 0:
        raise ConfigError("workflow.branching_factor must be greater than zero")


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent

    required = ["sequence", "models", "selection"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ConfigError(f"\u914d\u7f6e\u7f3a\u5c11\u5b57\u6bb5: {missing}")

    data["sequence"] = normalize_sequence(data["sequence"])
    reflection = data.setdefault("reflection", {})
    reflection.setdefault("enabled", False)
    reflection.setdefault("max_retries", 2)
    if type(reflection["enabled"]) is not bool:
        raise ConfigError("reflection.enabled must be boolean")
    if type(reflection["max_retries"]) is not int or reflection["max_retries"] < 0:
        raise ConfigError("reflection.max_retries must be a non-negative integer")
    verification = data.setdefault("verification", {})
    verification.setdefault("enabled", False)
    verification.setdefault("required_capabilities", [])
    verification.setdefault("constraints", {})
    if type(verification["enabled"]) is not bool:
        raise ConfigError("verification.enabled must be boolean")
    if not isinstance(verification["required_capabilities"], list) or not all(
            isinstance(item, str) for item in verification["required_capabilities"]):
        raise ConfigError("verification.required_capabilities must be a string array")
    if not isinstance(verification["constraints"], dict):
        raise ConfigError("verification.constraints must be an object")
    for field, rules in verification["constraints"].items():
        if field not in {"activity_proxy", "ph_opt", "ddg"} or not isinstance(rules, dict):
            raise ConfigError(f"Unsupported verification constraint: {field}")
        for operator, threshold in rules.items():
            if (operator not in {"gt", "ge", "lt", "le", "eq"} or
                    type(threshold) not in {int, float} or not math.isfinite(threshold)):
                raise ConfigError(f"Invalid verification constraint: {field}.{operator}")
    if "max_timeout_seconds" in reflection and (type(reflection["max_timeout_seconds"]) is not int or
                                               reflection["max_timeout_seconds"] <= 0):
        raise ConfigError("reflection.max_timeout_seconds must be a positive integer")
    data.setdefault("sequence_name", "enzyme")
    data.setdefault("user_request", None)
    data.setdefault("output_dir", "outputs/run")
    data["output_dir"] = _resolve_path(data["output_dir"], base)

    candidate = data.setdefault("candidate", {})
    candidate.setdefault("top_k", 64)
    candidate.setdefault("allowed_positions", None)
    candidate.setdefault("excluded_positions", [])
    candidate.setdefault("mutation_fasta", None)
    candidate["mutation_fasta"] = _resolve_path(candidate["mutation_fasta"], base)

    runtime = data.setdefault("runtime", {})
    runtime.setdefault("gpu_id", 0)
    runtime.setdefault("timeout_seconds", 7200)
    # keep_temp controls only raw external-model intermediate folders.
    # The tryN result directory is always retained.
    runtime.setdefault("keep_temp", True)
    runtime.setdefault("cache_dir", "cache")
    runtime["cache_dir"] = _resolve_path(runtime["cache_dir"], base)

    workflow = data.setdefault("workflow", {})
    workflow.setdefault("rounds", 2)
    workflow.setdefault("branching_factor", 3)
    workflow.setdefault("plugin_roots", [])
    workflow.setdefault("hooks", {
        "after_enzgfm": [],
        "after_ephod": [],
        "after_unistab": [],
    })
    workflow["plugin_roots"] = [
        _resolve_path(str(value), base) for value in workflow["plugin_roots"]
    ]

    models = data["models"]
    for key in ["ephod", "enzgfm", "unistab"]:
        if key not in models:
            raise ConfigError(f"models \u7f3a\u5c11 {key}")
        if "repo" not in models[key] or "python" not in models[key]:
            raise ConfigError(f"models.{key} \u5fc5\u987b\u5305\u542b repo \u548c python")
        models[key]["repo"] = _resolve_path(models[key]["repo"], base)
        models[key]["python"] = _resolve_python(models[key]["python"], base)

    if "model_location" not in models["enzgfm"]:
        raise ConfigError("models.enzgfm.model_location \u5fc5\u586b")
    models["enzgfm"]["model_location"] = _resolve_path(models["enzgfm"]["model_location"], base)

    if "checkpoint" not in models["unistab"]:
        raise ConfigError("models.unistab.checkpoint \u5fc5\u586b")
    models["unistab"]["checkpoint"] = _resolve_path(models["unistab"]["checkpoint"], base)
    models["unistab"].setdefault("batch_size", 1)
    if int(models["unistab"]["batch_size"]) <= 0:
        raise ConfigError("models.unistab.batch_size must be greater than zero")

    selection = data["selection"]
    selection.setdefault("target_ph", None)
    selection.setdefault("ph_tolerance", None)
    selection.setdefault("min_activity_proxy", None)
    selection.setdefault("stability_direction", "lower_is_better")
    selection.setdefault("stability_operator", "none")
    selection.setdefault("stability_threshold", None)
    selection.setdefault("top_k", 10)
    selection.setdefault("weights", {"ph": 0.5, "activity": 0.3, "stability": 0.2})
    selection.setdefault("minimum_ph_weight", 0.4)

    llm = data.setdefault("llm", {})
    llm.setdefault("enabled", False)
    llm.setdefault("base_url", "https://api.deepseek.com")
    llm.setdefault("parameter_parsing", True)
    llm.setdefault("result_analysis", True)
    llm.setdefault("parameter_model", "deepseek-v4-flash")
    llm.setdefault("analysis_model", "deepseek-v4-pro")
    llm.setdefault("thinking", True)
    llm.setdefault("reasoning_effort", "high")
    llm.setdefault("max_tokens", 4000)
    llm.setdefault("analysis_top_n", 20)
    llm.setdefault("analysis_fail_open", True)

    if llm["parameter_model"] not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        raise ConfigError("llm.parameter_model \u5fc5\u987b\u662f deepseek-v4-flash/deepseek-v4-pro")
    if llm["analysis_model"] not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        raise ConfigError("llm.analysis_model \u5fc5\u987b\u662f deepseek-v4-flash/deepseek-v4-pro")
    if llm["reasoning_effort"] not in {"high", "max"}:
        raise ConfigError("llm.reasoning_effort \u5fc5\u987b\u662f high/max")
    if int(llm["max_tokens"]) <= 0:
        raise ConfigError("llm.max_tokens \u5fc5\u987b > 0")
    if int(llm["analysis_top_n"]) <= 0:
        raise ConfigError("llm.analysis_top_n \u5fc5\u987b > 0")

    visualization = data.setdefault("visualization", {})
    visualization.setdefault("enabled", True)

    report = data.setdefault("report", {})
    report.setdefault("language", "zh")
    if report["language"] not in {"zh", "en"}:
        raise ConfigError("report.language \u53ea\u80fd\u662f zh/en")

    # If LLM parameter parsing is expected to supply target_ph, allow a temporary
    # missing target here; the resolved config is validated strictly before models run.
    strict_ph = not (
        bool(llm.get("enabled"))
        and bool(llm.get("parameter_parsing"))
        and bool(str(data.get("user_request") or "").strip())
    )
    validate_candidate_and_selection(data, strict_ph_target=strict_ph)

    data["_config_path"] = str(cfg_path)
    return data
