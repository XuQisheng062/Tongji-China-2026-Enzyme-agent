from __future__ import annotations

import copy
import json
from dataclasses import replace

from ..planner.models import PlanStep, WorkflowPlan
from ..planner.validator import validate_plan
from ..tools.base import ToolContext
from .controller import VerificationGate, serializable
from .schemas import VerificationResult
from .verifier import Verifier, fail, task_constraints


def enabled(config):
    return bool(config.get("reflection", {}).get("enabled", False) or
                config.get("verification", {}).get("enabled", False))


def artifact_rows(artifact):
    records = artifact.records
    if not records:
        return []
    reference_id = artifact.data.get("reference_id") or next(
        (r.id for r in records if r.annotations.get("role") == "reference"), records[0].id)
    reference = next(r for r in records if r.id == reference_id)
    rows = []
    for record in records:
        if record.id == reference_id or record.molecule_type != "protein":
            continue
        mutation = record.annotations.get("mutation") or ":".join(
            f"{a}{i}{b}" for i, (a, b) in enumerate(zip(reference.sequence, record.sequence), 1) if a != b)
        rows.append({"activity_proxy": artifact.data.get("activity_scores", {}).get(mutation),
                     "ph_opt": artifact.data.get("ph_predictions", {}).get(record.id),
                     "ddg": artifact.data.get("stability_predictions", {}).get(record.id)})
    return rows


def timeout_repairs(action, verdict, config):
    if verdict.evidence.get("exception_type") not in {"TimeoutError", "TimeoutExpired"}:
        return []
    timeout = action.get("arguments", {}).get("timeout_seconds", 7200)
    if type(timeout) is not int:
        return []
    ceiling = config.get("reflection", {}).get("max_timeout_seconds", timeout)
    if timeout >= ceiling:
        return []
    revised = copy.deepcopy(action)
    revised.setdefault("arguments", {})["timeout_seconds"] = min(timeout * 2, ceiling)
    return [{"id": "increase_timeout", "action": revised}]


def guarded_prediction(gate, *, name, step_id, arguments, expected_keys, call):
    verifier = Verifier()
    action = {"step_id": step_id, "tool": name, "arguments": arguments}

    def attempt(current):
        try:
            result = call(current["arguments"], gate.used_retries)
        except Exception as exc:
            return None, fail("tool_execution_failure", str(exc),
                              {"exception_type": type(exc).__name__}), "exception"
        return result, verifier.prediction_map(result, expected_keys), "returned"

    return gate.run(action, attempt, lambda a, v: timeout_repairs(a, v, gate.config))


def execute_verified(registry, plan, *, config, run_dir, initial_artifacts, reflector=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    task = {"user_request": config.get("user_request"), "plan": plan.to_dict(),
            "required_capabilities": list(dict.fromkeys([
                *plan.requested_capabilities,
                *config.get("verification", {}).get("required_capabilities", []),
            ])), "constraints": task_constraints(config)}
    for field, capability in {"ph_opt": "ph_prediction", "activity_proxy": "activity_ranking",
                              "ddg": "stability_prediction"}.items():
        if field in task["constraints"] and capability not in task["required_capabilities"]:
            task["required_capabilities"].append(capability)
    gate = VerificationGate(config, run_dir, reflector=reflector, task=task)
    (run_dir / "verification_input.json").write_text(json.dumps(serializable({
        "task": task, "initial_artifacts": initial_artifacts, "config": config,
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    verifier = Verifier()
    artifacts = copy.deepcopy(initial_artifacts)
    trace, status, error = [], "failed", None
    (run_dir / "00_workflow_plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    try:
        def plan_attempt(action):
            candidate = WorkflowPlan.from_dict(action["plan"])
            capabilities = []
            try:
                for step in candidate.steps:
                    tool = registry.get(step.tool)
                    spec = tool.spec
                    if candidate.initial_artifacts == ("artifact",) and (
                            spec.artifact_contract != "bioartifact/v1" or
                            spec.inputs != ("artifact",) or spec.outputs != ("artifact",)):
                        return None, fail("invalid_tool_argument", "Tool violates bioartifact/v1 contract",
                                          {"tool": step.tool}), "not_called"
                    capabilities.extend(spec.capabilities)
            except KeyError as exc:
                return None, fail("invalid_tool_argument", str(exc)), "not_called"
            verdict = verifier.verify(task=task, state=artifacts, planned_action=action,
                                      tool_call=None, tool_result=None, biological_constraints=task["constraints"],
                                      execution_history=gate.history, phase="plan",
                                      completed_capabilities=capabilities)
            if not verdict.passed:
                return None, verdict, "not_called"
            try:
                validate_plan(candidate, registry)
            except (ValueError, KeyError) as exc:
                return None, fail("invalid_tool_argument", str(exc)), "not_called"
            return candidate, VerificationResult(True), "not_called"

        def plan_repairs(action, verdict):
            if verdict.failure_type != "missing_required_step":
                return []
            candidate = WorkflowPlan.from_dict(action["plan"])
            steps = list(candidate.steps)
            for capability in verdict.evidence["missing_capabilities"]:
                providers = [t for t in registry.providers(capability)
                             if t.spec.inputs == ("artifact",) and t.spec.outputs == ("artifact",)
                             and t.spec.artifact_contract == "bioartifact/v1"]
                if not providers:
                    return []
                tool = providers[0]
                if any(s.tool == tool.spec.name for s in steps):
                    continue
                step_id = f"repair_{len(steps)}_{tool.spec.name}"
                steps.append(PlanStep(step_id, tool.spec.name, tuple(s.id for s in steps)))
            revised = replace(candidate, steps=tuple(steps), requested_capabilities=tuple(task["required_capabilities"]))
            return [{"id": "append_required_steps", "action": {**action, "plan": revised.to_dict()}}]

        plan = gate.run({"step_id": "plan_verification", "tool": "planner", "arguments": {},
                         "plan": plan.to_dict()}, plan_attempt, plan_repairs)
        for step in plan.steps:
            original_tool = registry.get(step.tool)
            base_arguments = {"timeout_seconds": int(config.get("runtime", {}).get("timeout_seconds", 7200))}
            base_arguments.update(config.get("tool_arguments", {}).get(step.tool, {}))
            action = {"step_id": step.id, "tool": step.tool, "arguments": base_arguments}

            def attempt(current):
                tool = registry.get(current["tool"])
                attempt_config = copy.deepcopy(config)
                attempt_config.setdefault("runtime", {})["timeout_seconds"] = current["arguments"].get("timeout_seconds", 7200)
                context = ToolContext(run_dir / f"attempt_{gate.used_retries}", attempt_config,
                                      copy.deepcopy(artifacts), step.id)
                verdict = verifier.verify(task=task, state=context.artifacts, planned_action=current,
                                          tool_call=context, tool_result=None, biological_constraints=task["constraints"],
                                          execution_history=gate.history, tool=tool, phase="argument")
                if not verdict.passed:
                    return None, verdict, "not_called"
                try:
                    result = tool.run(context)
                except Exception as exc:
                    verdict = verifier.verify(task=task, state=artifacts, planned_action=current,
                                              tool_call=context, tool_result=None, biological_constraints=task["constraints"],
                                              execution_history=gate.history, exception=exc)
                    return None, verdict, "exception"
                verdict = verifier.verify(task=task, state=artifacts, planned_action=current,
                                          tool_call=context, tool_result=result, biological_constraints=task["constraints"],
                                          execution_history=gate.history, tool=tool)
                return result, verdict, "returned"

            def repairs(current, verdict):
                options = timeout_repairs(current, verdict, config)
                if verdict.failure_type == "invalid_tool_argument":
                    args = current.get("arguments", {})
                    value = args.get("timeout_seconds")
                    if set(args) == {"timeout_seconds"} and (type(value) is not int or value <= 0):
                        revised = copy.deepcopy(current)
                        revised["arguments"]["timeout_seconds"] = int(config.get("runtime", {}).get("timeout_seconds", 7200))
                        options.append({"id": "restore_runtime_timeout", "action": revised})
                if verdict.failure_type == "tool_execution_failure":
                    for alternative in registry.all():
                        if (alternative.spec.name != current["tool"] and alternative.healthcheck().available
                                and set(original_tool.spec.capabilities) <= set(alternative.spec.capabilities)
                                and alternative.spec.inputs == original_tool.spec.inputs
                                and alternative.spec.outputs == original_tool.spec.outputs
                                and alternative.spec.artifact_contract == original_tool.spec.artifact_contract):
                            options.append({"id": f"switch_{alternative.spec.name}",
                                            "action": {**current, "tool": alternative.spec.name}})
                return options

            result = gate.run(action, attempt, repairs)
            artifacts.update(result)
            snapshot = run_dir / "verified_artifacts"
            snapshot.mkdir(exist_ok=True)
            (snapshot / f"step_{len(trace):04d}.json").write_text(
                json.dumps(serializable(result), ensure_ascii=False, indent=2), encoding="utf-8")
            trace.append({"step": step.id, "tool": gate.history[-1]["tool_name"], "outputs": list(result)})
        artifact = artifacts.get("artifact")
        if artifact is not None and task["constraints"]:
            rows = artifact_rows(artifact)
            gate.run({"step_id": "final_constraints", "tool": "verifier", "arguments": {}},
                     lambda _: (None, verifier.constraints(rows, task["constraints"]), "not_called"),
                     lambda a, v: [])
        status = "success"
        return artifacts
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        gate.finish(status, error)
        (run_dir / "01_execution_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        (run_dir / "verification_status.json").write_text(json.dumps({
            "status": status, "error": error, "trajectory": str(gate.path),
            "retries": gate.used_retries, "completed_steps": [row["step"] for row in trace],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
