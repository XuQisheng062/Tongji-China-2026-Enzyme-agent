"""Paired ablation. Offline fixtures are explicitly not enzyme predictions."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fixed_enzyme_agent.bioformats import BioArtifact, BioRecord, BioFormatConverter
from fixed_enzyme_agent.executor import WorkflowExecutor
from fixed_enzyme_agent.planner import GoalSpec, WorkflowPlanner, WorkflowPlan
from fixed_enzyme_agent.tools import AgentTool, ToolRegistry, ToolSpec, register_configured_tools
from fixed_enzyme_agent.verification import LLMReflector, Verifier
from fixed_enzyme_agent.verification.controller import serializable
from fixed_enzyme_agent.verification.verifier import PREDICTIONS, fail, task_constraints
from fixed_enzyme_agent.verification.execution import artifact_rows


class FixtureTool(AgentTool):
    def __init__(self, name, capability):
        self.spec = ToolSpec(name, "fixture-1", "Offline software test fixture; not a biological model",
                             (capability,), ("artifact",), ("artifact",), artifact_contract="bioartifact/v1")

    def run(self, context):
        artifact = copy.deepcopy(context.artifacts["artifact"])
        field = PREDICTIONS[self.spec.capabilities[0]]
        keys = [r.id for r in artifact.records] if field == "ph_predictions" else ["K2R"]
        artifact.data[field] = {key: 1.0 for key in keys}
        return {"artifact": artifact}


class FixtureReflector:
    """Deterministic protocol test double. Never used by production commands."""
    def reflect(self, payload):
        return {"failure_analysis": "\u7a0b\u5e8f\u8bc1\u636e\u8868\u660e\u672c\u6b21\u8c03\u7528\u5931\u8d25",
                "repair_strategy": "\u9009\u62e9\u7b2c\u4e00\u4e2a\u7a0b\u5e8f\u5141\u8bb8\u7684\u4fee\u590d",
                "repair_id": payload["allowed_repairs"][0]["id"]}


class FailureInjector:
    def __init__(self, seed, task_id, probability, kind):
        self.seed, self.task_id, self.probability, self.kind = seed, task_id, probability, kind
        self.counts = {}
        self.events = []

    def draw(self, key):
        index = self.counts.get(key, 0)
        self.counts[key] = index + 1
        digest = hashlib.sha256(f"{self.seed}:{self.task_id}:{key}:{index}".encode()).digest()
        sampled = random.Random(int.from_bytes(digest[:8], "big")).random() < self.probability
        self.events.append({"key": key, "attempt": index, "injected": sampled, "type": self.kind})
        return sampled


class ObservedTool(AgentTool):
    def __init__(self, tool, injector, observer, can_switch=False):
        self.tool, self.spec = tool, tool.spec
        self.injector, self.observer = injector, observer
        self.can_switch = can_switch

    def healthcheck(self):
        return self.tool.healthcheck()

    def validate_input(self, context):
        self.tool.validate_input(context)
        args = context.config.get("tool_arguments", {}).get(self.spec.name, {})
        value = context.config.get("runtime", {}).get("timeout_seconds", 7200)
        if not context.config.get("reflection", {}).get("enabled"):
            value = args.get("timeout_seconds", value)
        if type(value) is not int or value <= 0:
            raise ValueError("timeout_seconds must be a positive integer")

    def run(self, context):
        started = time.perf_counter()
        action = {"tool": self.spec.name, "step_id": context.step_id, "arguments": {}}
        status = "returned"
        try:
            injected = (self.injector.kind in {"tool_failure", "invalid_result"} and
                        self.injector.draw(self.spec.name))
            if injected and self.injector.kind == "tool_failure":
                raise RuntimeError("Injected benchmark tool failure")
            result = self.tool.run(context)
            if injected:
                result = {"artifact": copy.deepcopy(context.artifacts["artifact"])}
                for capability in self.spec.capabilities:
                    result["artifact"].data.pop(PREDICTIONS.get(capability), None)
            verdict = Verifier().verify(task={}, state=context.artifacts, planned_action=action,
                                        tool_call=context, tool_result=result, biological_constraints={},
                                        execution_history=[], tool=self.tool)
            return result
        except Exception as exc:
            status = "exception"
            verdict = fail("tool_execution_failure", str(exc), {"exception_type": type(exc).__name__})
            raise
        finally:
            self.observer.append({"step_id": context.step_id, "planned_action": action,
                                  "tool_name": self.spec.name, "tool_arguments": {},
                                  "tool_result_status": status, "verification_passed": verdict.passed,
                                  "failure_type": verdict.failure_type, "failure_reason": verdict.reason,
                                  "recoverable": not verdict.passed and self.can_switch,
                                  "reflection_triggered": False, "repair_action": None,
                                  "retry_index": 0, "final_status": "observed_only",
                                  "latency_seconds": time.perf_counter() - started})


def metrics(events, calls):
    attempts = [e for e in events if e.get("event") != "task_end"]
    tool_attempts = [e for e in attempts if e.get("tool_name") not in {"planner", "verifier"}]
    failures = [e for e in attempts if not e["verification_passed"] and e.get("recoverable")]
    recovered = 0
    repeated = 0
    repairs = [e for e in attempts if e.get("repair_action") is not None]
    for event in failures:
        later = attempts[attempts.index(event) + 1:]
        if any(e["step_id"] == event["step_id"] and e["verification_passed"] for e in later):
            recovered += 1
    for event in repairs:
        later = attempts[attempts.index(event) + 1:]
        following = next((e for e in later if e["step_id"] == event["step_id"]), None)
        repeated += bool(following and following["failure_type"] == event["failure_type"])
    return {"recoverable_failures": len(failures), "recovered_failures": recovered,
            "repeated_failures": repeated, "repair_attempts": len(repairs),
            "verification_passes": sum(e["verification_passed"] for e in tool_attempts),
            "verification_checks": len(tool_attempts),
            "invalid_calls": sum(e["failure_type"] == "invalid_tool_argument" for e in attempts),
            "attempted_calls": sum(e.get("tool_name") not in {"planner", "verifier"} for e in attempts),
            "tool_calls": calls,
            "reflections": sum(e["reflection_triggered"] for e in attempts), "retries": len(repairs)}


def run_task(*, method, probability, index, seed, kind, root, mode, config=None, manifest=None, client=None, model=None):
    task_id = f"task_{index:04d}"
    run_dir = root / "raw" / f"{kind}_{probability:g}" / method / task_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(config or {"runtime": {"timeout_seconds": 10}, "selection": {}})
    cfg["reflection"] = {**cfg.get("reflection", {}), "enabled": method == "VerifierReflection", "max_retries": 2}
    cfg["verification"] = {**cfg.get("verification", {}), "enabled": method == "VerifierReflection"}
    registry = ToolRegistry()
    if mode == "offline":
        artifact = BioArtifact(records=[BioRecord("WT", "MKW", "protein"),
                                        BioRecord("K2R", "MRW", "protein", annotations={"mutation": "K2R"})])
        for name, capability in [("activity", "activity_ranking"), ("ph", "ph_prediction"),
                                 ("stability", "stability_prediction")]:
            registry.register(FixtureTool(name, capability))
            registry.register(FixtureTool(name + "_backup", capability))
        plan = WorkflowPlanner(registry).deterministic(GoalSpec(("activity_ranking", "ph_prediction", "stability_prediction")))
        reflector = FixtureReflector()
    else:
        register_configured_tools(registry, cfg)
        registry.discover(cfg.get("workflow", {}).get("plugin_roots", []))
        artifact = BioFormatConverter().load(manifest["input"])
        plan = WorkflowPlan.from_dict(manifest["plan"])
        cfg["user_request"] = manifest.get("request", "")
        reflector = LLMReflector(client, model)
    injector = FailureInjector(seed, task_id, probability, kind)
    observer = []
    for tool in registry.all():
        can_switch = any(t.spec.name != tool.spec.name and
                         set(tool.spec.capabilities) <= set(t.spec.capabilities) and t.healthcheck().available
                         for t in registry.all())
        registry.register(ObservedTool(tool, injector, observer, can_switch), replace=True)
    if kind == "argument_failure" and injector.draw("argument") and plan.steps:
        cfg.setdefault("tool_arguments", {})[plan.steps[0].tool] = {"timeout_seconds": "invalid"}
    if kind == "missing_step" and injector.draw("plan") and plan.steps:
        plan = WorkflowPlan(plan.steps[:-1], plan.requested_capabilities, plan.initial_artifacts, plan.rationale)
    started, result, error = time.perf_counter(), None, None
    try:
        result = WorkflowExecutor(registry, reflector=reflector).execute(
            plan, config=cfg, run_dir=run_dir, initial_artifacts={"artifact": artifact})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    # The same external success oracle is used for both arms; baseline never sees it.
    success = result is not None and error is None
    if success:
        for capability in plan.requested_capabilities:
            field = PREDICTIONS.get(capability)
            if field and not result["artifact"].data.get(field):
                success = False
        for tool in registry.all():
            if set(tool.spec.capabilities) & set(plan.requested_capabilities):
                verdict = Verifier().verify(task={}, state={"artifact": artifact}, planned_action={},
                                            tool_call=None, tool_result=result, biological_constraints={},
                                            execution_history=[], tool=tool)
                success = success and verdict.passed
        success = success and Verifier.constraints(artifact_rows(result["artifact"]), task_constraints(cfg)).passed
    trajectory_files = list((run_dir / "trajectories").glob("*.jsonl"))
    if trajectory_files:
        events = [json.loads(line) for line in trajectory_files[0].read_text(encoding="utf-8").splitlines()]
    else:
        events = observer
        if error and not events:
            failure_type = "missing_required_step" if kind == "missing_step" else "invalid_tool_argument"
            events = [{"step_id": "plan_verification" if kind == "missing_step" else plan.steps[0].id,
                       "planned_action": plan.to_dict(), "tool_name": "planner" if kind == "missing_step" else plan.steps[0].tool,
                       "tool_arguments": cfg.get("tool_arguments", {}), "tool_result_status": "not_called",
                       "verification_passed": False, "failure_type": failure_type,
                       "failure_reason": error, "recoverable": True, "reflection_triggered": False,
                       "repair_action": None, "retry_index": 0, "final_status": "failed"}]
        events = [{"task_id": task_id, **event} for event in events]
        events.append({"task_id": task_id, "step_id": None, "planned_action": None,
                       "tool_name": None, "tool_arguments": {}, "tool_result_status": "not_called",
                       "verification_passed": success, "failure_type": None, "failure_reason": error,
                       "reflection_triggered": False, "repair_action": None, "retry_index": 0,
                       "final_status": "success" if success else "failed", "event": "task_end"})
    trajectory = root / "trajectories" / f"{kind}_{probability:g}_{method}_{task_id}.jsonl"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_text("".join(json.dumps(serializable(e), ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
    row = {"method": method, "failure_probability": probability, "task_id": task_id,
           "failure_type": kind, "mode": mode, "success": success,
           "latency_seconds": time.perf_counter() - started, "error": error,
           "token_usage": None, "api_cost": None, **metrics(events, len(observer))}
    (run_dir / "record.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (run_dir / "injections.json").write_text(json.dumps(injector.events, indent=2), encoding="utf-8")
    return row


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired verifier/reflection ablation")
    parser.add_argument("--mode", choices=["offline", "real"], default="offline")
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--failure-probability", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3])
    parser.add_argument("--failure-type", choices=["tool_failure", "invalid_result", "argument_failure", "missing_step"], default="tool_failure")
    parser.add_argument("--output-dir", type=Path, default=Path("results/verifier_reflection"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--tasks", help="JSON array of input paths and frozen WorkflowPlan objects")
    parser.add_argument("--model", default="deepseek-v4-flash")
    args = parser.parse_args(argv)
    if args.max_tasks < 1 or any(p < 0 or p > 1 for p in args.failure_probability):
        parser.error("max-tasks must be positive; probabilities must be between 0 and 1")
    if args.mode == "real" and not (args.config and args.tasks):
        parser.error("real mode requires --config and --tasks")
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "max_tasks": args.max_tasks, "seed": args.seed,
                          "failure_probability": args.failure_probability, "failure_type": args.failure_type,
                          "methods": ["Baseline", "VerifierReflection"], "execution": "NOT_RUN"}, indent=2))
        return
    cfg, tasks, client = None, None, None
    if args.mode == "real":
        from fixed_enzyme_agent.config import load_config
        from fixed_enzyme_agent.llm.deepseek import DeepSeekClient
        cfg = load_config(args.config)
        tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))[:args.max_tasks]
        if not tasks:
            parser.error("tasks must not be empty")
        for task in tasks:
            task["input"] = str((Path(args.tasks).resolve().parent / task["input"]).resolve())
        client = DeepSeekClient(base_url=cfg.get("llm", {}).get("base_url", "https://api.deepseek.com"))
    count = len(tasks) if tasks else args.max_tasks
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "experiment.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "offline_fixture_only": args.mode == "offline", "temperature": "unchanged_provider_default",
        "paired_design": "same frozen plan, tools, inputs, seed and injection schedule; only gate enabled differs",
        "token_usage": "NOT_AVAILABLE", "api_cost": "NOT_AVAILABLE",
    }, indent=2), encoding="utf-8")
    rows = []
    for probability in args.failure_probability:
        for method in ["Baseline", "VerifierReflection"]:
            group = [run_task(method=method, probability=probability, index=index, seed=args.seed,
                              kind=args.failure_type, root=args.output_dir, mode=args.mode,
                              config=cfg, manifest=tasks[index] if tasks else None, client=client, model=args.model)
                     for index in range(count)]
            total = lambda key: sum(row[key] for row in group)
            rows.append({"method": method, "failure_probability": probability, "num_tasks": count,
                         "success_rate": total("success") / count, "final_success_rate": total("success") / count,
                         "failure_recovery_rate": ratio(total("recovered_failures"), total("recoverable_failures")),
                         "verification_pass_rate": ratio(total("verification_passes"), total("verification_checks")),
                         "repeated_failure_rate": ratio(total("repeated_failures"), total("repair_attempts")),
                         "invalid_tool_call_rate": ratio(total("invalid_calls"), total("attempted_calls")),
                         "avg_tool_calls": total("tool_calls") / count, "avg_reflections": total("reflections") / count,
                         "avg_retries": total("retries") / count, "avg_latency_seconds": total("latency_seconds") / count,
                         "recoverable_failures": total("recoverable_failures"), "recovered_failures": total("recovered_failures"),
                         "mode": args.mode, "failure_type": args.failure_type, "token_usage": None, "api_cost": None})
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "rows": len(rows), "mode": args.mode}))


if __name__ == "__main__":
    main()
