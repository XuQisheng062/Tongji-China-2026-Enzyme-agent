import json
from pathlib import Path

import pytest

from fixed_enzyme_agent.bioformats import BioArtifact, BioRecord
from fixed_enzyme_agent.executor import WorkflowExecutor
from fixed_enzyme_agent.planner import GoalSpec, WorkflowPlanner, WorkflowPlan, PlanStep
from fixed_enzyme_agent.tools import ToolContext, ToolRegistry
from fixed_enzyme_agent.verification import VerificationGate, VerificationResult, VerifiedExecutionError, Verifier
from fixed_enzyme_agent.verification.verifier import fail, task_constraints
from fixed_enzyme_agent.verification.execution import guarded_prediction
from experiments.run_verifier_reflection_ablation import FixtureTool, FixtureReflector, FailureInjector, main


def artifact():
    return BioArtifact(records=[BioRecord("WT", "MKW", "protein"),
                                BioRecord("K2R", "MRW", "protein", annotations={"mutation": "K2R"})])


def verify(result=None, exception=None, phase="result", input_artifact=None):
    tool = FixtureTool("ph", "ph_prediction")
    state = {"artifact": input_artifact or artifact()}
    return Verifier().verify(task={}, state=state, planned_action={"tool": "ph"},
                             tool_call=ToolContext(Path("."), {}, state, "step_1"),
                             tool_result=result, biological_constraints={}, execution_history=[],
                             tool=tool, exception=exception, phase=phase)


def test_normal_result_passes():
    value = artifact()
    value.data["ph_predictions"] = {"WT": 7.0, "K2R": 7.2}
    assert verify({"artifact": value}).passed


@pytest.mark.parametrize("result", [{}, {"artifact": artifact()}, {"wrong": 2}, None])
def test_missing_required_field_fails(result):
    assert verify(result).failure_type == "tool_execution_failure"


def test_exception_and_nan_fail():
    assert verify(exception=TimeoutError("expired")).failure_type == "tool_execution_failure"
    value = artifact()
    value.data["ph_predictions"] = {"WT": float("nan"), "K2R": 7.0}
    assert not verify({"artifact": value}).passed


def test_constraints_only_come_from_task():
    assert task_constraints({"selection": {}}) == {}
    assert Verifier.constraints([{"ph_opt": 2.0}], {}).passed
    verdict = Verifier.constraints([{"ph_opt": 2.0}], {"ph_opt": {"ge": 6.0, "le": 8.0}})
    assert verdict.failure_type == "constraint_violation"
    assert not verdict.recoverable
    assert verdict.scope == "scientific_objective_satisfaction"


def test_invalid_residue_index_and_sequence_conflict():
    value = artifact()
    value.records[1].annotations["mutation"] = "K99R"
    assert verify(phase="argument", input_artifact=value).failure_type == "invalid_tool_argument"
    value.records[1].annotations["mutation"] = "K2A"
    assert verify(phase="argument", input_artifact=value).failure_type == "inconsistent_result"


def test_multiple_substitutions_are_valid():
    value = artifact()
    value.records[1].sequence = "ARA"
    value.records[1].annotations["mutation"] = "M1A:K2R:W3A"
    assert verify(phase="argument", input_artifact=value).passed


def test_pass_does_not_reflect(tmp_path):
    class MustNotReflect:
        def reflect(self, payload):
            raise AssertionError("PASS cannot reflect")
    gate = VerificationGate({"reflection": {"enabled": True}}, tmp_path, reflector=MustNotReflect())
    assert gate.run({"tool": "ph"}, lambda _: (42, VerificationResult(True), "returned"), lambda a, v: []) == 42
    assert not gate.history[0]["reflection_triggered"]


def test_recoverable_failure_reflects_and_logs(tmp_path):
    gate = VerificationGate({"reflection": {"enabled": True}}, tmp_path, reflector=FixtureReflector())
    result = gate.run({"step_id": "a", "tool": "primary", "arguments": {}},
                      lambda a: (42, VerificationResult(True), "returned") if a["tool"] == "backup"
                      else (None, fail("tool_execution_failure", "broken"), "exception"),
                      lambda a, v: [{"id": "fallback", "action": {**a, "tool": "backup"}}])
    assert result == 42
    rows = [json.loads(line) for line in gate.path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["reflection_triggered"] and rows[-1]["final_status"] == "recovered"
    assert rows[0]["task_id"] == rows[1]["task_id"]
    assert rows[0]["action_hash"] != rows[1]["action_hash"]


def test_retry_budget_terminates(tmp_path):
    gate = VerificationGate({"reflection": {"enabled": True, "max_retries": 2}}, tmp_path,
                            reflector=FixtureReflector())
    with pytest.raises(VerifiedExecutionError):
        gate.run({"tool": "primary", "arguments": {"timeout_seconds": 1}},
                 lambda a: (None, fail("tool_execution_failure", "broken"), "exception"),
                 lambda a, v: [{"id": "increase", "action": {**a, "arguments": {
                     "timeout_seconds": a["arguments"]["timeout_seconds"] * 2}}}])
    assert gate.used_retries == 2
    assert len(gate.history) == 3
    assert gate.history[-1]["termination_reason"] == "retry_budget_exhausted"


def test_identical_repair_and_constraint_rewrite_rejected(tmp_path):
    gate = VerificationGate({"reflection": {"enabled": True}}, tmp_path, reflector=FixtureReflector())
    action = {"tool": "primary", "arguments": {}}
    with pytest.raises(VerifiedExecutionError):
        gate.run(action, lambda a: (None, fail("tool_execution_failure", "broken"), "exception"),
                 lambda a, v: [{"id": "same", "action": a}])
    assert not gate.history[0]["reflection_triggered"]


def test_executor_preserves_successful_step_and_repairs_missing_step(tmp_path):
    registry = ToolRegistry()
    registry.register(FixtureTool("ph", "ph_prediction"))
    registry.register(FixtureTool("activity", "activity_ranking"))
    plan = WorkflowPlan((PlanStep("first", "ph"),), ("ph_prediction", "activity_ranking"), ("artifact",))
    result = WorkflowExecutor(registry, reflector=FixtureReflector()).execute(
        plan, config={"reflection": {"enabled": True}}, run_dir=tmp_path,
        initial_artifacts={"artifact": artifact()})
    assert "ph_predictions" in result["artifact"].data
    assert "activity_scores" in result["artifact"].data
    rows = json.loads((tmp_path / "01_execution_trace.json").read_text())
    assert [row["tool"] for row in rows] == ["ph", "activity"]


def test_failed_tool_cannot_mutate_successful_artifact(tmp_path):
    registry = ToolRegistry()
    class Broken(FixtureTool):
        def run(self, context):
            context.artifacts["artifact"].records[0].sequence = "AAAA"
            raise RuntimeError("broken")
    registry.register(FixtureTool("activity", "activity_ranking"))
    registry.register(Broken("ph", "ph_prediction"))
    registry.register(FixtureTool("ph_backup", "ph_prediction"))
    plan = WorkflowPlan((PlanStep("a", "activity"), PlanStep("p", "ph", ("a",))),
                        ("activity_ranking", "ph_prediction"), ("artifact",))
    result = WorkflowExecutor(registry, reflector=FixtureReflector()).execute(
        plan, config={"reflection": {"enabled": True}}, run_dir=tmp_path,
        initial_artifacts={"artifact": artifact()})
    assert result["artifact"].records[0].sequence == "MKW"
    assert result["artifact"].data["activity_scores"] == {"K2R": 1.0}


def test_fixed_prediction_timeout_repair(tmp_path):
    gate = VerificationGate({"reflection": {"enabled": True, "max_timeout_seconds": 20}},
                            tmp_path, reflector=FixtureReflector())
    def call(args, retry):
        if args["timeout_seconds"] < 20:
            raise TimeoutError("expired")
        return {"a": 0.2}
    assert guarded_prediction(gate, name="ephod", step_id="p", arguments={"timeout_seconds": 10},
                              expected_keys=["a"], call=call) == {"a": 0.2}


def test_failure_injection_seed_reproducible():
    first = FailureInjector(42, "task_0", 0.3, "tool_failure")
    second = FailureInjector(42, "task_0", 0.3, "tool_failure")
    assert [first.draw("ph") for _ in range(50)] == [second.draw("ph") for _ in range(50)]


def test_reflection_cannot_choose_unapproved_action(tmp_path):
    class InvalidReflector:
        def reflect(self, payload):
            return {"failure_analysis": "failure", "repair_strategy": "relax constraints",
                    "repair_id": "invented", "constraints": {}}
    gate = VerificationGate({"reflection": {"enabled": True}}, tmp_path,
                            reflector=InvalidReflector(), task={"constraints": {"ph_opt": {"ge": 8}}})
    with pytest.raises(VerifiedExecutionError):
        gate.run({"tool": "primary"}, lambda _: (None, fail("tool_execution_failure", "broken"), "exception"),
                 lambda a, v: [{"id": "switch", "action": {"tool": "backup"}}])
    assert gate.task["constraints"] == {"ph_opt": {"ge": 8}}
    assert gate.history[-1]["final_status"] == "failed"


def test_disabled_feature_uses_original_executor(tmp_path):
    registry = ToolRegistry()
    registry.register(FixtureTool("ph", "ph_prediction"))
    plan = WorkflowPlanner(registry).deterministic(GoalSpec(("ph_prediction",)))
    result = WorkflowExecutor(registry).execute(plan, config={"reflection": {"enabled": False}},
                                               run_dir=tmp_path, initial_artifacts={"artifact": artifact()})
    assert result["artifact"].data["ph_predictions"]
    assert not (tmp_path / "trajectories").exists()


def test_unknown_tool_is_a_logged_failure(tmp_path):
    plan = WorkflowPlan((PlanStep("unknown", "not_installed"),), ("ph_prediction",), ("artifact",))
    with pytest.raises(VerifiedExecutionError) as error:
        WorkflowExecutor(ToolRegistry(), reflector=FixtureReflector()).execute(
            plan, config={"reflection": {"enabled": True}}, run_dir=tmp_path,
            initial_artifacts={"artifact": artifact()})
    assert error.value.verification.failure_type == "invalid_tool_argument"
    assert Path(error.value.trajectory).exists()


def test_constraint_failure_does_not_relax_or_reflect(tmp_path):
    registry = ToolRegistry()
    registry.register(FixtureTool("ph", "ph_prediction"))
    plan = WorkflowPlanner(registry).deterministic(GoalSpec(("ph_prediction",)))
    config = {"reflection": {"enabled": True}, "verification": {"constraints": {"ph_opt": {"ge": 8}}}}
    with pytest.raises(VerifiedExecutionError) as error:
        WorkflowExecutor(registry, reflector=FixtureReflector()).execute(
            plan, config=config, run_dir=tmp_path, initial_artifacts={"artifact": artifact()})
    assert error.value.verification.failure_type == "constraint_violation"
    events = [json.loads(line) for line in Path(error.value.trajectory).read_text(encoding="utf-8").splitlines()]
    assert not any(e["reflection_triggered"] for e in events)


@pytest.mark.parametrize("kind", ["invalid_result", "argument_failure", "missing_step"])
def test_each_injection_type_is_exercised(tmp_path, kind):
    root = tmp_path / kind
    main(["--max-tasks", "1", "--failure-probability", "1", "--failure-type", kind, "--output-dir", str(root)])
    rows = json.loads((root / "summary.json").read_text())
    assert rows[0]["success_rate"] == 0
    if kind in {"argument_failure", "missing_step"}:
        assert rows[1]["success_rate"] == 1
        assert rows[1]["avg_reflections"] == 1


def test_ablation_smoke_and_required_outputs(tmp_path):
    root = tmp_path / "experiment"
    main(["--max-tasks", "2", "--failure-probability", "0", "1", "--output-dir", str(root)])
    rows = json.loads((root / "summary.json").read_text())
    assert len(rows) == 4
    assert rows[0]["success_rate"] == rows[1]["success_rate"] == 1
    assert (root / "summary.csv").is_file()
    assert list((root / "trajectories").glob("*.jsonl"))
