from pathlib import Path

import pytest

from fixed_enzyme_agent.executor import WorkflowExecutor
from fixed_enzyme_agent.planner import GoalSpec, PlanValidationError, WorkflowPlanner
from fixed_enzyme_agent.tools import AgentTool, ToolContext, ToolRegistry, ToolSpec
from fixed_enzyme_agent.tools.builtin import CodonOptimizationTool


class ExampleTool(AgentTool):
    spec = ToolSpec("example", "1.0", "Example provider", ("score",), ("protein_sequence",), ("scores",))

    def run(self, context: ToolContext):
        return {"scores": [len(context.artifacts["protein_sequence"])]}


def test_register_plan_execute_and_remove(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(ExampleTool())
    plan = WorkflowPlanner(registry).deterministic(GoalSpec(("score",)))
    output = WorkflowExecutor(registry).execute(
        plan, config={}, run_dir=tmp_path, initial_artifacts={"protein_sequence": "ACD"}
    )
    assert output["scores"] == [3]
    assert (tmp_path / "00_workflow_plan.json").is_file()
    registry.unregister("example")
    with pytest.raises(PlanValidationError):
        WorkflowPlanner(registry).deterministic(GoalSpec(("score",)))


def test_duplicate_registration_is_rejected():
    registry = ToolRegistry()
    registry.register(ExampleTool())
    with pytest.raises(ValueError):
        registry.register(ExampleTool())


def test_codon_optimizer_is_selected_and_executed(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(CodonOptimizationTool())
    goal = GoalSpec(
        ("codon_optimization",),
        ("protein_sequence", "selected_candidates", "host_organisms"),
    )
    plan = WorkflowPlanner(registry).deterministic(goal)
    assert [step.tool for step in plan.steps] == ["codon_optimizer"]
    artifacts = WorkflowExecutor(registry).execute(
        plan,
        config={"sequence_name": "demo", "runtime": {"cache_dir": str(tmp_path / "cache")}},
        run_dir=tmp_path / "run",
        initial_artifacts={
            "protein_sequence": "MKW",
            "selected_candidates": [{"mutation": "K2R", "mutant_sequence": "MRW"}],
            "host_organisms": ["Escherichia coli", "Bacillus subtilis"],
        },
    )
    assert len(artifacts["optimized_cds_set"]) == 4
    assert artifacts["codon_optimization_result"]["host_count"] == 2


def test_codon_optimizer_is_not_available_after_removal():
    registry = ToolRegistry()
    registry.register(CodonOptimizationTool())
    registry.unregister("codon_optimizer")
    with pytest.raises(PlanValidationError):
        WorkflowPlanner(registry).deterministic(GoalSpec(
            ("codon_optimization",),
            ("protein_sequence", "selected_candidates", "host_organisms"),
        ))
