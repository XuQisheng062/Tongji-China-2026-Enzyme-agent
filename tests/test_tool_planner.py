from pathlib import Path

import pytest

from fixed_enzyme_agent.bioformats import BioArtifact, BioRecord, SCHEMA_VERSION
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
    plan = WorkflowPlanner(registry).deterministic(
        GoalSpec(("score",), ("protein_sequence",))
    )
    output = WorkflowExecutor(registry).execute(
        plan, config={}, run_dir=tmp_path, initial_artifacts={"protein_sequence": "ACD"}
    )
    assert output["scores"] == [3]
    assert (tmp_path / "00_workflow_plan.json").is_file()
    registry.unregister("example")
    with pytest.raises(PlanValidationError):
        WorkflowPlanner(registry).deterministic(
            GoalSpec(("score",), ("protein_sequence",))
        )


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
        ("artifact",),
    )
    plan = WorkflowPlanner(registry).deterministic(goal)
    assert [step.tool for step in plan.steps] == ["codon_optimizer"]
    artifacts = WorkflowExecutor(registry).execute(
        plan,
        config={"sequence_name": "demo", "runtime": {"cache_dir": str(tmp_path / "cache")}},
        run_dir=tmp_path / "run",
        initial_artifacts={"artifact": BioArtifact(
            records=[
                BioRecord("WT", "MKW", "protein", annotations={"role": "reference"}),
                BioRecord("K2R", "MRW", "protein", annotations={"mutation": "K2R"}),
            ],
            data={"host_organisms": ["Escherichia coli", "Bacillus subtilis"]},
        )},
    )
    result = artifacts["artifact"]
    assert isinstance(result, BioArtifact)
    assert len(result.data["optimized_cds_set"]) == 4
    assert result.data["codon_optimization_result"]["host_count"] == 2


def test_codon_optimizer_is_not_available_after_removal():
    registry = ToolRegistry()
    registry.register(CodonOptimizationTool())
    registry.unregister("codon_optimizer")
    with pytest.raises(PlanValidationError):
        WorkflowPlanner(registry).deterministic(GoalSpec(
            ("codon_optimization",),
            ("artifact",),
        ))


class CanonicalTool(AgentTool):
    def __init__(self, name: str, capability: str):
        self.spec = ToolSpec(
            name, "1.0", name, (capability,), ("artifact",), ("artifact",),
            artifact_contract=SCHEMA_VERSION,
        )

    def run(self, context: ToolContext):
        return {"artifact": context.artifacts["artifact"].clone()}


class FakeRouteClient:
    def json_object(self, **kwargs):
        return {
            "interpretation": "Predict pH first.",
            "format_plan": {"input_format": "fasta", "standards": ["SBOL"]},
            "plan": {
                "requested_capabilities": ["ph_prediction"],
                "initial_artifacts": ["anything"],
                "rationale": "The request prioritizes pH.",
                "steps": [{"id": "step_1", "tool": "ph_tool", "depends_on": []}],
            },
            "available_additions": [
                {"tool": "structure_tool", "reason": "Inspect structure", "insertion_after": "step_1"}
            ],
            "external_suggestions": [
                {"name": "ExternalDB", "capability": "database_search", "reason": "Find homologs",
                 "user_action": "Run it separately and import its FASTA output"}
            ],
            "guidance": ["Validate the final candidates experimentally."],
        }


def test_llm_route_reads_catalog_and_separates_suggestions():
    registry = ToolRegistry()
    registry.register(CanonicalTool("ph_tool", "ph_prediction"))
    registry.register(CanonicalTool("structure_tool", "structure_prediction"))
    route = WorkflowPlanner(registry).route_request(
        "Find variants near pH 8",
        client=FakeRouteClient(),
        model="test-model",
        input_summary={"source_format": "fasta"},
        output_formats=["json", "sbol"],
    )
    assert route.plan.initial_artifacts == ("artifact",)
    assert route.plan.steps[0].tool == "ph_tool"
    assert route.available_additions[0]["tool"] == "structure_tool"
    assert route.external_suggestions[0]["name"] == "ExternalDB"
    assert route.format_plan["output_formats"] == ["json", "sbol"]


def test_routed_tool_must_use_canonical_contract():
    registry = ToolRegistry()
    registry.register(ExampleTool())

    class InvalidRouteClient:
        def json_object(self, **kwargs):
            return {
                "plan": {
                    "requested_capabilities": ["score"],
                    "steps": [{"id": "step_1", "tool": "example", "depends_on": []}],
                }
            }

    with pytest.raises(PlanValidationError, match="bioartifact/v1"):
        WorkflowPlanner(registry).route_request(
            "Score the sequence",
            client=InvalidRouteClient(),
            model="test-model",
            input_summary={"source_format": "fasta"},
            output_formats=["json"],
        )


def test_canonical_contract_rejects_plain_mapping(tmp_path: Path):
    tool = CanonicalTool("canonical", "normalize")
    context = ToolContext(tmp_path, {}, {"artifact": {"records": []}}, "step_1")
    with pytest.raises(TypeError, match="bioartifact/v1"):
        tool.validate_input(context)
