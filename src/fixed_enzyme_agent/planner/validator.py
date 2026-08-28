from __future__ import annotations

from .models import WorkflowPlan
from ..tools import ToolRegistry


class PlanValidationError(ValueError):
    pass


def validate_plan(plan: WorkflowPlan, registry: ToolRegistry) -> None:
    step_ids = [step.id for step in plan.steps]
    if len(step_ids) != len(set(step_ids)):
        raise PlanValidationError("Plan step IDs must be unique")
    known_steps: set[str] = set()
    artifacts = set(plan.initial_artifacts)
    capabilities: set[str] = set()
    for step in plan.steps:
        unknown_dependencies = set(step.depends_on) - known_steps
        if unknown_dependencies:
            raise PlanValidationError(
                f"Step {step.id!r} has missing or forward dependencies: {sorted(unknown_dependencies)}"
            )
        try:
            tool = registry.get(step.tool)
        except KeyError as exc:
            raise PlanValidationError(str(exc)) from exc
        health = tool.healthcheck()
        if not health.available:
            raise PlanValidationError(f"Tool {step.tool!r} is unavailable: {list(health.details)}")
        missing_inputs = set(tool.spec.inputs) - artifacts
        if missing_inputs:
            raise PlanValidationError(f"Step {step.id!r} is missing artifacts: {sorted(missing_inputs)}")
        artifacts.update(tool.spec.outputs)
        capabilities.update(tool.spec.capabilities)
        known_steps.add(step.id)
    missing_capabilities = set(plan.requested_capabilities) - capabilities
    if missing_capabilities:
        raise PlanValidationError(f"Plan does not satisfy capabilities: {sorted(missing_capabilities)}")
