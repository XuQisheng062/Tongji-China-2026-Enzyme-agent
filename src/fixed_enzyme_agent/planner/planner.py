from __future__ import annotations

import json
from typing import Any, Protocol

from .models import GoalSpec, PlanStep, WorkflowPlan
from .validator import PlanValidationError, validate_plan
from ..tools import ToolRegistry


class JsonPlannerClient(Protocol):
    def json_object(self, **kwargs: Any) -> dict[str, Any]: ...


class WorkflowPlanner:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def deterministic(self, goal: GoalSpec) -> WorkflowPlan:
        artifacts = set(goal.initial_artifacts)
        remaining = list(dict.fromkeys(goal.capabilities))
        steps: list[PlanStep] = []
        used: set[str] = set()
        while remaining:
            progress = False
            for capability in list(remaining):
                candidates = [
                    tool for tool in self.registry.providers(capability)
                    if tool.spec.name not in used and set(tool.spec.inputs) <= artifacts
                ]
                if not candidates:
                    continue
                tool = candidates[0]
                step_id = f"step_{len(steps) + 1}_{tool.spec.name}"
                steps.append(PlanStep(step_id, tool.spec.name, tuple(step.id for step in steps)))
                used.add(tool.spec.name)
                artifacts.update(tool.spec.outputs)
                remaining.remove(capability)
                progress = True
            if not progress:
                unavailable = {
                    capability: [tool.spec.name for tool in self.registry.providers(capability, available_only=False)]
                    for capability in remaining
                }
                raise PlanValidationError(f"Cannot satisfy requested capabilities: {unavailable}")
        plan = WorkflowPlan(tuple(steps), goal.capabilities, goal.initial_artifacts, "Rule-based provider selection")
        validate_plan(plan, self.registry)
        return plan

    def with_llm(self, goal: GoalSpec, *, client: JsonPlannerClient, model: str) -> WorkflowPlan:
        system_prompt = (
            "You plan a typed scientific workflow. Return JSON only. Select only listed tool names. "
            "Never invent tools or executable code. Steps must be topologically ordered. Each step has "
            "id, tool, and depends_on. Preserve requested_capabilities and initial_artifacts."
        )
        raw = client.json_object(
            model=model,
            system_prompt=system_prompt,
            user_prompt=json.dumps(
                {"goal": {"capabilities": goal.capabilities, "initial_artifacts": goal.initial_artifacts,
                          "constraints": goal.constraints}, "tool_catalog": self.registry.catalog()},
                ensure_ascii=False,
            ),
        )
        raw["requested_capabilities"] = list(goal.capabilities)
        raw["initial_artifacts"] = list(goal.initial_artifacts)
        plan = WorkflowPlan.from_dict(raw)
        validate_plan(plan, self.registry)
        return plan
