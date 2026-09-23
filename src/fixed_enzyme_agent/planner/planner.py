from __future__ import annotations

import json
from typing import Any, Protocol

from ..bioformats import SCHEMA_VERSION
from .models import GoalSpec, PlanStep, TaskRoute, WorkflowPlan
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

    def route_request(
        self,
        user_request: str,
        *,
        client: JsonPlannerClient,
        model: str,
        input_summary: dict[str, Any],
        output_formats: list[str],
    ) -> TaskRoute:
        if not user_request.strip():
            raise ValueError("user_request must not be empty")
        system_prompt = """
You are the routing planner for a synthetic-biology tool agent. Read the complete
installed tool catalog and the user request. Return one JSON object only.

Rules:
1. The executable plan may use only exact installed tool names marked available.
2. Every executable model tool consumes and produces the same bioartifact/v1 artifact.
3. Never generate code, shell commands, biological predictions, or invented tool IDs.
4. Put useful installed tools that the user did not request in available_additions;
   do not silently execute them.
5. Put tools not present in the catalog in external_suggestions with a clear user_action.
6. Treat SBOL as a data standard. Treat iGEM/BioBricks RFCs as design/assembly
   standard metadata, not as a single parseable file format.
7. Preserve the requested output formats in format_plan.output_formats.

Required JSON structure:
{
  "interpretation": "short explanation in the user's language",
  "format_plan": {
    "input_format": "detected format",
    "canonical_contract": "bioartifact/v1",
    "output_formats": ["json"],
    "standards": ["SBOL", "RFC10"]
  },
  "plan": {
    "requested_capabilities": ["capability"],
    "initial_artifacts": ["artifact"],
    "rationale": "why this order",
    "steps": [{"id": "step_1", "tool": "exact_installed_name", "depends_on": []}]
  },
  "available_additions": [
    {"tool": "exact_installed_name", "reason": "why useful", "insertion_after": "step id or input"}
  ],
  "external_suggestions": [
    {"name": "tool name", "capability": "capability", "reason": "why useful", "user_action": "what user must do"}
  ],
  "guidance": ["one practical idea"]
}
""".strip()
        raw = client.json_object(
            model=model,
            system_prompt=system_prompt,
            user_prompt=json.dumps(
                {
                    "user_request": user_request,
                    "input_summary": input_summary,
                    "requested_output_formats": output_formats,
                    "tool_catalog": self.registry.catalog(),
                },
                ensure_ascii=False,
            ),
        )
        plan_raw = raw.get("plan")
        if not isinstance(plan_raw, dict):
            raise ValueError("LLM route must contain a plan object")
        plan_raw["initial_artifacts"] = ["artifact"]
        plan = WorkflowPlan.from_dict(plan_raw)
        for step in plan.steps:
            spec = self.registry.get(step.tool).spec
            if (spec.artifact_contract != SCHEMA_VERSION or
                    spec.inputs != ("artifact",) or spec.outputs != ("artifact",)):
                raise PlanValidationError(
                    f"Routed tool {spec.name!r} does not satisfy the "
                    f"{SCHEMA_VERSION} input/output contract"
                )
        validate_plan(plan, self.registry)

        additions = raw.get("available_additions", [])
        if not isinstance(additions, list):
            raise ValueError("available_additions must be an array")
        planned = {step.tool for step in plan.steps}
        for addition in additions:
            if not isinstance(addition, dict):
                raise ValueError("Each available addition must be an object")
            tool_name = str(addition.get("tool", ""))
            if tool_name in planned:
                raise ValueError(f"Suggested installed tool is already planned: {tool_name}")
            tool = self.registry.get(tool_name)
            health = tool.healthcheck()
            if not health.available:
                raise ValueError(f"Suggested installed tool is unavailable: {tool_name}")

        external = raw.get("external_suggestions", [])
        if not isinstance(external, list):
            raise ValueError("external_suggestions must be an array")
        installed_names = {tool.spec.name for tool in self.registry.all()}
        for suggestion in external:
            if not isinstance(suggestion, dict):
                raise ValueError("Each external suggestion must be an object")
            if str(suggestion.get("name", "")) in installed_names:
                raise ValueError(
                    f"External suggestion is already installed: {suggestion.get('name')}"
                )
        format_plan = dict(raw.get("format_plan", {}))
        format_plan["canonical_contract"] = "bioartifact/v1"
        format_plan["output_formats"] = list(output_formats)
        return TaskRoute(
            plan=plan,
            interpretation=str(raw.get("interpretation", "")),
            format_plan=format_plan,
            available_additions=tuple(dict(item) for item in additions),
            external_suggestions=tuple(dict(item) for item in external),
            guidance=tuple(map(str, raw.get("guidance", []))),
        )
