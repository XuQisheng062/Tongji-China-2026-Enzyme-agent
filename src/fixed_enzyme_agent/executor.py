from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bioformats import BioArtifact, BioFormatConverter
from .planner import WorkflowPlan, validate_plan
from .tools import ToolContext, ToolRegistry


class WorkflowExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, plan: WorkflowPlan, *, config: dict[str, Any], run_dir: Path,
                initial_artifacts: dict[str, Any]) -> dict[str, Any]:
        validate_plan(plan, self.registry)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "00_workflow_plan.json").write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        artifacts = dict(initial_artifacts)
        trace: list[dict[str, Any]] = []
        for step in plan.steps:
            tool = self.registry.get(step.tool)
            context = ToolContext(run_dir, config, artifacts, step.id)
            tool.validate_input(context)
            result = dict(tool.run(context))
            tool.validate_output(result)
            artifacts.update(result)
            trace.append({"step": step.id, "tool": step.tool, "outputs": list(result)})
        (run_dir / "01_execution_trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return artifacts

    def execute_and_export(
        self,
        plan: WorkflowPlan,
        *,
        config: dict[str, Any],
        run_dir: Path,
        input_artifact: BioArtifact,
        output_formats: list[str],
    ) -> dict[str, Any]:
        artifacts = self.execute(
            plan,
            config=config,
            run_dir=run_dir,
            initial_artifacts={"artifact": input_artifact},
        )
        result = artifacts.get("artifact")
        if not isinstance(result, BioArtifact):
            raise TypeError("Workflow did not return a bioartifact/v1 artifact")
        outputs = BioFormatConverter().dump_many(
            result, run_dir / "exports", output_formats, stem="result"
        )
        return {"artifact": result, "outputs": outputs}
