from __future__ import annotations

import math
import numbers
from typing import Any

from ..bioformats import BioArtifact
from ..mutations import apply_mutation, parse_mutation, normalize_sequence
from .schemas import VerificationResult


PREDICTIONS = {
    "mutation_effect_prediction": "activity_scores",
    "activity_ranking": "activity_scores",
    "ph_prediction": "ph_predictions",
    "stability_prediction": "stability_predictions",
    "codon_optimization": "optimized_cds_set",
}


def fail(kind, reason, evidence=None, *, scope="execution_validity"):
    return VerificationResult(False, kind, reason, evidence or {}, False, scope)


def finite_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


class Verifier:
    """Only deterministic evidence is accepted as a verdict."""

    def verify(self, *, task, state, planned_action, tool_call, tool_result,
               biological_constraints, execution_history, tool=None, exception=None,
               phase="result", expected_keys=None, completed_capabilities=None):
        if exception is not None:
            return fail("invalid_tool_argument" if phase == "argument" else "tool_execution_failure",
                        str(exception), {"exception_type": type(exception).__name__,
                                         "tool": planned_action.get("tool")})
        if phase == "plan":
            missing = sorted(set(task.get("required_capabilities", [])) -
                             set(completed_capabilities or []))
            if missing:
                return fail("missing_required_step", "Required capabilities are absent",
                            {"missing_capabilities": missing})
            return VerificationResult(True)
        if phase == "argument":
            try:
                if tool is not None:
                    tool.validate_input(tool_call)
                artifact = state.get("artifact")
                if isinstance(artifact, BioArtifact):
                    ids = [record.id for record in artifact.records]
                    if len(ids) != len(set(ids)):
                        return fail("invalid_tool_argument", "Duplicate record IDs", {"ids": ids})
                    if not artifact.records:
                        return fail("invalid_tool_argument", "No sequence records")
                    proteins = [r for r in artifact.records if r.molecule_type == "protein"]
                    if tool and "protein" in tool.spec.properties.get("molecule_types", []):
                        if len(proteins) != len(artifact.records):
                            return fail("invalid_tool_argument", "Tool requires protein records")
                    reference_id = artifact.data.get("reference_id")
                    if reference_id and reference_id not in ids:
                        return fail("invalid_tool_argument", "Unknown reference_id", {"reference_id": reference_id})
                    reference = next((r for r in artifact.records if r.id == reference_id or
                                      r.annotations.get("role") == "reference"), artifact.records[0])
                    for record in proteins:
                        normalize_sequence(record.sequence)
                        if tool and set(tool.spec.capabilities) & {
                                "activity_ranking", "mutation_effect_prediction", "stability_prediction"}:
                            if len(record.sequence) != len(reference.sequence):
                                return fail("invalid_tool_argument", "Substitution models require equal sequence lengths",
                                            {"record_id": record.id})
                        mutation = record.annotations.get("mutation")
                        if mutation and record is not reference:
                            positions = set()
                            sequence = reference.sequence
                            for component in mutation.split(":"):
                                _, position, _ = parse_mutation(component)
                                if position in positions:
                                    raise ValueError("Duplicate residue index in mutation")
                                positions.add(position)
                                sequence = apply_mutation(sequence, component)
                            if sequence != record.sequence:
                                return fail("inconsistent_result", "Mutation label disagrees with sequence",
                                            {"record_id": record.id, "mutation": mutation})
                arguments = planned_action.get("arguments", {})
                if set(arguments) - {"timeout_seconds"}:
                    return fail("invalid_tool_argument", "Unsupported tool argument", {"keys": list(arguments)})
                if "timeout_seconds" in arguments:
                    value = arguments["timeout_seconds"]
                    if type(value) is not int or value <= 0:
                        return fail("invalid_tool_argument", "timeout_seconds must be a positive integer")
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                return fail("invalid_tool_argument", str(exc), {"exception_type": type(exc).__name__})
            return VerificationResult(True)
        if phase == "objective":
            return self.constraints(tool_result, biological_constraints)
        if expected_keys is not None:
            return self.prediction_map(tool_result, expected_keys)
        try:
            if not isinstance(tool_result, dict) or not tool_result:
                return fail("tool_execution_failure", "Empty or invalid result")
            if tool:
                tool.validate_output(tool_result)
            artifact = tool_result.get("artifact")
            if isinstance(artifact, BioArtifact) and tool:
                original = state.get("artifact")
                before = {r.id: r.sequence for r in original.records} if isinstance(original, BioArtifact) else {}
                after = {r.id: r.sequence for r in artifact.records}
                if any(after.get(key) != value for key, value in before.items()):
                    return fail("inconsistent_result", "Prediction tool changed or removed input sequences")
                for capability in tool.spec.capabilities:
                    field = PREDICTIONS.get(capability)
                    if not field:
                        continue
                    if field not in artifact.data or not artifact.data[field]:
                        return fail("tool_execution_failure", "Missing required prediction field", {"field": field})
                    if field == "optimized_cds_set":
                        if not isinstance(artifact.data[field], list):
                            return fail("tool_execution_failure", "Invalid CDS result type", {"field": field})
                        continue
                    records = original.records if isinstance(original, BioArtifact) else artifact.records
                    reference_id = original.data.get("reference_id") if isinstance(original, BioArtifact) else None
                    reference = next((r for r in records if r.id == reference_id or
                                      r.annotations.get("role") == "reference"), records[0])
                    keys = []
                    for record in records:
                        if field != "ph_predictions" and record.id == reference.id:
                            continue
                        if field == "activity_scores":
                            mutation = ":".join(f"{a}{i}{b}" for i, (a, b) in
                                                enumerate(zip(reference.sequence, record.sequence), 1) if a != b)
                            keys.append(record.annotations.get("mutation") or mutation)
                        else:
                            keys.append(record.id)
                    verdict = self.prediction_map(artifact.data[field], keys)
                    if not verdict.passed:
                        return verdict
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            return fail("tool_execution_failure", str(exc), {"exception_type": type(exc).__name__})
        return VerificationResult(True, evidence={"scientific_validation": "NOT_IMPLEMENTED",
                                                  "prediction_accuracy": "NOT_IMPLEMENTED"})

    @staticmethod
    def prediction_map(value, expected_keys):
        if not isinstance(value, dict) or not value:
            return fail("tool_execution_failure", "Empty or invalid prediction map")
        missing = sorted(set(expected_keys) - set(value))
        invalid = [key for key, score in value.items() if not finite_number(score)]
        if missing or invalid:
            return fail("tool_execution_failure", "Missing or non-finite predictions",
                        {"missing_keys": missing, "invalid_keys": invalid})
        return VerificationResult(True)

    @staticmethod
    def constraints(rows, constraints):
        if not constraints:
            return VerificationResult(True, scope="scientific_objective_satisfaction",
                                      evidence={"constraints": "NOT_REQUESTED"})
        if not rows:
            return fail("constraint_violation", "No candidate satisfies the requested selection",
                        {"constraints": constraints}, scope="scientific_objective_satisfaction")
        violations = []
        for index, row in enumerate(rows):
            for field, rules in constraints.items():
                value = row.get(field)
                if not finite_number(value):
                    violations.append({"row": index, "field": field, "value": str(value), "rule": "finite_required"})
                    continue
                for operator, threshold in rules.items():
                    comparisons = {"gt": value > threshold, "ge": value >= threshold,
                                   "lt": value < threshold, "le": value <= threshold,
                                   "eq": value == threshold}
                    if operator not in comparisons:
                        raise ValueError(f"Unsupported constraint operator: {operator}")
                    if not comparisons[operator]:
                        violations.append({"row": index, "field": field, "value": value,
                                           "operator": operator, "threshold": threshold})
        if violations:
            return fail("constraint_violation", "Explicit task constraints are not satisfied",
                        {"violations": violations}, scope="scientific_objective_satisfaction")
        return VerificationResult(True, scope="scientific_objective_satisfaction",
                                  evidence={"basis": "computational_predictions_only"})


def task_constraints(config):
    """Only explicit configured thresholds; no scientific defaults."""
    constraints = dict(config.get("verification", {}).get("constraints", {}))
    selection = config.get("selection", {})
    target, tolerance = selection.get("target_ph"), selection.get("ph_tolerance")
    if target is not None and tolerance is not None:
        constraints.setdefault("ph_opt", {"ge": target - tolerance, "le": target + tolerance})
    if selection.get("min_activity_proxy") is not None:
        constraints.setdefault("activity_proxy", {"ge": selection["min_activity_proxy"]})
    op = selection.get("stability_operator")
    if op in {"gt", "ge", "lt", "le", "eq"} and selection.get("stability_threshold") is not None:
        constraints.setdefault("ddg", {op: selection["stability_threshold"]})
    return constraints
