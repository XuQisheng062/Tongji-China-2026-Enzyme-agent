from __future__ import annotations

import math
from .types import Candidate


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rank_candidates(candidates: list[Candidate], selection: dict) -> list[Candidate]:
    if not candidates:
        return []
    for c in candidates:
        if c.activity_proxy is None or c.ph_opt is None or c.ddg is None:
            raise ValueError(f"\u5019\u9009 {c.mutation} \u7684\u6027\u8d28\u9884\u6d4b\u4e0d\u5b8c\u6574")
        if not all(math.isfinite(float(v)) for v in [c.activity_proxy, c.ph_opt, c.ddg]):
            raise ValueError(f"\u5019\u9009 {c.mutation} \u542b NaN/Inf")

    activity_norm = _minmax([float(c.activity_proxy) for c in candidates])
    stability_raw = [float(c.ddg) for c in candidates]
    stability_norm = _minmax(stability_raw)
    if selection["stability_direction"] == "lower_is_better":
        stability_norm = [1.0 - x for x in stability_norm]

    target_ph = selection.get("target_ph")
    ph_tol = selection.get("ph_tolerance")
    weights = selection["weights"]
    minimum_ph_weight = float(selection.get("minimum_ph_weight", 0.4))
    if target_ph is not None and float(weights["ph"]) < minimum_ph_weight:
        remaining = float(weights["activity"]) + float(weights["stability"])
        weights = dict(weights)
        weights["ph"] = minimum_ph_weight
        if remaining > 0:
            scale_other = (1.0 - minimum_ph_weight) / remaining
            weights["activity"] = float(weights["activity"]) * scale_other
            weights["stability"] = float(weights["stability"]) * scale_other
    denom = float(weights["ph"]) + float(weights["activity"]) + float(weights["stability"])

    for i, c in enumerate(candidates):
        c.activity_component = activity_norm[i]
        c.stability_component = stability_norm[i]

        if target_ph is None:
            c.ph_component = 0.0
        else:
            distance = abs(float(c.ph_opt) - float(target_ph))
            scale = float(ph_tol) if ph_tol is not None else max(1.0, float(target_ph))
            # A smooth reciprocal penalty preserves pH discrimination even
            # outside the hard tolerance instead of clipping every value to zero.
            c.ph_component = 1.0 / (1.0 + (distance / scale) ** 2)

        c.passed = True
        c.reasons = []
        if target_ph is not None and ph_tol is not None:
            if abs(float(c.ph_opt) - float(target_ph)) > float(ph_tol):
                c.passed = False
                c.reasons.append("pH \u8d85\u51fa\u5bb9\u5dee")
        min_activity = selection.get("min_activity_proxy")
        if min_activity is not None and float(c.activity_proxy) < float(min_activity):
            c.passed = False
            c.reasons.append("EnzGFM \u6d3b\u6027\u4ee3\u7406\u5206\u6570\u4f4e\u4e8e\u9608\u503c")
        op = selection.get("stability_operator", "none")
        threshold = selection.get("stability_threshold")
        if op == "le" and float(c.ddg) > float(threshold):
            c.passed = False
            c.reasons.append("UniStab \u0394\u0394G \u9ad8\u4e8e\u9608\u503c")
        elif op == "ge" and float(c.ddg) < float(threshold):
            c.passed = False
            c.reasons.append("UniStab \u0394\u0394G \u4f4e\u4e8e\u9608\u503c")

        c.final_score = (
            float(weights["ph"]) * float(c.ph_component)
            + float(weights["activity"]) * float(c.activity_component)
            + float(weights["stability"]) * float(c.stability_component)
        ) / denom

    return sorted(candidates, key=lambda c: (not c.passed, -(c.final_score or 0.0)))
