#!/usr/bin/env python3
from fixed_enzyme_agent.mutations import generate_single_substitutions, apply_mutation
from fixed_enzyme_agent.scoring import rank_candidates
from fixed_enzyme_agent.types import Candidate

seq = "ACDEFGHIK"
mutations = generate_single_substitutions(seq)
assert len(mutations) == len(seq) * 19
m = mutations[0]
mut = apply_mutation(seq, m)
assert mut != seq and len(mut) == len(seq)

cs = [
    Candidate("A1C", seq, apply_mutation(seq, "A1C"), 1.0, 7.0, 0.0, -0.5),
    Candidate("A1D", seq, apply_mutation(seq, "A1D"), 0.2, 8.5, 1.5, 0.8),
]
ranked = rank_candidates(cs, {
    "target_ph": 7.0,
    "ph_tolerance": 1.0,
    "min_activity_proxy": None,
    "stability_direction": "lower_is_better",
    "stability_operator": "none",
    "stability_threshold": None,
    "weights": {"ph": 0.3, "activity": 0.4, "stability": 0.3},
})
assert ranked[0].mutation == "A1C"
print("[PASS] \u6838\u5fc3\u4ee3\u7801\u81ea\u68c0\u901a\u8fc7\uff08\u4e0d\u52a0\u8f7d\u4e09\u4e2a\u6a21\u578b\uff09")
