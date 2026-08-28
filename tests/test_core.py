from fixed_enzyme_agent.mutations import generate_single_substitutions, apply_mutation
from fixed_enzyme_agent.scoring import rank_candidates
from fixed_enzyme_agent.types import Candidate


def test_mutation_generation_and_apply():
    seq = "ACD"
    muts = generate_single_substitutions(seq)
    assert len(muts) == 57
    assert "A1C" in muts
    assert apply_mutation(seq, "A1C") == "CCD"


def test_rank():
    seq = "ACD"
    a = Candidate("A1C", seq, "CCD", 1.0, 7.1, 0.1, -0.3)
    b = Candidate("A1D", seq, "DCD", -0.2, 8.8, 1.8, 1.0)
    out = rank_candidates([a, b], {
        "target_ph": 7.0,
        "ph_tolerance": 1.0,
        "min_activity_proxy": None,
        "stability_direction": "lower_is_better",
        "stability_operator": "none",
        "stability_threshold": None,
        "weights": {"ph": 0.3, "activity": 0.4, "stability": 0.3},
    })
    assert out[0].mutation == "A1C"
    assert out[0].passed is True
    assert out[1].passed is False
