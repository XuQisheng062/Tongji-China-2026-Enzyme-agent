import pytest
import csv

from fixed_enzyme_agent.mutations import (
    apply_mutation,
    generate_single_substitutions,
    load_single_substitution_fasta,
)
from fixed_enzyme_agent.scoring import rank_candidates
from fixed_enzyme_agent.types import Candidate
from fixed_enzyme_agent.adapters.enzgfm import EnzGFMAdapter


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


def test_ph_remains_discriminative_outside_tolerance():
    selection = {
        "target_ph": 7.0,
        "ph_tolerance": 0.5,
        "min_activity_proxy": None,
        "stability_direction": "lower_is_better",
        "stability_operator": "none",
        "stability_threshold": None,
        "weights": {"ph": 0.0, "activity": 0.5, "stability": 0.5},
        "minimum_ph_weight": 0.4,
    }
    closer = Candidate("A1C", "AC", "CC", 0.5, 8.0, 1.0, 0.0)
    farther = Candidate("A1D", "AC", "DC", 0.5, 10.0, 3.0, 0.0)
    ranked = rank_candidates([farther, closer], selection)
    assert closer.ph_component > farther.ph_component > 0.0
    assert ranked[0].mutation == "A1C"


def test_load_custom_single_substitution_fasta(tmp_path):
    fasta = tmp_path / "library.fasta"
    fasta.write_text(">first\nCCD\n>second\nAED\n", encoding="utf-8")
    library = load_single_substitution_fasta(fasta, "ACD")
    assert library == {"A1C": "CCD", "C2E": "AED"}


def test_custom_fasta_supports_multiple_substitutions(tmp_path):
    fasta = tmp_path / "multiple.fasta"
    fasta.write_text(">double\nCEE\n", encoding="utf-8")
    assert load_single_substitution_fasta(fasta, "ACD") == {"A1C:C2E:D3E": "CEE"}


def test_custom_fasta_rejects_wt_record(tmp_path):
    fasta = tmp_path / "invalid.fasta"
    fasta.write_text(">wt\nACD\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identical to the WT"):
        load_single_substitution_fasta(fasta, "ACD")


def test_enzgfm_adds_component_scores_for_multi_site_mutants(tmp_path, monkeypatch):
    adapter = EnzGFMAdapter(str(tmp_path), "python", str(tmp_path / "model"))

    def fake_run_checked(command, **kwargs):
        input_csv = tmp_path / "work" / "input" / "mutations.csv"
        output_csv = tmp_path / "work" / "output" / "processed_mutations.csv"
        with input_csv.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        assert [row["mutant"] for row in rows] == ["A1C", "C2E"]
        with output_csv.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=["mutant", "pred_0"])
            writer.writeheader()
            writer.writerow({"mutant": "A1C", "pred_0": 1.25})
            writer.writerow({"mutant": "C2E", "pred_0": 0.75})

    monkeypatch.setattr("fixed_enzyme_agent.adapters.enzgfm.run_checked", fake_run_checked)
    scores = adapter.score_mutations(
        "ACD",
        ["A1C:C2E"],
        work_dir=tmp_path / "work",
        env={},
        timeout=10,
        force_cpu=True,
    )
    assert scores == {"A1C:C2E": 2.0}
