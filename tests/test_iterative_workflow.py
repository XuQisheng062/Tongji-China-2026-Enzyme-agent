from pathlib import Path

from fixed_enzyme_agent.tools import ToolRegistry
from fixed_enzyme_agent.workflow import FixedEnzymeWorkflow


class FakeEnzGFM:
    def __init__(self):
        self.mutations = []

    def score_mutations(self, sequence, mutations, **kwargs):
        self.mutations = list(mutations)
        return {mutation: float(index) for index, mutation in enumerate(mutations)}


class FakeEpHod:
    def predict(self, sequences, **kwargs):
        return {name: 7.0 + index * 0.01 for index, name in enumerate(sequences)}


class FakeUniStab:
    def __init__(self):
        self.calls = 0
        self.records = []
        self.batch_size = None

    def predict_ddg_batch(self, records, **kwargs):
        self.calls += 1
        self.records = records
        self.batch_size = kwargs.get("batch_size")
        return {record["name"]: -float(index) for index, record in enumerate(records)}


def test_round_output_preserves_lineage_and_hook_position(tmp_path: Path):
    workflow = FixedEnzymeWorkflow.__new__(FixedEnzymeWorkflow)
    workflow.force_cpu = True
    workflow.enzgfm = FakeEnzGFM()
    workflow.ephod = FakeEpHod()
    workflow.unistab = FakeUniStab()
    workflow.tool_registry = ToolRegistry()
    workflow.tool_registry.discover([Path("plugins")])
    cfg = {
        "candidate": {"top_k": 3, "allowed_positions": None, "excluded_positions": []},
        "workflow": {"hooks": {"after_enzgfm": ["sequence_length"]}},
    }
    candidates, wt_ph, trace, _ = workflow._evaluate_parent_before_unistab(
        cfg=cfg,
        parent_id="A1C",
        parent_sequence="CCD",
        round_number=2,
        work_dir=tmp_path,
        env={},
        timeout=10,
    )
    assert len(candidates) == 3
    assert all(candidate.round_number == 2 for candidate in candidates)
    assert all(candidate.lineage.startswith("A1C+") for candidate in candidates)
    assert wt_ph == 7.0
    assert trace[0]["point"] == "after_enzgfm"
    assert trace[0]["tool"] == "sequence_length"


def test_three_parents_use_one_unistab_call(tmp_path: Path):
    from fixed_enzyme_agent.types import Candidate

    workflow = FixedEnzymeWorkflow.__new__(FixedEnzymeWorkflow)
    workflow.force_cpu = False
    workflow.unistab = FakeUniStab()
    candidates = [
        Candidate(
            mutation="A1C",
            wt_sequence=f"ACD{index}",
            mutant_sequence=f"CCD{index}",
            lineage=f"parent_{index}+A1C",
        )
        for index in range(3)
    ]
    result = workflow._predict_round_stability(
        cfg={"models": {"unistab": {"batch_size": 1}}},
        candidates=candidates,
        round_dir=tmp_path,
        env={},
        timeout=10,
    )
    assert workflow.unistab.calls == 1
    assert workflow.unistab.records
    assert workflow.unistab.batch_size == 1
    assert len(workflow.unistab.records) == 3
    assert len({record["parent_sequence"] for record in workflow.unistab.records}) == 3
    assert len(result) == 3
    assert all(candidate.ddg is not None for candidate in candidates)


def test_round_one_uses_custom_fasta_library(tmp_path: Path):
    fasta = tmp_path / "library.fasta"
    fasta.write_text(">one\nCCD\n>two\nAED\n", encoding="utf-8")
    workflow = FixedEnzymeWorkflow.__new__(FixedEnzymeWorkflow)
    workflow.force_cpu = True
    workflow.enzgfm = FakeEnzGFM()
    workflow.ephod = FakeEpHod()
    workflow.tool_registry = ToolRegistry()
    cfg = {
        "candidate": {
            "top_k": 10,
            "allowed_positions": None,
            "excluded_positions": [],
            "mutation_fasta": str(fasta),
        },
        "workflow": {"hooks": {}},
    }
    candidates, _, _, _ = workflow._evaluate_parent_before_unistab(
        cfg=cfg,
        parent_id="WT",
        parent_sequence="ACD",
        round_number=1,
        work_dir=tmp_path / "round1",
        env={},
        timeout=10,
    )
    assert workflow.enzgfm.mutations == ["A1C", "C2E"]
    assert {candidate.mutant_sequence for candidate in candidates} == {"CCD", "AED"}

    workflow._evaluate_parent_before_unistab(
        cfg=cfg,
        parent_id="A1C",
        parent_sequence="CCD",
        round_number=2,
        work_dir=tmp_path / "round2",
        env={},
        timeout=10,
    )
    assert len(workflow.enzgfm.mutations) == 57
