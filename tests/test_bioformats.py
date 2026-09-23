from pathlib import Path

from fixed_enzyme_agent.bioformats import BioArtifact, BioFormatConverter, BioRecord, SCHEMA_VERSION


def _artifact():
    return BioArtifact(
        records=[
            BioRecord("WT", "ACDE", "protein", "reference", {"role": "reference"}),
            BioRecord("A1C", "CCDE", "protein", annotations={"mutation": "A1C"}),
        ],
        standards=["SBOL", "RFC10"],
        data={"target_ph": 8.0},
    )


def test_fasta_to_canonical_json_and_csv(tmp_path: Path):
    source = tmp_path / "input.fasta"
    source.write_text(">WT reference\nACDE\n>A1C\nCCDE\n", encoding="utf-8")
    converter = BioFormatConverter()
    artifact = converter.load(source, standards=["RFC10"])
    assert artifact.schema == SCHEMA_VERSION
    assert len(artifact.records) == 2
    assert artifact.standards == ["RFC10"]
    json_path = converter.dump(artifact, tmp_path / "output.json")
    csv_path = converter.dump(artifact, tmp_path / "output.csv")
    assert converter.load(json_path).records[1].sequence == "CCDE"
    assert converter.load(csv_path).records[0].id == "WT"


def test_genbank_and_sbol_round_trip(tmp_path: Path):
    converter = BioFormatConverter()
    artifact = _artifact()
    genbank = converter.dump(artifact, tmp_path / "output.gb")
    sbol = converter.dump(artifact, tmp_path / "output.xml", format_name="sbol")
    assert [record.sequence for record in converter.load(genbank).records] == ["ACDE", "CCDE"]
    loaded_sbol = converter.load(sbol, format_name="sbol")
    assert [record.id for record in loaded_sbol.records] == ["WT", "A1C"]
    assert "SBOL" in loaded_sbol.standards
