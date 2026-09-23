from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "bioartifact/v1"
DNA_ALPHABET = set("ACGTUN")


@dataclass
class BioRecord:
    id: str
    sequence: str
    molecule_type: str
    description: str = ""
    annotations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sequence = re.sub(r"\s+", "", self.sequence).upper()
        if not self.id or not self.sequence:
            raise ValueError("BioRecord id and sequence must be non-empty")
        if self.molecule_type not in {"dna", "rna", "protein"}:
            raise ValueError("molecule_type must be dna, rna, or protein")


@dataclass
class BioArtifact:
    records: list[BioRecord] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    standards: list[str] = field(default_factory=list)
    source_format: str = "memory"
    schema: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact schema: {self.schema}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_format": self.source_format,
            "standards": self.standards,
            "metadata": self.metadata,
            "records": [asdict(record) for record in self.records],
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BioArtifact":
        if value.get("schema", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact schema: {value.get('schema')}")
        records = []
        for index, raw_record in enumerate(value.get("records", []), start=1):
            record = dict(raw_record)
            sequence = str(record.get("sequence", ""))
            record.setdefault("id", record.pop("name", f"record_{index}"))
            record.setdefault("molecule_type", infer_molecule_type(sequence))
            records.append(BioRecord(**record))
        return cls(
            records=records,
            data=dict(value.get("data", {})),
            metadata=dict(value.get("metadata", {})),
            standards=list(value.get("standards", [])),
            source_format=str(value.get("source_format", "json")),
        )

    def clone(self) -> "BioArtifact":
        return BioArtifact.from_dict(self.to_dict())


def infer_molecule_type(sequence: str) -> str:
    letters = set(re.sub(r"\s+", "", sequence).upper())
    return "dna" if letters and letters <= DNA_ALPHABET else "protein"


class BioFormatConverter:
    EXTENSIONS = {
        ".fa": "fasta", ".faa": "fasta", ".fna": "fasta", ".fasta": "fasta",
        ".gb": "genbank", ".gbk": "genbank", ".genbank": "genbank",
        ".csv": "csv", ".json": "json", ".jsonld": "sbol-json",
        ".xml": "sbol", ".rdf": "sbol", ".sbol": "sbol",
    }

    def detect(self, path: str | Path, format_name: str | None = None) -> str:
        if format_name:
            return format_name.lower()
        suffix = Path(path).suffix.lower()
        if suffix not in self.EXTENSIONS:
            raise ValueError(f"Cannot detect biological format from extension: {suffix}")
        return self.EXTENSIONS[suffix]

    def load(self, path: str | Path, *, format_name: str | None = None,
             standards: list[str] | None = None) -> BioArtifact:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Input file does not exist: {source}")
        fmt = self.detect(source, format_name)
        parser = getattr(self, f"_load_{fmt.replace('-', '_')}", None)
        if parser is None:
            raise ValueError(f"Unsupported input format: {fmt}")
        artifact = parser(source)
        artifact.source_format = fmt
        artifact.metadata.setdefault("source_path", str(source))
        artifact.standards = list(dict.fromkeys([*artifact.standards, *(standards or [])]))
        return artifact

    def dump(self, artifact: BioArtifact, path: str | Path, *, format_name: str | None = None) -> Path:
        target = Path(path).expanduser().resolve()
        fmt = self.detect(target, format_name)
        writer = getattr(self, f"_dump_{fmt.replace('-', '_')}", None)
        if writer is None:
            raise ValueError(f"Unsupported output format: {fmt}")
        target.parent.mkdir(parents=True, exist_ok=True)
        writer(artifact, target)
        return target

    def dump_many(self, artifact: BioArtifact, output_dir: str | Path,
                  formats: list[str], stem: str = "result") -> dict[str, str]:
        extensions = {"fasta": ".fasta", "genbank": ".gb", "csv": ".csv",
                      "json": ".json", "sbol": ".xml", "sbol-json": ".jsonld"}
        root = Path(output_dir).expanduser().resolve()
        outputs = {}
        for fmt in dict.fromkeys(formats):
            if fmt not in extensions:
                raise ValueError(f"Unsupported output format: {fmt}")
            path = self.dump(artifact, root / f"{stem}{extensions[fmt]}", format_name=fmt)
            outputs[fmt] = str(path)
        return outputs

    def _load_fasta(self, path: Path) -> BioArtifact:
        records = []
        header = None
        chunks: list[str] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(self._fasta_record(header, "".join(chunks)))
                header, chunks = line[1:].strip(), []
            elif header is None:
                raise ValueError("FASTA sequence appears before its header")
            else:
                chunks.append(line)
        if header is not None:
            records.append(self._fasta_record(header, "".join(chunks)))
        if not records:
            raise ValueError("FASTA contains no records")
        return BioArtifact(records=records)

    @staticmethod
    def _fasta_record(header: str, sequence: str) -> BioRecord:
        parts = header.split(maxsplit=1)
        return BioRecord(parts[0] or "sequence", sequence, infer_molecule_type(sequence),
                         parts[1] if len(parts) > 1 else "")

    def _dump_fasta(self, artifact: BioArtifact, path: Path) -> None:
        lines = []
        for record in artifact.records:
            description = f" {record.description}" if record.description else ""
            lines.append(f">{record.id}{description}")
            lines.extend(record.sequence[index:index + 80] for index in range(0, len(record.sequence), 80))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _load_json(self, path: Path) -> BioArtifact:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, list):
            value = {"records": value}
        if not isinstance(value, dict):
            raise ValueError("JSON biological input must be an object or record array")
        if "records" not in value and all(isinstance(item, str) for item in value.values()):
            value = {"records": [{"id": key, "sequence": sequence}
                                 for key, sequence in value.items()]}
        return BioArtifact.from_dict(value)

    def _dump_json(self, artifact: BioArtifact, path: Path) -> None:
        path.write_text(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_csv(self, path: Path) -> BioArtifact:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "sequence" not in rows[0]:
            raise ValueError("CSV biological input requires a sequence column")
        records = []
        for index, row in enumerate(rows, start=1):
            sequence = row.pop("sequence")
            record_id = row.pop("id", None) or row.pop("name", None) or f"record_{index}"
            molecule = row.pop("molecule_type", None) or infer_molecule_type(sequence)
            description = row.pop("description", "")
            annotations_json = row.pop("annotations_json", "")
            annotations = json.loads(annotations_json) if annotations_json else {}
            annotations.update({key: value for key, value in row.items() if value not in (None, "")})
            records.append(BioRecord(record_id, sequence, molecule, description, annotations))
        return BioArtifact(records=records)

    def _dump_csv(self, artifact: BioArtifact, path: Path) -> None:
        fields = ["id", "sequence", "molecule_type", "description", "mutation",
                  "activity_proxy", "ph_opt", "ddg", "annotations_json"]
        activity = artifact.data.get("activity_scores", {})
        ph_values = artifact.data.get("ph_predictions", {})
        stability = artifact.data.get("stability_predictions", {})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in artifact.records:
                mutation = record.annotations.get("mutation", "")
                score_key = mutation or record.id
                writer.writerow({"id": record.id, "sequence": record.sequence,
                                 "molecule_type": record.molecule_type,
                                 "description": record.description,
                                 "mutation": mutation,
                                 "activity_proxy": activity.get(score_key, ""),
                                 "ph_opt": ph_values.get(record.id, ""),
                                 "ddg": stability.get(record.id, ""),
                                 "annotations_json": json.dumps(record.annotations, ensure_ascii=False)})

    def _load_genbank(self, path: Path) -> BioArtifact:
        records = []
        for block in path.read_text(encoding="utf-8-sig").split("//"):
            if "ORIGIN" not in block:
                continue
            locus = re.search(r"^LOCUS\s+(\S+).*?(\baa\b|\bbp\b)", block, re.MULTILINE)
            definition = re.search(r"^DEFINITION\s+(.+?)(?=\n\S)", block, re.MULTILINE | re.DOTALL)
            origin = block.split("ORIGIN", 1)[1]
            sequence = "".join(re.findall(r"[A-Za-z]+", origin))
            molecule = "protein" if locus and locus.group(2) == "aa" else "dna"
            record_id = locus.group(1) if locus else f"record_{len(records) + 1}"
            description = " ".join(definition.group(1).split()) if definition else ""
            records.append(BioRecord(record_id, sequence, molecule, description))
        if not records:
            raise ValueError("GenBank input contains no ORIGIN records")
        return BioArtifact(records=records)

    def _dump_genbank(self, artifact: BioArtifact, path: Path) -> None:
        blocks = []
        for record in artifact.records:
            unit = "aa" if record.molecule_type == "protein" else "bp"
            lines = [f"LOCUS       {record.id[:16]:<16} {len(record.sequence):>7} {unit}",
                     f"DEFINITION  {record.description or record.id}.", "ORIGIN"]
            sequence = record.sequence.lower()
            for index in range(0, len(sequence), 60):
                chunk = sequence[index:index + 60]
                grouped = " ".join(chunk[offset:offset + 10] for offset in range(0, len(chunk), 10))
                lines.append(f"{index + 1:>9} {grouped}")
            lines.append("//")
            blocks.append("\n".join(lines))
        path.write_text("\n".join(blocks) + "\n", encoding="utf-8")

    def _load_sbol(self, path: Path) -> BioArtifact:
        root = ET.parse(path).getroot()
        records = []
        for element in root.iter():
            if element.tag.split("}")[-1] != "Sequence":
                continue
            children = {child.tag.split("}")[-1]: (child.text or "").strip() for child in element}
            sequence = children.get("elements")
            if not sequence:
                continue
            record_id = children.get("displayId") or element.attrib.get("about", "sequence").rsplit("/", 1)[-1]
            encoding = children.get("encoding", "")
            molecule = "protein" if "protein" in encoding.lower() else infer_molecule_type(sequence)
            records.append(BioRecord(record_id, sequence, molecule, children.get("description", "")))
        if not records:
            raise ValueError("SBOL input contains no readable Sequence elements")
        return BioArtifact(records=records, standards=["SBOL"])

    def _dump_sbol(self, artifact: BioArtifact, path: Path) -> None:
        rdf = ET.Element("RDF", {"xmlns:sbol": "http://sbols.org/v2#"})
        for record in artifact.records:
            sequence = ET.SubElement(rdf, "sbol:Sequence")
            ET.SubElement(sequence, "sbol:displayId").text = record.id
            ET.SubElement(sequence, "sbol:elements").text = record.sequence
            encoding = "http://www.chem.qmul.ac.uk/iubmb/misc/naseq.html"
            if record.molecule_type == "protein":
                encoding = "http://www.chem.qmul.ac.uk/iupac/AminoAcid/"
            ET.SubElement(sequence, "sbol:encoding").text = encoding
        ET.ElementTree(rdf).write(path, encoding="utf-8", xml_declaration=True)

    def _load_sbol_json(self, path: Path) -> BioArtifact:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        nodes = value.get("@graph", []) if isinstance(value, dict) else value
        records = []
        for index, node in enumerate(nodes or [], start=1):
            types = node.get("@type", [])
            types = [types] if isinstance(types, str) else types
            if not any("Sequence" in item for item in types):
                continue
            sequence = node.get("elements", "")
            if sequence:
                records.append(BioRecord(node.get("displayId", f"sequence_{index}"), sequence,
                                         infer_molecule_type(sequence)))
        if not records:
            raise ValueError("SBOL JSON-LD input contains no readable Sequence objects")
        return BioArtifact(records=records, standards=["SBOL"])

    def _dump_sbol_json(self, artifact: BioArtifact, path: Path) -> None:
        graph = [{"@type": "sbol:Sequence", "displayId": record.id,
                  "elements": record.sequence, "moleculeType": record.molecule_type}
                 for record in artifact.records]
        path.write_text(json.dumps({"@context": {"sbol": "http://sbols.org/v2#"},
                                         "@graph": graph}, indent=2), encoding="utf-8")
