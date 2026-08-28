from __future__ import annotations

import csv
import json
from pathlib import Path

from .types import Candidate


CANDIDATE_CSV_FIELDS = [
    "mutation",
    "position",
    "wt_aa",
    "mutant_aa",
    "activity_proxy",
    "ph_opt",
    "delta_ph_from_wt",
    "ddg",
    "activity_component",
    "ph_component",
    "stability_component",
    "final_score",
    "passed",
    "reasons",
    "blosum62",
    "conservative_substitution",
    "pareto_optimal",
    "wt_residue_class",
    "mutant_residue_class",
    "residue_class_change",
    "wt_hydrophobicity",
    "mutant_hydrophobicity",
    "delta_hydrophobicity",
    "wt_volume",
    "mutant_volume",
    "delta_volume",
    "wt_charge",
    "mutant_charge",
    "delta_charge",
    "mutant_sequence",
]


def write_candidates_csv(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_CSV_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            data = candidate.to_dict()
            data["reasons"] = "; ".join(candidate.reasons)
            writer.writerow({key: data.get(key) for key in CANDIDATE_CSV_FIELDS})


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
