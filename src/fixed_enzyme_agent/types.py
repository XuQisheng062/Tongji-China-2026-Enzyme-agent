from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Candidate:
    mutation: str
    wt_sequence: str
    mutant_sequence: Optional[str] = None

    # External model predictions.
    activity_proxy: Optional[float] = None
    ph_opt: Optional[float] = None
    delta_ph_from_wt: Optional[float] = None
    ddg: Optional[float] = None

    # Deterministic scoring components.
    activity_component: Optional[float] = None
    ph_component: Optional[float] = None
    stability_component: Optional[float] = None
    final_score: Optional[float] = None
    passed: bool = True
    reasons: list[str] = field(default_factory=list)

    # Deterministic point-mutation descriptors.
    position: Optional[int] = None
    wt_aa: Optional[str] = None
    mutant_aa: Optional[str] = None
    wt_residue_class: Optional[str] = None
    mutant_residue_class: Optional[str] = None
    residue_class_change: Optional[str] = None
    wt_hydrophobicity: Optional[float] = None
    mutant_hydrophobicity: Optional[float] = None
    delta_hydrophobicity: Optional[float] = None
    wt_volume: Optional[float] = None
    mutant_volume: Optional[float] = None
    delta_volume: Optional[float] = None
    wt_charge: Optional[float] = None
    mutant_charge: Optional[float] = None
    delta_charge: Optional[float] = None
    blosum62: Optional[int] = None
    conservative_substitution: Optional[bool] = None
    pareto_optimal: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
