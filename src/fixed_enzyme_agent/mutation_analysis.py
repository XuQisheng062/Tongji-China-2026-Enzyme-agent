from __future__ import annotations

import csv
import re
from pathlib import Path

from .types import Candidate

# Kyte-Doolittle hydropathy index.
_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Approximate residue volumes (\u00c5^3). These are used only as a deterministic
# comparative descriptor between WT and mutant residues.
_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Simplified nominal side-chain charge class near neutral pH.
_CHARGE = {aa: 0.0 for aa in _HYDROPHOBICITY}
_CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0})

_RESIDUE_CLASS = {
    **{aa: "nonpolar" for aa in "AVLIMGPFW"},
    **{aa: "polar_uncharged" for aa in "STCNQY"},
    **{aa: "positive" for aa in "KRH"},
    **{aa: "negative" for aa in "DE"},
}

_BLOSUM_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_ROWS = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]
_BLOSUM62 = {
    (a, b): _BLOSUM62_ROWS[i][j]
    for i, a in enumerate(_BLOSUM_ORDER)
    for j, b in enumerate(_BLOSUM_ORDER)
}

_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_point_mutation(mutation: str) -> tuple[str, int, str]:
    match = _MUTATION_RE.fullmatch(str(mutation).strip().upper())
    if not match:
        raise ValueError(f"\u4e0d\u652f\u6301\u7684\u70b9\u7a81\u53d8\u683c\u5f0f: {mutation!r}; \u671f\u671b\u4f8b\u5982 A23V")
    wt, pos, mutant = match.groups()
    if wt not in _HYDROPHOBICITY or mutant not in _HYDROPHOBICITY:
        raise ValueError(f"\u70b9\u7a81\u53d8\u5305\u542b\u975e\u6807\u51c6\u6c28\u57fa\u9178: {mutation!r}")
    return wt, int(pos), mutant


def annotate_mutation_features(candidate: Candidate) -> Candidate:
    wt, position, mutant = parse_point_mutation(candidate.mutation)
    if position < 1 or position > len(candidate.wt_sequence):
        raise ValueError(f"\u7a81\u53d8\u4f4d\u7f6e\u8d85\u51fa\u5e8f\u5217\u957f\u5ea6: {candidate.mutation}")
    actual_wt = candidate.wt_sequence[position - 1]
    if actual_wt != wt:
        raise ValueError(
            f"\u7a81\u53d8 {candidate.mutation} \u7684 WT \u6b8b\u57fa\u4e0e\u8f93\u5165\u5e8f\u5217\u4e0d\u4e00\u81f4: sequence={actual_wt}"
        )

    candidate.position = position
    candidate.wt_aa = wt
    candidate.mutant_aa = mutant
    candidate.wt_residue_class = _RESIDUE_CLASS[wt]
    candidate.mutant_residue_class = _RESIDUE_CLASS[mutant]
    candidate.residue_class_change = (
        f"{candidate.wt_residue_class}->{candidate.mutant_residue_class}"
    )

    candidate.wt_hydrophobicity = _HYDROPHOBICITY[wt]
    candidate.mutant_hydrophobicity = _HYDROPHOBICITY[mutant]
    candidate.delta_hydrophobicity = (
        candidate.mutant_hydrophobicity - candidate.wt_hydrophobicity
    )

    candidate.wt_volume = _VOLUME[wt]
    candidate.mutant_volume = _VOLUME[mutant]
    candidate.delta_volume = candidate.mutant_volume - candidate.wt_volume

    candidate.wt_charge = _CHARGE[wt]
    candidate.mutant_charge = _CHARGE[mutant]
    candidate.delta_charge = candidate.mutant_charge - candidate.wt_charge

    candidate.blosum62 = int(_BLOSUM62[(wt, mutant)])
    candidate.conservative_substitution = candidate.blosum62 >= 0
    return candidate


def annotate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    for candidate in candidates:
        # Aggregate substitutions keep their full mutation label, while the
        # single-residue descriptor fields remain empty because one position
        # cannot represent a multi-site mutant.
        if ":" not in candidate.mutation:
            annotate_mutation_features(candidate)
    return candidates


def mark_pareto_optimal(candidates: list[Candidate]) -> list[Candidate]:
    """Mark the 3-objective Pareto front using normalized higher-is-better components.

    Hard-filter failures are excluded from the Pareto front when at least one
    candidate passes. If none pass, all candidates are considered so the plot
    remains informative for debugging.
    """
    for candidate in candidates:
        candidate.pareto_optimal = False

    eligible = [c for c in candidates if c.passed]
    if not eligible:
        eligible = list(candidates)

    def vector(c: Candidate) -> tuple[float, float, float]:
        values = (c.activity_component, c.ph_component, c.stability_component)
        if any(v is None for v in values):
            raise ValueError(f"\u5019\u9009 {c.mutation} \u7f3a\u5c11 Pareto \u6240\u9700\u8bc4\u5206\u5206\u91cf")
        return tuple(float(v) for v in values)  # type: ignore[return-value]

    vectors = {id(c): vector(c) for c in eligible}
    for candidate in eligible:
        current = vectors[id(candidate)]
        dominated = False
        for other in eligible:
            if other is candidate:
                continue
            ov = vectors[id(other)]
            if all(b >= a for a, b in zip(current, ov)) and any(
                b > a for a, b in zip(current, ov)
            ):
                dominated = True
                break
        candidate.pareto_optimal = not dominated
    return candidates


def write_mutation_features_csv(path: Path, candidates: list[Candidate]) -> Path:
    fields = [
        "mutation",
        "position",
        "wt_aa",
        "mutant_aa",
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
        "blosum62",
        "conservative_substitution",
        "pareto_optimal",
        "final_score",
        "passed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            data = candidate.to_dict()
            writer.writerow({name: data.get(name) for name in fields})
    return path
