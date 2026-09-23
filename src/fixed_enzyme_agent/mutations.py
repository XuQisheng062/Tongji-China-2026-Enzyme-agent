from __future__ import annotations

from pathlib import Path

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def normalize_sequence(sequence: str) -> str:
    seq = "".join(sequence.split()).upper()
    if not seq:
        raise ValueError("sequence \u4e0d\u80fd\u4e3a\u7a7a")
    bad = sorted(set(seq) - set(AMINO_ACIDS))
    if bad:
        raise ValueError(f"\u53ea\u652f\u6301 20 \u79cd\u6807\u51c6\u6c28\u57fa\u9178\uff1b\u53d1\u73b0\u975e\u6cd5\u5b57\u7b26: {bad}")
    # EpHod warns/truncates above 1022, and EnzGFM point-mutation code skips >1024.
    if len(seq) > 1022:
        raise ValueError("\u5f53\u524d\u56fa\u5b9a\u94fe\u8def\u8981\u6c42\u5e8f\u5217\u957f\u5ea6 <= 1022 aa\uff0c\u4ee5\u540c\u65f6\u6ee1\u8db3 EpHod/EnzGFM \u63a8\u7406\u9650\u5236")
    return seq


def generate_single_substitutions(
    sequence: str,
    allowed_positions: list[int] | None = None,
    excluded_positions: list[int] | None = None,
) -> list[str]:
    seq = normalize_sequence(sequence)
    allowed = set(allowed_positions) if allowed_positions else None
    excluded = set(excluded_positions or [])
    n = len(seq)
    for collection, label in [(allowed or set(), "allowed_positions"), (excluded, "excluded_positions")]:
        bad = sorted(p for p in collection if p < 1 or p > n)
        if bad:
            raise ValueError(f"{label} \u542b\u8d8a\u754c\u4f4d\u7f6e: {bad}")

    out: list[str] = []
    for i, wt in enumerate(seq, start=1):
        if allowed is not None and i not in allowed:
            continue
        if i in excluded:
            continue
        for mt in AMINO_ACIDS:
            if mt != wt:
                out.append(f"{wt}{i}{mt}")
    return out


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    if len(mutation) < 3:
        raise ValueError(f"\u975e\u6cd5\u7a81\u53d8\u683c\u5f0f: {mutation}")
    wt = mutation[0].upper()
    mt = mutation[-1].upper()
    if wt not in AMINO_ACIDS or mt not in AMINO_ACIDS:
        raise ValueError(f"\u975e\u6cd5\u6c28\u57fa\u9178\u7a81\u53d8: {mutation}")
    try:
        pos = int(mutation[1:-1])
    except ValueError as exc:
        raise ValueError(f"\u975e\u6cd5\u7a81\u53d8\u4f4d\u7f6e: {mutation}") from exc
    return wt, pos, mt


def apply_mutation(sequence: str, mutation: str) -> str:
    seq = normalize_sequence(sequence)
    wt, pos, mt = parse_mutation(mutation)
    if not 1 <= pos <= len(seq):
        raise ValueError(f"\u7a81\u53d8\u4f4d\u7f6e\u8d8a\u754c: {mutation}")
    if seq[pos - 1] != wt:
        raise ValueError(f"\u7a81\u53d8 {mutation} \u7684 WT \u4e0e\u8f93\u5165\u5e8f\u5217\u4e0d\u4e00\u81f4\uff1a\u5e8f\u5217\u4f4d\u7f6e {pos} \u662f {seq[pos-1]}")
    return seq[: pos - 1] + mt + seq[pos:]


def load_single_substitution_fasta(
    path: str | Path,
    wt_sequence: str,
    *,
    allowed_positions: list[int] | None = None,
    excluded_positions: list[int] | None = None,
) -> dict[str, str]:
    """Load an equal-length FASTA library containing substitution mutants."""
    fasta_path = Path(path).expanduser().resolve()
    if not fasta_path.is_file():
        raise ValueError(f"Mutation FASTA does not exist: {fasta_path}")
    wt = normalize_sequence(wt_sequence)
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw_line in fasta_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].strip() or f"record_{len(records) + 1}"
            chunks = []
        else:
            if header is None:
                raise ValueError("Mutation FASTA sequence appears before its header")
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    if not records:
        raise ValueError(f"Mutation FASTA contains no records: {fasta_path}")

    allowed = set(allowed_positions) if allowed_positions else None
    excluded = set(excluded_positions or [])
    library: dict[str, str] = {}
    for record_header, raw_sequence in records:
        sequence = normalize_sequence(raw_sequence)
        if len(sequence) != len(wt):
            raise ValueError(
                f"Mutation FASTA record {record_header!r} has length {len(sequence)}; "
                f"expected {len(wt)}"
            )
        differences = [index for index, pair in enumerate(zip(wt, sequence), start=1) if pair[0] != pair[1]]
        if not differences:
            raise ValueError(
                f"Mutation FASTA record {record_header!r} is identical to the WT sequence"
            )
        if allowed is not None and any(position not in allowed for position in differences):
            continue
        if any(position in excluded for position in differences):
            continue
        mutation = ":".join(
            f"{wt[position - 1]}{position}{sequence[position - 1]}"
            for position in differences
        )
        library[mutation] = sequence
    if not library:
        raise ValueError("Mutation FASTA has no candidates after position filtering")
    return library
