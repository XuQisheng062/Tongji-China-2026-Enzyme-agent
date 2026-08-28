from __future__ import annotations

import csv
from pathlib import Path

from ..runner import run_checked


class EpHodAdapter:
    def __init__(self, repo: str, python: str):
        self.repo = Path(repo)
        self.python = python

    @property
    def entry(self) -> Path:
        return self.repo / "ephod" / "run.py"

    def preflight(self) -> list[str]:
        errors = []
        if not self.repo.is_dir():
            errors.append(f"EpHod repo \u4e0d\u5b58\u5728: {self.repo}")
        if not self.entry.is_file():
            errors.append(f"EpHod \u5165\u53e3\u4e0d\u5b58\u5728: {self.entry}")
        svr = self.repo / "ephod" / "data" / "ESM1v-SVR.pkl"
        if not svr.is_file():
            errors.append(f"EpHod SVR \u6570\u636e\u6587\u4ef6\u4e0d\u5b58\u5728: {svr}")
        return errors

    def predict(
        self,
        sequences: dict[str, str],
        *,
        work_dir: Path,
        env: dict[str, str],
        timeout: int,
    ) -> dict[str, float]:
        work_dir.mkdir(parents=True, exist_ok=True)
        fasta = work_dir / "ephod_input.fasta"
        with fasta.open("w", encoding="utf-8") as f:
            for name, seq in sequences.items():
                f.write(f">{name}\n{seq}\n")
        out_name = "prediction.csv"
        cmd = [
            self.python,
            str(self.entry),
            "--fasta_path", str(fasta),
            "--save_dir", str(work_dir),
            "--csv_name", out_name,
            "--verbose", "1",
            "--save_attention_weights", "0",
            "--save_embeddings", "0",
        ]
        run_checked(cmd, cwd=self.repo, env=env, timeout=timeout, log_file=work_dir / "ephod.log")
        out_csv = work_dir / out_name
        if not out_csv.is_file():
            raise RuntimeError(f"EpHod \u672a\u751f\u6210\u7ed3\u679c\u6587\u4ef6: {out_csv}")
        result: dict[str, float] = {}
        with out_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if "Ensemble" not in (reader.fieldnames or []):
                raise RuntimeError(f"EpHod \u8f93\u51fa\u7f3a\u5c11 Ensemble \u5217: {reader.fieldnames}")
            first_col = (reader.fieldnames or [""])[0]
            for row in reader:
                name = row.get(first_col, "")
                result[name] = float(row["Ensemble"])
        missing = [k for k in sequences if k not in result]
        if missing:
            raise RuntimeError(f"EpHod \u8f93\u51fa\u7f3a\u5c11\u5e8f\u5217: {missing[:10]}")
        return result
