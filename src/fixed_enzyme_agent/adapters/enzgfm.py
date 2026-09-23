from __future__ import annotations

import csv
from pathlib import Path

from ..runner import run_checked


class EnzGFMAdapter:
    def __init__(self, repo: str, python: str, model_location: str):
        self.repo = Path(repo)
        self.python = python
        self.model_location = Path(model_location)

    def _entry(self) -> Path:
        # README has used predict_mutations.py, while the current repository contains predict.py.
        candidates = [
            self.repo / "point_mutation" / "predict.py",
            self.repo / "point_mutation" / "predict_mutations.py",
        ]
        for p in candidates:
            if p.is_file():
                return p
        return candidates[0]

    def preflight(self) -> list[str]:
        errors = []
        if not self.repo.is_dir():
            errors.append(f"EnzGFM repo \u4e0d\u5b58\u5728: {self.repo}")
        if not self._entry().is_file():
            errors.append(f"EnzGFM point_mutation \u5165\u53e3\u4e0d\u5b58\u5728: {self._entry()}")
        if not (self.repo / "EsmTokenizer").is_dir():
            errors.append(f"EnzGFM EsmTokenizer \u4e0d\u5b58\u5728: {self.repo / 'EsmTokenizer'}")
        if not self.model_location.exists():
            errors.append(f"EnzGFM \u6a21\u578b\u76ee\u5f55\u4e0d\u5b58\u5728: {self.model_location}")
        return errors

    def score_mutations(
        self,
        wt_sequence: str,
        mutations: list[str],
        *,
        work_dir: Path,
        env: dict[str, str],
        timeout: int,
        force_cpu: bool,
    ) -> dict[str, float]:
        components_by_mutation = {
            mutation: [part.strip() for part in mutation.split(":") if part.strip()]
            for mutation in mutations
        }
        component_mutations = list(dict.fromkeys(
            component
            for components in components_by_mutation.values()
            for component in components
        ))
        work_dir.mkdir(parents=True, exist_ok=True)
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        in_csv = input_dir / "mutations.csv"
        with in_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Sequence", "mutant"])
            writer.writeheader()
            for m in component_mutations:
                # Upstream point-mutation code groups on Sequence and scores each mutation on the WT logits.
                writer.writerow({"Sequence": wt_sequence, "mutant": m})

        cmd = [
            self.python,
            str(self._entry()),
            "--model-location", str(self.model_location),
            "--sequence", wt_sequence,
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--mutation-col", "mutant",
            "--scoring-strategy", "wt-marginals",
        ]
        # CUDA_VISIBLE_DEVICES is also cleared by the caller. --nogpu is added for semantic clarity.
        if force_cpu:
            cmd.append("--nogpu")
        # point_mutation/predict.py imports the repository-level `models` package.
        # Add the EnzGFM repo root to PYTHONPATH for subprocess execution.
        child_env = dict(env)
        old_pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = str(self.repo) + ((":" + old_pythonpath) if old_pythonpath else "")
        run_checked(cmd, cwd=self.repo, env=child_env, timeout=timeout, log_file=work_dir / "enzgfm.log")

        out_csv = output_dir / "processed_mutations.csv"
        if not out_csv.is_file():
            processed = sorted(output_dir.glob("processed_*.csv"))
            if len(processed) != 1:
                raise RuntimeError(f"\u65e0\u6cd5\u5b9a\u4f4d EnzGFM \u8f93\u51fa\u6587\u4ef6: {processed}")
            out_csv = processed[0]

        component_scores: dict[str, float] = {}
        with out_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            pred_cols = sorted([c for c in fields if c.startswith("pred_")])
            if not pred_cols:
                raise RuntimeError(f"EnzGFM \u8f93\u51fa\u7f3a\u5c11 pred_* \u5217: {fields}")
            for row in reader:
                vals = [float(row[c]) for c in pred_cols if row.get(c) not in (None, "")]
                if not vals:
                    continue
                component_scores[row["mutant"]] = sum(vals) / len(vals)
        missing = [m for m in component_mutations if m not in component_scores]
        if missing:
            raise RuntimeError(f"EnzGFM \u672a\u8fd4\u56de {len(missing)} \u4e2a\u7a81\u53d8\u5206\u6570\uff1b\u793a\u4f8b: {missing[:10]}")
        return {
            mutation: sum(component_scores[component] for component in components)
            for mutation, components in components_by_mutation.items()
        }
