from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from ..runner import run_checked


class UniStabAdapter:
    def __init__(self, repo: str, python: str, checkpoint: str):
        self.repo = Path(repo)
        self.python = python
        self.checkpoint = Path(checkpoint)

    @property
    def entry(self) -> Path:
        return self.repo / "src" / "inference.py"

    def preflight(self) -> list[str]:
        errors = []
        if not self.repo.is_dir():
            errors.append(f"UniStab repo \u4e0d\u5b58\u5728: {self.repo}")
        if not self.entry.is_file():
            errors.append(f"UniStab \u63a8\u7406\u5165\u53e3\u4e0d\u5b58\u5728: {self.entry}")
        if not self.checkpoint.is_file():
            errors.append(f"UniStab checkpoint \u4e0d\u5b58\u5728: {self.checkpoint}")
        return errors

    def predict_ddg(
        self,
        wt_sequence: str,
        mutants: dict[str, str],
        *,
        work_dir: Path,
        env: dict[str, str],
        timeout: int,
        force_cpu: bool,
    ) -> dict[str, float]:
        work_dir.mkdir(parents=True, exist_ok=True)
        in_csv = work_dir / "unistab_input.csv"
        names = list(mutants)
        with in_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["WT_name", "ddG_ML", "mut_seq", "wt_seq"])
            writer.writeheader()
            for name in names:
                # Upstream UniversalMutationDataset requires a finite ddG_ML target even for inference.
                # We use a dummy 0.0 target and only consume predictions.
                writer.writerow({
                    "WT_name": name,
                    "ddG_ML": 0.0,
                    "mut_seq": mutants[name],
                    "wt_seq": wt_sequence,
                })
        out_dir = work_dir / "output"
        device = "cpu" if force_cpu else "cuda:0"
        run_name = "agent"
        cmd = [
            self.python,
            str(self.entry),
            "--checkpoint", str(self.checkpoint),
            "--data", str(in_csv),
            "--output", str(out_dir),
            "--name", run_name,
            "--batch_size", "1",
            "--device", device,
        ]
        # UniStab inference imports modules from both repo root and src/.
        child_env = dict(env)
        old_pythonpath = child_env.get("PYTHONPATH", "")
        required = f"{self.repo}:{self.repo / 'src'}"
        child_env["PYTHONPATH"] = required + ((":" + old_pythonpath) if old_pythonpath else "")
        run_checked(cmd, cwd=self.repo, env=child_env, timeout=timeout, log_file=work_dir / "unistab.log")
        npz = out_dir / f"{run_name}_results.npz"
        if not npz.is_file():
            raise RuntimeError(f"UniStab \u672a\u751f\u6210\u7ed3\u679c: {npz}")
        with np.load(npz, allow_pickle=False) as z:
            preds = np.asarray(z["predictions"], dtype=float).reshape(-1)
        if len(preds) != len(names):
            raise RuntimeError(
                f"UniStab \u8fd4\u56de\u6570\u91cf {len(preds)} \u4e0e\u8f93\u5165 {len(names)} \u4e0d\u4e00\u81f4\u3002"
                "\u8bf7\u67e5\u770b unistab.log\uff1b\u4e0a\u6e38\u811a\u672c\u53ef\u80fd\u8df3\u8fc7\u4e86\u65e0\u6548 batch\u3002"
            )
        if not np.all(np.isfinite(preds)):
            raise RuntimeError("UniStab \u8fd4\u56de NaN/Inf")
        return dict(zip(names, map(float, preds)))
