from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

from .adapters import EpHodAdapter, EnzGFMAdapter, UniStabAdapter
from .mutations import generate_single_substitutions, apply_mutation
from .runner import build_env


def validate_three_models_cpu(cfg: dict) -> dict:
    seq = cfg["sequence"]
    mutation = generate_single_substitutions(seq)[0]
    mut_seq = apply_mutation(seq, mutation)
    m = cfg["models"]
    ephod = EpHodAdapter(m["ephod"]["repo"], m["ephod"]["python"])
    enzgfm = EnzGFMAdapter(m["enzgfm"]["repo"], m["enzgfm"]["python"], m["enzgfm"]["model_location"])
    unistab = UniStabAdapter(m["unistab"]["repo"], m["unistab"]["python"], m["unistab"]["checkpoint"])
    errors = ephod.preflight() + enzgfm.preflight() + unistab.preflight()
    if errors:
        raise RuntimeError("CPU \u9884\u9a8c\u8bc1\u524d\u8def\u5f84\u68c0\u67e5\u5931\u8d25:\n- " + "\n- ".join(errors))

    runtime = cfg["runtime"]
    env = build_env(force_cpu=True, gpu_id=0, cache_dir=runtime["cache_dir"])
    timeout = int(runtime["timeout_seconds"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="cpu_model_validation_", dir=str(out_dir)))
    try:
        activity = enzgfm.score_mutations(
            seq, [mutation], work_dir=temp_root / "enzgfm", env=env, timeout=timeout, force_cpu=True
        )[mutation]
        ph = ephod.predict(
            {"WT": seq, mutation: mut_seq}, work_dir=temp_root / "ephod", env=env, timeout=timeout
        )
        ddg = unistab.predict_ddg(
            seq, {mutation: mut_seq}, work_dir=temp_root / "unistab", env=env, timeout=timeout, force_cpu=True
        )[mutation]
        values = [activity, ph["WT"], ph[mutation], ddg]
        if not all(math.isfinite(float(v)) for v in values):
            raise RuntimeError(f"CPU \u771f\u5b9e\u63a8\u7406\u8fd4\u56de\u975e\u6709\u9650\u503c: {values}")
        return {
            "mutation": mutation,
            "enzgfm_activity_proxy": float(activity),
            "ephod_wt_ph": float(ph["WT"]),
            "ephod_mutant_ph": float(ph[mutation]),
            "unistab_ddg": float(ddg),
            "passed": True,
        }
    finally:
        if bool(runtime.get("keep_temp", False)):
            print(f"CPU \u9a8c\u8bc1\u4e34\u65f6\u76ee\u5f55\u4fdd\u7559: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
