from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .adapters import EpHodAdapter, EnzGFMAdapter, UniStabAdapter


def adapters_from_config(cfg: dict):
    m = cfg["models"]
    return (
        EpHodAdapter(m["ephod"]["repo"], m["ephod"]["python"]),
        EnzGFMAdapter(m["enzgfm"]["repo"], m["enzgfm"]["python"], m["enzgfm"]["model_location"]),
        UniStabAdapter(m["unistab"]["repo"], m["unistab"]["python"], m["unistab"]["checkpoint"]),
    )


def check_paths(cfg: dict) -> list[str]:
    errors: list[str] = []
    for adapter in adapters_from_config(cfg):
        errors.extend(adapter.preflight())
    return errors


def _check_python_cuda(python: str, env: dict[str, str], timeout: int = 60) -> tuple[bool, str]:
    cmd = [
        python,
        "-c",
        "import torch; print(torch.__version__); print(torch.cuda.is_available()); "
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')",
    ]
    try:
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except Exception as e:
        return False, repr(e)
    text = (p.stdout or "").strip()
    ok = p.returncode == 0 and "True" in text and "NO_CUDA" not in text
    return ok, text


def gpu_preflight(cfg: dict) -> dict[str, dict]:
    path_errors = check_paths(cfg)
    if path_errors:
        return {"paths": {"ok": False, "details": path_errors}}
    gpu_id = int(cfg["runtime"]["gpu_id"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    out: dict[str, dict] = {"paths": {"ok": True, "details": []}}
    for name in ["ephod", "enzgfm", "unistab"]:
        ok, detail = _check_python_cuda(cfg["models"][name]["python"], env)
        out[name] = {"ok": ok, "details": detail}
    return out
