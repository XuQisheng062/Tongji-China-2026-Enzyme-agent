from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


class ExternalCommandError(RuntimeError):
    pass


def build_env(*, force_cpu: bool, gpu_id: int, cache_dir: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cache = str(Path(cache_dir).expanduser().resolve())
    Path(cache).mkdir(parents=True, exist_ok=True)
    env.setdefault("TORCH_HOME", str(Path(cache) / "torch"))
    env.setdefault("HF_HOME", str(Path(cache) / "huggingface"))
    env.setdefault("TRANSFORMERS_CACHE", str(Path(cache) / "huggingface" / "transformers"))
    return env


def _apply_ca_bundle(env: dict[str, str], cmd: list[str]) -> dict[str, str]:
    """Provide external model subprocesses with a usable HTTPS CA bundle.

    Existing explicit CA variables are preserved. Otherwise system CA is used
    first, then certifi from the subprocess Python environment is attempted.
    SSL verification is never disabled.
    """
    out = dict(env)
    existing_ssl = out.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_FILE")
    existing_requests = out.get("REQUESTS_CA_BUNDLE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if existing_ssl:
        out["SSL_CERT_FILE"] = existing_ssl
        out.setdefault("REQUESTS_CA_BUNDLE", existing_ssl)
        return out
    if existing_requests:
        out["REQUESTS_CA_BUNDLE"] = existing_requests
        out.setdefault("SSL_CERT_FILE", existing_requests)
        return out

    for candidate in [
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/ca-bundle.pem",
    ]:
        if Path(candidate).is_file():
            out["SSL_CERT_FILE"] = candidate
            out["REQUESTS_CA_BUNDLE"] = candidate
            return out

    if cmd:
        python_exe = str(cmd[0])
        if Path(python_exe).is_file():
            try:
                probe = subprocess.run(
                    [python_exe, "-c", "import certifi; print(certifi.where())"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=5,
                    check=True,
                )
                ca = probe.stdout.strip()
                if ca and Path(ca).is_file():
                    out["SSL_CERT_FILE"] = ca
                    out["REQUESTS_CA_BUNDLE"] = ca
            except Exception:
                pass
    return out


def run_checked(
    cmd: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str],
    timeout: int,
    log_file: str | Path,
) -> subprocess.CompletedProcess[str]:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = _apply_ca_bundle(env, cmd)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    header = "$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n"
    log_path.write_text(header + (proc.stdout or ""), encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-40:])
        raise ExternalCommandError(
            f"\u5916\u90e8\u6a21\u578b\u547d\u4ee4\u5931\u8d25\uff0creturncode={proc.returncode}\n"
            f"\u65e5\u5fd7: {log_path}\n\u6700\u540e\u8f93\u51fa:\n{tail}"
        )
    return proc
