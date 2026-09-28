from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 50_000_000
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".sh"}
IGNORED_PARTS = {".git", ".venv", "cache", "outputs", "results", "__pycache__"}


def repository_files() -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return sorted({ROOT / line for line in output.splitlines() if line and (ROOT / line).is_file()})
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [path for path in ROOT.rglob("*") if path.is_file() and not (
            set(path.relative_to(ROOT).parts) & IGNORED_PARTS
        )]


def main() -> None:
    files = repository_files()
    oversized = [path for path in files if path.stat().st_size >= MAX_BYTES]
    total = sum(path.stat().st_size for path in files)
    non_ascii: list[str] = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        # User-requested documentation is Chinese; executable sources and configs remain ASCII.
        if path.suffix.lower() == ".md" or path.relative_to(ROOT).parts[0] == "docs":
            continue
        try:
            value = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(ord(char) > 127 for char in value):
            non_ascii.append(str(path.relative_to(ROOT)))
    errors = []
    if oversized:
        errors.append("Files at or above 50 MB: " + ", ".join(map(str, oversized)))
    if total >= MAX_BYTES:
        errors.append(f"Repository content is {total / 1024 / 1024:.2f} MB")
    if non_ascii:
        errors.append("Non-ASCII text files: " + ", ".join(non_ascii))
    if errors:
        raise SystemExit("Repository validation failed:\n- " + "\n- ".join(errors))
    print(f"Repository validation passed: {len(files)} files, {total / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
