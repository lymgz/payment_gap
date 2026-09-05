"""Shared file, output, and provenance helpers for the replication pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OUTPUT_SUBDIRECTORIES = ("tables", "figures", "diagnostics", "logs")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create and return the standard output directories."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"root": output_dir}
    for name in OUTPUT_SUBDIRECTORIES:
        child = output_dir / name
        child.mkdir(parents=True, exist_ok=True)
        result[name] = child
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def write_run_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write a stable, human-readable JSON manifest for one pipeline run."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return path


def write_text_log(output_dir: Path, filename: str, lines: list[str]) -> Path:
    """Write a UTF-8 log file under the standard logs directory."""

    dirs = ensure_output_dirs(Path(output_dir))
    path = dirs["logs"] / filename
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
