"""Input discovery and Stata data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyreadstat

from .reporting import sha256_file


INPUT_FILES = {
    "main": "data.dta",
    "province_map": "province_map.dta",
    "china_map": "china_map.dta",
    "china_label": "china_label.dta",
}


def load_dta(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a Stata file and return a frame plus provenance metadata."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Stata input not found: {path}")
    frame, metadata = pyreadstat.read_dta(path, apply_value_formats=False)
    provenance = {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "column_labels": metadata.column_names_to_labels,
    }
    return frame, provenance


def load_inputs(input_dir: Path, source_dir: Path | None = None) -> dict[str, Any]:
    """Load the standard analytical and map inputs from one directory."""

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    loaded: dict[str, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for key, filename in INPUT_FILES.items():
        frame, metadata = load_dta(input_dir / filename)
        if source_dir is not None:
            source_path = Path(source_dir) / filename
            if source_path.is_file():
                source_hash = sha256_file(source_path)
                metadata["source_path"] = str(source_path.resolve())
                metadata["source_sha256"] = source_hash
                metadata["source_hash_matches_copy"] = source_hash == metadata["sha256"]
            else:
                metadata["source_path"] = str(source_path.resolve())
                metadata["source_sha256"] = None
                metadata["source_hash_matches_copy"] = False
        loaded[key] = frame
        provenance[filename] = metadata
    loaded["provenance"] = provenance
    return loaded
