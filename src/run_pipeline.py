"""Command-line orchestration for the full replication pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .data_loader import load_inputs
from .data_validation import validate_main_data, validate_map_data, write_validation_report
from .descriptive_analysis import prepare_analysis_data, run_descriptive_analysis
from .figures import run_figure_suite
from .propensity_score import run_psm_suite
from .regression_models import run_regression_suite
from .reporting import ensure_output_dirs, write_run_manifest, write_text_log


PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "pyreadstat",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "openpyxl",
]


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _relative_outputs(output_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())


def _collect_warning_rows(output_dir: Path) -> list[dict[str, Any]]:
    """Collect non-empty warning rows from module diagnostics into the manifest."""

    rows: list[dict[str, Any]] = []
    for filename in ("model_warnings.csv", "psm_warnings.csv", "figure_warnings.csv"):
        path = Path(output_dir) / "diagnostics" / filename
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        for record in frame.to_dict(orient="records"):
            warning = record.get("warning")
            if warning is not None and not pd.isna(warning) and str(warning).strip():
                rows.append({"source": filename, **record})
    return rows


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    run_name: str = "legacy",
    seed: int = 123,
    source_dir: Path | None = None,
) -> int:
    """Run loading, validation, analyses, figures, and provenance reporting."""

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    ensure_output_dirs(output_dir)
    started = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "run_name": run_name,
        "status": "started",
        "started_at_utc": started,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "source_dir": source_dir,
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "warnings": [],
        "outputs": [],
    }
    log_lines = [f"run_name={run_name}", f"input_dir={input_dir}", f"output_dir={output_dir}"]
    try:
        inputs = load_inputs(input_dir, source_dir=source_dir)
        manifest["input_provenance"] = inputs["provenance"]
        log_lines.append(f"main_shape={inputs['main'].shape}")
        reports = {
            "main": validate_main_data(inputs["main"]),
            "province_map": validate_map_data(inputs["province_map"], "province_map"),
            "china_map": validate_map_data(inputs["china_map"], "china_map"),
            "china_label": validate_map_data(inputs["china_label"], "china_label"),
        }
        write_validation_report(reports, output_dir)
        manifest["validation"] = reports
        if not all(report["passed"] for report in reports.values()):
            manifest["status"] = "failed_validation"
            manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["warnings"] = [
                {"source": name, "warning": error}
                for name, report in reports.items()
                for error in report.get("errors", [])
            ]
            write_text_log(output_dir, "pipeline.log", log_lines + ["status=failed_validation"])
            manifest["outputs"] = sorted(set(_relative_outputs(output_dir) + ["run_manifest.json"]))
            write_run_manifest(output_dir, manifest)
            return 2

        analysis = prepare_analysis_data(inputs["main"])
        analysis.to_csv(output_dir / "diagnostics" / "analysis_frame.csv", index=False, encoding="utf-8-sig")
        manifest["derived_variable_warnings"] = analysis.attrs.get("analysis_warnings", [])
        descriptive_outputs = run_descriptive_analysis(analysis, output_dir)
        regression_outputs = run_regression_suite(analysis, output_dir)
        psm_outputs = run_psm_suite(analysis, output_dir, seed=seed)
        figure_outputs = run_figure_suite(inputs, analysis, output_dir)
        manifest["warnings"] = _collect_warning_rows(output_dir)
        manifest["warnings"].extend(
            {"source": "derived_variables", "warning": warning}
            for warning in analysis.attrs.get("analysis_warnings", [])
        )
        manifest["module_outputs"] = {
            "descriptive": descriptive_outputs,
            "regression": regression_outputs,
            "psm": psm_outputs,
            "figures": figure_outputs,
        }
        manifest["status"] = "completed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_text_log(output_dir, "pipeline.log", log_lines + ["status=completed"])
        manifest["outputs"] = sorted(set(_relative_outputs(output_dir) + ["run_manifest.json"]))
        write_run_manifest(output_dir, manifest)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["traceback"] = traceback.format_exc()
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["outputs"] = sorted(set(_relative_outputs(output_dir) + ["logs/pipeline.log", "run_manifest.json"]))
        write_run_manifest(output_dir, manifest)
        write_text_log(output_dir, "pipeline.log", log_lines + [f"status=failed", f"error={error!r}"])
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Python legacy-data replication pipeline.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/legacy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/legacy"))
    parser.add_argument("--run-name", default="legacy")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--source-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_pipeline(args.input_dir, args.output_dir, args.run_name, args.seed, args.source_dir)


if __name__ == "__main__":
    raise SystemExit(main())
