"""Schema and range validation for the legacy and future input datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .reporting import ensure_output_dirs


MAIN_REQUIRED_COLUMNS = [
    "y",
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "x7",
    "color",
    "Province",
    "Region",
    "Overduearea",
    "Planneddeliveryarea",
]

MAP_REQUIRED_COLUMNS = {
    "province_map": ["id", "ename", "x_coord", "y_coord", "Overduearea", "Planneddeliveryarea"],
    "china_map": ["_ID", "_X", "_Y"],
    "china_label": ["id", "ename", "x_coord", "y_coord"],
}


def _missing_by_column(frame: pd.DataFrame) -> dict[str, int]:
    return {str(column): int(count) for column, count in frame.isna().sum().items() if count}


def validate_main_data(df: pd.DataFrame) -> dict[str, Any]:
    """Return a non-mutating validation report for the main analytical frame."""

    missing_columns = [column for column in MAIN_REQUIRED_COLUMNS if column not in df.columns]
    missing_by_column = _missing_by_column(df)
    numeric_columns = ["y", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "Overduearea", "Planneddeliveryarea"]
    nonfinite_by_column: dict[str, int] = {}
    for column in numeric_columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            count = int((~values.replace([float("inf"), float("-inf")], pd.NA).notna()).sum())
            if count:
                nonfinite_by_column[column] = count
    nonpositive_planned = 0
    negative_overdue = 0
    if "Planneddeliveryarea" in df.columns:
        nonpositive_planned = int((pd.to_numeric(df["Planneddeliveryarea"], errors="coerce") <= 0).sum())
    if "Overduearea" in df.columns:
        negative_overdue = int((pd.to_numeric(df["Overduearea"], errors="coerce") < 0).sum())

    categories: dict[str, list[Any]] = {}
    for column in ("x2", "color", "Region"):
        if column in df.columns:
            categories[column] = [value for value in df[column].dropna().drop_duplicates().tolist()]

    warnings: list[str] = []
    errors: list[str] = []
    required_value_missing = {column: count for column, count in missing_by_column.items() if column in MAIN_REQUIRED_COLUMNS}
    if required_value_missing:
        errors.append(f"required_value_missing={required_value_missing}")
    if nonfinite_by_column:
        errors.append(f"nonfinite_numeric_values={nonfinite_by_column}")
    if nonpositive_planned:
        errors.append(f"nonpositive_planned_count={nonpositive_planned}")
        warnings.append("Planneddeliveryarea contains non-positive denominators")
    if negative_overdue:
        errors.append(f"negative_overdue_count={negative_overdue}")
    if "x2" in categories and not set(categories["x2"]).issubset({0, 1}):
        errors.append("x2 contains values outside the expected binary levels 0/1")
    if "color" in categories and not set(categories["color"]).issubset({1, 2, 3}):
        errors.append("color contains values outside the expected levels 1/2/3")
    if "Region" in categories and not set(categories["Region"]).issubset(
        {"Central area", "Western area", "Eastern area"}
    ):
        errors.append("Region contains values outside the expected three area labels")

    report = {
        "passed": not missing_columns and not errors,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_required_columns": missing_columns,
        "missing_by_column": missing_by_column,
        "category_levels": categories,
        "nonpositive_planned_count": nonpositive_planned,
        "negative_overdue_count": negative_overdue,
        "nonfinite_by_column": nonfinite_by_column,
        "errors": errors,
        "warnings": warnings,
    }
    return report


def validate_map_data(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """Return a schema report for one map-related data frame."""

    required = MAP_REQUIRED_COLUMNS.get(name, [])
    missing_columns = [column for column in required if column not in df.columns]
    errors: list[str] = []
    warnings: list[str] = []
    if name == "china_map" and {"_X", "_Y"}.issubset(df.columns):
        x_missing = df["_X"].isna()
        y_missing = df["_Y"].isna()
        unmatched_separator = int((x_missing ^ y_missing).sum())
        if unmatched_separator:
            errors.append(f"unmatched_coordinate_separator_count={unmatched_separator}")
        warnings.append(f"coordinate_separator_rows={int((x_missing & y_missing).sum())}")
    for column in ("x_coord", "y_coord", "Overduearea", "Planneddeliveryarea"):
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if int(values.isna().sum()):
                errors.append(f"non_numeric_or_missing_{column}={int(values.isna().sum())}")
    return {
        "name": name,
        "passed": not missing_columns and not errors,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_required_columns": missing_columns,
        "missing_by_column": _missing_by_column(df),
        "errors": errors,
        "warnings": warnings,
    }


def write_validation_report(reports: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and flat CSV validation summaries."""

    dirs = ensure_output_dirs(Path(output_dir))
    json_path = dirs["diagnostics"] / "validation_report.json"
    json_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for name, report in reports.items():
        rows.append(
            {
                "dataset": name,
                "passed": report.get("passed"),
                "row_count": report.get("row_count"),
                "column_count": report.get("column_count"),
                "missing_required_columns": ";".join(report.get("missing_required_columns", [])),
                "errors": ";".join(report.get("errors", [])),
                "warnings": ";".join(report.get("warnings", [])),
            }
        )
    csv_path = dirs["diagnostics"] / "validation_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return json_path, csv_path
