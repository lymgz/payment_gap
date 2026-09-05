"""Derived variables and non-model descriptive analyses."""

from __future__ import annotations

import warnings as python_warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .reporting import ensure_output_dirs


COLOR_LABELS = {1: "Red", 2: "Yellow", 3: "Green"}
REGION_LABELS = {"Central area": 1, "Western area": 2, "Eastern area": 3}
BASELINE_CONTINUOUS = ["x1", "x3", "x4", "x5", "x6", "x7"]


def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create the variables used by the historical Stata workflow."""

    result = df.copy(deep=True)
    warnings: list[str] = []
    result["Overdue"] = pd.to_numeric(result["Overduearea"], errors="coerce")
    result["Planned"] = pd.to_numeric(result["Planneddeliveryarea"], errors="coerce")
    valid_denominator = result["Planned"] > 0
    invalid_count = int((~valid_denominator).sum())
    result["O_rate"] = np.where(valid_denominator, result["Overdue"] / result["Planned"], np.nan)
    if invalid_count:
        warnings.append(f"nonpositive_planned_count={invalid_count}")
    result["color_text"] = result["color"].map(COLOR_LABELS)
    unmapped_colors = int(result["color_text"].isna().sum())
    if unmapped_colors:
        warnings.append(f"unmapped_color_count={unmapped_colors}")
    result["Region_n"] = result["Region"].map(REGION_LABELS)
    unmapped_regions = int(result["Region_n"].isna().sum())
    if unmapped_regions:
        warnings.append(f"unmapped_region_count={unmapped_regions}")
    result.attrs["analysis_warnings"] = warnings
    return result


def summarize_variables(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Return count and distribution summaries in a stable long format."""

    rows: list[dict[str, Any]] = []
    for variable in variables:
        if variable not in df.columns:
            continue
        values = pd.to_numeric(df[variable], errors="coerce")
        nonmissing = values.dropna()
        rows.append(
            {
                "variable": variable,
                "n": int(nonmissing.size),
                "missing": int(values.isna().sum()),
                "mean": float(nonmissing.mean()) if len(nonmissing) else np.nan,
                "std": float(nonmissing.std(ddof=1)) if len(nonmissing) > 1 else np.nan,
                "min": float(nonmissing.min()) if len(nonmissing) else np.nan,
                "p25": float(nonmissing.quantile(0.25)) if len(nonmissing) else np.nan,
                "median": float(nonmissing.median()) if len(nonmissing) else np.nan,
                "p75": float(nonmissing.quantile(0.75)) if len(nonmissing) else np.nan,
                "max": float(nonmissing.max()) if len(nonmissing) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_frequency_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the region/color and province/color frequency tables."""

    region_color = pd.crosstab(
        df["Region"],
        df["color_text"],
        dropna=False,
    ).reset_index()
    province_color = pd.crosstab(
        df["Province"],
        df["color_text"],
        dropna=False,
    ).reset_index()
    color_counts = df["color_text"].value_counts(dropna=False).rename_axis("color_text").reset_index(name="n")
    return {
        "region_color": region_color,
        "province_color": province_color,
        "color_counts": color_counts,
    }


def run_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate uncentered-style VIF values for the numeric predictors."""

    variables = ["x1", "x2", *BASELINE_CONTINUOUS[1:]]
    available = [variable for variable in variables if variable in df.columns]
    matrix = df[available].apply(pd.to_numeric, errors="coerce").dropna()
    if matrix.empty:
        return pd.DataFrame(columns=["variable", "vif", "n_obs"])
    values = matrix.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for index, variable in enumerate(available):
        with python_warnings.catch_warnings():
            python_warnings.simplefilter("ignore", RuntimeWarning)
            try:
                vif = float(variance_inflation_factor(values, index))
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
                vif = np.inf
        rows.append({"variable": variable, "vif": vif, "n_obs": int(len(matrix))})
    return pd.DataFrame(rows)


def run_grouped_ttests(df: pd.DataFrame, group: str, variables: list[str]) -> pd.DataFrame:
    """Run Welch t tests for numeric variables across binary group values 0 and 1."""

    rows: list[dict[str, Any]] = []
    for variable in variables:
        if variable not in df.columns or group not in df.columns:
            continue
        group_zero = pd.to_numeric(df.loc[df[group] == 0, variable], errors="coerce").dropna()
        group_one = pd.to_numeric(df.loc[df[group] == 1, variable], errors="coerce").dropna()
        if len(group_zero) and len(group_one):
            test = stats.ttest_ind(group_zero, group_one, equal_var=False, nan_policy="omit")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        else:
            statistic = np.nan
            p_value = np.nan
        rows.append(
            {
                "variable": variable,
                "group_0_n": int(len(group_zero)),
                "group_1_n": int(len(group_one)),
                "group_0_mean": float(group_zero.mean()) if len(group_zero) else np.nan,
                "group_1_mean": float(group_one.mean()) if len(group_one) else np.nan,
                "t_statistic": statistic,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_descriptive_analysis(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Write the descriptive and diagnostic tables and return their paths."""

    dirs = ensure_output_dirs(Path(output_dir))
    summary = summarize_variables(
        df,
        ["y", *BASELINE_CONTINUOUS, "x2", "color", "Overdue", "Planned", "O_rate"],
    )
    frequencies = build_frequency_tables(df)
    area_summary = (
        df.groupby("color_text", dropna=False)[["Overdue", "Planned"]]
        .agg(["count", "mean", "median", "sum"])
        .reset_index()
    )
    area_summary.columns = [
        "_".join(str(part) for part in column if str(part) != "").rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in area_summary.columns
    ]
    ttests = run_grouped_ttests(df, "x2", ["y", *BASELINE_CONTINUOUS])
    outputs = {
        "summary": _write_table(summary, dirs["tables"] / "descriptive_summary.csv"),
        "frequency_region_color": _write_table(
            frequencies["region_color"], dirs["tables"] / "frequency_region_color.csv"
        ),
        "frequency_province_color": _write_table(
            frequencies["province_color"], dirs["tables"] / "frequency_province_color.csv"
        ),
        "frequency_color": _write_table(frequencies["color_counts"], dirs["tables"] / "frequency_color.csv"),
        "color_area_summary": _write_table(area_summary, dirs["tables"] / "color_area_summary.csv"),
        "vif": _write_table(run_vif(df), dirs["diagnostics"] / "vif.csv"),
        "grouped_ttests": _write_table(ttests, dirs["diagnostics"] / "grouped_ttests.csv"),
    }
    return outputs
