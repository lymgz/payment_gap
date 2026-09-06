"""Runnable analysis built around the legacy database and supplemental fields.

The module deliberately keeps the two margins separate:

1. a full-sample model for whether all planned delivery area is overdue;
2. a conditional severity model estimated only below that upper mass point.

The supplemental workbook is treated as a source of coded variables. No field is
silently recoded when its name and codebook disagree.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import traceback
import warnings as pywarnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from .reporting import ensure_output_dirs, sha256_file, write_run_manifest, write_text_log
from .supplemental_figures import run_supplemental_figures


SUPPLEMENT_FIELDS = [
    "firm_id",
    "completion_pct",
    "months_overdue",
    "offset_discount",
    "gov_intervention",
    "contractor_litigation",
    "contractor_soe",
]

MAIN_FIELDS = [
    "code",
    "color",
    "y",
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "x7",
    "Overduearea",
    "Planneddeliveryarea",
    "Province",
    "City",
    "Region",
]

SUPPLEMENT_LOCATION_FIELDS = ["province_additional", "city_additional", "region_additional"]
NUMERIC_FIELDS = [
    "color",
    "y",
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "x7",
    "Overduearea",
    "Planneddeliveryarea",
    *SUPPLEMENT_FIELDS,
]


def _read_query(connection: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, connection)


def _open_readonly(database: Path) -> sqlite3.Connection:
    uri = f"file:{database.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_joined_database(database: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load main and supplemental rows and join them one-to-one by project code."""

    database = Path(database).resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {database}")
    connection = _open_readonly(database)
    try:
        main = _read_query(
            connection,
            "SELECT "
            + ", ".join(f'"{field}"' for field in MAIN_FIELDS)
            + " FROM all_data WHERE dataset_name = 'main'",
        )
        supplemental_fields = ["code", *SUPPLEMENT_LOCATION_FIELDS, *SUPPLEMENT_FIELDS]
        supplemental = _read_query(
            connection,
            "SELECT "
            + ", ".join(f'"{field}"' for field in supplemental_fields)
            + " FROM all_data WHERE dataset_name = 'supplemental'",
        )
        provenance = _read_query(
            connection,
            "SELECT source_file, source_sha256, dataset_name, COUNT(*) AS row_count "
            "FROM all_data GROUP BY source_file, source_sha256, dataset_name",
        )
    finally:
        connection.close()

    if main["code"].duplicated().any():
        raise ValueError("Main database rows contain duplicate code values")
    if supplemental["code"].duplicated().any():
        raise ValueError("Supplemental database rows contain duplicate code values")
    joined = main.merge(supplemental, on="code", how="outer", validate="one_to_one", indicator=True)
    unmatched = joined.loc[joined["_merge"] != "both", ["code", "_merge"]]
    if not unmatched.empty:
        raise ValueError(f"Main and supplemental rows do not match one-to-one: {unmatched.to_dict('records')[:5]}")
    joined = joined.drop(columns="_merge")
    return joined, {
        "database": str(database),
        "database_sha256": sha256_file(database),
        "main_rows": int(len(main)),
        "supplemental_rows": int(len(supplemental)),
        "joined_rows": int(len(joined)),
        "provenance": provenance.to_dict(orient="records"),
    }


def prepare_analysis_frame(frame: pd.DataFrame, completion_cutoff: float = 2.0) -> pd.DataFrame:
    """Create outcomes and transparent robustness proxies without changing raw fields."""

    result = frame.copy()
    conversion_warnings: list[str] = []
    for column in NUMERIC_FIELDS:
        if column in result.columns:
            raw = result[column]
            converted = pd.to_numeric(raw, errors="coerce")
            invalid = int((raw.notna() & converted.isna()).sum())
            if invalid:
                conversion_warnings.append(f"{column}_invalid_numeric_count={invalid}")
            result[column] = converted
    result["Overdue"] = result["Overduearea"]
    result["Planned"] = result["Planneddeliveryarea"]
    planned = result["Planned"]
    result["O_rate"] = np.where(planned > 0, result["Overdue"] / planned, np.nan)
    result["O_rate_clipped"] = result["O_rate"].clip(1e-4, 1 - 1e-4)
    result["full_overdue_incidence"] = np.where(
        result["O_rate"].notna(), (result["O_rate"] >= 0.9999).astype(int), np.nan
    )
    result["completion_stage"] = result["completion_pct"]
    result["completion_low_proxy"] = np.where(
        result["completion_stage"].notna(),
        (result["completion_stage"] <= completion_cutoff).astype(int),
        np.nan,
    )
    result["x1_int"] = np.where(planned > 0, result["x1"] / planned, np.nan)
    finite_x1_int = result["x1_int"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_x1_int) >= 5:
        low, high = finite_x1_int.quantile([0.01, 0.99])
        result["x1_int_w"] = result["x1_int"].clip(float(low), float(high))
    else:
        result["x1_int_w"] = result["x1_int"]
    result["supplement_complete"] = result[SUPPLEMENT_FIELDS].notna().all(axis=1).astype(int)
    result.attrs["completion_cutoff"] = completion_cutoff
    result.attrs["numeric_conversion_warnings"] = conversion_warnings
    result.attrs["x1_int_winsor_limits"] = (
        [float(low), float(high)] if len(finite_x1_int) >= 5 else [None, None]
    )
    return result


def merge_marketization(
    frame: pd.DataFrame,
    marketization_csv: Path | None,
    target_year: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Optionally merge marketization after year filtering and uniqueness checks."""

    result = frame.copy()
    if marketization_csv is None:
        return result, {"status": "not_supplied", "matched_rows": 0, "unmatched_rows": len(result)}, [
            "marketization_csv_not_supplied"
        ]
    marketization_csv = Path(marketization_csv).resolve()
    if not marketization_csv.is_file():
        raise FileNotFoundError(f"Marketization CSV not found: {marketization_csv}")
    market = pd.read_csv(marketization_csv)
    market.columns = [str(column).strip() for column in market.columns]
    province_column = next((c for c in ("Province", "province", "province_name") if c in market.columns), None)
    year_column = next((c for c in ("index_year", "year", "Year") if c in market.columns), None)
    value_column = next(
        (c for c in ("marketization", "marketization_index", "index", "market_index") if c in market.columns),
        None,
    )
    if province_column is None or value_column is None:
        raise ValueError("Marketization CSV must contain a province column and a marketization/index column")
    if year_column is not None:
        market = market.loc[pd.to_numeric(market[year_column], errors="coerce") == target_year].copy()
    market = market.rename(columns={province_column: "Province", value_column: "marketization_index"})
    if market["Province"].duplicated().any():
        duplicates = market.loc[market["Province"].duplicated(), "Province"].tolist()
        raise ValueError(f"Marketization data is not unique by Province after year filtering: {duplicates[:5]}")
    result = result.merge(
        market[["Province", "marketization_index"]],
        on="Province",
        how="left",
        validate="many_to_one",
    )
    matched = int(result["marketization_index"].notna().sum())
    info = {
        "status": "merged",
        "path": str(marketization_csv),
        "target_year": target_year,
        "matched_rows": matched,
        "unmatched_rows": int(len(result) - matched),
        "marketization_rows_after_year_filter": int(len(market)),
    }
    warnings = ["marketization_rows_unmatched"] if matched < len(result) else []
    return result, info, warnings


def _write_table(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    variables = [*SUPPLEMENT_FIELDS, "O_rate", "x1_int", "x1_int_w", "marketization_index"]
    rows = []
    for variable in variables:
        if variable not in frame.columns:
            continue
        series = frame[variable]
        rows.append(
            {
                "variable": variable,
                "n": int(len(series)),
                "missing_n": int(series.isna().sum()),
                "observed_rate": float(series.notna().mean()),
                "unique_n": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def selection_check_report(frame: pd.DataFrame) -> pd.DataFrame:
    core = ["y", "x1", "x2", "x3", "x4", "O_rate", "full_overdue_incidence"]
    rows: list[dict[str, Any]] = []
    for supplement in [*SUPPLEMENT_FIELDS, "supplement_complete"]:
        if supplement == "supplement_complete":
            complete = frame.loc[frame[supplement] == 1]
        else:
            complete = frame.loc[frame[supplement].notna()]
        for variable in core:
            full_values = pd.to_numeric(frame[variable], errors="coerce").dropna()
            complete_values = pd.to_numeric(complete[variable], errors="coerce").dropna()
            if len(full_values) < 2 or len(complete_values) < 2:
                continue
            pooled_sd = np.sqrt((full_values.var(ddof=1) + complete_values.var(ddof=1)) / 2)
            standardized_difference = (
                float((complete_values.mean() - full_values.mean()) / pooled_sd) if pooled_sd > 0 else np.nan
            )
            rows.append(
                {
                    "supplement_variable": supplement,
                    "core_variable": variable,
                    "full_n": int(len(full_values)),
                    "complete_n": int(len(complete_values)),
                    "full_mean": float(full_values.mean()),
                    "complete_mean": float(complete_values.mean()),
                    "standardized_difference": standardized_difference,
                }
            )
    return pd.DataFrame(rows)


def codebook_validation(frame: pd.DataFrame) -> dict[str, Any]:
    """Report observed codes and known template/workbook definition conflicts."""

    observed: dict[str, list[Any]] = {}
    for field in SUPPLEMENT_FIELDS:
        values = frame[field].dropna().unique().tolist()
        observed[field] = sorted(values)
    conflicts = [
        {
            "field": "firm_id",
            "issue": "template says F01-F10, workbook data and codebook use 0-3 operator/stage codes",
            "severity": "review_required",
        },
        {
            "field": "completion_pct",
            "issue": "field name/template say 0-100 percentage, workbook codebook uses construction-stage codes and observed values are 1-5",
            "severity": "review_required",
        },
        {
            "field": "months_overdue",
            "issue": "template wording suggests months, workbook codebook uses four duration bands 1-4",
            "severity": "review_required",
        },
        {
            "field": "gov_intervention",
            "issue": "plan says 0-2, workbook data/codebook use 0-3 with code 3 for fiscal capital injection",
            "severity": "review_required",
        },
    ]
    value_warnings = []
    if any(float(value) > 1 for value in frame["offset_discount"].dropna()):
        value_warnings.append("offset_discount_contains_values_above_one")
    for field, allowed in {
        "firm_id": {0, 1, 2, 3},
        "months_overdue": {1, 2, 3, 4},
        "gov_intervention": {0, 1, 2, 3},
        "contractor_litigation": {0, 1},
        "contractor_soe": {0, 1},
    }.items():
        actual = set(frame[field].dropna().astype(float).astype(int))
        unexpected = sorted(actual - allowed)
        if unexpected:
            value_warnings.append(f"{field}_unexpected_codes={unexpected}")
    return {
        "observed_codes": observed,
        "conflicts_requiring_definition_confirmation": conflicts,
        "value_warnings": value_warnings,
        "status": "review_required" if conflicts or value_warnings else "passed",
    }


def _manual_cluster_covariance(fit: Any, groups: pd.Series) -> pd.DataFrame:
    """Compute a finite CR1 covariance for canonical binomial GLMs.

    Statsmodels' GLM cluster path can produce all-NaN standard errors when one
    fitted probability rounds to exactly one.  The canonical logit score is
    still finite: x_i * (y_i - mu_i), so it can be aggregated by cluster
    directly without dividing by the Bernoulli variance.
    """

    design = np.asarray(fit.model.exog, dtype=float)
    observed = np.asarray(fit.model.endog, dtype=float)
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    group_values = np.asarray(groups)
    if len(group_values) != len(observed):
        raise ValueError("Cluster labels are not aligned with model observations")
    unique_groups = pd.unique(group_values)
    n_obs, n_params = design.shape
    if len(unique_groups) < 2 or n_obs <= n_params:
        raise ValueError("Cluster covariance requires at least two clusters and positive residual degrees of freedom")

    bread = np.asarray(fit.normalized_cov_params, dtype=float)
    scores = design * (observed - fitted)[:, None]
    meat = np.zeros((n_params, n_params), dtype=float)
    for group in unique_groups:
        cluster_score = scores[group_values == group].sum(axis=0)[:, None]
        meat += cluster_score @ cluster_score.T
    correction = (len(unique_groups) / (len(unique_groups) - 1)) * ((n_obs - 1) / (n_obs - n_params))
    covariance = correction * bread @ meat @ bread
    if not np.isfinite(covariance).all():
        raise ValueError("Manual cluster covariance contains non-finite values")
    names = list(fit.params.index)
    return pd.DataFrame(covariance, index=names, columns=names)


def _tidy_fit(
    model_name: str,
    fit: Any,
    n: int,
    covariance: str,
    warning: str | None = None,
    covariance_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    params = pd.Series(getattr(fit, "params", pd.Series(dtype=float)))
    if covariance_override is None:
        bse = pd.Series(getattr(fit, "bse", pd.Series(index=params.index, dtype=float)), index=params.index)
        pvalues = pd.Series(getattr(fit, "pvalues", pd.Series(index=params.index, dtype=float)), index=params.index)
        conf = fit.conf_int() if hasattr(fit, "conf_int") else pd.DataFrame(index=params.index, columns=[0, 1])
    else:
        bse = pd.Series(np.sqrt(np.clip(np.diag(covariance_override), 0, None)), index=params.index)
        z_values = params / bse.replace(0, np.nan)
        pvalues = pd.Series(2 * norm.sf(np.abs(z_values)), index=params.index)
        critical = norm.ppf(0.975)
        conf = pd.DataFrame({0: params - critical * bse, 1: params + critical * bse}, index=params.index)
    rows = []
    for term, coefficient in params.items():
        rows.append(
            {
                "model": model_name,
                "term": term,
                "coefficient": float(coefficient),
                "std_error": float(bse.get(term, np.nan)),
                "p_value": float(pvalues.get(term, np.nan)),
                "conf_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
                "conf_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
                "n": n,
                "covariance": covariance,
                "converged": bool(getattr(fit, "converged", True)),
                "warning": warning or "",
            }
        )
    return pd.DataFrame(rows)


def fit_glm_binomial(
    frame: pd.DataFrame,
    formula: str,
    model_name: str,
    cluster: str = "Province",
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    data = frame.replace([np.inf, -np.inf], np.nan).copy()
    data = data.dropna(
        subset=["full_overdue_incidence"] if "full_overdue_incidence" in formula else ["O_rate_clipped"]
    )
    design = patsy.dmatrices(formula, data=data, return_type="dataframe")
    usable = pd.concat([design[0], design[1]], axis=1).dropna()
    data = data.loc[usable.index]
    warnings: list[str] = []
    if len(data) < 10 or data[cluster].nunique() < 2:
        return pd.DataFrame(), {"model": model_name, "status": "underidentified", "n": len(data)}, [
            f"{model_name}_underidentified"
        ]
    model = smf.glm(formula=formula, data=data, family=sm.families.Binomial())
    covariance = "model_default"
    covariance_override: pd.DataFrame | None = None
    try:
        with pywarnings.catch_warnings(record=True) as captured:
            pywarnings.simplefilter("always")
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster]})
        covariance = f"cluster:{cluster}"
        warnings.extend(f"{model_name}_runtime_warning={item.message}" for item in captured)
        if not np.isfinite(np.asarray(fit.bse, dtype=float)).all():
            covariance_override = _manual_cluster_covariance(fit, data[cluster])
            warnings.append(f"{model_name}_used_manual_cluster_covariance_after_nonfinite_statsmodels_result")
    except Exception as error:
        warnings.append(f"{model_name}_cluster_covariance_failed={error!r}")
        covariance = "model_default_after_cluster_failure"
        covariance_override = None
        with pywarnings.catch_warnings(record=True) as captured:
            pywarnings.simplefilter("always")
            fit = model.fit()
        warnings.extend(f"{model_name}_runtime_warning={item.message}" for item in captured)
    tidy = _tidy_fit(
        model_name,
        fit,
        len(data),
        covariance,
        "; ".join(warnings),
        covariance_override=covariance_override,
    )
    metadata = {
        "model": model_name,
        "formula": formula,
        "n": len(data),
        "clusters": int(data[cluster].nunique()),
        "status": "completed",
        "covariance": covariance,
    }
    return tidy, metadata, warnings


def fit_fractional_severity(
    frame: pd.DataFrame,
    formula: str,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    severity = frame.loc[frame["full_overdue_incidence"] == 0].copy()
    return fit_glm_binomial(severity, formula, model_name)


def fit_ordered_duration(
    frame: pd.DataFrame,
    model_name: str = "duration_ordered_logit",
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    base_columns = [
        "x1",
        "x2",
        "x3",
        "x4",
        "contractor_litigation",
        "contractor_soe",
    ]
    data = frame[base_columns + ["completion_stage", "gov_intervention", "months_overdue"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    warnings: list[str] = []
    if len(data) < 20 or data["months_overdue"].nunique() < 3:
        return pd.DataFrame(), {"model": model_name, "status": "underidentified", "n": len(data)}, [
            f"{model_name}_underidentified"
        ]
    exog = data[base_columns].astype(float)
    completion_dummies = pd.get_dummies(data["completion_stage"], prefix="completion_stage", drop_first=True, dtype=float)
    government_dummies = pd.get_dummies(data["gov_intervention"], prefix="gov_intervention", drop_first=True, dtype=float)
    exog = pd.concat([exog, completion_dummies, government_dummies], axis=1)
    try:
        fit = OrderedModel(data["months_overdue"].astype(int), exog, distr="logit").fit(method="bfgs", disp=False)
        tidy = _tidy_fit(model_name, fit, len(data), "model_default")
        return tidy, {"model": model_name, "n": len(data), "status": "completed", "formula": "OrderedModel"}, warnings
    except Exception as error:
        warnings.append(f"{model_name}_failed={error!r}")
        return pd.DataFrame(), {"model": model_name, "status": "failed", "n": len(data)}, warnings


def wild_cluster_bootstrap_ols(
    frame: pd.DataFrame,
    formula: str,
    targets: list[str],
    cluster: str = "Province",
    repetitions: int = 199,
    seed: int = 123,
) -> tuple[pd.DataFrame, list[str]]:
    """Return exploratory wild-cluster bootstrap p-values for a severity OLS."""

    data = frame.loc[frame["full_overdue_incidence"] == 0].replace([np.inf, -np.inf], np.nan).copy()
    warnings: list[str] = []
    if len(data) < 20 or data[cluster].nunique() < 3:
        return pd.DataFrame(), ["wild_bootstrap_underidentified"]
    y, x = patsy.dmatrices(formula, data=data, return_type="dataframe", NA_action="drop")
    data = data.loc[x.index]
    y_values = y.iloc[:, 0].to_numpy(dtype=float)
    x_values = x.to_numpy(dtype=float)
    unrestricted = sm.OLS(y_values, x_values).fit()
    cluster_values = data[cluster].to_numpy()
    unique_clusters = np.unique(cluster_values)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for target in targets:
        matching = [index for index, name in enumerate(x.columns) if name == target or target in name]
        if not matching:
            warnings.append(f"wild_bootstrap_target_not_found={target}")
            continue
        for index in matching:
            restricted_x = np.delete(x_values, index, axis=1)
            restricted_beta = np.linalg.lstsq(restricted_x, y_values, rcond=None)[0]
            fitted_null = restricted_x @ restricted_beta
            residual = y_values - fitted_null
            bootstrap_coefficients = []
            for _ in range(repetitions):
                signs = {cluster_value: rng.choice([-1.0, 1.0]) for cluster_value in unique_clusters}
                y_star = fitted_null + residual * np.array([signs[value] for value in cluster_values])
                bootstrap_beta = np.linalg.lstsq(x_values, y_star, rcond=None)[0]
                bootstrap_coefficients.append(float(bootstrap_beta[index]))
            observed = float(unrestricted.params[index])
            p_value = float(np.mean(np.abs(bootstrap_coefficients) >= abs(observed)))
            rows.append(
                {
                    "model": "severity_ols",
                    "target": x.columns[index],
                    "observed_coefficient": observed,
                    "wild_cluster_p_value": p_value,
                    "repetitions": repetitions,
                    "clusters": int(len(unique_clusters)),
                    "n": int(len(data)),
                    "note": "Rademacher wild-cluster bootstrap; exploratory with few province clusters",
                }
            )
    return pd.DataFrame(rows), warnings


def run_models(
    frame: pd.DataFrame,
    output_dir: Path,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    dirs = ensure_output_dirs(Path(output_dir))
    warnings: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    model_frames: list[pd.DataFrame] = []
    market_term = " + marketization_index" if "marketization_index" in frame.columns else ""
    base_formula = "full_overdue_incidence ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + C(color)" + market_term
    supplement_formula = (
        base_formula
        + " + C(completion_stage) + C(gov_intervention) + C(firm_id) + contractor_litigation + contractor_soe"
    )
    severity_formula = (
        "O_rate_clipped ~ x1_int_w + x2 + x3 + x4 + C(completion_stage) + C(gov_intervention) "
        "+ offset_discount + contractor_litigation + contractor_soe"
    )
    proxy_formula = "completion_low_proxy ~ x1 + x2 + x3 + x4 + C(gov_intervention) + contractor_litigation + contractor_soe"
    for formula, model_name in (
        (base_formula, "full_overdue_logit_base"),
        (supplement_formula, "full_overdue_logit_supplement"),
        (proxy_formula, "completion_low_proxy_logit"),
    ):
        terms, metadata, current_warnings = fit_glm_binomial(frame, formula, model_name)
        model_frames.append(terms)
        metadata_rows.append(metadata)
        warnings.extend(current_warnings)
    terms, metadata, current_warnings = fit_fractional_severity(frame, severity_formula, "severity_fractional_logit")
    model_frames.append(terms)
    metadata_rows.append(metadata)
    warnings.extend(current_warnings)
    duration_terms, duration_metadata, duration_warnings = fit_ordered_duration(frame)
    model_frames.append(duration_terms)
    metadata_rows.append(duration_metadata)
    warnings.extend(duration_warnings)
    bootstrap_formula = (
        "O_rate ~ x1_int_w + x2 + x3 + x4 + C(completion_stage) + C(gov_intervention) "
        "+ contractor_litigation + contractor_soe"
    )
    bootstrap_targets = [
        "x1_int_w",
        "C(completion_stage)",
        "C(gov_intervention)",
        "contractor_litigation",
        "contractor_soe",
    ]
    bootstrap, bootstrap_warnings = wild_cluster_bootstrap_ols(
        frame, bootstrap_formula, bootstrap_targets, repetitions=bootstrap_repetitions, seed=seed
    )
    warnings.extend(bootstrap_warnings)
    _write_table(bootstrap, dirs["tables"] / "wild_cluster_bootstrap.csv")
    _write_table(pd.DataFrame(metadata_rows), dirs["tables"] / "model_metadata.csv")
    nonempty_frames = [part for part in model_frames if not part.empty]
    model_terms = pd.concat(nonempty_frames, ignore_index=True) if nonempty_frames else pd.DataFrame()
    _write_table(model_terms, dirs["tables"] / "model_terms.csv")
    return model_terms, metadata_rows, warnings


def run_supplemental_analysis(
    database: Path,
    output_dir: Path,
    completion_cutoff: float = 2.0,
    target_year: int = 2023,
    marketization_csv: Path | None = None,
    bootstrap_repetitions: int = 199,
    seed: int = 123,
) -> int:
    """Run the complete supplemental-variable workflow and generate figures."""

    output_dir = Path(output_dir).resolve()
    dirs = ensure_output_dirs(output_dir)
    manifest: dict[str, Any] = {
        "run_name": "supplemental_analysis",
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": str(Path(database).resolve()),
        "output_dir": str(output_dir),
        "completion_cutoff": completion_cutoff,
        "target_year": target_year,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "warnings": [],
        "outputs": [],
    }
    log_lines = [f"database={database}", f"output_dir={output_dir}"]
    try:
        raw, database_info = load_joined_database(database)
        frame = prepare_analysis_frame(raw, completion_cutoff=completion_cutoff)
        frame, marketization_info, merge_warnings = merge_marketization(frame, marketization_csv, target_year)
        manifest["database_info"] = database_info
        manifest["marketization"] = marketization_info
        manifest["x1_int_winsor_limits"] = frame.attrs.get("x1_int_winsor_limits")
        warnings = [*merge_warnings, *frame.attrs.get("numeric_conversion_warnings", [])]
        validation = codebook_validation(frame)
        manifest["codebook_validation"] = validation
        warnings.extend(validation["value_warnings"])
        interpretation_status = (
            "exploratory_only_codebook_review_required"
            if validation["status"] == "review_required"
            else "definitions_confirmed"
        )
        manifest["interpretation_status"] = interpretation_status
        if validation["status"] == "review_required":
            warnings.append("inferential_outputs_are_exploratory_until_codebook_is_confirmed")
        (dirs["diagnostics"] / "codebook_validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_table(
            pd.DataFrame(
                [
                    {
                        "main_rows": database_info["main_rows"],
                        "supplemental_rows": database_info["supplemental_rows"],
                        "joined_rows": database_info["joined_rows"],
                        "unmatched_rows": 0,
                        "join_key": "code",
                        "validation": "one_to_one_passed",
                    }
                ]
            ),
            dirs["diagnostics"] / "merge_report.csv",
        )
        _write_table(frame, dirs["diagnostics"] / "analysis_frame.csv")
        _write_table(missingness_report(frame), dirs["diagnostics"] / "supplement_missingness.csv")
        _write_table(selection_check_report(frame), dirs["diagnostics"] / "selection_check.csv")
        descriptive_fields = [
            "y",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "O_rate",
            "completion_stage",
            "months_overdue",
            "offset_discount",
            "gov_intervention",
        ]
        descriptive = frame[descriptive_fields].describe().T.reset_index(names="variable")
        _write_table(descriptive, dirs["tables"] / "supplemental_descriptives.csv")
        government_summary = (
            frame.groupby("gov_intervention", dropna=False)
            .agg(
                n=("code", "size"),
                full_overdue_incidence=("full_overdue_incidence", "mean"),
                mean_O_rate=("O_rate", "mean"),
                mean_completion_stage=("completion_stage", "mean"),
            )
            .reset_index()
        )
        _write_table(
            government_summary,
            dirs["tables"] / "gov_intervention_analysis.csv",
        )
        _write_table(
            frame.groupby("firm_id", dropna=False)
            .agg(
                n=("code", "size"),
                full_overdue_incidence=("full_overdue_incidence", "mean"),
                mean_O_rate=("O_rate", "mean"),
                mean_completion_stage=("completion_stage", "mean"),
            )
            .reset_index(),
            dirs["tables"] / "firm_id_summary.csv",
        )
        duration_summary = (
            frame.groupby("months_overdue", dropna=False)
            .agg(
                n=("code", "size"),
                full_overdue_incidence=("full_overdue_incidence", "mean"),
                mean_O_rate=("O_rate", "mean"),
                mean_completion_stage=("completion_stage", "mean"),
            )
            .reset_index()
        )
        _write_table(duration_summary, dirs["tables"] / "duration_robustness.csv")
        agreement = pd.crosstab(
            frame["full_overdue_incidence"], frame["completion_low_proxy"], dropna=False
        ).reset_index()
        _write_table(agreement, dirs["tables"] / "full_overdue_proxy_agreement.csv")
        model_terms, model_metadata, model_warnings = run_models(
            frame, output_dir, bootstrap_repetitions, seed
        )
        warnings.extend(model_warnings)
        if not model_terms.empty:
            model_terms["interpretation_status"] = interpretation_status
            _write_table(model_terms, dirs["tables"] / "model_terms.csv")
        _write_table(
            model_terms.loc[
                model_terms["model"].isin(
                    ["full_overdue_logit_base", "full_overdue_logit_supplement", "severity_fractional_logit"]
                )
            ],
            dirs["tables"] / "two_part_model.csv",
        )
        figure_outputs = run_supplemental_figures(frame, model_terms, output_dir)
        manifest["rows"] = {
            "raw_joined": len(raw),
            "analysis": len(frame),
            "severity_below_full_overdue": int((frame["full_overdue_incidence"] == 0).sum()),
        }
        manifest["model_metadata"] = model_metadata
        manifest["figures"] = {name: str(path) if path is not None else None for name, path in figure_outputs.items()}
        manifest["warnings"] = warnings
        manifest["status"] = "completed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_text_log(output_dir, "supplemental_analysis.log", log_lines + ["status=completed"])
        manifest["outputs"] = sorted(
            str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()
        )
        write_run_manifest(output_dir, manifest)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["traceback"] = traceback.format_exc()
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_run_manifest(output_dir, manifest)
        write_text_log(output_dir, "supplemental_analysis.log", log_lines + ["status=failed", f"error={error!r}"])
        return 1


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the supplemental-variable research analysis.")
    parser.add_argument(
        "--database",
        type=Path,
        default=project_root / "data" / "current" / "disorder_payment_data.db",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "supplemental",
    )
    parser.add_argument(
        "--completion-cutoff",
        type=float,
        default=2.0,
        help="Low-stage cutoff for the robustness proxy; default 2.",
    )
    parser.add_argument("--target-year", type=int, default=2023)
    parser.add_argument("--marketization-csv", type=Path, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=199)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_supplemental_analysis(
        args.database,
        args.output_dir,
        completion_cutoff=args.completion_cutoff,
        target_year=args.target_year,
        marketization_csv=args.marketization_csv,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
