"""Regression, censored-regression, and subgroup model specifications."""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_hess

from .reporting import ensure_output_dirs


BASE_TERMS = ["x1", "x3", "x4", "x5", "x6", "x7", "C(x2)"]


@dataclass
class ModelFit:
    """Small common result object for statsmodels and custom Tobit fits."""

    name: str
    params: pd.Series
    bse: pd.Series
    statistics: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    nobs: int
    warning: str = ""
    converged: bool | None = None


def baseline_formula(include_color: bool = True) -> tuple[str, list[str]]:
    terms = [*BASE_TERMS]
    if include_color:
        terms.append("C(color)")
    return "y ~ " + " + ".join(terms), terms


def _formula_frame(df: pd.DataFrame, outcome: str, terms: list[str], extra: list[str] | None = None) -> pd.DataFrame:
    columns = [outcome, *[term[2:-1] if term.startswith("C(") else term for term in terms], *(extra or [])]
    columns = list(dict.fromkeys(columns))
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for model: {missing}")
    return df[columns].copy().dropna()


def _series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(float)
    return pd.Series(np.asarray(value, dtype=float), index=index)


def _statsmodels_fit(
    name: str,
    result: Any,
    warning_messages: list[str],
    nobs: int,
) -> ModelFit:
    params = _series(result.params, result.params.index)
    bse = _series(result.bse, params.index)
    statistic = _series(getattr(result, "tvalues", params / bse), params.index)
    pvalues = _series(result.pvalues, params.index)
    intervals = result.conf_int()
    conf_int = pd.DataFrame(np.asarray(intervals, dtype=float), index=params.index, columns=["low", "high"])
    converged = getattr(result, "converged", None)
    if converged is False:
        warning_messages.append("optimizer did not report convergence")
    return ModelFit(
        name=name,
        params=params,
        bse=bse,
        statistics=statistic,
        pvalues=pvalues,
        conf_int=conf_int,
        nobs=int(nobs),
        warning="; ".join(dict.fromkeys(warning_messages)),
        converged=converged,
    )


def fit_gamma_glm(
    df: pd.DataFrame,
    predictors: list[str],
    cluster: str | None = "Region",
    outcome: str = "y",
    name: str = "gamma_glm",
) -> ModelFit:
    """Fit a Gamma-log GLM, using region-clustered covariance when possible."""

    frame = _formula_frame(df, outcome, predictors, [cluster] if cluster else None)
    warnings_found: list[str] = []
    nonpositive = int((pd.to_numeric(frame[outcome], errors="coerce") <= 0).sum())
    if nonpositive:
        frame = frame.loc[frame[outcome] > 0].copy()
        warnings_found.append(f"dropped_nonpositive_{outcome}={nonpositive}")
    design = patsy.dmatrix("1 + " + " + ".join(predictors), data=frame, return_type="dataframe")
    if len(frame) < 2 * design.shape[1]:
        raise ValueError(
            f"underidentified_model: n_obs={len(frame)} is below twice the parameter count {design.shape[1]}"
        )
    formula = f"{outcome} ~ " + " + ".join(predictors)
    model = smf.glm(formula=formula, data=frame, family=sm.families.Gamma(link=sm.families.links.Log()))
    with python_warnings.catch_warnings(record=True) as captured:
        python_warnings.simplefilter("always")
        if cluster and cluster in frame.columns and frame[cluster].nunique() >= 2:
            result = model.fit(cov_type="cluster", cov_kwds={"groups": frame[cluster]})
        else:
            if cluster:
                warnings_found.append(f"insufficient_clusters_for_{cluster}")
            result = model.fit()
        warnings_found.extend(str(item.message) for item in captured)
    return _statsmodels_fit(name, result, warnings_found, len(frame))


def fit_clustered_ols(
    df: pd.DataFrame,
    predictors: list[str],
    cluster: str = "Region",
    outcome: str = "y",
    name: str = "ols_clustered",
) -> ModelFit:
    """Fit the OLS robustness model with clustered covariance."""

    frame = _formula_frame(df, outcome, predictors, [cluster])
    formula = f"{outcome} ~ " + " + ".join(predictors)
    model = smf.ols(formula=formula, data=frame)
    warnings_found: list[str] = []
    with python_warnings.catch_warnings(record=True) as captured:
        python_warnings.simplefilter("always")
        if frame[cluster].nunique() >= 2:
            result = model.fit(cov_type="cluster", cov_kwds={"groups": frame[cluster]})
        else:
            warnings_found.append(f"insufficient_clusters_for_{cluster}")
            result = model.fit()
        warnings_found.extend(str(item.message) for item in captured)
    return _statsmodels_fit(name, result, warnings_found, len(frame))


def fit_mixed_model(
    df: pd.DataFrame,
    predictors: list[str],
    group: str = "Region",
    outcome: str = "y",
    name: str = "mixed_region",
    variance_component: str | None = None,
) -> ModelFit:
    """Fit a random-intercept model with an optional variance component."""

    extra = [group, variance_component] if variance_component else [group]
    frame = _formula_frame(df, outcome, predictors, extra)
    formula = f"{outcome} ~ " + " + ".join(predictors)
    vc_formula = None
    if variance_component:
        vc_formula = {variance_component: f"0 + C({variance_component})"}
    model = smf.mixedlm(formula=formula, data=frame, groups=frame[group], vc_formula=vc_formula)
    warnings_found: list[str] = []
    with python_warnings.catch_warnings(record=True) as captured:
        python_warnings.simplefilter("always")
        try:
            result = model.fit(reml=False, method="lbfgs", disp=False)
        except Exception as first_error:
            warnings_found.append(f"lbfgs_failed={first_error}")
            result = model.fit(reml=False, method="powell", disp=False)
        warnings_found.extend(str(item.message) for item in captured)
    return _statsmodels_fit(name, result, warnings_found, len(frame))


def fit_tobit(
    df: pd.DataFrame,
    predictors: list[str],
    lower: float = 0.0,
    upper: float | None = None,
    outcome: str | None = None,
    name: str = "tobit",
    cluster: str | None = None,
) -> ModelFit:
    """Fit a censored normal regression with observed-Hessian inference.

    The likelihood is optimized with L-BFGS-B. Standard errors are based on
    the observed numerical Hessian, with an optional cluster-robust sandwich
    covariance based on per-observation scores. This avoids treating the
    optimizer's limited-memory inverse-Hessian approximation as inference.
    """

    target = outcome or ("O_rate" if "O_rate" in df.columns else "y2")
    if target not in df.columns:
        raise KeyError("Tobit requires O_rate or y2 as the outcome")
    frame = _formula_frame(df, target, predictors, [cluster] if cluster else None)
    y_design, design = patsy.dmatrices(
        f"{target} ~ " + " + ".join(predictors),
        data=frame,
        return_type="dataframe",
    )
    y = y_design.iloc[:, 0].to_numpy(dtype=float)
    x = design.to_numpy(dtype=float)
    names = pd.Index(design.columns)
    if len(y) <= x.shape[1]:
        raise ValueError("Tobit requires more observations than design columns")

    left = y <= lower
    right = np.zeros_like(left, dtype=bool) if upper is None else y >= upper
    uncensored = ~(left | right)
    clipped = np.clip(y, lower + 1e-6, (upper - 1e-6) if upper is not None else None)
    beta_start = np.linalg.lstsq(x, clipped, rcond=None)[0]
    sigma_start = max(float(np.std(clipped - x @ beta_start)), 1e-3)
    initial = np.r_[beta_start, np.log(sigma_start)]

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        beta = parameters[:-1]
        sigma = float(np.exp(np.clip(parameters[-1], -20, 20)))
        mu = x @ beta
        z_left = (lower - mu) / sigma
        value = np.zeros_like(y, dtype=float)
        value[left] = stats.norm.logcdf(z_left[left])
        if upper is not None:
            z_right = (upper - mu) / sigma
            value[right] = stats.norm.logsf(z_right[right])
        value[uncensored] = stats.norm.logpdf(y[uncensored], loc=mu[uncensored], scale=sigma)
        if not np.isfinite(value).all():
            return 1e100
        return float(-value.sum())

    def observation_scores(parameters: np.ndarray) -> np.ndarray:
        beta = parameters[:-1]
        sigma = float(np.exp(np.clip(parameters[-1], -20, 20)))
        mu = x @ beta
        scores = np.zeros((len(y), len(parameters)), dtype=float)
        if uncensored.any():
            z = (y[uncensored] - mu[uncensored]) / sigma
            scores[uncensored, :-1] = x[uncensored] * (z / sigma)[:, None]
            scores[uncensored, -1] = -1.0 + z**2
        if left.any():
            z = (lower - mu[left]) / sigma
            mills = np.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z))
            scores[left, :-1] = x[left] * (-mills / sigma)[:, None]
            scores[left, -1] = -z * mills
        if right.any():
            z = (upper - mu[right]) / sigma
            mills = np.exp(stats.norm.logpdf(z) - stats.norm.logsf(z))
            scores[right, :-1] = x[right] * (mills / sigma)[:, None]
            scores[right, -1] = z * mills
        return scores

    with python_warnings.catch_warnings(record=True) as captured:
        python_warnings.simplefilter("always")
        result = optimize.minimize(
            negative_log_likelihood,
            initial,
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
        )
    warning_messages = [str(item.message) for item in captured]
    if not result.success:
        warning_messages.append(str(result.message))

    params = pd.Series(result.x[:-1], index=names, dtype=float)
    try:
        hessian = np.asarray(approx_hess(result.x, negative_log_likelihood), dtype=float)
        if not np.isfinite(hessian).all():
            raise ValueError("non-finite observed Hessian")
        bread = np.linalg.pinv((hessian + hessian.T) / 2)
        covariance = bread[:-1, :-1]
        if cluster and cluster in frame.columns:
            clusters = frame.loc[y_design.index, cluster].to_numpy()
            unique_clusters = pd.unique(clusters)
            if len(unique_clusters) >= 2:
                scores = observation_scores(result.x)
                cluster_scores = np.vstack(
                    [scores[clusters == label].sum(axis=0) for label in unique_clusters]
                )
                meat = cluster_scores.T @ cluster_scores
                covariance_full = bread @ meat @ bread
                correction = len(unique_clusters) / (len(unique_clusters) - 1)
                correction *= len(y) / max(len(y) - len(params), 1)
                covariance = covariance_full[:-1, :-1] * correction
            else:
                warning_messages.append(f"insufficient_clusters_for_{cluster}")
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        warning_messages.append(f"hessian_inference_failed={error}")
        standard_errors = np.full(len(params), np.nan)
    bse = pd.Series(standard_errors, index=names, dtype=float)
    statistic = params / bse.replace(0, np.nan)
    pvalues = pd.Series(2 * stats.norm.sf(np.abs(statistic)), index=names, dtype=float)
    conf_int = pd.DataFrame({"low": params - 1.96 * bse, "high": params + 1.96 * bse})
    return ModelFit(
        name=name,
        params=params,
        bse=bse,
        statistics=statistic,
        pvalues=pvalues,
        conf_int=conf_int,
        nobs=int(len(y)),
        warning="; ".join(dict.fromkeys(warning_messages)),
        converged=bool(result.success),
    )


def tidy_model_fit(fit: ModelFit) -> pd.DataFrame:
    """Convert one model result into a machine-readable coefficient table."""

    rows: list[dict[str, Any]] = []
    for term in fit.params.index:
        rows.append(
            {
                "model": fit.name,
                "term": term,
                "estimate": float(fit.params.loc[term]),
                "std_error": float(fit.bse.loc[term]),
                "statistic": float(fit.statistics.loc[term]),
                "p_value": float(fit.pvalues.loc[term]),
                "conf_low": float(fit.conf_int.loc[term, "low"]),
                "conf_high": float(fit.conf_int.loc[term, "high"]),
                "n_obs": fit.nobs,
                "converged": fit.converged,
                "warning": fit.warning,
            }
        )
    return pd.DataFrame(rows)


def _safe_fit(
    name: str,
    function: Callable[[], ModelFit],
    warnings_rows: list[dict[str, Any]],
    n_obs: int | None = None,
) -> ModelFit | None:
    try:
        fit = function()
        if fit.warning:
            warnings_rows.append({"model": name, "status": "warning", "n_obs": fit.nobs, "warning": fit.warning})
        return fit
    except Exception as error:
        warnings_rows.append({"model": name, "status": "failed", "n_obs": n_obs, "warning": f"model_failed={error!r}"})
        return None


def _write_models(fits: list[ModelFit], path: Path) -> Path:
    columns = [
        "model",
        "term",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "conf_low",
        "conf_high",
        "n_obs",
        "converged",
        "warning",
    ]
    table = pd.concat([tidy_model_fit(fit) for fit in fits], ignore_index=True) if fits else pd.DataFrame(columns=columns)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_regression_suite(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Run the supported baseline, validation, and subgroup specifications."""

    dirs = ensure_output_dirs(Path(output_dir))
    warnings_rows: list[dict[str, Any]] = []
    baseline_terms = [*BASE_TERMS, "C(color)"]
    baseline_fits: list[ModelFit] = []
    fit = _safe_fit("glm_baseline", lambda: fit_gamma_glm(df, baseline_terms, name="glm_baseline"), warnings_rows, len(df))
    if fit:
        baseline_fits.append(fit)
    fit = _safe_fit("ols_baseline", lambda: fit_clustered_ols(df, baseline_terms, name="ols_baseline"), warnings_rows, len(df))
    if fit:
        baseline_fits.append(fit)
    fit = _safe_fit("mixed_region", lambda: fit_mixed_model(df, baseline_terms, name="mixed_region"), warnings_rows, len(df))
    if fit:
        baseline_fits.append(fit)
    fit = _safe_fit(
        "mixed_region_color",
        lambda: fit_mixed_model(
            df,
            baseline_terms,
            name="mixed_region_color",
            variance_component="color_text",
        ),
        warnings_rows,
        len(df),
    )
    if fit:
        baseline_fits.append(fit)

    tobit_specs = [
        ("tobit_r0", [*BASE_TERMS[:-1]]),
        ("tobit_r1", [*BASE_TERMS[:-1], "C(color)"]),
        ("tobit_r2", [*BASE_TERMS[:-1], "C(x2)", "C(color)"]),
        ("tobit_r3", ["x1", "x3", "x4", "C(x2)", "C(color)"]),
        ("tobit_upper1", [*BASE_TERMS[:-1], "C(x2)", "C(color)"]),
    ]
    tobit_fits: list[ModelFit] = []
    for model_name, terms in tobit_specs:
        upper = 1.0 if model_name == "tobit_upper1" else None
        fit = _safe_fit(
            model_name,
            lambda model_name=model_name, terms=terms, upper=upper: fit_tobit(
                df, terms, lower=0.0, upper=upper, name=model_name, cluster="Region"
            ),
            warnings_rows,
            len(df),
        )
        if fit:
            tobit_fits.append(fit)

    subgroup_fits: list[ModelFit] = []
    for color, label in ((1, "red"), (2, "yellow"), (3, "green")):
        model_name = f"color_{label}"
        fit = _safe_fit(
            model_name,
            lambda model_name=model_name, color=color: fit_gamma_glm(
                df.loc[df["color"] == color], BASE_TERMS, name=model_name
            ),
            warnings_rows,
            int((df["color"] == color).sum()),
        )
        if fit:
            subgroup_fits.append(fit)
    for region in ("Central area", "Eastern area", "Western area"):
        model_name = f"region_{region.split()[0].lower()}"
        fit = _safe_fit(
            model_name,
            lambda model_name=model_name, region=region: fit_gamma_glm(
                df.loc[df["Region"] == region], BASE_TERMS, cluster="color", name=model_name
            ),
            warnings_rows,
            int((df["Region"] == region).sum()),
        )
        if fit:
            subgroup_fits.append(fit)

    outputs = {
        "baseline": _write_models(baseline_fits, dirs["tables"] / "regression_baseline.csv"),
        "tobit": _write_models(tobit_fits, dirs["tables"] / "regression_tobit.csv"),
        "subgroups": _write_models(subgroup_fits, dirs["tables"] / "regression_subgroups.csv"),
        "warnings": dirs["diagnostics"] / "model_warnings.csv",
        "workbook": dirs["tables"] / "regression_models.xlsx",
    }
    with pd.ExcelWriter(outputs["workbook"], engine="openpyxl") as writer:
        pd.read_csv(outputs["baseline"]).to_excel(writer, sheet_name="baseline", index=False)
        pd.read_csv(outputs["tobit"]).to_excel(writer, sheet_name="tobit", index=False)
        pd.read_csv(outputs["subgroups"]).to_excel(writer, sheet_name="subgroups", index=False)
    pd.DataFrame(warnings_rows, columns=["model", "status", "n_obs", "warning"]).to_csv(
        outputs["warnings"], index=False, encoding="utf-8-sig"
    )
    return outputs
