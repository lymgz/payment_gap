"""Propensity-score matching and covariate balance diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from .reporting import ensure_output_dirs


DEFAULT_COVARIATES = ["x1", "x3", "x4", "x5", "x6", "x7"]


def fit_propensity_scores(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    seed: int = 123,
) -> pd.DataFrame:
    """Estimate treatment probabilities and return a copy with diagnostics."""

    required = [treatment, *covariates, "color"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for propensity model: {missing}")
    result = df.copy(deep=True)
    usable = result[required].notna().all(axis=1)
    model_frame = result.loc[usable, required].copy()
    color_dummies = pd.get_dummies(model_frame["color"], prefix="color", drop_first=True, dtype=float)
    for column in ("color_2", "color_3"):
        result[column] = np.nan
        if column in color_dummies.columns:
            result.loc[usable, column] = color_dummies[column]
        else:
            result.loc[usable, column] = 0.0
    design = pd.concat([model_frame[covariates].astype(float), color_dummies], axis=1)
    design = sm.add_constant(design, has_constant="add")
    treatment_values = model_frame[treatment].astype(int)
    warnings: list[str] = []
    try:
        result_fit = sm.Logit(treatment_values, design).fit(disp=False, maxiter=300)
        scores = result_fit.predict(design)
    except Exception as error:
        warnings.append(f"statsmodels_logit_fallback={error}")
        fallback = LogisticRegression(random_state=seed, max_iter=2000)
        fallback.fit(design.drop(columns="const"), treatment_values)
        scores = fallback.predict_proba(design.drop(columns="const"))[:, 1]
    result["propensity_score"] = np.nan
    result.loc[usable, "propensity_score"] = np.asarray(scores, dtype=float)
    result.attrs["psm_warnings"] = warnings
    return result


def nearest_neighbor_match(
    df: pd.DataFrame,
    treatment: str,
    score: str,
    neighbors: int,
    caliper: float | None = None,
    seed: int = 123,
) -> pd.DataFrame:
    """Match each treated unit to nearest controls, allowing control reuse."""

    del seed  # The deterministic score ordering makes this operation reproducible.
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1")
    usable = df[treatment].isin([0, 1]) & df[score].notna()
    working = df.loc[usable].copy()
    treated = working.loc[working[treatment] == 1]
    controls = working.loc[working[treatment] == 0]
    if treated.empty or controls.empty:
        return pd.DataFrame(
            columns=[
                "source_index",
                "source_position",
                "treatment",
                "propensity_score",
                "match_id",
                "neighbor_rank",
                "matched",
                "match_weight",
            ]
        )

    lower = max(float(treated[score].min()), float(controls[score].min()))
    upper = min(float(treated[score].max()), float(controls[score].max()))
    treated = treated.loc[treated[score].between(lower, upper)]
    controls = controls.loc[controls[score].between(lower, upper)]
    control_scores = controls[score].to_numpy(dtype=float)
    positions = {index: int(position) for position, index in enumerate(df.index)}
    rows: list[dict[str, Any]] = []
    match_id = 0
    for treated_index, treated_row in treated.sort_values(score).iterrows():
        distances = np.abs(control_scores - float(treated_row[score]))
        order = np.argsort(distances, kind="stable")
        if caliper is not None:
            order = order[distances[order] <= caliper]
        order = order[:neighbors]
        if len(order) == 0:
            continue
        match_id += 1
        rows.append(
            {
                "source_index": treated_index,
                "source_position": positions[treated_index],
                "treatment": 1,
                "propensity_score": float(treated_row[score]),
                "match_id": match_id,
                "neighbor_rank": 0,
                "matched": True,
                "match_weight": 1.0,
            }
        )
        for rank, control_position in enumerate(order, start=1):
            control_index = controls.index[control_position]
            control_row = controls.iloc[control_position]
            rows.append(
                {
                    "source_index": control_index,
                    "source_position": positions[control_index],
                    "treatment": 0,
                    "propensity_score": float(control_row[score]),
                    "match_id": match_id,
                    "neighbor_rank": rank,
                    "matched": True,
                    "match_weight": 1.0 / len(order),
                }
            )
    return pd.DataFrame(rows)


def _weighted_mean_sd(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    clean = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "weight": weights}).dropna()
    if clean.empty or clean["weight"].sum() <= 0:
        return np.nan, np.nan
    mean = float(np.average(clean["value"], weights=clean["weight"]))
    variance = float(np.average((clean["value"] - mean) ** 2, weights=clean["weight"]))
    return mean, float(np.sqrt(max(variance, 0.0)))


def standardized_mean_differences(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    weights: str | None = None,
    stage: str = "before",
) -> pd.DataFrame:
    """Calculate treated/control standardized mean differences."""

    weight_values = pd.Series(1.0, index=df.index) if weights is None else pd.to_numeric(df[weights], errors="coerce")
    rows: list[dict[str, Any]] = []
    for covariate in covariates:
        if covariate not in df.columns:
            continue
        treated_mask = df[treatment] == 1
        control_mask = df[treatment] == 0
        treated_mean, treated_sd = _weighted_mean_sd(df.loc[treated_mask, covariate], weight_values.loc[treated_mask])
        control_mean, control_sd = _weighted_mean_sd(df.loc[control_mask, covariate], weight_values.loc[control_mask])
        pooled_sd = np.sqrt((treated_sd**2 + control_sd**2) / 2) if np.isfinite(treated_sd + control_sd) else np.nan
        smd = (treated_mean - control_mean) / pooled_sd if pooled_sd and np.isfinite(pooled_sd) else np.nan
        rows.append(
            {
                "stage": stage,
                "covariate": covariate,
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "smd": smd,
                "abs_smd": abs(smd) if np.isfinite(smd) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _weighted_outcome(frame: pd.DataFrame, treatment: int) -> float:
    subset = frame.loc[frame["x2"] == treatment]
    if subset.empty:
        return np.nan
    return float(np.average(subset["y"], weights=subset["match_weight"]))


def run_psm_suite(df: pd.DataFrame, output_dir: Path, seed: int = 123) -> dict[str, Path]:
    """Run 1:2 and 1:5 matching and write balance/treatment diagnostics."""

    dirs = ensure_output_dirs(Path(output_dir))
    scored = fit_propensity_scores(df, "x2", DEFAULT_COVARIATES, seed=seed)
    covariates = [*DEFAULT_COVARIATES, "color_2", "color_3"]
    balance_rows = [standardized_mean_differences(scored, "x2", covariates)]
    summary_rows: list[dict[str, Any]] = []
    match_paths: dict[str, Path] = {}
    warnings_rows = [{"model": "propensity_score", "warning": warning} for warning in scored.attrs.get("psm_warnings", [])]

    for neighbors in (2, 5):
        matches = nearest_neighbor_match(scored, "x2", "propensity_score", neighbors=neighbors, seed=seed)
        match_path = dirs["diagnostics"] / f"psm_matches_neighbors_{neighbors}.csv"
        matches.to_csv(match_path, index=False, encoding="utf-8-sig")
        match_paths[f"matches_{neighbors}"] = match_path
        if matches.empty:
            warnings_rows.append({"model": f"psm_neighbors_{neighbors}", "warning": "no matched units"})
            continue
        matched = scored.iloc[matches["source_position"].astype(int).to_numpy()].copy()
        matched["match_weight"] = matches["match_weight"].to_numpy()
        after = standardized_mean_differences(
            matched,
            "x2",
            covariates,
            weights="match_weight",
            stage=f"after_{neighbors}",
        )
        balance_rows.append(after)
        summary_rows.append(
            {
                "neighbors": neighbors,
                "treated_matched_n": int((matches["treatment"] == 1).sum()),
                "control_matched_rows": int((matches["treatment"] == 0).sum()),
                "unique_control_n": int(matches.loc[matches["treatment"] == 0, "source_index"].nunique()),
                "propensity_min": float(matches["propensity_score"].min()),
                "propensity_max": float(matches["propensity_score"].max()),
                "weighted_treated_y": _weighted_outcome(matched, 1),
                "weighted_control_y": _weighted_outcome(matched, 0),
                "weighted_difference": _weighted_outcome(matched, 1) - _weighted_outcome(matched, 0),
            }
        )

    outputs = {
        "summary": dirs["tables"] / "psm_summary.csv",
        "balance": dirs["tables"] / "psm_balance.csv",
        "warnings": dirs["diagnostics"] / "psm_warnings.csv",
        **match_paths,
    }
    pd.DataFrame(summary_rows).to_csv(outputs["summary"], index=False, encoding="utf-8-sig")
    pd.concat(balance_rows, ignore_index=True).to_csv(outputs["balance"], index=False, encoding="utf-8-sig")
    pd.DataFrame(warnings_rows, columns=["model", "warning"]).to_csv(
        outputs["warnings"], index=False, encoding="utf-8-sig"
    )
    return outputs
