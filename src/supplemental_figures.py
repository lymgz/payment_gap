"""Script-generated figures for the supplemental-variable analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .reporting import ensure_output_dirs


FIGURE_DPI = 300
VIRIDIS = plt.get_cmap("viridis")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": FIGURE_DPI,
    }
)


def _viridis_colors(n: int, start: float = 0.18, stop: float = 0.88) -> list[Any]:
    """Return evenly spaced, print-readable colors from the viridis palette."""

    return [VIRIDIS(value) for value in np.linspace(start, stop, max(n, 1))]


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def summarize_two_part_outcome(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Summarize incidence and conditional severity without mixing the margins."""

    incidence = (
        frame.groupby(group, dropna=False)
        .agg(
            n=("full_overdue_incidence", "size"),
            full_overdue_share=("full_overdue_incidence", "mean"),
        )
        .reset_index()
    )
    severity = (
        frame.loc[frame["full_overdue_incidence"] == 0]
        .groupby(group, dropna=False)
        .agg(
            below_full_overdue_n=("O_rate", "size"),
            below_full_overdue_mean_severity=("O_rate", "mean"),
        )
        .reset_index()
    )
    return incidence.merge(severity, on=group, how="left").fillna(
        {"below_full_overdue_n": 0, "below_full_overdue_mean_severity": np.nan}
    )


def plot_completion_full_overdue(frame: pd.DataFrame, path: Path) -> Path:
    summary = (
        frame.groupby("completion_stage", dropna=False)
        .agg(n=("code", "size"), full_overdue_share=("full_overdue_incidence", "mean"))
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.bar(
        summary["completion_stage"].astype(str),
        summary["full_overdue_share"],
        color=_viridis_colors(len(summary)),
    )
    for index, row in summary.iterrows():
        axis.text(index, row["full_overdue_share"] + 0.02, f"n={int(row['n'])}", ha="center", fontsize=9)
    axis.set_ylim(0, 1.08)
    axis.set_xlabel("Completion-stage code (raw workbook code)")
    axis.set_ylabel("Share with overdue rate >= 0.9999")
    axis.set_title("Full-overdue proxy by completion-stage code")
    axis.grid(axis="y", alpha=0.2)
    return _save(fig, path)


def plot_government_intervention(frame: pd.DataFrame, path: Path) -> Path:
    summary = summarize_two_part_outcome(frame, "gov_intervention")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.8), sharex=True)
    x = np.arange(len(summary))
    labels = summary["gov_intervention"].astype(str)
    axes[0].bar(x, summary["full_overdue_share"], color=VIRIDIS(0.28))
    axes[1].bar(x, summary["below_full_overdue_mean_severity"], color=VIRIDIS(0.72))
    for index, row in summary.iterrows():
        axes[0].text(
            index,
            row["full_overdue_share"] + 0.025,
            f"{row['full_overdue_share']:.0%}\nn={int(row['n'])}",
            ha="center",
            fontsize=8,
        )
        if np.isfinite(row["below_full_overdue_mean_severity"]):
            axes[1].text(
                index,
                row["below_full_overdue_mean_severity"] + 0.025,
                f"{row['below_full_overdue_mean_severity']:.2f}\nn={int(row['below_full_overdue_n'])}",
                ha="center",
                fontsize=8,
            )
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_ylim(0, 1.08)
        axis.set_xlabel("Government-intervention code")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Share with overdue rate >= 0.9999")
    axes[0].set_title("Incidence margin: all projects")
    axes[1].set_ylabel("Mean overdue rate")
    axes[1].set_title("Severity margin: below full overdue only")
    fig.suptitle("Two-part overdue outcomes by government intervention")
    fig.tight_layout()
    return _save(fig, path)


def plot_duration_severity(frame: pd.DataFrame, path: Path) -> Path:
    data = frame.loc[frame["full_overdue_incidence"] == 0, ["months_overdue", "O_rate"]].dropna()
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    groups = []
    labels = []
    for level in sorted(data["months_overdue"].unique()):
        groups.append(data.loc[data["months_overdue"] == level, "O_rate"])
        labels.append(str(int(level)))
    if groups:
        box = axis.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": VIRIDIS(0.95), "linewidth": 1.5},
        )
        for patch, color in zip(box["boxes"], _viridis_colors(len(groups))):
            patch.set_facecolor(color)
    axis.set_xlabel("Overdue-duration code (1–4; not actual months)")
    axis.set_ylabel("Overdue rate below the full-overdue mass point")
    axis.set_title("Severity margin by overdue-duration category")
    axis.grid(axis="y", alpha=0.2)
    return _save(fig, path)


def plot_missingness(frame: pd.DataFrame, path: Path) -> Path:
    fields = [
        "firm_id",
        "completion_pct",
        "months_overdue",
        "offset_discount",
        "gov_intervention",
        "contractor_litigation",
        "contractor_soe",
    ]
    observed = [float(frame[field].notna().mean() * 100) for field in fields]
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    bars = axis.bar(fields, observed, color=VIRIDIS(0.65))
    axis.bar_label(bars, labels=[f"{value:.0f}%" for value in observed], padding=2, fontsize=8)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Observed rows (%)")
    axis.set_title("Supplemental-variable coverage diagnostic")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.2)
    return _save(fig, path)


def plot_definition_agreement(frame: pd.DataFrame, path: Path) -> Path:
    table = pd.crosstab(frame["full_overdue_incidence"], frame["completion_low_proxy"], dropna=False)
    table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(table.to_numpy(), cmap="viridis")
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            axis.text(col, row, int(table.iloc[row, col]), ha="center", va="center", fontsize=13)
    axis.set_xticks([0, 1], ["Not low-stage", "Low-stage"])
    axis.set_yticks([0, 1], ["Below full overdue", "Full overdue proxy"])
    axis.set_xlabel("Completion-stage robustness proxy")
    axis.set_ylabel("O_rate >= 0.9999 proxy")
    axis.set_title("Agreement of two exploratory risk proxies")
    fig.colorbar(image, ax=axis, shrink=0.78, label="Number of projects")
    return _save(fig, path)


def plot_firm_outcomes(frame: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.8))
    firm_levels = sorted(frame["firm_id"].dropna().unique())
    severity = frame.loc[frame["full_overdue_incidence"] == 0]
    groups = [severity.loc[severity["firm_id"] == level, "O_rate"].dropna() for level in firm_levels]
    if groups:
        box = axes[0].boxplot(
            groups,
            tick_labels=[str(int(x)) for x in firm_levels],
            patch_artist=True,
            showfliers=False,
            medianprops={"color": VIRIDIS(0.95), "linewidth": 1.5},
        )
        for patch, color in zip(box["boxes"], _viridis_colors(len(groups))):
            patch.set_facecolor(color)
    axes[0].set_xlabel("firm_id code")
    axes[0].set_ylabel("Overdue rate")
    axes[0].set_title("Severity: below full overdue")
    summary = summarize_two_part_outcome(frame, "firm_id")
    bars = axes[1].bar(summary["firm_id"].astype(str), summary["full_overdue_share"], color=VIRIDIS(0.7))
    axes[1].bar_label(
        bars,
        labels=[f"{value:.0%}" for value in summary["full_overdue_share"]],
        padding=2,
        fontsize=8,
    )
    axes[1].set_ylim(0, 1.08)
    axes[1].set_xlabel("firm_id code")
    axes[1].set_ylabel("Share with overdue rate >= 0.9999")
    axes[1].set_title("Incidence: all projects")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _save(fig, path)


def plot_model_effects(model_terms: pd.DataFrame, path: Path) -> Path | None:
    if model_terms.empty:
        return None
    selected = model_terms.loc[
        model_terms["term"].astype(str).str.contains(
            "completion_stage|gov_intervention|contractor_litigation|contractor_soe", regex=True
        )
        & model_terms["model"].isin(["full_overdue_logit_supplement", "severity_fractional_logit"])
    ].copy()
    selected = selected.loc[np.isfinite(selected["std_error"])].copy()
    if selected.empty:
        return None
    selected = selected.sort_values(["model", "term"])
    model_order = list(selected["model"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(model_order),
        figsize=(7.4, max(5.5, min(10.2, selected.groupby("model").size().max() * 0.42 + 1.2))),
        squeeze=False,
    )
    for axis, model_name in zip(axes[0], model_order):
        subset = selected.loc[selected["model"] == model_name].copy()
        y = np.arange(len(subset))
        axis.errorbar(
            subset["coefficient"],
            y,
            xerr=1.96 * subset["std_error"].fillna(0),
            fmt="o",
            color=VIRIDIS(0.72),
            ecolor=VIRIDIS(0.35),
            capsize=3,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y, subset["term"])
        axis.set_xlabel("Coefficient with 95% CI")
        axis.set_title(model_name.replace("_", " "))
        axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _save(fig, path)


def run_supplemental_figures(
    frame: pd.DataFrame,
    model_terms: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path | None]:
    """Generate all planned figures and record figure-level failures."""

    dirs = ensure_output_dirs(Path(output_dir))
    specs = {
        "completion_full_overdue": lambda path: plot_completion_full_overdue(frame, path),
        "government_intervention": lambda path: plot_government_intervention(frame, path),
        "duration_severity": lambda path: plot_duration_severity(frame, path),
        "supplement_missingness": lambda path: plot_missingness(frame, path),
        "full_overdue_proxy_agreement": lambda path: plot_definition_agreement(frame, path),
        "firm_outcomes": lambda path: plot_firm_outcomes(frame, path),
        "model_effects": lambda path: plot_model_effects(model_terms, path),
    }
    outputs: dict[str, Path | None] = {}
    warnings: list[dict[str, str]] = []
    for name, function in specs.items():
        path = dirs["figures"] / f"{name}.png"
        try:
            result = function(path)
        except Exception as error:
            result = None
            warnings.append({"figure": name, "warning": repr(error)})
        outputs[name] = result
        if result is None and not any(row["figure"] == name for row in warnings):
            warnings.append({"figure": name, "warning": "not_enough_model_terms_or_data"})
    pd.DataFrame(warnings, columns=["figure", "warning"]).to_csv(
        dirs["diagnostics"] / "figure_warnings.csv", index=False, encoding="utf-8-sig"
    )
    return outputs
