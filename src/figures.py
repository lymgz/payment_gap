"""Matplotlib figures corresponding to the supported Stata graphics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize

from .reporting import ensure_output_dirs


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overdue_vs_planned(df: pd.DataFrame, path: Path) -> Path:
    """Plot overdue area against planned area with an overdue-rate axis."""

    data = df[["Planned", "Overdue", "O_rate"]].apply(pd.to_numeric, errors="coerce").dropna()
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.scatter(data["Planned"], data["Overdue"], color="#2f5597", alpha=0.75, label="Overdue area")
    if len(data) >= 3 and data["Planned"].nunique() > 1:
        x = data["Planned"].to_numpy(dtype=float)
        y = data["Overdue"].to_numpy(dtype=float)
        model = sm.OLS(y, sm.add_constant(x)).fit()
        grid = np.linspace(float(x.min()), float(x.max()), 120)
        prediction = model.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)
        axis.plot(grid, prediction["mean"], color="#1f4e79", linewidth=1.5, label="Linear fit")
        axis.fill_between(
            grid,
            prediction["mean_ci_lower"],
            prediction["mean_ci_upper"],
            color="#9dc3e6",
            alpha=0.35,
            label="95% CI",
        )
    axis.set_xlabel("Planned delivery area")
    axis.set_ylabel("Overdue delivery area")
    axis.grid(alpha=0.2)
    rate_axis = axis.twinx()
    rate_axis.scatter(
        data["Planned"],
        data["O_rate"],
        color="#c00000",
        marker="^",
        s=25,
        alpha=0.65,
        label="Overdue rate",
    )
    rate_axis.set_ylabel("Overdue rate", color="#c00000")
    rate_axis.tick_params(axis="y", labelcolor="#c00000")
    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = rate_axis.get_legend_handles_labels()
    axis.legend(handles_a + handles_b, labels_a + labels_b, loc="best", frameon=True)
    axis.set_title("Overdue delivery area and overdue rate")
    fig.tight_layout()
    return _save(fig, path)


def plot_color_boxplots(df: pd.DataFrame, path: Path) -> Path:
    """Plot overdue and planned area distributions by red/yellow/green group."""

    order = ["Red", "Yellow", "Green"]
    colors = ["#c00000", "#bf9000", "#70ad47"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for axis, variable, title in zip(axes, ["Overdue", "Planned"], ["Overdue area", "Planned area"]):
        groups = [pd.to_numeric(df.loc[df["color_text"] == label, variable], errors="coerce").dropna() for label in order]
        box = axis.boxplot(groups, tick_labels=order, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.set_title(title)
        axis.set_ylabel("Area")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Delivery area by color cluster")
    fig.tight_layout()
    return _save(fig, path)


def _geometry_segments(geometry_df: pd.DataFrame) -> list[tuple[int, list[list[float]]]]:
    required = {"_ID", "_X", "_Y"}
    if not required.issubset(geometry_df.columns):
        return []
    segments: list[tuple[int, list[list[float]]]] = []
    current_id: int | None = None
    current: list[list[float]] = []

    def flush() -> None:
        nonlocal current
        if current_id is not None and len(current) >= 3:
            segments.append((current_id, current))
        current = []

    for row in geometry_df[["_ID", "_X", "_Y"]].itertuples(index=False):
        identifier, x_value, y_value = row
        if pd.isna(x_value) or pd.isna(y_value):
            flush()
            continue
        try:
            numeric_id = int(identifier)
            numeric_x = float(x_value)
            numeric_y = float(y_value)
        except (TypeError, ValueError, OverflowError):
            return []
        if current_id is not None and numeric_id != current_id:
            flush()
        current_id = numeric_id
        current.append([numeric_x, numeric_y])
    flush()
    return segments


def plot_province_map(
    province_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    path: Path,
    value: str,
    title: str,
    cmap: str = "Blues",
) -> Path | None:
    """Render Stata-style `_ID`, `_X`, `_Y` geometry as a province map."""

    required_province = {"id", "ename", "x_coord", "y_coord", value}
    if not required_province.issubset(province_df.columns):
        return None
    segments = _geometry_segments(geometry_df)
    if not segments:
        return None
    values = pd.to_numeric(province_df.set_index("id")[value], errors="coerce")
    finite_values = values[np.isfinite(values)]
    if finite_values.empty:
        return None
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    patches: list[Polygon] = []
    patch_values: list[float] = []
    for identifier, coordinates in segments:
        if identifier not in values.index:
            continue
        patches.append(Polygon(coordinates, closed=True))
        patch_values.append(float(values.loc[identifier]))
    if not patches:
        return None
    fig, axis = plt.subplots(figsize=(8, 7))
    collection = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.25)
    collection.set_array(np.asarray(patch_values, dtype=float))
    axis.add_collection(collection)
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    axis.set_title(title)
    colorbar = fig.colorbar(collection, ax=axis, shrink=0.68, pad=0.02)
    colorbar.set_label(value)
    labels = province_df.drop_duplicates("id")
    for row in labels.itertuples(index=False):
        if np.isfinite(row.x_coord) and np.isfinite(row.y_coord):
            axis.text(row.x_coord, row.y_coord, str(row.ename), fontsize=3.8, ha="center", va="center")
    fig.tight_layout()
    return _save(fig, path)


def run_figure_suite(inputs: dict[str, Any], analysis_df: pd.DataFrame, output_dir: Path) -> dict[str, Path | None]:
    """Write the supported area, cluster, and province map figures."""

    dirs = ensure_output_dirs(Path(output_dir))
    outputs: dict[str, Path | None] = {
        "overdue_vs_planned": plot_overdue_vs_planned(analysis_df, dirs["figures"] / "overdue_vs_planned.png"),
        "color_boxplots": plot_color_boxplots(analysis_df, dirs["figures"] / "color_boxplots.png"),
    }
    warnings_rows: list[dict[str, str]] = []
    province = inputs.get("province_map")
    geometry = inputs.get("china_map")
    map_specs = [
        ("Overduearea", "Province overdue area", "Blues"),
        ("Red_O", "Overdue area in red cluster", "Reds"),
        ("Yellow_O", "Overdue area in yellow cluster", "YlOrBr"),
        ("Green_O", "Overdue area in green cluster", "Greens"),
    ]
    for value, title, cmap in map_specs:
        key = f"map_{value.lower()}"
        path = dirs["figures"] / f"{key}.png"
        try:
            result = plot_province_map(province, geometry, path, value, title, cmap=cmap)
        except (TypeError, ValueError, KeyError, OverflowError) as error:
            result = None
            warnings_rows.append({"figure": key, "warning": f"map_render_failed={error!r}"})
        outputs[key] = result
        if result is None:
            warnings_rows.append({"figure": key, "warning": "legacy geometry/value columns could not be rendered"})
    pd.DataFrame(warnings_rows, columns=["figure", "warning"]).to_csv(
        dirs["diagnostics"] / "figure_warnings.csv", index=False, encoding="utf-8-sig"
    )
    return outputs
