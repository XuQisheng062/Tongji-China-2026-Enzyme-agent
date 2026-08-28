from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_empty(path: Path, title: str, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_final_scores(selected: pd.DataFrame, path: Path) -> Path:
    if selected.empty or "final_score" not in selected.columns:
        return _save_empty(path, "Selected mutation final scores", "No selected candidates")
    data = selected.sort_values("final_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(data) + 2)))
    ax.barh(data["mutation"].astype(str), data["final_score"].astype(float))
    ax.set_xlabel("Final score")
    ax.set_ylabel("Mutation")
    ax.set_title("Selected mutation candidates")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_activity_stability(ranked: pd.DataFrame, selected: pd.DataFrame, path: Path) -> Path:
    needed = {"activity_proxy", "ddg", "mutation"}
    if ranked.empty or not needed.issubset(ranked.columns):
        return _save_empty(path, "Activity proxy vs UniStab ddG", "No plottable candidates")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        ranked["activity_proxy"].astype(float),
        ranked["ddg"].astype(float),
        alpha=0.55,
    )
    if not selected.empty and needed.issubset(selected.columns):
        ax.scatter(
            selected["activity_proxy"].astype(float),
            selected["ddg"].astype(float),
            marker="x",
            s=70,
        )
        for _, row in selected.iterrows():
            ax.annotate(
                str(row["mutation"]),
                (float(row["activity_proxy"]), float(row["ddg"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("EnzGFM activity_proxy")
    ax.set_ylabel("UniStab ddG")
    ax.set_title("Activity proxy vs stability prediction")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_heatmap(selected: pd.DataFrame, path: Path) -> Path:
    columns = ["activity_proxy", "ph_opt", "ddg", "final_score"]
    if selected.empty or not set(columns + ["mutation"]).issubset(selected.columns):
        return _save_empty(path, "Selected candidate property profile", "No selected candidates")

    data = selected[columns].astype(float).copy()
    for col in columns:
        arr = data[col].to_numpy(dtype=float)
        std = float(np.nanstd(arr))
        if np.isfinite(std) and std > 0:
            data[col] = (arr - float(np.nanmean(arr))) / std
        else:
            data[col] = 0.0

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(selected) * 0.5 + 2)))
    im = ax.imshow(data.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(columns)), labels=columns, rotation=20, ha="right")
    ax.set_yticks(range(len(selected)), labels=selected["mutation"].astype(str).tolist())
    ax.set_title("Selected candidate property profile (column-wise z-score)")
    fig.colorbar(im, ax=ax, label="Standardized value")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_mutation_sequence_map(selected: pd.DataFrame, sequence_length: int, path: Path) -> Path:
    needed = {"mutation", "position", "final_score"}
    if sequence_length <= 0 or selected.empty or not needed.issubset(selected.columns):
        return _save_empty(path, "Mutation sequence map", "No selected candidates")

    data = selected.copy()
    data["position"] = pd.to_numeric(data["position"], errors="coerce")
    data["final_score"] = pd.to_numeric(data["final_score"], errors="coerce")
    data = data.dropna(subset=["position", "final_score"])
    if data.empty:
        return _save_empty(path, "Mutation sequence map", "No valid mutation positions")

    fig, ax = plt.subplots(figsize=(11, max(4.2, 0.25 * len(data) + 3)))
    ax.hlines(0.0, 1, sequence_length, linewidth=3)

    ymax = max(1.0, float(data["final_score"].max()) * 1.25)
    for idx, (_, row) in enumerate(data.iterrows()):
        x = float(row["position"])
        y = float(row["final_score"])
        ax.vlines(x, 0.0, y, linewidth=1.4)
        ax.scatter([x], [y], s=55)
        offset = 5 if idx % 2 == 0 else -13
        ax.annotate(
            str(row["mutation"]),
            (x, y),
            xytext=(4, offset),
            textcoords="offset points",
            fontsize=8,
            rotation=30,
            ha="left",
            va="bottom" if offset > 0 else "top",
        )

    ax.set_xlim(0, sequence_length + 1)
    ax.set_ylim(-0.08 * ymax, ymax)
    ax.set_xlabel(f"Protein sequence position (1-{sequence_length})")
    ax.set_ylabel("Final score")
    ax.set_title("Selected mutation sequence map")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pareto_front(ranked: pd.DataFrame, selected: pd.DataFrame, path: Path) -> Path:
    needed = {
        "mutation",
        "activity_component",
        "ph_component",
        "stability_component",
        "pareto_optimal",
    }
    if ranked.empty or not needed.issubset(ranked.columns):
        return _save_empty(path, "Three-objective Pareto front", "No Pareto data")

    data = ranked.copy()
    for col in ["activity_component", "ph_component", "stability_component"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["activity_component", "ph_component", "stability_component"])
    if data.empty:
        return _save_empty(path, "Three-objective Pareto front", "No valid Pareto data")

    sizes = 45 + 150 * data["ph_component"].clip(0, 1).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(
        data["activity_component"].astype(float),
        data["stability_component"].astype(float),
        s=sizes,
        alpha=0.45,
        label="Candidate pool",
    )

    pareto_mask = data["pareto_optimal"].astype(str).str.lower().isin({"true", "1"})
    pareto = data[pareto_mask]
    if not pareto.empty:
        ax.scatter(
            pareto["activity_component"].astype(float),
            pareto["stability_component"].astype(float),
            s=95 + 150 * pareto["ph_component"].clip(0, 1).to_numpy(dtype=float),
            facecolors="none",
            linewidths=1.6,
            label="3-objective Pareto optimal",
        )

    if not selected.empty and {"mutation", "activity_component", "stability_component"}.issubset(selected.columns):
        ax.scatter(
            selected["activity_component"].astype(float),
            selected["stability_component"].astype(float),
            marker="x",
            s=80,
            label="Selected Top-K",
        )
        for _, row in selected.iterrows():
            ax.annotate(
                str(row["mutation"]),
                (float(row["activity_component"]), float(row["stability_component"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Normalized activity component (higher is better)")
    ax.set_ylabel("Normalized stability component (higher is better)")
    ax.set_title("Three-objective Pareto view (marker size = pH component)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_visualizations(
    *,
    ranked_csv: Path,
    selected_csv: Path,
    output_dir: Path,
    sequence_length: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_csv(ranked_csv)
    selected = pd.read_csv(selected_csv)
    paths = [
        _plot_final_scores(selected, output_dir / "01_final_score.png"),
        _plot_activity_stability(ranked, selected, output_dir / "02_activity_vs_stability.png"),
        _plot_heatmap(selected, output_dir / "03_property_heatmap.png"),
        _plot_mutation_sequence_map(selected, sequence_length, output_dir / "04_mutation_sequence_map.png"),
        _plot_pareto_front(ranked, selected, output_dir / "05_pareto_front.png"),
    ]
    return paths
