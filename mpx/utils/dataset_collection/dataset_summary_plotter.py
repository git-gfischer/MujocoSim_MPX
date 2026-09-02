#Usage: python -m mpx.utils.dataset_collection.dataset_summary_plotter datasets/dataset_summary.json --show
"""Visualize the persistent JSON summary produced by :class:`DatasetBucketSystem`.

Run from the repository root, for example::

    python -m mpx.utils.dataset_collection.dataset_summary_plotter \
        datasets/dataset_summary.json --show

The command writes ``dataset_summary.png`` alongside the input summary unless
``--output`` is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def load_dataset_summary(summary_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a DatasetBucketSystem summary JSON file."""
    path = Path(summary_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read dataset summary '{path}': {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
        raise ValueError(
            f"Dataset summary '{path}' must contain a top-level 'summary' object"
        )
    return payload


def _bar_chart(
    ax: plt.Axes,
    values: Mapping[str, Any],
    title: str,
    *,
    color: str = "tab:blue",
    horizontal: bool = False,
) -> None:
    """Draw a labelled bar chart, including a useful empty-state message."""
    items = sorted((str(name), int(count)) for name, count in values.items())
    if not items:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    labels, counts = zip(*items)
    if horizontal:
        positions = np.arange(len(labels))
        ax.barh(positions, counts, color=color)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Stored/exported windows (count)")
    else:
        positions = np.arange(len(labels))
        ax.bar(positions, counts, color=color)
        ax.set_xticks(positions, labels, rotation=35, ha="right")
        ax.set_ylabel("Windows")
    ax.set_title(title)
    ax.grid(axis="y" if not horizontal else "x", alpha=0.25)


def _run_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return run records in chronological order when timestamps are available."""
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, Mapping):
        return []
    return sorted(
        (record for record in datasets.values() if isinstance(record, Mapping)),
        key=lambda record: str(record.get("completed_at", "")),
    )


def _plot_distribution(
    mean_axis: plt.Axes,
    range_axis: plt.Axes,
    stats_by_bucket: Mapping[str, Any],
    *,
    quantity: str,
    color: str,
) -> None:
    """Plot mean/std and min/max charts for a per-bucket scalar statistic."""
    if not stats_by_bucket:
        for axis, title in (
            (mean_axis, f"{quantity} mean ± standard deviation"),
            (range_axis, f"{quantity} range by bucket"),
        ):
            axis.set_title(title)
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
            axis.set_axis_off()
        return

    items = sorted(stats_by_bucket.items())
    labels = [name for name, _ in items]
    means = np.array([float(stats.get("mean", 0.0)) for _, stats in items])
    stds = np.array([float(stats.get("std", 0.0)) for _, stats in items])
    minima = np.array([float(stats.get("min", 0.0)) for _, stats in items])
    maxima = np.array([float(stats.get("max", 0.0)) for _, stats in items])
    positions = np.arange(len(labels))

    mean_axis.errorbar(positions, means, yerr=stds, fmt="o", capsize=3, color=color)
    mean_axis.set_xticks(positions, labels, rotation=45, ha="right")
    mean_axis.set_ylabel(f"{quantity} [N]")
    mean_axis.set_title(f"{quantity} mean ± standard deviation")
    mean_axis.grid(axis="y", alpha=0.25)

    range_axis.vlines(positions, minima, maxima, color=color, linewidth=2)
    range_axis.scatter(positions, minima, color="tab:blue", label="min")
    range_axis.scatter(positions, maxima, color="tab:red", label="max")
    range_axis.set_xticks(positions, labels, rotation=45, ha="right")
    range_axis.set_ylabel(f"{quantity} [N]")
    range_axis.set_title(f"{quantity} range by bucket")
    range_axis.legend()
    range_axis.grid(axis="y", alpha=0.25)


def plot_dataset_summary(
    summary_path: str | Path,
    *,
    output_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Create a dashboard with every aggregate statistic in a dataset summary.

    Parameters
    ----------
    summary_path:
        Path to ``dataset_summary.json`` written by ``DatasetBucketSystem``.
    output_path:
        Optional PNG/PDF/SVG destination. Parent directories are created.
    show:
        Display the dashboard interactively after saving it.

    Returns
    -------
    matplotlib.figure.Figure
        The caller owns the returned figure and may further customize or close it.
    """
    payload = load_dataset_summary(summary_path)
    summary = payload["summary"]
    runs = _run_records(payload)

    figure, axes = plt.subplots(4, 3, figsize=(21, 20), layout="constrained")
    figure.suptitle(
        f"Dataset collection summary — {Path(summary_path).resolve()}",
        fontsize=16,
        fontweight="bold",
    )

    overview = axes[0, 0]
    overview.axis("off")
    perturbation_ratio = float(summary.get("perturbation_ratio", 0.0))
    force_stats = summary.get("perturbation_force_n", {})
    force_windows = int(force_stats.get("windows", 0))
    force_description = (
        f"{float(force_stats.get('mean', 0.0)):.1f} ± "
        f"{float(force_stats.get('std', 0.0)):.1f} N"
        if force_windows
        else "No perturbed windows"
    )
    overview.text(
        0.02,
        0.98,
        "\n".join(
            [
                "OVERVIEW",
                f"Dataset files:             {int(summary.get('dataset_files', 0)):,}",
                f"Total windows:             {int(summary.get('windows', 0)):,}",
                f"Collection windows stored: {int(summary.get('collection_windows_stored', 0)):,}",
                f"Collection windows seen:   {int(summary.get('collection_windows_seen', 0)):,}",
                f"Perturbation windows:      {int(summary.get('perturbation_windows', 0)):,}",
                f"Perturbation ratio:        {perturbation_ratio:.1%}",
                f"Perturbation force (mean ± std): {force_description}",
                f"Perturbation force range:  "
                f"{float(force_stats.get('min', 0.0)):.1f}–"
                f"{float(force_stats.get('max', 0.0)):.1f} N",
                f"Episodes:                  {int(summary.get('episodes', 0)):,}",
                f"Active buckets:            {int(summary.get('active_buckets', 0)):,}",
                f"Dataset size:              {float(summary.get('total_size_bytes', 0)) / 1e6:.2f} MB",
                f"Last updated: {payload.get('updated_at') or 'not yet written'}",
            ]
        ),
        va="top",
        family="monospace",
        fontsize=11,
    )

    perturbation = axes[0, 1]
    perturbed = int(summary.get("perturbation_windows", 0))
    total_windows = int(summary.get("windows", 0))
    unperturbed = max(total_windows - perturbed, 0)
    if total_windows:
        perturbation.pie(
            [perturbed, unperturbed],
            labels=["Perturbed", "Unperturbed"],
            autopct="%1.1f%%",
            colors=["tab:orange", "tab:blue"],
            startangle=90,
        )
    else:
        perturbation.text(0.5, 0.5, "No windows", ha="center", va="center")
    perturbation.set_title("Perturbation coverage")

    _bar_chart(
        axes[0, 2],
        summary.get("contact_state_counts", {}),
        "Windows by contact state",
        horizontal=True,
    )
    _bar_chart(axes[1, 0], summary.get("terrain_counts", {}), "Windows by terrain")
    _bar_chart(axes[1, 1], summary.get("gait_counts", {}), "Windows by gait", color="tab:green")

    bucket_counts = summary.get("bucket_counts", {})
    _bar_chart(
        axes[1, 2],
        bucket_counts,
        "Stored/exported windows by bucket",
        color="tab:purple",
        horizontal=True,
    )

    run_axis = axes[2, 0]
    if runs:
        run_windows = [int(record.get("export", {}).get("windows", 0)) for record in runs]
        run_axis.plot(range(1, len(runs) + 1), run_windows, marker="o", color="tab:blue")
        run_axis.set_xlabel("Collection run (chronological)")
        run_axis.set_ylabel("Exported windows")
        run_axis.grid(alpha=0.25)
    else:
        run_axis.text(0.5, 0.5, "No run records", ha="center", va="center")
    run_axis.set_title("Windows per collection run")

    _plot_distribution(
        axes[2, 1],
        axes[2, 2],
        summary.get("perturbation_force_n_by_bucket", {}),
        quantity="Perturbation force magnitude",
        color="tab:orange",
    )
    _plot_distribution(
        axes[3, 0],
        axes[3, 1],
        summary.get("grf_total_n_by_bucket", {}),
        quantity="Total GRF",
        color="tab:red",
    )

    run_ratio_axis = axes[3, 2]
    if runs:
        ratios = [
            float(record.get("export", {}).get("perturbation_ratio", 0.0))
            for record in runs
        ]
        run_ratio_axis.plot(
            range(1, len(runs) + 1), ratios, marker="o", color="tab:orange"
        )
        run_ratio_axis.set_xlabel("Collection run (chronological)")
        run_ratio_axis.set_ylabel("Perturbed-window ratio")
        run_ratio_axis.set_ylim(0.0, 1.0)
        run_ratio_axis.grid(alpha=0.25)
    else:
        run_ratio_axis.text(0.5, 0.5, "No run records", ha="center", va="center")
    run_ratio_axis.set_title("Perturbation coverage per collection run")

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset-summary plotting command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_path", type=Path, help="Path to dataset_summary.json")
    parser.add_argument(
        "--output",
        type=Path,
        help="Image destination (default: dataset_summary.png next to the JSON file)",
    )
    parser.add_argument("--show", action="store_true", help="Display the dashboard")
    args = parser.parse_args(argv)

    output_path = args.output or args.summary_path.with_name("dataset_summary.png")
    plot_dataset_summary(args.summary_path, output_path=output_path, show=args.show)
    print(f"Saved dataset summary dashboard to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
