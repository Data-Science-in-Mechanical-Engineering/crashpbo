"""
Generate the four-panel paper plot comparing GP and synthetic benchmarks.

The script aggregates normalized performance and cumulative crash curves across all
available GP seeds/dimensions as well as the configured synthetic functions, and
creates a 2x2 (four subplot) figure:

    +-----------------------------+-----------------------------+
    | GPs - normalized perf.      | GPs - cumulative crashes    |
    | Synthetic - normalized perf.| Synthetic - cumulative crash|
    +-----------------------------+-----------------------------+
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from .experiment_evaluation import (
    EvaluateMultipleAlgorithms,
    EvaluateMultipleAlgorithms_GP,
)

DEFAULT_RESULTS_ROOT = Path("results/")
DEFAULT_SYNTHETIC_FUNCTIONS = ["ackley_2D", "hartmann_6D", "branin_2D", "cosine8_8D"]
DEFAULT_ALGORITHMS = ["crashPBO", "EUBO", "MES", "SafeOpt", "random"]
DEFAULT_MODE = "compare to best"
FEASIBLE_SUFFIX = "feasiblep_0.5"
RESAMPLE_GRID = np.linspace(0.0, 1.0, 101)
RESAMPLE_INDEX = pd.Index(RESAMPLE_GRID, dtype=float)
DOUBLE_COLUMN_FIGSIZE = (6.8, 4.0)
TITLE_FONT_SIZE = 9
LABEL_FONT_SIZE = 9
TICK_FONT_SIZE = 9
LEGEND_FONT_SIZE = 9
DEFAULT_LINESTYLE = "-"
DEFAULT_COLOR = "#111111"
BW_FRIENDLY_ALGORITHM_COLORS = {
    "crashPBO": "blue",        # keep
    "EUBO": "green",           # keep
    "MES": "orange",          # orange (safe)
    "SafeOpt": "#B80166",      # reddish-purple (safe)
    "random": "black",         # ok
    "ISE": "magenta",          # magenta (safe)
}
TICK_FORMAT_STRING = "%.1f"


def _preferred_color(algorithm: str, fallback: str | None = None) -> str:
    color = BW_FRIENDLY_ALGORITHM_COLORS.get(algorithm)
    if color:
        return color
    if fallback:
        return fallback
    return DEFAULT_COLOR


def _parse_dimension(dim_value) -> int:
    """Return the integer dimension encoded in a file name fragment."""
    if isinstance(dim_value, str):
        dim_value = dim_value.rstrip("D")
    try:
        dim = int(dim_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to parse dimension from '{dim_value}'") from exc
    return max(dim, 1)


def _resample_curve(series: pd.Series) -> pd.Series:
    """Interpolate the series onto the shared RESAMPLE_GRID, clipping beyond ratio 1."""
    if series.empty:
        return series
    clipped = series[series.index <= 1.0]
    if clipped.empty:
        # fallback to the first entry to avoid returning an empty result
        clipped = series.iloc[[0]]
        clipped.index = pd.Index([0.0], dtype=float)
    union_index = clipped.index.union(RESAMPLE_INDEX)
    interpolated = (
        clipped.reindex(union_index)
        .interpolate(method="values")
        .reindex(RESAMPLE_INDEX)
        .ffill()
        .bfill()
    )
    interpolated.index = RESAMPLE_GRID
    return interpolated


def _series_with_normalized_index(series: pd.Series, dim_value) -> pd.Series:
    """Attach an index scaled by iterations / (dim * 10)."""
    dim = _parse_dimension(dim_value)
    total_iters = max(dim * 10, 1)
    values = series.reset_index(drop=True).to_numpy(copy=True)
    if values.size == 0:
        return pd.Series(dtype=float)
    index = np.arange(len(values), dtype=float) / total_iters
    normalized = pd.Series(values, index=index)
    return _resample_curve(normalized)


def _build_algorithm_store(
    algorithms: Sequence[str],
) -> Dict[str, MutableMapping[str, List[pd.Series]]]:
    """Return a dictionary that accumulates per-algorithm curves."""

    return {
        alg: {
            "performance": [],
            "crashes": [],
            "color": BW_FRIENDLY_ALGORITHM_COLORS.get(alg),
            "linestyle": None,
        }
        for alg in algorithms
    }


def _summarize_series(series_collection: List[pd.Series]) -> Tuple[pd.Series, pd.Series]:
    """Return the mean and std envelopes for the provided series."""

    if not series_collection:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    stacked = pd.concat(
        series_collection,
        axis=1,
    ).sort_index()
    mean = stacked.mean(axis=1, skipna=True)
    std = stacked.std(axis=1, skipna=True)
    mean = mean.ffill().bfill()
    std = std.fillna(0.0)
    return mean, std


def _finalize_store(
    store: Dict[str, MutableMapping[str, List[pd.Series]]],
) -> Dict[str, Dict[str, pd.Series]]:
    """Convert the raw store into a dictionary of averaged series."""

    summary: Dict[str, Dict[str, pd.Series]] = {}
    for algorithm, payload in store.items():
        perf_mean, perf_std = _summarize_series(payload["performance"])
        crash_mean, crash_std = _summarize_series(payload["crashes"])
        if perf_mean.empty or crash_mean.empty:
            continue
        summary[algorithm] = {
            "performance_mean": perf_mean,
            "performance_std": perf_std,
            "crash_mean": crash_mean,
            "crash_std": crash_std,
            "color": payload["color"],
            "linestyle": payload["linestyle"],
        }
    return summary


def _extract_legend_entries(axes: np.ndarray) -> Tuple[List, List]:
    """Collect unique legend entries from a grid of axes."""

    handles: List = []
    labels: List = []
    seen = set()
    for ax in axes.flatten():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if not label or label in seen:
                continue
            handles.append(handle)
            labels.append(label)
            seen.add(label)
    return handles, labels


def _discover_gp_dirs(results_root: Path) -> List[Path]:
    """Return all gp_seed directories under ``results_root``."""

    gp_dirs: List[Path] = []
    if not results_root.is_dir():
        return gp_dirs
    for child in results_root.iterdir():
        if (
            child.is_dir()
            and child.name.startswith("gp_seed_")
            and child.name.endswith(f"_{FEASIBLE_SUFFIX}")
        ):
            gp_dirs.append(child)
    return sorted(gp_dirs)


def _aggregate_gp_curves(
    results_root: Path,
    algorithms: Sequence[str],
    mode: str,
    gp_dirs: Sequence[Path] | None = None,
) -> Dict[str, Dict[str, pd.Series]]:
    """Average normalized GP curves across all available seed folders."""

    store = _build_algorithm_store(algorithms)
    gp_dirs = list(gp_dirs) if gp_dirs is not None else _discover_gp_dirs(results_root)
    if not gp_dirs:
        raise FileNotFoundError(
            f"No gp_seed folders found under '{results_root}'. "
            "Run the GP experiments before generating the plot."
        )

    for folder in gp_dirs:
        evaluation = EvaluateMultipleAlgorithms_GP(str(folder), algorithms)
        for single in evaluation.get_average_normalized_performance():
            if single.algorithm not in store or single.mode != mode:
                continue
            dim_value = getattr(single, "dim", None)
            if dim_value is None:
                continue
            perf_series = _series_with_normalized_index(
                single.df_y["mean"].reset_index(drop=True), dim_value
            )
            crash_series = _series_with_normalized_index(
                single.crashes["mean"].reset_index(drop=True), dim_value
            )
            store[single.algorithm]["performance"].append(perf_series)
            store[single.algorithm]["crashes"].append(crash_series)
            if store[single.algorithm]["color"] is None:
                store[single.algorithm]["color"] = _preferred_color(
                    single.algorithm, single.color
                )
            if store[single.algorithm]["linestyle"] is None:
                store[single.algorithm]["linestyle"] = single.line_style

    return _finalize_store(store)


def _resolve_function_dir(results_root: Path, function: str) -> Path:
    """Locate the directory containing the synthetic function pickles."""

    suffix = f"{function}_{FEASIBLE_SUFFIX}"
    direct = results_root / suffix
    if direct.is_dir():
        return direct

    candidates: List[Path] = []
    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        nested = child / suffix
        if nested.is_dir():
            candidates.append(nested)

    if not candidates:
        raise FileNotFoundError(
            f"Unable to locate results for '{function}' under '{results_root}'."
        )
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1]


def _aggregate_synthetic_curves(
    results_root: Path,
    functions: Sequence[str],
    algorithms: Sequence[str],
    mode: str,
    function_dirs: Mapping[str, Path] | None = None,
) -> Dict[str, Dict[str, pd.Series]]:
    """Average normalized synthetic curves across ``functions``."""

    store = _build_algorithm_store(algorithms)
    for function in functions:
        if function_dirs and function in function_dirs:
            folder = function_dirs[function]
        else:
            folder = _resolve_function_dir(results_root, function)
        evaluation = EvaluateMultipleAlgorithms(str(folder), algorithms)
        for single in evaluation.evaluations:
            if single.algorithm not in store or single.mode != mode:
                continue
            single.normalize_data(evaluation.worst_y, evaluation.best_y)
            dim_value = getattr(single, "dim", None)
            if dim_value is None:
                continue
            perf_series = _series_with_normalized_index(
                single.df_y["mean"].reset_index(drop=True), dim_value
            )
            crash_series = _series_with_normalized_index(
                single.crashes["mean"].reset_index(drop=True), dim_value
            )
            store[single.algorithm]["performance"].append(perf_series)
            store[single.algorithm]["crashes"].append(crash_series)
            if store[single.algorithm]["color"] is None:
                store[single.algorithm]["color"] = _preferred_color(
                    single.algorithm, single.color
                )
            if store[single.algorithm]["linestyle"] is None:
                store[single.algorithm]["linestyle"] = single.line_style

    return _finalize_store(store)


def _plot_metric(
    ax: plt.Axes,
    summary: Dict[str, Dict[str, pd.Series]],
    algorithms: Sequence[str],
    metric_prefix: str,
    title: str,
    ylabel: str | None = None,
) -> None:
    """Plot the mean and std envelope for one metric (performance or crashes)."""

    for algorithm in algorithms:
        stats = summary.get(algorithm)
        if stats is None:
            continue
        legend_label = "crashPBO (ours)" if algorithm == "crashPBO" else algorithm
        mean = stats[f"{metric_prefix}_mean"]
        std = stats[f"{metric_prefix}_std"]
        x = mean.index.to_numpy()
        y = mean.to_numpy()
        y_std = std.to_numpy()
        color = _preferred_color(algorithm, stats["color"])
        linestyle = stats["linestyle"] or DEFAULT_LINESTYLE
        ax.plot(
            x,
            y,
            label=legend_label,
            color=color,
            linestyle=linestyle,
        )
        if not std.empty:
            ax.fill_between(
                x,
                y - y_std,
                y + y_std,
                color=color,
                alpha=0.12,
            )
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.xaxis.set_major_formatter(FormatStrFormatter(TICK_FORMAT_STRING))
    if metric_prefix == "crash":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter(TICK_FORMAT_STRING))


def build_paper_figure(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    synthetic_functions: Sequence[str] = DEFAULT_SYNTHETIC_FUNCTIONS,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    mode: str = DEFAULT_MODE,
) -> Tuple[plt.Figure, np.ndarray]:
    """Return the four-panel figure used in the paper."""

    results_root = Path(results_root)
    gp_dirs = _discover_gp_dirs(results_root)
    function_dirs = {
        function: _resolve_function_dir(results_root, function)
        for function in synthetic_functions
    }
    gp_summary = _aggregate_gp_curves(results_root, algorithms, mode, gp_dirs)
    synthetic_summary = _aggregate_synthetic_curves(
        results_root,
        synthetic_functions,
        algorithms,
        mode,
        function_dirs,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=DOUBLE_COLUMN_FIGSIZE,
        sharex="col",
        constrained_layout=False,
    )

    _plot_metric(
        axes[0, 0],
        gp_summary,
        algorithms,
        metric_prefix="performance",
        title="Within model",
        ylabel="Normalized performance",
    )
    _plot_metric(
        axes[1, 0],
        gp_summary,
        algorithms,
        metric_prefix="crash",
        title=None,
        ylabel="Cumulative crashes",
    )
    _plot_metric(
        axes[0, 1],
        synthetic_summary,
        algorithms,
        metric_prefix="performance",
        title="Synthetic functions",
        ylabel="Normalized performance",
    )
    _plot_metric(
        axes[1, 1],
        synthetic_summary,
        algorithms,
        metric_prefix="crash",
        title=None,
        ylabel="Cumulative crashes",
    )

    for ax in axes.flatten():
        ax.set_xlim(0, 1)
    for ax in axes[0, :]:
        ax.tick_params(axis="x", labelbottom=True)
    for ax in axes[1, :]:
        ax.set_xlabel("Iterations / (dim * 10)", fontsize=LABEL_FONT_SIZE)

    handles, labels = _extract_legend_entries(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.08),
            frameon=False,
            columnspacing=0.6,
            handlelength=1.5,
            prop={"size": LEGEND_FONT_SIZE},
        )
    fig.subplots_adjust(
        hspace=0.16,
        wspace=0.26,
        top=0.96,
        bottom=0.18,
        left=0.1,
        right=0.98,
    )
    return fig, axes


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the paper figure script."""

    parser = argparse.ArgumentParser(
        description="Create the four-panel plot aggregating GP and synthetic results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Base directory that contains the pickled experiment results.",
    )
    parser.add_argument(
        "--functions",
        nargs="+",
        default=DEFAULT_SYNTHETIC_FUNCTIONS,
        help="Synthetic function prefixes to include in the lower row.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        help="Algorithms to include when averaging the curves.",
    )
    parser.add_argument(
        "--mode",
        choices=["compare to best", "compare to last", "two new parameters"],
        default=DEFAULT_MODE,
        help="Acquisition comparison mode to use when reading the pickles.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        help="Optional path to save the generated figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip plt.show(); useful for headless environments.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI usage."""

    args = _parse_args()
    fig, _ = build_paper_figure(
        results_root=args.results_root,
        synthetic_functions=args.functions,
        algorithms=args.algorithms,
        mode=args.mode,
    )
    if args.save_path:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=300, bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
