"""Utilities for summarizing crashPBO and EUBO experiments.

The previous version of this module contained code for a large collection of
plots and ranking experiments.  For the paper we only need three pieces of
functionality:

* load a pickled experiment file and expose normalized performance/crash curves
* aggregate those curves across folders to form the LaTeX table
* provide the data access helpers used by the plotting scripts

The classes and functions below focus solely on those tasks, which keeps the
module compact while retaining the public API that ``new_plot_paper`` and
``new_plot_rebuttal`` rely on.
"""

from __future__ import annotations

import argparse
import importlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import sys

import numpy as np
import pandas as pd

DEFAULT_ALGORITHMS: tuple[str, ...] = ("crashPBO", "EUBO")
DEFAULT_SYNTHETIC_FUNCTIONS: tuple[str, ...] = (
    "ackley_2D",
    "hartmann_6D",
    "branin_2D",
    "cosine8_8D",
)
DEFAULT_GP_DIMENSIONS: tuple[int, ...] = tuple(range(1, 9))
DEFAULT_MODES: tuple[str, ...] = (
    "compare to best",
    "compare to last",
    "two new parameters",
)
DEFAULT_RESULTS_ROOT = Path("results/20250818_results")
FEASIBLE_SUFFIX = "feasiblep_0.5"
GP_DIR_TEMPLATE = "gp_seed_{seed}_{dim}D_" + FEASIBLE_SUFFIX
ALGORITHM_COLORS = {"crashPBO": "blue", "EUBO": "green"}
MODE_LINE_STYLES = {
    "compare to best": "-",
    "compare to last": "--",
    "two new parameters": "-.",
}


def _ensure_numpy_core_aliases() -> None:
    """Recreate the ``numpy._core`` aliases that older pickles expect."""

    try:
        np_core = importlib.import_module("numpy._core")
    except ModuleNotFoundError:
        try:
            np_core = importlib.import_module("numpy.core")
        except ModuleNotFoundError:
            np_core = None

    if np_core is None:
        return

    modules = (
        ("numpy._core", np_core),
        ("numpy._core.multiarray", getattr(np_core, "multiarray", None)),
        ("numpy._core.numerictypes", getattr(np_core, "numerictypes", None)),
        ("numpy._core.umath", getattr(np_core, "umath", None)),
    )
    for name, module in modules:
        if module is not None:
            sys.modules.setdefault(name, module)


_ensure_numpy_core_aliases()


def _iteration_index(length: int, dim: int) -> int:
    """Return the iteration index corresponding to ``dim * 10``."""

    target = dim * 10
    return max(0, min(length - 1, target))


def _attach_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``mean`` and ``std`` columns to the provided samples."""

    summary = df.copy()
    summary["mean"] = summary.mean(axis=1)
    summary["std"] = summary.std(axis=1)
    return summary


def _downsample_for_compare_modes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep every second row after the initial burn-in."""

    if len(df) <= 2:
        return df.copy()
    return pd.concat([df.iloc[:2], df.iloc[2::2]], axis=0).reset_index(drop=True)


@dataclass
class _Metadata:
    function: str
    dim: int
    feasible_percentage: str
    algorithm: str
    mode: str


class SingleEvaluation:
    """Load and store the per-algorithm results contained in a pickle file."""

    color: str
    line_style: str

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata = self._parse_metadata()
        self.function = self.metadata.function
        self.dim = self.metadata.dim
        self.feasible_percentage = self.metadata.feasible_percentage
        self.algorithm = self.metadata.algorithm
        self.mode = self.metadata.mode
        self.color = ALGORITHM_COLORS.get(self.algorithm, "#333333")
        self.line_style = MODE_LINE_STYLES.get(self.mode, "-")
        self._performance_samples: pd.DataFrame | None = None
        self._crash_samples: pd.DataFrame | None = None
        self.df_y: pd.DataFrame
        self.crashes: pd.DataFrame
        self.best_y: float = -np.inf
        self.worst_y: float = np.inf
        self._load_curves()

    def _parse_metadata(self) -> _Metadata:
        """Parse function/dimension/algorithm/mode information from the file."""

        tokens = self.path.stem.split("_")
        if len(tokens) < 9:
            raise ValueError(f"Unexpected file name structure: '{self.path.name}'")
        function = tokens[1]
        dim = int(tokens[2].rstrip("D"))
        feasible_percentage = tokens[4]
        algorithm = tokens[5]
        mode = " ".join(tokens[6:9])
        return _Metadata(function, dim, feasible_percentage, algorithm, mode)

    def _load_curves(self) -> None:
        """Load raw data from pickle and pre-compute the cumulative curves."""

        with self.path.open("rb") as handle:
            raw_data = pickle.load(handle)

        performance = pd.DataFrame()
        crashes = pd.DataFrame()

        for key, df in raw_data.items():
            series = df.copy()
            series["crash"] = series["crash"].cumsum()
            crashes[key] = series["crash"]
            y = series["y"].replace(np.nan, -np.inf).cummax()
            performance[key] = y

        self._performance_samples = performance
        self._crash_samples = crashes
        self.best_y = float(performance.max().max())
        self.worst_y = float(performance.min().min())
        self.df_y = _attach_summary_columns(performance)
        self.crashes = _attach_summary_columns(crashes)

    def _prepare_samples(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return copies of the raw samples with optional down-sampling applied."""

        performance = self._performance_samples.copy()
        crashes = self._crash_samples.copy()
        if (
            self.algorithm in DEFAULT_ALGORITHMS
            and self.mode in ("compare to best", "compare to last")
        ):
            performance = _downsample_for_compare_modes(performance)
            crashes = _downsample_for_compare_modes(crashes)
        return performance, crashes

    def normalize_data(self, minimum: float, maximum: float) -> None:
        """Normalize the performance curves to the [0, 1] range."""

        spread = maximum - minimum
        if not np.isfinite(spread) or spread == 0:
            spread = 1.0

        performance, crashes = self._prepare_samples()
        performance = (performance - minimum) / spread
        self.df_y = _attach_summary_columns(performance)
        self.crashes = _attach_summary_columns(crashes)


class SingleEvaluation_GP(SingleEvaluation):
    """Same as :class:`SingleEvaluation` but with the GP specific naming scheme."""

    seed: str

    def _parse_metadata(self) -> _Metadata:
        tokens = self.path.stem.split("_")
        if len(tokens) < 11:
            raise ValueError(f"Unexpected GP file name: '{self.path.name}'")
        function = tokens[1]
        self.seed = tokens[3]
        dim = int(tokens[4].rstrip("D"))
        feasible_percentage = tokens[6]
        algorithm = tokens[7]
        mode = " ".join(tokens[8:11])
        return _Metadata(function, dim, feasible_percentage, algorithm, mode)


class _EvaluationDirectory:
    """Base helper that loads all pickles stored in a directory."""

    evaluation_cls = SingleEvaluation

    def __init__(
        self,
        directory: str | Path,
        algorithms: Sequence[str] | None = None,
    ):
        self.directory = Path(directory)
        self.algorithms = tuple(algorithms or DEFAULT_ALGORITHMS)
        self.evaluations: List[SingleEvaluation] = []
        self.best_y: float | None = None
        self.worst_y: float | None = None
        self._load_evaluations()

    def _load_evaluations(self) -> None:
        for file in sorted(self.directory.glob("*.pkl")):
            try:
                evaluation = self.evaluation_cls(file)
            except (EOFError, ModuleNotFoundError, pickle.UnpicklingError) as exc:
                print(f"Skipping {file.name}: {exc}")
                continue
            if evaluation.algorithm not in self.algorithms:
                continue
            self.evaluations.append(evaluation)
            if self.best_y is None or evaluation.best_y > self.best_y:
                self.best_y = evaluation.best_y
            if self.worst_y is None or evaluation.worst_y < self.worst_y:
                self.worst_y = evaluation.worst_y

    def get_average_performance_and_crashes(
        self,
        mode: str,
        algorithms: Sequence[str] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """Return the normalized statistics for the requested mode."""

        if self.best_y is None or self.worst_y is None:
            return {}
        focus = tuple(algorithms or self.algorithms)
        results: Dict[str, Dict[str, float]] = {}
        for evaluation in self.evaluations:
            if evaluation.mode != mode or evaluation.algorithm not in focus:
                continue
            evaluation.normalize_data(self.worst_y, self.best_y)
            results[evaluation.algorithm] = _summarize_evaluation(evaluation)
        return results

    def get_average_normalized_performance(self) -> List[SingleEvaluation]:
        """Return evaluations with normalized performance/crash curves."""

        if self.best_y is None or self.worst_y is None:
            return []
        normalized: List[SingleEvaluation] = []
        for evaluation in self.evaluations:
            evaluation.normalize_data(self.worst_y, self.best_y)
            normalized.append(evaluation)
        return normalized


class EvaluateMultipleAlgorithms(_EvaluationDirectory):
    """Load all synthetic-function evaluations contained in ``directory``."""

    evaluation_cls = SingleEvaluation


class EvaluateMultipleAlgorithms_GP(_EvaluationDirectory):
    """Load all GP evaluations for a single seed."""

    evaluation_cls = SingleEvaluation_GP


def _summarize_evaluation(evaluation: SingleEvaluation) -> Dict[str, float]:
    """Return statistics evaluated at ``dim * 10`` iterations."""

    idx = _iteration_index(len(evaluation.df_y), evaluation.dim)
    mean = float(evaluation.df_y["mean"].iloc[idx])
    std = float(evaluation.df_y["std"].iloc[idx])
    crash_mean = float(evaluation.crashes["mean"].iloc[idx] / (10 * evaluation.dim))
    crash_std = float(evaluation.crashes["std"].iloc[idx] / (10 * evaluation.dim))
    return {
        "mean": mean,
        "std": std,
        "mean_crashes": crash_mean,
        "std_crashes": crash_std,
    }


def evaluate_gp_seeds(
    dim: int,
    mode: str,
    algorithms: Sequence[str] | None = None,
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    seed_indices: Iterable[int] = range(10),
) -> Dict[str, Dict[str, float]]:
    """Aggregate GP results across the requested ``seed_indices``."""

    focus = tuple(algorithms or DEFAULT_ALGORITHMS)
    aggregated_performance: Dict[str, List[pd.DataFrame]] = {alg: [] for alg in focus}
    aggregated_crashes: Dict[str, List[pd.DataFrame]] = {alg: [] for alg in focus}
    root = Path(results_root)

    for seed in seed_indices:
        folder = root / GP_DIR_TEMPLATE.format(seed=seed, dim=dim)
        if not folder.is_dir():
            continue
        evaluator = EvaluateMultipleAlgorithms_GP(folder, focus)
        if evaluator.best_y is None or evaluator.worst_y is None:
            continue
        for evaluation in evaluator.evaluations:
            if evaluation.algorithm not in focus or evaluation.mode != mode:
                continue
            evaluation.normalize_data(evaluator.worst_y, evaluator.best_y)
            aggregated_performance[evaluation.algorithm].append(
                evaluation.df_y.drop(columns=["mean", "std"], errors="ignore").reset_index(drop=True)
            )
            aggregated_crashes[evaluation.algorithm].append(
                evaluation.crashes.drop(columns=["mean", "std"], errors="ignore").reset_index(drop=True)
            )

    results: Dict[str, Dict[str, float]] = {}
    for algorithm in focus:
        perf_samples = _concat_samples(aggregated_performance[algorithm])
        crash_samples = _concat_samples(aggregated_crashes[algorithm])
        if perf_samples is None or crash_samples is None:
            continue
        mean_perf = perf_samples.mean(axis=1)
        std_perf = perf_samples.std(axis=1)
        mean_crash = crash_samples.mean(axis=1)
        std_crash = crash_samples.std(axis=1)
        idx = _iteration_index(len(mean_perf), dim)
        results[algorithm] = {
            "mean": float(mean_perf.iloc[idx]),
            "std": float(std_perf.iloc[idx]),
            "mean_crashes": float(mean_crash.iloc[idx] / (10 * dim)),
            "std_crashes": float(std_crash.iloc[idx] / (10 * dim)),
        }
    return results


def _concat_samples(frames: List[pd.DataFrame]) -> pd.DataFrame | None:
    """Concatenate the provided sample DataFrames column-wise."""

    if not frames:
        return None
    return pd.concat(frames, axis=1)


def _resolve_function_dir(results_root: Path, function: str) -> Path:
    """Return the folder that stores the pickles for ``function``."""

    suffix = f"{function}_{FEASIBLE_SUFFIX}"
    direct = results_root / suffix
    if direct.is_dir():
        return direct

    for child in results_root.iterdir():
        if child.is_dir():
            candidate = child / suffix
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"Unable to locate results for '{function}' in '{results_root}'.")


def collect_table_results(
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    synthetic_functions: Sequence[str] = DEFAULT_SYNTHETIC_FUNCTIONS,
    gp_dimensions: Sequence[int] = DEFAULT_GP_DIMENSIONS,
    modes: Sequence[str] = DEFAULT_MODES,
    algorithms: Sequence[str] | None = None,
    gp_seed_indices: Iterable[int] = range(10),
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Collect the nested dictionary consumed by :func:`make_latex_table`."""

    focus = tuple(algorithms or DEFAULT_ALGORITHMS)
    root = Path(results_root)
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for mode in modes:
        summary[mode] = {}
        for dim in gp_dimensions:
            key = f"gp_{dim}D"
            summary[mode][key] = evaluate_gp_seeds(
                dim,
                mode,
                algorithms=focus,
                results_root=root,
                seed_indices=gp_seed_indices,
            )
        for function in synthetic_functions:
            folder = _resolve_function_dir(root, function)
            evaluator = EvaluateMultipleAlgorithms(folder, focus)
            summary[mode][function] = evaluator.get_average_performance_and_crashes(
                mode, focus
            )
    return summary


def make_latex_table(
    results: Mapping[str, Mapping[str, Mapping[str, Dict[str, float]]]],
    path: Path | str,
) -> None:
    """Write a small LaTeX table summarizing the provided ``results``."""

    if not results:
        raise ValueError("No results were provided, cannot create a table.")

    modes = list(results.keys())
    datasets = next(iter(results.values()))
    if len(datasets) == 0:
        raise ValueError("Results dictionary is missing the dataset entries.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("\\begin{table}[ht]\n")
        handle.write("\\centering\n")
        handle.write("\\begin{tabular}{|c|c|" + "c|" * len(modes) + "}\n")
        handle.write("\\hline\n")
        handle.write("Algorithm & Metric & " + " & ".join(modes) + " \\\\ \\hline\n")

        for algorithm in DEFAULT_ALGORITHMS:
            for metric, metric_key in (
                ("Performance", ("mean", "std")),
                ("Crashes", ("mean_crashes", "std_crashes")),
            ):
                row_cells = []
                for mode in modes:
                    values = []
                    for dataset in results[mode].values():
                        alg_values = dataset.get(algorithm)
                        if not alg_values:
                            continue
                        values.append(
                            (
                                alg_values[metric_key[0]],
                                alg_values[metric_key[1]],
                            )
                        )
                    if not values:
                        row_cells.append("--")
                        continue
                    mean_val = float(np.mean([val[0] for val in values]))
                    std_val = float(np.mean([val[1] for val in values]))
                    row_cells.append(f"{mean_val:.2f} \\pm {std_val:.2f}")
                handle.write(f"{algorithm} & {metric} & " + " & ".join(row_cells) + " \\\\ \\hline\n")

        handle.write("\\end{tabular}\n")
        handle.write("\\caption{crashPBO vs. EUBO summary}\n")
        handle.write("\\label{tab:crashpbo_eubo}\n")
        handle.write("\\end{table}\n")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for generating the LaTeX table."""

    parser = argparse.ArgumentParser(
        description="Create the crashPBO vs. EUBO LaTeX table."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, required=True, help="Destination *.tex file.")
    parser.add_argument(
        "--functions",
        nargs="+",
        default=list(DEFAULT_SYNTHETIC_FUNCTIONS),
        help="Synthetic functions to include.",
    )
    parser.add_argument(
        "--gp-dims",
        nargs="+",
        type=int,
        default=list(DEFAULT_GP_DIMENSIONS),
        help="GP dimensions to include.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        help="Acquisition comparison modes to summarize.",
    )
    parser.add_argument(
        "--gp-seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="GP seed indices to average.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point used when executing the module directly."""

    args = _parse_args()
    summary = collect_table_results(
        results_root=args.results_root,
        synthetic_functions=args.functions,
        gp_dimensions=args.gp_dims,
        modes=args.modes,
        gp_seed_indices=args.gp_seeds,
    )
    make_latex_table(summary, args.output)


if __name__ == "__main__":
    main()
