"""Evaluation utilities for BO experiments.

This module is a cleaned, shorter and documented rewrite of your script.
Functionality is preserved, but paths are now flexible and provided once.

Main features
-------------
* Read multiple pickle runs and compute mean/std of performance and crashes.
* Normalise performance to [0, 1] using global best/worst.
* Plot (performance, crashes, ranks) for algorithms and modes.
* Produce LaTeX tables summarising results.
* Works for both synthetic functions and GP benchmarks.

Usage example
-------------
>>> from pathlib import Path
>>> base = Path("results")
>>> eva = EvaluateMultipleAlgorithms(base/"hartmann_6D_feasiblep_0.5")
>>> fig, ax = plt.subplots(2, 1)
>>> eva.plot_average_performance(ax[0])
>>> eva.plot_average_cumulated_crashes(ax[1])
>>> plt.show()

For GP batches:
>>> ax0, ax1 = evaluate_gp(8, base_dir=base)

If you run this file as a script it will reproduce the original tables/plots
(using the same folder structure) without hard–coded paths inside functions.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------
PLOT_PARAMS = {"font.family": "serif"}
mpl.rcParams.update(PLOT_PARAMS)

COLOR_MAP = {
    "crashPBO": "blue",
    "EUBO": "green",
    "EI": "red",
    "MES": "orange",
    "sobolRandom": "black",
    "random": "black",
}
LINESTYLE_MAP = {
    "compare to best": "-",
    "compare to last": "--",
    "two new parameters": "-.",
}
MODE_LIST = ["compare to best", "compare to last", "two new parameters"]
ALG_DEFAULT = ["EUBO", "crashPBO", "EI", "MES", "random", "sobolRandom"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_every_second_row(df: pd.DataFrame, algorithm: str, mode: str) -> pd.DataFrame:
    """Mimic old behaviour: for EUBO/crashPBO + (best/last) drop every 2nd row after row 1.
    Returns a copy with reset index.
    """
    if algorithm in {"EUBO", "crashPBO"} and mode in {"compare to best", "compare to last"}:
        df = pd.concat([df.iloc[:2], df.iloc[2::2]], axis=0)
        df.index = range(len(df))
    return df


def _norm(series: pd.Series, worst: float, best: float) -> pd.Series:
    return (series - worst) / (best - worst)


def _rank_mean(resorted: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Average rank (ascending=False -> bigger is better) across seeds."""
    rank_dfs = {s: df.rank(axis=1, ascending=False) for s, df in resorted.items()}
    first = next(iter(rank_dfs.values()))
    mean_rank = pd.DataFrame(index=first.index, columns=first.columns, dtype=float)
    for col in first.columns:
        mean_rank[col] = sum(rank_dfs[s][col] for s in rank_dfs) / len(rank_dfs)
    return mean_rank


# ---------------------------------------------------------------------------
# Single evaluation file
# ---------------------------------------------------------------------------

@dataclass
class SingleEvaluation:
    """Hold data for one *.pkl result file and provide plotting helpers.

    Attributes
    ----------
    path : Path
        Pickle file path.
    function : str
    dim : int
    feasible_percentage : str
    algorithm : str
    mode : str
    data : dict[str, pd.DataFrame]
        Raw per-seed/replicate frames.
    df_y : pd.DataFrame
        Cumulative best y per iteration (columns: runs + mean/std).
    crashes : pd.DataFrame
        Cumulated crashes per iteration (columns: runs + mean/std).
    best_y, worst_y : float
        Per file extrema.
    color, line_style : str
        For plotting.
    """

    path: Path
    function: str = ""
    dim: int = 0
    feasible_percentage: str = ""
    algorithm: str = ""
    mode: str = ""
    data: Dict[str, pd.DataFrame] = None
    df_y: pd.DataFrame = None
    crashes: pd.DataFrame = None
    best_y: float = None
    worst_y: float = None
    color: str = "purple"
    line_style: str = "-"

    def __post_init__(self) -> None:
        parts = self.path.name.split("_")
        # "<fn>_<function>_<dim>D_<...>_<feasiblep>_<algo>_<mode words>.pkl"
        # original indexing: [1] func, [2] dim, [4] feasible %, [5] algo, [6..8] mode
        self.function = parts[1]
        self.dim = int(parts[2][:-1])
        self.feasible_percentage = parts[4]
        self.algorithm = parts[5]
        self.mode = " ".join(parts[6:9]).replace(".pkl", "")
        self.color = COLOR_MAP.get(self.algorithm, "purple")
        self.line_style = LINESTYLE_MAP.get(self.mode, "-.")
        self._read_pickle()

    # ------------------------------------------------------------------
    # IO / processing
    # ------------------------------------------------------------------
    def _read_pickle(self) -> None:
        with self.path.open("rb") as f:
            self.data = pickle.load(f)

        crashes = pd.DataFrame()
        df_y = pd.DataFrame()
        best_y = worst_y = None

        for key, df in self.data.items():
            df = df.copy()
            df["crash"] = df["crash"].cumsum()
            crashes[key] = df["crash"]

            # check nans (keep print behaviour but warn once)
            nan_rows = df["y"].isna()
            if nan_rows.any():
                for row in df.loc[nan_rows].iterrows():
                    print(row)  # keep original side-effect
            df["y"] = df["y"].replace(np.nan, -np.inf).cummax()
            df_y[key] = df["y"]

            best_y = df["y"].max() if best_y is None else max(best_y, df["y"].max())
            worst_y = df["y"].min() if worst_y is None else min(worst_y, df["y"].min())

        self.best_y = df_y.iloc[-1].max()
        self.worst_y = worst_y

        df_y["mean"] = df_y.mean(axis=1)
        df_y["std"] = df_y.std(axis=1)
        crashes["mean"] = crashes.mean(axis=1)
        crashes["std"] = crashes.std(axis=1)

        self.df_y = df_y
        self.crashes = crashes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def normalize_data(self, minimum: float, maximum: float) -> None:
        """Normalize cumulative performance columns in-place to [0,1]."""
        self.df_y = _skip_every_second_row(self.df_y, self.algorithm, self.mode)
        self.crashes = _skip_every_second_row(self.crashes, self.algorithm, self.mode)
        rng = maximum - minimum
        self.df_y = self.df_y.apply(lambda c: (c - minimum) / rng)

    # ----------------------------- plotting -----------------------------
    def plot_average_performance(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        ax = ax or plt.gca()
        df = _skip_every_second_row(self.df_y, self.algorithm, self.mode)
        ax.plot(df["mean"], label=self.algorithm, color=self.color, linestyle=self.line_style, marker=".")
        ax.fill_between(df.index, df["mean"] - df["std"], df["mean"] + df["std"], alpha=0.15, color=self.color)
        return ax

    def plot_average_cumulated_crashes(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        ax = ax or plt.gca()
        df = _skip_every_second_row(self.crashes, self.algorithm, self.mode)
        ax.plot(df["mean"], label=self.algorithm, color=self.color, linestyle=self.line_style)
        ax.fill_between(df.index, df["mean"] - df["std"], df["mean"] + df["std"], alpha=0.15, color=self.color)
        return ax

    def plot_density_of_evaluated_points(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        ax = ax or plt.gca()
        for df in self.data.values():
            ax.scatter(df["X0"], df["X1"], label=f"{self.algorithm} {self.mode}", color=self.color, alpha=0.15)
        return ax

    def plot_normalized_performance(
        self,
        ax: Optional[plt.Axes],
        global_best: float,
        global_worst: float,
        *,
        color: Optional[str] = None,
        line_style: Optional[str] = None,
    ) -> plt.Axes:
        ax = ax or plt.gca()
        df = _skip_every_second_row(self.df_y, self.algorithm, self.mode)
        color = color or self.color
        line_style = line_style or self.line_style
        mean = _norm(df["mean"], global_worst, global_best)
        std = df["std"] / (global_best - global_worst)
        ax.plot(mean, label=f"{self.algorithm} {self.mode}", color=color, linestyle=line_style)
        ax.fill_between(df.index, mean - std, mean + std, alpha=0.15, color=color)
        return ax


# GP variant ---------------------------------------------------------------

@dataclass
class SingleEvaluationGP(SingleEvaluation):
    """File name layout differs for GP runs; override parsing only."""

    def __post_init__(self) -> None:  # type: ignore[override]
        parts = self.path.name.split("_")
        # pattern: gp_seed_<seed>_<dim>D_feasiblep_<p>_<algo>_<mode words>.pkl
        self.function = parts[1]  # actually 'seed'
        self.seed = parts[3]
        self.dim = int(parts[4][:-1])
        self.feasible_percentage = parts[6]
        self.algorithm = parts[7]
        self.mode = " ".join(parts[8:11]).replace(".pkl", "")
        self.color = COLOR_MAP.get(self.algorithm, "purple")
        self.line_style = LINESTYLE_MAP.get(self.mode, "-.")
        self._read_pickle()


# ---------------------------------------------------------------------------
# Collections of evaluations (folders)
# ---------------------------------------------------------------------------

class EvaluateMultipleAlgorithms:
    """Handle a folder with many *.pkl runs for one synthetic test function."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.function = self.path.name.split("_")[0]
        self.evaluations: List[SingleEvaluation] = []
        self.best_y: float | None = None
        self.worst_y: float | None = None
        self._extract_files()

    # ------------------------------------------------------------------
    def _extract_files(self) -> None:
        for file in self.path.glob("*.pkl"):
            ev = SingleEvaluation(file)
            self.evaluations.append(ev)
            self.best_y = ev.best_y if self.best_y is None else max(self.best_y, ev.best_y)
            self.worst_y = ev.worst_y if self.worst_y is None else min(self.worst_y, ev.worst_y)

    # ------------------------------------------------------------------
    def get_average_performance_and_crashes(
        self,
        mode: str,
        algorithms: Sequence[str] = ALG_DEFAULT,
    ) -> Dict[str, Dict[str, float]]:
        """Return metrics at iteration dim*10, matching original behaviour."""
        results: Dict[str, Dict[str, float]] = {}
        for ev in self.evaluations:
            if ev.mode != mode or ev.algorithm not in algorithms:
                continue
            ev.normalize_data(self.worst_y, self.best_y)  # type: ignore[arg-type]
            ev.df_y["mean"] = ev.df_y.mean(axis=1)
            ev.df_y["std"] = ev.df_y.std(axis=1)
            ev.crashes["mean"] = ev.crashes.mean(axis=1)
            ev.crashes["std"] = ev.crashes.std(axis=1)
            idx = ev.dim * 10
            results[ev.algorithm] = {
                "mean": float(ev.df_y["mean"].iloc[idx]),
                "std": float(ev.df_y["std"].iloc[idx]),
                "mean_crashes": float(ev.crashes["mean"].iloc[idx]) / (10 * ev.dim),
                "std_crashes": float(ev.crashes["std"].iloc[idx]) / (10 * ev.dim),
            }
        return results

    # ------------------------------------------------------------------
    def _collect_dfs(
        self, mode: str, algorithms: Sequence[str]
    ) -> Dict[str, pd.DataFrame]:
        df_dict: Dict[str, pd.DataFrame] = {}
        for ev in self.evaluations:
            if ev.mode == mode and ev.algorithm in algorithms:
                ev.normalize_data(self.worst_y, self.best_y)  # type: ignore[arg-type]
                df_dict[ev.algorithm] = ev.df_y
        return df_dict

    def get_ranking_of_algorithms(
        self, mode: str = "compare to best", algorithms: Sequence[str] = ("EUBO", "crashPBO", "EI", "random")
    ) -> pd.DataFrame:
        """Ranking (mean across seeds) of algorithms per iteration."""
        df_dict = self._collect_dfs(mode, algorithms)
        # build per-seed frames
        resorted = {}
        for seed in range(len(algorithms)):
            tmp = pd.DataFrame({algo: df.iloc[:, seed] for algo, df in df_dict.items()}, columns=algorithms)
            resorted[seed] = tmp
        return _rank_mean(resorted)

    def get_ranking_modes(self, algorithm: str = "crashPBO") -> pd.DataFrame:
        df_dict: Dict[str, pd.DataFrame] = {}
        for ev in self.evaluations:
            if ev.algorithm == algorithm:
                ev.normalize_data(self.worst_y, self.best_y)  # type: ignore[arg-type]
                df_dict[ev.mode] = ev.df_y
        resorted = {}
        for seed in range(20):
            tmp = pd.DataFrame({m: df.iloc[:, seed] for m, df in df_dict.items()}, columns=MODE_LIST)
            resorted[seed] = tmp
        return _rank_mean(resorted)

    # ------------------------------------------------------------------
    # plots
    def plot_average_performance(
        self,
        ax: Optional[plt.Axes] = None,
        mode: str = "compare to best",
        algorithms: Sequence[str] = ALG_DEFAULT,
        *,
        color: Optional[str] = None,
        line_style: Optional[str] = None,
    ) -> plt.Axes:
        ax = ax or plt.gca()
        for ev in self.evaluations:
            if ev.algorithm in algorithms and ev.mode == mode:
                ev.plot_normalized_performance(ax, self.best_y, self.worst_y, color=color, line_style=line_style)  # type: ignore[arg-type]
        return ax

    def plot_average_cumulated_crashes(
        self,
        ax: Optional[plt.Axes] = None,
        mode: str = "compare to best",
        algorithms: Sequence[str] = ALG_DEFAULT,
    ) -> plt.Axes:
        ax = ax or plt.gca()
        for ev in self.evaluations:
            if ev.algorithm in algorithms and ev.mode == mode:
                ev.plot_average_cumulated_crashes(ax)
        return ax

    def create_plots(self) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        self.plot_average_performance(axes[0])
        self.plot_average_cumulated_crashes(axes[1])
        plt.tight_layout()
        plt.show()


class EvaluateMultipleAlgorithmsGP(EvaluateMultipleAlgorithms):
    """Folder with GP runs: file parsing differs, logic identical."""

    def _extract_files(self) -> None:  # type: ignore[override]
        for file in self.path.glob("*.pkl"):
            ev = SingleEvaluationGP(file)
            self.evaluations.append(ev)
            self.best_y = ev.best_y if self.best_y is None else max(self.best_y, ev.best_y)
            self.worst_y = ev.worst_y if self.worst_y is None else min(self.worst_y, ev.worst_y)

    def get_average_normalized_performance(self) -> List[SingleEvaluationGP]:
        for ev in self.evaluations:
            ev.normalize_data(self.worst_y, self.best_y)  # type: ignore[arg-type]
        return self.evaluations


# ---------------------------------------------------------------------------
# Public convenience functions (kept from original script, simplified)
# ---------------------------------------------------------------------------

def evaluate_gp(
    dim: int,
    ax1: Optional[plt.Axes] = None,
    ax2: Optional[plt.Axes] = None,
    mode: str = "compare to best",
    algorithms: Sequence[str] = ALG_DEFAULT,
    *,
    color: Optional[str] = None,
    line_style: Optional[str] = None,
    base_dir: Path | str = "results",
) -> Tuple[plt.Axes, plt.Axes]:
    """Plot mean±std performance + crashes across seeds for GP dim.

    Returns the two axes.
    """
    base_dir = Path(base_dir)
    axes = [ax1 or plt.subplot(2, 1, 1), ax2 or plt.subplot(2, 1, 2)]
    results = {}

    for algo in algorithms:
        perf_df = pd.DataFrame()
        crash_df = pd.DataFrame()
        algo_color, algo_ls = color, line_style
        for seed in range(10):
            path = base_dir / f"gp_seed_{seed}_{dim}D_feasiblep_0.5"
            evm = EvaluateMultipleAlgorithmsGP(path)
            for ev in evm.get_average_normalized_performance():
                if ev.algorithm != algo or ev.mode != mode:
                    continue
                perf_df = pd.concat([perf_df, ev.df_y.drop(columns=["mean", "std"], errors="ignore")], axis=1)
                crash_df = pd.concat([crash_df, ev.crashes.drop(columns=["mean", "std"], errors="ignore")], axis=1)
                algo_color = algo_color or ev.color
                algo_ls = algo_ls or ev.line_style
        mean_p, std_p = perf_df.mean(axis=1), perf_df.std(axis=1)
        mean_c, std_c = crash_df.mean(axis=1), crash_df.std(axis=1)
        axes[0].plot(mean_p, label=algo, color=algo_color, linestyle=algo_ls)
        axes[0].fill_between(mean_p.index, mean_p - std_p, mean_p + std_p, alpha=0.15, color=algo_color)
        axes[1].plot(mean_c, label=algo, color=algo_color, linestyle=algo_ls)
        axes[1].fill_between(mean_c.index, mean_c - std_c, mean_c + std_c, alpha=0.15, color=algo_color)

    axes[0].set_xlim(0, dim * 10)
    axes[1].set_xlim(0, dim * 10)
    return axes[0], axes[1]


def evaluate_gp_performance(
    dim: int,
    ax: Optional[plt.Axes] = None,
    mode: str = "compare to best",
    algorithms: Sequence[str] = ALG_DEFAULT,
    *,
    color: Optional[str] = None,
    line_style: Optional[str] = None,
    base_dir: Path | str = "results",
) -> plt.Axes:
    ax = ax or plt.gca()
    base_dir = Path(base_dir)
    for algo in algorithms:
        perf_df = pd.DataFrame()
        for seed in range(10):
            path = base_dir / f"gp_seed_{seed}_{dim}D_feasiblep_0.5"
            evm = EvaluateMultipleAlgorithmsGP(path)
            for ev in evm.get_average_normalized_performance():
                if ev.algorithm == algo and ev.mode == mode:
                    perf_df = pd.concat([perf_df, ev.df_y.drop(columns=["mean", "std"], errors="ignore")], axis=1)
                    c = color or ev.color
                    ls = line_style or ev.line_style
        mean_p, std_p = perf_df.mean(axis=1), perf_df.std(axis=1)
        ax.plot(mean_p, label=algo, color=c, linestyle=ls)
        ax.fill_between(mean_p.index, mean_p - std_p, mean_p + std_p, alpha=0.15, color=c)
    ax.set_xlim(0, dim * 10)
    return ax


def evaluate_gp_seeds(
    dim: int,
    mode: str,
    algorithms: Sequence[str] = ALG_DEFAULT,
    *,
    base_dir: Path | str = "results",
) -> Dict[str, Dict[str, float]]:
    """Return summary numbers (performance/crashes at dim*10) per algorithm."""
    base_dir = Path(base_dir)
    out: Dict[str, Dict[str, float]] = {}
    for algo in algorithms:
        perf_df = pd.DataFrame()
        crash_df = pd.DataFrame()
        color = line_style = None
        for seed in range(10):
            path = base_dir / f"gp_seed_{seed}_{dim}D_feasiblep_0.5"
            evm = EvaluateMultipleAlgorithmsGP(path)
            for ev in evm.get_average_normalized_performance():
                if ev.algorithm == algo and ev.mode == mode:
                    perf_df = pd.concat([perf_df, ev.df_y.drop(columns=["mean", "std"], errors="ignore")], axis=1)
                    crash_df = pd.concat([crash_df, ev.crashes.drop(columns=["mean", "std"], errors="ignore")], axis=1)
                    color = color or ev.color
                    line_style = line_style or ev.line_style
        idx = 10 * dim
        out[algo] = {
            "mean": float(perf_df.mean(axis=1).iloc[idx]),
            "std": float(perf_df.std(axis=1).iloc[idx]),
            "mean_crashes": float(crash_df.mean(axis=1).iloc[idx]) / (10 * dim),
            "std_crashes": float(crash_df.std(axis=1).iloc[idx]) / (10 * dim),
            "color": color or "black",
            "line_style": line_style or "-",
        }
    return out


# ---------------------------- Ranking helpers ----------------------------

def get_ranking_for_synthetic_functions(base_dir: Path | str = "results") -> pd.DataFrame:
    base_dir = Path(base_dir)
    mean_rank: Dict[str, pd.DataFrame] = {}
    for func in ["ackley_2D", "hartmann_6D", "branin_2D", "cosine8_8D"]:
        eva = EvaluateMultipleAlgorithms(base_dir / f"{func}_feasiblep_0.5")
        mr = eva.get_ranking_of_algorithms(mode="compare to best", algorithms=["EUBO", "crashPBO", "MES", "random"])
        mr["rank_index"] = mr.index / eva.evaluations[0].dim
        mr = mr.set_index("rank_index")
        X = np.linspace(0, 10, 11)
        mr = mr.reindex(mr.index.union(X)).interpolate("values").loc[X]
        mean_rank[func] = mr

    out = pd.DataFrame(columns=["EUBO", "crashPBO", "MES", "random"])
    for algo in out.columns:
        out[algo] = sum(mean_rank[f][algo] for f in mean_rank) / len(mean_rank)
    return out


def get_ranking_for_synthetic_functions_modes(base_dir: Path | str = "results") -> pd.DataFrame:
    base_dir = Path(base_dir)
    mean_rank: Dict[str, pd.DataFrame] = {}
    for func in ["ackley_2D", "hartmann_6D", "branin_2D", "cosine8_8D"]:
        eva = EvaluateMultipleAlgorithms(base_dir / f"{func}_feasiblep_0.5")
        mr = eva.get_ranking_modes(algorithm="crashPBO")
        mr["rank_index"] = mr.index / eva.evaluations[0].dim
        mr = mr.set_index("rank_index")
        X = np.linspace(0, 10, 11)
        mr = mr.reindex(mr.index.union(X)).interpolate("values").loc[X]
        mean_rank[func] = mr

    out = pd.DataFrame(columns=MODE_LIST)
    for mode in MODE_LIST:
        out[mode] = sum(mean_rank[f][mode] for f in mean_rank) / len(mean_rank)
    return out


def ranking_plot(ax: plt.Axes, base_dir: Path | str = "results") -> plt.Axes:
    base_dir = Path(base_dir)
    mean_rank_dim: Dict[int, pd.DataFrame] = {}
    for dim in [1,2,3,4,5,6,7,8]:
        seed_rank: Dict[int, pd.DataFrame] = {}
        for seed in range(10):
            eva = EvaluateMultipleAlgorithmsGP(base_dir / f"gp_seed_{seed}_{dim}D_feasiblep_0.5")
            mr = eva.get_ranking_of_algorithms(mode="compare to best", algorithms=["EUBO", "crashPBO", "MES", "random"])
            mr["rank_index"] = mr.index / dim
            mr = mr.set_index("rank_index")
            X = np.linspace(0, 10, 11)
            mr = mr.reindex(mr.index.union(X)).interpolate("values").loc[X]
            seed_rank[seed] = mr
        # average across seeds
        mean_rank_dim[dim] = sum(seed_rank[s] for s in seed_rank) / len(seed_rank)

    mean_rank_all = sum(mean_rank_dim[d] for d in [1,2,3,4,5,6,7,8]) / 8
    # merge with synthetic
    syn = get_ranking_for_synthetic_functions(base_dir)
    for algo in ["EUBO", "crashPBO", "MES", "random"]:
        mean_rank_all[algo] = (mean_rank_all[algo] + 0.5 * syn[algo]) / 1.5

    ax.plot(mean_rank_all["EUBO"], label="EUBO", color="green")
    ax.plot(mean_rank_all["crashPBO"], label="crashPBO", color="blue")
    ax.plot(mean_rank_all["MES"], label="MES", color="orange")
    ax.plot(mean_rank_all["random"], label="random", color="black")
    return ax


def ranking_plot_modes(ax: plt.Axes, base_dir: Path | str = "results") -> plt.Axes:
    base_dir = Path(base_dir)
    mean_rank_dim: Dict[int, pd.DataFrame] = {}
    for dim in [1,2,3,4,5,6,7,8]:
        seed_rank: Dict[int, pd.DataFrame] = {}
        for seed in range(10):
            eva = EvaluateMultipleAlgorithmsGP(base_dir / f"gp_seed_{seed}_{dim}D_feasiblep_0.5")
            mr = eva.get_ranking_modes(algorithm="crashPBO")
            mr["rank_index"] = mr.index / dim
            mr = mr.set_index("rank_index")
            X = np.linspace(0, 10, 11)
            mr = mr.reindex(mr.index.union(X)).interpolate("values").loc[X]
            seed_rank[seed] = mr
        mean_rank_dim[dim] = sum(seed_rank[s] for s in seed_rank) / len(seed_rank)

    mean_rank_all = sum(mean_rank_dim[d] for d in [1,2,3,4,5,6,7,8]) / 8
    syn = get_ranking_for_synthetic_functions_modes(base_dir)
    for mode in MODE_LIST:
        mean_rank_all[mode] = (mean_rank_all[mode] + 0.5 * syn[mode]) / 1.5

    ax.plot(mean_rank_all["compare to best"], label="best", color="#00549F")
    ax.plot(mean_rank_all["compare to last"], label="last", color="#646567")
    ax.plot(mean_rank_all["two new parameters"], label="two new", color="#E30066")
    return ax


# Placeholders retained ----------------------------------------------------

def place_holder_plot(ax: plt.Axes) -> plt.Axes:
    ax.plot([0, 1], [0, 1], label="best", color="#00549F")
    ax.plot([0, 1], [0, 1], label="last", color="#000000")
    ax.plot([0, 1], [0, 1], label="two new", color="#E30066")
    ax.set_title("Mean Rank for GPs and Synthetic Functions")
    return ax


def place_holder_algorithms(ax: plt.Axes) -> plt.Axes:
    ax.plot([0, 1], [0, 1], label="EUBO", color="green")
    ax.plot([0, 1], [0, 1], label="crashPBO", color="blue")
    ax.plot([0, 1], [0, 1], label="MES", color="red")
    ax.plot([0, 1], [0, 1], label="random", color="black")
    return ax


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def make_latex_table(results: Mapping[str, Mapping[str, Mapping[str, Dict[str, float]]]], path: Path | str) -> None:
    """Write LaTeX table identical to original logic.

    Parameters
    ----------
    results : dict
        Outer keys: mode; inner keys: dataset name; leaf: algo metrics dict.
    path : str or Path
        Output .tex path.
    """
    path = Path(path)
    with path.open("w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|}\\hline\n")
        f.write("Algorithm & Metric & " + " & ".join(results.keys()) + " \\ \\ \n\\hline\n")
        for algorithm in ["crashPBO", "EUBO"]:
            for metric in ["performance", "crashes"]:
                row_data = []
                for mode in results.keys():
                    means = []
                    stds = []
                    for key in results[mode].keys():
                        d = results[mode][key][algorithm]
                        if metric == "performance":
                            means.append(d["mean"]) ; stds.append(d["std"])
                        else:
                            means.append(d["mean_crashes"]) ; stds.append(d["std_crashes"])
                    row_data.append(f"{np.mean(means):.2f} ± {np.mean(stds):.2f}")
                f.write(f"{algorithm} - {metric.capitalize()} & " + " & ".join(row_data) + " \\ \\ \n\\hline\n")
        f.write("\\end{tabular}\n\\caption{Performance and crashes summary}\n\\label{tab:results}\n\\end{table}\n")


# ---------------------------------------------------------------------------
# Paper style plots
# ---------------------------------------------------------------------------

def generate_paper_plot(base_dir: Path | str = "results") -> None:
    base_dir = Path(base_dir)
    fig, ax = plt.subplots(2, 5, figsize=(7, 3), sharex="col", sharey="row")
    for i, function in enumerate(["ackley_2D", "hartmann_6D"]):
        eva = EvaluateMultipleAlgorithms(base_dir / f"{function}_feasiblep_0.5")
        ax[0][i] = eva.plot_average_performance(ax[0][i], mode="compare to best")
        ax[1][i] = eva.plot_average_cumulated_crashes(ax[1][i], mode="compare to best")
    for i, dim in enumerate([2, 4, 8]):
        ax[0][i + 2], ax[1][i + 2] = evaluate_gp(dim, ax[0][i + 2], ax[1][i + 2], mode="compare to best", base_dir=base_dir)
    ax[1][0].legend(loc="lower left", bbox_to_anchor=(0, -0.5), fontsize=12, ncol=6)
    ax[1][0].set_ylim(0, 40)
    ax[0][0].set_ylabel("Performance", fontsize=12)
    ax[1][0].set_ylabel("Average Crashes", fontsize=12)
    ax[0][0].set_title("Ackley")
    ax[0][1].set_title("Hartmann")
    ax[0][2].set_title("GP 2D")
    ax[0][3].set_title("GP 4D")
    ax[0][4].set_title("GP 8D")
    ax[0][0].set_xlabel("Iterations", fontsize=12)
    plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.2, top=0.9, left=0.1, right=0.99)
    plt.show()


# ---------------------------------------------------------------------------
# Main script block – mirrors original behaviour but with base_dir arg
# ---------------------------------------------------------------------------

def _main(base_dir: Path | str = "results") -> None:
    base_dir = Path(base_dir)
    # Full results (GP + synthetic)
    results_all: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for mode in MODE_LIST:
        results_all[mode] = {}
        for dim in [1,2,3,4,5,6,7,8]:
            results_all[mode][f"gp_{dim}D"] = evaluate_gp_seeds(dim, mode, algorithms=["EUBO", "crashPBO"], base_dir=base_dir)
        for func in ["ackley_2D", "hartmann_6D", "branin_2D", "cosine8_8D"]:
            eva = EvaluateMultipleAlgorithms(base_dir / f"{func}_feasiblep_0.5")
            results_all[mode][func] = eva.get_average_performance_and_crashes(mode, algorithms=["EUBO", "crashPBO"])
    with (base_dir / "results_all.pkl").open("wb") as f:
        pickle.dump(results_all, f)
    make_latex_table(results_all, base_dir / "latex_table_all.tex")

    # GP only
    results_gp: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for mode in MODE_LIST:
        results_gp[mode] = {f"gp_{d}D": evaluate_gp_seeds(d, mode, algorithms=["EUBO", "crashPBO"], base_dir=base_dir) for d in [1,2,3,4,5,6,7,8]}
    with (base_dir / "results_gp.pkl").open("wb") as f:
        pickle.dump(results_gp, f)
    make_latex_table(results_gp, base_dir / "latex_table_gp.tex")

    # Synthetic only
    results_syn: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for mode in MODE_LIST:
        results_syn[mode] = {}
        for func in ["ackley_2D", "hartmann_6D", "branin_2D", "cosine8_8D"]:
            eva = EvaluateMultipleAlgorithms(base_dir / f"{func}_feasiblep_0.5")
            results_syn[mode][func] = eva.get_average_performance_and_crashes(mode, algorithms=["EUBO", "crashPBO"])
    with (base_dir / "results_synthetic.pkl").open("wb") as f:
        pickle.dump(results_syn, f)
    make_latex_table(results_syn, base_dir / "latex_table_synthetic.tex")

    # Example composite figure (3 columns x 2 rows) – unchanged layout
    fig, axs = plt.subplots(2, 3, figsize=(7.17, 3), sharex="col")
    eva = EvaluateMultipleAlgorithms(base_dir / "hartmann_6D_feasiblep_0.5")
    axs[0][0] = eva.plot_average_performance(axs[0][0], mode="compare to best", algorithms=["crashPBO"], color="#00549F", line_style="solid")
    axs[0][0] = eva.plot_average_performance(axs[0][0], mode="compare to last", algorithms=["crashPBO"], color="#646567", line_style="solid")
    axs[0][0] = eva.plot_average_performance(axs[0][0], mode="two new parameters", algorithms=["crashPBO"], color="#E30066", line_style="solid")

    axs[0][1] = evaluate_gp_performance(8, axs[0][1], mode="compare to best", algorithms=["crashPBO"], color="#00549F", line_style="solid", base_dir=base_dir)
    axs[0][1] = evaluate_gp_performance(8, axs[0][1], mode="compare to last", algorithms=["crashPBO"], color="#646567", line_style="solid", base_dir=base_dir)
    axs[0][1] = evaluate_gp_performance(8, axs[0][1], mode="two new parameters", algorithms=["crashPBO"], color="#E30066", line_style="solid", base_dir=base_dir)

    axs[0][2] = ranking_plot_modes(axs[0][2], base_dir)
    axs[0][2].legend(loc="lower left", bbox_to_anchor=(0, -0.5), fontsize=12, ncol=1)

    axs[1][0] = eva.plot_average_performance(axs[1][0], mode="compare to best", algorithms=["crashPBO", "EUBO", "MES", "random"])
    axs[1][1] = evaluate_gp_performance(8, axs[1][1], mode="compare to best", algorithms=["crashPBO", "EUBO", "MES", "random"], base_dir=base_dir)
    axs[1][2] = ranking_plot(axs[1][2], base_dir)

    axs[0][0].set_title("Hartmann 6D", fontsize=12)
    axs[0][1].set_title("GP 8D", fontsize=12)
    axs[0][2].set_title("Average Rank", fontsize=12)
    for a in [axs[0][0], axs[1][0], axs[0][1], axs[1][1]]:
        a.set_ylabel("Performance", fontsize=12)
    axs[0][2].set_ylabel("Rank", fontsize=12)
    axs[1][2].set_ylabel("Rank", fontsize=12)
    axs[1][0].set_xlabel("Iterations", fontsize=12)
    axs[1][1].set_xlabel("Iterations", fontsize=12)
    axs[1][2].set_xlabel("Iterations / 10 * dim", fontsize=12)

    leg = axs[0][2].legend(loc="center right", bbox_to_anchor=(2.2, 0.5), fontsize=11, ncol=1, title="Comparison \n    modes", title_fontsize=11)
    leg._legend_box.align = "right"
    axs[1][2].legend(loc="center right", bbox_to_anchor=(2.2, 0.5), fontsize=11, ncol=1, title="Algorithms", title_fontsize=11)

    for a in axs.flat:
        a.tick_params(axis="both", which="major", labelsize=12)

    plt.subplots_adjust(hspace=0.1, wspace=0.6, bottom=0.16, top=0.9, left=0.1, right=0.8)
    plt.show()


if __name__ == "__main__":
    _main("results") # Change to your results directory
