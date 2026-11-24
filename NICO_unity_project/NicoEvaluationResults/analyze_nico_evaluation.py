#!/usr/bin/env python3
"""Compare per-episode CSVs produced by NicoAgentReach2 fixed evaluation mode.

Usage:
    python analyze_nico_evaluation.py /path/to/NicoEvaluationResults
    python analyze_nico_evaluation.py run_a.csv run_b.csv

Requires: pandas, numpy, matplotlib
    python -m pip install pandas numpy matplotlib
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "Episode",
    "TargetSeed",
    "TargetWorldX",
    "TargetWorldY",
    "TargetWorldZ",
    "InitialTargetDistanceM",
    "DecisionSteps",
    "LegacyReachAndHoldSuccess",
    "CumulativeReward",
    "StableHoldSuccess",
    "SettlingTimeSteps",
    "HoldDistanceStdDevValid",
    "HoldDistanceStdDevM",
    "HoldDistanceSampleCount",
    "ReachedReachDistance",
    "JerkValid",
    "CartesianDimensionlessSquaredJerk",
    "SPARCValid",
    "SPARC",
    "ReachDurationSeconds",
    "ReachAmplitudeMeters",
    "RawTrajectorySamples",
    "AnalysisSamples",
    "PhysicsSamplingFrequencyHz",
    "FixedDeltaTime",
    "TimeScale",
]

METRICS = {
    "Stable-hold success rate": ("StableHoldSuccess", None),
    "Settling time (successful episodes only)": ("SettlingTimeSteps", "StableHoldSuccess"),
    "Settling time (all episodes; failure sentinel = 1)": ("SettlingTimeSteps", None),
    "Hold distance standard deviation (m)": ("HoldDistanceStdDevM", "HoldDistanceStdDevValid"),
    "Cartesian dimensionless squared jerk": ("CartesianDimensionlessSquaredJerk", "JerkValid"),
    "SPARC": ("SPARC", "SPARCValid"),
    "Initial target distance (m)": ("InitialTargetDistanceM", None),
    "Episode decisions": ("DecisionSteps", None),
    "Episode cumulative reward (diagnostic only)": ("CumulativeReward", None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more per-episode CSV files, or directories containing NicoEvalEpisodes_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <first input directory>/Analysis)",
    )
    return parser.parse_args()


def collect_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("NicoEvalEpisodes_*.csv")))
        elif path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)
        else:
            warnings.warn(f"Skipping unsupported or missing input: {path}")

    # Keep first occurrence of each absolute path.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def load_run(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"No episode rows in {path}")

    missing = [column for column in ("RunLabel", "Episode", "TargetSeed") if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["RunLabel"] = frame["RunLabel"].fillna(path.stem).astype(str)
    frame = frame.sort_values("Episode").reset_index(drop=True)
    frame["_source_file"] = path.name
    return frame


def check_target_sets(runs: list[pd.DataFrame]) -> bool:
    """Return True if all runs appear to use the same ordered target set."""
    reference = runs[0]
    target_columns = ["Episode", "TargetWorldX", "TargetWorldY", "TargetWorldZ"]
    all_match = True

    for candidate in runs[1:]:
        same_seed = (
            reference["TargetSeed"].nunique() == 1
            and candidate["TargetSeed"].nunique() == 1
            and int(reference["TargetSeed"].iloc[0]) == int(candidate["TargetSeed"].iloc[0])
        )
        if not same_seed:
            warnings.warn(
                f"Target seed differs between {reference['_source_file'].iloc[0]} "
                f"and {candidate['_source_file'].iloc[0]}. Comparisons are not paired."
            )
            all_match = False

        if len(reference) != len(candidate):
            warnings.warn(
                f"Episode counts differ ({len(reference)} vs {len(candidate)}): "
                f"{reference['_source_file'].iloc[0]} / {candidate['_source_file'].iloc[0]}"
            )
            all_match = False
            continue

        left = reference[target_columns].to_numpy(dtype=float)
        right = candidate[target_columns].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-6, equal_nan=True):
            warnings.warn(
                f"The ordered target positions differ between "
                f"{reference['_source_file'].iloc[0]} and {candidate['_source_file'].iloc[0]}. "
                "Check spawn-volume geometry, seed, region fraction, boundary margin and scene transforms."
            )
            all_match = False

    return all_match


def selected_values(frame: pd.DataFrame, value_column: str, filter_column: str | None) -> np.ndarray:
    if value_column not in frame.columns:
        return np.array([], dtype=float)

    selected = frame
    if filter_column:
        if filter_column not in frame.columns:
            return np.array([], dtype=float)
        selected = selected[selected[filter_column] == 1]

    values = pd.to_numeric(selected[value_column], errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return values


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else math.nan


def describe(values: np.ndarray) -> dict[str, float | int]:
    if not values.size:
        return {"N": 0, "Mean": math.nan, "Median": math.nan, "SampleStdDev": math.nan,
                "P25": math.nan, "P75": math.nan, "P90": math.nan}

    return {
        "N": int(values.size),
        "Mean": float(np.mean(values)),
        "Median": float(np.median(values)),
        "SampleStdDev": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "P25": percentile(values, 25),
        "P75": percentile(values, 75),
        "P90": percentile(values, 90),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    margin = (z / denominator) * math.sqrt((p * (1.0 - p) / trials) + (z * z / (4.0 * trials * trials)))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def build_summary(runs: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame in runs:
        label = str(frame["RunLabel"].iloc[0])
        source = str(frame["_source_file"].iloc[0])

        for metric_name, (value_column, filter_column) in METRICS.items():
            values = selected_values(frame, value_column, filter_column)
            result = describe(values)
            rows.append({"RunLabel": label, "SourceFile": source, "Metric": metric_name, **result})

        success_values = selected_values(frame, "StableHoldSuccess", None)
        trials = int(success_values.size)
        successes = int(np.sum(success_values == 1))
        low, high = wilson_interval(successes, trials)
        rows.append({
            "RunLabel": label,
            "SourceFile": source,
            "Metric": "Stable-hold success rate (95% Wilson CI lower)",
            "N": trials,
            "Mean": low,
            "Median": math.nan,
            "SampleStdDev": math.nan,
            "P25": math.nan,
            "P75": math.nan,
            "P90": math.nan,
        })
        rows.append({
            "RunLabel": label,
            "SourceFile": source,
            "Metric": "Stable-hold success rate (95% Wilson CI upper)",
            "N": trials,
            "Mean": high,
            "Median": math.nan,
            "SampleStdDev": math.nan,
            "P25": math.nan,
            "P75": math.nan,
            "P90": math.nan,
        })

    return pd.DataFrame(rows)


def make_boxplot(runs: list[pd.DataFrame], value_column: str, filter_column: str | None,
                 title: str, ylabel: str, output_path: Path) -> None:
    data: list[np.ndarray] = []
    labels: list[str] = []
    for frame in runs:
        values = selected_values(frame, value_column, filter_column)
        if values.size:
            data.append(values)
            labels.append(str(frame["RunLabel"].iloc[0]))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    matplotlib_version = tuple(int(part) for part in matplotlib.__version__.split(".")[:2])
    label_argument = "tick_labels" if matplotlib_version >= (3, 9) else "labels"
    ax.boxplot(data, showmeans=True, showfliers=True, **{label_argument: labels})
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_success_plot(runs: list[pd.DataFrame], output_path: Path) -> None:
    labels: list[str] = []
    rates: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []

    for frame in runs:
        values = selected_values(frame, "StableHoldSuccess", None)
        if not values.size:
            continue
        successes = int(np.sum(values == 1))
        trials = int(values.size)
        rate = successes / trials
        low, high = wilson_interval(successes, trials)
        labels.append(str(frame["RunLabel"].iloc[0]))
        rates.append(rate)
        lower_errors.append(rate - low)
        upper_errors.append(high - rate)

    if not rates:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    positions = np.arange(len(labels))
    ax.bar(positions, rates)
    ax.errorbar(positions, rates, yerr=np.array([lower_errors, upper_errors]), fmt="none", capsize=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Stable-hold success rate")
    ax.set_title("Fixed-set evaluation: stable-hold success rate (95% Wilson intervals)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    files = collect_files(args.inputs)
    if not files:
        print("No NicoEvalEpisodes_*.csv files found. Select the folder containing the per-episode files.", file=sys.stderr)
        return 2

    runs: list[pd.DataFrame] = []
    for file_path in files:
        try:
            runs.append(load_run(file_path))
        except Exception as exc:  # keep other valid runs processable
            warnings.warn(f"Could not load {file_path}: {exc}")

    if not runs:
        print("No usable evaluation files were loaded.", file=sys.stderr)
        return 2

    same_targets = check_target_sets(runs)
    if same_targets:
        print("Target-set check: PASS — runs have the same ordered target positions and seed.")
    else:
        print("Target-set check: WARNING — at least one run does not share the same ordered target set.")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        first_input = Path(args.inputs[0]).expanduser().resolve()
        output_dir = first_input if first_input.is_dir() else first_input.parent
        output_dir = output_dir / "Analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(runs)
    summary_path = output_dir / "fixed_evaluation_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    make_success_plot(runs, output_dir / "stable_hold_success_rate.png")
    make_boxplot(runs, "CartesianDimensionlessSquaredJerk", "JerkValid",
                 "Cartesian dimensionless squared jerk (valid reach episodes)",
                 "Dimensionless squared jerk (lower is smoother)",
                 output_dir / "dimensionless_jerk_boxplot.png")
    make_boxplot(runs, "SPARC", "SPARCValid",
                 "SPARC (valid reach episodes)", "SPARC (higher / closer to zero is smoother)",
                 output_dir / "sparc_boxplot.png")
    make_boxplot(runs, "HoldDistanceStdDevM", "HoldDistanceStdDevValid",
                 "Hold-distance standard deviation", "Distance standard deviation (m; lower is steadier)",
                 output_dir / "hold_distance_std_boxplot.png")
    make_boxplot(runs, "SettlingTimeSteps", "StableHoldSuccess",
                 "Settling time (successful episodes only)", "Agent decisions (lower is faster)",
                 output_dir / "settling_time_boxplot.png")

    print(f"Processed {len(runs)} run(s).")
    print(f"Comparison summary: {summary_path}")
    print(f"Plots: {output_dir}")
    if not same_targets:
        print("Do not interpret these plots as a paired checkpoint comparison until target sets match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
