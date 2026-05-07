"""Plot dimension-dependence convergence curves from Algorithm 3 log files.

This script reads CSV files created by ``main.py --log_path ...`` and plots the
normalized hyper-objective value ``varphi(x) / d`` against the outer iteration.

Examples
--------
Read log paths directly::

    python plot_dimension_convergence.py \
        --log_paths logs/dims/run_001.csv logs/dims/run_002.csv \
        --output_path plots/dims/joint_normalized_hyper_objective.png

Or parse log paths from a shell script containing commands with ``--log_path``::

    python plot_dimension_convergence.py \
        --script_path dimension_dependence.sh \
        --output_path plots/dims/joint_normalized_hyper_objective.png
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


VALUE_COLUMN_CANDIDATES = (
    "hyper_objective_value",
    "hyper_objective_loss",
    "estimated_hyper_objective_value",
    "value",
    "value_history",
    "objective",
    "loss",
)

ITERATION_COLUMN_CANDIDATES = (
    "iteration",
    "outer_iteration",
    "iter",
    "t",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Algorithm 3 CSV logs and plot dimension-normalized "
            "hyper-objective traces."
        )
    )
    parser.add_argument(
        "--log_paths",
        nargs="*",
        default=None,
        help="CSV log files produced by main.py --log_path.",
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default=None,
        help=(
            "Optional shell script containing commands with --log_path. "
            "Used to extract log paths automatically."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="plots/dims/joint_normalized_hyper_objective.png",
        help="Where to save the joint convergence plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Dimension dependence of Algorithm 3",
        help="Plot title.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        help=(
            "Optional moving-average window. Use 1 to plot raw curves. "
            "Values larger than 1 smooth noisy traces."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving.",
    )
    return parser.parse_args()


def extract_log_paths_from_script(script_path: str | Path) -> list[str]:
    """Extract values passed to --log_path from a shell script."""
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"script_path does not exist: {script_path}")

    log_paths: list[str] = []
    for raw_line in script_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "--log_path" not in line:
            continue

        try:
            tokens = shlex.split(line)
        except ValueError:
            # Fallback for simple shell commands.
            tokens = line.split()

        for i, token in enumerate(tokens):
            if token == "--log_path" and i + 1 < len(tokens):
                log_paths.append(tokens[i + 1])
            elif token.startswith("--log_path="):
                log_paths.append(token.split("=", 1)[1])

    return log_paths


def choose_column(columns: Iterable[str], candidates: Iterable[str], kind: str) -> str:
    columns = list(columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Could not find a {kind} column. Available columns are: {columns}"
    )


def infer_dim_from_log(df: pd.DataFrame, log_path: str | Path) -> int:
    """Infer dimension from config columns, config_json, or filename."""
    if "config_dim" in df.columns:
        return int(df["config_dim"].iloc[0])

    if "dim" in df.columns:
        return int(df["dim"].iloc[0])

    if "config_json" in df.columns:
        first = df["config_json"].dropna()
        if len(first) > 0:
            try:
                config = json.loads(first.iloc[0])
                if "dim" in config:
                    return int(config["dim"])
                if "config_dim" in config:
                    return int(config["config_dim"])
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    # Fallback: infer from filenames like run_00100.csv or directory dims/100/.
    path = Path(log_path)
    parent_match = re.search(r"(?:^|/)(\d+)(?:/|$)", str(path.parent))
    if parent_match is not None:
        return int(parent_match.group(1))

    name_match = re.search(r"run_0*(\d+)", path.stem)
    if name_match is not None:
        return int(name_match.group(1))

    raise ValueError(f"Could not infer dimension for log file: {log_path}")


def load_trace(log_path: str | Path) -> tuple[int, pd.Series, pd.Series]:
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(
            f"Log file not found: {log_path}. Run main.py with --log_path first."
        )

    df = pd.read_csv(log_path)
    iteration_col = choose_column(df.columns, ITERATION_COLUMN_CANDIDATES, "iteration")
    value_col = choose_column(df.columns, VALUE_COLUMN_CANDIDATES, "hyper-objective value")
    dim = infer_dim_from_log(df, log_path)

    iterations = df[iteration_col]
    normalized_value = df[value_col] / dim
    return dim, iterations, normalized_value


def resolve_log_paths(args: argparse.Namespace) -> list[str]:
    log_paths: list[str] = []
    if args.script_path is not None:
        log_paths.extend(extract_log_paths_from_script(args.script_path))
    if args.log_paths:
        log_paths.extend(args.log_paths)

    # Preserve order but remove duplicates.
    seen = set()
    unique_paths = []
    for path in log_paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)

    if not unique_paths:
        raise ValueError("Provide --log_paths or --script_path containing --log_path entries.")
    return unique_paths


def main() -> None:
    args = parse_args()
    log_paths = resolve_log_paths(args)

    traces = []
    for log_path in log_paths:
        dim, iterations, normalized_value = load_trace(log_path)
        if args.smooth_window > 1:
            normalized_value = normalized_value.rolling(
                window=args.smooth_window,
                min_periods=1,
                center=False,
            ).mean()
        traces.append((dim, iterations, normalized_value, log_path))

    traces.sort(key=lambda item: item[0])

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9.0, 5.5))
    for dim, iterations, normalized_value, _ in traces:
        plt.plot(
            iterations,
            normalized_value,
            linewidth=1.8,
            label=rf"$d={dim}$ ($\varphi(x)/d$)",
        )

    plt.xlabel("Outer iteration")
    plt.ylabel(r"Normalized hyper-objective value $\varphi(x)/d$")
    plt.title(args.title)
    plt.grid(True, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi)
    print(f"Saved joint normalized convergence plot to: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
