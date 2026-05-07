"""Run experiments for Algorithms 1, 2, and 3.

This entry point is intentionally configurable from the command line. Example:

    python main.py --dim 5 --alg3_T 100 --alg2_K 80 --log_path logs/run.csv

The optional CSV log stores the hyper-objective value at every outer iteration
alongside the full experiment configuration, so later analysis can be done
without guessing which hyperparameters produced the run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from grad_estimator import GradientEstimator
from main_routine import Algorithm3, HyperObjectiveOracle
from sub_routine import Algorithm2
from utils import Array, as_vector, make_box_projection, make_quadratic_problem


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the zeroth-order bilevel toy experiment with Algorithms 1, 2, and 3."
    )

    # Problem configuration
    parser.add_argument("--dim", type=int, default=3, help="Dimension of x and y.")
    parser.add_argument("--beta", type=float, default=1.0, help="Lower-level quadratic beta.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Upper-level quadratic alpha.")
    parser.add_argument(
        "--x_target_value",
        type=float,
        default=2.0,
        help="Scalar target used to initialize x_target = value * ones(dim).",
    )
    parser.add_argument(
        "--y_target_value",
        type=float,
        default=1.0,
        help="Scalar target used to initialize y_target = value * ones(dim).",
    )

    # Algorithm 1 / gradient estimator configuration for Algorithm 2
    parser.add_argument(
        "--grad_mode",
        choices=["zeroth_order", "true"],
        default="zeroth_order",
        help="Gradient source for Algorithm 2. Use 'true' for debugging/comparison.",
    )
    parser.add_argument(
        "--grad_f_gamma",
        type=float,
        default=1e-2,
        help="Smoothing radius for the f-gradient estimator with respect to y.",
    )
    parser.add_argument(
        "--grad_g_gamma",
        type=float,
        default=1e-2,
        help="Smoothing radius for the g-gradient estimator with respect to y.",
    )
    parser.add_argument(
        "--grad_f_samples",
        type=int,
        default=2,
        help="Number of samples for the f-gradient estimator in Algorithm 2.",
    )
    parser.add_argument(
        "--grad_g_samples",
        type=int,
        default=2,
        help="Number of samples for the g-gradient estimator in Algorithm 2.",
    )
    parser.add_argument(
        "--seed_grad_f",
        type=int,
        default=10,
        help="Random seed for the f-gradient estimator in Algorithm 2.",
    )
    parser.add_argument(
        "--seed_grad_g",
        type=int,
        default=20,
        help="Random seed for the g-gradient estimator in Algorithm 2.",
    )

    # Algorithm 2 configuration
    parser.add_argument("--alg2_K", type=int, default=60, help="Number of inner iterations.")
    parser.add_argument("--alg2_mu", type=float, default=0.08, help="Algorithm 2 step size.")
    parser.add_argument(
        "--epsilon_g",
        type=float,
        default=0.25,
        help="Switching threshold epsilon_g; Algorithm 2 checks ||G_g|| <= 2 epsilon_g.",
    )
    parser.add_argument(
        "--alg2_output_rule",
        choices=["best_g", "average", "last_feasible", "last"],
        default="best_g",
        help="How Algorithm 2 selects y_out.",
    )
    parser.add_argument(
        "--y_box_lower",
        type=float,
        default=-8.0,
        help="Lower bound for projection onto Y.",
    )
    parser.add_argument(
        "--y_box_upper",
        type=float,
        default=8.0,
        help="Upper bound for projection onto Y.",
    )
    parser.add_argument(
        "--y0_value",
        type=float,
        default=0.0,
        help="Scalar initialization used by the hyper-objective oracle: y0 = value * ones(dim).",
    )

    # Algorithm 3 configuration
    parser.add_argument("--alg3_T", type=int, default=50, help="Number of outer iterations.")
    parser.add_argument("--alg3_eta", type=float, default=0.04, help="Algorithm 3 step size.")
    parser.add_argument(
        "--rho",
        type=float,
        default=5e-2,
        help="Outer smoothing radius used in Algorithm 3.",
    )
    parser.add_argument(
        "--alg3_samples",
        type=int,
        default=2,
        help="Number of two-point samples used for each Algorithm 3 gradient estimate.",
    )
    parser.add_argument(
        "--seed_outer",
        type=int,
        default=123,
        help="Random seed for Algorithm 3 outer directions and random output rule.",
    )
    parser.add_argument(
        "--alg3_output_rule",
        choices=["random", "last", "best_value"],
        default="best_value",
        help="How Algorithm 3 selects x_out.",
    )
    parser.add_argument(
        "--x_box_lower",
        type=float,
        default=-8.0,
        help="Lower bound for projection onto X.",
    )
    parser.add_argument(
        "--x_box_upper",
        type=float,
        default=8.0,
        help="Upper bound for projection onto X.",
    )
    parser.add_argument(
        "--x0_value",
        type=float,
        default=4.0,
        help="Scalar initialization for Algorithm 3: x0 = value * ones(dim).",
    )

    # Logging / diagnostics
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help=(
            "Optional CSV path. If provided, saves the hyper-objective value, "
            "gradient-estimator norm, step norm, x coordinates, and full config at every outer iteration."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional run information and ask Algorithm 3 to retain estimator details.",
    )
    parser.add_argument(
        "--run_alg2_demo",
        action="store_true",
        help="Run a single verbose Algorithm 2 demo before Algorithm 3.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save diagnostic plots after running Algorithm 3.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=".",
        help="Directory where plots are saved when --plot is used.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Experiment construction
# -----------------------------------------------------------------------------


def make_vector(value: float, dim: int) -> Array:
    return np.ones(dim, dtype=float) * float(value)


def build_problem(args: argparse.Namespace):
    return make_quadratic_problem(
        dim=args.dim,
        beta=args.beta,
        alpha=args.alpha,
        x_target=make_vector(args.x_target_value, args.dim),
        y_target=make_vector(args.y_target_value, args.dim),
    )


def build_gradient_estimators(problem, args: argparse.Namespace) -> tuple[GradientEstimator, GradientEstimator]:
    true_grad_f = problem.grad_f_y if args.grad_mode == "true" else None
    true_grad_g = problem.grad_g_y if args.grad_mode == "true" else None

    grad_f_y = GradientEstimator(
        func=problem.f,
        variable="y",
        dim=args.dim,
        gamma=args.grad_f_gamma,
        n_samples=args.grad_f_samples,
        mode=args.grad_mode,
        true_grad_func=true_grad_f,
        random_state=args.seed_grad_f,
        name="grad f wrt y",
    )
    grad_g_y = GradientEstimator(
        func=problem.g,
        variable="y",
        dim=args.dim,
        gamma=args.grad_g_gamma,
        n_samples=args.grad_g_samples,
        mode=args.grad_mode,
        true_grad_func=true_grad_g,
        random_state=args.seed_grad_g,
        name="grad g wrt y",
    )
    return grad_f_y, grad_g_y


def build_algorithm2(problem, args: argparse.Namespace) -> Algorithm2:
    grad_f_y, grad_g_y = build_gradient_estimators(problem, args)
    return Algorithm2(
        f=problem.f,
        g=problem.g,
        grad_f=grad_f_y,
        grad_g=grad_g_y,
        K=args.alg2_K,
        mu=args.alg2_mu,
        epsilon_g=args.epsilon_g,
        projection_y=make_box_projection(lower=args.y_box_lower, upper=args.y_box_upper),
        output_rule=args.alg2_output_rule,
    )


def build_algorithm3(problem, args: argparse.Namespace) -> Algorithm3:
    lower_solver = build_algorithm2(problem, args)
    hyper_oracle = HyperObjectiveOracle(
        f=problem.f,
        lower_solver=lower_solver,
        y0=make_vector(args.y0_value, args.dim),
    )
    return Algorithm3(
        hyper_objective=hyper_oracle,
        x_dim=args.dim,
        T=args.alg3_T,
        eta=args.alg3_eta,
        rho=args.rho,
        n_samples=args.alg3_samples,
        projection_x=make_box_projection(lower=args.x_box_lower, upper=args.x_box_upper),
        random_state=args.seed_outer,
        output_rule=args.alg3_output_rule,
    )


# -----------------------------------------------------------------------------
# Running / logging
# -----------------------------------------------------------------------------


def run_algorithm2_verbose_demo(problem, args: argparse.Namespace) -> None:
    algorithm2 = build_algorithm2(problem, args)
    x = make_vector(args.x0_value, args.dim)
    y0 = make_vector(args.y0_value, args.dim)
    result = algorithm2(x, y0, verbose=True)

    print("\n=== Algorithm 2 verbose demo ===")
    print(f"y_out: {result['y_out']}")
    print(f"output source: {result['output_source']}")
    print(f"f-branch count: {result['f_branch_count']}")
    print(f"g-branch count: {result['g_branch_count']}")
    print(f"|I|: {result['I_count']}, |Ic|: {result['Ic_count']}")
    print("first 10 branches:", result["history"]["branch"][:10])
    print("last f(x,y):", result["history"]["f_value"][-1])
    print("last g(x,y):", result["history"]["g_value"][-1])
    print("last ||grad_f||:", result["history"]["grad_f_norm"][-1])
    print("last ||grad_g||:", result["history"]["grad_g_norm"][-1])


def run_algorithm3_experiment(problem, args: argparse.Namespace) -> dict:
    algorithm3 = build_algorithm3(problem, args)
    x0 = make_vector(args.x0_value, args.dim)
    result = algorithm3(x0, verbose=args.verbose)

    print("\n=== Algorithm 3 experiment ===")
    print(f"x_out: {result['x_out']}")
    print(f"value_out: {result['value_out']:.6f}")
    print(f"x_last: {result['x_last']}")
    print(f"value_last: {result['value_last']:.6f}")
    print(f"initial hyper value: {result['value_history'][0]:.6f}")
    print(f"best logged hyper value: {np.min(result['value_history']):.6f}")

    return result


def config_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Return a JSON-serializable copy of the command-line configuration."""
    return {key: value for key, value in vars(args).items()}


def write_log(result: dict, args: argparse.Namespace, log_path: str) -> None:
    """Write per-outer-iteration logs and a sidecar JSON config file."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = config_dict(args)
    config_json = json.dumps(config, sort_keys=True)

    x_history = np.asarray(result["x_history"])
    grad_history = np.asarray(result["grad_estimator_history"])
    value_history = np.asarray(result["value_history"])
    grad_norm_history = np.asarray(result["grad_estimator_norm_history"])
    step_norm_history = np.asarray(result["step_norm_history"])

    fieldnames = [
        "iteration",
        "hyper_objective_value",
        "hyper_objective_loss",
        "grad_estimator_norm",
        "step_norm",
    ]
    fieldnames += [f"x_{i}" for i in range(x_history.shape[1])]
    fieldnames += [f"grad_estimator_{i}" for i in range(grad_history.shape[1])]
    fieldnames += [f"config_{key}" for key in sorted(config.keys())]
    fieldnames += ["config_json"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in range(value_history.shape[0]):
            row: dict[str, Any] = {
                "iteration": t,
                "hyper_objective_value": float(value_history[t]),
                # Alias kept because many experiment scripts call this quantity a loss.
                "hyper_objective_loss": float(value_history[t]),
                "grad_estimator_norm": float(grad_norm_history[t]),
                "step_norm": float(step_norm_history[t]),
                "config_json": config_json,
            }
            row.update({f"x_{i}": float(x_history[t, i]) for i in range(x_history.shape[1])})
            row.update(
                {f"grad_estimator_{i}": float(grad_history[t, i]) for i in range(grad_history.shape[1])}
            )
            row.update({f"config_{key}": config[key] for key in sorted(config.keys())})
            writer.writerow(row)

    config_path = path.with_suffix(path.suffix + ".config.json")
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print(f"Saved per-iteration log to: {path}")
    print(f"Saved configuration to: {config_path}")


def maybe_plot(result: dict, plot_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - only relevant when matplotlib is unavailable
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    output_dir = Path(plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    value_path = output_dir / "algorithm3_hyper_objective_trace.png"
    grad_path = output_dir / "algorithm3_grad_estimator_norm_trace.png"

    plt.figure()
    plt.plot(result["value_history"])
    plt.xlabel("Outer iteration")
    plt.ylabel("estimated hyper-objective value")
    plt.title("Algorithm 3 hyper-objective trace")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(value_path, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(result["grad_estimator_norm_history"])
    plt.xlabel("Outer iteration")
    plt.ylabel("outer gradient-estimator norm")
    plt.title("Algorithm 3 estimator norm trace")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(grad_path, dpi=150)
    plt.close()

    print("Saved plots:")
    print(f"  {value_path}")
    print(f"  {grad_path}")


def main() -> None:
    args = parse_args()
    problem = build_problem(args)

    if args.verbose:
        print("\n=== Configuration ===")
        print(json.dumps(config_dict(args), indent=2, sort_keys=True))

    if args.run_alg2_demo:
        run_algorithm2_verbose_demo(problem, args)

    result = run_algorithm3_experiment(problem, args)

    if args.log_path is not None:
        write_log(result, args, args.log_path)

    if args.plot:
        maybe_plot(result, args.plot_dir)


if __name__ == "__main__":
    main()
