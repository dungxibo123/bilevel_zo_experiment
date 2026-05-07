"""Algorithm 3 and the hyper-objective oracle built from Algorithm 2."""

from __future__ import annotations

from typing import Callable, Literal, Optional, Union

import numpy as np

from utils import Array, as_vector, identity_projection, sample_unit_sphere

Algorithm3OutputRule = Literal["random", "last", "best_value"]


class HyperObjectiveOracle:
    """
    Callable sub-algorithm for evaluating the hyper-objective at x.

    Given x, it runs Algorithm 2 to compute y_out(x), then returns f(x, y_out(x)).
    """

    def __init__(
        self,
        f: Callable[[Array, Array], float],
        lower_solver: Callable,
        y0: Optional[Array] = None,
        y0_func: Optional[Callable[[Array], Array]] = None,
    ) -> None:
        if y0 is None and y0_func is None:
            raise ValueError("Either y0 or y0_func must be provided")
        self.f = f
        self.lower_solver = lower_solver
        self.y0 = None if y0 is None else as_vector(y0)
        self.y0_func = y0_func

    def __call__(self, x: Array, verbose: bool = False):
        x = as_vector(x)
        y0 = self._initial_y(x)
        result = self.lower_solver(x, y0, verbose=verbose)

        if verbose:
            y_out = as_vector(result["y_out"])
            value = float(self.f(x, y_out))
            return {
                "value": value,
                "x": x.copy(),
                "y0": y0.copy(),
                "y_out": y_out,
                "algorithm2": result,
            }

        y_out = as_vector(result)
        return float(self.f(x, y_out))

    def _initial_y(self, x: Array) -> Array:
        if self.y0_func is not None:
            return as_vector(self.y0_func(x))
        return self.y0.copy()


class Algorithm3:
    """
    Callable implementation of Algorithm 3.

    For each t, sample u_t uniformly from the unit sphere and form

        grad_t = n / (2 rho) * (phi(x_t + rho u_t) - phi(x_t - rho u_t)) u_t.

    Then update

        x_{t+1} = P_X(x_t - eta grad_t).
    """

    def __init__(
        self,
        hyper_objective: Callable[[Array], float],
        x_dim: int,
        T: int = 100,
        eta: float = 0.05,
        rho: float = 1e-2,
        projection_x: Optional[Callable[[Array], Array]] = None,
        n_samples: int = 1,
        random_state: Optional[int] = None,
        output_rule: Algorithm3OutputRule = "random",
    ) -> None:
        if x_dim <= 0:
            raise ValueError("x_dim must be positive")
        if T <= 0:
            raise ValueError("T must be positive")
        if eta <= 0:
            raise ValueError("eta must be positive")
        if rho <= 0:
            raise ValueError("rho must be positive")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if output_rule not in {"random", "last", "best_value"}:
            raise ValueError("Invalid output_rule")

        self.hyper_objective = hyper_objective
        self.x_dim = int(x_dim)
        self.T = int(T)
        self.eta = float(eta)
        self.rho = float(rho)
        self.projection_x = projection_x or identity_projection
        self.n_samples = int(n_samples)
        self.rng = np.random.default_rng(random_state)
        self.output_rule = output_rule

    def __call__(self, x0: Array, verbose: bool = False) -> dict:
        x = as_vector(x0)
        if x.shape[0] != self.x_dim:
            raise ValueError(f"Expected x0 to have dimension {self.x_dim}, got {x.shape[0]}")

        x_history = []
        value_history = []
        grad_history = []
        step_norm_history = []
        estimator_details = []

        for t in range(self.T):
            value = float(self.hyper_objective(x))
            grad_t, details = self._estimate_gradient(x)
            x_next = self.projection_x(x - self.eta * grad_t)

            x_history.append(x.copy())
            value_history.append(value)
            grad_history.append(grad_t.copy())
            step_norm_history.append(float(np.linalg.norm(x_next - x)))
            if verbose:
                estimator_details.append(details)

            x = x_next

        x_history_arr = np.vstack(x_history)
        value_history_arr = np.asarray(value_history, dtype=float)
        grad_history_arr = np.vstack(grad_history)
        step_norm_history_arr = np.asarray(step_norm_history, dtype=float)

        x_out, output_index = self._select_output(x_history_arr, value_history_arr)

        return {
            "x_out": x_out,
            "output_index": output_index,
            "output_rule": self.output_rule,
            "x_last": x.copy(),
            "value_out": float(self.hyper_objective(x_out)),
            "value_last": float(self.hyper_objective(x)),
            "x_history": x_history_arr,
            "value_history": value_history_arr,
            "grad_estimator_history": grad_history_arr,
            "grad_estimator_norm_history": np.linalg.norm(grad_history_arr, axis=1),
            "step_norm_history": step_norm_history_arr,
            "estimator_details": estimator_details if verbose else None,
        }

    def _estimate_gradient(self, x: Array) -> tuple[Array, list[dict]]:
        grad = np.zeros(self.x_dim, dtype=float)
        details = []

        for _ in range(self.n_samples):
            direction = sample_unit_sphere(self.x_dim, self.rng)
            value_plus = float(self.hyper_objective(x + self.rho * direction))
            value_minus = float(self.hyper_objective(x - self.rho * direction))
            sample_grad = (
                self.x_dim / (2.0 * self.rho)
            ) * (value_plus - value_minus) * direction
            grad += sample_grad
            details.append(
                {
                    "direction": direction.copy(),
                    "value_plus": value_plus,
                    "value_minus": value_minus,
                    "sample_grad": sample_grad.copy(),
                }
            )

        return grad / self.n_samples, details

    def _select_output(self, x_history: Array, value_history: Array) -> tuple[Array, int]:
        if self.output_rule == "last":
            idx = x_history.shape[0] - 1
        elif self.output_rule == "best_value":
            idx = int(np.argmin(value_history))
        else:
            idx = int(self.rng.integers(0, x_history.shape[0]))
        return x_history[idx].copy(), idx
