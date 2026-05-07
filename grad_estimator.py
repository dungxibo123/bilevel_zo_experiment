"""Algorithm 1: callable zeroth-order gradient estimator."""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np

from utils import Array, as_vector, sample_unit_sphere

VariableName = Literal["x", "y"]
ModeName = Literal["zeroth_order", "true"]


class GradientEstimator:
    """
    Callable implementation of Algorithm 1.

    The default estimator is the two-point random-direction estimator on the unit sphere:

        dim / (2 gamma) * (phi(z + gamma u) - phi(z - gamma u)) * u.

    The estimator can be used with respect to either x or y by setting variable="x" or variable="y".
    For debugging/comparison, mode="true" can be used with an exact gradient callable.
    """

    def __init__(
        self,
        func: Callable[[Array, Array], float],
        variable: VariableName = "y",
        gamma: float = 1e-3,
        dim: Optional[int] = None,
        n_samples: int = 1,
        mode: ModeName = "zeroth_order",
        true_grad_func: Optional[Callable[[Array, Array], Array]] = None,
        random_state: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if variable not in {"x", "y"}:
            raise ValueError("variable must be either 'x' or 'y'")
        if mode not in {"zeroth_order", "true"}:
            raise ValueError("mode must be either 'zeroth_order' or 'true'")
        if mode == "true" and true_grad_func is None:
            raise ValueError("true_grad_func must be provided when mode='true'")

        self.func = func
        self.variable = variable
        self.gamma = float(gamma)
        self.dim = dim
        self.n_samples = int(n_samples)
        self.mode = mode
        self.true_grad_func = true_grad_func
        self.rng = np.random.default_rng(random_state)
        self.name = name or f"GradientEstimator({variable})"

    def __call__(self, x: Array, y: Array) -> Array:
        x = as_vector(x)
        y = as_vector(y)

        if self.mode == "true":
            return as_vector(self.true_grad_func(x, y))

        dim = self._infer_dim(x, y)
        grad = np.zeros(dim, dtype=float)

        for _ in range(self.n_samples):
            direction = sample_unit_sphere(dim, self.rng)
            grad += self._two_point_estimate(x, y, direction)

        return grad / self.n_samples

    def _infer_dim(self, x: Array, y: Array) -> int:
        if self.dim is not None:
            return int(self.dim)
        return x.shape[0] if self.variable == "x" else y.shape[0]

    def _two_point_estimate(self, x: Array, y: Array, direction: Array) -> Array:
        gamma = self.gamma
        dim = direction.shape[0]

        if self.variable == "y":
            value_plus = self.func(x, y + gamma * direction)
            value_minus = self.func(x, y - gamma * direction)
        else:
            value_plus = self.func(x + gamma * direction, y)
            value_minus = self.func(x - gamma * direction, y)

        return (dim / (2.0 * gamma)) * (value_plus - value_minus) * direction
