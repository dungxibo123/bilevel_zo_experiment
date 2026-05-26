"""Utility functions and toy vector-valued test problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

Array = np.ndarray


def as_vector(value: Union[float, list, tuple, Array]) -> Array:
    """Convert an input into a 1-D float numpy array."""
    return np.asarray(value, dtype=float).reshape(-1)


def l2_norm(value: Union[float, list, tuple, Array]) -> float:
    """Euclidean norm as a Python float."""
    return float(np.linalg.norm(as_vector(value)))


def sample_unit_sphere(dim: int, rng: np.random.Generator) -> Array:
    """Sample a random vector uniformly from the Euclidean unit sphere."""
    if dim <= 0:
        raise ValueError("dim must be positive")

    while True:
        direction = rng.normal(size=dim)
        norm = np.linalg.norm(direction)
        if norm > 0:
            return direction / norm


def identity_projection(value: Array) -> Array:
    """Projection map that does nothing."""
    return as_vector(value)


def make_box_projection(
    lower: Union[float, Array],
    upper: Union[float, Array],
) -> Callable[[Array], Array]:
    """Create a projection onto a box [lower, upper]."""
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)

    def projection(value: Array) -> Array:
        return np.clip(as_vector(value), lower_arr, upper_arr)

    return projection

@dataclass
class QuadraticNonConvexBilevelProblem:
    """
    Smooth nonconvex toy problem for vector-valued x and y.

    Lower-level objective:
        g(x, y) = 0.5 ||y - x||^2 - 0.5 beta ||y||^2

    Upper-level objective:
        f(x, y) = 0.5 ||x - x_target||^2 + 0.5 alpha ||y - y_target||^2

    The dimensions of x and y are assumed equal in this toy problem.
    """

    dim: int
    beta: float = 1.0
    alpha: float = 1.0
    x_target: Optional[Array] = None
    y_target: Optional[Array] = None

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.beta < 0:
            raise ValueError("beta must be nonnegative")
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative")

        if self.x_target is None:
            self.x_target = np.ones(self.dim) * 2.0
        else:
            self.x_target = as_vector(self.x_target)

        if self.y_target is None:
            self.y_target = np.ones(self.dim) * 1.0
        else:
            self.y_target = as_vector(self.y_target)

        if self.x_target.shape[0] != self.dim:
            raise ValueError("x_target dimension mismatch")
        if self.y_target.shape[0] != self.dim:
            raise ValueError("y_target dimension mismatch")

    def _check(self, x: Array, y: Array) -> tuple[Array, Array]:
        x = as_vector(x)
        y = as_vector(y)
        if x.shape[0] != self.dim or y.shape[0] != self.dim:
            raise ValueError(
                f"Expected x and y to have dimension {self.dim}, got {x.shape[0]} and {y.shape[0]}"
            )
        return x, y

    def f(self, x: Array, y: Array) -> float:
        x, y = self._check(x, y)
        return float(
            0.5 * np.linalg.norm(x - self.x_target) ** 2
            + 0.5 * self.alpha * np.linalg.norm(y - self.y_target) ** 2
        )

    def g(self, x: Array, y: Array) -> float:
        x, y = self._check(x, y)
        return float(
            0.5 * np.linalg.norm(y - x) ** 2
            - 0.5 * self.beta * np.linalg.norm(y) ** 2
        )

    def grad_f_y(self, x: Array, y: Array) -> Array:
        x, y = self._check(x, y)
        return self.alpha * (y - self.y_target)

    def grad_g_y(self, x: Array, y: Array) -> Array:
        x, y = self._check(x, y)
        return (y - x) - self.beta * y

    def grad_f_x(self, x: Array, y: Array) -> Array:
        x, y = self._check(x, y)
        return x - self.x_target

    def lower_solution(self, x: Array) -> Array:
        """Closed-form lower solution for sanity checks."""
        x = as_vector(x)
        return x / (1.0 - self.beta)

@dataclass
class QuadraticBilevelProblem:
    """
    Smooth convex toy problem for vector-valued x and y.

    Lower-level objective:
        g(x, y) = 0.5 ||y - x||^2 + 0.5 beta ||y||^2

    Upper-level objective:
        f(x, y) = 0.5 ||x - x_target||^2 + 0.5 alpha ||y - y_target||^2

    The dimensions of x and y are assumed equal in this toy problem.
    """

    dim: int
    beta: float = 1.0
    alpha: float = 1.0
    x_target: Optional[Array] = None
    y_target: Optional[Array] = None

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.beta < 0:
            raise ValueError("beta must be nonnegative")
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative")

        if self.x_target is None:
            self.x_target = np.ones(self.dim) * 2.0
        else:
            self.x_target = as_vector(self.x_target)

        if self.y_target is None:
            self.y_target = np.ones(self.dim) * 1.0
        else:
            self.y_target = as_vector(self.y_target)

        if self.x_target.shape[0] != self.dim:
            raise ValueError("x_target dimension mismatch")
        if self.y_target.shape[0] != self.dim:
            raise ValueError("y_target dimension mismatch")

    def _check(self, x: Array, y: Array) -> tuple[Array, Array]:
        x = as_vector(x)
        y = as_vector(y)
        if x.shape[0] != self.dim or y.shape[0] != self.dim:
            raise ValueError(
                f"Expected x and y to have dimension {self.dim}, got {x.shape[0]} and {y.shape[0]}"
            )
        return x, y

    def f(self, x: Array, y: Array) -> float:
        x, y = self._check(x, y)
        return float(
            0.5 * np.linalg.norm(x - self.x_target) ** 2
            + 0.5 * self.alpha * np.linalg.norm(y - self.y_target) ** 2
        )

    def g(self, x: Array, y: Array) -> float:
        x, y = self._check(x, y)
        return float(
            0.5 * np.linalg.norm(y - x) ** 2
            + 0.5 * self.beta * np.linalg.norm(y) ** 2
        )

    def grad_f_y(self, x: Array, y: Array) -> Array:
        x, y = self._check(x, y)
        return self.alpha * (y - self.y_target)

    def grad_g_y(self, x: Array, y: Array) -> Array:
        x, y = self._check(x, y)
        return (y - x) + self.beta * y

    def grad_f_x(self, x: Array, y: Array) -> Array:
        x, y = self._check(x, y)
        return x - self.x_target

    def lower_solution(self, x: Array) -> Array:
        """Closed-form lower solution for sanity checks."""
        x = as_vector(x)
        return x / (1.0 + self.beta)

def make_quadratic_nonconvex_problem(
    dim: int = 3,
    beta: float = 1.0,
    alpha: float = 1.0,
    x_target: Optional[Array] = None,
    y_target: Optional[Array] = None,
) -> QuadraticBilevelProblem:
    """Factory for the default smooth convex experiment."""
    return QuadraticNonConvexBilevelProblem(
        dim=dim,
        beta=beta,
        alpha=alpha,
        x_target=x_target,
        y_target=y_target,
    )


def make_quadratic_problem(
    dim: int = 3,
    beta: float = 1.0,
    alpha: float = 1.0,
    x_target: Optional[Array] = None,
    y_target: Optional[Array] = None,
) -> QuadraticBilevelProblem:
    """Factory for the default smooth convex experiment."""
    return QuadraticBilevelProblem(
        dim=dim,
        beta=beta,
        alpha=alpha,
        x_target=x_target,
        y_target=y_target,
    )
