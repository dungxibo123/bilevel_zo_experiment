"""Algorithm 2: callable lower-level switching sub-routine."""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np

from utils import Array, as_vector, identity_projection, l2_norm

OutputRule = Literal["best_g", "average", "last_feasible", "last"]


class Algorithm2:
    """
    Callable implementation of Algorithm 2.

    The switching rule is:
        if ||GradEstimator_g(x, y_k)|| <= 2 epsilon_g:
            y_{k+1} = P_Y(y_k - mu GradEstimator_f(x, y_k))
            k is added to I
        else:
            y_{k+1} = P_Y(y_k - mu GradEstimator_g(x, y_k))

    If verbose=True, returns a dictionary with y_out and detailed histories.
    If verbose=False, returns only y_out.
    """

    def __init__(
        self,
        f: Callable[[Array, Array], float],
        g: Callable[[Array, Array], float],
        grad_f: Callable[[Array, Array], Array],
        grad_g: Callable[[Array, Array], Array],
        K: int = 100,
        mu: float = 0.05,
        epsilon_g: float = 1e-2,
        projection_y: Optional[Callable[[Array], Array]] = None,
        output_rule: OutputRule = "best_g",
    ) -> None:
        if K <= 0:
            raise ValueError("K must be positive")
        if mu <= 0:
            raise ValueError("mu must be positive")
        if epsilon_g < 0:
            raise ValueError("epsilon_g must be nonnegative")
        if output_rule not in {"best_g", "average", "last_feasible", "last"}:
            raise ValueError("Invalid output_rule")

        self.f = f
        self.g = g
        self.grad_f = grad_f
        self.grad_g = grad_g
        self.K = int(K)
        self.mu = float(mu)
        self.epsilon_g = float(epsilon_g)
        self.projection_y = projection_y or identity_projection
        self.output_rule = output_rule

    def __call__(self, x: Array, y0: Array, verbose: bool = False):
        x = as_vector(x)
        y = as_vector(y0)

        feasible_indices: list[int] = []
        feasible_y: list[Array] = []
        feasible_g_norms: list[float] = []

        history = {
            "iteration": [],
            "branch": [],
            "x": [],
            "y_before": [],
            "y_after": [],
            "f_value": [],
            "g_value": [],
            "grad_f": [],
            "grad_g": [],
            "grad_f_norm": [],
            "grad_g_norm": [],
            "step_norm": [],
        }

        f_branch_count = 0
        g_branch_count = 0

        for k in range(self.K):
            y_before = y.copy()

            # Both estimators are evaluated every iteration. This keeps verbose=True and verbose=False
            # on the same algorithmic path and gives complete logs when requested.
            grad_g_k = as_vector(self.grad_g(x, y_before))
            grad_f_k = as_vector(self.grad_f(x, y_before))

            grad_g_norm = l2_norm(grad_g_k)
            grad_f_norm = l2_norm(grad_f_k)
            f_value = float(self.f(x, y_before))
            g_value = float(self.g(x, y_before))

            if grad_g_norm <= 2.0 * self.epsilon_g:
                branch = "f"
                f_branch_count += 1
                feasible_indices.append(k)
                feasible_y.append(y_before.copy())
                feasible_g_norms.append(grad_g_norm)
                y = self.projection_y(y_before - self.mu * grad_f_k)
            else:
                branch = "g"
                g_branch_count += 1
                y = self.projection_y(y_before - self.mu * grad_g_k)

            if verbose:
                history["iteration"].append(k)
                history["branch"].append(branch)
                history["x"].append(x.copy())
                history["y_before"].append(y_before.copy())
                history["y_after"].append(y.copy())
                history["f_value"].append(f_value)
                history["g_value"].append(g_value)
                history["grad_f"].append(grad_f_k.copy())
                history["grad_g"].append(grad_g_k.copy())
                history["grad_f_norm"].append(grad_f_norm)
                history["grad_g_norm"].append(grad_g_norm)
                history["step_norm"].append(l2_norm(y - y_before))

        y_out, output_source = self._select_output(
            final_y=y,
            feasible_y=feasible_y,
            feasible_g_norms=feasible_g_norms,
        )

        if not verbose:
            return y_out

        return {
            "y_out": y_out,
            "output_source": output_source,
            "output_rule": self.output_rule,
            "I": np.asarray(feasible_indices, dtype=int),
            "I_count": len(feasible_indices),
            "Ic_count": self.K - len(feasible_indices),
            "f_branch_count": f_branch_count,
            "g_branch_count": g_branch_count,
            "history": self._finalize_history(history),
        }

    def _select_output(
        self,
        final_y: Array,
        feasible_y: list[Array],
        feasible_g_norms: list[float],
    ) -> tuple[Array, str]:
        if self.output_rule == "last":
            return final_y.copy(), "last"

        if len(feasible_y) == 0:
            return final_y.copy(), "fallback_last_no_feasible_iterate"

        if self.output_rule == "average":
            return np.mean(np.vstack(feasible_y), axis=0), "average_over_I"

        if self.output_rule == "last_feasible":
            return feasible_y[-1].copy(), "last_feasible"

        if self.output_rule == "best_g":
            best_index = int(np.argmin(np.asarray(feasible_g_norms)))
            return feasible_y[best_index].copy(), "argmin_grad_g_over_I"

        raise ValueError("Invalid output_rule")

    @staticmethod
    def _finalize_history(history: dict) -> dict:
        finalized = dict(history)
        for key in [
            "f_value",
            "g_value",
            "grad_f_norm",
            "grad_g_norm",
            "step_norm",
        ]:
            finalized[key] = np.asarray(finalized[key], dtype=float)
        for key in ["x", "y_before", "y_after", "grad_f", "grad_g"]:
            if len(finalized[key]) > 0:
                finalized[key] = np.vstack(finalized[key])
            else:
                finalized[key] = np.empty((0,))
        finalized["iteration"] = np.asarray(finalized["iteration"], dtype=int)
        return finalized
