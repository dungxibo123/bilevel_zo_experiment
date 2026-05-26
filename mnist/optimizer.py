"""Zeroth-order PyTorch optimizers.

This module contains optimizer-style implementations that can be plugged into
any ``torch.nn.Module``. Closures must return scalar losses and must not call
``backward()``.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Literal, Optional

import torch
from torch.optim import Optimizer

TensorClosure = Callable[[], torch.Tensor]
BranchName = Literal["f", "g"]


def trainable_params(params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    """Return only trainable parameters."""
    return [p for p in params if p.requires_grad]


def numel(params: Iterable[torch.nn.Parameter]) -> int:
    """Total number of scalar parameters."""
    return sum(p.numel() for p in params)


@torch.no_grad()
def sample_unit_direction(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    """Sample one unit vector in the flattened parameter space."""
    if len(params) == 0:
        raise ValueError("Cannot sample a direction from an empty parameter list.")

    directions = [torch.randn_like(p) for p in params]
    norm_sq = torch.zeros((), device=params[0].device)
    for d in directions:
        norm_sq += torch.sum(d * d)
    norm = torch.sqrt(norm_sq).clamp_min(1e-12)
    return [d / norm for d in directions]


@torch.no_grad()
def add_scaled_direction(
    params: list[torch.nn.Parameter],
    directions: list[torch.Tensor],
    scale: float,
) -> None:
    """Add ``scale * direction`` to every parameter tensor."""
    for p, d in zip(params, directions):
        p.add_(d, alpha=scale)


@torch.no_grad()
def apply_dense_update(
    params: list[torch.nn.Parameter],
    gradient_estimate: list[torch.Tensor],
    lr: float,
    weight_decay: float = 0.0,
    max_update_norm: Optional[float] = None,
    clamp: Optional[tuple[float, float]] = None,
) -> float:
    """Apply ``theta <- theta - lr * gradient_estimate`` and return update norm."""
    update_norm_sq = torch.zeros((), device=params[0].device)
    for g in gradient_estimate:
        update_norm_sq += torch.sum(g * g)
    grad_norm = torch.sqrt(update_norm_sq).clamp_min(1e-12)

    scale = 1.0
    if max_update_norm is not None and grad_norm.item() * lr > max_update_norm:
        scale = float(max_update_norm / (lr * grad_norm.item()))

    for p, g in zip(params, gradient_estimate):
        if weight_decay > 0.0:
            p.mul_(1.0 - lr * weight_decay)
        p.add_(g, alpha=-lr * scale)
        if clamp is not None:
            p.clamp_(clamp[0], clamp[1])

    return float(lr * grad_norm.item() * scale)


class ZerothOrderSGD(Optimizer):
    """Two-point random-direction zeroth-order SGD.

    For a scalar closure ``L(theta)``, each sample computes

        d / (2 gamma) * [L(theta + gamma u) - L(theta - gamma u)] * u,

    where ``u`` is sampled uniformly from the unit sphere in the flattened
    parameter space and ``d`` is the number of trainable parameters.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        gamma: float = 1e-3,
        n_samples: int = 1,
        weight_decay: float = 0.0,
        max_update_norm: Optional[float] = None,
        clamp: Optional[tuple[float, float]] = None,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            n_samples=n_samples,
            weight_decay=weight_decay,
            max_update_norm=max_update_norm,
            clamp=clamp,
        )
        super().__init__(params, defaults)
        self.params_list = trainable_params(self.param_groups[0]["params"])
        if not self.params_list:
            raise ValueError("optimizer got an empty parameter list")
        self.dim = numel(self.params_list)
        self.last_stats: dict = {}

    @torch.no_grad()
    def _estimate(self, closure: TensorClosure) -> tuple[list[torch.Tensor], float, list[dict]]:
        group = self.param_groups[0]
        gamma = float(group["gamma"])
        n_samples = int(group["n_samples"])

        grad_sum = [torch.zeros_like(p) for p in self.params_list]
        sample_stats: list[dict] = []

        for _ in range(n_samples):
            direction = sample_unit_direction(self.params_list)

            add_scaled_direction(self.params_list, direction, +gamma)
            loss_plus = float(closure().detach().item())

            add_scaled_direction(self.params_list, direction, -2.0 * gamma)
            loss_minus = float(closure().detach().item())

            add_scaled_direction(self.params_list, direction, +gamma)

            coeff = self.dim * (loss_plus - loss_minus) / (2.0 * gamma)
            for acc, d in zip(grad_sum, direction):
                acc.add_(d, alpha=coeff)
            sample_stats.append(
                {
                    "loss_plus": loss_plus,
                    "loss_minus": loss_minus,
                    "coeff": float(coeff),
                    "sample_grad_norm": float(abs(coeff)),
                }
            )

        for acc in grad_sum:
            acc.div_(n_samples)
        grad_norm = math.sqrt(sum(float(torch.sum(g * g).item()) for g in grad_sum))
        return grad_sum, grad_norm, sample_stats

    @torch.no_grad()
    def step(self, closure: TensorClosure) -> dict:  # type: ignore[override]
        group = self.param_groups[0]
        grad_estimate, grad_norm, samples = self._estimate(closure)
        update_norm = apply_dense_update(
            self.params_list,
            grad_estimate,
            lr=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            max_update_norm=group["max_update_norm"],
            clamp=group["clamp"],
        )
        stats = {
            "optimizer": "zo-sgd",
            "branch": "zo",
            "grad_norm": grad_norm,
            "grad_step_norm": grad_norm,
            "update_norm": update_norm,
            "samples": samples,
            "dim": self.dim,
        }
        self.last_stats = stats
        return stats


class SwitchingZerothOrderOptimizer(Optimizer):
    """Algorithm-2-style switching zeroth-order optimizer.

    Required closures:
        ``g_closure``: lower objective, e.g. training mini-batch loss.
        ``f_closure``: upper objective, e.g. validation mini-batch loss.

    Rule:
        if ||G_g(theta)|| <= 2 epsilon_g: update by G_f(theta)
        else:                            update by G_g(theta)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        gamma_f: float = 1e-3,
        gamma_g: float = 1e-3,
        epsilon_g: float = 1e-2,
        n_samples_f: int = 1,
        n_samples_g: int = 1,
        weight_decay: float = 0.0,
        max_update_norm: Optional[float] = None,
        clamp: Optional[tuple[float, float]] = None,
        evaluate_f_every_step: bool = False,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if gamma_f <= 0 or gamma_g <= 0:
            raise ValueError("gamma_f and gamma_g must be positive")
        if epsilon_g < 0:
            raise ValueError("epsilon_g must be nonnegative")
        if n_samples_f <= 0 or n_samples_g <= 0:
            raise ValueError("n_samples_f and n_samples_g must be positive")

        defaults = dict(
            lr=lr,
            gamma_f=gamma_f,
            gamma_g=gamma_g,
            epsilon_g=epsilon_g,
            n_samples_f=n_samples_f,
            n_samples_g=n_samples_g,
            weight_decay=weight_decay,
            max_update_norm=max_update_norm,
            clamp=clamp,
            evaluate_f_every_step=evaluate_f_every_step,
        )
        super().__init__(params, defaults)
        self.params_list = trainable_params(self.param_groups[0]["params"])
        if not self.params_list:
            raise ValueError("optimizer got an empty parameter list")
        self.dim = numel(self.params_list)
        self.f_branch_count = 0
        self.g_branch_count = 0
        self.I: list[int] = []
        self.iteration = 0
        self.last_stats: dict = {}

    @torch.no_grad()
    def _estimate_with_gamma(
        self,
        closure: TensorClosure,
        gamma: float,
        n_samples: int,
    ) -> tuple[list[torch.Tensor], float, list[dict]]:
        grad_sum = [torch.zeros_like(p) for p in self.params_list]
        sample_stats: list[dict] = []

        for _ in range(n_samples):
            direction = sample_unit_direction(self.params_list)

            add_scaled_direction(self.params_list, direction, +gamma)
            loss_plus = float(closure().detach().item())

            add_scaled_direction(self.params_list, direction, -2.0 * gamma)
            loss_minus = float(closure().detach().item())

            add_scaled_direction(self.params_list, direction, +gamma)

            coeff = self.dim * (loss_plus - loss_minus) / (2.0 * gamma)
            for acc, d in zip(grad_sum, direction):
                acc.add_(d, alpha=coeff)
            sample_stats.append(
                {
                    "loss_plus": loss_plus,
                    "loss_minus": loss_minus,
                    "coeff": float(coeff),
                    "sample_grad_norm": float(abs(coeff)),
                }
            )

        for acc in grad_sum:
            acc.div_(n_samples)
        grad_norm = math.sqrt(sum(float(torch.sum(g * g).item()) for g in grad_sum))
        return grad_sum, grad_norm, sample_stats

    @torch.no_grad()
    def step(self, f_closure: TensorClosure, g_closure: TensorClosure) -> dict:  # type: ignore[override]
        group = self.param_groups[0]
        gamma_f = float(group["gamma_f"])
        gamma_g = float(group["gamma_g"])
        epsilon_g = float(group["epsilon_g"])

        grad_g, grad_g_norm, g_samples = self._estimate_with_gamma(
            g_closure,
            gamma=gamma_g,
            n_samples=int(group["n_samples_g"]),
        )

        if grad_g_norm <= 2.0 * epsilon_g:
            branch: BranchName = "f"
            self.f_branch_count += 1
            self.I.append(self.iteration)
            grad_step, grad_step_norm, f_samples = self._estimate_with_gamma(
                f_closure,
                gamma=gamma_f,
                n_samples=int(group["n_samples_f"]),
            )
        else:
            branch = "g"
            self.g_branch_count += 1
            grad_step = grad_g
            grad_step_norm = grad_g_norm
            f_samples = []

        update_norm = apply_dense_update(
            self.params_list,
            grad_step,
            lr=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            max_update_norm=group["max_update_norm"],
            clamp=group["clamp"],
        )

        f_value = float(f_closure().detach().item()) if group["evaluate_f_every_step"] else math.nan
        g_value = float(g_closure().detach().item())

        stats = {
            "optimizer": "zo-switch",
            "iteration": self.iteration,
            "branch": branch,
            "I_count": len(self.I),
            "f_branch_count": self.f_branch_count,
            "g_branch_count": self.g_branch_count,
            "grad_g_norm": grad_g_norm,
            "grad_step_norm": grad_step_norm,
            "update_norm": update_norm,
            "f_value": f_value,
            "g_value": g_value,
            "g_samples": g_samples,
            "f_samples": f_samples,
            "dim": self.dim,
        }
        self.iteration += 1
        self.last_stats = stats
        return stats
