"""Utilities for training, datasets, logging, plotting, and reproducibility."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

try:
    import torchvision
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    torchvision = None
    T = None

TensorClosure = Callable[[], torch.Tensor]


def set_seed(seed: int) -> None:
    """Set Python and PyTorch random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(use_cuda: bool) -> torch.device:
    """Use CUDA only when requested and available; otherwise use CPU."""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_datasets(name: str, data_dir: str, val_fraction: float, seed: int):
    """Create train/validation/test datasets and return metadata."""
    if torchvision is None or T is None:
        raise RuntimeError("torchvision is required. Install torchvision before running this experiment.")

    name = name.lower()
    if name == "fake":
        transform = T.ToTensor()
        train_full = torchvision.datasets.FakeData(
            size=4096,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        test_set = torchvision.datasets.FakeData(
            size=1024,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        input_shape = (1, 28, 28)
        num_classes = 10
    elif name == "mnist":
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train_full = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        input_shape = (1, 28, 28)
        num_classes = 10
    elif name == "fashionmnist":
        transform = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
        train_full = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        input_shape = (1, 28, 28)
        num_classes = 10
    elif name == "cifar10":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train_full = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        input_shape = (3, 32, 32)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")

    val_size = max(1, int(len(train_full) * val_fraction))
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=generator)
    return train_set, val_set, test_set, input_shape, num_classes


def make_loaders(
    train_set,
    val_set,
    test_set,
    batch_size: int,
    device: torch.device,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build dataloaders."""
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """Classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def make_batch_closure(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    l2_reg: float = 0.0,
) -> TensorClosure:
    """Create a scalar loss closure for zeroth-order optimizers."""

    def closure() -> torch.Tensor:
        logits = model(x)
        loss = criterion(logits, y)
        if l2_reg > 0.0:
            reg = torch.zeros((), device=x.device)
            for p in model.parameters():
                reg = reg + torch.sum(p * p)
            loss = loss + 0.5 * l2_reg * reg
        return loss

    return closure


def train_one_epoch(
    model: nn.Module,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer_name: str,
    epoch: int,
    max_train_batches: Optional[int],
    l2_reg_f: float,
    l2_reg_g: float,
    log_writer: Optional[csv.DictWriter],
) -> dict:
    """Train for one epoch using either backprop or zeroth-order optimizer."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0
    val_iter = iter(val_loader)

    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        if max_train_batches is not None and batch_idx >= max_train_batches:
            break

        x_train = x_train.to(device)
        y_train = y_train.to(device)

        try:
            x_val, y_val = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            x_val, y_val = next(val_iter)
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        if optimizer_name in {"sgd", "adam"}:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train)
            loss = criterion(logits, y_train)
            if l2_reg_g > 0.0:
                reg = torch.zeros((), device=device)
                for p in model.parameters():
                    reg = reg + torch.sum(p * p)
                loss = loss + 0.5 * l2_reg_g * reg
            loss.backward()
            optimizer.step()
            stats = {"branch": "backprop", "grad_g_norm": math.nan, "update_norm": math.nan}
            loss_value = float(loss.detach().item())
        elif optimizer_name == "zo-sgd":
            closure = make_batch_closure(model, x_train, y_train, criterion, l2_reg=l2_reg_g)
            stats = optimizer.step(closure)
            loss_value = float(closure().detach().item())
        elif optimizer_name == "zo-switch":
            g_closure = make_batch_closure(model, x_train, y_train, criterion, l2_reg=l2_reg_g)
            f_closure = make_batch_closure(model, x_val, y_val, criterion, l2_reg=l2_reg_f)
            stats = optimizer.step(f_closure=f_closure, g_closure=g_closure)
            loss_value = float(g_closure().detach().item())
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        with torch.no_grad():
            pred = model(x_train).argmax(dim=1)
            batch_acc = float((pred == y_train).float().mean().item())

        running_loss += loss_value
        running_acc += batch_acc
        n_batches += 1

        if log_writer is not None:
            row = {
                "epoch": epoch,
                "batch": batch_idx,
                "optimizer": optimizer_name,
                "train_loss": loss_value,
                "train_acc": batch_acc,
                "branch": stats.get("branch", ""),
                "grad_g_norm": stats.get("grad_g_norm", math.nan),
                "grad_step_norm": stats.get("grad_step_norm", stats.get("grad_norm", math.nan)),
                "update_norm": stats.get("update_norm", math.nan),
                "I_count": stats.get("I_count", math.nan),
                "f_branch_count": stats.get("f_branch_count", math.nan),
                "g_branch_count": stats.get("g_branch_count", math.nan),
                "f_value": stats.get("f_value", math.nan),
                "g_value": stats.get("g_value", math.nan),
            }
            log_writer.writerow(row)

    return {
        "train_loss": running_loss / max(n_batches, 1),
        "train_acc": running_acc / max(n_batches, 1),
        "num_batches": n_batches,
    }


def plot_training_curves(log_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Create simple diagnostic plots from the CSV log."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plots because plotting dependencies are unavailable: {exc}")
        return []

    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)
    if df.empty:
        print("Skipping plots because the log file is empty.")
        return []
    df["global_step"] = range(len(df))

    saved_paths: list[Path] = []

    loss_path = output_dir / "train_loss.png"
    plt.figure()
    plt.plot(df["global_step"], df["train_loss"])
    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.title("Training loss")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=160)
    plt.close()
    saved_paths.append(loss_path)

    acc_path = output_dir / "train_accuracy.png"
    plt.figure()
    plt.plot(df["global_step"], df["train_acc"])
    plt.xlabel("Training step")
    plt.ylabel("Training accuracy")
    plt.title("Training accuracy")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=160)
    plt.close()
    saved_paths.append(acc_path)

    if "grad_g_norm" in df.columns and df["grad_g_norm"].notna().any():
        grad_path = output_dir / "grad_g_norm.png"
        plt.figure()
        plt.plot(df["global_step"], df["grad_g_norm"])
        plt.xlabel("Training step")
        plt.ylabel("Estimated ||G_g||")
        plt.title("Lower-objective zeroth-order gradient norm")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(grad_path, dpi=160)
        plt.close()
        saved_paths.append(grad_path)

    if "branch" in df.columns and set(df["branch"].dropna().unique()).intersection({"f", "g"}):
        branch_path = output_dir / "branch_counts.png"
        counts = df["branch"].value_counts()
        plt.figure()
        plt.bar(counts.index.astype(str), counts.values)
        plt.xlabel("Branch")
        plt.ylabel("Count")
        plt.title("Switching branch counts")
        plt.tight_layout()
        plt.savefig(branch_path, dpi=160)
        plt.close()
        saved_paths.append(branch_path)

    return saved_paths
