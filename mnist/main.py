"""Main training and plotting entry point for deep-learning ZO experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from model import build_model
from optimizer import SwitchingZerothOrderOptimizer, ZerothOrderSGD
from utils import (
    accuracy,
    build_datasets,
    count_parameters,
    make_loaders,
    plot_training_curves,
    resolve_device,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep-learning testbed for Algorithm-2-style zeroth-order switching optimization."
    )

    # Dataset/model configuration.
    parser.add_argument("--dataset", choices=["fake", "mnist", "fashionmnist", "cifar10"], default="fake")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    # Training configuration.
    parser.add_argument("--optimizer", choices=["sgd", "adam", "zo-sgd", "zo-switch"], default="zo-switch")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=2)

    # Zeroth-order optimizer configuration.
    parser.add_argument("--gamma", type=float, default=1e-3, help="Default ZO radius.")
    parser.add_argument("--gamma_f", type=float, default=None, help="ZO radius for f-branch.")
    parser.add_argument("--gamma_g", type=float, default=None, help="ZO radius for g-branch.")
    parser.add_argument("--epsilon_g", type=float, default=1e-2)
    parser.add_argument("--zo_samples", type=int, default=1)
    parser.add_argument("--zo_samples_f", type=int, default=None)
    parser.add_argument("--zo_samples_g", type=int, default=None)
    parser.add_argument("--max_update_norm", type=float, default=None)
    parser.add_argument("--clamp_params", nargs=2, type=float, default=None)
    parser.add_argument("--l2_reg_f", type=float, default=0.0)
    parser.add_argument("--l2_reg_g", type=float, default=0.0)
    parser.add_argument("--evaluate_f_every_step", action="store_true")

    # Device/reproducibility/logging.
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available; otherwise fall back to CPU.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log_path", type=str, default="logs/deep_zo_run.csv")
    parser.add_argument("--plot", action="store_true", help="Save diagnostic plots after training.")
    parser.add_argument("--plot_dir", type=str, default="plots/deep_zo")

    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "zo-sgd":
        return ZerothOrderSGD(
            model.parameters(),
            lr=args.lr,
            gamma=args.gamma,
            n_samples=args.zo_samples,
            weight_decay=args.weight_decay,
            max_update_norm=args.max_update_norm,
            clamp=tuple(args.clamp_params) if args.clamp_params is not None else None,
        )
    if args.optimizer == "zo-switch":
        return SwitchingZerothOrderOptimizer(
            model.parameters(),
            lr=args.lr,
            gamma_f=args.gamma_f if args.gamma_f is not None else args.gamma,
            gamma_g=args.gamma_g if args.gamma_g is not None else args.gamma,
            epsilon_g=args.epsilon_g,
            n_samples_f=args.zo_samples_f if args.zo_samples_f is not None else args.zo_samples,
            n_samples_g=args.zo_samples_g if args.zo_samples_g is not None else args.zo_samples,
            weight_decay=args.weight_decay,
            max_update_norm=args.max_update_norm,
            clamp=tuple(args.clamp_params) if args.clamp_params is not None else None,
            evaluate_f_every_step=args.evaluate_f_every_step,
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def write_summary(summary_path: Path, summary: dict) -> None:
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.cuda)

    train_set, val_set, test_set, input_shape, num_classes = build_datasets(
        args.dataset,
        args.data_dir,
        args.val_fraction,
        args.seed,
    )
    train_loader, val_loader, test_loader = make_loaders(
        train_set,
        val_set,
        test_set,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, input_shape, num_classes, args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args)

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "batch",
        "optimizer",
        "train_loss",
        "train_acc",
        "branch",
        "grad_g_norm",
        "grad_step_norm",
        "update_norm",
        "I_count",
        "f_branch_count",
        "g_branch_count",
        "f_value",
        "g_value",
    ]

    print("=== Configuration ===")
    printable_config = vars(args).copy()
    printable_config["resolved_device"] = str(device)
    printable_config["cuda_available"] = torch.cuda.is_available()
    print(json.dumps(printable_config, indent=2, sort_keys=True))
    print(f"model parameters: {count_parameters(model):,}")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(args.epochs):
            train_stats = train_one_epoch(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer_name=args.optimizer,
                epoch=epoch,
                max_train_batches=args.max_train_batches,
                l2_reg_f=args.l2_reg_f,
                l2_reg_g=args.l2_reg_g,
                log_writer=writer,
            )
            val_acc = accuracy(model, val_loader, device, max_batches=args.max_eval_batches)
            test_acc = accuracy(model, test_loader, device, max_batches=args.max_eval_batches)
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_stats['train_loss']:.4f} "
                f"train_acc={train_stats['train_acc']:.4f} "
                f"val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
            )

    summary = {
        "config": printable_config,
        "num_parameters": count_parameters(model),
        "final_val_acc": accuracy(model, val_loader, device, max_batches=args.max_eval_batches),
        "final_test_acc": accuracy(model, test_loader, device, max_batches=args.max_eval_batches),
        "log_path": str(log_path),
    }
    summary_path = log_path.with_suffix(log_path.suffix + ".summary.json")
    write_summary(summary_path, summary)

    print(f"saved log: {log_path}")
    print(f"saved summary: {summary_path}")

    if args.plot:
        saved_plots = plot_training_curves(log_path, args.plot_dir)
        for path in saved_plots:
            print(f"saved plot: {path}")


if __name__ == "__main__":
    main()
