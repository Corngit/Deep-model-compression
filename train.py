import argparse
import csv
import json
import random

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from datasets.cifar100 import get_cifar100_loaders
from models.model_factory import create_model
from collections.abc import Sized

def set_seed(seed):
    """
    Make experiments more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_accuracy,
    args,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_val_accuracy":
            best_val_accuracy,

        "args": vars(args),
    }

    torch.save(
        checkpoint,
        path,
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train baseline image "
            "classification models."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v3_small",
        choices=[
            "resnet50",
            "mobilenet_v3_small",
        ],
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="baseline",
    )

    # These three are mainly for CPU debugging.
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Run    : {args.run_name}")
    print("=" * 60)

    # -------------------------------------------------
    # Output directories
    # -------------------------------------------------

    result_dir = (
        Path("results")
        / args.run_name
    )

    checkpoint_dir = (
        Path("checkpoints")
        / args.run_name
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save experiment configuration.
    with open(
        result_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            vars(args),
            f,
            indent=4,
        )

    # -------------------------------------------------
    # Data
    # -------------------------------------------------

    train_loader, val_loader, test_loader = (
        get_cifar100_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=(
                device.type == "cuda"
            ),
            max_train_samples=(
                args.max_train_samples
            ),
            max_val_samples=(
                args.max_val_samples
            ),
            max_test_samples=(
                args.max_test_samples
            ),
        )
    )
    assert isinstance(train_loader.dataset, Sized)
    assert isinstance(val_loader.dataset, Sized)
    assert isinstance(test_loader.dataset, Sized)
    print(
        f"Train samples : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Val samples   : "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test samples  : "
        f"{len(test_loader.dataset)}"
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = create_model(
        model_name=args.model,
        num_classes=100,
        pretrained=False,
    )

    model = model.to(device)

    # -------------------------------------------------
    # Loss / Optimizer / Scheduler
    # -------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    # -------------------------------------------------
    # CSV logging
    # -------------------------------------------------

    csv_path = (
        result_dir / "history.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ])

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    best_val_accuracy = -1.0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        val_loss, val_accuracy = (
            evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )
        )

        print(
            f"\nEpoch "
            f"[{epoch}/{args.epochs}]"
        )

        print(
            f"LR        : "
            f"{current_lr:.6f}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Acc : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss  : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Acc   : "
            f"{val_accuracy * 100:.2f}%"
        )

        # Save results to CSV.
        with open(
            csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.writer(
                csv_file
            )

            writer.writerow([
                epoch,
                current_lr,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            ])

        # Always save last checkpoint.
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_accuracy=(
                best_val_accuracy
            ),
            args=args,
        )

        # Save best model.
        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )

            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_accuracy=(
                    best_val_accuracy
                ),
                args=args,
            )

            print(
                "Saved new best model."
            )

        scheduler.step()

    # -------------------------------------------------
    # Final Test
    # -------------------------------------------------

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        checkpoint_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test_loss, test_accuracy = (
        evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )
    )

    print("\n" + "=" * 60)
    print("Final Test Result")
    print("=" * 60)

    print(
        f"Test Loss : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Acc  : "
        f"{test_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()