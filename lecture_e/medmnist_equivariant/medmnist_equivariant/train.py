"""
Training script for MedMNIST equivariance experiments.

Usage:
    python -m medmnist_equivariant.train --model baseline
    python -m medmnist_equivariant.train --model equivariant --rotation-augment
"""
import os
import sys
import pathlib as pl
from argparse import ArgumentParser
from datetime import datetime

import torch
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from medmnist_equivariant.data import MedMNISTDataModule
from medmnist_equivariant.baseline_model import PLBaselineModule
from medmnist_equivariant.equivariant_model import PLC4EquivariantModule
from medmnist_equivariant.utils import get_wandb_key

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


def main(args):
    L.seed_everything(42, workers=True)
    torch.hub.set_dir(args.torch_cache_dir)

    if "LOG_PATH" in os.environ:
        wandb_save_dir = os.path.dirname(os.environ["LOG_PATH"])
    else:
        wandb_save_dir = "."

    # Initialize WandB
    wandb.login(key=get_wandb_key())

    # Setup data module
    dm = MedMNISTDataModule(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rotation_augment=args.rotation_augment,
    )

    # Select model
    if args.model == "baseline":
        model = PLBaselineModule(
            in_channels=dm.in_channels if hasattr(dm, "in_channels") else 3,
            num_classes=dm.num_classes if hasattr(dm, "num_classes") else 9,
            lr=args.lr,
        )
    elif args.model == "equivariant":
        model = PLC4EquivariantModule(
            in_channels=dm.in_channels if hasattr(dm, "in_channels") else 3,
            num_classes=dm.num_classes if hasattr(dm, "num_classes") else 9,
            lr=args.lr,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Setup logger
    logger = WandbLogger(
        project="ms-in-dnns-medmnist-equivariant",
        name=args.run_name,
        config=vars(args),
        save_dir=wandb_save_dir,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            RichProgressBar(),
            RichModelSummary(max_depth=2),
        ],
    )

    # Train
    trainer.fit(model, dm)

    # Test on best checkpoint
    trainer.test(model, dm, ckpt_path="best")

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Data arguments
    if "LOG_PATH" in os.environ:
        bucket_name = os.environ["BUCKET"].split("gs://")[1]
        data_root = str(pl.PurePosixPath("/gcs", bucket_name, "medmnist_data"))
    else:
        data_root = str(pl.PurePath("..", "..", "data", "medmnist"))
    parser.add_argument("--data-root", type=str, default=data_root)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="pathmnist",
        choices=MedMNISTDataModule.DATASET_CHOICES,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "equivariant"],
        help="Model type to train",
    )
    parser.add_argument("--rotation-augment", action="store_true")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)

    # Logging
    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)

    if "LOG_PATH" in os.environ:
        bucket_name = os.environ["BUCKET"].split("gs://")[1]
        torch_cache_dir = str(pl.PurePosixPath("/gcs", bucket_name, "torch_cache"))
    else:
        torch_cache_dir = str(pl.PurePath("..", "..", "torch_cache"))
    parser.add_argument("--torch-cache-dir", type=str, default=torch_cache_dir)

    args = parser.parse_args()
    main(args)
