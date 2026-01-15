"""
Training script for SimpleCIFARNet.

This script is provided for training the model checkpoint.
Students do not need to run this - they will use the pre-trained checkpoint.
"""
import os
import sys
import pathlib as pl
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from cifar10_simple import SimpleCIFARNet, get_cifar10_dataloaders
from cifar10_simple.utils import get_wandb_key

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(val_loader), 100.0 * correct / total


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if "LOG_PATH" in os.environ:
        wandb_save_dir = os.path.dirname(os.environ["LOG_PATH"])
    else:
        wandb_save_dir = "."

    wandb.login(key=get_wandb_key())
    wandb.init(
        project="ms-in-dnns-cifar10-simple",
        config=vars(args),
        name=args.run_name,
        dir=wandb_save_dir,
    )

    train_loader, val_loader = get_cifar10_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = SimpleCIFARNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        wandb.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "epoch": epoch + 1,
        })

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, args.ckpt_path)
            print(f"Saved best checkpoint with val_acc: {val_acc:.2f}%")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    if "LOG_PATH" in os.environ:
        bucket_name = os.environ["BUCKET"].split("gs://")[1]
        data_root = str(pl.PurePosixPath("/gcs", bucket_name, "cifar10_data"))
    else:
        data_root = str(pl.PurePath("..", "..", "data", "cifar10"))

    parser.add_argument("--data-root", type=str, default=data_root)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt-path", type=str, default="pretrained_cifar10.ckpt")

    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)

    args = parser.parse_args()
    main(args)
