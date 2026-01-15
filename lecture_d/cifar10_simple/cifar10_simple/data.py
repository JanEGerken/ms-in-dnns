import os
import pathlib as pl

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_cifar10_dataloaders(
    data_root: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Get CIFAR10 data loaders for training and validation.

    Args:
        data_root: Path to the directory where CIFAR10 data is stored.
                   If None, uses Google Cloud path if LOG_PATH is set,
                   otherwise defaults to "../data/cifar10".
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if data_root is None:
        if "LOG_PATH" in os.environ:
            bucket_name = os.environ["BUCKET"].split("gs://")[1]
            data_root = str(pl.PurePosixPath("/gcs", bucket_name, "cifar10_data"))
        else:
            data_root = "../data/cifar10"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    val_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
