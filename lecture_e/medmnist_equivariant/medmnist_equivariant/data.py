"""
MedMNIST DataModule for PyTorch Lightning.

This module is provided - students do not need to modify it.
"""
import medmnist
from medmnist import INFO
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms


class MedMNISTDataModule(L.LightningDataModule):
    """
    DataModule for MedMNIST datasets.

    Supports rotation augmentation for equivariance experiments.
    """

    DATASET_CHOICES = ["pathmnist", "dermamnist", "bloodmnist", "tissuemnist"]

    def __init__(
        self,
        data_root: str,
        dataset_name: str = "pathmnist",
        batch_size: int = 64,
        num_workers: int = 4,
        rotation_augment: bool = False,
    ):
        """
        Args:
            data_root: Path to store/load the dataset
            dataset_name: Name of MedMNIST dataset (default: pathmnist)
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            rotation_augment: Whether to apply random rotation augmentation
        """
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rotation_augment = rotation_augment

        info = INFO[dataset_name]
        self.num_classes = len(info["label"])
        self.in_channels = info["n_channels"]
        self.class_names = list(info["label"].values())

    def setup(self, stage: str):
        DataClass = getattr(medmnist, INFO[self.dataset_name]["python_class"])

        # Base transform for all splits
        base_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * self.in_channels, std=[0.5] * self.in_channels),
        ]

        # Training transform with optional rotation augmentation
        if self.rotation_augment:
            train_transform = transforms.Compose(
                [transforms.RandomRotation(degrees=180)] + base_transform
            )
        else:
            train_transform = transforms.Compose(base_transform)

        val_transform = transforms.Compose(base_transform)

        self.train_dataset = DataClass(
            split="train", transform=train_transform, download=True, root=self.data_root
        )
        self.val_dataset = DataClass(
            split="val", transform=val_transform, download=True, root=self.data_root
        )
        self.test_dataset = DataClass(
            split="test", transform=val_transform, download=True, root=self.data_root
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
