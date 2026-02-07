"""
Baseline (non-equivariant) CNN model for MedMNIST.

TODO: Implement the BaselineCNN and PLBaselineModule classes.
"""
import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import MulticlassAccuracy


class BaselineCNN(nn.Module):
    """
    Standard CNN baseline for MedMNIST classification.

    TODO: Implement a simple CNN with 3 convolutional blocks:
    - Each block: Conv2d -> ReLU -> MaxPool2d
    - Followed by a classifier (Linear layers)
    - Input: 28x28 images with `in_channels` channels
    - Output: `num_classes` logits
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # TODO: Implement the model architecture
        # Suggested structure:
        # - Block 1: in_channels -> 32 channels
        # - Block 2: 32 -> 64 channels
        # - Block 3: 64 -> 128 channels
        # - Classifier: flatten -> Linear -> ReLU -> Linear
        raise NotImplementedError("TODO: Implement BaselineCNN")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        raise NotImplementedError("TODO: Implement forward pass")


class PLBaselineModule(L.LightningModule):
    """
    PyTorch Lightning wrapper for BaselineCNN.

    TODO: Implement training_step, validation_step, test_step, and configure_optimizers.
    """

    def __init__(self, in_channels: int, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        # TODO: Initialize model, loss function, and metrics
        raise NotImplementedError("TODO: Implement PLBaselineModule.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Forward pass through the model
        raise NotImplementedError("TODO: Implement forward")

    def training_step(self, batch, batch_idx):
        # TODO: Implement training step
        # - Compute loss and accuracy
        # - Log metrics with self.log()
        raise NotImplementedError("TODO: Implement training_step")

    def validation_step(self, batch, batch_idx):
        # TODO: Implement validation step
        raise NotImplementedError("TODO: Implement validation_step")

    def test_step(self, batch, batch_idx):
        # TODO: Implement test step
        raise NotImplementedError("TODO: Implement test_step")

    def configure_optimizers(self):
        # TODO: Return optimizer (e.g., Adam)
        raise NotImplementedError("TODO: Implement configure_optimizers")
