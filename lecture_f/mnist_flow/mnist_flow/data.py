import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
import lightning as L


class MNISTDataModule(L.LightningDataModule):

    CLASS_NAMES = list(range(10))

    def __init__(self, data_root, batch_size=128, pred_batch_size=10):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size

    def prepare_data(self):
        MNIST(self.data_root, train=True, download=True)
        MNIST(self.data_root, train=False, download=True)

    def setup(self, stage):

        self.val_dataset = MNIST(
            self.data_root, train=False, download=True, transform=transforms.ToTensor()
        )

        self.train_dataset = MNIST(
            self.data_root, train=True, download=True, transform=transforms.ToTensor()
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        dummy_pred_dataset = TensorDataset(torch.zeros(self.pred_batch_size, 1, 28, 28))
        return DataLoader(dummy_pred_dataset, batch_size=self.pred_batch_size)
