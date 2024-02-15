import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import lightning as L


class IncomeNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class PLIncomeModule(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = IncomeNet(108, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.classification.BinaryAccuracy(),
                "precision": torchmetrics.classification.BinaryPrecision(),
                "recall": torchmetrics.classification.BinaryRecall(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.best_metrics = metrics.clone(prefix="best/")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        self.train_metrics.update(torch.argmax(outputs, dim=-1), torch.argmax(labels, dim=-1))
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)
        self.log("step", float(self.current_epoch + 1), on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)

        acc = (preds == torch.argmax(labels, dim=-1)).sum() / inputs.shape[0]
        self.val_metrics.update(preds, torch.argmax(labels, dim=-1))
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log_dict(self.val_metrics, on_epoch=True, on_step=False)
        self.log("step", float(self.current_epoch + 1), on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)

        self.best_metrics.update(preds, torch.argmax(labels, dim=-1))
        self.log("best/loss", loss, on_epoch=True, on_step=False)
        self.log_dict(self.best_metrics, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
