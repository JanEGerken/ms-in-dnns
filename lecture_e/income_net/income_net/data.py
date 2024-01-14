import os
import urllib
import zipfile

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import lightning as L


class AdultDataset(Dataset):
    """Adult UCI dataset, download data from https://archive.ics.uci.edu/dataset/2/adult"""

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # one-hot encoding of categorical variables (including label)
        df = pd.get_dummies(df).astype("int32")

        data_columns = df.columns[:-2]
        labels_column = df.columns[-2:]
        self.data = torch.tensor(df[data_columns].values, dtype=torch.float32)
        self.labels = torch.tensor(df[labels_column].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class AdultDataModule(L.LightningDataModule):
    def __init__(self, data_root, batch_size=32, train_share=0.8):
        super().__init__()
        self.data_root = data_root
        self.csv_file = os.path.join(self.data_root, "adult.data")
        self.train_share = train_share
        self.batch_size = batch_size

    def prepare_data(self):
        if not os.path.isfile(self.csv_file):
            os.makedirs(self.data_root, exist_ok=True)

            zip_filename = os.path.join(self.data_root, "adult_data.zip")
            url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
            urllib.request.urlretrieve(url, zip_filename)

            with zipfile.ZipFile(zip_filename, "r") as zip_ref:
                zip_ref.extractall(self.data_root)

            os.remove(zip_filename)

    def setup(self, stage):
        entire_dataset = AdultDataset(self.csv_file)
        split_generator = torch.Generator()
        split_generator.manual_seed(0xDEADBEEF)
        self.train_dataset, self.val_dataset = random_split(
            entire_dataset, [self.train_share, 1 - self.train_share], generator=split_generator
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return self.val_dataloader()
