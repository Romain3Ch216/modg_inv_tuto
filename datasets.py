import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, scale_factor=4):
        super().__init__()
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.hr_size = 28
        self.lr_size = self.hr_size // scale_factor
        self.setup()

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()) # conversion en images binaires
        ])

        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
            )

        self.test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=4
        )
    
    def hr2lr(self, x):
        x = x.view(-1, 1, self.hr_size, self.hr_size)
        x =  F.interpolate(
            x, scale_factor=1/self.scale_factor, mode='bilinear', antialias=True
        ).view(-1, self.lr_size, self.lr_size)
        return x