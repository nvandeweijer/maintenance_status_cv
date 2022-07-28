"""
Inspired by https://github.com/tae898/room-classification
"""

import argparse
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
from collections import Counter
from PIL import Image
from sklearn.metrics import f1_score
from statistics import mode

from utils import ImageDataset

# torch
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# A datamodule is a shareable, reusable class that encapsulates all the steps needed to process data
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
class DataModule(pl.LightningDataModule):
    """Pytorch-lightning data module for the maintenance status dataset.
    This does initial setup, image augmentation, and loading dataloaser classes.
    """

    def __init__(self, image_size: int, batch_size: int) -> None:
        """
        Args
        ----
        image_size: image size (width and height)
        batch_size: batch size
        """
        super().__init__()
        self.image_size = image_size
        self.train_transform = A.Compose(
        [
            A.HorizontalFlip(),
            A.ShiftScaleRotate(),
            A.Normalize(mean=[0.6319, 0.5944, 0.5529], std=[0.2207, 0.2254, 0.2380]),
            ToTensor(),
        ]
    )
        self.test_transform = A.Compose(
        [
            A.Normalize(mean=[0.6319, 0.5944, 0.5529], std=[0.2207, 0.2254, 0.2380]),
            ToTensor(),
        ]
    )
        self.batch_size = batch_size

    def setup(
        self,
        data_path: str ="/content/drive/MyDrive/nicole/data_after90.json", 
        image_root_dir: str ="/content/drive/MyDrive/nicole/part1/images_all/final_images",
        stage=None
    ) -> None:
        """
        Args
        ----
        data_path: path to the splits.json
        image_root_dir: path to the images directory

        """

        self.train_dataset = ImageDataset(
            split="train",
            data_path=data_path,
            image_root_dir=image_root_dir,
            data_size="large",
            balance_classes = True,
            image_size=self.image_size,
            transform=self.train_transform,
        )

        self.val_dataset = ImageDataset(
            split = "val",
            data_path=data_path,
            image_root_dir=image_root_dir,
            balance_classes = True,
            image_size=self.image_size,
            transform=self.test_transform,
        )

        self.test_dataset = ImageDataset(
            split = "test",
            data_path=data_path,
            image_root_dir=image_root_dir,
            balance_classes = False,
            image_size=self.image_size,
            transform=self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


# A LightningModule organizes your PyTorch code into 6 sections:
## Computations (init).
## Train Loop (training_step)
## Validation Loop (validation_step)
## Test Loop (test_step)
## Prediction Loop (predict_step)
## Optimizers and LR Schedulers (configure_optimizers)
# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
class ConditionEfficientNet(pl.LightningModule):
    """EfficientNet class for training and inference."""

    def __init__(
        self, 
        num_classes: int, 
        efficientnet: str, 
        weights_path: str = None,   ## focus puntje
        classes: dict = {"good": 1, "excellent": 1, "bad": 0}
    ) -> None:
        """
        Args
        ----
        num_classes: number of classes
        efficientnet: EfficientNet type (e.g., efficientnet-b3)
        """
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained(
            efficientnet, num_classes=num_classes, weights_path=weights_path
        )
        in_features = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Linear(in_features, num_classes)
        self.calc_accuracy = torchmetrics.Accuracy(num_classes=num_classes, average="macro")
        # self.lr = lr
        self.classes = classes

    def forward(self, x):
        out = self.efficient_net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-05)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=torch.tensor([float(1),1]).to("cuda:0"))
        acc = self.calc_accuracy(logits, y)
        preds = torch.argmax(logits, dim=1)
        f1score = f1_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')

        self.log(
            "train_acc",
            acc,
            prog_bar=True,
            logger=True,
        ),
        self.log(
            "train_f1score",
            f1score,
            prog_bar=True,
            logger=True,
        ),
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch["image"], batch["label"]
        logits = self(x)
        val_loss = F.cross_entropy(logits, y, weight=torch.tensor([float(1),1]).to("cuda:0"))
        val_acc = self.calc_accuracy(logits, y)
        preds = torch.argmax(logits, dim=1)
        val_f1score = f1_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_f1score", val_f1score, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        x, y = batch["image"], batch["label"]
        logits = self(x)
        test_loss = F.cross_entropy(logits, y, weight=torch.tensor([float(1),1]).to("cuda:0"))
        test_acc = self.calc_accuracy(logits, y)
        preds = torch.argmax(logits, dim=1)
        test_f1score = f1_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')

        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)
        self.log("test_f1score", test_f1score, prog_bar=True)


def main(
    seed: int,
    batch_size: int,
    image_size: int,
    num_classes: int,
    efficientnet: str,
    epochs: int,
    use_gpu: bool,
    precision: int,
    patience: int,
) -> None:
    """Run training with the given arguments."""
    seed_everything(seed)

    # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss", 
        verbose=True,
        mode="min",
        filename="{epoch}_{val_loss:.4f}_{val_acc:02f}",
    )
    # https://pytorch-lightning.readthedocs.io/en/stable/common/early_stopping.html
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=patience, verbose=True, mode="min"
    ) 

    dm = DataModule(image_size=image_size, batch_size=batch_size)
    model = ConditionEfficientNet(num_classes=num_classes, efficientnet=efficientnet)

    if use_gpu:
        gpus = -1
    else:
        gpus = 0
        precision = 32

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=epochs,
        callbacks=[model_checkpoint, early_stop_callback],
        flush_logs_every_n_steps=1,
        log_every_n_steps=1,
        precision=precision,
    )
    trainer.fit(model=model, datamodule=dm)
    print("Validation:")
    trainer.test(dataloaders=dm.val_dataloader())
    print("Testing:")
    trainer.test(dataloaders=dm.test_dataloader())


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run training wiht pytorch-lightning.")
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Set the random seed for reproducibility",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="batch size",
        )
        parser.add_argument(
            "--image-size",
            type=int,
            default=300,
            help="height and width of target image.",
        )
        parser.add_argument(
            "--num-classes",
            type=int,
            default=7,
            help="number of classes.",
        )
        parser.add_argument(
            "--efficientnet",
            type=str,
            default="efficientnet-b3",
            help="efficientnet-b0, efficientnet-b1, efficientnet-b2, ...",
        )
        parser.add_argument(
            "--epochs", type=int, default=10, help="number of epochs to train"
        )
        parser.add_argument(
            "--use-gpu", action="store_true", help="whether to use GPU or not"
        )
        parser.add_argument(
            "--precision", type=int, default=16, help="GPU floating point precision"
        )
        parser.add_argument(
            "--patience", type=int, default=3, help="Early stopping patience epochs."
        )

        args = vars(parser.parse_args())

        main(**args)
