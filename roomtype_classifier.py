""" Copied from https://github.com/tae898/room-classification/blob/main/train.py """

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from efficientnet_pytorch import EfficientNet
from torch import nn


class RoomEfficientNet(pl.LightningModule):
    """
    EfficientNet class for training and inference.
    
    """

    def __init__(
        self, num_classes: int, efficientnet: str, weights_path: str = None) -> None:
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
        self.calc_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        out = self.efficient_net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        acc = self.calc_accuracy(y_hat, y)
        self.log(
            "train_acc",
            acc,
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
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.calc_accuracy(y_hat, y)

        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx) -> None:
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.calc_accuracy(y_hat, y)

        self.log("test_acc", acc, prog_bar=True, logger=True),
        self.log("test_loss", loss, prog_bar=True, logger=True)
        