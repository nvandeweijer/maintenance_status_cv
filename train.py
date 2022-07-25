import numpy as np
import copy
import time
import json

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt

from cleanlab.classification import CleanLearning

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor

import timm

from utils import balance_classes, ConditionDataset, image_paths_labels


class ConditionImgClassifier(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(ConditionImgClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 2)
        
    @torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.model(x)
        return x


class Classifier(BaseEstimator):
    def __init__(self, image_size, batch_size, epochs, model, val_img_paths, val_labels, data_size='large'):
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model
        self.val_img_paths = val_img_paths
        self.val_labels = val_labels
        self.data_size = data_size
        self.device = "cuda:0" #if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.best_model = None
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

    def fit(self, img_paths, labels):
        img_paths, labels = balance_classes(img_paths, labels)
        
        # train
        train_dataset = ConditionDataset(image_size=self.image_size, image_paths=img_paths, labels=labels, data_size=self.data_size, transform=self.train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)

        # validation
        val_dataset = ConditionDataset(image_size=self.image_size, image_paths=self.val_img_paths, labels=self.val_labels, data_size=self.data_size, transform=self.test_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, pin_memory=True)

        # loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1,1]).to(self.device)) 
        loss_fn = nn.CrossEntropyLoss() 

        optimizer = optim.AdamW(self.model.parameters())
        scaler = torch.cuda.amp.GradScaler()

        best_loss = 1000.0
        best_score = 0.0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        for epoch in range(int(self.epochs)):
            self.model.train()
            train_loss = 0.0
            train_score = 0.0
            t0 = time.time()
            for data in train_dataloader:
                x = data["image"].to(device=self.device) # error?                
                y = data["label"].to(device=self.device)
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                    loss = loss_fn(pred, y.to(torch.int64))
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                train_score += f1_score(y.detach().cpu().numpy(), pred.softmax(dim=1).argmax(dim=1).detach().cpu().numpy(), average='macro')
  
            train_loss = train_loss / len(train_dataloader)
            train_score = train_score / len(train_dataloader)

            
            val_loss = 0.0
            val_score = 0.0
            t1 = time.time()
            with torch.no_grad():
              for data in val_dataloader:
                val_x = data["image"].to(self.device)
                val_y = data["label"].to(self.device)
                with torch.cuda.amp.autocast():
                  val_pred = self.model(val_x)
                  val_loss = loss_fn(val_pred, val_y)
                val_loss += val_loss.item()
                val_score += f1_score(val_y.detach().cpu().numpy(), val_pred.softmax(dim=1).argmax(dim=1).detach().cpu().numpy(), average='macro')
              val_loss = val_loss / len(val_dataloader)
              val_score = val_score / len(val_dataloader)


            if best_score < val_score:
                self.best_model = copy.copy(self.model)
            print(f"{epoch} epoch | train loss: {train_loss:.4f} | train acc: {train_score:.4f} |val loss: {val_loss:.4f}| val acc: {val_score:.4f} | {time.time() - t0:.1f}s")


            train_losses.append(train_loss)
            if type(val_loss) == "float":
                val_losses.append(val_loss)
            else:
              val_loss = val_loss.item()
              val_losses.append(val_loss)
            train_accuracies.append(train_score)
            val_accuracies.append(val_score)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

        ax1.plot(train_losses, 'red', label='Training loss')
        ax1.plot(val_losses, 'blue', label='Validation loss')
        ax1.set_title("Train and val loss per epoch")
        ax1.set_ylabel("Loss")
        ax1.set_xlabel("Epochs")
        ax1.legend()

        ax2.plot(train_accuracies, 'red', label='Training accuracy')
        ax2.plot(val_accuracies, 'blue', label='Validation accuracy')
        ax2.set_title("Train and val accuracy per epoch")
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Epochs")
        ax2.legend()

        plt.show()

        return self
            
        
    def predict_proba(self, img_idx, phase="train"):
        breakpoint()
        dataset = ConditionDataset(image_size=self.image_size, image_paths=img_idx, transform=self.test_transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True)
  
        self.model.eval()
        preds = []
        with torch.no_grad():
            for data in dataloader:
                x = data["image"].to(self.device)
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                preds.append(pred.softmax(dim=1).detach().cpu().numpy())
        prob = np.concatenate(preds)
        return prob

    def predict(self, img_idx, phase="train"):
        prob = self.predict_proba(img_idx, phase=phase)
        preds = np.argmax(prob, axis=1)
        return preds

    def score(self, img_idx, label, phase="train"):
        preds = self.predict(img_idx, phase=phase)
        return accuracy_score(label, preds)

def main(model_name):
  data_path: str ="/content/drive/MyDrive/nicole/excellent_good.json" 
  image_root_dir: str ="/content/drive/MyDrive/nicole/part1/images_all/final_images"

  with open(data_path, "r") as stream:
    data = json.load(stream)

  train_img, train_y = image_paths_labels(data["train"])
  val_img, val_y = image_paths_labels(data["val"])

  condition_model = ConditionImgClassifier(model_name=model_name)
  model = Classifier(image_size=380, batch_size=32, epochs=15, model=condition_model, val_img_paths=val_img, val_labels=val_y, data_size='large')
  
  lnl = CleanLearning(clf=model) 
  lnl.fit(train_img, train_y)
