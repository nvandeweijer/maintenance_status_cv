import json
import numpy as np
import os
import torch

from collections import Counter
from PIL import Image
from roomtype.roomtype_classifier import RoomEfficientNet
from torch.utils.data import Dataset
from torchvision import transforms


def get_key(index, labels):
    for key, value in labels.items():
         if index == value:
             return key
 
    return "key doesn't exist"


def crop_center_square(img: Image):
    if img.mode != "RGB":
        return None

    width, height = img.size

    if width > height:
        margin = (width - height) // 2
        left = margin
        right = width - margin
        top = 0
        bottom = height

    else:
        margin = (height - width) // 2
        left = 0
        right = width
        top = margin
        bottom = height - margin

    img = img.crop((left, top, right, bottom))

    return img


def balance_classes(img_paths, labels):
    classes: dict = {"excellent": 1, "bad": 0}
    balanced_img_paths = []
    balanced_labels = []

    counts = Counter(labels)
    min_class = min(counts, key=counts.get)
    class_counts = {class_int: 0 for class_int in list(classes.values())}

    for img_path, lab in zip(img_paths, labels):
      if class_counts[lab] >= counts[min_class]:
        continue
      class_counts[lab] += 1
      balanced_img_paths.append(img_path)              
      balanced_labels.append(lab)

    return balanced_img_paths, balanced_labels


class ImageDataset(Dataset):
    def __init__(self, 
                 split, 
                 image_size, 
                 image_paths=None, 
                 labels=None, 
                 data_size='large', 
                 transform=None, 
                 data_path="/content/drive/MyDrive/nicole/data_after90.json",
                 image_root_dir = "/content/drive/MyDrive/nicole/part1/images_all/final_images",
                 classes: dict = {"excellent": 1, "bad": 0},
                 desired_rooms: list = {"bathroom": 0.8, "kitchen": 0.8, "living_room": 0.8},
                 balance_classes: bool = True):
        self.split = split
        self.image_size = image_size
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.data_size = data_size
        self.data_path = data_path
        self.classes = classes
        self.balance_classes = balance_classes
        self.image_root_dir = image_root_dir
        self.desired_rooms = desired_rooms
        self._load_data()

    def _load_data(self):
        with open(self.data_path, "r") as stream:
            self.data = json.load(stream)[self.split]
        
        self.image_paths = []
        self.labels = []

        counts = []
        for sample in self.data:
          if sample["maintenance_status"] == "excellent":
            cond = 1
          elif sample["maintenance_status"] == "bad":
            cond = 0
          counts.extend([cond] * len(sample["room_types"]))
        counts = dict(Counter(counts))  
        min_class = min(counts, key=counts.get)
        self.class_counts = {class_int: 0 for class_int in list(self.classes.values())}

        for house in self.data:
            label_str = house["maintenance_status"]
            label = self.classes[label_str.lower()]

            for rt in house["room_types"]:
                pred_type = max(rt["annotation"], key=rt["annotation"].get)
                pred_prob = rt["annotation"][pred_type]

                if pred_type not in list(self.desired_rooms):
                    continue
                if pred_prob < self.desired_rooms[pred_type]:
                    continue

                if self.split.lower() == "train":
                    if self.data_size.lower() == "small":
                        if self.balance_classes and self.class_counts[label] >= counts[0]*0.2:
                            continue
                    elif self.data_size.lower() == "medium":
                        if self.balance_classes and self.class_counts[label] >= counts[0]*0.5:
                            continue
                    else:
                        if self.balance_classes and self.class_counts[label] >= counts[min_class]:
                            continue
                else:
                    if self.balance_classes and self.class_counts[label] >= counts[min_class]:
                        continue 
                
                self.class_counts[label] += 1

                self.image_paths.append(
                    os.path.join(self.image_root_dir, label_str, house["house"], rt["image_path"])
                )
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = img.convert("RGB")
        img = crop_center_square(img)
        img = img.resize(size=(self.image_size, self.image_size))
        img = np.array(img)

        if self.transform:
            img = self.transform(image=img)["image"]
        if self.labels is not None:
            label = self.labels[idx]
            return {"image_path": self.image_paths[idx], "image": img, "label": label}
        else:
            return {"image": img}


def image_paths_labels(data, image_root_dir):
    image_paths = []
    labels = []
    classes: dict = {"good": 2, "excellent": 1, "bad": 0}

    for house in data:
        label_str = house["maintenance_status"]
        label = classes[label_str.lower()]
        for rt in house["room_types"]:
            if label_str == "excellent" or label_str == "bad":
                image_paths.append(os.path.join(image_root_dir, label_str, house["house"], rt["image_path"]))
                labels.append(label)
            else: 
              continue
    return np.array(image_paths), np.array(labels)


def create_annotations_dict(image_path):
    labels: dict = {
            "interior": 0,
            "bathroom": 1,
            "bedroom": 2,
            "exterior": 3,
            "living_room": 4,
            "kitchen": 5,
            "dining_room": 6,
        }

    transform = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
    mean = [0.5, 0.5, 0.5],
    std = [0.22, 0.22, 0.22])])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RoomEfficientNet.load_from_checkpoint(
    checkpoint_path="/content/drive/MyDrive/nicole/part1/model_v2.ckpt", num_classes=7, efficientnet="efficientnet-b3")
    net.to(device)
    net.eval()
    net.freeze()

    imgI = Image.open(image_path)
    try:
        img_t = transform(imgI)
    except:
        return

    batch_t = torch.unsqueeze(img_t, 0)
    output = net(batch_t.to(device))
    prob = torch.nn.functional.softmax(output, dim=1)
    prob = prob[0]

    percentage_list = []
    for i in range(len(prob)):
        percentage_list.append(round(prob[i].item()*100, 4))

    annotations = {}
    for label, idx_label in labels.items(): 
        annotations[label]=prob[idx_label]

    return annotations


def balance_classes(img_paths, labels):
    classes: dict = {"excellent": 1, "bad": 0}
    balanced_img_paths = []
    balanced_labels = []

    counts = Counter(labels)
    min_class = min(counts, key=counts.get)
    class_counts = {class_int: 0 for class_int in list(classes.values())}

    for img_path, lab in zip(img_paths, labels):
      if class_counts[lab] >= counts[min_class]:
        continue
      class_counts[lab] += 1
      balanced_img_paths.append(img_path)              
      balanced_labels.append(lab)

    return balanced_img_paths, balanced_labels


def image_paths_labels(image_root_dir, data):
    image_paths = []
    labels = []
    classes: dict = {"excellent": 1, "bad": 0}

    for house in data:
        label_str = house["maintenance_status"]
        label = classes[label_str.lower()]
        for rt in house["room_types"]:
            if label_str == "excellent" or label_str == "bad":
                image_paths.append(os.path.join(image_root_dir, label_str, house["house"], rt["image_path"]))
                labels.append(label)
            else: 
              continue
    return np.array(image_paths), np.array(labels)

    