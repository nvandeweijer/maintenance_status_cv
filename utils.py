import numpy as np
import os
import torch

from collections import Counter
from PIL import Image
from roomtype.roomtype_classifier import RoomEfficientNet
from torch.utils.data import Dataset
from torchvision import transforms


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


class ConditionDataset(Dataset):
    def __init__(self, image_size, image_paths, labels=None, data_size='large', transform=None):
        self.image_size = image_size
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.data_size = data_size

        if data_size == "small":
          self.image_paths = image_paths[:20]
        elif data_size == "medium":
          self.image_paths = image_paths[:50]
        else:
          pass

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
    