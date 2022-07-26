import argparse
import os
import shutil
import torch

from PIL import Image
from glob import glob
from roomtype_classifier import RoomEfficientNet
from torchvision import transforms


transform = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
    mean = [0.5, 0.5, 0.5],
    std = [0.22, 0.22, 0.22])
])

def get_key(val, my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key
    return "key doesn't exist"

def main(images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RoomEfficientNet.load_from_checkpoint(
    checkpoint_path="/content/drive/MyDrive/nicole/part1/model_v2.ckpt", num_classes=7, efficientnet="efficientnet-b3")
    net.to(device)
    net.eval()
    net.freeze()

    labels: dict = {
            "interior": 0,
            "bathroom": 1,
            "bedroom": 2,
            "exterior": 3,
            "living_room": 4,
            "kitchen": 5,
            "dining_room": 6,
        }

    false_images = []
    for idx, img in enumerate(images):
        print(idx)
        label_condition = img.split(os.sep)[-3]
        imgI = Image.open(img)
        try:
            img_t = transform(imgI)
        except:
            false_images.append(img)
            print(img)
            continue

        batch_t = torch.unsqueeze(img_t, 0)
        out = net(batch_t.to(device))
        _, index = torch.max(out,1)
        
        label_type_room = get_key(index, labels)
        percentage = round((torch.nn.functional.softmax(out, dim=1)[0] * 100)[int(index)].item(), 2)

        if percentage <= 90 or label_type_room == "exterior" or label_type_room == "interior" or label_type_room == "bedroom" or label_type_room == "dining_room":
            imgI.close()
            shutil.move(img, "/content/drive/MyDrive/nicole/part1/images_all/excluded_images_roomtype")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from the csv.")
    parser.add_argument(
        "--images",
        type=str,
        default="/content/drive/MyDrive/nicole/part1/images_all/final_images/excellent/*/*",
        help="The location of the images you wish to apply the classifier to"
    )

    args = vars(parser.parse_args())

    main(**args)
