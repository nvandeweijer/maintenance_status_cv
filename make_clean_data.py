"""This scripts load tabular data and images, and save them into a nice / clean format."""
import pandas as pd
import os
from glob import glob
import json
from tqdm.notebook import tqdm
import random
import argparse


def main(
    tabular_path: str = "/content/drive/MyDrive/nicole/after_businessrules.csv",
    image_dir_path: str = "/content/drive/MyDrive/nicole/part1/images_all/final_images/*/",
    save_at: str = "/content/drive/MyDrive/nicole/excellent_good.json",
    val_ratio: float=0.1,
    test_ratio: float=0.1,
    features_of_interest: list = [
        # "Unnamed: 0.1",
        # "Unnamed: 0",
        "published_on",
        "publication_date",
        "listing_price",
        "listing_size_m2",
        "total_rooms",
        "total_bedrooms",
        "listing_street_name",
        "listing_house_nr",
        "listing_postal_code",
        "listing_city_name",
        "interior_type",
        "listing_residential_type",
        "listing_volume_m3",
        "balcony",
        "number_of_floors_building",
        "floor_level",
        "energy_label",
        "garage",
        "storage",
        "garden",
        "parking_availability",
        "parking_type",
        # "source",
        # "bag_id",
        "total_bathrooms",
        "listing_residential_type_detailed",
        "construction_year",
        # "listing_photo_urls",
        # "id",
        "maintenance_status",
        "acceptance_category",
        "service_costs",
        "max_rent_period",
        "min_rent_periode",
        "acceptance_date",
        "deposit_amount",
        "available_since",
        "building_type",
        "energy_index",
        "lon",
        "lat",
        "furnished",
        "monumental building",
        "protected townscape",
        "shell",
        "upholstered",
        "publication_published",
        "listing_price_sqm",
        "publication_year",
        "buurtnaam2020",
        "gm_naam",
        "pc6",
        "wk_naam",
        "woonplaats",
        "number_of_photos",
        "foldername",
        # "pred_part1_class0"
    ],
) -> None:
    """Run the data cleaning pipeline.
    Args
    ----
    tabular_path:
    image_dir_path:
    features_of_interest:
    save_at:
    """

    tabular_data = pd.read_csv(tabular_path)
    tabular_data = tabular_data.dropna(how="all", axis=1)
    imagefolder_paths = glob(os.path.join(image_dir_path, "*"))
    data = []
    for path in tqdm(imagefolder_paths):
        foldername = path.split("/")[-1]

        rows_matched = tabular_data.loc[tabular_data["foldername"] == foldername]

        if len(rows_matched) != 1:
            continue

        condition = rows_matched.maintenance_status.values.item()
        if condition != path.split("/")[-2]:
          continue

        # if you want to include the 'good' maintenance status, comment the following line
        if condition == "good":
          continue

        features = rows_matched[features_of_interest].to_dict(orient="records")
        assert len(features) == 1

        images_paths = glob(os.path.join(path, "*"))
        room_types = []
 
        for path in images_paths:
            image_path = path.split("/")[-1]
            room_types.append({"image_path": image_path})

        data.append(
            {
                "house": foldername,
                "maintenance_status": condition,
                "features": features[0],
                "room_types": room_types,
            }
        )

    random.seed(42)
    random.shuffle(data)
    train_idx = int(len(data) * (1 - val_ratio - test_ratio))
    val_idx = int(len(data) * (1 - test_ratio))
    test_idx = int(len(data))

    train_split = data[:train_idx]
    val_split = data[train_idx:val_idx]
    test_split = data[val_idx:test_idx]

    assert len(data) == len(train_split) + len(val_split) + len(test_split), "lol"

    data = {"train": train_split, "val": val_split, "test": test_split}

    with open(save_at, "w") as stream:
        json.dump(data, stream, indent=4)

    print(f"Done! data saved at {save_at}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from the csv.")
    
    parser.add_argument(
        "--tabular_path",
        type=str,
        default="/content/drive/MyDrive/nicole/after_businessrules.csv",
        help="Path to the csv file.",
    )

    parser.add_argument(
        "--image_dir_path",
        type=str,
        default="/content/drive/MyDrive/nicole/part1/images_all/final_images/*/",
        help="Directory where images are saved.",
    )

    parser.add_argument(
        "--save_at",
        type=str,
        default="/content/drive/MyDrive/nicole/excellent_good.json",
        help="Directory where the json file will be saved",
    )

    args = vars(parser.parse_args())

    main(**args)
