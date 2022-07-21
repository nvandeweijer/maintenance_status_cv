import argparse
import ast
import os
import pandas as pd
import requests
from tqdm import tqdm


def main(image_dir: str, csv_file: str) -> None:
    """Download images from the csv."""

    if ".csv" in csv_file:
        data = pd.read_csv(csv_file)
    elif ".parquet" in csv_file:
        data = pd.read_parquet(csv_file, engine="fastparquet")
    else:
        raise ValueError("extension not correct!")

    os.makedirs(image_dir, exist_ok=True)

    for foo, status in zip(
        tqdm(data["listing_photo_urls"]), data["maintenance_status"]
    ):
        urls = ast.literal_eval(foo)
        for url in tqdm(urls):
            filename = url["resource"]
            uri = url["original_uri"]
            uri = uri.split("?")[0]

            try:
                # put timeout for 5 seconds to prevent hanging
                response = requests.get(uri, timeout=10)
                os.makedirs(
                    f"{image_dir}/{status}/{os.path.dirname(filename)}", exist_ok=True
                )
                file = open(f"{image_dir}/{status}/{filename}", "wb")
                file.write(response.content)
            except Exception as e:
                print(f"can't process {uri}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from the csv.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images",
        help="directory where images will be saved",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="photo_data_set_start_nicole.parquet",
        help="csv file name where image paths are stored. "
        "Use 'photo_data_set_start_nicole.parquet'",
    )

    args = vars(parser.parse_args())

    main(**args)
    