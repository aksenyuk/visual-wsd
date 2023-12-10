"""
This module downloads Visual-WSD dataset and renames/restructures/simplifies it. 
All trial images are renamed (so they don't cross with train images) and moved to train ones.
All txt files are parsed and combined to single csv.

Final simplified structure of dataset is:
    -data
        -visual_wsd
            -images
                -image.number.jpg
                ...
            -dataset.csv
            
Columns of csv are: ['ambigues word', 'context (phrase)', 'target_image', {image_1 - image_9}(wrong images)]
"""

import asyncio
import os
import re
import shutil
from typing import Literal
from zipfile import ZipFile

import aiohttp
import pandas as pd
from aiohttp import ClientResponse

###### UNCOMMENT THIS IF YOU RUN IN JUPYTER ENVIROMENT
# import nest_asyncio
# nest_asyncio.apply()


class VisualWSDDownloader:
    """
    This class handles the downloading of the Visual-WSD dataset from Google Drive.
    It manages the virus scan page for large files, downloads the dataset in zip format, unzips it, and cleans up by removing the zip file.
    Additionally, it provides functionality to rename the dataset directories to a more manageable format.
    """

    def __init__(
        self, file_gdrive_id: str, zip_file_path: str, extract_to_path: str
    ) -> None:
        self.file_gdrive_id = file_gdrive_id
        self.zip_file_path = zip_file_path
        self.extract_to_path = extract_to_path

    async def download_file_from_google_drive(self) -> None:
        URL = "https://docs.google.com/uc?export=download"

        async with aiohttp.ClientSession() as session:
            initial_response = await session.get(
                URL, params={"id": self.file_gdrive_id}
            )
            token = await self.get_confirm_token(initial_response)

            if token:
                params = {"id": self.file_gdrive_id, "confirm": token}
                response = await session.get(URL, params=params)
            else:
                response = initial_response

            await self.save_response_content(response)

    async def get_confirm_token(self, response: ClientResponse) -> str:
        if "text/html" in response.headers.get("Content-Type", ""):
            text = await response.text()
            match = re.search("confirm=([0-9A-Za-z_]+)&", text)
            return match.group(1) if match else None
        return None

    async def save_response_content(self, response: ClientResponse) -> None:
        CHUNK_SIZE = 32768

        with open(self.zip_file_path, "wb") as f:
            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    def unzip_file(self) -> None:
        with ZipFile(self.zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.extract_to_path)

    def rename_directories(self) -> None:
        os.chdir("./data/")
        os.rename("./semeval-2023-task-1-V-WSD-train-v1", "./visual_wsd")
        os.rename("./visual_wsd/train_v1", "./visual_wsd/train")
        os.rename("./visual_wsd/trial_v1", "./visual_wsd/trial")
        os.rename("./visual_wsd/train/train_images_v1", "./visual_wsd/train/images")
        os.rename("./visual_wsd/trial/trial_images_v1", "./visual_wsd/trial/images")
        os.chdir("../")
        if os.path.exists(self.zip_file_path):
            print("visual_wsd zip file removed")
            os.remove(self.zip_file_path)

    async def run(self) -> None:
        print("===> Starting Visual_WSD Downloader")
        await self.download_file_from_google_drive()
        print("Visual_WSD dataset downloaded")
        self.unzip_file()
        print("Visual_WSD dataset unzipped")
        self.rename_directories()
        print("Visual_WSD dataset folders renamed\n")


class VisualWSDRestructurer:
    """
    This class is responsible for reorganizing the Visual-WSD dataset. 
    It renames and moves trial images to avoid name conflicts with training images, parses text files related to the dataset, 
    and combines this information into a single CSV file. The class also restructures the dataset into a simplified format 
    with a specific folder structure and dataset CSV.
    """

    def __init__(self, data_path: str, dataset_name: str) -> None:
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.path = os.path.join(data_path, dataset_name)

        self.max_num = self.find_max_image_number(
            os.path.join(self.path, "train", "images")
        )

    def find_max_image_number(self, images_path: str) -> int:
        max_num = 0
        for image_file in os.listdir(images_path):
            num = int(re.search(r"\d+", image_file).group())
            if num > max_num:
                max_num = num
        return max_num + 1

    def rename_move_trial_images(self) -> int:
        trial_images_path = os.path.join(self.path, "trial", "images")
        train_images_path = os.path.join(self.path, "train", "images")
        for filename in os.listdir(trial_images_path):
            match = re.search(r"\d+", filename)
            if match:
                number = int(match.group())
                new_number = self.max_num + number
                new_filename = filename.replace(str(number), str(new_number))
                shutil.move(
                    os.path.join(trial_images_path, filename),
                    os.path.join(train_images_path, new_filename),
                )

        if not os.listdir(trial_images_path):
            shutil.rmtree(trial_images_path)

    def txt2csv(
        self, path: str, datafile: str, goldfile: str, mode: Literal["train", "trial"]
    ) -> None:
        column_names = ["word", "context", "target"] + [
            f"image_{i}" for i in range(1, 11)
        ]

        data_file_path = os.path.join(path, datafile)
        df1 = pd.read_csv(data_file_path, sep="\t", header=None)

        gold_file_path = os.path.join(path, goldfile)
        df2 = pd.read_csv(gold_file_path, sep="\t", header=None)

        combined_df = pd.concat([df1.iloc[:, :2], df2, df1.iloc[:, 2:12]], axis=1)
        combined_df.columns = column_names

        def update_image_name(image_name: str) -> None:
            if mode == "trial":
                num = int(image_name.split(".")[1]) + self.max_num
                return f"image.{num}.jpg"
            return image_name

        combined_df["target"] = combined_df["target"].apply(update_image_name)
        for i in range(1, 11):
            combined_df[f"image_{i}"] = combined_df[f"image_{i}"].apply(
                update_image_name
            )

        combined_df["images"] = combined_df[
            [f"image_{i}" for i in range(1, 11)]
        ].values.tolist()
        combined_df["images"] = combined_df.apply(
            lambda row: [img for img in row["images"] if img != row["target"]], axis=1
        )

        for i in range(1, 10):
            combined_df[f"image_{i}"] = combined_df["images"].apply(
                lambda x: x[i - 1] if i <= len(x) else None
            )
        combined_df.drop(columns=["images", "image_10"], inplace=True)

        combined_df.to_csv(os.path.join(path, "dataset.csv"), index=False)

    def restructure(self) -> None:
        shutil.move(
            os.path.join(self.path, "train", "images"),
            os.path.join(self.path, "images"),
        )
        shutil.move(
            os.path.join(self.path, "train", "dataset.csv"),
            os.path.join(self.path, "dataset.csv"),
        )
        shutil.rmtree(os.path.join(self.path, "train"))
        shutil.rmtree(os.path.join(self.path, "trial"))

    def run(self) -> None:
        print("===> Starting Visual_WSD Restructurer")
        self.rename_move_trial_images()
        print("Visual_WSD dataset trial images reanamed")
        self.txt2csv(
            path=os.path.join(self.path, "train"),
            datafile="train.data.v1.txt",
            goldfile="train.gold.v1.txt",
            mode="train",
        )
        print("Visual_WSD dataset txt files parsed to csv")
        self.restructure()
        print("Visual_WSD dataset restructured\n")


async def main():
    visual_wsd_downloader = VisualWSDDownloader(
        "1byX4wpe1UjyCVyYrT04sW17NnycKAK7N", "./visual_wsd.zip", "./data/"
    )
    await visual_wsd_downloader.run()
    visual_wsd_restructurer = VisualWSDRestructurer("./data", "visual_wsd")
    visual_wsd_restructurer.run()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
