import os
from typing import Literal, Optional

import pandas as pd
import torch
import PIL
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import Compose

PIL.Image.MAX_IMAGE_PIXELS = 1000000000


class VisualWSDDataset(Dataset):
    """
    This class implements a torch dataset for the Visual-WSD dataset, inheriting from PyTorch's Dataset class.
    The class supports both training and evaluation modes and includes functionality for splitting the dataset
    into training and evaluation subsets. It also supports custom transformations on the images.
    """

    def __init__(
        self,
        path: str,
        csv_file: str,
        images_folder: str,
        transform: Optional[Compose] = None,
        mode: Literal["train", "eval"] = "eval",
        train_ratio: float = 0.8,
    ) -> None:
        self.path = path
        self.df = pd.read_csv(os.path.join(path, csv_file))
        self.images_folder = images_folder
        self.transform = transform
        self.mode = mode
        self.train_ratio = train_ratio

        if mode == "train":
            self.train_data, self.test_data = train_test_split(
                self.df, train_size=train_ratio
            )
        elif mode == "eval":
            self.data = self.df
        else:
            raise ValueError(
                f"Invalid mode. Choose 'train' or 'eval'. Provided mode: {mode}"
            )

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.train_data)
        else:
            return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "train":
            row = self.train_data.iloc[idx]
        else:
            row = self.data.iloc[idx]

        target_img_name = os.path.join(self.path, self.images_folder, row["target"])
        target_image = Image.open(target_img_name).convert("RGB")
        if self.transform:
            target_image = self.transform(target_image)

        candidate_images = []
        for i in range(1, 10):
            img_name = os.path.join(self.path, self.images_folder, row[f"image_{i}"])
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            candidate_images.append(image)
        candidate_images = torch.stack(candidate_images)

        sample = {
            "word": row["word"],
            "context": row["context"],
            "target": torch.Tensor(target_image),
            "candidate_images": candidate_images,
        }
        return sample
