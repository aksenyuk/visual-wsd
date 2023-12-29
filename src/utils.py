import os
import random
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision.transforms import (  # Normalize,
    CenterCrop,
    Compose,
    InterpolationMode,
    Resize,
    ToTensor,
)

from datasets import VisualWSDDataset

transform = Compose(
    [
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        # Normalize(
        #     (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        # ),
    ]
)


def get_loaders(
    path: str,
    csv_file: str,
    images_folder: str,
    transform: Compose = transform,
    mode: Literal["train", "eval"] = "eval",
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    train_ratio: float = 0.9,
) -> DataLoader | tuple[DataLoader, DataLoader]:
    if mode == "eval":
        eval_dataset = VisualWSDDataset(
            path=path,
            csv_file=csv_file,
            images_folder=images_folder,
            transform=transform,
            mode="eval",
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return eval_loader

    elif mode == "train":
        train_dataset = VisualWSDDataset(
            path=path,
            csv_file=csv_file,
            images_folder=images_folder,
            transform=transform,
            mode="train",
            train_ratio=train_ratio,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        test_dataset = VisualWSDDataset(
            path=path,
            csv_file=csv_file,
            images_folder=images_folder,
            transform=transform,
            mode="train",
            train_ratio=train_ratio,
            test_split=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, test_loader

    else:
        raise ValueError(
            f"Invalid mode. Choose 'train' or 'eval'. Provided mode: {mode}"
        )


def get_metrics(targets: list, ranks: list) -> tuple[float]:
    accuracy = sum(targets) / len(targets)

    true_targets = [1] * len(targets)

    f1 = f1_score(true_targets, targets)
    prec = precision_score(true_targets, targets)
    rec = recall_score(true_targets, targets)

    mrr = np.mean([1 / rank for rank in ranks])

    return accuracy, f1, prec, rec, mrr


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
