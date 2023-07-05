"""Provides a custom Dataset class for building a PyTorch dataset"""
from pathlib import Path
from PIL import Image
import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for building a PyTorch dataset

    Args:
        root_path (str): The root directory path of the dataset.
        split (str, optional): The split of the dataset, either "train" or "test".
            Default is "train".
        transform (callable, optional): A callable function or transformation to 
            be applied to the data. Default is None.

    Raises:
        ValueError: If the provided split is not valid (neither "train" nor "test").

    Attributes:
        split (str): The split of the dataset ("train" or "test").
        transform (callable): The transformation to be applied to the data.
        sample_images (list): A list containing the paths to the sample images in the dataset.
        sample_targets (list): A list containing the paths to the corresponding target images
            in the dataset.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Returns the image and its target segmentation at the given index.

    Usage:
        dataset = Dataset(root_path="path/to/dataset", split="train", 
            transform=albumentations.Compose([ToTensorV2()]))
        image, target = dataset[idx]
    """
    def __init__(self, root_path: str, split="train", transform=None):
        super().__init__()

        # Check if split is valid
        if split not in ["train", "test"]:
            raise ValueError("Split must be either train or test")

        self.split = split
        self.transform = transform

        path = Path(root_path) / split

        self.sample_images = [file for file in sorted((path / "images").iterdir())]
        self.sample_targets = [file for file in sorted((path / "labels").iterdir())]
        assert len(self.sample_images) == len(self.sample_targets)
        
    def __len__(self):
        return len(self.sample_images)

    def __getitem__(self, idx):
        image_path = self.sample_images[idx]
        target_path = self.sample_targets[idx]

        image = Image.open(str(image_path))
        image = np.array(image)
        target = Image.open(str(target_path))
        target = np.array(target)

        if self.transform:
            transformed = self.transform(image=image, mask=target)
            image = transformed["image"]
            target = transformed["mask"]

        return image, target
