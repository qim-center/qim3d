"""Fixtures shared across the test suite"""

from pathlib import Path

import numpy as np
import pytest

from qim3d.io import save


@pytest.fixture
def make_dataset(tmp_path):
    """
    Factory fixture that creates a temporary dataset for testing deep learning tools.

    Creates two folders, 'train' and 'test', who each also have two subfolders 'images' and 'labels'.
    n random images are then added to all four subfolders.
    The dataset is placed in a temporary directory that pytest removes after the test.

    Args:
        folder (str or Path, optional): Where to create the dataset. Defaults to the test's `tmp_path`.
        n (int, optional): Number of random images and labels in the temporary dataset.
        img_shape (tuple, optional): Tuple with the depth, height and width of the images and labels.

    Returns:
        Path: The folder containing the dataset.

    Example:
        >>> def test_something(make_dataset):
        ...     folder = make_dataset(n=10, img_shape=(16, 16, 16))

    """

    def _make(folder=None, n=3, img_shape=(32, 32, 32)):
        folder = tmp_path if folder is None else Path(folder)

        # Random image
        img = np.random.randint(2, size=img_shape, dtype=np.uint8)

        for split in ["train", "test"]:
            for sub_folder in ["images", "labels"]:
                path = folder / split / sub_folder
                path.mkdir(parents=True)
                for i in range(n):
                    save(
                        str(path / f"img_{split}{i}.nii.gz"),
                        img,
                        compression=True,
                        replace=True,
                    )

        return folder

    return _make
