"""Provides a custom Dataset class for building a PyTorch dataset."""

from collections.abc import Callable
from pathlib import Path

import numpy as np

import qim3d
from qim3d.utils import log
from qim3d.utils._dependencies import optional_import

from ._augmentations import Augmentation

torch = optional_import('torch', extra='deep-learning')


class Dataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for building a PyTorch dataset.

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

    """

    def __init__(
        self, root_path: str, split: str = 'train', transform: Callable | None = None
    ):
        super().__init__()

        # Check if split is valid
        if split not in ['train', 'test']:
            msg = f"Invalid split: {split}. Use either 'train' or 'test'."
            raise ValueError(msg)

        self.split = split
        self.transform = transform

        path = Path(root_path) / split

        self.sample_images = sorted((path / 'images').iterdir())
        self.sample_targets = sorted((path / 'labels').iterdir())
        assert len(self.sample_images) == len(self.sample_targets)

        # Checking the characteristics of the dataset
        self.check_shape_consistency(self.sample_images)

    def __len__(self):
        return len(self.sample_images)

    def __getitem__(self, idx: int):
        image_path = self.sample_images[idx]
        target_path = self.sample_targets[idx]

        # Load 3D volume
        image = qim3d.io.load(image_path)
        target = qim3d.io.load(target_path)

        # Add extra channel dimension
        image = np.expand_dims(image, axis=0)
        target = np.expand_dims(target, axis=0)

        if self.transform:
            # Apply augmentation
            transformed = self.transform({'image': image, 'label': target})
            image = transformed['image']
            target = transformed['label']

        image = image.clone().detach().to(dtype=torch.float32)
        target = target.clone().detach().to(dtype=torch.float32)

        return image, target

    # TODO: working with images of different sizes
    def check_shape_consistency(self, sample_images: tuple[str]) -> bool:
        image_shapes = []
        for image_path in sample_images:
            image_shape = self._get_shape(image_path)
            image_shapes.append(image_shape)

        # Check if all images have the same size
        consistency_check = all(i == image_shapes[0] for i in image_shapes)

        if not consistency_check:
            msg = 'Only images of all the same size can be processed at the moment'
            raise NotImplementedError(msg)
        else:
            log.debug('Images are all the same size!')
        return consistency_check

    def _get_shape(self, image_path: str) -> tuple:
        # Load 3D volume
        image = qim3d.io.load(image_path)
        return image.shape


def check_resize(
    orig_shape: tuple,
    resize: tuple,
    n_channels: int,
) -> tuple:
    """
    Checks and adjusts the resize dimensions based on the original shape and the number of channels.

    Args:
        orig_shape (tuple): Original shape of the image.
        resize (tuple): Desired resize dimensions.
        n_channels (int): Number of channels in the model.

    Returns:
        tuple: Final resize dimensions.

    Raises:
        ValueError: If the image size is smaller than minimum required for the model's depth.

    """

    # 3D images
    orig_d, orig_h, orig_w = orig_shape
    final_d = resize[0] if resize[0] else orig_d
    final_h = resize[1] if resize[1] else orig_h
    final_w = resize[2] if resize[2] else orig_w

    # Finding suitable size to upsize with padding
    if resize == 'padding':
        final_d = (orig_d // 2**n_channels + 1) * 2**n_channels
        final_h = (orig_h // 2**n_channels + 1) * 2**n_channels
        final_w = (orig_w // 2**n_channels + 1) * 2**n_channels

    # Finding suitable size to downsize with crop / resize
    else:
        final_d = (orig_d // 2**n_channels) * 2**n_channels
        final_h = (orig_h // 2**n_channels) * 2**n_channels
        final_w = (orig_w // 2**n_channels) * 2**n_channels

        # Check if the image size is too small compared to the model's depth
        if final_d == 0 or final_h == 0 or final_w == 0:
            msg = (
                "The size of the image is too small compared to the depth of the UNet. \
                   Choose a different 'resize' and/or a smaller model."
            )

            raise ValueError(msg)

        if final_d != orig_d or final_h != orig_h or final_w != orig_w:
            log.warning(f"The image size doesn't match the Unet model's depth. \
                          The image is changed with '{resize}', from {orig_h, orig_w} to {final_h, final_w}.")

    return final_d, final_h, final_w


def prepare_datasets(
    path: str,
    val_fraction: float,
    model: torch.nn.Module,
    augmentation: Augmentation,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Loads, splits, and applies augmentations to the dataset.

    This function automates the creation of PyTorch datasets for training, validation, and testing. It handles:
    
    1.  **Loading**: Reads data from the specified directory structure (expects `train/images`, `train/labels`, `test/images`, `test/labels`).
    2.  **Splitting**: Divides the training data into a training set and a validation set based on `val_fraction`.
    3.  **Augmentation**: Applies the transformation pipelines defined in the `augmentation` object to each split.

    Args:
        path (str): The root directory of the dataset.
        val_fraction (float): The proportion of the training data to reserve for validation (0.0 to 1.0).
        model (torch.nn.Module): The PyTorch model. (Used to determine the expected input dimensions/channels for resizing checks).
        augmentation (Augmentation): An instance of `qim3d.ml.Augmentation` containing the transformation rules for each split.

    Returns:
        (train_set, val_set, test_set): A tuple of `torch.utils.data.Subset` objects ready for use with a DataLoader.

    Raises:
        ValueError: If `val_fraction` is not a float between 0 and 1.

    Example:
        ```python
        import qim3d
        
        # Define parameters
        base_path = "dataset"
        model = qim3d.ml.models.UNet(size='small')
        
        # Configure augmentation
        aug_pipeline = qim3d.ml.Augmentation(resize='crop', transform_train='light')

        # Generate datasets
        train_ds, val_ds, test_ds = qim3d.ml.prepare_datasets(
            path=base_path, 
            val_fraction=0.2, 
            model=model, 
            augmentation=aug_pipeline
        )
        
        print(f"Training samples: {len(train_ds)}")
        print(f"Validation samples: {len(val_ds)}")
        ```
    """

    if not isinstance(val_fraction, float) or not (0 <= val_fraction < 1):
        msg = 'The validation fraction must be a float between 0 and 1.'
        raise ValueError(msg)

    resize = augmentation.resize
    n_channels = len(model.channels)

    # Get the first image to check the shape
    im_path = Path(path) / 'train'
    first_img = sorted((im_path / 'images').iterdir())[0]

    # Load 3D volume
    image = qim3d.io.load(first_img)
    orig_shape = image.shape

    final_shape = check_resize(orig_shape, resize, n_channels)

    train_set = Dataset(
        root_path=path,
        transform=augmentation.augment(final_shape, level=augmentation.transform_train),
    )
    val_set = Dataset(
        root_path=path,
        transform=augmentation.augment(
            final_shape, level=augmentation.transform_validation
        ),
    )
    test_set = Dataset(
        root_path=path,
        split='test',
        transform=augmentation.augment(final_shape, level=augmentation.transform_test),
    )

    split_idx = int(np.floor(val_fraction * len(train_set)))
    indices = torch.randperm(len(train_set))

    train_set = torch.utils.data.Subset(train_set, indices[split_idx:])
    val_set = torch.utils.data.Subset(val_set, indices[:split_idx])

    return train_set, val_set, test_set


def prepare_dataloaders(
    train_set: torch.utils.data,
    val_set: torch.utils.data,
    test_set: torch.utils.data,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 8,
    pin_memory: bool = False,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    Wraps the datasets into PyTorch DataLoaders for efficient batch processing.

    DataLoaders handle the complexity of batching, shuffling, and parallel data loading. This function ensures that your model receives data in the correct format and speed during training.

    **Note:** It is recommended to keep `shuffle_train=True` to improve model generalization.

    Args:
        train_set (torch.utils.data.Dataset): The training dataset.
        val_set (torch.utils.data.Dataset): The validation dataset.
        test_set (torch.utils.data.Dataset): The testing dataset.
        batch_size (int): The number of samples to process in a single step (batch).
        shuffle_train (bool, optional): If `True`, the training data is reshuffled at every epoch. This prevents the model from learning the order of the data. Defaults to `True`.
        num_workers (int, optional): The number of subprocesses to use for data loading. `0` means that the data will be loaded in the main process. Increasing this speeds up data loading but consumes more RAM. Defaults to 8.
        pin_memory (bool, optional): If `True`, the data loader will copy Tensors into CUDA pinned memory before returning them. This can speed up data transfer to the GPU. Defaults to `False`.

    Returns:
        (train_loader, val_loader, test_loader): A tuple containing the three `torch.utils.data.DataLoader` objects.

    Example:
        ```python
        import qim3d

        # ... (assume datasets are already created as train_set, val_set, test_set) ...

        # Create DataLoaders
        train_loader, val_loader, test_loader = qim3d.ml.prepare_dataloaders(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            batch_size=4,
            num_workers=4
        )
        
        # Iterate through the training loader
        for batch in train_loader:
            inputs, labels = batch['image'], batch['label']
            # training step...
        ```
    """
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
