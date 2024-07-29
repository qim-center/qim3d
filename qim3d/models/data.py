"""Provides a custom Dataset class for building a PyTorch dataset."""
from pathlib import Path
from PIL import Image
from qim3d.utils.logger import log
import torch
import numpy as np


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

        # checking the characteristics of the dataset
        self.check_shape_consistency(self.sample_images)

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


    # TODO: working with images of different sizes
    def check_shape_consistency(self,sample_images):
        image_shapes= []
        for image_path in sample_images:
            image_shape = self._get_shape(image_path)
            image_shapes.append(image_shape)

        # check if all images have the same size.
        consistency_check = all(i == image_shapes[0] for i in image_shapes)

        if not consistency_check:
            raise NotImplementedError(
                "Only images of all the same size can be processed at the moment"
            )
        else:
            log.debug(
                "Images are all the same size!"
            )
        return consistency_check
    
    def _get_shape(self,image_path):
        return Image.open(str(image_path)).size


def check_resize(im_height: int, im_width: int, resize: str, n_channels: int):
    """
    Checks the compatibility of the image shape with the depth of the model.
    If the image height and width cannot be divided by 2 `n_channels` times, then the image size is inappropriate.
    If so, the image is changed to the closest appropriate dimension, and the user is notified with a warning.

    Args:
        im_height (int) : Height of the original image from the dataset.
        im_width (int)  : Width of the original image from the dataset.
        resize (str)    : Type of resize to be used on the image if the size doesn't fit the model.
        n_channels (int): Number of channels in the model.

    Raises:
        ValueError: If the image size is smaller than minimum required for the model's depth.               
    """
    # finding suitable size to upsize with padding 
    if resize == 'padding':
        h_adjust, w_adjust = (im_height // 2**n_channels+1) * 2**n_channels , (im_width // 2**n_channels+1) * 2**n_channels
        
    # finding suitable size to downsize with crop / resize
    else:
        h_adjust, w_adjust = (im_height // 2**n_channels) * 2**n_channels , (im_width // 2**n_channels) * 2**n_channels    
    
    if h_adjust == 0 or w_adjust == 0:
        raise ValueError("The size of the image is too small compared to the depth of the UNet. Choose a different 'resize' and/or a smaller model.")
    
    elif h_adjust != im_height or w_adjust != im_width:
        log.warning(f"The image size doesn't match the Unet model's depth. The image is changed with '{resize}', from {im_height, im_width} to {h_adjust, w_adjust}.")
    
    return h_adjust, w_adjust 


def prepare_datasets(path: str, val_fraction: float, model, augmentation):
    """
    Splits and augments the train/validation/test datasets.

    Args:
        path (str): Path to the dataset.
        val_fraction (float): Fraction of the data for the validation set.
        model (torch.nn.Module): PyTorch Model.
        augmentation (albumentations.core.composition.Compose): Augmentation class for the dataset with predefined augmentation levels.

    Raises:
        ValueError: if the validation fraction is not a float, and is not between 0 and 1. 
    """
    
    if not isinstance(val_fraction,float) or not (0 <= val_fraction < 1):
        raise ValueError("The validation fraction must be a float between 0 and 1.")

    resize = augmentation.resize
    n_channels = len(model.channels)

    # taking the size of the 1st image in the dataset
    im_path = Path(path) / 'train'
    first_img = sorted((im_path / "images").iterdir())[0]
    image = Image.open(str(first_img))
    orig_h, orig_w = image.size[:2]
        
    final_h, final_w = check_resize(orig_h, orig_w, resize, n_channels)

    train_set = Dataset(root_path = path, transform = augmentation.augment(final_h, final_w, augmentation.transform_train))
    val_set   = Dataset(root_path = path, transform = augmentation.augment(final_h, final_w, augmentation.transform_validation))
    test_set  = Dataset(root_path = path, split='test', transform = augmentation.augment(final_h, final_w, augmentation.transform_test))

    split_idx = int(np.floor(val_fraction * len(train_set)))
    indices = torch.randperm(len(train_set))

    train_set = torch.utils.data.Subset(train_set, indices[split_idx:])
    val_set = torch.utils.data.Subset(val_set, indices[:split_idx])
    
    return train_set, val_set, test_set


def prepare_dataloaders(train_set, val_set, test_set, batch_size, shuffle_train = True, num_workers = 8, pin_memory = False):  
    """
    Prepares the dataloaders for model training.

    Args:
        train_set (torch.utils.data): Training dataset. 
        val_set (torch.utils.data):   Validation dataset.
        test_set (torch.utils.data):  Testing dataset.
        batch_size (int): Size of the batches that should be trained upon.
        shuffle_train (bool, optional): Optional input to shuffle the training data (training robustness).
        num_workers (int, optional): Defines how many processes should be run in parallel.
        pin_memory (bool, optional): Loads the datasets as CUDA tensors.
    """
    from torch.utils.data import DataLoader

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader,val_loader,test_loader