"""Provides a custom Dataset class for building a PyTorch dataset."""
from pathlib import Path
from PIL import Image
from qim3d.io.logger import log
from qim3d.utils.internal_tools import find_one_image
from torch.utils.data import DataLoader

import os
import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    ''' Custom Dataset class for building a PyTorch dataset.
    Args:
        root_path (str): The root directory path of the dataset.
        transform (callable, optional): A callable function or transformation to 
            be applied to the data. Default is None.

    Raises:
        ValueError: If the provided split is not valid (neither "train" nor "test").

    Attributes:
        root_path (str): root directory path to the dataset.
        transform (callable): The transformation to be applied to the data.
        sample_images (list): A list containing the paths to the sample images in the dataset.
        sample_targets (list): A list containing the paths to the corresponding target images
            in the dataset.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Returns the image and its target segmentation at the given index.
        _data_scan(): Finds how many folders are in the directory path as well as their names.
        _find_samples(): Finds the images and targets according to one of the 3 datastructure cases.

    Usage:
        dataset = Dataset(root_path="path/to/dataset",
            transform=albumentations.Compose([ToTensorV2()]))
        image, target = dataset[idx]
    
    Notes:
        Case 1: There are no folder - all images and targets are stored in the same data directory. 
                The image and corresponding target have similar names (eg: data1.tif, data1mask.tif)
        
        |-- data
            |-- img1.tif
            |-- img1_mask.tif
            |-- img2.tif
            |-- img2_mask.tif
            |-- ...
        
        Case 2: There are two folders - one with all the images and one with all the targets.

        |-- data
            |-- images
                |-- img1.tif
                |-- img2.tif
                |-- ...
            |-- masks
                |-- img1_mask.tif
                |-- img2_mask.tif
                |-- ...
        
        Case 3: There are many folders - each folder with a case (eg. patient) and multiple images.

        |-- data
            |-- patient1
                |-- p1_img1.tif
                |-- p1_img1_mask.tif
                |-- p1_img2.tif
                |-- p1_img2_mask.tif
                |-- p1_img3.tif
                |-- p1_img3_mask.tif
                |-- ...

            |-- patient2
                |-- p2_img1.tif
                |-- p2_img1_mask.tif
                |-- p2_img2.tif
                |-- p2_img2_mask.tif
                |-- p2_img3.tif
                |-- p2_img3_mask.tif
                |-- ...
            |-- ...
    '''
    def __init__(self, root_path: str, transform=None):
        super().__init__()

        self.root_path = root_path
        self.transform = transform

        # scans folders
        self._data_scan()
        # finds the images and targets given the folder setup
        self._find_samples()

        assert len(self.sample_images)==len(self.sample_targets)
        
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
    

    def _data_scan(self):
        ''' Find out which of the three categories the data belongs to.
        '''

        # how many folders there are:
        files = os.listdir(self.root_path)
        n_folders = 0
        folder_names = []
        for f in files:
            if os.path.isdir(Path(self.root_path,f)):
                n_folders += 1
                folder_names.append(f)

        self.n_folders = n_folders
        self.folder_names = folder_names


    def _find_samples(self):
        ''' Scans and retrieves the images and targets from their given folder configuration.

        Raises:
            ValueError: in Case 2, if no folder contains any of the labels 'mask', 'label', 'target'.
            NotImplementedError: in Case 3, if a file is found among the list of folders.
            NotImplementedError: in Case 3, if a folder is found among the list of files.
            NotImplementedError: If the data structure does not fall into one of the three cases.
        '''
        
        target_folder_names = ['mask','label','target']

        # Case 1
        if self.n_folders == 0:
            sample_images = []
            sample_targets = []

            for file in os.listdir(self.root_path):

                # checks if a label extension is in the filename
                if any(ext in file.lower() for ext in target_folder_names):
                    sample_targets.append(Path(self.root_path,file))
                
                # otherwise the file is assumed to be the image
                else:
                    sample_images.append(Path(self.root_path,file))
            
            self.sample_images = sorted(sample_images)
            self.sample_targets = sorted(sample_targets)
        
        # Case 2
        elif self.n_folders == 2:

            # if the first folder contains the targets:
            if any(ext in self.folder_names[0].lower() for ext in target_folder_names):
                images = self.folders_names[1]
                targets  = self.folder_names[0]
            
            # if the second folder contains the targets:
            elif any(ext in self.folder_names[1].lower() for ext in target_folder_names):
                images = self.folder_names[0]
                targets  = self.folder_names[1]

            else:
                raise ValueError('Folder names do not match categories such as "mask", "label" or "target".')

            self.sample_images = [image for image in sorted(Path(self.root_path,images).iterdir())]
            self.sample_targets = [target for target in sorted(Path(self.root_path,targets).iterdir())]

        # Case 3
        elif self.n_folders > 2:
            sample_images = []
            sample_targets = []

            for folder in os.listdir(self.root_path):
                
                # if some files are not a folder
                if not os.path.isdir(Path(self.root_path,folder)):
                    raise NotImplementedError(f'The current data structure is not supported. {Path(self.root_path,folder)} is not a folder.')

                for file in os.listdir(Path(self.root_path,folder)):

                    # if files are not images:
                    if not os.path.isfile(Path(self.root_path,folder,file)):
                        raise NotImplementedError(f'The current data structure is not supported. {Path(self.root_path,folder,file)} is not a file.')

                    # checks if a label extension is in the filename
                    if any(ext in file for ext in target_folder_names):
                        sample_targets.append(Path(self.root_path,folder,file))
                    
                    # otherwise the file is assumed to be the image
                    else:
                        sample_images.append(Path(self.root_path,folder,file))

            self.sample_images = sorted(sample_images)
            self.sample_targets = sorted(sample_targets)
        
        else:
            raise NotImplementedError('The current data structure is not supported.')
    
    # TODO: working with images of different sizes
    def check_shape_consistency(self,sample_images):
        image_shapes = []
        for image_path in sample_images[:100]:
            image_shape = self._get_shape(image_path)
            image_shapes.append(image_shape)

        # check if all images have the same size.
        unique_shapes = len(set(image_shapes))
               
        if unique_shapes>1:
            raise NotImplementedError(
                "Only images of all the same size can be processed at the moment"
            )
        else:
            log.debug(
                "Images are all the same size!"
            )
    
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


def prepare_datasets(
    path:str,
    model,
    augmentation,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    train_folder:str = None,
    val_folder:str = None,
    test_folder:str = None
):
    '''Splits and augments the train/validation/test datasets

    Args:
        path (str): Path to the dataset.
        model (torch.nn.Module): PyTorch Model.
        augmentation (albumentations.core.composition.Compose): Augmentation class for the dataset with predefined augmentation levels.
        val_fraction (float, optional): Fraction of the data for the validation set.
        test_fraction (float, optional): Fraction of the data for the test set.
        train_folder (str, optional): Can be used to specify where the data for training data is located. 
        val_folder (str, optional): Can be used to specify where the data for validation data is located.
        test_folder (str, optional): Can be used to specify where the data for testing data is located.

    Raises:
        ValueError: If the validation fraction is not a float, and is not between 0 and 1.
        ValueError: If the test fraction is not a float, and is not between 0 and 1.
        ValueError: If the sum of the validation and test fractions is equal or larger than 1.
        ValueError: If the combination of train/val/test_folder strings isn't enough to prepare the data for model training.
    
    Usage:
        # if all data stored together:
        prepare_datasets(path="path/to/dataset", val_fraction = 0.2, test_fraction = 0.1,
            model = qim3d.models.UNet(), augmentation = qim3d.utils.Augmentation())
        
        # if data has be pre-configured into training/testing:
        prepare_datasets(path="path/to/dataset", val_fraction = 0.2, test_fraction = 0.1,
            model = qim3d.models.UNet(), augmentation = qim3d.utils.Augmentation(),
            train_folder = 'training_folder_name', test_folder = 'test_folder_name')
    '''

    if not isinstance(val_fraction,float) or not (0 <= val_fraction < 1):
        raise ValueError("The validation fraction must be a float between 0 and 1.")
    
    if not isinstance(test_fraction,float) or not (0 <= test_fraction < 1):
        raise ValueError("The test fraction must be a float between 0 and 1.")

    if (val_fraction + test_fraction)>=1:
        print(int(val_fraction+test_fraction)*100)
        raise ValueError(f"The validation and test fractions cover {int((val_fraction+test_fraction)*100)}%. "
                         "Make sure to lower it below 100%, and include some place for the training data.")
    
    # Finds one image:
    image = Image.open(find_one_image(path = path))
    orig_h,orig_w = image.size[:2]
    
    resize = augmentation.resize
    n_channels = len(model.channels)

    final_h, final_w = check_resize(orig_h, orig_w, resize, n_channels)

    # Change number of channels in UNet if needed
    if len(np.array(image).shape)>2:
        model.img_channels = np.array(image).shape[2]
        model.update_params()

    # Only Train and Test folders are given, splits Train into Train/Val.
    if train_folder and test_folder and not val_folder:
        
        log.info('Only train and test given, splitting train_folder with val_fraction.')
        train_set = Dataset(root_path=Path(path,train_folder),transform=augmentation.augment(final_h, final_w,type = 'train'))
        val_set = Dataset(root_path=Path(path,train_folder),transform=augmentation.augment(final_h, final_w,type = 'validation'))
        test_set = Dataset(root_path=Path(path,test_folder),transform=augmentation.augment(final_h, final_w,type = 'test'))

        indices = torch.randperm(len(train_set))
        split_idx = int(np.floor(val_fraction * len(train_set)))

        train_set = torch.utils.data.Subset(train_set,indices[split_idx:])
        val_set = torch.utils.data.Subset(val_set,indices[:split_idx])        

    # Only Train and Val folder are given.
    elif train_folder and val_folder and not test_folder:

        log.info('Only train and validation folder provided, will not be able to make inference on test data.')
        train_set = Dataset(root_path=Path(path,train_folder),transform=augmentation.augment(final_h, final_w,type = 'train'))
        val_set = Dataset(root_path=Path(path,train_folder),transform=augmentation.augment(final_h, final_w,type = 'validation'))
        test_set = None

    # All Train/Val/Test folders are given.
    elif train_folder and val_folder and test_folder:
        
        log.info('Retrieving data from train, validation and test folder.')
        train_set = Dataset(root_path=Path(path,train_folder),transform=augmentation.augment(final_h, final_w,type = 'train'))
        val_set = Dataset(root_path=Path(path,train_folder),transform=augmentation.augment(final_h, final_w,type = 'validation'))
        test_set = Dataset(root_path=Path(path,test_folder),transform=augmentation.augment(final_h, final_w,type = 'test'))

    # None of the train/val/test folders are given:
    elif not(train_folder or val_folder or test_folder):

        log.info('No specific train/validation/test folders given. Splitting the data into train/validation/test sets.')
        train_set = Dataset(root_path=path,transform=augmentation.augment(final_h, final_w,type = 'train'))
        val_set = Dataset(root_path=path,transform=augmentation.augment(final_h, final_w,type = 'validation'))
        test_set =Dataset(root_path=path,transform=augmentation.augment(final_h, final_w,type = 'test'))

        indices = torch.randperm(len(train_set))
        
        train_idx = int(np.floor((1-val_fraction-test_fraction)*len(train_set)))
        val_idx = train_idx + int(np.floor(val_fraction*len(train_set)))

        train_set = torch.utils.data.Subset(train_set,indices[:train_idx])
        val_set = torch.utils.data.Subset(val_set,indices[train_idx:val_idx])
        test_set = torch.utils.data.Subset(test_set,indices[val_idx:])
    
    else:
        raise ValueError("Your folder configuration cannot be recognized. "
                         "Give a path to the dataset, or paths to the train/validation/test folders.")

    return train_set,val_set,test_set



def prepare_dataloaders(train_set, val_set, test_set, batch_size, shuffle_train = True, num_workers = 0, pin_memory = False):  
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
    
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader,val_loader,test_loader