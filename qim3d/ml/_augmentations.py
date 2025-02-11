"""Class for choosing the level of data augmentations with MONAI"""

class Augmentation:
    """
    Class for defining image augmentation transformations using the MONAI library.
        
    Args:
        resize (str, optional): Specifies how the images should be reshaped to the appropriate size.
        trainsform_train (str, optional): level of transformation for the training set.
        transform_validation (str, optional): level of transformation for the validation set.
        transform_test (str, optional): level of transformation for the test set.
        mean (float, optional): The mean value for normalizing pixel intensities.
        std (float, optional): The standard deviation value for normalizing pixel intensities.
        is_3d (bool, optional): Specifies if the images are 3D or 2D. Default is True.

    Raises:
        ValueError: If the ´resize´ is neither 'crop', 'resize' or 'padding'.
    
    Example:
        my_augmentation = Augmentation(resize = 'crop', transform_train = 'heavy')
    """
    
    def __init__(self, 
                 resize: str = 'crop', 
                 transform_train: str = 'moderate', 
                 transform_validation: str | None = None,
                 transform_test: str | None = None,
                 mean: float = 0.5, 
                 std: float = 0.5,
                 is_3d: bool = True,
                ):

        if resize not in ['crop', 'reshape', 'padding']:
            raise ValueError(f"Invalid resize type: {resize}. Use either 'crop', 'resize' or 'padding'.")

        self.resize = resize
        self.mean = mean
        self.std = std
        self.transform_train = transform_train
        self.transform_validation = transform_validation
        self.transform_test = transform_test
        self.is_3d = is_3d
    
    def augment(self, im_h: int, im_w: int, im_d: int | None = None, level: str | None = None):
        """
        Creates an augmentation pipeline based on the specified level.

        Args:
            im_h (int): Height of the image.
            im_w (int): Width of the image.
            im_d (int, optional): Depth of the image (for 3D).
            level (str, optional): Level of augmentation. One of [None, 'light', 'moderate', 'heavy'].

        Raises:
            ValueError: If `level` is neither None, light, moderate nor heavy.
        """
        from monai.transforms import (
            Compose, RandRotate90, RandFlip, RandAffine, ToTensor, \
            RandGaussianSmooth, NormalizeIntensity, Resize, CenterSpatialCrop, SpatialPad
        )

        # Check if one of standard augmentation levels
        if level not in [None,'light','moderate','heavy']:
            raise ValueError(f"Invalid transformation level: {level}. Please choose one of the following levels: None, 'light', 'moderate', 'heavy'.")

        # Baseline augmentations
        baseline_aug = [ToTensor()]

        # For 2D, add normalization to the baseline augmentations
        # TODO: Figure out how to properly do this in 3D (normalization should be done channel-wise)
        if not self.is_3d:
            baseline_aug.append(NormalizeIntensity(subtrahend=self.mean, divisor=self.std))

        # Resize augmentations
        if self.resize == 'crop':
            resize_aug = [CenterSpatialCrop((im_d, im_h, im_w))] if self.is_3d else [CenterSpatialCrop((im_h, im_w))]
        
        elif self.resize == 'reshape':
            resize_aug = [Resize((im_d, im_h, im_w))] if self.is_3d else [Resize((im_h, im_w))]
        
        elif self.resize == 'padding':
            resize_aug = [SpatialPad((im_d, im_h, im_w))] if self.is_3d else [SpatialPad((im_h, im_w))]

        # Level of augmentation
        if level == None:
            level_aug = []

        elif level == 'light':
            level_aug = [RandRotate90(prob=1, spatial_axes=(0, 1, 2))] if self.is_3d else [RandRotate90(prob=1)]
        
        elif level == 'moderate':
            level_aug = [
                RandRotate90(prob=1, spatial_axes=(0, 1, 2)) if self.is_3d else RandRotate90(prob=1),
                RandFlip(prob=0.3, spatial_axis=0),
                RandFlip(prob=0.3, spatial_axis=1),
                RandGaussianSmooth(sigma_x=(0.7, 0.7), prob=0.1),
                RandAffine(prob=0.5, translate_range=(0.1, 0.1), scale_range=(0.9, 1.1)),
            ]

        elif level == 'heavy':
            level_aug = [
                RandRotate90(prob=1, spatial_axes=(0, 1, 2)) if self.is_3d else RandRotate90(prob=1),
                RandFlip(prob=0.7, spatial_axis=0),
                RandFlip(prob=0.7, spatial_axis=1),
                RandGaussianSmooth(sigma_x=(1.2, 1.2), prob=0.3),
                RandAffine(prob=0.5, translate_range=(0.2, 0.2), scale_range=(0.8, 1.4), shear_range=(-15, 15))
            ]

        return Compose(baseline_aug + resize_aug + level_aug)