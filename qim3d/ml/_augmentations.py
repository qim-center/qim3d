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
                ):

        if resize not in ['crop', 'reshape', 'padding']:
            raise ValueError(f"Invalid resize type: {resize}. Use either 'crop', 'resize' or 'padding'.")

        self.resize = resize
        self.mean = mean
        self.std = std
        self.transform_train = transform_train
        self.transform_validation = transform_validation
        self.transform_test = transform_test
    
    def augment(self, img_shape: tuple, level: str | None = None):
        """
        Creates an augmentation pipeline based on the specified level.

        Args:
            img_shape (tuple): Dimensions of the image.
            level (str, optional): Level of augmentation. One of [None, 'light', 'moderate', 'heavy'].

        Raises:
            ValueError: If `level` is neither None, light, moderate nor heavy.
        """
        from monai.transforms import (
            Compose, RandRotate90d, RandFlipd, RandAffined, ToTensor, \
            RandGaussianSmoothd, NormalizeIntensityd, Resized, CenterSpatialCropd, SpatialPadd
        )

        # Check if image is 3D
        if len(img_shape) == 3:
            im_d, im_h, im_w = img_shape
        
        else: 
            msg = f"Invalid image shape: {img_shape}. Must be 3D."
            raise ValueError(msg)

        # Check if one of standard augmentation levels
        if level not in [None,'light', 'moderate', 'heavy']:
            msg = f"Invalid transformation level: {level}. Please choose one of the following levels: None, 'light', 'moderate', 'heavy'."
            raise ValueError(msg)

        # Baseline augmentations
        # TODO: Figure out how to properly do normalization in 3D (normalization should be done channel-wise)
        baseline_aug = [ToTensor()]

        # Resize augmentations
        if self.resize == 'crop':
            resize_aug = [CenterSpatialCropd(keys=["image", "label"], roi_size=(im_d, im_h, im_w))]
        
        elif self.resize == 'reshape':
            resize_aug = [Resized(keys=["image", "label"], spatial_size=(im_d, im_h, im_w))]
        
        elif self.resize == 'padding':
            resize_aug = [SpatialPadd(keys=["image", "label"], spatial_size=(im_d, im_h, im_w))]

        # Level of augmentation
        if level == None:

            # No augmentation for the validation and test sets
            level_aug = []
            resize_aug = []

        elif level == 'light':
            # TODO: Do rotations along other axes?
            level_aug = [RandRotate90d(keys=["image", "label"], prob=1, spatial_axes=(0, 1))]
        
        elif level == 'moderate':
            level_aug = [
                RandRotate90d(keys=["image", "label"], prob=1, spatial_axes=(0, 1)),
                RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
                RandGaussianSmoothd(keys=["image"], sigma_x=(0.7, 0.7), prob=0.1),
                RandAffined(keys=["image", "label"], prob=0.5, translate_range=(0.1, 0.1), scale_range=(0.9, 1.1)),
                ]
            
        elif level == 'heavy':
            level_aug = [
                RandRotate90d(keys=["image", "label"], prob=1, spatial_axes=(0, 1)),
                RandFlipd(keys=["image", "label"], prob=0.7, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.7, spatial_axis=1),
                RandGaussianSmoothd(keys=["image"], sigma_x=(1.2, 1.2), prob=0.3),
                RandAffined(keys=["image", "label"], prob=0.5, translate_range=(0.2, 0.2), scale_range=(0.8, 1.4), shear_range=(-15, 15))
                ]
            
        return Compose(baseline_aug + resize_aug + level_aug)