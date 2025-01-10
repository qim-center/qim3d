"""Class for choosing the level of data augmentations with albumentations"""

class Augmentation:
    """
    Class for defining image augmentation transformations using the Albumentations library.
        
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
                 std: float = 0.5
                ):

        if resize not in ['crop', 'reshape', 'padding']:
            raise ValueError(f"Invalid resize type: {resize}. Use either 'crop', 'resize' or 'padding'.")

        self.resize = resize
        self.mean = mean
        self.std = std
        self.transform_train = transform_train
        self.transform_validation = transform_validation
        self.transform_test = transform_test
    
    def augment(self, im_h: int, im_w: int, level: bool | None = None):
        """
        Returns an albumentations.core.composition.Compose class depending on the augmentation level.
        A baseline augmentation is implemented regardless of the level, and a set of augmentations are added depending of the level.
        The A.Resize() function is used if the user has specified a 'resize' int or tuple at the creation of the Augmentation class.

        Args:
            im_h (int): image height for resize.
            im_w (int): image width for resize.
            level (str, optional): level of augmentation.

        Raises:
            ValueError: If `level` is neither None, light, moderate nor heavy.
        """
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # Check if one of standard augmentation levels
        if level not in [None,'light','moderate','heavy']:
            raise ValueError(f"Invalid transformation level: {level}. Please choose one of the following levels: None, 'light', 'moderate', 'heavy'.")

        # Baseline
        baseline_aug = [
            A.Normalize(mean = (self.mean),std = (self.std)),
            ToTensorV2()
        ]
        if self.resize == 'crop':
            resize_aug = [
                A.CenterCrop(im_h,im_w)
            ]
        elif self.resize == 'reshape':
            resize_aug =[
                A.Resize(im_h,im_w)
            ]
        elif self.resize == 'padding':
            resize_aug = [
                A.PadIfNeeded(im_h,im_w,border_mode = 0) # OpenCV border mode
            ]
        
        # Level of augmentation
        if level == None:
            level_aug = []
        elif level == 'light':
            level_aug = [
                A.RandomRotate90()
            ]
        elif level == 'moderate':
            level_aug = [
                A.RandomRotate90(),
                A.HorizontalFlip(p = 0.3),
                A.VerticalFlip(p = 0.3),
                A.GlassBlur(sigma = 0.7, p = 0.1),
                A.Affine(scale = [0.9,1.1], translate_percent = (0.1,0.1))
            ]
        elif level == 'heavy':
            level_aug = [
                A.RandomRotate90(),
                A.HorizontalFlip(p = 0.7),
                A.VerticalFlip(p = 0.7),
                A.GlassBlur(sigma = 1.2, iterations = 2, p = 0.3),
                A.Affine(scale = [0.8,1.4], translate_percent = (0.2,0.2), shear = (-15,15))
            ]

        augment = A.Compose(level_aug + resize_aug + baseline_aug)
        
        return augment