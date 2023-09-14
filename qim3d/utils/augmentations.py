"""Class for choosing the level of data augmentations with albumentations"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
from qim3d.io.logger import log

class Augmentation:
    """
    Class for defining image augmentation transformations using the Albumentations library.
        
    Args:
        resize ((int,tuple), optional): The target size to resize the image.
        trainsform_train (str, optional): level of transformation for the training set.
        transform_validation (str, optional): level of transformation for the validation set.
        transform_test (str, optional): level of transformation for the test set.
        mean (float, optional): The mean value for normalizing pixel intensities.
        std (float, optional): The standard deviation value for normalizing pixel intensities. 

    Raises:
        ValueError: If `resize` is neither a None, int nor tuple.  
    
    Example:
        my_augmentation = Augmentation(resize = (256,256), transform_train = 'heavy')
    """
    
    def __init__(self, 
                 resize = None, 
                 transform_train = 'moderate', 
                 transform_validation = None,
                 transform_test = None,
                 mean: float = 0.5, 
                 std: float = 0.5
                ):
        
        if not isinstance(resize,(type(None),int,tuple)):
            raise ValueError(f"Invalid input for resize: {resize}. Use an integer or tuple to modify the data.")

        self.resize = resize
        self.mean = mean
        self.std = std
        self.transform_train = transform_train
        self.transform_validation = transform_validation
        self.transform_test = transform_test                     
    
    def augment(self, im_h, im_w, level=None):
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
        
        # Check if one of standard augmentation levels
        if level not in [None,'light','moderate','heavy']:
            raise ValueError(f"Invalid transformation level: {level}. Please choose one of the following levels: None, 'light', 'moderate', 'heavy'.")

        # Baseline
        baseline_aug = [
            A.Resize(im_h, im_w),
            A.Normalize(mean = (self.mean),std = (self.std)),
            ToTensorV2()
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

        augment = A.Compose(level_aug + baseline_aug)
        
        return augment