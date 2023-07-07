"""Class for choosing or customizing data augmentations with albumentations"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Augmentation:
    """
    Class for defining image augmentation transformations using Albumentations library.

    Raises:
        ValueError: If the provided level is neither None, 'light', 'moderate', 'heavy', nor a custom augmentation.
        
    Attributes:
        resize (int): The target size to resize the image.
        mean (float): The mean value for normalizing pixel intensities.
        std (float): The standard deviation value for normalizing pixel intensities.

    Methods:
        augment(level=None): Apply image augmentation transformations based on the specified level, or on a 
            custom albumentations augmentation. The available levels are None, 'light', 'moderate', and 'heavy'.

    Usage:
        my_augmentation = Augmentation()
        moderate_augment = augmentation.augment(level='moderate')
    """
    def __init__(self, resize=256, mean=0.5, std=0.5):
        
        self.resize = resize
        self.mean = mean
        self.std = std
                
    def augment(self, level=None):

        # Check if one of standard augmentation levels
        if level not in [None,'light','moderate','heavy']:

            # Check if the custom transformation is an albumentation:
            if not isinstance(level, A.core.composition.Compose):
                raise ValueError("Custom Transformations need to be an instance of Albumentations Compose class, "
                                 "or one of the following levels: None, 'light', 'moderate', 'heavy'")
            # Custom transformation
            else:
                return level
        
        # Default transformation
        elif level is None:
            augment = A.Compose([
                A.Resize(self.resize, self.resize),
                A.Normalize(mean = (self.mean), std = (self.std)),
                ToTensorV2()
            ])

        # Choosing light augmentation
        elif level == 'light':
            augment = A.Compose([
                A.Resize(self.resize, self.resize),
                A.RandomRotate90(),
                A.Normalize(mean = (self.mean), std = (self.std)),
                ToTensorV2()
            ])

        # Choosing moderate augmentation
        elif level == 'moderate':
            augment = A.Compose([
                A.Resize(self.resize, self.resize),
                A.RandomRotate90(),
                A.HorizontalFlip(p = 0.3),
                A.VerticalFlip(p = 0.3),
                A.GlassBlur(sigma = 0.7, p = 0.1),
                A.Affine(scale = [0.8,1.2], translate_percent = (0.1,0.1)),
                A.Normalize(mean = (self.mean), std = (self.std)),
                ToTensorV2()
            ])

        # Choosing heavy augmentation
        elif level == 'heavy':
            augment = A.Compose([
                A.Resize(self.resize,self.resize),
                A.RandomRotate90(),
                A.HorizontalFlip(p = 0.7),
                A.VerticalFlip(p = 0.7),
                A.GlassBlur(sigma = 1.2, iterations = 2, p = 0.3),
                A.Affine(scale = [0.8,1.4], translate_percent = (0.2,0.2), shear = (-15,15)),
                A.Normalize(mean = (self.mean), std = (self.std)),
                ToTensorV2()
            ])

        return augment