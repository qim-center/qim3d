"""Class for choosing the level of data augmentations with MONAI."""

from qim3d.utils._dependencies import optional_import

monai = optional_import('monai', extra='deep-learning')


class Augmentation:
    """
    Configures data augmentation pipelines for 3D deep learning using MONAI.

    This class simplifies the creation of augmentation strategies for training, validation, and testing. It allows you to select preset levels of intensity ('light', 'moderate', 'heavy') and define how input volumes are resized or cropped to match the model's input requirements.

    Args:
        resize (str, optional): The method used to conform input images to a specific size. Defaults to 'crop'.
            * **'crop'**: Extracts a central crop of the desired size.
            * **'reshape'**: Resizes (interpolates) the image to the desired size.
            * **'padding'**: Pads the image with zeros to reach the desired size.
        transform_train (str | None, optional): The intensity of augmentation applied to the training set. Options: 'light', 'moderate', 'heavy', or `None`. Defaults to 'moderate'.
        transform_validation (str | None, optional): The intensity of augmentation applied to the validation set. Defaults to `None` (no augmentation).
        transform_test (str | None, optional): The intensity of augmentation applied to the test set. Defaults to `None`.

    Raises:
        ValueError: If `resize` is not one of 'crop', 'reshape', or 'padding'.

    Example:
        ```python
        import qim3d

        # Create an augmentation strategy that crops images and applies moderate
        # transformations during training.
        augmentation = qim3d.ml.Augmentation(resize='crop', transform_train='moderate')
        ```
    """

    def __init__(
        self,
        resize: str = 'crop',
        transform_train: str | None = 'moderate',
        transform_validation: str | None = None,
        transform_test: str | None = None,
    ):
        if resize not in ['crop', 'reshape', 'padding']:
            msg = f"Invalid resize type: {resize}. Use either 'crop', 'resize' or 'padding'."
            raise ValueError(msg)

        self.resize = resize
        self.transform_train = transform_train
        self.transform_validation = transform_validation
        self.transform_test = transform_test

    def augment(
        self, img_shape: tuple, level: str | None = None
    ) -> monai.transforms.Compose:
        """
        Builds a MONAI composition of transforms based on the specified intensity level.

        This method constructs the actual pipeline of operations (e.g., rotations, flips, smoothing) that will be applied to the data.

        **Augmentation Levels:**

        * **None**: No augmentation. Only baseline formatting (ToTensor) is applied.
        * **'light'**: Random 90-degree rotations.
        * **'moderate'**: Rotations, flips, slight Gaussian smoothing, and minor affine transformations (scaling/translation).
        * **'heavy'**: Aggressive rotations, flips, stronger smoothing, and significant affine transformations including shearing.

        Args:
            img_shape (tuple): The target dimensions of the volume as `(Depth, Height, Width)`.
            level (str | None, optional): The specific augmentation level to generate. Must be one of `None`, 'light', 'moderate', or 'heavy'. Defaults to `None`.

        Returns:
            Compose (monai.transforms.Compose): A MONAI `Compose` object containing the sequence of transforms.

        Raises:
            ValueError: If `img_shape` is not 3D or if `level` is invalid.
        """
        from monai.transforms import (
            CenterSpatialCropd,
            Compose,
            RandAffined,
            RandFlipd,
            RandGaussianSmoothd,
            RandRotate90d,
            Resized,
            SpatialPadd,
            ToTensor,
        )

        # Check if image is 3D
        if len(img_shape) == 3:
            im_d, im_h, im_w = img_shape

        else:
            msg = f'Invalid image shape: {img_shape}. Must be 3D.'
            raise ValueError(msg)

        # Check if one of standard augmentation levels
        if level not in [None, 'light', 'moderate', 'heavy']:
            msg = f"Invalid transformation level: {level}. Please choose one of the following levels: None, 'light', 'moderate', 'heavy'."
            raise ValueError(msg)

        # Baseline augmentations
        # TODO: Figure out how to properly do normalization in 3D (normalization should be done channel-wise)
        baseline_aug = [ToTensor()]  # , NormalizeIntensityd(keys=["image"])]

        # Resize augmentations
        if self.resize == 'crop':
            resize_aug = [
                CenterSpatialCropd(keys=['image', 'label'], roi_size=(im_d, im_h, im_w))
            ]

        elif self.resize == 'reshape':
            resize_aug = [
                Resized(keys=['image', 'label'], spatial_size=(im_d, im_h, im_w))
            ]

        elif self.resize == 'padding':
            resize_aug = [
                SpatialPadd(keys=['image', 'label'], spatial_size=(im_d, im_h, im_w))
            ]

        # Level of augmentation
        if level is None:
            # No augmentation for the validation and test sets
            level_aug = []
            resize_aug = []

        elif level == 'light':
            # TODO: Do rotations along other axes?
            level_aug = [
                RandRotate90d(keys=['image', 'label'], prob=1, spatial_axes=(0, 1))
            ]

        elif level == 'moderate':
            level_aug = [
                RandRotate90d(keys=['image', 'label'], prob=1, spatial_axes=(0, 1)),
                RandFlipd(keys=['image', 'label'], prob=0.3, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.3, spatial_axis=1),
                RandGaussianSmoothd(keys=['image'], sigma_x=(0.7, 0.7), prob=0.1),
                RandAffined(
                    keys=['image', 'label'],
                    prob=0.5,
                    translate_range=(0.1, 0.1),
                    scale_range=(0.9, 1.1),
                ),
            ]

        elif level == 'heavy':
            level_aug = [
                RandRotate90d(keys=['image', 'label'], prob=1, spatial_axes=(0, 1)),
                RandFlipd(keys=['image', 'label'], prob=0.7, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.7, spatial_axis=1),
                RandGaussianSmoothd(keys=['image'], sigma_x=(1.2, 1.2), prob=0.3),
                RandAffined(
                    keys=['image', 'label'],
                    prob=0.5,
                    translate_range=(0.2, 0.2),
                    scale_range=(0.8, 1.4),
                    shear_range=(-15, 15),
                ),
            ]

        return Compose(baseline_aug + resize_aug + level_aug)
