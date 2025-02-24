import albumentations
import pytest

import qim3d


# unit tests for Augmentation()
def test_augmentation():
    augment_class = qim3d.ml.Augmentation()

    assert augment_class.resize == 'crop'


def test_augment():
    augment_class = qim3d.ml.Augmentation()

    album_augment = augment_class.augment(256, 256)

    assert type(album_augment) == albumentations.core.composition.Compose


# unit tests for ValueErrors in Augmentation()
def test_resize():
    resize_str = 'not valid resize'

    with pytest.raises(
        ValueError,
        match=f"Invalid resize type: {resize_str}. Use either 'crop', 'resize' or 'padding'.",
    ):
        augment_class = qim3d.ml.Augmentation(resize=resize_str)


def test_levels():
    augment_class = qim3d.ml.Augmentation()

    level = 'Not a valid level'

    with pytest.raises(
        ValueError,
        match=f"Invalid transformation level: {level}. Please choose one of the following levels: None, 'light', 'moderate', 'heavy'.",
    ):
        augment_class.augment(256, 256, level)
