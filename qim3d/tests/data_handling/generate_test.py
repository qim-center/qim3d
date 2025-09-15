import numpy as np
import pytest

import qim3d
import qim3d.generate


# Unit test for background() apply_to ValueError
def test_background_apply_to_error():
    background_shape = (64, 64, 64)
    msg = f'Supply both apply_method and apply_to when applying background to a volume.'

    with pytest.raises(ValueError, match=msg):
        qim3d.generate.background(background_shape=background_shape, apply_method='add')


# Unit test for background() voxel intensities
def test_background_intensities():
    baseline_value = 10
    min_noise_value = 25
    max_noise_value = 50

    background = qim3d.generate.background(
        background_shape=(64, 64, 64),
        baseline_value=baseline_value,
        generate_method='add',
        min_noise_value=min_noise_value,
        max_noise_value=max_noise_value,
    )

    # Assertions
    assert np.min(background) >= baseline_value + min_noise_value
    assert np.max(background) <= baseline_value + max_noise_value
    assert np.mean(background) >= baseline_value

def test_data_not_fit_raises():
    # A single custom volume larger than the collection → error
    large = np.zeros((10, 10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match=r'No custom volumes fit within collection size'):
        qim3d.generate.volume_collection(
            num_volumes=1,
            collection_shape=(5, 5, 5),
            data=large
        )

def test_data_single_fit_multiple_placements():
    # Single small volume fits repeatedly
    vol = np.full((3, 3, 3), fill_value=42, dtype=np.uint8)
    coll, labels = qim3d.generate.volume_collection(
        num_volumes=2,
        collection_shape=(10, 10, 10),
        data=vol
    )

    # Collection and labels have correct shape
    assert coll.shape == (10, 10, 10)
    assert labels.shape == coll.shape

    # Intensities are either 0 or 42
    unique_int = set(np.unique(coll))
    assert unique_int == {0, 42}

    # Two distinct labels plus background
    unique_lbl = set(np.unique(labels))
    assert unique_lbl == {0, 1, 2}
