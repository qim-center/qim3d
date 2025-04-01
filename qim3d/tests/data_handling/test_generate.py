import numpy as np
import pytest

import qim3d


# Unit test for background() shape mismatch ValueError
def test_background_shape_mismatch():
    vol = np.ones([128, 128, 128])
    background_shape = (64, 64, 64)

    msg = f'Shape of input volume {vol.shape} does not match background_shape {background_shape}.'

    with pytest.raises(ValueError, match=msg):
        qim3d.generate.background(background_shape=background_shape, apply_to=vol)


# Unit test for background() voxel intensities
def test_background_intensities():
    baseline_value = 10
    min_noise_value = 25
    max_noise_value = 50

    background = qim3d.generate.background(
        background_shape=(64, 64, 64),
        baseline_value=baseline_value,
        min_noise_value=min_noise_value,
        max_noise_value=max_noise_value,
    )

    # Assertions
    assert np.min(background) >= baseline_value + min_noise_value
    assert np.max(background) <= baseline_value + max_noise_value
    assert np.mean(background) >= baseline_value
